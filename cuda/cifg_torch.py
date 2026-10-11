"""Luke Steuber's CUDA port of the frozen CIFG equations and MLX optimizer.

Tensor names/layouts match MLX safetensors. AdamW intentionally has no moment
bias correction: that is the existing trainer's default, unlike torch.optim.
"""
import math
import torch
from torch import nn


class CIFGLM(nn.Module):
    def __init__(self, vocab_size=16384, embed_dim=96, hidden=670):
        super().__init__()
        self.hidden = hidden
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.wx = nn.Parameter(torch.empty(embed_dim, 3 * hidden))
        self.wr = nn.Parameter(torch.empty(embed_dim, 3 * hidden))
        self.b = nn.Parameter(torch.zeros(3 * hidden))
        self.proj = nn.Parameter(torch.empty(hidden, embed_dim))
        # Real jobs load an explicit MLX-seeded initialization before any update.
        with torch.no_grad():
            self.embedding.weight.normal_(std=math.sqrt(1 / embed_dim))
            self.wx.uniform_(-1 / math.sqrt(embed_dim), 1 / math.sqrt(embed_dim))
            self.wr.uniform_(-1 / math.sqrt(embed_dim), 1 / math.sqrt(embed_dim))
            self.proj.uniform_(-1 / math.sqrt(hidden), 1 / math.sqrt(hidden))
            self.b[:hidden].fill_(1)

    def forward(self, tokens, state=None):
        xg = self.embedding(tokens) @ self.wx + self.b
        batch, length, _ = xg.shape
        if state is None:
            c = xg.new_zeros((batch, self.hidden))
            r = xg.new_zeros((batch, self.embed_dim))
        else:
            c, r = state
        outputs = []
        for t in range(length):
            f, o, u = (xg[:, t] + r @ self.wr).split(self.hidden, dim=-1)
            f, o, u = f.sigmoid(), o.sigmoid(), u.tanh()
            c = f * c + (1 - f) * u
            r = (o * c.tanh()) @ self.proj
            outputs.append(r)
        return torch.stack(outputs, dim=1) @ self.embedding.weight.T, (c, r)


def loss_fn(model, x, y, mask):
    logits, _ = model(x)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction='none')
    return (loss.reshape_as(mask) * mask).sum() / mask.sum()


class MLXAdamW:
    """Capture-safe fp32 AdamW, matching the frozen MLX defaults exactly in form."""
    def __init__(self, model, weight_decay=.01, clip=1.0):
        self.params = dict(model.named_parameters())
        self.m = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.weight_decay, self.clip = weight_decay, clip

    @torch.no_grad()
    def step(self, lr):
        norm = sum(p.grad.square().sum() for p in self.params.values()).sqrt()
        scale = (self.clip / (norm + 1e-6)).clamp(max=1) if self.clip > 0 else 1
        for k, p in self.params.items():
            g = p.grad * scale
            self.m[k].copy_(.9 * self.m[k] + .1 * g)
            self.v[k].copy_(.999 * self.v[k] + .001 * g.square())
            p.copy_(p * (1 - lr * self.weight_decay) - lr * self.m[k] / (self.v[k].sqrt() + 1e-8))

    def state_dict(self):
        return {kind: {k: t.detach().cpu().clone() for k, t in values.items()} for kind, values in [('m', self.m), ('v', self.v)]}

    def load_state_dict(self, state):
        for kind in ('m', 'v'):
            values = getattr(self, kind)
            if set(values) != set(state[kind]):
                raise ValueError('Optimizer tensor keys differ')
            for k, target in values.items():
                target.copy_(state[kind][k])


class CapturedStep:
    """Fixed-shape graph for a model not previously used by autograd.

    The first differentiable use must occur on the capture stream. Resetting
    parameter values or .grad does not reset an existing AccumulateGrad stream.
    Warm-up/capture updates are restored, never counted as training steps.
    """
    def __init__(self, model, optimizer, x, y, mask):
        self.model, self.optimizer = model, optimizer
        self.x, self.y, self.mask = x.clone(), y.clone(), mask.clone()
        self.lr = torch.zeros((), device=x.device)
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt_initial = optimizer.state_dict()
        self.capture_stream = stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                loss_fn(model, self.x, self.y, self.mask).backward()
                optimizer.step(self.lr)
        torch.cuda.current_stream().wait_stream(stream)
        model.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.loss = loss_fn(model, self.x, self.y, self.mask)
            self.loss.backward()
            optimizer.step(self.lr)
        model.load_state_dict(initial)
        optimizer.load_state_dict(opt_initial)
        torch.cuda.synchronize()

    def __call__(self, x, y, mask, lr):
        self.x.copy_(x)
        self.y.copy_(y)
        self.mask.copy_(mask)
        self.lr.fill_(lr)
        self.graph.replay()
        return self.loss
