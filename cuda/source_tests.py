"""Synthetic CPU checks for the x8 source review. Author: Luke Steuber.

Run from the prepared source root. No data bundle, network or CUDA is used.
These checks require NumPy, PyTorch and safetensors; nothing is silently skipped.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import cifg_data as data
from cifg_torch import CIFGLM, MLXAdamW, loss_fn
from train_cuda import lr_at


class SyntheticSourceChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def test_cifg_state_and_padding_gradients(self):
        model = CIFGLM(11, 4, 7)
        x = torch.tensor([[1, 4, 5], [1, 6, 0]])
        y = torch.tensor([[4, 5, 2], [6, 2, 0]])
        mask = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
        logits, state = model(x)
        first, partial = model(x[:, :1])
        last, continued = model(x[:, 1:], partial)
        torch.testing.assert_close(torch.cat([first, last], 1), logits)
        for a, b in zip(state, continued):
            torch.testing.assert_close(a, b)
        original = loss_fn(model, x, y, mask)
        original.backward()
        gradients = {k: p.grad.clone() for k, p in model.named_parameters()}
        model.zero_grad()
        padded = loss_fn(model, torch.nn.functional.pad(x, (0, 5)),
                         torch.nn.functional.pad(y, (0, 5)),
                         torch.nn.functional.pad(mask, (0, 5)))
        padded.backward()
        torch.testing.assert_close(original, padded)
        for k, p in model.named_parameters():
            torch.testing.assert_close(gradients[k], p.grad, atol=2e-7, rtol=2e-5)

    def test_uncorrected_adamw_against_numpy(self):
        model = CIFGLM(11, 4, 7)
        opt = MLXAdamW(model, weight_decay=.01, clip=.1)
        expected = {k: p.detach().numpy().copy() for k, p in model.named_parameters()}
        moments = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in expected.items()}
        for step in range(3):
            grads = {k: np.full_like(v, .2 + step * .1) for k, v in expected.items()}
            norm = np.sqrt(sum((g * g).sum() for g in grads.values()))
            scale = min(1., .1 / (norm + 1e-6))
            for k, p in model.named_parameters():
                p.grad = torch.from_numpy(grads[k].copy())
                g = grads[k] * scale
                m, v = moments[k]
                m[:] = .9 * m + .1 * g
                v[:] = .999 * v + .001 * g * g
                expected[k] = expected[k] * (1 - .002 * .01) - .002 * m / (np.sqrt(v) + 1e-8)
            opt.step(.002)
            for k, p in model.named_parameters():
                np.testing.assert_allclose(p.detach().numpy(), expected[k], atol=2e-7, rtol=2e-6)

    def test_synthetic_checkpoint_roundtrip(self):
        model = CIFGLM(11, 4, 7)
        opt = MLXAdamW(model)
        x = torch.tensor([[1, 4, 5]])
        y = torch.tensor([[4, 5, 2]])
        mask = torch.ones((1, 3))
        loss_fn(model, x, y, mask).backward()
        opt.step(.001)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_file(model.state_dict(), str(root / 'tiny.safetensors'))
            torch.save(opt.state_dict(), root / 'tiny.pt')
            restored = CIFGLM(11, 4, 7)
            restored.load_state_dict(load_file(str(root / 'tiny.safetensors')))
            other = MLXAdamW(restored)
            other.load_state_dict(torch.load(root / 'tiny.pt', weights_only=True))
        for current, optimizer in ((model, opt), (restored, other)):
            current.zero_grad()
            loss_fn(current, x, y, mask).backward()
            optimizer.step(.0005)
        for k, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[k], rtol=0, atol=0)

    def test_batch_contract_seeded_order_and_cursor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'vocab.txt'
            path.write_text('<pad>\n<s>\n</s>\n<unk>\nhello\nthere\n')
            vocab = data.Vocab.load(str(path))
        x, y, mask = data.make_batch([np.array([4, 5]), np.array([4])], vocab, 64, round_to=64)
        self.assertEqual(x.shape, (2, 64))
        np.testing.assert_array_equal(x[0, :3], [1, 4, 5])
        np.testing.assert_array_equal(y[0, :3], [4, 5, 2])
        np.testing.assert_array_equal(mask.sum(axis=1), [3, 2])
        codes = np.array([0, 1, 2, 3, 4])
        repeats = data.oversample_repeats_by_source(codes, ['nus', 'taskmaster', 'oasst2', 'taskmaster3', 'soda'],
                                                   dict(nus=8, taskmaster=8, oasst2=8, taskmaster3=1))
        np.testing.assert_array_equal(repeats, [8, 8, 8, 1, 1])
        # Unique one-token sequences expose sampler order without any real text.
        tokens, offsets = np.arange(4, 9), np.arange(6)
        def iterator(seed):
            return iter(data.Batcher(tokens, offsets, 2, 64, seed, pool_batches=2, repeats=repeats))
        def signature(batch):
            return tuple(tuple(s) for s in batch)
        a, b = iterator(20260925), iterator(20260925)
        order = [signature(next(a)) for _ in range(20)]
        self.assertEqual(order, [signature(next(b)) for _ in range(20)])
        different = iterator(99)
        self.assertNotEqual(order, [signature(next(different)) for _ in range(20)])
        resumed = iterator(20260925)
        for _ in range(7):
            next(resumed)
        self.assertEqual(order[7:], [signature(next(resumed)) for _ in range(13)])

    def test_schedule_boundaries(self):
        cfg = dict(warmup=200, lr=.002, min_lr_ratio=.1, max_steps=102195)
        self.assertAlmostEqual(lr_at(0, cfg), .00001)
        self.assertAlmostEqual(lr_at(199, cfg), .002)
        self.assertAlmostEqual(lr_at(200, cfg), .002)
        self.assertAlmostEqual(lr_at(102195, cfg), .0002)
        self.assertGreater(lr_at(50000, cfg), lr_at(98000, cfg))


if __name__ == '__main__':
    unittest.main()
