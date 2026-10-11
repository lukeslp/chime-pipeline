"""Bounded private Hugging Face training runner. Author: Luke Steuber.

Run only from a verified package produced by prepare_cuda_training.py.
CUDA floating-point trajectories are not claimed identical to MLX.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from cifg_data import Vocab, Batcher, make_batch
from cifg_torch import CIFGLM, MLXAdamW, CapturedStep, loss_fn


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify_package(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['schema'] == 'localtype-cuda-input-v1' and manifest['private']
    for name, entry in manifest['files'].items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], name
    assert not list((root / 'data').glob('test.*'))
    return sha(root / 'manifest.json')


def lr_at(step, cfg):
    if step < cfg['warmup']:
        return cfg['lr'] * (step + 1) / cfg['warmup']
    progress = min(1., (step - cfg['warmup']) / max(1, cfg['max_steps'] - cfg['warmup']))
    floor = cfg['lr'] * cfg['min_lr_ratio']
    return floor + .5 * (1 + math.cos(math.pi * progress)) * (cfg['lr'] - floor)


def tensors(batch):
    return tuple(torch.as_tensor(a, device='cuda', dtype=torch.float32 if i == 2 else torch.long)
                 for i, a in enumerate(batch))


def parity(data, cfg, progress=lambda phase: None):
    """Fail before training if full-size MLX or CUDA graph parity is outside bounds."""
    golden = np.load(data / 'parity.npz')
    model = CIFGLM(cfg['vocab_size'], cfg['embed_dim'], cfg['hidden']).cuda()
    initial = load_file(str(data / 'initial.safetensors'), device='cuda')
    model.load_state_dict(initial)
    x, y, mask = tensors(tuple(golden[k] for k in ('x', 'y', 'mask')))
    logits, (c, r) = model(x)
    errors = {}
    def compare(name, actual, expected, atol=5e-6, rtol=5e-4):
        arr = actual.detach().cpu().numpy()
        np.testing.assert_allclose(arr, expected, atol=atol, rtol=rtol, err_msg=name)
        errors[name] = float(np.max(np.abs(arr - expected)))
    for name, actual in [('logits', logits), ('c', c), ('r', r)]:
        compare(name, actual, golden[name])
    loss = loss_fn(model, x, y, mask)
    compare('loss', loss, golden['loss'])
    loss.backward()
    for k, p in model.named_parameters():
        compare('gradient:' + k, p.grad, golden['grad:' + k], atol=1e-7)
    opt = MLXAdamW(model, cfg['weight_decay'], cfg['clip'])
    opt.step(cfg['lr'])
    for k, p in model.named_parameters():
        compare('update:' + k, p, golden['updated:' + k], atol=3e-5)
    progress('eager_mlx_comparison')
    # The eager comparisons above bind gradient accumulation to the default
    # stream. Fresh parameters avoid retaining those autograd stream bindings;
    # load_state_dict/zero_grad alone cannot reset them.
    model = CIFGLM(cfg['vocab_size'], cfg['embed_dim'], cfg['hidden']).cuda()
    model.load_state_dict(initial)
    opt = MLXAdamW(model, cfg['weight_decay'], cfg['clip'])
    eager = CIFGLM(cfg['vocab_size'], cfg['embed_dim'], cfg['hidden']).cuda()
    eager.load_state_dict(initial)
    eager_opt = MLXAdamW(eager, cfg['weight_decay'], cfg['clip'])
    graph = CapturedStep(model, opt, x, y, mask)
    for k, p in model.named_parameters():
        assert torch.equal(p, initial[k]), 'Capture changed initialization'
        assert torch.count_nonzero(opt.m[k]) == 0 and torch.count_nonzero(opt.v[k]) == 0
    for step in range(3):
        xx = (x + step) % cfg['vocab_size']
        eager.zero_grad(set_to_none=True)
        expected_loss = loss_fn(eager, xx, y, mask)
        expected_loss.backward()
        eager_opt.step(lr_at(step, cfg))
        got = graph(xx, y, mask, lr_at(step, cfg))
        torch.cuda.synchronize()
        compare(f'graph_loss:{step}', got, expected_loss.detach().cpu().numpy())
        for k, p in model.named_parameters():
            compare(f'graph_update:{step}:{k}', p, eager.state_dict()[k].cpu().numpy())
    progress('capture_and_replay')
    del graph, eager, eager_opt, opt, model, initial, logits, loss
    torch.cuda.empty_cache()
    return {'passed': True, 'absolute_errors': errors, 'atol': 5e-6, 'update_atol': 3e-5, 'gradient_atol': 1e-7, 'rtol': 5e-4}


@torch.no_grad()
def perplexity(model, vocab, tokens, offsets, ids):
    ids = ids[np.argsort(offsets[ids + 1] - offsets[ids], kind='stable')]
    total, count = 0., 0
    for start in range(0, len(ids), 128):
        seqs = [tokens[offsets[i]:offsets[i + 1]] for i in ids[start:start + 128]]
        batch = make_batch(seqs, vocab, 64)
        x, y, mask = tensors(batch)
        n = int(batch[2].sum())
        total += float(loss_fn(model, x, y, mask)) * n
        count += n
    assert count > 0
    nll = total / count
    assert math.isfinite(nll)
    return {'nll': nll, 'ppl': math.exp(nll), 'targets': count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['smoke', 'full'], required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    assert not a.out.exists(), 'A new output directory is required'
    a.out.mkdir(parents=True)
    from huggingface_hub import HfApi
    api = HfApi()
    assert api.repo_info(a.repo, repo_type='dataset').private, 'Output must remain private'
    manifest_sha = verify_package(a.package)
    data = a.package / 'data'
    cfg = json.loads((data / 'config.json').read_text())
    assert (cfg['max_steps'], cfg['batch_size'], cfg['max_len'], cfg['params']) == (102195, 256, 64, 2025114)
    assert cfg['oversample'] == dict(nus=8, taskmaster=8, oasst2=8, taskmaster3=1)
    cfg.update(backend='torch-cuda-float32', schedule_total_steps=102195, input_manifest_sha256=manifest_sha)
    write_json(a.out / 'config.json', cfg)
    started = time.monotonic()
    def event(kind, **values):
        record = dict(event=kind, elapsed=round(time.monotonic() - started, 2), **values)
        line = json.dumps(record, allow_nan=False)
        print(line, flush=True)
        with (a.out / 'train.jsonl').open('a') as f:
            f.write(line + '\n')
    def upload():
        files = {f.name: sha(f) for f in a.out.iterdir() if f.is_file() and f.name != 'checksums.json'}
        write_json(a.out / 'checksums.json', files)
        api.upload_folder(repo_id=a.repo, repo_type='dataset', folder_path=a.out,
                          path_in_repo=a.prefix, commit_message=f'Private {a.mode} checkpoint')
    state = dict(completed=False, state='preflight', step=0, mode=a.mode, input_manifest_sha256=manifest_sha)
    write_json(a.out / 'status.json', state)
    step, tokens_seen, train_seconds, best, evals = 0, 0, 0., float('inf'), []
    model = opt = None
    def checkpoint():
        if model is None or opt is None:
            return
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(a.out / 'latest.safetensors'))
        # Cursor = completed batches; the unchanged seeded Batcher replays to it.
        torch.save(dict(optimizer=opt.state_dict(), step=step, tokens_seen=tokens_seen,
                        best=best, evals=evals, train_seconds=train_seconds,
                        input_manifest_sha256=manifest_sha), a.out / 'training-state.pt')
    def interrupted(signum, frame):
        raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        assert torch.cuda.is_available(), 'CUDA required'
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        event('hardware', torch=torch.__version__, gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
        report = parity(data, cfg, progress=lambda phase: event('parity_phase', phase=phase, passed=True))
        write_json(a.out / 'parity.json', report)
        event('parity', passed=True)
        vocab = Vocab.load(str(data / 'vocab.txt'))
        splits = {s: tuple(np.load(data / f'{s}.{k}.npy', mmap_mode='r') for k in ('tokens', 'offsets')) for s in ('train', 'dev')}
        train_tokens, train_offsets = splits['train']
        dev_tokens, dev_offsets = splits['dev']
        names = json.loads((data / 'train.sources.json').read_text())
        codes = np.load(data / 'train.sources.npy')
        repeats = np.ones(len(codes), dtype=np.int32)
        for name, factor in cfg['oversample'].items():
            repeats[codes == names.index(name)] = factor
        lens = np.diff(train_offsets)
        epoch_lines = int(repeats[lens > 0].sum())
        epoch_tokens = int(((np.minimum(lens, 63) + 1) * repeats * (lens > 0)).sum())
        assert epoch_lines == cfg['epoch_lines'] and epoch_tokens == cfg['epoch_tokens']
        assert epoch_lines // 256 == cfg['batches_per_epoch'] == 34065
        batcher = Batcher(train_tokens, train_offsets, 256, 64, cfg['seed'], repeats=repeats)
        iterator = iter(batcher)
        ids = np.nonzero(np.diff(dev_offsets) > 0)[0]
        dev_ids = np.sort(np.random.default_rng(cfg['seed']).choice(ids, size=cfg['dev_lines'], replace=False))
        dev_names = json.loads((data / 'dev.sources.json').read_text())
        dev_codes = np.load(data / 'dev.sources.npy')
        human = {name: np.intersect1d(ids, np.nonzero(dev_codes == dev_names.index(name))[0]) for name in cfg['human_sources']}
        assert {k: len(v) for k, v in human.items()} == cfg['human_dev_lines']
        model = CIFGLM(cfg['vocab_size'], cfg['embed_dim'], cfg['hidden']).cuda()
        model.load_state_dict(load_file(str(data / 'initial.safetensors'), device='cuda'))
        opt = MLXAdamW(model, cfg['weight_decay'], cfg['clip'])
        if a.resume:
            saved = torch.load(a.resume / 'training-state.pt', weights_only=True, map_location='cpu')
            assert saved['input_manifest_sha256'] == manifest_sha
            model.load_state_dict(load_file(str(a.resume / 'latest.safetensors'), device='cuda'))
            opt.load_state_dict(saved['optimizer'])
            step, tokens_seen, best, evals, train_seconds = (saved[k] for k in ('step', 'tokens_seen', 'best', 'evals', 'train_seconds'))
            assert 0 <= step < 102195
            import shutil
            if (a.resume / 'best.safetensors').exists():
                shutil.copy2(a.resume / 'best.safetensors', a.out / 'best.safetensors')
            for _ in range(step):
                next(iterator)
        first = make_batch(next(iterator), vocab, 64, round_to=64)
        graph = CapturedStep(model, opt, *tensors(first))
        event('capture', seconds=round(time.monotonic() - started, 2))
        def evaluate():
            nonlocal best
            t = time.monotonic()
            rec = perplexity(model, vocab, dev_tokens, dev_offsets, dev_ids)
            sources = {k: perplexity(model, vocab, dev_tokens, dev_offsets, v) for k, v in human.items()}
            count = sum(r['targets'] for r in sources.values())
            nll = sum(r['targets'] * r['nll'] for r in sources.values()) / count
            macro = sum(r['nll'] for r in sources.values()) / len(sources)
            rec.update(human=dict(nll=nll, ppl=math.exp(nll), targets=count, macro_nll=macro,
                                  macro_ppl=math.exp(macro), per_source=sources),
                       step=step, tokens=tokens_seen, eval_seconds=time.monotonic() - t,
                       selected_on='human-macro', selection_ppl=math.exp(macro))
            if math.exp(macro) < best:
                best = math.exp(macro)
                save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(a.out / 'best.safetensors'))
            evals.append(rec)
            event('eval', **rec)
            return rec
        limit = 300 if a.mode == 'smoke' else 102195
        warm_start = None
        initial_step = step
        state['state'] = 'running'
        for batch_index in range(step, limit):
            t = time.monotonic()
            batch = first if batch_index == initial_step else make_batch(next(iterator), vocab, 64, round_to=64)
            loss = float(graph(*tensors(batch), lr_at(batch_index, cfg)))
            assert math.isfinite(loss), 'Nonfinite loss'
            tokens_seen += int(batch[2].sum())
            step = batch_index + 1
            train_seconds += time.monotonic() - t
            if step == initial_step + 30:
                warm_start = time.monotonic()
            if step % 50 == 0:
                event('train', step=step, loss=loss, tokens=tokens_seen, train_seconds=round(train_seconds, 3))
            if a.mode == 'full' and step % 1000 == 0:
                evaluate()
                checkpoint()
                state.update(step=step)
                write_json(a.out / 'status.json', state)
                upload()
        warm_seconds = time.monotonic() - warm_start
        final = evaluate()
        checkpoint()
        if a.mode == 'smoke':
            seconds_per_step = warm_seconds / (limit - initial_step - 30)
            estimate = seconds_per_step * 102195 + final['eval_seconds'] * 104 + 600
            write_json(a.out / 'benchmark.json', dict(seconds_per_step=seconds_per_step,
                       eval_seconds=final['eval_seconds'], estimated_full_seconds=estimate,
                       estimated_full_cost_usd=estimate * 5 / 3600,
                       passed=estimate < 270 * 60 * .85))
            event('benchmark', estimated_full_seconds=estimate, seconds_per_step=seconds_per_step)
        completed = a.mode == 'full' and step == 102195
        write_json(a.out / 'metrics.json', dict(config=cfg, steps=step, stop_reason='max_steps' if completed else 'smoke',
                   epochs=round(tokens_seen / int(train_offsets[-1]), 3), batcher_epochs=round(tokens_seen / epoch_tokens, 3),
                   tokens_seen=tokens_seen, wall_seconds=time.monotonic() - started, train_seconds=train_seconds,
                   final_dev=final, selected_on='human-macro', best_dev_ppl=best, evals=evals,
                   keyboard=None, keyboard_evaluation_pending=True))
        state.update(completed=completed, state='complete' if completed else 'smoke_complete', step=step)
        write_json(a.out / 'status.json', state)
        upload()
        event('done', **state)
    except BaseException as exc:
        state.update(completed=False, state='failed', step=step, error=type(exc).__name__ + ': ' + str(exc))
        write_json(a.out / 'status.json', state)
        event('failed', **state)
        checkpoint()
        upload()
        raise


if __name__ == '__main__':
    main()
