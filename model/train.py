#!/usr/bin/env python3
"""Train and evaluate the CIFG-LSTM next-word model on the LocalType corpus.

    python model/train.py prepare [--train-lines N]
    python model/train.py train --name smoke --max-minutes 10 [--max-steps N]
    python model/train.py train --name v3 --corpus <root>/corpus-v3 --vocab <root>/vocab-v3/vocab.txt \\
        --select-on human-macro --oversample taskmaster3=1 --max-steps N
    python model/train.py eval --name smoke [--split dev|test] [--which best|latest]

Data root: ~/.local/share/localtype-model (LOCALTYPE_MODEL_DATA).
The corpus is <root>/corpus unless --corpus names another build; its
manifest.json says which sources are human-written. Checkpoints, logs and
metrics go to <root>/runs/<name>/, and `eval` reads the vocabulary and corpus
a run was trained on from its config.json.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Keep the standard library ahead of this directory on sys.path: a sibling
# model/tokenize.py would otherwise shadow the stdlib tokenize module that
# inspect/linecache import.
if sys.path and os.path.abspath(sys.path[0]) == _HERE:
    sys.path.pop(0)
if _HERE not in sys.path:
    sys.path.append(_HERE)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from functools import partial  # noqa: E402

import numpy as np  # noqa: E402

import cifg  # noqa: E402
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402


# --------------------------------------------------------------------------
# Shared setup
# --------------------------------------------------------------------------

def resolve_vocab(root: str, explicit: str | None, workers: int, corpus: str | None = None) -> tuple[cifg.Vocab, bool]:
    """The shared vocab when it exists, else a temporary frequency vocab built here."""
    if explicit:
        return cifg.Vocab.load(explicit), False
    shared = cifg.shared_vocab_path(root)
    if os.path.exists(shared):
        return cifg.Vocab.load(shared), False
    tmp = cifg.temp_vocab_path(root)
    if not os.path.exists(tmp):
        print(f"[vocab] {shared} is absent; building a temporary frequency vocab at {tmp}", flush=True)
        cifg.build_temp_vocab(root, os.path.join(cifg.corpus_dir(root, corpus), "train.txt"), cifg.VOCAB_SIZE, workers=workers)
        with open(os.path.join(os.path.dirname(tmp), "meta.json")) as f:
            print("[vocab] " + json.dumps(json.load(f)), flush=True)
    else:
        print(f"[vocab] {shared} is absent; using the temporary frequency vocab at {tmp}", flush=True)
    return cifg.Vocab.load(tmp), True


def setup(args) -> dict:
    root = args.root
    corpus = cifg.corpus_dir(root, getattr(args, "corpus", None))
    vocab, temporary = resolve_vocab(root, args.vocab, args.workers, corpus)
    tokenizer, source = cifg.load_tokenizer(vocab)
    fp = cifg.cache_fingerprint(vocab, source, root, corpus)
    human = cifg.human_sources(corpus)
    print(f"[vocab] {vocab.path}: {len(vocab)} entries, {len(vocab.word_ids)} words, start={vocab.start} end={vocab.end} unk={vocab.unk} pad={vocab.pad}; tokenizer {source}; cache {cifg.cache_dir(root, fp)}", flush=True)
    print(f"[corpus] {corpus}: human-written sources {', '.join(human)}", flush=True)
    return {"root": root, "corpus": corpus, "vocab": vocab, "temporary_vocab": temporary, "tokenizer_source": source,
            "fingerprint": fp, "human_sources": human}


def prepare(ctx: dict, split: str, max_lines: int | None, workers: int) -> dict:
    meta = cifg.prepare_split(ctx["root"], ctx["vocab"], ctx["fingerprint"], split, max_lines, workers, corpus=ctx["corpus"])
    print(f"[prepare] {meta['tag']}: {meta['lines']} lines, {meta['tokens']} tokens, {meta['unk_tokens']} unk ({100 * meta['unk_tokens'] / max(1, meta['tokens']):.2f}%), {meta['empty_lines']} empty, {meta['seconds']}s", flush=True)
    return meta


def sample_lines(offsets: np.ndarray, n: int | None, seed: int, within: np.ndarray | None = None) -> np.ndarray:
    lens = np.diff(offsets)
    ids = np.nonzero(lens > 0)[0]
    if within is not None:
        ids = np.intersect1d(ids, within)
    if n is None or n >= len(ids):
        return ids
    return np.sort(np.random.default_rng(seed).choice(ids, size=n, replace=False))


def dev_sample(offsets: np.ndarray, n: int, seed: int) -> np.ndarray:
    """The dev lines for periodic perplexity: n sampled lines, or all of them when n is 0 or less.

    That perplexity also picks best.safetensors, so an empty sample (which
    scores as perplexity 1.0 and freezes best at the first eval) is refused.
    """
    ids = sample_lines(offsets, n if n > 0 else None, seed)
    if len(ids) == 0:
        raise SystemExit("the dev split has no non-empty lines; periodic perplexity (and best.safetensors) would be meaningless")
    return ids


# The human-written sources of the v2 corpus, reported as their own columns
# beside the pooled one; a corpus with a manifest names its own. The NUS column
# is the one that predicts the harness: real text messages.
HUMAN_SOURCES = cifg.DEFAULT_HUMAN_SOURCES
HUMAN_OVERSAMPLE = 4
SELECT_ON = ("pooled", "human", "human-macro")


def parse_oversample(spec: str | None) -> dict[str, int]:
    """``taskmaster3=1,nus=6`` -> {"taskmaster3": 1, "nus": 6}."""
    out: dict[str, int] = {}
    for part in (spec or "").split(","):
        if not part.strip():
            continue
        name, eq, value = part.strip().partition("=")
        if not eq or not name:
            raise SystemExit(f"--oversample wants source=factor pairs, got {part!r}")
        out[name.strip()] = int(value)
    return out


def oversample_factors(human: tuple[str, ...], default: int, overrides: dict[str, int], names: list[str]) -> dict[str, int]:
    """Every human-written source at `default`, then the per-source overrides (which may name any source)."""
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise SystemExit(f"--oversample names {unknown}, which the training split does not have ({', '.join(names)})")
    factors = {s: default for s in human}
    factors.update(overrides)
    return factors


def human_dev_sets(codes: np.ndarray, names: list[str], offsets: np.ndarray, human: tuple[str, ...], seed: int) -> dict[str, np.ndarray]:
    """Every non-empty dev line of each human-written source, keyed by source."""
    sets = {}
    for src in human:
        if src in names:
            ids = sample_lines(offsets, None, seed, within=np.nonzero(codes == names.index(src))[0])
            if len(ids):
                sets[src] = ids
    if not sets:
        raise SystemExit("the dev split has no human-written lines; --select-on human would select on nothing")
    return sets


def human_perplexity(per_source: dict[str, dict]) -> dict:
    """Pooled (every human-written target counts once) and macro (every source counts once) perplexity.

    Each entry of per_source is a dev_perplexity result: mean NLL and target count.
    """
    scored = {s: r for s, r in per_source.items() if r["targets"]}
    targets = sum(r["targets"] for r in scored.values())
    pooled = sum(r["nll"] * r["targets"] for r in scored.values()) / targets if targets else float("nan")
    macro = sum(r["nll"] for r in scored.values()) / len(scored) if scored else float("nan")
    return {
        "nll": pooled,
        "ppl": math.exp(pooled),
        "targets": targets,
        "macro_nll": macro,
        "macro_ppl": math.exp(macro),
        "per_source": {s: {"ppl": r["ppl"], "nll": r["nll"], "targets": r["targets"]} for s, r in per_source.items()},
    }


def selection_ppl(record: dict, select_on: str) -> float:
    """The number best.safetensors is chosen on: pooled dev perplexity, or one of the human-written ones."""
    if select_on == "pooled":
        return record["ppl"]
    if select_on == "human":
        return record["human"]["ppl"]
    if select_on == "human-macro":
        return record["human"]["macro_ppl"]
    raise ValueError(f"select_on must be one of {SELECT_ON}, not {select_on!r}")


def evaluate_split(model, vocab, root: str, fingerprint: str, split: str, *, batch_size: int, max_len: int,
                   pooled_lines: int | None, seed: int, columns=HUMAN_SOURCES, corpus: str | None = None) -> dict:
    """Perplexity and keyboard metrics for one split: pooled and per human source, whole-line and sentence scope.

    The pooled column is a seeded sample of `pooled_lines` lines (all when
    None); the per-source columns use every line of that source in the split,
    because they are small. Perplexity uses every line.
    """
    tokens, offsets, oov = cifg.load_split(root, fingerprint, split, None)
    codes, names = cifg.load_sources(root, split, None, corpus=corpus)
    if len(codes) != len(offsets) - 1:
        raise SystemExit(f"{split}.sources.txt has {len(codes)} lines, the tokenized split {len(offsets) - 1}")
    sets: dict[str, np.ndarray] = {"pooled": sample_lines(offsets, pooled_lines, seed)}
    for src in columns:
        if src in names:
            sets[src] = sample_lines(offsets, None, seed, within=np.nonzero(codes == names.index(src))[0])
    out: dict = {"split": split, "lines": int(len(offsets) - 1), "perplexity": {}, "keyboard": {}}
    t0 = time.time()
    for name, ids in sets.items():
        all_ids = sample_lines(offsets, None, seed) if name == "pooled" else ids
        r = cifg.dev_perplexity(model, vocab, tokens, offsets, all_ids, batch_size=batch_size, max_len=max_len)
        r["lines"] = int(len(all_ids))
        out["perplexity"][name] = r
    out["perplexity"]["seconds"] = round(time.time() - t0, 1)
    for scope in ("line", "sentence"):
        t0 = time.time()
        cols = {}
        for name, ids in sets.items():
            cols[name] = cifg.keyboard_metrics(model, vocab, tokens, offsets, oov, ids, batch_size=max(1, batch_size // 4), max_len=max_len, scope=scope)
        cols["seconds"] = round(time.time() - t0, 1)
        out["keyboard"][scope] = cols
    return out


def print_evaluation(ev: dict, tag: str) -> None:
    ppl = ev["perplexity"]
    print(f"[ppl {tag}] " + "  ".join(f"{k} {v['ppl']:.2f} ({v['targets']} targets, {v['lines']} lines)" for k, v in ppl.items() if k != "seconds") + f"  {ppl['seconds']}s", flush=True)
    for scope in ("line", "sentence"):
        cols = {k: v for k, v in ev["keyboard"][scope].items() if k != "seconds"}
        label = "sentence scope (harness-comparable)" if scope == "sentence" else "whole-line scope"
        print(f"[keyboard {tag}] {label}, {ev['keyboard'][scope]['seconds']}s\n" + cifg.format_table(cols), flush=True)
        for k, v in cols.items():
            print(f"  {k}: {v['items']} items, cut at {v['cut']['max_len']}: {v['cut']['items_cut']} items / {v['cut']['targets_not_scored']} targets not scored; one-letter unofferable {v['unofferable_one_letter_targets']}", flush=True)


def add_common(p: argparse.ArgumentParser):
    p.add_argument("--root", default=cifg.data_root())
    p.add_argument("--vocab", default=None, help="vocab.txt to use instead of the shared or temporary one")
    p.add_argument("--corpus", default=None, help="corpus directory to use instead of <root>/corpus (e.g. <root>/corpus-v3)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--train-lines", type=int, default=None, help="tokenize only the first N training lines")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def lr_at(step: int, total: int, peak: float, warmup: int, floor_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    floor = peak * floor_ratio
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (peak - floor)


def plan_steps(step: int, rate: float, remaining_seconds: float, eval_seconds: float | None, eval_every: int, eval_reserve: float, max_steps: int, warmup: int) -> int:
    """How many steps the cosine should span so a time-bounded run ends near its floor.

    rate is training steps per second of training time. With a measured eval
    cost, the remaining time is split so that one eval per eval_every steps
    fits; without one, eval_reserve of the time is kept back.
    """
    if eval_seconds is None:
        train_time = remaining_seconds * (1.0 - eval_reserve)
    else:
        train_time = (remaining_seconds - eval_seconds) / (1.0 + rate * eval_seconds / max(1, eval_every))
    est = step + int(max(0.0, train_time) * rate)
    return max(step + 1, warmup + 1, min(max_steps, est))


def cmd_train(args):
    ctx = setup(args)
    root, vocab = ctx["root"], ctx["vocab"]
    prepare(ctx, "train", args.train_lines, args.workers)
    prepare(ctx, "dev", None, args.workers)
    corpus, human = ctx["corpus"], ctx["human_sources"]
    train_tokens, train_offsets, _ = cifg.load_split(root, ctx["fingerprint"], "train", args.train_lines)
    dev_tokens, dev_offsets, dev_oov = cifg.load_split(root, ctx["fingerprint"], "dev", None)
    # Checked here as well as in evaluate_split, so a mismatch stops the run before training, not after it.
    dev_codes, dev_names = cifg.load_sources(root, "dev", None, corpus=corpus)
    if len(dev_codes) != len(dev_offsets) - 1:
        raise SystemExit(f"dev.sources.txt has {len(dev_codes)} lines, the tokenized split {len(dev_offsets) - 1}")
    dev_ids = dev_sample(dev_offsets, args.dev_lines, args.seed)
    # Human-written dev lines for --select-on human / human-macro: every one, per source.
    human_dev = human_dev_sets(dev_codes, dev_names, dev_offsets, human, args.seed) if args.select_on != "pooled" else {}
    src_codes, src_names = cifg.load_sources(root, "train", args.train_lines, corpus=corpus)
    if len(src_codes) != len(train_offsets) - 1:
        raise SystemExit(f"train.sources.txt has {len(src_codes)} lines, the tokenized split {len(train_offsets) - 1}")
    factors = oversample_factors(human, args.human_oversample, parse_oversample(args.oversample), src_names)
    repeats = cifg.oversample_repeats_by_source(src_codes, src_names, factors)
    source_lines = {s: int((src_codes == i).sum()) for i, s in enumerate(src_names)}
    human_lines = sum(source_lines.get(s, 0) for s in human)
    epoch_lines = int(repeats.sum())
    human_epoch_share = sum(source_lines.get(s, 0) * max(1, factors.get(s, 1)) for s in human) / max(1, epoch_lines)
    lens = np.minimum(np.diff(train_offsets), args.max_len - 1)
    nonempty = lens > 0
    # What one epoch of the batcher feeds: every non-empty line `repeats` times, cut at max_len, plus END.
    epoch_tokens = int(((lens + 1) * repeats)[nonempty].sum())
    batches_per_epoch = int(repeats[nonempty].sum()) // args.batch_size

    run_dir = os.path.join(root, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    mx.random.seed(args.seed)
    model = cifg.CIFGLM(len(vocab), args.embed_dim, args.hidden)
    mx.eval(model.parameters())
    n_params = cifg.param_count(model)
    config = {
        **model.config(),
        "params": n_params,
        "vocab_path": vocab.path,
        "temporary_vocab": ctx["temporary_vocab"],
        "tokenizer": ctx["tokenizer_source"],
        "cache_fingerprint": ctx["fingerprint"],
        "train_lines": args.train_lines,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "lr": args.lr,
        "min_lr_ratio": args.min_lr_ratio,
        "warmup": args.warmup,
        "weight_decay": args.weight_decay,
        "clip": args.clip,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "max_minutes": args.max_minutes,
        "dev_lines": int(len(dev_ids)),
        "corpus": corpus,
        "human_sources": list(human),
        "human_oversample": args.human_oversample,
        "oversample": factors,
        "source_lines": source_lines,
        "human_lines": human_lines,
        "epoch_lines": epoch_lines,
        "epoch_tokens": epoch_tokens,
        "batches_per_epoch": batches_per_epoch,
        "human_epoch_share": round(human_epoch_share, 5),
        "select_on": args.select_on,
        "human_dev_lines": {s: int(len(ids)) for s, ids in human_dev.items()},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"[model] {config['arch']} vocab={len(vocab)} embed={args.embed_dim} hidden={args.hidden} params={n_params:,}", flush=True)
    print(f"[data] {len(train_offsets) - 1:,} train lines, {human_lines:,} human-written ({', '.join(f'{s} x{factors.get(s, 1)}' for s in human)}) = {100 * human_epoch_share:.2f}% of every epoch's {epoch_lines:,} lines; one epoch is {epoch_tokens:,} tokens in {batches_per_epoch:,} batches", flush=True)
    if human_dev:
        print(f"[select] best.safetensors by {args.select_on} perplexity on {sum(len(v) for v in human_dev.values()):,} human-written dev lines ({', '.join(f'{s} {len(v)}' for s, v in human_dev.items())})", flush=True)

    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    loss_and_grad = nn.value_and_grad(model, cifg.loss_fn)
    state = [model.state, optimizer.state]

    def step_fn(x, y, mask):
        loss, grads = loss_and_grad(model, x, y, mask)
        if args.clip > 0:
            grads, _ = optim.clip_grad_norm(grads, args.clip)
        optimizer.update(model, grads)
        return loss

    step_fn = step_fn if args.no_compile else partial(mx.compile, inputs=state, outputs=state)(step_fn)

    batcher = iter(cifg.Batcher(train_tokens, train_offsets, args.batch_size, args.max_len, args.seed, repeats=repeats))
    log_path = os.path.join(run_dir, "log.jsonl")
    log = open(log_path, "a")
    budget = args.max_minutes * 60 if args.max_minutes else None
    total_steps = args.max_steps or 10**9
    schedule_total = total_steps
    best_ppl = float("inf")
    evals = []
    t_start = time.time()
    train_seconds = 0.0
    tokens_seen = 0
    step = 0
    window_loss = 0.0
    window_n = 0
    last_print = t_start
    last_tokens = 0

    def run_eval(tag: str):
        nonlocal best_ppl
        t0 = time.time()
        r = cifg.dev_perplexity(model, vocab, dev_tokens, dev_offsets, dev_ids, batch_size=args.eval_batch, max_len=args.max_len)
        if human_dev:
            r["human"] = human_perplexity({
                s: cifg.dev_perplexity(model, vocab, dev_tokens, dev_offsets, ids, batch_size=args.eval_batch, max_len=args.max_len)
                for s, ids in human_dev.items()
            })
        r.update({"step": step, "elapsed": round(time.time() - t_start, 1), "tokens": tokens_seen, "eval_seconds": round(time.time() - t0, 1)})
        sel = selection_ppl(r, args.select_on)
        if args.select_on != "pooled":
            r["selected_on"] = args.select_on
            r["selection_ppl"] = sel
        evals.append(r)
        model.save_weights(os.path.join(run_dir, "latest.safetensors"))
        improved = sel < best_ppl
        if improved:
            best_ppl = sel
            model.save_weights(os.path.join(run_dir, "best.safetensors"))
        log.write(json.dumps({"event": "eval", **r}) + "\n")
        log.flush()
        human_note = ""
        if human_dev:
            h = r["human"]
            human_note = (f"; human ppl {h['ppl']:.2f}, macro {h['macro_ppl']:.2f} ("
                          + ", ".join(f"{s} {v['ppl']:.1f}" for s, v in h["per_source"].items()) + ")")
        print(f"[eval {tag}] step {step} dev ppl {r['ppl']:.2f} (nll {r['nll']:.4f}, {r['targets']} targets, {r['eval_seconds']}s){human_note}{' *best' if improved else ''}", flush=True)
        return r

    plan = {"step": 0, "train_seconds": 0.0}

    def replan(tag: str):
        """Time-bounded runs: size the cosine so it reaches its floor near the deadline.

        Planned at the probe step and again at every periodic eval until the
        schedule is half done, from the training-only rate since the last plan
        and the time evals are expected to take from here on.
        """
        nonlocal schedule_total
        if budget is None:
            return
        if tag != "probe" and step < schedule_total and (step - args.warmup) / max(1, schedule_total - args.warmup) >= 0.5:
            return
        d_steps = step - plan["step"]
        d_secs = train_seconds - plan["train_seconds"]
        if d_steps <= 0 or d_secs <= 0:
            return
        rate = d_steps / d_secs
        remaining = budget - (time.time() - t_start)
        eval_cost = (sum(e["eval_seconds"] for e in evals) / len(evals)) if evals else None
        schedule_total = plan_steps(step, rate, remaining, eval_cost, args.eval_every, args.eval_reserve, total_steps, args.warmup)
        plan.update(step=step, train_seconds=train_seconds)
        config["schedule_total_steps"] = schedule_total
        print(f"[schedule/{tag}] {rate:.2f} steps/s over the last {d_steps} steps; cosine planned over {schedule_total} steps", flush=True)

    stop_reason = "max_steps"
    while step < total_steps:
        if budget is not None and time.time() - t_start >= budget:
            stop_reason = "max_minutes"
            break
        seqs = next(batcher)
        x, y, mask = cifg.make_batch(seqs, vocab, args.max_len)
        lr = lr_at(step, schedule_total, args.lr, args.warmup, args.min_lr_ratio)
        optimizer.learning_rate = mx.array(lr, dtype=mx.float32)
        t0 = time.time()
        loss = step_fn(mx.array(x), mx.array(y), mx.array(mask))
        mx.eval(loss, state)
        train_seconds += time.time() - t0
        lv = loss.item()
        step += 1
        n_tok = int(mask.sum())
        tokens_seen += n_tok
        window_loss += lv
        window_n += 1
        if step == args.probe_steps and budget is not None:
            replan("probe")
        if step % args.log_every == 0:
            now = time.time()
            rec = {"event": "train", "step": step, "loss": window_loss / window_n, "lr": lr, "tokens": tokens_seen, "elapsed": round(now - t_start, 1), "tok_per_s": round((tokens_seen - last_tokens) / max(1e-6, now - last_print))}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"[train] step {step} loss {rec['loss']:.4f} ppl {math.exp(rec['loss']):.1f} lr {lr:.2e} tokens {tokens_seen:,} {rec['elapsed']}s ({rec['tok_per_s']:,} tok/s)", flush=True)
            window_loss = 0.0
            window_n = 0
            last_print = now
            last_tokens = tokens_seen
        if step % args.eval_every == 0:
            run_eval("periodic")
            replan("eval")

    final = run_eval("final")
    keyboard = None
    if args.keyboard_lines != 0:
        best_model, _ = cifg.load_model(run_dir, "best")
        keyboard = evaluate_split(best_model, vocab, root, ctx["fingerprint"], "dev", batch_size=args.eval_batch,
                                  max_len=args.eval_max_len, pooled_lines=args.keyboard_lines, seed=args.seed + 1,
                                  columns=human, corpus=corpus)
        keyboard["checkpoint"] = "best"
        print_evaluation(keyboard, "dev, best checkpoint")
    config["schedule_total_steps"] = schedule_total
    summary = {
        "config": config,
        "steps": step,
        "stop_reason": stop_reason,
        # Training tokens over the split's raw tokens (the measure every earlier run reports), and
        # over what one oversampled batcher epoch feeds (lines x repeats, with END).
        "epochs": round(tokens_seen / max(1, train_offsets[-1]), 3),
        "batcher_epochs": round(tokens_seen / max(1, epoch_tokens), 3),
        "tokens_seen": tokens_seen,
        "wall_seconds": round(time.time() - t_start, 1),
        "train_seconds": round(train_seconds, 1),
        "final_dev": final,
        "selected_on": args.select_on,
        # The best value of the selection metric: pooled dev perplexity unless --select-on says otherwise.
        "best_dev_ppl": best_ppl,
        "evals": evals,
        "keyboard": keyboard,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.close()
    print(f"[done] {step} steps ({stop_reason}), {tokens_seen:,} tokens, {summary['epochs']} epochs ({summary['batcher_epochs']} batcher epochs), best {args.select_on} dev ppl {best_ppl:.2f}, wall {summary['wall_seconds']}s; metrics at {os.path.join(run_dir, 'metrics.json')}", flush=True)


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

def run_defaults(args, cfg: dict) -> None:
    """Fill --vocab and --corpus from the run's config.json when they aren't given.

    Two vocabularies of the same size load into the same checkpoint without
    complaint, so scoring a run against anything but its own vocabulary would
    be silently wrong. Runs from before the corpus was recorded used <root>/corpus.
    """
    if args.vocab is None and cfg.get("vocab_path"):
        args.vocab = cfg["vocab_path"]
    if args.corpus is None and cfg.get("corpus"):
        args.corpus = cfg["corpus"]


def cmd_eval(args):
    run_dir = os.path.join(args.root, "runs", args.name)
    with open(os.path.join(run_dir, "config.json")) as f:
        run_defaults(args, json.load(f))
    ctx = setup(args)
    root, vocab = ctx["root"], ctx["vocab"]
    model, cfg = cifg.load_model(run_dir, args.which)
    if cfg["vocab_size"] != len(vocab):
        raise SystemExit(f"checkpoint vocab {cfg['vocab_size']} != vocab file {len(vocab)}")
    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for split in splits:
        prepare(ctx, split, None, args.workers)
        ev = evaluate_split(model, vocab, root, ctx["fingerprint"], split, batch_size=args.eval_batch,
                            max_len=args.max_len, pooled_lines=args.keyboard_lines, seed=args.seed + 1,
                            columns=ctx["human_sources"], corpus=ctx["corpus"])
        print_evaluation(ev, f"{split}, {args.which} checkpoint")
        out = {"run": args.name, "checkpoint": args.which, "split": split, "params": cifg.param_count(model),
               "max_len": args.max_len, "vocab": vocab.path, "corpus": ctx["corpus"], "tokenizer": ctx["tokenizer_source"], **ev}
        path = os.path.join(run_dir, f"eval.{split}.{args.which}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[done] wrote {path}", flush=True)


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------

def cmd_prepare(args):
    ctx = setup(args)
    for split in args.splits.split(","):
        prepare(ctx, split, args.train_lines if split == "train" else None, args.workers)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("prepare", help="tokenize splits into the cache")
    add_common(pp)
    pp.add_argument("--splits", default="train,dev")
    pp.set_defaults(fn=cmd_prepare)

    pt = sub.add_parser("train", help="train a run")
    add_common(pt)
    pt.add_argument("--name", required=True)
    pt.add_argument("--embed-dim", type=int, default=cifg.EMBED_DIM)
    pt.add_argument("--hidden", type=int, default=cifg.HIDDEN)
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--max-len", type=int, default=64, help="tokens per line including END; longer lines are cut")
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.add_argument("--min-lr-ratio", type=float, default=0.1)
    pt.add_argument("--warmup", type=int, default=200)
    pt.add_argument("--weight-decay", type=float, default=0.01)
    pt.add_argument("--clip", type=float, default=1.0)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument("--max-minutes", type=float, default=None)
    pt.add_argument("--probe-steps", type=int, default=30, help="steps used to estimate the step budget for a time-bounded run")
    pt.add_argument("--eval-reserve", type=float, default=0.12, help="fraction of a time budget kept back for evals")
    pt.add_argument("--eval-every", type=int, default=500)
    pt.add_argument("--eval-batch", type=int, default=128)
    pt.add_argument("--dev-lines", type=int, default=20000, help="dev lines for periodic perplexity, which also picks best.safetensors (0 = all)")
    pt.add_argument("--keyboard-lines", type=int, default=5000, help="pooled dev lines for the final keyboard metrics (0 = skip; the per-source columns always use every line of that source)")
    pt.add_argument("--eval-max-len", type=int, default=128, help="tokens per item for the final keyboard metrics; no dev line exceeds 127")
    pt.add_argument("--human-oversample", type=int, default=HUMAN_OVERSAMPLE, help="how many times each line of the corpus's human-written sources (its manifest says which; nus and taskmaster in the v2 corpus) enters an epoch (1 = off)")
    pt.add_argument("--oversample", default=None, metavar="SRC=K,...", help="per-source factors that override --human-oversample, e.g. taskmaster3=1")
    pt.add_argument("--select-on", default="pooled", choices=SELECT_ON,
                    help="what picks best.safetensors: pooled dev perplexity on the --dev-lines sample (the default), "
                    "perplexity on every human-written dev line (human), or the mean of the per-source log perplexities "
                    "of the human-written sources, so each source counts once (human-macro)")
    pt.add_argument("--log-every", type=int, default=50)
    pt.add_argument("--seed", type=int, default=20260925)
    pt.add_argument("--no-compile", action="store_true")
    pt.set_defaults(fn=cmd_train)

    pe = sub.add_parser("eval", aliases=["score"], help="dev perplexity and keyboard metrics for a run")
    add_common(pe)
    pe.add_argument("--name", required=True)
    pe.add_argument("--which", default="best", choices=["best", "latest"])
    pe.add_argument("--split", default="dev", choices=["dev", "test", "both"])
    pe.add_argument("--max-len", type=int, default=128, help="tokens per item; no dev or test line exceeds 127, so nothing is cut by default")
    pe.add_argument("--eval-batch", type=int, default=128)
    pe.add_argument("--keyboard-lines", type=int, default=None, help="pooled lines for keyboard metrics (default all; the per-source columns always use every line of that source)")
    pe.add_argument("--seed", type=int, default=20260925)
    pe.set_defaults(fn=cmd_eval)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
