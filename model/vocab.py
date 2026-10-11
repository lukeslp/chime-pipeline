#!/usr/bin/env python3
"""Build the word-level vocabulary for the LocalType next-word model.

Reads ``<corpus>/train.txt`` (with ``train.sources.txt`` beside it) through
``model.tokenize``, counts every token, and writes

    <out>/vocab.txt       one token per line, id = line number
    <out>/coverage.json   token coverage at several sizes, per source and on
                          dev/test, and the overlap with the keyboard's word list

The vocabulary is laid out as specials, punctuation, every emoji the keyboard
can offer, then words by descending train count (ties alphabetical) until the
requested size is full. Coverage is reported at 8k, 16k and 32k so the size is
a measured choice, not a guess.

Two options change the ranking, and both are off by default:

    --weights src=w,...   rank by a mixture of per-source word frequencies
                          (each source's counts divided by its own word total,
                          times its weight) instead of raw pooled counts, so a
                          small human-written source is not drowned by a large
                          synthetic one
    --cover FILE          interleave a keyboard word list with that ranking:
                          --cover-share of the word picks go to the list's
                          words in the list's own frequency order (only those
                          seen at least --cover-min-count times in train, so
                          every word gets some training signal), the rest to
                          the ranking itself; a word both want counts once

--baseline VOCAB scores an existing vocabulary on the same counts, so a
rebuild reports its coverage before and after.

Usage (from the repo root, so the package import works):

    python -m model.vocab                        # full build, 16,384 tokens
    python -m model.vocab --size 32768
    python -m model.vocab --limit 200000 --out ~/.local/share/localtype-model/vocab-smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import date
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Iterator

from model import tokenize as tk

DATA_ROOT = Path(os.environ.get("LOCALTYPE_MODEL_DATA", str(Path.home() / ".local" / "share" / "localtype-model")))
ASSETS = Path(__file__).resolve().parents[1] / "app" / "src" / "main" / "assets"
DEFAULT_SIZES = (8192, 16384, 32768)
BATCH_LINES = 20_000


# ------------------------------------------------------------------ keyboard assets


def read_emoji_list(path: Path) -> list[str]:
    """Every distinct emoji in emoji_en.txt, in first-seen order (the keyboard's parse rule)."""
    seen: dict[str, None] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            _, tab, rest = line.rstrip("\n").partition("\t")
            if not tab:
                continue
            for e in rest.strip().split(" "):
                if e:
                    seen.setdefault(e, None)
    return list(seen)


def read_word_list(path: Path) -> dict[str, int]:
    """word -> count from a keyboard word list (``word<TAB>count``, ``#`` comments)."""
    words: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            word, _, count = line.rstrip("\n").partition("\t")
            norm = tk.normalize(word.strip())
            try:
                n = int(count.strip() or 0)
            except ValueError:
                n = 0
            words[norm] = words.get(norm, 0) + n
    return words


# ------------------------------------------------------------------ counting


def count_batch(batch: tuple[list[str], list[str]]) -> tuple[dict[str, Counter], Counter, int]:
    """Count word tokens per source and every other token, for one batch of lines."""
    lines, sources = batch
    words: dict[str, Counter] = {}
    other: Counter = Counter()
    for line, src in zip(lines, sources):
        toks = tk.tokenize(line)
        w = [t for t in toks if tk.is_word(t)]
        if len(w) != len(toks):
            other.update(t for t in toks if not tk.is_word(t))
        words.setdefault(src, Counter()).update(w)
    return words, other, len(lines)


def batches(corpus: Path, split: str, limit: int | None) -> Iterator[tuple[list[str], list[str]]]:
    text = open(corpus / f"{split}.txt", encoding="utf-8")
    src_path = corpus / f"{split}.sources.txt"
    sources = open(src_path, encoding="utf-8") if src_path.exists() else None
    done = 0
    try:
        while True:
            take = BATCH_LINES if limit is None else min(BATCH_LINES, limit - done)
            if take <= 0:
                return
            lines = [l.rstrip("\n") for l in islice(text, take)]
            if not lines:
                return
            if sources is not None:
                srcs = [s.strip() for s in islice(sources, len(lines))]
            else:
                srcs = ["all"] * len(lines)
            done += len(lines)
            yield lines, srcs
    finally:
        text.close()
        if sources is not None:
            sources.close()


def count_split(corpus: Path, split: str, limit: int | None, workers: int) -> tuple[dict[str, Counter], Counter, int]:
    """Returns (words per source, other tokens, lines read)."""
    per_source: dict[str, Counter] = {}
    other: Counter = Counter()
    lines = 0
    t0 = time.time()

    def merge(result: tuple[dict[str, Counter], Counter, int]) -> None:
        nonlocal lines
        w, o, n = result
        for src, c in w.items():
            per_source.setdefault(src, Counter()).update(c)
        other.update(o)
        lines += n
        if lines % (BATCH_LINES * 25) == 0:
            print(f"  {split}: {lines:,} lines, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    if workers <= 1:
        for b in batches(corpus, split, limit):
            merge(count_batch(b))
    else:
        with Pool(workers) as pool:
            for result in pool.imap(count_batch, batches(corpus, split, limit)):
                merge(result)
    return per_source, other, lines


def total(per_source: dict[str, Counter]) -> Counter:
    c: Counter = Counter()
    for v in per_source.values():
        c.update(v)
    return c


# ------------------------------------------------------------------ assembling


def rank_words(counts: Counter) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def reserved_tokens(emoji: Iterable[str]) -> list[str]:
    toks = list(tk.SPECIALS) + list(tk.PUNCT)
    for e in emoji:
        if e not in toks:
            toks.append(e)
    return toks


def parse_weights(spec: str | None) -> dict[str, float] | None:
    """``nus=1,soda=0.5`` -> {"nus": 1.0, "soda": 0.5}; None or empty -> None (raw counts)."""
    if not spec:
        return None
    out: dict[str, float] = {}
    for part in spec.split(","):
        name, eq, value = part.strip().partition("=")
        if not eq or not name:
            raise ValueError(f"--weights wants source=weight pairs, got {part!r}")
        out[name.strip()] = float(value)
        if out[name.strip()] < 0:
            raise ValueError(f"negative weight for {name}")
    return out


def rank_by_mix(by_source: dict[str, Counter], weights: dict[str, float]) -> list[tuple[str, float]]:
    """Words by mixture score, the sum over sources of weight x the word's share of that source's words.

    Every source in the counts needs a weight, so a typo can't silently drop
    a source; a weight of 0 drops it on purpose. Ties are alphabetical, and a
    word seen only in zero-weight sources is left out.
    """
    missing = sorted(set(by_source) - set(weights))
    if missing:
        raise ValueError(f"--weights has no weight for {missing}; give each train source one (0 drops it)")
    score: dict[str, float] = {}
    for src, counts in by_source.items():
        w = weights[src]
        n = sum(counts.values())
        if w <= 0 or n == 0:
            continue
        for word, c in counts.items():
            score[word] = score.get(word, 0.0) + w * c / n
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def interleave(primary: list[str], secondary: list[str], share: float) -> list[str]:
    """Merge two rankings into one list of distinct words.

    ``secondary`` supplies ``share`` of the picks and ``primary`` the rest,
    each in its own order; a word already taken is skipped and does not count
    as a pick. Every prefix of the result is the merge at that size, so one
    merged list gives the vocabulary at every size.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must be between 0 and 1")
    out: list[str] = []
    seen: set[str] = set()
    i = j = taken2 = 0
    while i < len(primary) or j < len(secondary):
        use2 = j < len(secondary) and (i >= len(primary) or taken2 < share * (len(out) + 1))
        if use2:
            w = secondary[j]
            j += 1
        else:
            w = primary[i]
            i += 1
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if use2:
            taken2 += 1
    return out


def coverage_of(ranked: list[tuple[str, int]], slots: int, counters: dict[str, Counter]) -> dict[str, dict[str, float]]:
    """Token and type coverage of the top ``slots`` words against each counter."""
    return coverage_of_set({w for w, _ in ranked[:slots]}, counters)


def coverage_of_set(keep: set[str], counters: dict[str, Counter]) -> dict[str, dict[str, float]]:
    """Token and type coverage of a word set against each counter."""
    out: dict[str, dict[str, float]] = {}
    for name, c in counters.items():
        tokens = sum(c.values())
        types = len(c)
        hit_tokens = sum(n for w, n in c.items() if w in keep)
        hit_types = sum(1 for w in c if w in keep)
        out[name] = {
            "tokens": tokens,
            "types": types,
            "token_coverage": round(hit_tokens / tokens, 6) if tokens else None,
            "type_coverage": round(hit_types / types, 6) if types else None,
        }
    return out


def build(args: argparse.Namespace) -> dict:
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    emoji = read_emoji_list(ASSETS / "emoji_en.txt")
    reserved = reserved_tokens(emoji)
    print(f"reserved {len(reserved)} tokens: {len(tk.SPECIALS)} specials, {len(tk.PUNCT)} punct, "
          f"{len(reserved) - len(tk.SPECIALS) - len(tk.PUNCT)} emoji", file=sys.stderr)

    print(f"counting train ({args.workers} workers)", file=sys.stderr)
    train_by_src, train_other, train_lines = count_split(corpus, "train", args.limit, args.workers)
    train_words = total(train_by_src)
    ranked = rank_words(train_words)
    print(f"  train: {train_lines:,} lines, {sum(train_words.values()):,} word tokens, "
          f"{len(train_words):,} types, {time.time() - t0:.0f}s", file=sys.stderr)

    held: dict[str, Counter] = {}
    held_by_src: dict[str, Counter] = {}
    held_other: dict[str, Counter] = {}
    for split in ("dev", "test"):
        if (corpus / f"{split}.txt").exists():
            by_src, other, _ = count_split(corpus, split, args.limit, 1)
            held[split] = total(by_src)
            held_other[split] = other
            if len(by_src) > 1 or "all" not in by_src:
                held_by_src.update({f"{split}_{s}": c for s, c in sorted(by_src.items())})

    # Optional word lists forced into the vocabulary ahead of any ranking, so a
    # trainer can keep every strip candidate from a keyboard list even when the corpus
    # rarely or never uses it. Default off: the plain build is the measured one.
    included: list[str] = []
    for path in getattr(args, "include", None) or []:
        for w in read_word_list(Path(path)):
            if w and tk.is_word(w) and tk.tokenize(w) == [w] and w not in included:
                included.append(w)
    included_set = set(included)

    # The ranking: raw pooled counts by default, else the weighted mixture, then
    # optionally interleaved with keyboard word lists (--cover).
    weights = parse_weights(getattr(args, "weights", None))
    scored = rank_by_mix(train_by_src, weights) if weights else ranked
    order = [w for w, _ in scored]
    cover_words: list[str] = []
    cover_listed = cover_unseen = 0
    cover_share = getattr(args, "cover_share", 0.5)
    cover_min = getattr(args, "cover_min_count", 1)
    if getattr(args, "cover", None):
        listed: dict[str, int] = {}
        for path in args.cover:
            for w, n in read_word_list(Path(path)).items():
                if w and tk.is_word(w) and tk.tokenize(w) == [w]:
                    listed[w] = max(listed.get(w, 0), n)
        cover_listed = len(listed)
        cover_unseen = sum(1 for w in listed if train_words.get(w, 0) < cover_min)
        # The list's own frequency order (the keyboard's counts), not the corpus ranking: a list
        # ranked like the corpus is a subsequence of it, and interleaving it would change nothing.
        cover_words = sorted((w for w in listed if train_words.get(w, 0) >= cover_min),
                             key=lambda w: (-listed[w], w))
        order = interleave(order, cover_words, cover_share)
    order = included + [w for w in order if w not in included_set]

    sizes = sorted(set(list(args.sizes) + [args.size]))
    counters = {"train": train_words, **{f"train_{s}": c for s, c in sorted(train_by_src.items())},
                **held, **held_by_src}
    coverage = {}
    for size in sizes:
        slots = max(0, size - len(reserved))
        top = order[:slots]
        coverage[str(size)] = {
            "vocab_size": size,
            "word_slots": slots,
            "words_available": len(order),
            "min_train_count_kept": min((train_words.get(w, 0) for w in top), default=None) if slots <= len(order) else 0,
            **coverage_of_set(set(top), counters),
        }
    coverage["all_train_words"] = {
        "vocab_size": len(reserved) + len(ranked),
        "word_slots": len(ranked),
        "words_available": len(ranked),
        "min_train_count_kept": ranked[-1][1] if ranked else None,
        **coverage_of(ranked, len(ranked), counters),
    }

    slots = args.size - len(reserved)
    chosen = order[: max(0, slots)]
    vocab = reserved + chosen

    baseline = None
    if getattr(args, "baseline", None):
        with open(args.baseline, encoding="utf-8") as f:
            base_words = {t for t in (line.rstrip("\n") for line in f) if t and tk.is_word(t)}
        baseline = {"vocab_file": str(args.baseline), "words": len(base_words),
                    **coverage_of_set(base_words, counters)}
    tokenizer = tk.Tokenizer(vocab)
    tokenizer.save(out / "vocab.txt")

    # Overlap with the keyboard's word lists: what strip candidates the model can never produce.
    vocab_words = set(chosen)
    overlap: dict[str, dict] = {}
    lists = {
        "words_en": ["words_en.txt"],
        "words_en_modern": ["words_en_modern.txt"],
        "words_en_profanity": ["words_en_profanity.txt"],
        "device_dictionary": ["words_en.txt", "words_en_modern.txt", "words_en_profanity.txt"],
    }
    for name, files in lists.items():
        merged: dict[str, int] = {}
        raw_entries = 0
        for fn in files:
            path = ASSETS / fn
            if not path.exists():
                continue
            part = read_word_list(path)
            raw_entries += sum(1 for line in open(path, encoding="utf-8") if line.strip() and not line.startswith("#"))
            for w, n in part.items():
                merged[w] = max(merged.get(w, 0), n)
        mass = sum(merged.values())
        missing = {w: n for w, n in merged.items() if w not in vocab_words}
        missing_ranked = sorted(missing.items(), key=lambda kv: (-kv[1], kv[0]))
        overlap[name] = {
            "entries": raw_entries,
            "distinct_normalized": len(merged),
            "in_vocab": len(merged) - len(missing),
            "missing": len(missing),
            "missing_frequency_mass": round(sum(missing.values()) / mass, 6) if mass else None,
            "missing_unseen_in_train": sum(1 for w in missing if w not in train_words),
            "missing_examples": [{"word": w, "list_count": n, "train_count": train_words.get(w, 0)}
                                 for w, n in missing_ranked[:40]],
        }
    device_words = set()
    for fn in lists["device_dictionary"]:
        if (ASSETS / fn).exists():
            device_words |= set(read_word_list(ASSETS / fn))
    vocab_only = [w for w in chosen if w not in device_words]

    report = {
        "built_on": date.today().isoformat(),
        "corpus": str(corpus),
        "limit_lines": args.limit,
        "train_lines_read": train_lines,
        "vocab_file": str(out / "vocab.txt"),
        "vocab_size": len(vocab),
        "layout": {
            "specials": list(tk.SPECIALS),
            "punct": list(tk.PUNCT),
            "emoji": len(reserved) - len(tk.SPECIALS) - len(tk.PUNCT),
            "reserved": len(reserved),
            "word_slots": slots,
            "words_used": len(chosen),
        },
        "train_tokens": {
            "words": sum(train_words.values()),
            "word_types": len(train_words),
            "non_word": sum(train_other.values()),
            "by_kind": {
                "num": train_other.get(tk.NUM, 0),
                "sym": train_other.get(tk.SYM, 0),
                "punct": {p: train_other.get(p, 0) for p in tk.PUNCT},
                "emoji_in_train": {e: train_other[e] for e in emoji if train_other.get(e)},
                "emoji_tokens_total": sum(n for t, n in train_other.items() if tk.is_emoji(t)),
                "unknown_emoji_types": sum(1 for t in train_other if tk.is_emoji(t) and t not in reserved),
            },
        },
        "held_out_non_word": {s: {"num": c.get(tk.NUM, 0), "sym": c.get(tk.SYM, 0),
                                  "emoji": sum(n for t, n in c.items() if tk.is_emoji(t))}
                              for s, c in held_other.items()},
        "coverage": coverage,
        "ranking": {
            "weights": weights,
            "rule": ("raw pooled train counts" if not weights else
                     "sum over sources of weight x the word's share of that source's train word tokens"),
            "cover_files": [str(p) for p in (getattr(args, "cover", None) or [])],
            "cover_share": cover_share if cover_words else None,
            "cover_min_count": cover_min if cover_words else None,
            "cover_listed_words": cover_listed,
            "cover_candidates": len(cover_words),
            "cover_below_min_count": cover_unseen,
        },
        "baseline": baseline,
        "chosen": {
            "vocab_size": args.size,
            "word_slots": slots,
            "include_files": [str(p) for p in (getattr(args, "include", None) or [])],
            "included_words": len(included),
            "included_never_in_train": sum(1 for w in included if w not in train_words),
            "min_train_count_kept": min((train_words.get(w, 0) for w in chosen if w not in included_set), default=None),
            "first_word_dropped": next(({"word": w, "count": n} for w, n in ranked if w not in vocab_words), None),
            "words_with_apostrophe": sum("'" in w for w in chosen),
            "words_with_hyphen": sum("-" in w for w in chosen),
            "single_letter_words": [w for w in chosen if len(w) == 1],
        },
        "dictionary_overlap": overlap,
        "vocab_words_not_in_device_dictionary": {
            "count": len(vocab_only),
            "examples": [{"word": w, "train_count": train_words[w]} for w in vocab_only[:40]],
        },
        "seconds": round(time.time() - t0, 1),
    }
    with open(out / "coverage.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(out / "word_counts.tsv", "w", encoding="utf-8") as f:
        for w, n in ranked:
            f.write(f"{w}\t{n}\n")
    return report


def summarize(report: dict) -> str:
    lines = [f"vocab {report['vocab_size']} tokens -> {report['vocab_file']}"]
    lay = report["layout"]
    lines.append(f"  reserved {lay['reserved']} ({len(lay['specials'])} specials, {len(lay['punct'])} punct, "
                 f"{lay['emoji']} emoji), {lay['words_used']} words")
    tt = report["train_tokens"]
    lines.append(f"  train: {tt['words']:,} word tokens, {tt['word_types']:,} types, "
                 f"{tt['non_word']:,} non-word tokens, emoji tokens {tt['by_kind']['emoji_tokens_total']}")
    for size, cov in report["coverage"].items():
        cells = [f"{name} {c['token_coverage']:.4f}" for name, c in cov.items()
                 if isinstance(c, dict) and c.get("token_coverage") is not None]
        lines.append(f"  size {size} ({cov['word_slots']} word slots, min count {cov['min_train_count_kept']}): "
                     + ", ".join(cells))
    for name, o in report["dictionary_overlap"].items():
        lines.append(f"  {name}: {o['in_vocab']}/{o['distinct_normalized']} in vocab, {o['missing']} missing "
                     f"(frequency mass {o['missing_frequency_mass']}, {o['missing_unseen_in_train']} never in train)")
    base = report.get("baseline")
    if base:
        cells = [f"{name} {c['token_coverage']:.4f}" for name, c in base.items()
                 if isinstance(c, dict) and c.get("token_coverage") is not None]
        lines.append(f"  baseline {base['vocab_file']} ({base['words']} words): " + ", ".join(cells))
    lines.append(f"  {report['seconds']}s")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(DATA_ROOT / "corpus"))
    p.add_argument("--out", default=str(DATA_ROOT / "vocab"))
    p.add_argument("--size", type=int, default=16384, help="total vocabulary size including reserved tokens")
    p.add_argument("--sizes", default=",".join(map(str, DEFAULT_SIZES)),
                   help="comma-separated sizes to report coverage for")
    p.add_argument("--limit", type=int, default=None, help="read at most this many lines per split")
    p.add_argument("--include", action="append", default=[], metavar="FILE",
                   help="keyboard word list (word<TAB>count) whose words are always in the vocabulary; repeatable")
    p.add_argument("--weights", default=None, metavar="SRC=W,...",
                   help="rank words by a mixture of per-source frequencies with these weights (every train source "
                   "needs one; 0 drops it) instead of raw pooled counts")
    p.add_argument("--cover", action="append", default=[], metavar="FILE",
                   help="keyboard word list whose words get --cover-share of the word slots, interleaved with the ranking; repeatable")
    p.add_argument("--cover-share", type=float, default=0.5, help="share of word picks that go to --cover words")
    p.add_argument("--cover-min-count", type=int, default=1,
                   help="--cover words seen fewer times than this in train are left to the ranking")
    p.add_argument("--baseline", default=None, metavar="VOCAB",
                   help="an existing vocab.txt to score on the same counts, for a before and after")
    p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    args = p.parse_args(argv)
    args.sizes = [int(s) for s in args.sizes.split(",") if s]
    report = build(args)
    print(summarize(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
