"""Per-source counts of what the keyboard metrics score in sentence scope, through model/tokenize.py.

    python -m model.corpus_counts dev [--corpus DIR]

Prints one line per source of the split, in the order the sources first appear:
the lines that hold a token, the sentences (cifg's sentence-scope items), the
word targets, the one-letter targets the strip never offers, the non-word tokens
skipped inside sentences (in total and by kind), the ladder positions at prefix
lengths 1 to 4 and 5+ (offerable targets only, up to prefix 16) and the baseline
characters (each word's length plus one). No ranker affects any of these.

The Kotlin corpus scorer (CorpusEvaluationTest) writes the same lines for the
same split to app/build/reports/corpus-eval/<split>.counts.txt, so parity
between its tokenizer and this one is a plain diff. A one-letter word is counted
under one_letter here whatever the vocabulary holds; cifg's eval JSON counts one
outside the vocabulary as OOV instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Iterable

from model import tokenize as tk
from model.cifg import SENTENCE_END, data_root, offerable_word

MAX_PREFIX = 16
KINDS = ("punct", "num", "emoji", "sym")
FIELDS = ("lines", "sentences", "targets", "one_letter", "skipped", *KINDS, "p1", "p2", "p3", "p4", "p5+", "baseline")


def kind(token: str) -> str:
    """word, punct, num, emoji or sym: the five kinds tokenize() emits."""
    if tk.is_word(token):
        return "word"
    if token == tk.NUM:
        return "num"
    if token == tk.SYM:
        return "sym"
    if token in tk.PUNCT:
        return "punct"
    if tk.is_emoji(token):
        return "emoji"
    raise ValueError(f"tokenize() gave a token of no known kind: {token!r}")


def add_line(tokens: list[str], c: Counter) -> None:
    """Adds the tokens of one corpus line to `c`."""
    c["lines"] += 1
    sentence: list[str] = []
    # As cifg.scope_items: split at . ! ? …, the mark itself belongs to neither side, and an empty piece is no sentence.
    for token in [*tokens, None]:
        if token is not None and token not in SENTENCE_END:
            sentence.append(token)
            continue
        if not sentence:
            continue
        c["sentences"] += 1
        for t in sentence:
            k = kind(t)
            if k != "word":
                c["skipped"] += 1
                c[k] += 1
                continue
            c["targets"] += 1
            c["baseline"] += len(t) + 1
            if not offerable_word(t):
                c["one_letter"] += 1
                continue
            for p in range(1, min(len(t), MAX_PREFIX + 1)):
                c[f"p{p}" if p < 5 else "p5+"] += 1
        sentence = []


def count_lines(lines: Iterable[str], sources: Iterable[str]) -> dict[str, Counter]:
    """Counts per source, in the order the sources first appear. A line without a token is skipped, as train.sample_lines skips it."""
    out: dict[str, Counter] = {}
    for line, source in zip(lines, sources, strict=True):
        tokens = tk.tokenize(line)
        if tokens:
            add_line(tokens, out.setdefault(source.strip(), Counter()))
    return out


def format_counts(source: str, c: Counter) -> str:
    return source + " " + " ".join(f"{f}={c[f]}" for f in FIELDS)


def read_lines(path: str) -> list[str]:
    """One entry per line, as cifg._line_chunks reads a split."""
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("split", nargs="?", default="dev", help="dev, test or train (default dev)")
    ap.add_argument("--corpus", default=os.path.join(data_root(), "corpus"), help="the corpus directory (default <root>/corpus)")
    args = ap.parse_args(argv)
    lines = read_lines(os.path.join(args.corpus, f"{args.split}.txt"))
    sources = read_lines(os.path.join(args.corpus, f"{args.split}.sources.txt"))
    if len(lines) != len(sources):
        raise SystemExit(f"{len(lines)} corpus lines but {len(sources)} source labels")
    for source, c in count_lines(lines, sources).items():
        print(format_counts(source, c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
