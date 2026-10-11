import argparse

import json

import math

import os

import struct

import zlib

from collections import Counter, defaultdict

from itertools import islice

from pathlib import Path

import numpy as np

import pytest

from model import prior

from model import tokenize as tk

from model.prior import START, context_events, word_spans

def test_context_events_follow_the_keyboard():
    assert context_events("Hi there. How are you, Sam? ok") == [
        (START, "hi"), ("hi", "there"), (START, "how"), ("how", "are"), ("are", "you"), (None, "sam"), (START, "ok"),
    ]
    assert context_events("so 12 apples") == [(START, "so"), (None, "apples")]
    assert context_events("Don’t go") == [(START, "don't"), ("don't", "go")]
    assert context_events("well-known fact") == [(START, "well-known"), ("well-known", "fact")]
    # A word ending in an apostrophe or hyphen is not a context: the keyboard checks the last character.
    assert context_events("the dogs' bowl") == [(START, "the"), ("the", "dogs'"), (None, "bowl")]
    assert context_events("wait... what") == [(START, "wait"), (START, "what")]
    assert context_events("  so\tthen  ") == [(START, "so"), ("so", "then")]
    assert context_events("“quoted” word") == [(None, "quoted"), (None, "word")]
    assert context_events("") == []
    # ² is a letter to Python's \w but not to Java's \p{L}, so it splits the word and resets context.
    assert context_events("ab²cd ef") == [(START, "ab"), (None, "cd"), ("cd", "ef")]

TRICKY = [
    "Hi there. How are you, Sam? ok", "so 12 apples", "Don’t go", "the dogs' bowl", "wait... what",
    "  so\tthen  ", "“quoted” word", "ab²cd ef", "e-mail me -- now", "rock 'n' roll", "x²y z", "Ⅻ o'clock",
    "café au lait!Yes", "(aside) then", "A.B. Smith", "it’s ok?!no", "end.\tstart", "x  - y",
]

def device_contexts(line: str) -> list[tuple[str | None, str]]:
    """The O(n^2) way: what predictScored computes from the text before each word."""
    out = []
    for start, _, word in word_spans(line):
        before = line[:start]
        ctx = tk.last_completed_word(before)
        if ctx is None and tk.is_sentence_start(before):
            ctx = START
        out.append((ctx, word))
    return out

@pytest.mark.parametrize("line", TRICKY)
def test_context_events_match_the_device_mirrors(line):
    assert context_events(line) == device_contexts(line)

def test_word_spans_are_the_tokenizer_words():
    for line in TRICKY:
        assert [w for _, _, w in word_spans(line)] == [t for t in tk.tokenize(line) if tk.is_word(t)], line

def write_assets(dir_: Path, words: dict[str, int], modern: dict[str, int], swears: dict[str, int]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    for name, entries in (("words_en.txt", words), ("words_en_modern.txt", modern), ("words_en_profanity.txt", swears)):
        body = "# test list\n" + "".join(f"{w}\t{c}\n" for w, c in entries.items())
        (dir_ / name).write_text(body, encoding="utf-8")
    return dir_

def test_dictionary_drops_swears_and_keeps_device_words(tmp_path):
    assets = write_assets(tmp_path, {"You": 9, "it’s": 5, "damn": 4, "two words": 1, "u": 1}, {"ok": 3}, {"damn": 1})
    words, info = prior.load_dictionary(assets)
    assert words == {"you", "it's", "u", "ok"}
    assert info["swear_words_excluded"] == 1
    assert set(info["files"]) == {"words_en.txt", "words_en_modern.txt", "words_en_profanity.txt"}

def write_corpus(dir_: Path, lines: list[str], sources: list[str]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dir_ / "train.sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")
    return dir_

def test_count_batch_weights_human_lines():
    words, keys, weighted, raw = prior.count_batch((["Hi there", "hi there, you"], [4, 1]))
    pairs = {(words[k >> prior.SHIFT], words[k & 0xFFFFFFFF]): (int(w), int(r)) for k, w, r in zip(keys, weighted, raw)}
    assert pairs == {(START, "hi"): (5, 2), ("hi", "there"): (5, 2)}  # "you" follows a comma: no context

def test_count_pairs_merges_batches_and_counts_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(prior, "BATCH_LINES", 2)
    corpus = write_corpus(tmp_path, ["a cat", "a cat", "a dog", "the cat"], ["soda", "nus", "soda", "taskmaster"])
    pairs = prior.count_pairs(corpus, None, 1, 4)
    got = {(pairs.words[c], pairs.words[w]): (int(n), int(r))
           for c, w, n, r in zip(pairs.ctx, pairs.word, pairs.counts, pairs.raw)}
    assert got == {(START, "a"): (6, 3), ("a", "cat"): (5, 2), ("a", "dog"): (1, 1),
                   (START, "the"): (4, 1), ("the", "cat"): (4, 1)}
    assert pairs.lines == 4 and pairs.weighted_lines == 10
    assert prior.count_pairs(corpus, 3, 1, 4).lines == 3

def test_weighting_needs_a_line_parallel_sources_file(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "train.txt").write_text("a cat\nthe cat\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="missing"):
        prior.count_pairs(corpus, None, 1, 4)  # without sources every line would count once under a header saying 4
    assert prior.count_pairs(corpus, None, 1, 1).weighted_lines == 2
    (corpus / "train.sources.txt").write_text("nus\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="shorter"):
        prior.count_pairs(corpus, None, 1, 4)
    (corpus / "train.sources.txt").write_text("nus\nsoda\nsoda\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="longer"):
        prior.count_pairs(corpus, None, 1, 4)
    assert prior.count_pairs(corpus, 2, 1, 4).weighted_lines == 5  # a limit stops before the extra source line

def test_estimate_discounts_uses_the_count_of_counts():
    raw = np.array([1] * 40 + [2] * 10 + [3] * 5 + [4] * 2 + [9] * 3)
    y = 40 / (40 + 2 * 10)
    expected = (1 - 2 * y * 10 / 40, 2 - 3 * y * 5 / 10, 3 - 4 * y * 2 / 5)
    assert prior.estimate_discounts(raw) == pytest.approx(expected)
    with pytest.raises(ValueError):
        prior.estimate_discounts(np.array([1, 1, 2, 3]))  # no fours

def test_select_keeps_top_contexts_and_offerable_followers(tmp_path):
    assets = write_assets(tmp_path / "assets", {"you": 9, "u": 8, "ok": 7, "see": 6, "so": 5, "it": 4}, {}, {"damn": 1})
    lines = ["see you", "see you", "see u", "see damn", "see so", "you ok", "you so", "damn it", "so it", "ok"]
    corpus = write_corpus(tmp_path / "corpus", lines, ["soda"] * len(lines))
    dictionary, _ = prior.load_dictionary(assets)
    pairs = prior.count_pairs(corpus, None, 1, 4)
    chosen = prior.select(pairs, prior.kneser_ney(pairs, (0.5, 1.0, 1.5)), dictionary, contexts=2, followers=2)
    by_word = {chosen.words[c]: [chosen.words[w] for w, _ in lst] for c, lst in chosen.contexts}
    # "see" (5 followers) and "you" (2) are the contexts, plus <s>; "so" (1) and the swear "damn" (1) are not.
    assert set(by_word) == {START, "see", "you"}
    assert by_word["see"] == ["you", "so"]  # "u" is never offered and "damn" is a swear
    # One count each, but "so" follows three distinct words and "ok" two, so the Kneser-Ney lower order puts "so" first.
    assert by_word["you"] == ["so", "ok"]
    assert by_word[START] == ["see", "you"]  # at most two followers
    assert "damn" not in chosen.words and "u" not in chosen.words
    assert chosen.words == sorted(chosen.words, key=prior.utf16_key)
    for _, lst in chosen.contexts:
        assert [lp for _, lp in lst] == sorted((lp for _, lp in lst), reverse=True)

def test_quantize_error_is_at_most_half_a_step():
    values = np.linspace(-11.3, -0.01, 1000)
    q, lo, step = prior.quantize(values)
    back = prior.dequantize(q, lo, step)
    assert q.min() == 0 and q.max() == 255
    assert np.abs(back - values).max() <= step / 2 + 1e-6
    q1, lo1, step1 = prior.quantize([-2.0, -2.0])
    assert step1 == 0 and prior.dequantize(q1, lo1, step1).tolist() == [-2.0, -2.0]
