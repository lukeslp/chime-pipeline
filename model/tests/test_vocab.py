import argparse

import json

from collections import Counter

import pytest

from model import tokenize as tk

from model import vocab

from model.tokenize import Tokenizer

def write_corpus(dir_, lines, sources):
    for split in ("train", "dev", "test"):
        (dir_ / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (dir_ / f"{split}.sources.txt").write_text("\n".join(sources) + "\n", encoding="utf-8")

def test_count_batch_separates_words_from_the_rest():
    words, other, n = vocab.count_batch((["Hi there! ☕ 12", "hi HI"], ["nus", "soda"]))
    assert n == 2
    assert words["nus"] == Counter({"hi": 1, "there": 1})
    assert words["soda"] == Counter({"hi": 2})
    assert other == Counter({"!": 1, "☕": 1, tk.NUM: 1})

def test_rank_words_is_count_then_alphabetical():
    ranked = vocab.rank_words(Counter({"b": 2, "a": 2, "c": 5}))
    assert ranked == [("c", 5), ("a", 2), ("b", 2)]

def test_reserved_layout_starts_with_specials_then_punct_then_emoji():
    reserved = vocab.reserved_tokens(["☕", "☕", "❤️"])
    assert tuple(reserved[: len(tk.SPECIALS)]) == tk.SPECIALS
    assert reserved[len(tk.SPECIALS): len(tk.SPECIALS) + len(tk.PUNCT)] == list(tk.PUNCT)
    assert reserved[-2:] == ["☕", "❤️"]

def test_build_writes_vocab_and_coverage(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = ["the cat sat", "the dog sat", "the end.", "a cat ☕ 12"]
    write_corpus(corpus, lines, ["nus", "soda", "soda", "soda"])
    monkeypatch.setattr(vocab, "ASSETS", tmp_path / "assets")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "emoji_en.txt").write_text("# c\ncoffee\t☕ 🍵\n", encoding="utf-8")
    (tmp_path / "assets" / "words_en.txt").write_text("# c\nthe\t100\ncat\t50\nzebra\t1\n", encoding="utf-8")

    reserved = len(tk.SPECIALS) + len(tk.PUNCT) + 2
    args = argparse.Namespace(corpus=str(corpus), out=str(tmp_path / "out"), size=reserved + 3,
                              sizes=[reserved + 1, reserved + 3, reserved + 100], limit=None, workers=1)
    report = vocab.build(args)

    t = Tokenizer.load(tmp_path / "out" / "vocab.txt")
    assert len(t) == reserved + 3
    assert t.decode(t.word_ids) == ["the", "cat", "sat"]  # 3, 2, 2 (cat before sat alphabetically)
    assert t.decode(t.emoji_ids) == ["☕", "🍵"]

    cov = json.loads((tmp_path / "out" / "coverage.json").read_text(encoding="utf-8"))
    assert cov == report
    assert cov["train_tokens"]["words"] == 10
    assert cov["train_tokens"]["word_types"] == 6
    assert cov["train_tokens"]["by_kind"]["emoji_in_train"] == {"☕": 1}
    assert cov["train_tokens"]["by_kind"]["num"] == 1
    assert cov["train_tokens"]["by_kind"]["punct"]["."] == 1
    c3 = cov["coverage"][str(reserved + 3)]
    assert c3["word_slots"] == 3
    assert c3["train"]["token_coverage"] == round(7 / 10, 6)
    assert c3["train_nus"]["token_coverage"] == 1.0
    assert cov["coverage"]["all_train_words"]["train"]["token_coverage"] == 1.0
    assert cov["coverage"][str(reserved + 1)]["train"]["token_coverage"] == round(3 / 10, 6)

    overlap = cov["dictionary_overlap"]["words_en"]
    assert overlap["entries"] == 3
    assert overlap["in_vocab"] == 2 and overlap["missing"] == 1
    assert overlap["missing_examples"][0]["word"] == "zebra"
    assert overlap["missing_unseen_in_train"] == 1
    assert cov["vocab_words_not_in_device_dictionary"]["count"] == 1  # "sat"
    assert (tmp_path / "out" / "word_counts.tsv").read_text(encoding="utf-8").splitlines()[0] == "the\t3"

def test_read_word_list_normalizes_and_merges(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("# header\nDon’t\t5\ndon't\t7\nx\n", encoding="utf-8")
    assert vocab.read_word_list(p) == {"don't": 12, "x": 0}

def test_include_forces_list_words_ahead_of_frequency(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    write_corpus(corpus, ["the cat sat", "the dog sat", "the end."], ["soda", "soda", "soda"])
    monkeypatch.setattr(vocab, "ASSETS", tmp_path / "assets")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "emoji_en.txt").write_text("coffee\t☕\n", encoding="utf-8")
    (tmp_path / "assets" / "words_en.txt").write_text("the\t100\n", encoding="utf-8")
    listing = tmp_path / "modern.txt"
    listing.write_text("# c\nzebra\t9\nWi-Fi\t5\n12\t1\nthe\t1\n", encoding="utf-8")

    reserved = len(tk.SPECIALS) + len(tk.PUNCT) + 1
    args = argparse.Namespace(corpus=str(corpus), out=str(tmp_path / "out"), size=reserved + 3,
                              sizes=[reserved + 3], limit=None, workers=1, include=[str(listing)])
    report = vocab.build(args)
    t = Tokenizer.load(tmp_path / "out" / "vocab.txt")
    # zebra, wi-fi and the come from the list ("12" is not a word) and fill every slot.
    assert t.decode(t.word_ids) == ["zebra", "wi-fi", "the"]
    assert report["chosen"]["included_words"] == 3
    assert report["chosen"]["included_never_in_train"] == 2
    assert report["chosen"]["min_train_count_kept"] is None  # no frequency-ranked word fit
    assert report["chosen"]["first_word_dropped"] == {"word": "sat", "count": 2}

def test_weights_rank_by_per_source_share():
    by_source = {"nus": Counter({"lol": 3, "the": 1}), "soda": Counter({"the": 90, "indeed": 10})}
    # Raw counts would put "indeed" (10) ahead of "lol" (3); equal weights rank by share of each source.
    ranked = vocab.rank_by_mix(by_source, {"nus": 1, "soda": 1})
    assert [w for w, _ in ranked] == ["the", "lol", "indeed"]
    assert abs(dict(ranked)["the"] - (1 / 4 + 90 / 100)) < 1e-9
    assert [w for w, _ in vocab.rank_by_mix(by_source, {"nus": 0, "soda": 1})] == ["the", "indeed"]
    with pytest.raises(ValueError):
        vocab.rank_by_mix(by_source, {"nus": 1})  # every source needs a weight
    assert vocab.parse_weights("nus=0.25, soda=0.75") == {"nus": 0.25, "soda": 0.75}
    assert vocab.parse_weights("") is None
    with pytest.raises(ValueError):
        vocab.parse_weights("nus")

def test_interleave_gives_the_second_list_its_share():
    primary = ["a", "b", "c", "d", "e", "f"]
    secondary = ["x", "b", "y", "z"]
    merged = vocab.interleave(primary, secondary, 0.5)
    # x (list), a (ranking), b (list), c (ranking), y, d, z, e, f
    assert merged == ["x", "a", "b", "c", "y", "d", "z", "e", "f"]
    assert vocab.interleave(primary, secondary, 0.0)[:6] == primary
    assert vocab.interleave(primary, secondary, 1.0)[:4] == secondary
    with pytest.raises(ValueError):
        vocab.interleave(primary, secondary, 1.5)

def test_build_with_weights_cover_and_baseline(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    lines = ["lol ok", "the cat sat on the mat", "the dog sat on the mat", "the end of the day"]
    write_corpus(corpus, lines, ["nus", "soda", "soda", "soda"])
    monkeypatch.setattr(vocab, "ASSETS", tmp_path / "assets")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "emoji_en.txt").write_text("coffee\t☕\n", encoding="utf-8")
    (tmp_path / "assets" / "words_en.txt").write_text("# c\nday\t50\ncat\t40\nzebra\t30\n", encoding="utf-8")
    base = tmp_path / "base.txt"
    base.write_text("\n".join(list(tk.SPECIALS) + list(tk.PUNCT) + ["the", "sat"]) + "\n", encoding="utf-8")

    reserved = len(tk.SPECIALS) + len(tk.PUNCT) + 1
    args = argparse.Namespace(corpus=str(corpus), out=str(tmp_path / "out"), size=reserved + 5, sizes=[reserved + 5],
                              limit=None, workers=1, weights="nus=1,soda=1",
                              cover=[str(tmp_path / "assets" / "words_en.txt")], cover_share=0.5, cover_min_count=1,
                              baseline=str(base))
    report = vocab.build(args)
    t = Tokenizer.load(tmp_path / "out" / "vocab.txt")
    # By share, nus's "lol" and "ok" (half of nus each) outrank soda's "the" (6 of 17 words); the
    # cover list ("day" 50, "cat" 40 in the list's own order; "zebra" never occurs) takes every other pick.
    assert t.decode(t.word_ids) == ["day", "lol", "cat", "ok", "the"]
    assert report["ranking"]["cover_candidates"] == 2 and report["ranking"]["cover_below_min_count"] == 1
    cov = report["coverage"][str(reserved + 5)]
    assert cov["dev_nus"]["token_coverage"] == 1.0
    assert cov["dev_soda"]["token_coverage"] == round(8 / 17, 6)
    assert report["baseline"]["words"] == 2
    assert report["baseline"]["dev_nus"]["token_coverage"] == 0.0
    assert report["baseline"]["train_soda"]["token_coverage"] == round(8 / 17, 6)
