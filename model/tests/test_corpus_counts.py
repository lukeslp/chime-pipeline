"""The per-source counter that the Kotlin corpus scorer diffs against (model/corpus_counts.py).

The hand-made lines and the expected output are repeated in CorpusEvaluationTest.kt, so a change
to either tokenizer fails one side or the other.
"""

import numpy as np

import pytest

pytest.importorskip("mlx.core")

from model import corpus_counts as cc

from model import tokenize as tk

LINES = [
    "Hi u. How are you?? ok 🙂 at 12.30",
    "Done!! Really?! , . yes",
    "   ",
    "wait… what. k lol ☕ 5€",
    "pneumonoultramicroscopicsilicovolcanoconiosis 𝑥",
]

SOURCES = ["nus", "taskmaster", "soda", "nus", "soda"]

EXPECTED = [
    "nus lines=2 sentences=6 targets=11 one_letter=2 skipped=5 punct=0 num=2 emoji=2 sym=1 p1=9 p2=6 p3=2 p4=0 p5+=0 baseline=39",
    "taskmaster lines=1 sentences=4 targets=3 one_letter=0 skipped=1 punct=1 num=0 emoji=0 sym=0 p1=3 p2=3 p3=2 p4=1 p5+=1 baseline=16",
    "soda lines=1 sentences=1 targets=2 one_letter=0 skipped=0 punct=0 num=0 emoji=0 sym=0 p1=1 p2=1 p3=1 p4=1 p5+=12 baseline=48",
]

def test_counts_on_hand_made_lines():
    counts = cc.count_lines(LINES, SOURCES)
    # Sources in order of first appearance on a line with a token; the blank soda line counts nowhere.
    # 𝑥 is one code point but two UTF-16 units, so like the device the counter doesn't call it one-letter.
    assert [cc.format_counts(s, c) for s, c in counts.items()] == EXPECTED

def test_every_token_has_a_kind():
    assert [cc.kind(t) for t in tk.tokenize("Hi, 12 🙂 € …")] == ["word", "punct", "num", "emoji", "sym", "punct"]

def test_main_reads_a_split(tmp_path, capsys):
    (tmp_path / "dev.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    (tmp_path / "dev.sources.txt").write_text("\n".join(SOURCES) + "\n", encoding="utf-8")
    assert cc.main(["dev", "--corpus", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines() == EXPECTED
    (tmp_path / "test.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "test.sources.txt").write_text("nus\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        cc.main(["test", "--corpus", str(tmp_path)])

def test_counts_match_cifg_keyboard_metrics(tmp_path, monkeypatch):
    """The counter sees the sentences, targets, skipped tokens and ladder that cifg.keyboard_metrics scores."""
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from model import cifg

    # "k", 𝑥 and the long word are left out of the vocabulary, as are ☕ and €.
    words = ["a", "i", "u", "hi", "how", "are", "you", "ok", "at", "done", "really", "yes", "wait", "what", "lol"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join([*tk.SPECIALS, ".", "!", "?", "…", ",", "🙂", *words]) + "\n", encoding="utf-8")
    vocab = cifg.Vocab.load(str(path))
    monkeypatch.setattr(cifg, "_worker_tok", None)
    monkeypatch.setattr(cifg, "_worker_vocab", None)
    cifg._init_worker(str(path))
    tokens, lens, oovlen = cifg._encode_chunk(LINES)
    offsets = np.concatenate([[0], np.cumsum(lens)])
    mx.random.seed(0)
    model = cifg.CIFGLM(vocab_size=len(vocab), embed_dim=4, hidden=6)
    mx.eval(model.parameters())
    counts = cc.count_lines(LINES, SOURCES)
    for source, c in counts.items():
        line_ids = np.array([i for i, s in enumerate(SOURCES) if s == source and lens[i] > 0])
        m = cifg.keyboard_metrics(model, vocab, tokens, offsets, oovlen, line_ids, batch_size=4, max_len=64, scope="sentence")
        assert (m["lines"], m["items"], m["targets"], m["non_word_targets_skipped"]) == (c["lines"], c["sentences"], c["targets"], c["skipped"]), source
        assert [m["completion"][p]["n"] for p in ("1", "2", "3", "4", "5+")] == [c["p1"], c["p2"], c["p3"], c["p4"], c["p5+"]], source
    # cifg counts the one-letter "k" outside its vocabulary as OOV; the counter counts it as one-letter.
    nus = cifg.keyboard_metrics(model, vocab, tokens, offsets, oovlen, np.array([0, 3]), batch_size=4, max_len=64, scope="sentence")
    assert (nus["unofferable_one_letter_targets"], nus["oov_targets"], counts["nus"]["one_letter"]) == (1, 1, 2)
