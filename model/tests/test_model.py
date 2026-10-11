"""Shape, parameter-count, statefulness and quantization checks for the CIFG model.

Run from the repository root so the standard library stays ahead of model/ on
sys.path:  python -m pytest model/tests
"""

import json

import math

import os

import sys

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

import numpy as np  # noqa: E402

import pytest  # noqa: E402

mx = pytest.importorskip("mlx.core")

import cifg  # noqa: E402

def test_forward_shape_and_state():
    model = cifg.CIFGLM(vocab_size=50, embed_dim=8, hidden=16)
    tokens = mx.array(np.random.default_rng(0).integers(0, 50, size=(2, 5)), dtype=mx.int32)
    logits, (c, r) = model(tokens)
    mx.eval(logits, c, r)
    assert logits.shape == (2, 5, 50)
    assert c.shape == (2, 16)
    assert r.shape == (2, 8)
    assert bool(mx.isfinite(logits).all())

def test_param_count_matches_production_shape():
    model = cifg.CIFGLM()
    n = cifg.param_count(model)
    assert n == cifg.expected_param_count()
    assert 1_900_000 < n < 2_100_000

def test_step_by_step_matches_full_sequence():
    model = cifg.CIFGLM(vocab_size=30, embed_dim=6, hidden=10)
    tokens = mx.array(np.random.default_rng(1).integers(0, 30, size=(3, 7)), dtype=mx.int32)
    full, _ = model(tokens)
    state = None
    pieces = []
    for t in range(7):
        out, state = model(tokens[:, t : t + 1], state)
        pieces.append(out)
    stepped = mx.concatenate(pieces, axis=1)
    mx.eval(full, stepped)
    assert np.allclose(np.array(full), np.array(stepped), atol=1e-5)

def test_loss_is_masked_mean_cross_entropy():
    model = cifg.CIFGLM(vocab_size=20, embed_dim=4, hidden=6)
    x = mx.zeros((2, 4), dtype=mx.int32)
    y = mx.ones((2, 4), dtype=mx.int32)
    mask = mx.array([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=mx.float32)
    loss = cifg.loss_fn(model, x, y, mask)
    mx.eval(loss)
    assert loss.shape == ()
    assert 0.0 < loss.item() < 10.0

def test_make_batch_adds_start_and_end(tmp_path):
    tokens = ["<pad>", "<s>", "</s>", "<unk>", "hey", "there", "ok"]
    path = str(tmp_path / "vocab.txt")
    with open(path, "w") as f:
        f.write("\n".join(tokens) + "\n")
    try:
        vocab = cifg.Vocab.load(path)
    finally:
        os.remove(path)
    assert (vocab.pad, vocab.start, vocab.end, vocab.unk) == (0, 1, 2, 3)
    assert list(vocab.sorted_words) == ["hey", "ok", "there"]
    assert vocab.prefix_range("th") == (2, 3)
    x, y, mask = cifg.make_batch([np.array([4, 5]), np.array([6])], vocab, max_len=8)
    assert x.shape == (2, 8) and y.shape == (2, 8)
    assert x[0, :3].tolist() == [1, 4, 5] and y[0, :3].tolist() == [4, 5, 2]
    assert mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert x[1, :2].tolist() == [1, 6] and y[1, :2].tolist() == [6, 2]
    assert mask[1].sum() == 2

def test_int8_roundtrip_error_is_bounded():
    w = np.random.default_rng(2).normal(size=(64, 96)).astype(np.float32)
    q, s = cifg.quantize_int8(w, axis=0)
    assert q.dtype == np.int8 and s.shape == (64, 1)
    back = cifg.dequantize_int8(q, s)
    assert np.abs(back - w).max() <= (np.abs(w).max(axis=1) / 127.0 / 2 + 1e-6).max()

def test_time_bounded_schedule_planning():
    import train

    # No eval cost measured yet: keep eval_reserve of the time back.
    assert train.plan_steps(30, 7.0, 570.0, None, 500, 0.12, 10**9, 200) == 30 + int(570 * 0.88 * 7.0)
    # With a measured eval cost, one eval per eval_every steps fits in the budget.
    n = train.plan_steps(500, 7.0, 500.0, 2.0, 500, 0.12, 10**9, 200)
    assert n == 500 + int((500.0 - 2.0) / (1.0 + 7.0 * 2.0 / 500) * 7.0)
    # Never shorter than the warmup, never past --max-steps, never behind the current step.
    assert train.plan_steps(30, 7.0, -5.0, None, 500, 0.12, 10**9, 200) == 201
    assert train.plan_steps(30, 7.0, 570.0, None, 500, 0.12, 1000, 200) == 1000
    assert train.plan_steps(4000, 7.0, 0.0, 2.0, 500, 0.12, 10**9, 200) == 4001
    # The learning rate reaches its floor at the planned end.
    assert abs(train.lr_at(3000, 3000, 2e-3, 200, 0.1) - 2e-4) < 1e-9
    assert abs(train.lr_at(199, 3000, 2e-3, 200, 0.1) - 2e-3) < 1e-9

def test_surface_tokens_mirror_keyboard_rules():
    assert cifg.surface_tokens("Don’t worry, it's OK 👍🏽!") == ["don't", "worry", "it's", "ok", "👍"]

def _write_vocab(tmp_path, tokens):
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return cifg.Vocab.load(str(path))

def test_offerable_words_follow_predict_rule(tmp_path):
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", ".", "?", "a", "i", "u", "k", "hi", "how", "are", "you"])
    assert cifg.offerable_word("u") is False and cifg.offerable_word("a") and cifg.offerable_word("i") and cifg.offerable_word("ok")
    # String.length counts UTF-16 units: one code point outside the BMP is two of them, so the device offers it.
    assert cifg.offerable_word("𝑎") and cifg.offerable_word("é") is False
    assert [vocab.tokens[j] for j in np.nonzero(vocab.offerable)[0]] == ["a", "i", "hi", "how", "are", "you"]
    assert [vocab.tokens[j] for j in vocab.sorted_ids[vocab.sorted_offerable]] == ["a", "are", "hi", "how", "i", "you"]
    assert [vocab.tokens[j] for j in vocab.boundary_ids] == [".", "?"]

def test_scope_items_split_at_sentence_ends(tmp_path):
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", ".", "!", "?", "…", "hi", "how", "are", "you", "ok"])
    ids = lambda *ws: [vocab.index[w] for w in ws]  # noqa: E731
    line1 = ids("hi", ".", "how", "are", "you", "?", "ok")
    line2 = ids("!", "ok", "…")
    tokens = np.array(line1 + line2, dtype=np.uint16)
    offsets = np.array([0, len(line1), len(line1) + len(line2)])
    assert cifg.scope_items(tokens, offsets, np.array([0, 1]), vocab.boundary_ids, "line") == [(0, 7), (7, 10)]
    # The boundary token belongs to no chunk; empty chunks are skipped; every chunk restarts from <s>.
    assert cifg.scope_items(tokens, offsets, np.array([0, 1]), vocab.boundary_ids, "sentence") == [(0, 1), (2, 5), (6, 7), (8, 9)]
    # Without boundary tokens in the vocabulary, sentence scope equals line scope.
    assert cifg.scope_items(tokens, offsets, np.array([1]), np.zeros(0, dtype=np.int64), "sentence") == [(7, 10)]

def test_keyboard_metrics_counts_scope_cut_and_one_letter_targets(tmp_path):
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", ".", "?", "a", "i", "u", "hi", "how", "are", "you", "ok", "🙂"])
    ids = lambda *ws: [vocab.index[w] for w in ws]  # noqa: E731
    # "hi u . how are you ? ok 🙂" then an OOV word of length 5 and an emoji-OOV.
    line = ids("hi", "u", ".", "how", "are", "you", "?", "ok", "🙂") + [vocab.unk, vocab.unk]
    oovlen = np.zeros(len(line), dtype=np.uint8)
    oovlen[-2] = 5  # an OOV word of five letters
    oovlen[-1] = 255  # an unknown emoji
    tokens = np.array(line, dtype=np.uint16)
    offsets = np.array([0, len(line)])
    mx.random.seed(0)
    model = cifg.CIFGLM(vocab_size=len(vocab), embed_dim=4, hidden=6)
    mx.eval(model.parameters())
    ln = cifg.keyboard_metrics(model, vocab, tokens, offsets, oovlen, np.array([0]), batch_size=4, max_len=64, scope="line")
    assert ln["items"] == 1
    # Word targets: hi u how are you ok + the OOV word = 7; "u" is one-letter unofferable, 🙂 and the emoji-OOV are skipped.
    assert ln["targets"] == 7 and ln["unofferable_one_letter_targets"] == 1 and ln["oov_targets"] == 1
    assert ln["in_vocab_targets"] == 5 and ln["non_word_targets_skipped"] == 4  # . ? 🙂 and the emoji-OOV
    assert ln["next"]["n"] == 7 and ln["cut"] == {"max_len": 64, "items_cut": 0, "targets_not_scored": 0}
    sn = cifg.keyboard_metrics(model, vocab, tokens, offsets, oovlen, np.array([0]), batch_size=4, max_len=64, scope="sentence")
    assert sn["items"] == 3 and sn["targets"] == 7 and sn["scope"] == "sentence"
    # A cut at 4 tokens per item scores only the first three tokens of the line and counts the rest.
    cut = cifg.keyboard_metrics(model, vocab, tokens, offsets, oovlen, np.array([0]), batch_size=4, max_len=4, scope="line")
    assert cut["cut"] == {"max_len": 4, "items_cut": 1, "targets_not_scored": 5}
    assert cut["targets"] == 2
    table = cifg.format_table({"pooled": ln, "nus": sn})
    assert table.splitlines()[0].split() == ["pooled", "nus"]
    assert "potential savings rate" in table
    assert cifg.format_table({"pooled": ln}, markdown=True).startswith("| | pooled |")

def test_batcher_repeats_oversample_lines():
    tokens = np.arange(30, dtype=np.uint16)
    offsets = np.arange(0, 31, 3)  # 10 lines of 3 tokens
    repeats = np.ones(10, dtype=np.int64)
    repeats[[2, 7]] = 4
    b = cifg.Batcher(tokens, offsets, batch_size=2, max_len=8, seed=1, pool_batches=8, repeats=repeats)
    assert len(b.indices) == 16
    assert b.indices.tolist().count(2) == 4 and b.indices.tolist().count(0) == 1
    seen = []
    it = iter(b)
    for _ in range(8):
        seen.extend(int(s[0]) // 3 for s in next(it))
    assert seen.count(2) == 4 and seen.count(7) == 4 and seen.count(0) == 1

def test_oversample_repeats_and_sources(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "train.sources.txt").write_text("nus\nsoda\ntaskmaster\nsoda\n", encoding="utf-8")
    codes, names = cifg.load_sources(str(tmp_path), "train")
    assert names == ["nus", "soda", "taskmaster"] and codes.tolist() == [0, 1, 2, 1]
    assert cifg.oversample_repeats(codes, names, 4, ("nus", "taskmaster")).tolist() == [4, 1, 4, 1]
    assert cifg.oversample_repeats(codes, names, 1, ("nus", "taskmaster")).tolist() == [1, 1, 1, 1]
    codes2, _ = cifg.load_sources(str(tmp_path), "train", max_lines=2)
    assert codes2.tolist() == [0, 1]
    assert cifg.surface_tokens("call 555-1234 now") == ["call", "now"]

def test_per_source_oversampling_and_factors(tmp_path):
    import train

    codes = np.array([0, 1, 2, 3, 1], dtype=np.int8)
    names = ["nus", "soda", "taskmaster", "taskmaster3"]
    factors = train.oversample_factors(("nus", "taskmaster", "taskmaster3"), 4, train.parse_oversample("taskmaster3=1"), names)
    assert factors == {"nus": 4, "taskmaster": 4, "taskmaster3": 1}
    assert cifg.oversample_repeats_by_source(codes, names, factors).tolist() == [4, 1, 4, 1, 1]
    # A factor of 1 or less is "no oversampling", as --human-oversample 1 always meant.
    assert cifg.oversample_repeats_by_source(codes, names, {"nus": 0}).tolist() == [1, 1, 1, 1, 1]
    with pytest.raises(SystemExit):
        train.oversample_factors(("nus",), 4, {"persona": 2}, names)
    with pytest.raises(SystemExit):
        train.parse_oversample("taskmaster3")

def test_corpus_dir_cache_key_and_human_sources(tmp_path):
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", "hi"])
    root = str(tmp_path)
    base = vocab.fingerprint("tok")
    # The default corpus keeps the key every existing cache and run config already uses.
    assert cifg.cache_fingerprint(vocab, "tok", root) == base
    assert cifg.cache_fingerprint(vocab, "tok", root, str(tmp_path / "corpus")) == base
    v3 = cifg.cache_fingerprint(vocab, "tok", root, str(tmp_path / "corpus-v3"))
    assert v3.startswith(base + "-corpus-v3-") and v3 != base
    assert cifg.corpus_dir(root) == str(tmp_path / "corpus")

    (tmp_path / "corpus-v3").mkdir()
    assert cifg.human_sources(str(tmp_path / "corpus-v3")) == ("nus", "taskmaster")  # no manifest: the v2 pair
    (tmp_path / "corpus-v3" / "manifest.json").write_text(json.dumps({"sources": {
        "nus": {"human_written": True}, "taskmaster": {"human_written": True},
        "oasst2": {"human_written": True}, "soda": {"human_written": False}}}), encoding="utf-8")
    assert cifg.human_sources(str(tmp_path / "corpus-v3")) == ("nus", "taskmaster", "oasst2")
    (tmp_path / "corpus-v3" / "dev.sources.txt").write_text("soda\noasst2\n", encoding="utf-8")
    codes, names = cifg.load_sources(root, "dev", corpus=str(tmp_path / "corpus-v3"))
    assert names == ["soda", "oasst2"] and codes.tolist() == [0, 1]

def test_selection_on_human_written_dev_lines():
    import train

    per_source = {
        "nus": {"nll": math.log(100.0), "ppl": 100.0, "targets": 100},
        "taskmaster3": {"nll": math.log(10.0), "ppl": 10.0, "targets": 900},
    }
    h = train.human_perplexity(per_source)
    # Pooled: every target once, so the big source dominates; macro: each source once.
    assert h["targets"] == 1000
    assert abs(h["ppl"] - math.exp(0.1 * math.log(100) + 0.9 * math.log(10))) < 1e-9
    assert abs(h["macro_ppl"] - math.sqrt(100 * 10)) < 1e-9
    record = {"ppl": 20.0, "human": h}
    assert train.selection_ppl(record, "pooled") == 20.0
    assert train.selection_ppl(record, "human") == h["ppl"]
    assert train.selection_ppl(record, "human-macro") == h["macro_ppl"]

    offsets = np.array([0, 2, 2, 5, 7, 9])  # five lines, the second empty
    codes = np.array([0, 0, 1, 2, 1], dtype=np.int8)
    sets = train.human_dev_sets(codes, ["nus", "soda", "oasst2"], offsets, ("nus", "oasst2", "taskmaster"), 1)
    assert {k: v.tolist() for k, v in sets.items()} == {"nus": [0], "oasst2": [3]}
    with pytest.raises(SystemExit):
        train.human_dev_sets(codes, ["soda", "x", "y"], offsets, ("nus",), 1)

def test_eval_defaults_to_the_runs_own_vocab_and_corpus():
    import argparse
    import train

    args = argparse.Namespace(vocab=None, corpus=None)
    train.run_defaults(args, {"vocab_path": "/v3/vocab.txt", "corpus": "/data/corpus-v3"})
    assert (args.vocab, args.corpus) == ("/v3/vocab.txt", "/data/corpus-v3")
    explicit = argparse.Namespace(vocab="/mine.txt", corpus=None)
    train.run_defaults(explicit, {"vocab_path": "/v3/vocab.txt"})  # an old run: no corpus recorded
    assert (explicit.vocab, explicit.corpus) == ("/mine.txt", None)

def test_dev_sample_reads_zero_as_all_and_refuses_an_empty_sample(tmp_path):
    import train

    offsets = np.array([0, 2, 2, 5, 6])  # four lines, the second one empty
    assert train.dev_sample(offsets, 0, 1).tolist() == [0, 2, 3]
    assert train.dev_sample(offsets, -1, 1).tolist() == [0, 2, 3]
    assert len(train.dev_sample(offsets, 2, 1)) == 2
    # An empty sample would score as perplexity 1.0 and freeze best.safetensors at the first eval.
    with pytest.raises(SystemExit):
        train.dev_sample(np.array([0, 0, 0]), 0, 1)
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", "hi"])
    model = cifg.CIFGLM(vocab_size=len(vocab), embed_dim=4, hidden=6)
    r = cifg.dev_perplexity(model, vocab, np.zeros(0, dtype=np.uint16), np.array([0]), np.zeros(0, dtype=np.int64))
    assert r["targets"] == 0 and np.isnan(r["ppl"])

def test_prepare_split_retokenizes_when_the_corpus_changes(tmp_path, monkeypatch):
    vocab = _write_vocab(tmp_path, ["<pad>", "<s>", "</s>", "<unk>", "hi", "how", "are", "you"])
    (tmp_path / "corpus").mkdir()
    dev = tmp_path / "corpus" / "dev.txt"
    dev.write_text("hi you\nhow are you\n", encoding="utf-8")
    meta = cifg.prepare_split(str(tmp_path), vocab, "fp", "dev", None, workers=1)
    assert meta["lines"] == 2 and meta["source_size"] == dev.stat().st_size

    def no_tokenizing(*_args, **_kwargs):
        raise AssertionError("an unchanged corpus file must come from the cache")

    with monkeypatch.context() as m:
        m.setattr(cifg, "_line_chunks", no_tokenizing)
        assert cifg.prepare_split(str(tmp_path), vocab, "fp", "dev", None, workers=1) == meta
    # A rebuilt corpus under the same vocabulary and tokenizer (same fingerprint) is tokenized again.
    mapped, _, _ = cifg.load_split(str(tmp_path), "fp", "dev", None)
    before = np.array(mapped)
    dev.write_text("hi\nhow are you\nhi you\n", encoding="utf-8")
    meta = cifg.prepare_split(str(tmp_path), vocab, "fp", "dev", None, workers=1)
    _, offsets, _ = cifg.load_split(str(tmp_path), "fp", "dev", None)
    assert meta["lines"] == 3 and len(offsets) == 4
    # A run that already mapped the old tokens keeps reading them: the new files replace the old by rename.
    assert np.array_equal(mapped, before)
    assert not [p for p in (tmp_path / "cache" / "fp").iterdir() if p.suffix == ".tmp"]
