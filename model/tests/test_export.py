"""The flat binary for the Kotlin port, the float64 reference behind its golden files, and the
committed fixtures under app/src/test/fixtures.

Run from the repository root:  python -m pytest model/tests
"""

import os

import sys

from pathlib import Path

MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

import numpy as np  # noqa: E402

import pytest  # noqa: E402

mx = pytest.importorskip("mlx.core")

import cifg  # noqa: E402

import export  # noqa: E402

def test_bin_round_trip_and_layout():
    q = export.tiny_quantized()
    data = export.bin_bytes(q, export.TINY_TOKENS)
    assert data[:4] == b"LTCF"
    V, E, H = len(export.TINY_TOKENS), export.TINY_EMBED, export.TINY_HIDDEN
    assert export.HEADER.unpack_from(data, 0)[1:5] == (1, V, E, H)
    back, tokens = export.read_bin(data)
    assert tokens == export.TINY_TOKENS
    for k, v in q.items():
        assert np.array_equal(back[k], v.reshape(back[k].shape)), k
    # Every int8 block is padded to a multiple of 4: V*E = 112 needs none, E*3H = 60 none, H*E = 20 none,
    # so the size is the plain sum here; the odd case is checked below.
    vocab = len("".join(t + "\n" for t in tokens).encode())
    assert len(data) == 24 + 4 * V + V * E + 2 * (4 * E + E * 3 * H) + 4 * H + H * E + 12 * H + vocab + 4

def test_bin_pads_odd_matrices_and_refuses_damage():
    rng = np.random.default_rng(0)
    tokens = ["<pad>", "<s>", "</s>", "<unk>", "<num>", "<sym>", "hi"]
    f32 = {"embedding": rng.normal(size=(7, 3)), "wx": rng.normal(size=(3, 3)), "wr": rng.normal(size=(3, 3)),
           "b": rng.normal(size=3), "proj": rng.normal(size=(1, 3))}
    q = export.quantize_all({k: v.astype(np.float32) for k, v in f32.items()})
    data = export.bin_bytes(q, tokens)
    back, back_tokens = export.read_bin(data)
    assert back_tokens == tokens and np.array_equal(back["wx_q"], q["wx_q"])
    damaged = bytearray(data)
    damaged[30] ^= 1
    with pytest.raises(ValueError, match="CRC"):
        export.read_bin(bytes(damaged))
    with pytest.raises(ValueError):
        export.bin_bytes(q, tokens[:-1])

def test_reference_logits_match_the_mlx_model():
    q = export.tiny_quantized()
    w = export.dequantize64(q)
    V = len(export.TINY_TOKENS)
    model = cifg.CIFGLM(vocab_size=V, embed_dim=export.TINY_EMBED, hidden=export.TINY_HIDDEN)
    export.load_into_model(model, export.dequantize_all(q))
    ids = [1, 23, 17, 24, 6, 1, 19]
    logits, _ = model(mx.array([ids], dtype=mx.int32))
    mx.eval(logits)
    full = np.array(logits)[0]
    for t in range(1, len(ids) + 1):
        assert np.allclose(export.reference_logits(w, ids[:t]), full[t - 1], atol=1e-4), t

def test_context_tokens_are_the_current_sentence_without_the_typed_word():
    ctx = export.context_tokens
    assert ctx("") == ([], "")
    assert ctx("Thanks for the h") == (["thanks", "for", "the"], "h")
    assert ctx("Done. The ") == (["the"], "")
    assert ctx("Wait... what") == ([], "what")
    assert ctx("first line\nsecond ") == (["second"], "")
    assert ctx("I don’t t") == (["i", "don't"], "t")
    assert ctx("at 12:30, ok ") == (["at", "<num>", ",", "ok"], "")
    long_ctx, _ = ctx(" ".join(["one", "two", "three"] * 40) + " ")
    assert len(long_ctx) == 120 - 64 and long_ctx[-3:] == ["one", "two", "three"]
    tk = export.tokenize_module().Tokenizer(export.TINY_TOKENS)
    ids, prefix = export.context_ids(tk, "hey ❤ y")
    assert ids == [tk.bos_id, tk.ids["hey"], tk.ids["❤️"]] and prefix == "y"

def test_the_window_jumps_rather_than_slides():
    ws = export.window_start
    assert [ws(n) for n in (0, 1, 63)] == [0, 0, 0]
    assert [ws(n) for n in (64, 95, 96, 127, 128)] == [32, 32, 64, 64, 96]
    assert all(32 <= n - ws(n) <= 63 for n in range(64, 400))
    assert [ws(n, hop=1) for n in (63, 64, 70)] == [0, 1, 7]
    with pytest.raises(ValueError):
        ws(70, hop=0)
    # A session following a sentence re-reads the window only when its start moves: once per 32 tokens.
    moves = [n for n in range(1, 400) if ws(n) != ws(n - 1)]
    assert moves == list(range(64, 400, 32))
    words, _ = export.context_tokens(export.counting(96) + " ")
    assert words == list(export.COUNTING[4:]) + list(export.COUNTING) * 2 + list(export.COUNTING[:6])
    assert export.context_tokens("Done. " + export.counting(64) + " th") == (export.context_tokens(export.counting(64) + " ")[0], "th")

def test_offerable_follows_the_kotlin_rule():
    assert [t for t in export.TINY_TOKENS if export.offerable(t)] == [
        "i", "a", "the", "to", "you", "thanks", "that", "don't", "hello", "help", "hey", "we", "it"]
    assert not export.offerable("u") and not export.offerable("<s>") and not export.offerable("❤️")
    assert export.offerable("ça") and not export.offerable("x²")

def test_escape_keeps_rows_on_one_line():
    s = "a\tb\nc\\d\re\x1ff g\x85h"
    e = export.escape(s)
    assert "\t" not in e and "\n" not in e and "\r" not in e and " " not in e and "\x85" not in e
    assert e == "a\\tb\\nc\\\\d\\re\\u001ff\\u2028g\\u0085h"

def test_golden_rows_rank_and_filter():
    q = export.tiny_quantized()
    rows, gap = export.golden_rows(q, export.TINY_TOKENS, ["Hello th", "hey ❤ "])
    assert gap > 1e-3
    fields = rows[0].split("\t")
    assert fields[0] == "C" and fields[1] == "Hello th" and fields[2] == "th"
    assert fields[3] == "1 23"
    logits = [float(x) for x in fields[5].split()]
    assert logits == sorted(logits, reverse=True) and len(logits) == export.TOP_IDS
    assert len(fields[6].split()) == export.TOP_WORDS
    assert sorted(fields[7].split()) == ["thanks", "that", "the"]

def test_cifg_files_are_never_replaced_without_force(tmp_path):
    q = export.tiny_quantized()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "dev.txt").write_text("Hello there, how are you?\nI am fine. Thanks\n", encoding="utf-8")
    out = tmp_path / "export"
    out.mkdir()
    report = export.write_cifg_files(str(out), q, export.TINY_TOKENS, str(corpus), 2, 2, 7, False, "test")
    assert report["golden_contexts"] == len(export.FIXED_CONTEXTS) + 2 and report["golden_token_lines"] == 2
    assert sorted(p.name for p in out.iterdir()) == ["cifg_golden.tsv", "cifg_golden_tokens.tsv", "cifg_int8.bin", "cifg_int8.json"]
    before = (out / "cifg_int8.bin").read_bytes()
    with pytest.raises(SystemExit):
        export.write_cifg_files(str(out), q, export.TINY_TOKENS, str(corpus), 2, 2, 7, False, "test")
    assert (out / "cifg_int8.bin").read_bytes() == before
    export.write_cifg_files(str(out), q, export.TINY_TOKENS, str(corpus), 2, 2, 7, True, "test")

def test_bin_only_writes_to_out_and_leaves_the_export_alone(tmp_path):
    root = tmp_path / "root"
    export_dir = root / "runs" / "tiny" / "export"
    export_dir.mkdir(parents=True)
    (root / "corpus").mkdir()
    (root / "corpus" / "dev.txt").write_text("Hello there, how are you?\nI am fine. Thanks\n", encoding="utf-8")
    np.savez(export_dir / "weights_int8.npz", **export.tiny_quantized())
    (export_dir / "vocab.txt").write_text("".join(t + "\n" for t in export.TINY_TOKENS), encoding="utf-8")
    before = sorted(p.name for p in export_dir.iterdir())
    out = tmp_path / "elsewhere"
    args = ["--root", str(root), "--name", "tiny", "--bin-only", "--out", str(out), "--golden-contexts", "2", "--token-lines", "2"]
    export.main(args)
    assert sorted(p.name for p in export_dir.iterdir()) == before
    assert (out / "cifg_int8.bin").read_bytes() == export.bin_bytes(export.tiny_quantized(), export.TINY_TOKENS)
    header = {r.split("\t")[1]: r.split("\t")[2:] for r in (out / "cifg_golden.tsv").read_text(encoding="utf-8").splitlines() if r.startswith("H")}
    assert header["context_tokens"] == ["64"] and header["context_hop"] == ["32"]
    with pytest.raises(SystemExit):
        export.main(args)
    with pytest.raises(SystemExit):
        export.main(["--root", str(root), "--name", "tiny", "--out", str(out)])
