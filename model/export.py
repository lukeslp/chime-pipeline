#!/usr/bin/env python3
"""Export a trained run as plain float32 .npz, an int8-quantized variant, and the
flat binary the Kotlin port (CifgModel.kt) loads, with golden outputs to check it.

    python model/export.py --name smoke [--which best] [--check-lines 5000]
    python model/export.py --name smoke-10min-v2 --bin-only     # binary + golden from an existing export
    python model/export.py --name smoke-10min-v2 --bin-only --out DIR   # the same, written to DIR
    python model/export.py --fixtures app/src/test/fixtures      # the tiny committed test model

Writes <root>/runs/<name>/export/:

    weights_f32.npz    embedding (V, E), wx (E, 3H), wr (E, 3H), b (3H,), proj (H, E)
    weights_int8.npz   the four matrices as int8 with a float32 scale per row
                       (embedding, wx, wr: axis 0; proj: axis 0), b as float32
    vocab.txt          the vocabulary the ids refer to, copied
    export.json        parameter count, bytes per tensor, file sizes and, with
                       --check-lines, dev perplexity for float32 and for the
                       int8 weights dequantized back into the model
    cifg_int8.bin      the int8 weights and the vocabulary in one little-endian
                       file (layout in BIN_LAYOUT below and in model/README.md)
    cifg_golden.tsv    for fixed and dev-sampled contexts: the context ids, the
                       top-10 ids and logits over the whole vocabulary, and the
                       top-6 offerable words without and with the typed prefix,
                       all computed in float64 from the dequantized int8 weights
    cifg_golden_tokens.tsv   dev lines with their tokens, for the Kotlin tokenizer

--bin-only reads weights_int8.npz and vocab.txt from an earlier export and writes
only the cifg_* files, to the export directory or to --out; it refuses to replace one
that exists unless --force.

Gate order in wx / wr / b is [forget, output, update]; the input gate is 1 - forget.
No LiteRT conversion is attempted here; the Kotlin port runs the int8 weights itself.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _HERE:
    sys.path.pop(0)
if _HERE not in sys.path:
    sys.path.append(_HERE)

import argparse  # noqa: E402
import functools  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
import zlib  # noqa: E402

import numpy as np  # noqa: E402

import cifg  # noqa: E402
import mlx.core as mx  # noqa: E402

TENSORS = ("embedding", "wx", "wr", "b", "proj")
MATRICES = ("embedding", "wx", "wr", "proj")

# --------------------------------------------------------------------------
# The flat binary the Kotlin port loads
# --------------------------------------------------------------------------

BIN_NAME = "cifg_int8.bin"
BIN_MAGIC = b"LTCF"
BIN_VERSION = 1
BIN_LAYOUT = """\
All numbers little-endian. V = vocabulary size, E = embedding width, H = hidden units.

    offset 0   4 bytes  magic "LTCF"
               u32      version (1)
               u32      V
               u32      E
               u32      H
               u32      N, the byte length of the vocabulary section
    then, for embedding (V x E), wx (E x 3H), wr (E x 3H) and proj (H x E) in that order:
               f32[rows]       one scale per row (symmetric: weight = q * scale)
               i8[rows * cols] the matrix, row-major
               0-3 zero bytes  so the next section starts on a multiple of 4
    then       f32[3H]  b, the gate bias
               N bytes  the vocabulary, UTF-8, every token followed by a newline; id = line number
               u32      CRC-32 (zlib) of every byte before it

Gate order in wx, wr and b is [forget, output, update], and the input gate is 1 - forget.
"""
HEADER = struct.Struct("<4s5I")


def _pad4(n: int) -> int:
    return (-n) % 4


def bin_bytes(q: dict[str, np.ndarray], tokens: list[str]) -> bytes:
    """Serialize int8 weights (quantize_all's layout) and their vocabulary as BIN_LAYOUT."""
    V, E = q["embedding_q"].shape
    H = q["proj_q"].shape[0]
    shapes = {"embedding": (V, E), "wx": (E, 3 * H), "wr": (E, 3 * H), "proj": (H, E)}
    for name, shape in shapes.items():
        if q[f"{name}_q"].shape != shape or q[f"{name}_scale"].reshape(-1).shape != (shape[0],):
            raise ValueError(f"{name}: expected {shape} with {shape[0]} scales")
    if q["b"].shape != (3 * H,):
        raise ValueError(f"b: expected ({3 * H},)")
    if len(tokens) != V:
        raise ValueError(f"{len(tokens)} tokens for a {V}-row embedding")
    if any("\n" in t or not t for t in tokens):
        raise ValueError("a token is empty or holds a newline")
    vocab = "".join(t + "\n" for t in tokens).encode("utf-8")
    parts = [HEADER.pack(BIN_MAGIC, BIN_VERSION, V, E, H, len(vocab))]
    for name in MATRICES:
        m = np.ascontiguousarray(q[f"{name}_q"], dtype=np.int8)
        parts += [np.ascontiguousarray(q[f"{name}_scale"].reshape(-1), dtype="<f4").tobytes(), m.tobytes(), b"\0" * _pad4(m.nbytes)]
    parts += [np.ascontiguousarray(q["b"], dtype="<f4").tobytes(), vocab]
    body = b"".join(parts)
    return body + struct.pack("<I", zlib.crc32(body))


def read_bin(data: bytes) -> tuple[dict[str, np.ndarray], list[str]]:
    """The inverse of bin_bytes, checking the magic, version, length and CRC."""
    magic, version, V, E, H, n_vocab = HEADER.unpack_from(data, 0)
    if magic != BIN_MAGIC or version != BIN_VERSION:
        raise ValueError(f"not a version {BIN_VERSION} {BIN_MAGIC!r} file")
    (crc,) = struct.unpack_from("<I", data, len(data) - 4)
    if crc != zlib.crc32(data[:-4]):
        raise ValueError("CRC mismatch")
    q: dict[str, np.ndarray] = {}
    pos = HEADER.size
    for name, (rows, cols) in (("embedding", (V, E)), ("wx", (E, 3 * H)), ("wr", (E, 3 * H)), ("proj", (H, E))):
        q[f"{name}_scale"] = np.frombuffer(data, dtype="<f4", count=rows, offset=pos).astype(np.float32)
        pos += 4 * rows
        q[f"{name}_q"] = np.frombuffer(data, dtype=np.int8, count=rows * cols, offset=pos).reshape(rows, cols).copy()
        pos += rows * cols + _pad4(rows * cols)
    q["b"] = np.frombuffer(data, dtype="<f4", count=3 * H, offset=pos).astype(np.float32)
    pos += 12 * H
    tokens = data[pos : pos + n_vocab].decode("utf-8").split("\n")
    if tokens[-1] != "" or len(tokens) != V + 1 or pos + n_vocab + 4 != len(data):
        raise ValueError("vocabulary section does not match the header")
    return q, tokens[:-1]


# --------------------------------------------------------------------------
# Context encoding and the float64 reference the golden files come from
# --------------------------------------------------------------------------

# A context is <s> plus at most CONTEXT_TOKENS - 1 tokens of the current sentence, the
# length train.py cuts lines at (--max-len 64). A longer sentence is read from a start
# that moves CONTEXT_HOP tokens at a time (window_start), so a Kotlin session following
# the sentence re-reads its window once every CONTEXT_HOP tokens rather than on every word.
CONTEXT_TOKENS = 64
CONTEXT_HOP = 32
SENTENCE_END_TOKENS = (".", "!", "?", "…")


@functools.lru_cache(maxsize=None)
def tokenize_module():
    """model/tokenize.py, loaded by path as cifg.load_tokenizer does, so it never shadows the stdlib."""
    spec = importlib.util.spec_from_file_location("localtype_tokenize", os.path.join(_HERE, "tokenize.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def window_start(length: int, limit: int = CONTEXT_TOKENS - 1, hop: int = CONTEXT_HOP) -> int:
    """Where the context starts in a sentence of ``length`` tokens (CifgTokenizer.windowStart).

    0 while the sentence fits in ``limit`` tokens, then the smallest multiple of ``hop`` that
    leaves at most ``limit``: a 63-token sentence is read whole, a 64-token one from token 32,
    a 96-token one from token 64. ``hop=1`` is a window that slides one token at a time.
    """
    if not 1 <= hop <= limit:
        raise ValueError(f"hop {hop} must be in 1..{limit}")
    return 0 if length <= limit else -(-(length - limit) // hop) * hop


def context_tokens(text: str, limit: int = CONTEXT_TOKENS - 1, hop: int = CONTEXT_HOP) -> tuple[list[str], str]:
    """The tokens a model sees for the text before the cursor, and the normalized typed prefix.

    The word being typed (activePrefix) is left out and returned as the prefix. The context
    is the current sentence: what follows the last newline, then the tokens after the last
    . ! ? or … token. That is the split keyboard_metrics(scope="sentence") makes, so for a
    sentence of up to ``limit`` tokens (every word target in dev) the context is exactly what
    the evaluation scores; past that, keyboard_metrics keeps reading from the sentence start
    (up to its max_len - 1 = 127 tokens) and this reads from window_start.

    It is not the typing log's Context Scope rule (docs/prediction-log-spec.md,
    PredictionLogSession). The log cuts its stored text after the last . ! ? or newline
    character, so it cuts inside a number ("I paid 3.5 for the " keeps "5 for the "), where
    the tokenizer reads 3.5 as one <num>, and it does not cut at …, which ends a sentence
    here. A replay of logged tails through CifgModel.encode has to allow for both.
    """
    tk = tokenize_module()
    prefix = tk.active_prefix(text)
    before = text[: len(text) - len(prefix)]
    toks = tk.tokenize(before.rsplit("\n", 1)[-1])
    for i in range(len(toks) - 1, -1, -1):
        if toks[i] in SENTENCE_END_TOKENS:
            toks = toks[i + 1 :]
            break
    return toks[window_start(len(toks), limit, hop) :], tk.normalize(prefix)


def context_ids(tokenizer, text: str) -> tuple[list[int], str]:
    toks, prefix = context_tokens(text)
    return [tokenizer.bos_id] + [tokenizer.id_of(t) for t in toks], prefix


def dequantize64(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {name: q[f"{name}_q"].astype(np.float64) * q[f"{name}_scale"].astype(np.float64).reshape(-1, 1) for name in MATRICES}
    out["b"] = q["b"].astype(np.float64)
    return out


def reference_logits(w: dict[str, np.ndarray], ids: list[int]) -> np.ndarray:
    """CIFGLM.__call__ for one sequence from the zero state, in float64; logits after the last id."""
    H = w["proj"].shape[0]
    c = np.zeros(H)
    r = np.zeros(w["embedding"].shape[1])
    for t in ids:
        g = w["embedding"][t] @ w["wx"] + r @ w["wr"] + w["b"]
        f = 1.0 / (1.0 + np.exp(-g[:H]))
        o = 1.0 / (1.0 + np.exp(-g[H : 2 * H]))
        u = np.tanh(g[2 * H :])
        c = f * c + (1.0 - f) * u
        r = (o * np.tanh(c)) @ w["proj"]
    return w["embedding"] @ r


def offerable(token: str) -> bool:
    """A word the keyboard could offer, by Kotlin's rules: the whole token matches
    [\\p{L}][\\p{L}'’-]* and its UTF-16 length is 2 or more, or it is a or i
    (AdaptiveNgramModel.predict). Specials, punctuation and emoji are not words."""
    if not token or not unicodedata.category(token[0]).startswith("L"):
        return False
    if not all(unicodedata.category(ch).startswith("L") or ch in "'’-" for ch in token):
        return False
    return len(token.encode("utf-16-le")) // 2 >= 2 or token in ("a", "i")


def top_words(logits: np.ndarray, tokens: list[str], offer_ids: list[int], prefix: str, k: int) -> list[str]:
    """The k best offerable words that start with prefix, by logit, ties to the lower id."""
    cand = [i for i in offer_ids if tokens[i].startswith(prefix)]
    cand.sort(key=lambda i: (-logits[i], i))
    return [tokens[i] for i in cand[:k]]


# --------------------------------------------------------------------------
# Golden files: line-based, tab-separated, so the JVM tests parse them with no library
# --------------------------------------------------------------------------

GOLDEN_NAME = "cifg_golden.tsv"
GOLDEN_TOKENS_NAME = "cifg_golden_tokens.tsv"
GOLDEN_FORMAT = "localtype-cifg-golden"
TOP_IDS = 10
TOP_WORDS = 6


_ESCAPES = {"\\": "\\\\", "\t": "\\t", "\n": "\\n", "\r": "\\r"}


def escape(s: str) -> str:
    """Backslash escapes for \\ tab newline CR, and \\uXXXX for every other control character and
    U+2028/U+2029, so no reader splits a row anywhere but at the newline that ends it."""
    out = []
    for ch in s:
        if ch in _ESCAPES:
            out.append(_ESCAPES[ch])
        elif ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F or ch in "  ":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def golden_rows(q: dict[str, np.ndarray], tokens: list[str], contexts: list[str]) -> tuple[list[str], float]:
    """One C row per context, and the smallest logit gap among each context's top 11.

    C <before cursor> <prefix> <context ids> <top-10 ids> <their logits> <top-6 words> <top-6 words with the prefix>
    """
    tokenizer = tokenize_module().Tokenizer(tokens)
    w = dequantize64(q)
    offer_ids = [i for i, t in enumerate(tokens) if offerable(t)]
    rows = []
    min_gap = float("inf")
    for text in contexts:
        ids, prefix = context_ids(tokenizer, text)
        logits = reference_logits(w, ids)
        order = np.argsort(-logits, kind="stable")[: TOP_IDS + 1]
        min_gap = min(min_gap, float(np.min(-np.diff(logits[order]))))
        top = order[:TOP_IDS]
        rows.append("\t".join([
            "C",
            escape(text),
            escape(prefix),
            " ".join(str(i) for i in ids),
            " ".join(str(int(i)) for i in top),
            " ".join(f"{logits[i]:.9g}" for i in top),
            escape(" ".join(top_words(logits, tokens, offer_ids, "", TOP_WORDS))),
            escape(" ".join(top_words(logits, tokens, offer_ids, prefix, TOP_WORDS))),
        ]))
    return rows, min_gap


def golden_text(header: list[tuple[str, ...]], rows: list[str]) -> str:
    lines = ["\t".join(("H", "format", GOLDEN_FORMAT, "1"))] + ["\t".join(("H",) + h) for h in header] + rows
    return "\n".join(lines) + "\n"


def token_rows(texts: list[str]) -> list[str]:
    """T <text> <tokenize(text)> <typed prefix> <context_tokens(text)>; tokens never hold a space."""
    tk = tokenize_module()
    rows = []
    for text in texts:
        ctx, prefix = context_tokens(text)
        rows.append("\t".join(["T", escape(text), escape(" ".join(tk.tokenize(text))), escape(prefix), escape(" ".join(ctx))]))
    return rows


# Contexts every golden file includes: sentence scope, the prefix, emoji, numbers, a long sentence.
FIXED_CONTEXTS = [
    "",
    "I",
    "I ",
    "Thanks for the ",
    "Thanks for the h",
    "see you tomorrow at ",
    "Done. The ",
    "Wait... what are you ",
    "Really?! I can't ",
    "I don’t think so ",
    "He said “hello” and ",
    "Meet at 12:30 near the ",
    "I paid $1,000 for ",
    "love you ❤ ",
    "love you ❤️ so ",
    "great job 👍🏽 see ",
    "first line\nsecond ",
    "first line.\n",
    "e-mail me when you're re",
    " ".join(["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"] * 8) + " ",
]


def sample_contexts(corpus_file: str, n: int, seed: int) -> list[str]:
    """n dev lines cut at a random character, so some end mid-word and some after a space."""
    with open(corpus_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    rng = np.random.default_rng(seed)
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    picks = rng.choice(len(nonempty), size=min(n, len(nonempty)), replace=False)
    out = []
    for p in sorted(int(x) for x in picks):
        line = lines[nonempty[p]]
        out.append(line[: int(rng.integers(0, len(line) + 1))])
    return out


def sample_token_texts(corpus_file: str, n: int, seed: int) -> list[str]:
    """Half whole dev lines (the full tokenizer), half cut at a random character (the context rule)."""
    with open(corpus_file, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    rng = np.random.default_rng(seed + 1)
    picks = sorted(int(x) for x in rng.choice(len(lines), size=min(n, len(lines)), replace=False))
    return [lines[p] if k % 2 == 0 else lines[p][: int(rng.integers(0, len(lines[p]) + 1))] for k, p in enumerate(picks)]


def _write(path: str, data: bytes, force: bool) -> None:
    if os.path.exists(path) and not force:
        raise SystemExit(f"{path} exists; pass --force to replace it")
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def write_cifg_files(out: str, q: dict[str, np.ndarray], tokens: list[str], corpus_dir: str, contexts: int,
                     token_lines: int, seed: int, force: bool, source: str) -> dict:
    """cifg_int8.bin, cifg_golden.tsv, cifg_golden_tokens.tsv and cifg_int8.json under out."""
    names = (BIN_NAME, GOLDEN_NAME, GOLDEN_TOKENS_NAME, "cifg_int8.json")
    if not force:
        taken = [n for n in names if os.path.exists(os.path.join(out, n))]
        if taken:
            raise SystemExit(f"{', '.join(taken)} already in {out}; pass --force to replace")
    t0 = time.time()
    data = bin_bytes(q, tokens)
    digest = hashlib.sha256(data).hexdigest()
    dev = os.path.join(corpus_dir, "dev.txt")
    ctx = FIXED_CONTEXTS + (sample_contexts(dev, contexts, seed) if contexts and os.path.exists(dev) else [])
    rows, gap = golden_rows(q, tokens, ctx)
    V, E = q["embedding_q"].shape
    header = [
        ("model", BIN_NAME, digest),
        ("shape", str(V), str(E), str(q["proj_q"].shape[0])),
        ("context_tokens", str(CONTEXT_TOKENS)),
        ("context_hop", str(CONTEXT_HOP)),
        ("min_top11_gap", f"{gap:.6g}"),
        ("source", escape(source)),
        ("contexts", str(len(FIXED_CONTEXTS)), f"fixed, then {len(ctx) - len(FIXED_CONTEXTS)} dev lines cut at a random character, seed {seed}"),
    ]
    texts = sample_token_texts(dev, token_lines, seed) if token_lines and os.path.exists(dev) else []
    _write(os.path.join(out, BIN_NAME), data, force)
    _write(os.path.join(out, GOLDEN_NAME), golden_text(header, rows).encode("utf-8"), force)
    _write(os.path.join(out, GOLDEN_TOKENS_NAME), golden_text([("source", escape(dev))], token_rows(texts)).encode("utf-8"), force)
    report = {
        "file": BIN_NAME,
        "layout_version": BIN_VERSION,
        "bytes": len(data),
        "sha256": digest,
        "source": source,
        "golden_contexts": len(rows),
        "golden_min_top11_gap": gap,
        "golden_token_lines": len(texts),
        "context_tokens": CONTEXT_TOKENS,
        "context_hop": CONTEXT_HOP,
        "seconds": round(time.time() - t0, 1),
    }
    _write(os.path.join(out, "cifg_int8.json"), (json.dumps(report, indent=2) + "\n").encode(), force)
    return report


# --------------------------------------------------------------------------
# The tiny committed model: a few hundred random parameters, so the JVM tests
# check parity on every run without the data volume
# --------------------------------------------------------------------------

TINY_TOKENS = [
    "<pad>", "<s>", "</s>", "<unk>", "<num>", "<sym>",
    ".", ",", "?", "!", "…", "'", "❤️", "😀",
    "i", "a", "u", "the", "to", "you", "thanks", "that", "don't", "hello", "help", "hey", "we", "it",
]
TINY_EMBED, TINY_HIDDEN, TINY_SEED = 4, 5, 20261006  # a seed whose top-11 logits are all 0.0036 or more apart
TINY_CONTEXTS = [
    "",
    "h",
    "hello",
    "hello ",
    "Hello th",
    "Thanks you ",
    "I don't know. you ",
    "I don’t t",
    "hey ❤ ",
    "hey ❤️ ",
    "hey 😀😀 w",
    "we 12:30 t",
    "it.\nthe ",
    "it!\nU",
    "you, you, the ",
    "Zebra quokka ",
    "the the the the the the the the the the the the the ",
    "hey #1 & ",
    "WE ARE HE",
    "a",
    # Sentences past the 63-token window: 70 tokens and a typed prefix, then 95 and 96 (window_start 32 and 64).
    " ".join(["we", "help", "you", "to", "the", "hey", "it"] * 10) + " th",
    " ".join(["thanks", "to", "you", "we", "help"] * 19) + " ",
    " ".join(["thanks", "to", "you", "we", "help"] * 19) + " it ",
]
TINY_BIN = "cifg_tiny.bin"
TINY_GOLDEN = "cifg_tiny_golden.tsv"
TOKENIZER_CASES = "cifg_tokenizer_cases.tsv"

# Hand-written tokenizer cases for the Kotlin port; no corpus text, so the file can be committed.
COUNTING = ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def counting(n: int, sep: str = " ") -> str:
    """n counting words joined by sep: a sentence of known length for the context window cases."""
    return sep.join(COUNTING[i % len(COUNTING)] for i in range(n))


TOKENIZER_TEXTS = [
    "", " ", "hello", "Hello World", "HELLO wor", "hello ", "\t hi \n",
    "don't", "Don’t you", "rock'n'roll", "'tis", "-hello", "hello-", "e-mail", "o’", "‘this’ and “that”",
    "12:30", "1,000.50", "3.5", "abc123def", "v2", "٣٤ apples", "1/2", "10%", "$5", "#1", "@me", "a&b",
    "...", "..", ". . .", "…", "‥", "wait...what", "?!", "!!!", "??", ",,", "--", "—", "a – b", "1 - 2",
    "。", "、", "，", "：", "？", "！", "（hi）", "¿qué?", "¡sí!",
    "x²", "x²y", "²abc", "a'²b", "²'a", "Ⅻ", "½ cup", "❶ first",
    "café", "cafe\u0301", "naïve", "Straße", "İstanbul", "ΟΔΟΣ", "ǅungla", "𝒜bc", "日本語", "한국어 단어",
    "😀", "😀😀", "❤", "❤️", "❤︎", "👍🏽", "🧑‍💻", "👨‍👩‍👧", "🏳️‍🌈", "🇺🇸", "🇺", "🇺🇸🇬", "🏴\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "#️⃣", "1⃣", "1️⃣2", "*️⃣", "©", "®", "™", "↔️", "☀", "⭐", "〰", "Ⓜ", "㊗", "\u200d", "\ufe0f", "a\u200db",
    "<3", ":)", ";-)", "°C", "[x]", "~", "^", "\\", "|", "_", "+=", "<>", "€5", "£", "¥", "§",
    "a\u00a0b", "a\u0085b", "a\u2028b", "a\u3000b", "a\x1fb", "a\u200bb", "a\u2060b", "\x00",
    "I ", "I'm going to the ", "I’m going to the st", "Thanks. See you ", "OK!\nsee you l", "first.\nsecond. third ",
    "Done. The", "Done.The", "What?? Really... ok ", "x\n..y", "a.\n.", "a,\n,b", "U.S. army ", "mr. smith ",
    "hey 😀😀 w", "it costs 12,5 € ", "call 555-1234 now", "I'm 25 years ", "2nd place ", "the 90s ",
    # The context window: sentences of 63, 64, 95, 96 and 130 words, one that ends in a typed word,
    # one after a sentence end, and one where commas make 79 tokens of 40 words.
    counting(63) + " ", counting(64) + " ", counting(95) + " ", counting(96) + " ", counting(130) + " ",
    counting(70) + " th", "Done. " + counting(64) + " ", counting(40, ", ") + " ",
]


def tiny_quantized(seed: int = TINY_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    V, E, H = len(TINY_TOKENS), TINY_EMBED, TINY_HIDDEN
    f32 = {
        "embedding": rng.normal(0.0, 2.0, (V, E)),
        "wx": rng.normal(0.0, 0.8, (E, 3 * H)),
        "wr": rng.normal(0.0, 0.8, (E, 3 * H)),
        "b": rng.normal(0.0, 0.5, (3 * H,)),
        "proj": rng.normal(0.0, 1.0, (H, E)),
    }
    return quantize_all({k: v.astype(np.float32) for k, v in f32.items()})


def fixture_files() -> dict[str, bytes]:
    """The committed test fixtures, by file name: the tiny model, its golden file, the tokenizer cases."""
    q = tiny_quantized()
    data = bin_bytes(q, TINY_TOKENS)
    rows, gap = golden_rows(q, TINY_TOKENS, TINY_CONTEXTS)
    header = [
        ("model", TINY_BIN, hashlib.sha256(data).hexdigest()),
        ("shape", str(len(TINY_TOKENS)), str(TINY_EMBED), str(TINY_HIDDEN)),
        ("context_tokens", str(CONTEXT_TOKENS)),
        ("context_hop", str(CONTEXT_HOP)),
        ("min_top11_gap", f"{gap:.6g}"),
        ("source", f"random weights, seed {TINY_SEED}; regenerate with python model/export.py --fixtures app/src/test/fixtures"),
    ]
    cases = golden_text([("source", "TOKENIZER_TEXTS in model/export.py")], token_rows(TOKENIZER_TEXTS))
    return {
        TINY_BIN: data,
        TINY_GOLDEN: golden_text(header, rows).encode("utf-8"),
        TOKENIZER_CASES: cases.encode("utf-8"),
    }


def tensors_from_model(model: cifg.CIFGLM) -> dict[str, np.ndarray]:
    p = model.parameters()
    return {
        "embedding": np.array(p["embedding"]["weight"], dtype=np.float32),
        "wx": np.array(p["wx"], dtype=np.float32),
        "wr": np.array(p["wr"], dtype=np.float32),
        "b": np.array(p["b"], dtype=np.float32),
        "proj": np.array(p["proj"], dtype=np.float32),
    }


def quantize_all(t: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for name in ("embedding", "wx", "wr", "proj"):
        q, s = cifg.quantize_int8(t[name], axis=0)
        out[f"{name}_q"] = q
        out[f"{name}_scale"] = s.reshape(-1)
    out["b"] = t["b"]
    return out


def dequantize_all(q: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for name in ("embedding", "wx", "wr", "proj"):
        out[name] = cifg.dequantize_int8(q[f"{name}_q"], q[f"{name}_scale"].reshape(-1, 1))
    out["b"] = q["b"]
    return out


def load_into_model(model: cifg.CIFGLM, t: dict[str, np.ndarray]):
    model.update({
        "embedding": {"weight": mx.array(t["embedding"])},
        "wx": mx.array(t["wx"]),
        "wr": mx.array(t["wr"]),
        "b": mx.array(t["b"]),
        "proj": mx.array(t["proj"]),
    })
    mx.eval(model.parameters())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=cifg.data_root())
    p.add_argument("--name", help="the run under <root>/runs to export")
    p.add_argument("--which", default="best", choices=["best", "latest"])
    p.add_argument("--check-lines", type=int, default=5000, help="dev lines for the f32 vs int8 perplexity check (0 = skip)")
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--bin-only", action="store_true", help="write only the cifg_* files, from export/weights_int8.npz")
    p.add_argument("--force", action="store_true", help="with --bin-only, replace cifg_* files that exist")
    p.add_argument("--out", metavar="DIR", help="with --bin-only, write the cifg_* files to DIR instead of the run's export/")
    p.add_argument("--golden-contexts", type=int, default=100, help="dev contexts in cifg_golden.tsv, after the fixed ones")
    p.add_argument("--token-lines", type=int, default=2000, help="dev lines in cifg_golden_tokens.tsv")
    p.add_argument("--fixtures", metavar="DIR", help="write the tiny test model, its golden file and the tokenizer cases to DIR")
    args = p.parse_args(argv)

    if args.fixtures:
        os.makedirs(args.fixtures, exist_ok=True)
        for name, data in fixture_files().items():
            _write(os.path.join(args.fixtures, name), data, force=True)
            print(f"[done] wrote {os.path.join(args.fixtures, name)} ({len(data):,} bytes)")
        return
    if not args.name:
        p.error("--name or --fixtures is required")
    if args.out and not args.bin_only:
        p.error("--out goes with --bin-only")

    run_dir = os.path.join(args.root, "runs", args.name)
    out = os.path.join(run_dir, "export")
    corpus_dir = os.path.join(args.root, "corpus")
    if args.bin_only:
        q = dict(np.load(os.path.join(out, "weights_int8.npz")))
        tokens = tokenize_module().Tokenizer.load(os.path.join(out, "vocab.txt")).tokens
        dest = args.out or out
        os.makedirs(dest, exist_ok=True)
        report = write_cifg_files(dest, q, tokens, corpus_dir, args.golden_contexts, args.token_lines, args.seed,
                                  args.force, f"runs/{args.name}/export/weights_int8.npz")
        print(json.dumps(report, indent=2))
        print(f"[done] wrote {dest}")
        return

    model, cfg = cifg.load_model(run_dir, args.which)
    os.makedirs(out, exist_ok=True)

    f32 = tensors_from_model(model)
    q = quantize_all(f32)
    f32_path = os.path.join(out, "weights_f32.npz")
    q_path = os.path.join(out, "weights_int8.npz")
    np.savez(f32_path, **f32)
    np.savez(q_path, **q)
    if os.path.exists(cfg["vocab_path"]):
        shutil.copyfile(cfg["vocab_path"], os.path.join(out, "vocab.txt"))

    report = {
        "run": args.name,
        "checkpoint": args.which,
        "arch": cfg.get("arch"),
        "vocab_size": cfg["vocab_size"],
        "embed_dim": cfg["embed_dim"],
        "hidden": cfg["hidden"],
        "gate_order": ["forget", "output", "update"],
        "params": int(sum(a.size for a in f32.values())),
        "tensors": {k: {"shape": list(v.shape), "f32_bytes": int(v.nbytes)} for k, v in f32.items()},
        "f32_raw_bytes": int(sum(v.nbytes for v in f32.values())),
        "int8_raw_bytes": int(sum(v.nbytes for v in q.values())),
        "files": {
            "weights_f32.npz": os.path.getsize(f32_path),
            "weights_int8.npz": os.path.getsize(q_path),
        },
    }
    for name in ("embedding", "wx", "wr", "proj"):
        deq = cifg.dequantize_int8(q[f"{name}_q"], q[f"{name}_scale"].reshape(-1, 1))
        err = np.abs(deq - f32[name])
        report["tensors"][name].update({
            "int8_bytes": int(q[f"{name}_q"].nbytes + q[f"{name}_scale"].nbytes),
            "quant_max_abs_err": float(err.max()),
            "quant_rel_rms_err": float(np.sqrt((err**2).mean()) / max(1e-12, np.sqrt((f32[name] ** 2).mean()))),
        })

    if args.check_lines != 0:
        vocab = cifg.Vocab.load(cfg["vocab_path"])
        tokens, offsets, _ = cifg.load_split(args.root, cfg["cache_fingerprint"], "dev", None)
        lens = np.diff(offsets)
        ids = np.nonzero(lens > 0)[0]
        if args.check_lines is not None and args.check_lines < len(ids):
            ids = np.sort(np.random.default_rng(args.seed).choice(ids, size=args.check_lines, replace=False))
        t0 = time.time()
        ppl_f32 = cifg.dev_perplexity(model, vocab, tokens, offsets, ids)
        load_into_model(model, dequantize_all(q))
        ppl_int8 = cifg.dev_perplexity(model, vocab, tokens, offsets, ids)
        report["check"] = {"dev_lines": int(len(ids)), "ppl_f32": ppl_f32["ppl"], "ppl_int8": ppl_int8["ppl"], "seconds": round(time.time() - t0, 1)}

    with open(os.path.join(out, "export.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    vocab_file = os.path.join(out, "vocab.txt")
    if os.path.exists(vocab_file):
        tokens = tokenize_module().Tokenizer.load(vocab_file).tokens
        # A full export rewrites every file in export/, so the cifg_* files are replaced along with the rest.
        cifg_report = write_cifg_files(out, q, tokens, corpus_dir, args.golden_contexts, args.token_lines, args.seed,
                                       True, f"runs/{args.name}/{args.which}.safetensors")
        print(json.dumps(cifg_report, indent=2))
    else:
        print(f"[skip] {cfg['vocab_path']} is gone, so no {BIN_NAME}")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
