"""CIFG-LSTM next-word model for the LocalType keyboard, in MLX.

The shape is the one the mobile-keyboard literature settled on: one coupled
input-forget gate (CIFG) LSTM layer of 670 units whose output is projected to
96 dimensions before it is fed back as the recurrent input, a 96-dimensional
input embedding of the word vocabulary, and the same embedding matrix used as
the output projection (tied weights). With a 16,384-entry vocabulary that is
about 2.0M parameters.

Everything here reads and writes under the data root (default
~/.local/share/localtype-model, override with LOCALTYPE_MODEL_DATA):

    <root>/vocab/vocab.txt       the shared vocabulary, one token per line
    <root>/vocab-tmp/vocab.txt   a temporary frequency vocabulary built here
                                 when the shared one does not exist yet
    <root>/cache/<fingerprint>/  tokenized splits (uint16 ids + offsets)
    <root>/runs/<name>/          checkpoints, logs, metrics, exports

Vocabulary contract: one token per line, the line number is the id. Special
tokens are found by name (see SPECIAL_NAMES). Tokenizer contract: a function
tokenize(line) -> list[int]; when model/tokenize.py exists next to this file
it is loaded by path and used, otherwise a word-and-emoji tokenizer that
mirrors the keyboard's own word regex is used.
"""

from __future__ import annotations

import bisect
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = str(Path.home() / ".local" / "share" / "localtype-model")
VOCAB_SIZE = 16384
EMBED_DIM = 96
HIDDEN = 670


def data_root() -> str:
    return os.environ.get("LOCALTYPE_MODEL_DATA", DEFAULT_ROOT)


# --------------------------------------------------------------------------
# Text rules, mirrored from the keyboard (AdaptiveNgramModel.WORD_REGEX and
# normalize) and from corpus.py
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’-])*")
# Single emoji code points and flags; variation selectors and skin tones are dropped.
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF⭐⭕‼⁉〰〽㊗㊙]"
)
TOKEN_RE = re.compile(f"{WORD_RE.pattern}|{EMOJI_RE.pattern}")
STRIP_RE = re.compile("[️︎\U0001F3FB-\U0001F3FF‍]")
FULL_WORD_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’-])*$")


def normalize(word: str) -> str:
    return word.replace("’", "'").lower()


SENTENCE_END = (".", "!", "?", "…")


def offerable_word(word: str) -> bool:
    """AdaptiveNgramModel.predict: `if (word.length < 2 && word != "a" && word != "i") return`.

    Kotlin's length counts UTF-16 units, so a one-letter word outside the Basic Multilingual Plane
    (a mathematical letter such as 𝑎) is two units long and offerable on the device.
    """
    return len(word.encode("utf-16-le")) >= 4 or word in ("a", "i")


def surface_tokens(line: str) -> list[str]:
    """Words (normalized) and emoji (as typed) in order."""
    out = []
    for m in TOKEN_RE.finditer(STRIP_RE.sub("", line)):
        tok = m.group(0)
        out.append(normalize(tok) if FULL_WORD_RE.match(tok) else tok)
    return out


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

SPECIAL_NAMES = {
    "pad": ("<pad>", "[PAD]", "<PAD>", "PAD"),
    "start": ("<s>", "<start>", "[START]", "<bos>", "<START>", "START", "<BOS>"),
    "end": ("</s>", "<end>", "[END]", "<eos>", "<END>", "END", "<EOS>"),
    "unk": ("<unk>", "[UNK]", "<UNK>", "UNK"),
}


@dataclass
class Vocab:
    tokens: list[str]
    path: str
    pad: int | None = None
    start: int = 0
    end: int | None = None
    unk: int | None = None
    index: dict[str, int] = field(default_factory=dict)
    word_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    sorted_words: list[str] = field(default_factory=list)
    sorted_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    is_word: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Words the keyboard can offer: AdaptiveNgramModel.predict drops every
    # one-letter word except "a" and "i", so the model's candidates do the same.
    offerable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    sorted_offerable: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Sentence-ending tokens (. ! ? …) when the vocabulary has them; the
    # harness scores on the current sentence only, cut after the last of these.
    boundary_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def load(cls, path: str) -> "Vocab":
        with open(path, encoding="utf-8") as f:
            tokens = [line.rstrip("\n") for line in f]
        while tokens and tokens[-1] == "":
            tokens.pop()
        v = cls(tokens=tokens, path=path)
        v.index = {t: i for i, t in enumerate(tokens)}
        found = {}
        for key, names in SPECIAL_NAMES.items():
            for name in names:
                if name in v.index:
                    found[key] = v.index[name]
                    break
        if "start" not in found:
            raise ValueError(f"{path}: no START token (looked for {SPECIAL_NAMES['start']})")
        v.start = found["start"]
        v.end = found.get("end")
        v.unk = found.get("unk")
        v.pad = found.get("pad")
        special = {i for i in (v.pad, v.start, v.end, v.unk) if i is not None}
        words = [(t, i) for i, t in enumerate(tokens) if i not in special and FULL_WORD_RE.match(t)]
        v.is_word = np.zeros(len(tokens), dtype=bool)
        v.word_ids = np.array([i for _, i in words], dtype=np.int64)
        v.is_word[v.word_ids] = True
        words.sort()
        v.sorted_words = [w for w, _ in words]
        v.sorted_ids = np.array([i for _, i in words], dtype=np.int64)
        v.offerable = v.is_word & np.array([offerable_word(t) for t in tokens], dtype=bool)
        v.sorted_offerable = v.offerable[v.sorted_ids] if len(v.sorted_ids) else np.zeros(0, dtype=bool)
        v.boundary_ids = np.array([v.index[t] for t in SENTENCE_END if t in v.index], dtype=np.int64)
        return v

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        """Input padding id; masked out of the loss, so START works when there is no pad."""
        return self.pad if self.pad is not None else self.start

    def prefix_range(self, prefix: str) -> tuple[int, int]:
        lo = bisect.bisect_left(self.sorted_words, prefix)
        hi = bisect.bisect_left(self.sorted_words, prefix + "￿")
        return lo, hi

    def fingerprint(self, tokenizer_source: str) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            h.update(f.read())
        h.update(b"\0" + tokenizer_source.encode())
        return h.hexdigest()[:12]


def shared_vocab_path(root: str) -> str:
    return os.path.join(root, "vocab", "vocab.txt")


def corpus_dir(root: str, corpus: str | None = None) -> str:
    """The corpus directory: `corpus` when given, else <root>/corpus (the v2 build)."""
    return os.path.abspath(corpus) if corpus else os.path.join(root, "corpus")


def cache_fingerprint(vocab: "Vocab", tokenizer_source: str, root: str, corpus: str | None = None) -> str:
    """The cache key: vocabulary and tokenizer, plus the corpus when it isn't <root>/corpus.

    The default corpus keeps the key it always had, so existing caches and
    run configs stay valid; any other corpus gets its own cache directory
    (its name and a hash of its path), so two corpora never share token files.
    """
    fp = vocab.fingerprint(tokenizer_source)
    path = corpus_dir(root, corpus)
    if os.path.realpath(path) == os.path.realpath(corpus_dir(root)):
        return fp
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(os.path.normpath(path)))
    return f"{fp}-{tag}-{hashlib.sha256(os.path.realpath(path).encode()).hexdigest()[:6]}"


# The human-written sources of the v2 corpus, used when a corpus has no manifest.
DEFAULT_HUMAN_SOURCES = ("nus", "taskmaster")


def human_sources(corpus: str) -> tuple[str, ...]:
    """The sources a corpus's manifest.json marks human-written, in build order."""
    path = os.path.join(corpus, "manifest.json")
    if not os.path.exists(path):
        return DEFAULT_HUMAN_SOURCES
    with open(path, encoding="utf-8") as f:
        sources = json.load(f).get("sources", {})
    return tuple(k for k, v in sources.items() if v.get("human_written"))


def temp_vocab_path(root: str) -> str:
    return os.path.join(root, "vocab-tmp", "vocab.txt")


def _count_chunk(lines: list[str]) -> Counter:
    c: Counter = Counter()
    for line in lines:
        c.update(surface_tokens(line))
    return c


def build_temp_vocab(root: str, train_path: str, size: int = VOCAB_SIZE, max_lines: int | None = None, workers: int = 8) -> str:
    """Frequency vocabulary over the training split: specials, then words and emoji by count."""
    from multiprocessing import get_context

    out = temp_vocab_path(root)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    counts: Counter = Counter()
    with get_context("spawn").Pool(workers) as pool:
        for part in pool.imap_unordered(_count_chunk, _line_chunks(train_path, 50_000, max_lines)):
            counts.update(part)
    specials = ["<pad>", "<s>", "</s>", "<unk>"]
    ranked = [t for t, _ in counts.most_common(size - len(specials))]
    tokens = specials + ranked
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens) + "\n")
    with open(os.path.join(os.path.dirname(out), "counts.tsv"), "w", encoding="utf-8") as f:
        for t, n in counts.most_common():
            f.write(f"{t}\t{n}\n")
    total = sum(counts.values())
    covered = sum(counts[t] for t in ranked)
    meta = {
        "source": train_path,
        "max_lines": max_lines,
        "size": len(tokens),
        "unique_tokens": len(counts),
        "token_occurrences": total,
        "coverage": covered / max(1, total),
        "emoji_entries": sum(1 for t in ranked if not FULL_WORD_RE.match(t)),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(os.path.dirname(out), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out


def _line_chunks(path: str, chunk: int, max_lines: int | None) -> Iterable[list[str]]:
    buf: list[str] = []
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf.append(line.rstrip("\n"))
            n += 1
            if len(buf) >= chunk:
                yield buf
                buf = []
            if max_lines is not None and n >= max_lines:
                break
    if buf:
        yield buf


# --------------------------------------------------------------------------
# Tokenizer: the sibling module by path when it exists, else the fallback
# --------------------------------------------------------------------------

SIBLING_TOKENIZER = os.path.join(HERE, "tokenize.py")


class FallbackTokenizer:
    """Words and emoji mapped through the vocabulary; OOV -> unk."""

    name = "fallback"

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def __call__(self, line: str) -> list[int]:
        return [i for _, i in self.with_surface(line)]

    def with_surface(self, line: str) -> list[tuple[str, int]]:
        idx = self.vocab.index
        unk = self.vocab.unk
        out = []
        for tok in surface_tokens(line):
            i = idx.get(tok, unk)
            if i is not None:
                out.append((tok, i))
        return out


class SharedTokenizer:
    """model/tokenize.py's Tokenizer: words, punctuation, emoji, <num>, <sym>; OOV words -> unk."""

    name = "shared"

    def __init__(self, tk):
        self.tk = tk

    def __call__(self, line: str) -> list[int]:
        return self.tk.encode(line, bos=False, eos=False)

    def with_surface(self, line: str) -> list[tuple[str, int]]:
        tk = self.tk
        return [(tok, tk.id_of(tok)) for tok in tk.tokens_of(line)]


def load_tokenizer(vocab: Vocab) -> tuple[Callable[[str], list[int]], str]:
    """Returns (tokenize, source_description).

    model/tokenize.py is loaded by path under its own name so it never shadows
    the standard library. Its Tokenizer class is used when the vocabulary file
    has its layout; failing that its plain tokenize(line) -> list[int] is used.
    """
    if os.path.exists(SIBLING_TOKENIZER):
        spec = importlib.util.spec_from_file_location("localtype_tokenize", SIBLING_TOKENIZER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        with open(SIBLING_TOKENIZER, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:12]
        cls = getattr(mod, "Tokenizer", None)
        if cls is not None and hasattr(cls, "load"):
            try:
                return SharedTokenizer(cls.load(vocab.path)), f"model/tokenize.py@{digest}"
            except ValueError:
                pass  # a vocabulary without that layout (the temporary one)
        fn = getattr(mod, "tokenize", None)
        if callable(fn):
            probe = fn("hey")
            if probe and isinstance(probe[0], int):
                return fn, f"model/tokenize.py@{digest}"
    return FallbackTokenizer(vocab), "fallback"


# --------------------------------------------------------------------------
# Tokenized cache: <root>/cache/<fingerprint>/<split>.{tokens,offsets,oovlen}.npy
# --------------------------------------------------------------------------

_worker_tok = None
_worker_vocab = None


def _init_worker(vocab_path: str):
    global _worker_tok, _worker_vocab
    _worker_vocab = Vocab.load(vocab_path)
    _worker_tok, _ = load_tokenizer(_worker_vocab)


def _encode_chunk(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tok = _worker_tok
    v = _worker_vocab
    ids: list[int] = []
    oov: list[int] = []
    lens: list[int] = []
    strip = {v.start, v.end, v.pad}
    for line in lines:
        if hasattr(tok, "with_surface"):
            pairs = tok.with_surface(line)
            seq = [i for _, i in pairs]
            o = [
                0 if i != v.unk else (min(len(s), 200) if FULL_WORD_RE.match(s) else 255)
                for s, i in pairs
            ]
        else:
            seq = list(tok(line))
            # A tokenizer that adds START/END itself: strip them, the batcher adds them back.
            while seq and seq[0] in strip:
                seq.pop(0)
            while seq and seq[-1] in strip:
                seq.pop()
            o = [0] * len(seq)
        ids.extend(seq)
        oov.extend(o)
        lens.append(len(seq))
    return np.array(ids, dtype=np.uint16), np.array(lens, dtype=np.int64), np.array(oov, dtype=np.uint8)


def cache_dir(root: str, fingerprint: str) -> str:
    return os.path.join(root, "cache", fingerprint)


def _tag(split: str, max_lines: int | None) -> str:
    return split if max_lines is None else f"{split}.first{max_lines}"


def _replace_file(path: str, write: Callable, mode: str = "wb") -> None:
    """Write path through a temp file beside it, then rename it into place.

    A running train.py memory-maps the cached arrays (load_split), so a cache
    file is never rewritten on its own inode: the run keeps the old one until
    it exits instead of reading new data, or dying, mid-run.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, mode) as f:
            write(f)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def prepare_split(root: str, vocab: Vocab, fingerprint: str, split: str, max_lines: int | None, workers: int = 8,
                  corpus: str | None = None) -> dict:
    """Tokenize <corpus>/<split>.txt into the cache, or return the cached meta.

    The fingerprint covers the vocabulary and tokenizer (and the corpus
    directory when it isn't the default; see cache_fingerprint), so the
    corpus file's size and mtime are stored in the meta too: a rebuilt corpus
    re-tokenizes instead of reading the old corpus's tokens. The meta is
    written last, so an interrupted re-tokenize keeps the old stamp and
    runs again on the next call.
    """
    src = os.path.join(corpus_dir(root, corpus), f"{split}.txt")
    st = os.stat(src)
    stamp = {"source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns}
    out = cache_dir(root, fingerprint)
    os.makedirs(out, exist_ok=True)
    tag = _tag(split, max_lines)
    meta_path = os.path.join(out, f"{tag}.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            cached = json.load(f)
        if all(cached.get(k) == v for k, v in stamp.items()):
            return cached
        print(f"[prepare] {tag}: {src} changed since it was tokenized; re-tokenizing", flush=True)
    from multiprocessing import get_context

    t0 = time.time()
    parts_ids, parts_lens, parts_oov = [], [], []
    with get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(vocab.path,)) as pool:
        for ids, lens, oov in pool.imap(_encode_chunk, _line_chunks(src, 20_000, max_lines)):
            parts_ids.append(ids)
            parts_lens.append(lens)
            parts_oov.append(oov)
    tokens = np.concatenate(parts_ids) if parts_ids else np.zeros(0, dtype=np.uint16)
    lens = np.concatenate(parts_lens) if parts_lens else np.zeros(0, dtype=np.int64)
    oov = np.concatenate(parts_oov) if parts_oov else np.zeros(0, dtype=np.uint8)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    for name, arr in (("tokens", tokens), ("offsets", offsets), ("oovlen", oov)):
        # A file handle, so np.save doesn't append .npy to the temp name.
        _replace_file(os.path.join(out, f"{tag}.{name}.npy"), lambda f, a=arr: np.save(f, a))
    unk = vocab.unk
    meta = {
        "split": split,
        "tag": tag,
        "source": src,
        "lines": int(len(lens)),
        "tokens": int(len(tokens)),
        "unk_tokens": int((tokens == unk).sum()) if unk is not None else 0,
        "empty_lines": int((lens == 0).sum()),
        "seconds": round(time.time() - t0, 1),
        **stamp,
    }
    _replace_file(meta_path, lambda f: json.dump(meta, f, indent=2), mode="w")
    return meta


def load_split(root: str, fingerprint: str, split: str, max_lines: int | None):
    out = cache_dir(root, fingerprint)
    tag = _tag(split, max_lines)
    tokens = np.load(os.path.join(out, f"{tag}.tokens.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(out, f"{tag}.offsets.npy"))
    oov = np.load(os.path.join(out, f"{tag}.oovlen.npy"), mmap_mode="r")
    return tokens, offsets, oov


# --------------------------------------------------------------------------
# Batches: [START] + ids -> ids + [END], padded, with a loss mask
# --------------------------------------------------------------------------

def make_batch(seqs: list[np.ndarray], vocab: Vocab, max_len: int, round_to: int = 8):
    """Returns x, y (int32, B x L) and mask (float32, B x L)."""
    has_end = vocab.end is not None
    lens = [min(len(s), max_len - 1) + (1 if has_end else 0) for s in seqs]
    L = max(1, max(lens))
    L = ((L + round_to - 1) // round_to) * round_to
    B = len(seqs)
    x = np.full((B, L), vocab.pad_id, dtype=np.int32)
    y = np.full((B, L), vocab.pad_id, dtype=np.int32)
    mask = np.zeros((B, L), dtype=np.float32)
    for b, s in enumerate(seqs):
        s = np.asarray(s[: max_len - 1], dtype=np.int32)
        n = len(s)
        x[b, 0] = vocab.start
        x[b, 1 : n + 1] = s
        y[b, :n] = s
        if has_end:
            y[b, n] = vocab.end
            mask[b, : n + 1] = 1.0
        else:
            mask[b, :n] = 1.0
    return x, y, mask


def load_sources(root: str, split: str, max_lines: int | None = None, corpus: str | None = None) -> tuple[np.ndarray, list[str]]:
    """Per-line source codes for a split from <corpus>/<split>.sources.txt, plus the code names."""
    path = os.path.join(corpus_dir(root, corpus), f"{split}.sources.txt")
    names: list[str] = []
    codes: dict[str, int] = {}
    out: list[int] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f):
            if max_lines is not None and n >= max_lines:
                break
            key = line.strip()
            c = codes.get(key)
            if c is None:
                c = codes[key] = len(names)
                names.append(key)
            out.append(c)
    return np.array(out, dtype=np.int8), names


def oversample_repeats(source_codes: np.ndarray, names: list[str], factor: int, sources: Iterable[str]) -> np.ndarray:
    """How many times each line enters an epoch: factor for the named sources, 1 otherwise."""
    return oversample_repeats_by_source(source_codes, names, {s: factor for s in sources})


def oversample_repeats_by_source(source_codes: np.ndarray, names: list[str], factors: dict[str, int]) -> np.ndarray:
    """How many times each line enters an epoch: its source's factor, 1 for a source not named.

    A factor of 1 or less means no oversampling, as `--human-oversample 1` always has.
    """
    rep = np.ones(len(source_codes), dtype=np.int64)
    for s, k in factors.items():
        if s in names and k > 1:
            rep[source_codes == names.index(s)] = k
    return rep


class Batcher:
    """Length-bucketed random batches over a tokenized split.

    `repeats` (one int per line) enters a line that many times into every
    epoch's permutation; it is how the small human-written sources are
    oversampled against the synthetic one.
    """

    def __init__(self, tokens, offsets, batch_size: int, max_len: int, seed: int, pool_batches: int = 64, repeats: np.ndarray | None = None):
        self.tokens = tokens
        self.offsets = offsets
        self.batch_size = batch_size
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)
        self.pool = pool_batches
        lens = np.diff(offsets)
        self.indices = np.nonzero(lens > 0)[0]
        if repeats is not None:
            self.indices = np.repeat(self.indices, repeats[self.indices])
        self.epoch = 0

    def __iter__(self):
        while True:
            order = self.rng.permutation(self.indices)
            pool = self.batch_size * self.pool
            for start in range(0, len(order) - self.batch_size + 1, pool):
                chunk = order[start : start + pool]
                lens = self.offsets[chunk + 1] - self.offsets[chunk]
                chunk = chunk[np.argsort(lens, kind="stable")]
                batches = [chunk[i : i + self.batch_size] for i in range(0, len(chunk) - self.batch_size + 1, self.batch_size)]
                self.rng.shuffle(batches)
                for idx in batches:
                    yield [self.tokens[self.offsets[i] : self.offsets[i + 1]] for i in idx]
            self.epoch += 1


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402


class CIFGLM(nn.Module):
    """One CIFG-LSTM layer with a recurrent projection and tied embeddings.

    gates = x_t @ wx + r_{t-1} @ wr + b, split into f, o, u
    f = sigmoid(f)  (input gate is 1 - f)   o = sigmoid(o)   u = tanh(u)
    c_t = f * c_{t-1} + (1 - f) * u
    h_t = o * tanh(c_t)                      (hidden units)
    r_t = h_t @ proj                         (embed_dim, fed back and used for logits)
    logits_t = r_t @ embedding.T
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM, hidden: int = HIDDEN, forget_bias: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        kx = 1.0 / math.sqrt(embed_dim)
        kh = 1.0 / math.sqrt(hidden)
        self.wx = mx.random.uniform(-kx, kx, (embed_dim, 3 * hidden))
        self.wr = mx.random.uniform(-kx, kx, (embed_dim, 3 * hidden))
        b = mx.zeros((3 * hidden,))
        b[:hidden] = forget_bias
        self.b = b
        self.proj = mx.random.uniform(-kh, kh, (hidden, embed_dim))

    def initial_state(self, batch: int):
        return mx.zeros((batch, self.hidden)), mx.zeros((batch, self.embed_dim))

    def __call__(self, tokens: mx.array, state=None):
        x = self.embedding(tokens)
        xg = x @ self.wx + self.b
        B, L, _ = xg.shape
        if state is None:
            c, r = self.initial_state(B)
        else:
            c, r = state
        H = self.hidden
        outs = []
        for t in range(L):
            g = xg[:, t] + r @ self.wr
            f = mx.sigmoid(g[:, :H])
            o = mx.sigmoid(g[:, H : 2 * H])
            u = mx.tanh(g[:, 2 * H :])
            c = f * c + (1.0 - f) * u
            h = o * mx.tanh(c)
            r = h @ self.proj
            outs.append(r)
        R = mx.stack(outs, axis=1)
        logits = self.embedding.as_linear(R)
        return logits, (c, r)

    def config(self) -> dict:
        return {"vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "hidden": self.hidden, "arch": "cifg-lstmp-tied"}


def param_count(model: nn.Module) -> int:
    from mlx.utils import tree_flatten

    return int(sum(v.size for _, v in tree_flatten(model.parameters())))


def expected_param_count(vocab_size: int = VOCAB_SIZE, embed_dim: int = EMBED_DIM, hidden: int = HIDDEN) -> int:
    return vocab_size * embed_dim + 2 * embed_dim * 3 * hidden + 3 * hidden + hidden * embed_dim


def loss_fn(model: CIFGLM, x: mx.array, y: mx.array, mask: mx.array):
    logits, _ = model(x)
    ce = nn.losses.cross_entropy(logits, y, reduction="none")
    return (ce * mask).sum() / mask.sum()


def load_model(run_dir: str, which: str = "best") -> tuple[CIFGLM, dict]:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    model = CIFGLM(cfg["vocab_size"], cfg["embed_dim"], cfg["hidden"])
    path = os.path.join(run_dir, f"{which}.safetensors")
    if not os.path.exists(path):
        path = os.path.join(run_dir, "latest.safetensors")
    model.load_weights(path)
    mx.eval(model.parameters())
    return model, cfg


# --------------------------------------------------------------------------
# Evaluation: perplexity and keyboard-shaped metrics
# --------------------------------------------------------------------------

def _eval_batches(tokens, offsets, line_ids: np.ndarray, batch_size: int):
    lens = offsets[line_ids + 1] - offsets[line_ids]
    order = line_ids[np.argsort(lens, kind="stable")]
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        yield idx, [tokens[offsets[j] : offsets[j + 1]] for j in idx]


def dev_perplexity(model: CIFGLM, vocab: Vocab, tokens, offsets, line_ids: np.ndarray, batch_size: int = 128, max_len: int = 64) -> dict:
    total = 0.0
    count = 0.0
    for _, seqs in _eval_batches(tokens, offsets, line_ids, batch_size):
        x, y, mask = make_batch(seqs, vocab, max_len)
        xm, ym, mm = mx.array(x), mx.array(y), mx.array(mask)
        logits, _ = model(xm)
        ce = nn.losses.cross_entropy(logits, ym, reduction="none")
        s = (ce * mm).sum()
        n = mm.sum()
        mx.eval(s, n)
        total += s.item()
        count += n.item()
    if count == 0:
        # Nothing scored: NaN, never the perplexity 1.0 that 0/0 would otherwise report.
        return {"nll": float("nan"), "ppl": float("nan"), "targets": 0}
    nll = total / count
    return {"nll": nll, "ppl": math.exp(nll), "targets": int(count)}


def scope_items(tokens, offsets, line_ids: np.ndarray, boundary_ids: np.ndarray, scope: str) -> list[tuple[int, int]]:
    """Token slices to score: one per line, or one per sentence of each line.

    "line" scores every word with the whole line so far as context. "sentence"
    is what the replay harness does (docs/prediction-log-spec.md, Context
    Scope): the context is the current sentence only, cut after the last
    . ! ? or newline, so the token sequence is split at those tokens and every
    chunk restarts from <s>. The boundary token itself belongs to no chunk.
    """
    if scope == "line":
        return [(int(offsets[l]), int(offsets[l + 1])) for l in line_ids]
    if scope != "sentence":
        raise ValueError(f"scope must be line or sentence, not {scope!r}")
    items: list[tuple[int, int]] = []
    for l in line_ids:
        s, e = int(offsets[l]), int(offsets[l + 1])
        if e <= s:
            continue
        marks = np.nonzero(np.isin(np.asarray(tokens[s:e]), boundary_ids))[0] if len(boundary_ids) else []
        cur = s
        for m in marks:
            end = s + int(m)
            if end > cur:
                items.append((cur, end))
            cur = end + 1
        if e > cur:
            items.append((cur, e))
    return items


def _item_batches(items: list[tuple[int, int]], batch_size: int):
    lens = np.array([e - s for s, e in items], dtype=np.int64)
    order = np.argsort(lens, kind="stable")
    for i in range(0, len(order), batch_size):
        yield [items[j] for j in order[i : i + batch_size]]


def keyboard_metrics(
    model: CIFGLM,
    vocab: Vocab,
    tokens,
    offsets,
    oovlen,
    line_ids: np.ndarray,
    batch_size: int = 32,
    max_len: int = 128,
    max_prefix: int = 16,
    scope: str = "line",
) -> dict:
    """Next-word hit@1/3/6 and MRR at prefix 0, completion hit@3 by prefix
    length with candidates restricted to vocabulary words starting with that
    prefix, and the potential-savings rate from the full ladder.

    Candidates are the words the keyboard could offer: vocabulary words minus
    every one-letter word except a and i (AdaptiveNgramModel.predict). Targets
    are the word tokens of each item, in order, with the true left context of
    that item; `scope` decides whether an item is the whole line or the current
    sentence. END, punctuation, number, symbol and emoji tokens are skipped and
    counted. An OOV word (unk with a known length) and an unofferable one-letter
    word are misses at every prefix length and count in every denominator,
    which is how the replay harness will treat a word the model cannot
    produce. OOV targets of unknown length (from a tokenizer that gives ids
    only) are misses at prefix 0 and are left out of the ladder. Items longer
    than max_len - 1 tokens are cut there; the targets past the cut are
    counted in `cut` and not scored.
    """
    V = len(vocab)
    word_bias = np.full(V, -1e30, dtype=np.float32)
    word_bias[vocab.offerable] = 0.0
    word_bias_mx = mx.array(word_bias)
    sorted_ids = vocab.sorted_ids
    sorted_off = vocab.sorted_offerable
    tokens_str = vocab.tokens
    unk = vocab.unk

    n_targets = n_inv = n_oov_known = n_oov_unknown = n_emoji = n_unofferable = 0
    n_items_cut = n_targets_cut = 0
    hit1 = hit3 = hit6 = 0
    rr = 0.0
    comp_hits = np.zeros(max_prefix + 1, dtype=np.int64)
    comp_cnt = np.zeros(max_prefix + 1, dtype=np.int64)
    comp_cnt_inv = np.zeros(max_prefix + 1, dtype=np.int64)
    saved = 0
    baseline = 0
    baseline_inv = 0
    per_instance_comp = 0.0  # mean hit@3 over the instance's ladder positions 1..len-1
    per_instance_n = 0

    items = scope_items(tokens, offsets, line_ids, vocab.boundary_ids, scope)
    for batch in _item_batches(items, batch_size):
        seqs = [tokens[s:e] for s, e in batch]
        x, y, mask = make_batch(seqs, vocab, max_len)
        logits, _ = model(mx.array(x))
        masked = logits + word_bias_mx
        top6 = mx.argpartition(-masked, kth=5, axis=-1)[..., :6]
        mx.eval(logits, top6)
        logits_np = np.array(logits, copy=False)
        top6_np = np.array(top6)
        for b, (s, e) in enumerate(batch):
            seq = seqs[b]
            oov_item = oovlen[s:e]
            n = min(len(seq), max_len - 1)
            if len(seq) > n:
                n_items_cut += 1
                for t in range(n, len(seq)):
                    tid = int(seq[t])
                    if vocab.is_word[tid] or (tid == unk and int(oov_item[t]) != 255):
                        n_targets_cut += 1
            for t in range(n):
                tid = int(seq[t])
                if vocab.is_word[tid]:
                    word = tokens_str[tid]
                    known = True
                    if not vocab.offerable[tid]:
                        # A one-letter word other than a/i: the device never offers it.
                        n_targets += 1
                        n_unofferable += 1
                        baseline += len(word) + 1
                        continue
                elif tid == unk:
                    ol = int(oov_item[t])
                    if ol == 255:
                        n_emoji += 1
                        continue
                    if ol == 0:
                        n_oov_unknown += 1
                        n_targets += 1
                        continue  # miss at p=0, no ladder
                    word = None
                    known = False
                    wlen = ol
                else:
                    n_emoji += 1
                    continue
                n_targets += 1
                row = logits_np[b, t]
                if known:
                    n_inv += 1
                    wlen = len(word)
                    cand = top6_np[b, t]
                    cand = cand[np.argsort(-row[cand], kind="stable")]
                    pos = np.nonzero(cand == tid)[0]
                    p_star = None
                    if len(pos):
                        rank = int(pos[0]) + 1
                        rr += 1.0 / rank
                        hit6 += 1
                        if rank <= 3:
                            hit3 += 1
                            p_star = 0
                        if rank == 1:
                            hit1 += 1
                    ladder_hits = 0
                    ladder_n = 0
                    for p in range(1, min(wlen, max_prefix + 1)):
                        lo, hi = vocab.prefix_range(word[:p])
                        ids = sorted_ids[lo:hi][sorted_off[lo:hi]]
                        if len(ids) <= 3:
                            hit = True
                        else:
                            vals = row[ids]
                            top = ids[np.argpartition(-vals, 2)[:3]]
                            hit = bool((top == tid).any())
                        comp_cnt[p] += 1
                        comp_cnt_inv[p] += 1
                        ladder_n += 1
                        if hit:
                            comp_hits[p] += 1
                            ladder_hits += 1
                            if p_star is None:
                                p_star = p
                    if p_star is None and wlen > max_prefix + 1:
                        p_star = max_prefix + 1  # past the searched ladder, credit the rest
                    if ladder_n:
                        per_instance_comp += ladder_hits / ladder_n
                        per_instance_n += 1
                    baseline += wlen + 1
                    baseline_inv += wlen + 1
                    if p_star is not None:
                        saved += max(0, wlen - p_star)
                else:
                    n_oov_known += 1
                    for p in range(1, min(wlen, max_prefix + 1)):
                        comp_cnt[p] += 1
                    baseline += wlen + 1
                    if wlen >= 2:
                        per_instance_n += 1

    def rate(a, b):
        return a / b if b else 0.0

    comp = {str(p): {"hit3": rate(comp_hits[p], comp_cnt[p]), "n": int(comp_cnt[p])} for p in range(1, 5)}
    pooled5 = comp_hits[5:].sum(), comp_cnt[5:].sum()
    comp["5+"] = {"hit3": rate(*pooled5), "n": int(pooled5[1])}
    comp["pooled"] = {"hit3": rate(comp_hits[1:].sum(), comp_cnt[1:].sum()), "n": int(comp_cnt[1:].sum())}
    comp_inv = {str(p): {"hit3": rate(comp_hits[p], comp_cnt_inv[p]), "n": int(comp_cnt_inv[p])} for p in range(1, 5)}
    return {
        "scope": scope,
        "lines": int(len(line_ids)),
        "items": len(items),
        "targets": n_targets,
        "in_vocab_targets": n_inv,
        "oov_targets": n_oov_known + n_oov_unknown,
        "oov_targets_unknown_length": n_oov_unknown,
        "oov_rate": rate(n_oov_known + n_oov_unknown, n_targets),
        "unofferable_one_letter_targets": n_unofferable,
        "non_word_targets_skipped": n_emoji,
        "cut": {"max_len": max_len, "items_cut": n_items_cut, "targets_not_scored": n_targets_cut},
        "next": {"hit1": rate(hit1, n_targets), "hit3": rate(hit3, n_targets), "hit6": rate(hit6, n_targets), "mrr6": rate(rr, n_targets), "n": n_targets},
        "next_in_vocab_only": {"hit1": rate(hit1, n_inv), "hit3": rate(hit3, n_inv), "n": n_inv},
        "completion": comp,
        "completion_in_vocab_only": comp_inv,
        "per_instance_completion_hit3": rate(per_instance_comp, per_instance_n),
        "potential_savings_rate": rate(saved, baseline),
        "potential_savings_rate_in_vocab_only": rate(saved, baseline_inv),
        "max_prefix_searched": max_prefix,
    }


def format_metrics(m: dict) -> str:
    lines = [
        f"scope {m['scope']}: {m['lines']} lines, {m['items']} scored items; targets {m['targets']} (in-vocab {m['in_vocab_targets']}, OOV {m['oov_targets']} = {100 * m['oov_rate']:.2f}%, unknown-length OOV {m['oov_targets_unknown_length']}, one-letter unofferable {m['unofferable_one_letter_targets']}, non-word targets skipped {m['non_word_targets_skipped']}); cut at {m['cut']['max_len']} tokens: {m['cut']['items_cut']} items, {m['cut']['targets_not_scored']} targets not scored",
        f"next word (prefix 0): hit@1 {100 * m['next']['hit1']:.2f}%  hit@3 {100 * m['next']['hit3']:.2f}%  hit@6 {100 * m['next']['hit6']:.2f}%  mrr@6 {m['next']['mrr6']:.4f}",
        "completion hit@3 by prefix length: " + "  ".join(f"p={p} {100 * m['completion'][p]['hit3']:.2f}% (n={m['completion'][p]['n']})" for p in ("1", "2", "3", "4", "5+")),
        f"pooled completion hit@3 {100 * m['completion']['pooled']['hit3']:.2f}%  per-instance mean {100 * m['per_instance_completion_hit3']:.2f}%",
        f"potential savings rate {100 * m['potential_savings_rate']:.2f}% of characters (in-vocab only {100 * m['potential_savings_rate_in_vocab_only']:.2f}%)",
        f"in-vocab only: next hit@1 {100 * m['next_in_vocab_only']['hit1']:.2f}%  hit@3 {100 * m['next_in_vocab_only']['hit3']:.2f}%",
    ]
    return "\n".join(lines)


METRIC_ROWS: list[tuple[str, Callable[[dict], str]]] = [
    ("lines", lambda m: f"{m['lines']:,}"),
    ("word targets", lambda m: f"{m['targets']:,}"),
    ("OOV rate", lambda m: f"{100 * m['oov_rate']:.2f}%"),
    ("next hit@1", lambda m: f"{100 * m['next']['hit1']:.2f}%"),
    ("next hit@3", lambda m: f"{100 * m['next']['hit3']:.2f}%"),
    ("next hit@6", lambda m: f"{100 * m['next']['hit6']:.2f}%"),
    ("next MRR@6", lambda m: f"{m['next']['mrr6']:.4f}"),
    ("completion hit@3 p=1", lambda m: f"{100 * m['completion']['1']['hit3']:.2f}%"),
    ("completion hit@3 p=2", lambda m: f"{100 * m['completion']['2']['hit3']:.2f}%"),
    ("completion hit@3 p=3", lambda m: f"{100 * m['completion']['3']['hit3']:.2f}%"),
    ("completion hit@3 p=4", lambda m: f"{100 * m['completion']['4']['hit3']:.2f}%"),
    ("completion hit@3 p=5+", lambda m: f"{100 * m['completion']['5+']['hit3']:.2f}%"),
    ("pooled completion hit@3", lambda m: f"{100 * m['completion']['pooled']['hit3']:.2f}%"),
    ("per-instance completion hit@3", lambda m: f"{100 * m['per_instance_completion_hit3']:.2f}%"),
    ("potential savings rate", lambda m: f"{100 * m['potential_savings_rate']:.2f}%"),
]


def format_table(columns: dict[str, dict], markdown: bool = False) -> str:
    """One row per metric, one column per named metrics dict (pooled, nus, ...)."""
    names = list(columns)
    rows = [[label] + [fn(columns[n]) for n in names] for label, fn in METRIC_ROWS]
    if markdown:
        out = ["| | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)
    widths = [max(len(r[i]) for r in rows + [[""] + names]) for i in range(len(names) + 1)]

    def fmt(r):
        return "  ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r))

    return "\n".join([fmt([""] + names)] + [fmt(r) for r in rows])


# --------------------------------------------------------------------------
# Quantization helpers used by export.py and its test
# --------------------------------------------------------------------------

def quantize_int8(w: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-slice int8: q = round(w / scale), scale = max|w| / 127 along every other axis."""
    w = np.asarray(w, dtype=np.float32)
    reduce_axes = tuple(i for i in range(w.ndim) if i != axis)
    amax = np.abs(w).max(axis=reduce_axes, keepdims=True) if reduce_axes else np.abs(w).max(keepdims=True)
    scale = np.maximum(amax, 1e-12) / 127.0
    q = np.clip(np.rint(w / scale), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


def dequantize_int8(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale
