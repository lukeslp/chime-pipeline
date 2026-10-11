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
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.expanduser("~/.local/share/localtype-model")
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

