import itertools

import os

import sys

from pathlib import Path

import pytest

from model import tokenize as tk

from model.tokenize import (
    BOS, EOS, NUM, PAD, SYM, UNK, Tokenizer, active_prefix, detokenize, is_sentence_start,
    is_word, last_completed_word, normalize, normalize_word, tokenize,
)

DATA_ROOT = Path(os.environ.get("LOCALTYPE_MODEL_DATA", str(Path.home() / ".local" / "share" / "localtype-model")))

ASSETS = Path(__file__).resolve().parents[2] / "app" / "src" / "main" / "assets"

def test_normalize_matches_device():
    assert normalize("Don’t") == "don't"
    assert normalize("ÉCOLE") == "école"
    assert normalize_word("  Hello, world") == "hello"
    assert normalize_word("123") is None
    assert normalize_word("'quoted") == "quoted"

def test_active_prefix_matches_device():
    assert active_prefix("Please writ") == "writ"
    assert active_prefix("Please ") == ""
    assert active_prefix("it’s") == "it’s"  # raw, not normalized
    assert active_prefix("e-mai") == "e-mai"
    assert active_prefix("12ab") == "ab"
    assert active_prefix("'hello") == "hello"
    assert active_prefix("") == ""
    assert active_prefix("so 12") == ""

def test_last_completed_word_matches_device():
    assert last_completed_word("Please writ") == "please"
    assert last_completed_word("Please ") == "please"
    assert last_completed_word("All done. Th") is None
    assert last_completed_word("All done! ") is None
    assert last_completed_word("so 12 ") is None
    assert last_completed_word("thanks\nsee") is None
    assert last_completed_word("Don’t you") == "don't"
    assert last_completed_word("") is None

def test_is_sentence_start_matches_device():
    assert is_sentence_start("")
    assert is_sentence_start("Hel")
    assert is_sentence_start("thanks\n")
    assert is_sentence_start("thanks\n  wh")
    assert not is_sentence_start("thanks wh")
    assert is_sentence_start("Done. ")
    assert is_sentence_start("Really?! ")
    assert not is_sentence_start("Really, ")

def test_number_letters_are_not_letters():
    # Ⅻ (Nl) and ² (No) are letters to Python's \w but not to Java's \p{L}.
    assert normalize_word("Ⅻth") == "th"
    assert active_prefix("Ⅻth") == "th"
    assert tokenize("Ⅻth x²") == [SYM, "th", "x", SYM]
    assert last_completed_word("x² ") is None

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hello, world!", ["hello", ",", "world", "!"]),
        ("Don’t you DARE", ["don't", "you", "dare"]),
        ("e-mail rock'n'roll goin'", ["e-mail", "rock'n'roll", "goin'"]),
        ("wait... what?!", ["wait", "…", "what", "?", "!"]),
        ("no!!! way??", ["no", "!", "way", "?"]),
        ("hmm…. ok", ["hmm", "…", "ok"]),
        ("2nd at 12:30 on 9/11 for $1,000", [NUM, "nd", "at", NUM, "on", NUM, "for", "$", NUM]),
        ("50% off — really", [NUM, "%", "off", "-", "really"]),
        ("“quoted” and ‘this’", ['"', "quoted", '"', "and", "'", "this'"]),  # ’ continues a word, as on the device
        ("coffee ☕ please", ["coffee", "☕", "please"]),
        ("love ❤️ u", ["love", "❤️", "u"]),
        ("dev 🧑‍💻 life", ["dev", "🧑‍💻", "life"]),
        ("run 🏃‍♀️ now", ["run", "🏃‍♀️", "now"]),
        ("go 🇺🇸 1️⃣ #tag @you", ["go", "🇺🇸", "1️⃣", "#", "tag", "@", "you"]),
        ("café naïve über ünd", ["café", "naïve", "über", "ünd"]),
        ("a\tb\n c", ["a", "b", "c"]),
        ("¿Qué? ¡Sí!", ["?", "qué", "?", "!", "sí", "!"]),
        ("weird © 5€ x", ["weird", "©", NUM, SYM, "x"]),
        ("", []),
        ("   ", []),
        ("...", ["…"]),
    ],
)
def test_tokenize_cases(text, expected):
    assert tokenize(text) == expected

def test_is_word():
    assert is_word("hello") and is_word("don't") and is_word("e-mail") and is_word("é")
    assert not is_word(".") and not is_word("☕") and not is_word(NUM) and not is_word(UNK)
    assert not is_word("<s>") and not is_word("")

def test_word_tokens_match_device_word_rule():
    for tok in tokenize("Hello there, it's a well-known e-mail from Zoë 2nd 12 …"):
        if is_word(tok):
            assert normalize_word(tok) == tok
            assert active_prefix(tok) == tok

TRICKY = [
    "Hello, world!", "wait... what?!", "Don’t you DARE?", "coffee ☕ please ❤️ 🧑‍💻",
    "2nd at 12:30 on 9/11 for $1,000 (50% off)", "“quoted” — and ‘this’ - that", "a - b -- c",
    "goin' home wait- what", "x²  Ⅻth", "#tag @you [ok] {no} <hi> ~ ^ | \\ ° 5€", "¿Qué? ¡Sí! 。，：",
    "☕☕ ☕. ☕!", "email me: luke at example dot com", "1️⃣ 🇺🇸 🏃‍♀️ 👨‍👩‍👧",
]

@pytest.mark.parametrize("text", TRICKY)
def test_round_trip_is_stable(text):
    tokens = tokenize(text)
    text2 = detokenize(tokens)
    assert tokenize(text2) == tokens
    assert detokenize(tokenize(text2)) == text2  # normalized text is a fixed point

def test_detokenize_spacing():
    assert detokenize(["hello", ",", "world", "!"]) == "hello, world!"
    assert detokenize(["(", "yes", ")", "…"]) == "(yes)…"
    assert detokenize(["$", NUM, "%"]) == "$0%"
    assert detokenize([BOS, "hi", EOS, PAD]) == "hi"
    assert detokenize(["a", SYM, "b"]) == "a ¤ b"
    assert tokenize(detokenize([NUM, SYM])) == [NUM, SYM]

def small_tokenizer() -> Tokenizer:
    vocab = list(tk.SPECIALS) + list(tk.PUNCT) + ["☕", "❤️", "🧑‍💻", "the", "coffee", "i", "love"]
    return Tokenizer(vocab)

def test_tokenizer_layout_rules():
    with pytest.raises(ValueError):
        Tokenizer(["<s>", "<pad>"])
    with pytest.raises(ValueError):
        Tokenizer(list(tk.SPECIALS) + ["a", "a"])
    t = small_tokenizer()
    assert t.pad_id == 0 and t.bos_id == 1 and t.eos_id == 2 and t.unk_id == 3
    assert t.num_id == t.ids[NUM] and t.sym_id == t.ids[SYM]
    assert t.decode(t.word_ids) == ["the", "coffee", "i", "love"]
    assert t.decode(t.emoji_ids) == ["☕", "❤️", "🧑‍💻"]
    assert len(t.punct_ids) == len(tk.PUNCT)

def test_encode_decode_round_trip():
    t = small_tokenizer()
    ids = t.encode("I love coffee ☕!")
    assert ids[0] == t.bos_id and ids[-1] == t.eos_id
    assert t.decode(ids) == [BOS, "i", "love", "coffee", "☕", "!", EOS]
    assert t.text(ids) == "i love coffee ☕!"
    assert t.encode("i love coffee ☕!", bos=False, eos=False) == ids[1:-1]
    unk = t.encode("I love tea", bos=False, eos=False)
    assert unk == [t.ids["i"], t.ids["love"], t.unk_id]
    assert all(0 <= i < len(t) for i in unk)

def test_emoji_presentation_folding():
    t = small_tokenizer()
    # Typed without the variation selector, still the vocabulary's ❤️ token.
    assert t.tokens_of("love ❤") == ["love", "❤️"]
    assert t.id_of("❤") == t.ids["❤️"]
    assert t.id_of("🔥") == t.unk_id  # an emoji outside the vocabulary
    assert t.encode("🧑‍💻", bos=False, eos=False) == [t.ids["🧑‍💻"]]

def test_save_and_load(tmp_path):
    t = small_tokenizer()
    path = tmp_path / "vocab.txt"
    t.save(path)
    again = Tokenizer.load(path)
    assert again.tokens == t.tokens
    assert path.read_text(encoding="utf-8").count("\n") == len(t)

@pytest.mark.skipif(not (ASSETS / "emoji_en.txt").exists(), reason="keyboard assets not present")
def test_every_keyboard_emoji_is_one_token():
    from model.vocab import read_emoji_list

    emoji = read_emoji_list(ASSETS / "emoji_en.txt")
    assert len(emoji) > 250
    for e in emoji:
        assert tokenize(f"word {e} word") == ["word", e, "word"], e
        assert tk.is_emoji(e), e
        assert not is_word(e), e

@pytest.mark.skipif(not (ASSETS / "words_en.txt").exists(), reason="keyboard assets not present")
def test_every_dictionary_word_is_one_word_token():
    from model.vocab import read_word_list

    words = read_word_list(ASSETS / "words_en.txt")
    assert len(words) > 19_000
    bad = [w for w in words if tokenize(w) != [w]]
    assert bad == [], bad[:20]
