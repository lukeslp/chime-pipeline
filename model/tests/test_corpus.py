"""Corpus rules: the name-masked template key, the per-dialogue split, the source labels and the v3 readers."""

import json

import pytest

pytest.importorskip("pyarrow")

pytest.importorskip("xxhash")

from model import corpus  # noqa: E402

def test_template_key_masks_dialogue_names_and_capitalized_words():
    a = corpus.template_key("Hey, Anahi. I wanted to talk to you about something important.", frozenset({"anahi"}))
    b = corpus.template_key("Hey, Maria. I wanted to talk to you about something important.", frozenset({"maria"}))
    assert a == b
    assert "<name>" in a and "anahi" not in a
    # A capitalized word that is not the first word is masked even without a name list.
    c = corpus.template_key("Hey, Maria. I wanted to talk to you about something important.", None)
    assert c == a
    # Forms of "I" are never names. "I'd" and "I'm" pass because the corpus regex splits them into "I" and a lowercase piece.
    assert corpus.template_key("So I said I'd go and I'm still going, I'll be there.", None).count("<name>") == 0
    # A possessive of a known name is masked too.
    assert "<name>" in corpus.template_key("That is Anahi's car and it is the same car as before.", frozenset({"anahi"}))

def test_template_key_exempts_short_lines():
    assert corpus.template_key("Thanks, Mom!", frozenset({"mom"})) is None
    assert corpus.template_key("one two three four five six seven", None) is None
    assert corpus.template_key("one two three four five six seven eight", None) is not None

def test_template_key_keeps_lines_that_differ_beyond_the_name():
    a = corpus.template_key("Hey, Anahi. I wanted to talk to you about the party.", frozenset({"anahi"}))
    b = corpus.template_key("Hey, Maria. I wanted to talk to you about the game.", frozenset({"maria"}))
    assert a != b

def test_split_is_by_unit_and_stable():
    unit = "soda:123"
    assert corpus.split_for("soda", unit) == corpus.split_for("soda", unit)
    # Every line of a dialogue shares the dialogue's split, whatever its text.
    splits = {corpus.split_for("soda", unit) for _ in range(5)}
    assert len(splits) == 1
    buckets = {corpus.split_for("soda", f"soda:{i}") for i in range(2000)}
    assert buckets == {"train", "dev", "test"}

def test_soda_names_collects_speakers_and_persons():
    names = corpus.soda_names({"speakers": ["Veda", "Priest", "Veda"], "PersonX": "Veda", "PersonY": "Mary Ann", "PersonZ": ""})
    assert names == frozenset({"veda", "priest", "mary", "ann"})

def test_sources_are_labelled_and_smoltalk_is_refused():
    assert set(corpus.SOURCES) == {"soda", "nus", "taskmaster", "oasst2", "taskmaster3"}
    assert corpus.HUMAN_SOURCES == ["nus", "taskmaster", "oasst2", "taskmaster3"]
    assert not corpus.SOURCES["soda"].human_written
    assert corpus.SOURCE_ORDER == ["nus", "taskmaster", "oasst2", "taskmaster3", "soda"]
    assert set(corpus.READERS) == set(corpus.SOURCES)
    assert any("smol-smoltalk" in what and "Llama" in why for what, why in corpus.REFUSED)
    text = corpus.licenses_markdown({})
    assert "Smol-SmolTalk" not in text.split("## Refused")[0]
    assert "human-written" in text

def test_presets_keep_v2_and_put_human_sources_first():
    v2, v2_dir = corpus.PRESETS["v2"]
    v3, v3_dir = corpus.PRESETS["v3"]
    # The v2 build is the published corpus: same sources, same order, same directory.
    assert (v2, v2_dir) == (["nus", "taskmaster", "soda"], "corpus")
    assert v3_dir == "corpus-v3" and v3[:2] == v2[:2] and v3[-1] == "soda"
    # Every human source precedes SODA, so a typed line wins each dedupe tie.
    assert all(corpus.SOURCES[k].human_written for k in v3[:-1])
    assert all(v3.index(k) < v3.index("soda") for k in corpus.HUMAN_SOURCES)

def test_refusals_say_why():
    refused = dict(corpus.REFUSED)
    persona = next(why for what, why in refused.items() if what.startswith("PersonaChat"))
    assert "license unclear" in persona and "parl.ai" in persona
    oanc = next(why for what, why in refused.items() if "Switchboard" in what)
    assert "conflict" in oanc and "spoken" in oanc
    assert any(what.startswith("Taskmaster-2") and "spoken" in why for what, why in refused.items())
    assert any(what.startswith("QuAC") and "CC BY-SA" in why for what, why in refused.items())

def test_licenses_list_only_the_build_sources():
    v2 = corpus.licenses_markdown({}, corpus.PRESETS["v2"][0])
    assert "OpenAssistant" not in v2.split("## Refused")[0] and "TicketTalk" not in v2
    assert "Two of the three sources are human-written (NUS SMS and Taskmaster-1)" in v2
    v3 = corpus.licenses_markdown({}, corpus.PRESETS["v3"][0])
    assert "Four of the five sources are human-written (NUS SMS, Taskmaster-1, OASST2 and Taskmaster-3)" in v3
    assert "Apache-2.0, CC BY 4.0" in v3 and "2026-09-25 and 2026-10-03" in v3
    # SODA's generator terms are recorded with the source.
    soda = v3.split("## SODA")[1].split("## Refused")[0]
    assert "Terms beyond the license" in soda and "compete with OpenAI" in soda
    attribution = v3.split("## Attribution")[1]
    assert all(f"- {corpus.SHORT_NAMES[k]}:" in attribution for k in corpus.PRESETS["v3"][0])
    assert corpus.human_written_note(["nus", "soda"]) == "SODA is synthetic dialogue generated by a language model; only NUS SMS was typed by people"

def test_oasst2_takes_english_human_prompter_turns_only():
    rows = [
        {"role": "prompter", "lang": "en", "message_tree_id": "t1", "text": "how do i fix my bike"},
        {"role": "assistant", "lang": "en", "message_tree_id": "t1", "text": "Check the chain."},
        {"role": "prompter", "lang": "es", "message_tree_id": "t2", "text": "hola"},
        {"role": "prompter", "lang": "en", "message_tree_id": "t3", "text": "made up", "synthetic": True},
        {"role": "prompter", "lang": "en", "message_tree_id": "t4", "text": "gone", "deleted": True},
        {"role": "prompter", "lang": "en", "message_tree_id": "t5", "text": "rejected", "review_result": False},
        {"role": "prompter", "lang": "en", "message_tree_id": "t1", "text": "thanks!", "review_result": None},
    ]
    counts = corpus.Counter()
    got = list(corpus.oasst2_prompter_turns((json.dumps(r) for r in rows), counts))
    assert got == [("oasst2:t1", None, "how do i fix my bike"), ("oasst2:t1", None, "thanks!")]
    assert counts == {"taken": 2, "assistant": 1, "not_english": 1, "synthetic": 1, "deleted": 1, "failed_review": 1}

def test_taskmaster_takes_the_customer_side_of_either_release():
    dialogs = [{"conversation_id": "c1", "utterances": [
        {"speaker": "USER", "text": "two tickets please"},
        {"speaker": "ASSISTANT", "text": "Sure."},
        {"speaker": "user", "text": "for tonight"},
    ]}]
    speakers = corpus.Counter()
    got = list(corpus.taskmaster_user_turns(dialogs, "taskmaster3", speakers))
    assert got == [("taskmaster3:c1", None, "two tickets please"), ("taskmaster3:c1", None, "for tonight")]
    assert speakers == {"USER": 1, "ASSISTANT": 1, "user": 1}
    # A dialogue id is hashed with its source, so the two Taskmaster releases split independently.
    assert corpus.split_for("taskmaster3", "taskmaster3:c1") == corpus.split_for("taskmaster3", "taskmaster3:c1")

def test_output_dir_built_from_other_sources_is_refused(tmp_path):
    corpus.check_output_dir(tmp_path, ["nus"])  # nothing there yet
    (tmp_path / "manifest.json").write_text(json.dumps({"sources": {"nus": {}, "taskmaster": {}, "soda": {}}}))
    corpus.check_output_dir(tmp_path, ["nus", "taskmaster", "soda"])  # the same build may be rebuilt
    with pytest.raises(SystemExit):
        corpus.check_output_dir(tmp_path, corpus.PRESETS["v3"][0])

def test_filter_and_keys_unchanged():
    assert corpus.filter_reason("see http://x.com now") == "url"
    assert corpus.filter_reason("call me on 555 123 4567") == "phone"
    assert corpus.filter_reason("Hello there!") is None
    # The corpus regex splits at apostrophes, unlike the keyboard's WORD_REGEX; pinned so corpus keys stay reproducible.
    assert corpus.normalized_key("Don’t stop, DON'T!") == "don t stop don t"
