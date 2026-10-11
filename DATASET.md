---
language:
  - en
license: other
license_name: mixed-cc-by-4.0-and-apache-2.0
license_link: https://github.com/lukeslp/chime-pipeline/blob/main/model/licenses.md
pretty_name: Chime training corpus (recipe)
task_categories:
  - text-generation
size_categories:
  - 100M<n<1B
tags:
  - keyboard
  - next-word-prediction
  - sms
  - dialogue
---

# Chime training corpus: a dataset card for a build recipe

This card describes the corpus that trained [Chime-SMS-2M](https://huggingface.co/lukeslp/chime-sms-2m).
The text is not redistributed. `python -m model.corpus --preset v3` from the
[pipeline repository](https://github.com/lukeslp/chime-pipeline) rebuilds it from the pinned
upstream versions below, and `model/licenses.md` there records each source's licence as read on
the date it was checked. By Luke Steuber.

## Sources

| Source | Written by | Licence | Pinned version | Taken |
| --- | --- | --- | --- | --- |
| NUS SMS Corpus, English portion | people (volunteers' real text messages, 2003 to 2015, mostly Singapore) | CC BY 4.0 | release 2015.03.09, GitHub mirror commit `bf0a365` | every `en` message's text, with the release's own redaction tags removed |
| Taskmaster-1, written self-dialogs | people (crowd workers typing both sides of a task dialogue) | CC BY 4.0 | commit `d92cb6a` (2024-03-19) | `USER` utterances of `self-dialogs.json` |
| OpenAssistant Conversations 2 | people (volunteers writing requests to an assistant) | Apache-2.0 | revision `179dd21` (2024-01-11) | `prompter` turns, English, not synthetic, not deleted, review not failed |
| Taskmaster-3 (TicketTalk) self dialogs | people (crowd workers, one movie-ticketing task) | CC BY 4.0 | commit `d92cb6a` | `user` utterances |
| SODA | a language model (GPT-3.5 dialogues from commonsense triples) | CC BY 4.0 | revision `fdc848a` (2023-01-04) | every turn of `dialogue` in train, valid and test |

Refused on review: anything with a NonCommercial clause, an unclear licence or no licence, and
anything written by a model but offered as human text. Assistant turns of OASST2 and the ticketing
agent's turns of Taskmaster-3 are not used.

## Processing

Each utterance is one line. The filter drops empty lines, lines over the length cap, lines with no
words, phone numbers, lines that are mostly symbols, code or markup, URLs, non-Latin text,
credential-looking strings and email addresses. Deduplication removes exact duplicates, then
name-masked template duplicates (the same line with character names swapped, which SODA produces in
quantity), then near duplicates. Splits are by dialogue or message so that no normalized line appears
in more than one split. A leak check that masks capitalized words and looks every dev and test line
of eight or more words up in train finds 0.04% of them.

## Counts (v3 build, 2026-10-03)

| Split | Lines | Words | Unique words |
| --- | ---: | ---: | ---: |
| train | 7,858,648 | 161,625,719 | 97,883 |
| dev | 79,810 | 1,636,229 | 20,920 |
| test | 80,290 | 1,652,754 | 21,112 |

| Source | Raw utterances | Kept lines | Words | train / dev / test lines |
| --- | ---: | ---: | ---: | --- |
| NUS SMS | 55,835 | 46,651 | 526,296 | 45,697 / 463 / 491 |
| Taskmaster-1 | 85,693 | 57,692 | 527,196 | 56,541 / 567 / 584 |
| OASST2 | 22,741 | 21,386 | 421,533 | 20,935 / 261 / 190 |
| Taskmaster-3 | 239,439 | 139,762 | 1,493,272 | 136,813 / 1,587 / 1,362 |
| SODA | 11,333,873 | 7,753,257 | 161,946,405 | 7,598,662 / 76,932 / 77,663 |

Human-written text is 265,491 lines (3.31%) and 2,968,297 words (1.80%) of the corpus. The trainer
repeats the human-written sources (eight times in the published run, Taskmaster-3 once) and scores
them as their own columns; the vocabulary ranks words by a mixture that gives the four human sources
half the weight.

## Known properties and defects

- 98% of the corpus by line is SODA: fluent, clean synthetic dialogue. The human-written sources are
  the register a keyboard sees and are small.
- NUS SMS is Singapore English texting from 2003 to 2015, not current American texting, and its
  senders appear in more than one split, so NUS development and test scores are flattered.
- Taskmaster-3 is a single task; 38% of its user turns are exact duplicates ("yes", "ok", "thank
  you") and are removed.
- The Latin-script filter keys on Unicode names containing "LETTER", which CJK ideographs lack, so
  635 train lines (0.008%) keep Chinese or mojibake characters, and one mojibake token took a
  vocabulary slot.
- The filter removes text that looks sensitive; it does not prove the source text is anonymous.
  The NUS release's own redaction tags are honoured.

## Intended use

Training small on-device next-word and completion models for English messaging. It is not a
chat or instruction dataset and contains no personal typing logs.

## Citation and credits

The authors of NUS SMS, Taskmaster, OpenAssistant and SODA do not endorse this work. Cite the
originals: Chen and Kan (2013) for NUS SMS; Byrne et al. (2019) and (2021) for Taskmaster-1 and
TicketTalk; Köpf et al. (2023) for OpenAssistant; Kim et al. (2023) for SODA.
