# Chime pipeline

The training pipeline behind [Chime-SMS-2M](https://huggingface.co/lukeslp/chime-sms-2m), the
2 MB next-word and completion model that ranks the suggestion strip in
[LocalType](https://dr.eamer.dev/downloads/apps/localtype/), my Android keyboard. By Luke Steuber.

**November 2026: Chime 2.** LocalType has bundled Chime 2 since 0.5.58. Its weights and card are
at [lukeslp/chime](https://huggingface.co/lukeslp/chime) (tag `v2.0`), which lists every version.
Chime 2 has the same shape, vocabulary and tokenizer as the model described here. It trained on
this pipeline's corpus plus 1,340,659 English sentences from Tatoeba (CC BY 2.0 FR) and 1,000,000
sentences from C4 web text (ODC-BY 1.0, with the Common Crawl terms of use). A filter built from my
own sent text messages chose those C4 sentences: the messages only ranked public sentences, and
none of their words are in the training text. The Tatoeba and C4 steps are not in this repository
yet, and because the messages stay private, the C4 choice can't be repeated exactly. The rest of
this README describes the October model, Chime-SMS-2M (Chime 1.1).

It goes from five licence-checked public text sources to an int8 model file:

1. `model/corpus.py` downloads pinned upstream datasets, filters, deduplicates and splits them
   (`python -m model.corpus --preset v3`).
2. `model/vocab.py` builds the 16,384-entry word-level vocabulary, ranking by a human-weighted
   mixture of the sources and interleaving a keyboard word list as cover.
3. `model/tokenize.py` is the tokenizer the keyboard's Kotlin port and browser port copy.
4. `model/train.py` trains the CIFG-LSTM on Apple silicon through MLX; `cuda/train_cuda.py` is the
   PyTorch trainer, with the same equations and sampler, that trained the published run on one A100.
5. `model/export.py` writes the int8 file and its parity goldens; `model/prior.py` builds the
   small corpus prior the keyboard's n-gram ranker uses alongside the model.

What this repository does not contain: corpus text, typing logs, checkpoints or weights. The
weights, vocabulary and a NumPy reference loader are on Hugging Face. Every corpus source, its
licence, pinned version and what was taken from it is in [`model/licenses.md`](model/licenses.md),
and [`DATASET.md`](DATASET.md) is the dataset card for the assembled corpus.

## The model

One CIFG-LSTM layer (670 units, recurrent projection to 96) over a 96-dimensional embedding tied to
the output layer: 2,025,114 parameters, 2,221,879 bytes as int8 with the vocabulary. The published
checkpoint is step 98,000 of a 102,195-step run (three passes over the corpus with the human-written
sources repeated eight times, 7,343 s on one A100), selected on the mean per-source development
perplexity of the human-written sources. Int8 export raised development perplexity by 0.101%. On a
Pixel 9a the warm next-word call with retained context takes 3.09 ms at p95.

## Setup

Python 3.12 or later. Install dependencies in your own environment; `model/pyproject.toml` records
the minimum versions. Corpus processing uses NumPy, PyArrow, xxhash and `huggingface_hub`;
training, export and the corpus counts need MLX on a supported Mac; the prior uses NumPy; tests use
pytest and skip the MLX modules when MLX is absent. The CUDA trainer needs NumPy, PyTorch and
safetensors, and CUDA to train.

Run from this directory with package imports (`from model import tokenize`) and
`python -m model.<module>`: `model/tokenize.py` would otherwise shadow the standard library module.
The data root defaults to `~/.local/share/localtype-model`; `LOCALTYPE_MODEL_DATA` overrides it.
Importing never downloads or trains. Only the corpus command fetches the pinned upstream datasets;
the other commands read local inputs. Builds overwrite their own outputs, so use new corpus,
vocabulary and run names, and never reuse a stopped run directory: `train.py` has no resume.

## Reproducing the published model

The completed reference recipe is `reference_recipe.json`. These commands rebuild the corpus and
vocabulary and train the MLX ×4 reference (`cifg-v3-3ep-r2`); they are not run by anything here:

```sh
export LOCALTYPE_MODEL_DATA="$HOME/.local/share/localtype-model-reproduction"
python -m model.corpus --preset v3
python -m model.vocab --corpus "$LOCALTYPE_MODEL_DATA/corpus-v3" --out "$LOCALTYPE_MODEL_DATA/vocab-v3" \
  --size 16384 --weights nus=0.125,taskmaster=0.125,oasst2=0.125,taskmaster3=0.125,soda=0.5 \
  --cover app/src/main/assets/words_en.txt --cover-share 0.5 --cover-min-count 5
python -m model.train train --name reproduction-4x --corpus "$LOCALTYPE_MODEL_DATA/corpus-v3" \
  --vocab "$LOCALTYPE_MODEL_DATA/vocab-v3/vocab.txt" --human-oversample 4 --oversample taskmaster3=1 \
  --select-on human-macro --max-steps 96423 --eval-every 1000
python -m model.export --name reproduction-4x --check-lines 5000
```

The published ×8 checkpoint came from `cuda/train_cuda.py` on a frozen input bundle (tokenized
splits, vocabulary, MLX-seeded initial tensors, parity fixtures) that is hashed in
`cuda/required_inputs.json` but not distributed; see [`cuda/README.md`](cuda/README.md) for the
exact invocation, the recipe and its numerical limits. Neither path promises bit-identical weights:
GPU and CPU reductions and library versions move the trajectory.

The vocabulary builder reports coverage on the dev and test splits as part of its command line;
do not point it at a split you intend to keep sealed.

## Tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider model/tests
cd cuda && PYTHONDONTWRITEBYTECODE=1 python source_tests.py && python train_cuda.py --help
```

The tests use synthetic inputs and the included word lists, never a corpus or weights. The CUDA
checks build tiny CPU models and an independent NumPy optimizer reference; they do not touch CUDA.

## Provenance

`provenance-mlx-package-manifest.json` and `cuda/provenance-package-manifest.json` are the manifests
of the two private review trees this repository was published from, kept as they were written on
2026-10-05 and 2026-10-07. They pin each tree's files by SHA-256, and the first names the source
commit. Every source, test, config, word-list and provenance file here matches its digest byte for
byte. The entries that do not match are the review trees' own `LICENSE-REVIEW.md` files, replaced
here by `LICENSE` and `NOTICE`, and the three READMEs, rewritten for publication. The manifests'
`license_status` field records the review state at the time, not the present licence.
`cuda/source_provenance.json` records the two portability edits to `cifg_data.py` and hashes the
inputs and outputs of the completed run. The MLX tree is pinned by commit and digests; it is not
claimed to be the bit-identical launch snapshot of the reference run.

## Licences

- Source code (`model/`, `cuda/`): Apache License 2.0, see [`LICENSE`](LICENSE).
- `app/src/main/assets/words_en.txt` and `words_en_profanity.txt` derive from
  [FrequencyWords](https://github.com/hermitdave/FrequencyWords) (OpenSubtitles 2018) and are
  CC BY-SA 4.0; `words_en_modern.txt` and `emoji_en.txt` are mine and carry the same licence. See
  [`NOTICE`](NOTICE).
- The corpus sources keep their own terms (NUS SMS Corpus, Taskmaster-1, Taskmaster-3 and SODA:
  CC BY 4.0; OpenAssistant Conversations 2: Apache-2.0), recorded in `model/licenses.md`. No author
  of those datasets endorses this work.
- The published weights and vocabulary are CC BY-SA 4.0 and the reference inference code is
  Apache-2.0, on the [model page](https://huggingface.co/lukeslp/chime-sms-2m).
