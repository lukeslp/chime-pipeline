# Chime CUDA trainer (the published ×8 run)

Author: Luke Steuber. Apache-2.0; see ../LICENSE and ../NOTICE.
This is the CUDA trainer from the completed x8 experiment, with its equation
port and NumPy sampler. x8 means oversampling, not model size: NUS, Taskmaster-1
and OASST2 repeat eight times; Taskmaster-3 and SODA remain once per epoch.
The shape is 16,384 tokens, embedding/projection 96, hidden 670, 2,025,114
parameters. The completed run reached 102,195 updates (three batcher epochs),
selecting step 98,000 on human-macro development perplexity, not test data.
Development loss and training completion do not establish everyday typing benefit.

cifg_torch.py and train_cuda.py are the exact hashed launch bytes. cifg_data.py
has only two portability edits: a private-path docstring and the default data
root. Config JSON paths and serialization are transformed separately. Every
original input and output digest, transformation and provenance is recorded.
The NumPy helper was extracted before the first MLX import from the original
CIFG module; it retains the original MLX docstring but does not import MLX.
No MLX library is needed for the copied source checks or trainer CLI help.
The CUDA trainer consumes pretokenized inputs; its sampler does not use the
optional sibling tokenizer or its fallback. Corpus construction and export
are outside this package. There are no static app assets or learned vocabulary.

Safe checks from this directory with an already installed environment:

    PYTHONDONTWRITEBYTECODE=1 python source_tests.py
    PYTHONDONTWRITEBYTECODE=1 python train_cuda.py --help

The tests require NumPy, PyTorch and safetensors and fail if unavailable. They
use invented token IDs, tiny CPU models, temporary synthetic weights and an
independent NumPy optimizer reference. They do not touch original checkpoints,
run a training job, download data, call a service, test CUDA capture or assert
MLX parity. Redirect TMPDIR and bytecode outside this read-only tree.

requirements-reference.txt records the historical bootstrap pins; installation
is an explicit separate action. The recorded container is in reference_recipe.json.
Local CPU checks in a different environment do not validate that CUDA container.
The exact trainer requires CUDA for training and imports NumPy, torch and
safetensors at startup. huggingface_hub is imported only when its main entry
point is executed past argument parsing. That entry point requires a private
dataset destination and uploads checkpoints. No launch/upload command is run
by source preparation. --help exits before network, data access or output creation.
The historical trainer uses assertions for its private-destination and new-folder
checks. Never run it with -O or PYTHONOPTIMIZE: Python optimization removes those
checks. It creates the output directory before checking destination privacy.
These original source bytes are preserved; no hardened launcher is included.

For a separately authorized reproduction, the original invocation is:

    python train_cuda.py --package <verified-private-input-root> --mode full --out <new-output-directory> --repo <private-dataset> --prefix <new-prefix>

This command starts training and uploads results; it is documentation only.
The exact private input root contains the three original source files,
manifest.json and every required_inputs.json entry at its original digest.
The original runner verifies that manifest. This sanitized source tree is
NOT that input root: it omits tokenized train/dev data, vocabulary, corpus
metadata, explicit MLX-seeded tensors and parity tensors. It contains no test
split, corpus text, typing logs, personal state, secrets, checkpoints, source
history or internal instructions. Hashes describe omitted inputs without
redistributing their contents. They were read from the frozen manifest,
not rehashed from data during this preparation.

config.input.json and config.completed.json preserve all numerical fields;
vocab_path becomes data/vocab.txt and corpus becomes corpus-v3. The historical
cache fingerprint is retained as an opaque identity, including its path hash.
These configs are reference metadata, not a substitute for the original input
manifest. Sanitized helper/config bytes cannot match that original manifest.
An exact-input reconstruction needs independently authorized matching input
bytes; generating new tensors from a seed is not proven byte-identical.
Source hashes alone cannot reproduce missing data or grant its redistribution.

The trainer uses float32, uncorrected AdamW moments, clipping, fixed masked
64-position CUDA graphs, deterministic algorithms and disabled TF32. See the
recipe for the schedule, sampler, evaluation and environment settings. CPU/GPU
reductions and library/hardware differences limit numerical reproduction;
neither identical weights nor training trajectories are promised. The preserved
MLX r3 stopped run is not a resume input. The CUDA runner's optional resume
requires matching optimizer/cursor and input-manifest identity, not weights alone.

provenance-package-manifest.json is the manifest of the private review tree this directory was
published from, kept as it was written on 2026-10-07. Its digests match every source, config and
provenance file here byte for byte. Two entries no longer match by design: the review tree's
LICENSE-REVIEW.md, replaced by the repository's LICENSE and NOTICE when Apache-2.0 was confirmed on
2026-10-11, and this README, whose first lines changed for the same reason. Its `license_status`
field records the review state at the time, not the present licence. The tree digest in it is
SHA-256 of canonical UTF-8 JSON of path-to-SHA-256 entries (sorted keys, indent 2, unescaped
Unicode, trailing newline), excluding the manifest itself.
