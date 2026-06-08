# codenames-interpretability

Layer-wise word representation geometry across seven transformer
architectures, evaluated on the Codenames Duet clue-guessing task. This
repository contains the source code for the experiments reported in:

> **João Pedro Ferreira Pereira Königs.** *Cross-Architectural Analysis of
Layer-wise Representations in
Transformer Language Models on a
Word Association Task.* Undergraduate thesis (TCC), UFRGS, 2026.

The repository accompanies the thesis and is the canonical record of how
the experimental measurements were produced. The thesis itself is the
source for the methodology's motivation, the empirical findings, and the
discussion of what those findings imply for cross-architectural
interpretability research.

---

## What this work does

The experiment extracts hidden state representations at every layer of
seven transformer-based language models when those models are presented
with Codenames Duet clue-guessing turns. For each turn, the cosine
similarity between the hint word and each candidate word is computed at
every layer, the resulting geometry is summarised through a small set of
scalar metrics, and the metrics are compared across the architectural
properties that distinguish the models in the suite.

The models span three attention patterns (causal, bidirectional,
mixed local+global) and three positional encoding schemes (absolute,
relative, rotary), with randomly initialised controls in both the
bidirectional and causal families. The design isolates the contributions
of attention pattern and positional encoding; pretraining objective is
recorded as descriptive metadata in the model table but is not isolated
as a controlled axis. Random-weight baselines distinguish the
contribution of training from that of architecture alone.

| # | Model | Notebook | Prefix | Attention | PE | Notes |
|---|-------|----------|--------|-----------|------|-------|
| 1 | Mistral-7B-Instruct-v0.2 | `notebooks/01_mistral.ipynb` | `mistral` | Causal | RoPE | Generation enabled |
| 2 | Qwen2.5-7B-Instruct | `notebooks/02_qwen.ipynb` | `qwen` | Causal | RoPE | Generation enabled |
| 3 | Qwen2.5-7B Random Baseline | `notebooks/03_qwen_random.ipynb` | `random_qwen` | Causal | RoPE | Random init, no generation |
| 4 | BERT-base-uncased | `notebooks/04_bert.ipynb` | `bert` | Bidirectional | APE | |
| 5 | BERT-base Random Baseline | `notebooks/05_bert_random.ipynb` | `random_bert` | Bidirectional | APE | Random init |
| 6 | T5-base (encoder only) | `notebooks/06_t5.ipynb` | `t5` | Bidirectional | RPE | |
| 7 | ModernBERT-base | `notebooks/07_modernbert.ipynb` | `modernbert` | Local + global | RoPE | |

---

## Repository structure

```
codenames-interpretability/
├── codenames/     # The Python package — the methodology
│   ├── contract.py                 # Frozen experimental parameters
│   ├── data.py                     # Dataset loading and sampling
│   ├── prompts.py                  # Byte-identical prompt construction
│   ├── spans.py                    # Token span detection and pooling
│   ├── extraction.py               # Per-board forward pass and metrics
│   ├── loop.py                     # Main extraction loop, both conditions
│   ├── generation.py               # Free-form generation, causal only
│   ├── sanity.py                   # SC1–SC7 diagnostic functions
│   ├── persistence.py              # File I/O for CSVs, parquet, NPZ
│   └── models/                     # One file per model: load_<name>()
│       ├── mistral.py
│       ├── qwen.py
│       ├── qwen_random.py
│       ├── bert.py
│       ├── bert_random.py
│       ├── t5.py
│       └── modernbert.py
├── notebooks/                      # Thin orchestration shells
│   ├── 00_validation.ipynb         # Bit-identity check against existing runs
│   ├── 01_mistral.ipynb
│   ├── 02_qwen.ipynb
│   ├── 03_qwen_random.ipynb
│   ├── 04_bert.ipynb
│   ├── 05_bert_random.ipynb
│   ├── 06_t5.ipynb
│   └── 07_modernbert.ipynb
├── tests/                          # Sanity tests for pure functions
├── docs/
│   ├── contract.md                 # Human-readable parameter contract
│   ├── runtime.md                  # The Colab-from-GitHub workflow
│   └── methodology.md              # Brief — points to thesis chapter
├── pyproject.toml
├── LICENSE
└── README.md
```

The package contains the methodology; the notebooks contain the
orchestration. Each notebook is a short shell (roughly fifteen cells)
that imports from the package, loads a model, runs the extraction loop
with full per-cell visibility into intermediate outputs, and writes
results to Google Drive. The same package is consumed by all seven
notebooks; the methodology has one implementation.

---

## Methodology in brief

This summary covers what the code does in operational terms. The full
methodology, with motivations, mathematical definitions, and
architectural rationale, is in Chapter 5 (Geometric Analysis Framework)
of the thesis. Chapter 4 of the thesis describes the dataset and the
per-turn data unit that the methodology consumes.

The procedure proceeds in five steps. For each turn in a sample of
2,000 Codenames Duet clue-giving turns drawn from CULTURAL CODES
\[Shaikh, 2023\], a prompt is constructed containing the hint word and
the candidate list. The forward pass of each model on the prompt is
intercepted, and the hidden state tensor at every layer is retained.
The contiguous token span of each word in the prompt is identified, and
the word's representation at a given layer is approximated by pooling
hidden states across that span; two pooling procedures (mean and
maximum-norm) are computed in parallel.

The cosine similarity between the hint vector and each candidate vector
is then computed at every layer, and a per-layer separation margin is
derived as the difference between the mean hint-to-target cosine and
the mean hint-to-non-target cosine. The margin is reported alongside an
anisotropy-adjusted form that divides by the layer-local standard
deviation of pairwise cosines, factoring out the architecture-specific
inflation of cosine magnitudes that varies across layers.

A residual confound is addressed by an ordering-perturbation
procedure: candidates are presented to the model in three orderings per
turn --- alphabetical (canonical) and two random permutations --- and
the variance of each candidate's cosine-to-hint across orderings is
decomposed into the part attributable to position (when a fixed word
appears at different list positions) and the part attributable to
identity (when distinct words are compared at fixed positions). The
ratio of identity-variance to total variance yields a per-layer semantic
signal ratio.

Three families of complementary metrics are computed. Rank-based
behavioural metrics --- top-1 accuracy, Hit@*k*, mean reciprocal rank
--- treat the final-layer cosine ordering as a prediction and quantify
its agreement with the ground-truth target set. Positional confound
diagnostics --- per-layer Spearman correlation between alphabetical
candidate position and cosine-to-hint similarity, and the variance
decomposition above --- quantify how much of the cosine signal is
attributable to position rather than identity. For the
instruction-tuned causal decoders, a generation-geometry concordance
procedure compares the cosine-based prediction against the model's own
free-form continuation under greedy decoding.

Two experimental conditions differ only in the presence of a
demographic preamble describing the clue-giver; the prompt is otherwise
byte-identical between conditions for a given turn. All seven models
are evaluated under both conditions, except where noted in the model
table above (random-init models omit generation).

---

## Reproducing the experiments

The experiments are designed to run on Google Colab with GPU
acceleration. The package is not designed for local execution.

### Quick start

1. Open one of the model notebooks from `notebooks/` directly in Colab
   (use Colab's "Open from GitHub" dialog and paste this repository's
   URL).
2. Run the first three cells, which clone the package into the Colab
   session, install it, and mount Google Drive.
3. Configure the `DATASET_PATH` in cell 4 to point to your local copy
   of the CULTURAL CODES `clue_generation.csv` file in Drive.
4. Run the subsequent cells in order. Each sanity check is a separate
   cell so its output is independently inspectable.

### Required data

The experiment depends on the CULTURAL CODES dataset \[Shaikh et al.,
2023\], which is not redistributed in this repository. The dataset is
available at <https://github.com/SALT-NLP/codenames> under a CC BY-SA
4.0 licence. The expected path in Drive is
`/content/drive/MyDrive/TCC/clue_generation.csv`, configurable per
notebook.

### Drive layout

Each notebook writes its outputs to a model-specific directory:

```
/content/drive/MyDrive/TCC/
├── clue_generation.csv             # Input dataset
├── mistral_outputs/
│   ├── mistral_general_no_social.csv
│   ├── mistral_general_with_social.csv
│   ├── mistral_metrics_no_social.parquet
│   ├── mistral_metrics_with_social.parquet
│   ├── mistral_generation_no_social.csv
│   ├── mistral_generation_with_social.csv
│   ├── mistral_vectors_subsample_*_f16.npz
│   ├── mistral_layer_margins_*.csv
│   ├── mistral_position_confound_by_layer.csv
│   └── mistral_shuffle_decomposition_by_layer.csv
├── qwen_outputs/
├── random_qwen_outputs/
├── bert_outputs/
├── random_bert_outputs/
├── t5_outputs/
└── modernbert_outputs/
```

The output schema is documented in `docs/contract.md`. The synthesis
notebook that consumes these outputs to produce the figures in the
thesis Results chapter is maintained separately from this repository.

### Resuming an interrupted run

A full-dataset run of a 7B model takes hours, and a Colab runtime can die
mid-run. The `run` subcommand checkpoints incrementally so a dead run can be
continued instead of restarted:

```bash
# First attempt (or after a runtime death, just re-run with --resume):
codenames-experiment run \
    --model mistral \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv \
    --output-dir /content/drive/MyDrive/TCC/mistral_outputs \
    --full --resume
```

How it works:

- Every `shard_boards` boards (Contract default 200), all output streams
  (metrics, general, generation, vectors, errors) are flushed to atomic
  checkpoint files in `--output-dir`, and a per-condition manifest
  (`{prefix}_{mode}_manifest.json`) records the committed board prefix.
- With `--resume`, the run skips boards already committed and reuses a
  condition that already finished, continuing from the last checkpoint. The
  result is **byte-identical** to an uninterrupted run: board sampling, shuffle
  seeds, and the vector subsample are all re-derived deterministically from the
  contract, independent of where the run was interrupted.
- Without `--resume`, any stale checkpoints/manifests in `--output-dir` are
  wiped before the run, so a fresh run is never contaminated by a previous
  aborted one. A successfully completed run removes all checkpoint and manifest
  files, leaving exactly the documented output files.

Checkpoints live in `--output-dir` (i.e. in Drive), so they survive the runtime
dying. Because flushes are batched (every `shard_boards` boards) rather than
per-board, the extra Drive I/O is negligible.

`--resume` continues a run of the **same** size: it is tied to the
`--sample-size`/`--full` the run was started with (the manifest records it). If
you point `--resume` at a directory whose checkpoints came from a different run
size, the run aborts with a clear error rather than risk mixing boards — re-run
without `--resume` (which wipes and starts fresh), or use the matching size.

### Reusing canonical results across run sizes

`--reuse-canonical` enables cross-size reuse: a later, larger run reuses the
per-board **canonical** (`permutation_id=0`) metrics, general, and generation
results computed by an earlier, smaller run, for the boards they share — instead
of recomputing those forward passes.

```bash
# 1. Explore on 1000 boards, populating the cache:
codenames-experiment run --model mistral --dataset .../clue_generation.csv \
    --output-dir .../mistral_outputs --sample-size 1000 --reuse-canonical

# 2. Later scale to the full dataset, reusing the shared boards' canonicals:
codenames-experiment run --model mistral --dataset .../clue_generation.csv \
    --output-dir .../mistral_outputs --full --reuse-canonical
```

How it works and what it guarantees:

- The canonical records are cached per condition in `--output-dir`, keyed by
  `row_id` (`{prefix}_canoncache_*` files). The cache is **persistent** (it is
  not deleted between runs) and only grows.
- When a board's `row_id` is in the cache, its canonical forward pass is
  skipped and the cached records are reused. **Shuffle permutations are always
  recomputed** for the current run (their seeds are sample-size-dependent), and
  boards in the vector subsample are never reused (they need a fresh pass to
  produce vectors). The output is therefore **byte-identical** to a run computed
  without the cache — reuse only skips work, it never changes results.
- Pass `--reuse-canonical` on **both** runs: the earlier run writes the cache,
  the later run reads it (and adds its own new boards).
- Reuse is **disabled under `--batch-size > 1`** (batched canonical values
  depend on which boards are padded together, so they are not interchangeable
  with the reference path); such runs neither read nor write the cache.
- `--reuse-canonical` composes with `--resume`: a run can both continue from its
  own checkpoints and reuse canonicals cached from a different-size run.

Because the cache duplicates the canonical third of the metrics, it uses extra
Drive space; only enable it when you intend to reuse across sizes.

### Validating against existing runs

The `notebooks/00_validation.ipynb` notebook runs the refactored
extraction pipeline on a 50-turn subsample and compares the resulting
outputs against the corresponding rows of the existing N=2000 runs
saved in Drive. Run this before trusting the refactored code with a
new experiment to verify that the package produces outputs interchangeable
with the original notebook-based runs.

### Acceleration (optional)

Three opt-in flags trade bit-identity with the reference path for
wall-clock speedup. None of them change the methodology; each is
characterised against the reference path by the `compare` subcommand
before being used in a production run.

| Flag | What it does | Expected drift |
|---|---|---|
| `--vectorize-anisotropy` | Replace nested O(n²) Python loop over candidate pairs with a single `M @ M.T` matrix product | ~1e-6 on `layer_mean_pairwise_cosine` |
| `--flash-attn` | Load Mistral or Qwen with `attn_implementation="flash_attention_2"` (requires `flash_attn` installed) | ~1-ULP per attention head per token, compounding through 28-32 layers |
| `--batch-size N` | Run N boards through one forward pass (with right-padding and attention mask). Default 1 = reference. | fp16 reduction-order drift on the sequence dim; >99% of final ranks agree |

Quantify the per-cell tolerance on a 50-board subsample first:

```
codenames-experiment compare \
    --model bert \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv \
    -n 50 --batch-size 8 --vectorize-anisotropy
```

Then enable the same flags on the production `run`:

```
codenames-experiment run \
    --model bert \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv \
    --output-dir /content/drive/MyDrive/TCC/bert_outputs \
    --batch-size 8 --vectorize-anisotropy
```

`--flash-attn` for Mistral or Qwen requires two model loads in `compare`
(the reference pass uses eager attention, the fast pass uses FA2). The
CLI handles this by freeing the reference model between passes so only
one 7B model is in GPU memory at a time.

---

## Mapping the thesis to the code

| Thesis chapter / section | Implemented in |
|---|---|
| Ch. 4: The Codenames Duet Task (data, per-turn unit, attributes) | `data.py` (dataset loading, candidate construction) |
| Ch. 5.3: Turn Sampling and Prompt Construction | `data.py`, `prompts.py` |
| Ch. 5.4: Hidden State Extraction and Pooling | `extraction.py`, `spans.py` |
| Ch. 5.5: Geometric Metrics (margin, adjusted margin) | `extraction.py`, `sanity.py` (SC5) |
| Ch. 5.6.1: Rank-based Behavioural Metrics | `extraction.py`, `sanity.py` (SC4) |
| Ch. 5.6.2: Positional Confound Diagnostics | `extraction.py`, `sanity.py` (SC6, SC7) |
| Ch. 5.6.3: Between-Condition Diagnostics | `loop.py` (with_social condition handling) |
| Ch. 5.7: Generation–Geometry Concordance | `generation.py`, `sanity.py` (SC4 concordance block) |
| Ch. 5.8: Implementation and Reproducibility | `contract.py`, all of the above |

The model loaders in `codenames/models/` are not described in any
individual thesis section; they are the operational realisations of the
model suite described in Chapter 5.2 (The Model Suite).

---

## Software environment

The package is designed for Colab's default scientific Python
environment and does not pin its dependencies. Tested against the
versions of `torch`, `transformers`, `numpy`, `pandas`, `tqdm`, `scipy`,
`pyarrow`, and `accelerate` that ship with Google Colab as of
\<date of final run\>. ModernBERT requires `transformers>=4.48.0`, which
Colab provides; other models work with earlier versions.

GPU memory: the seven-billion-parameter causal models (Mistral, Qwen,
Random Qwen) require a GPU with at least 16 GB of memory. The
encoder-only models run on Colab's free-tier GPU.

---

## License

MIT. See `LICENSE`.

The CULTURAL CODES dataset \[Shaikh, 2023\] is not redistributed in
this repository and is subject to its own licensing terms.
