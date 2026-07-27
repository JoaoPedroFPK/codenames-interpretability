# codenames-interpretability

Layer-wise word representation geometry across seven transformer
architectures, measured on the clue-selection turns of the Codenames Duet
word-association game. This repository holds the experimental code for the
thesis:

> **João Pedro Ferreira Pereira Königs.** *Cross-Architectural Analysis of
> Layer-wise Representations in Transformer Language Models on a Word
> Association Task.* Undergraduate thesis (TCC), Instituto de Informática,
> UFRGS, 2026. Advisor: Prof. Dr. Bruno Iochins Grisci.

The thesis is the reference for the motivation, the full methodology, and the
empirical findings; this repository is the record of how the measurements were
produced.

## What this work asks

Where in a transformer's layer stack do the representations of a hint word and
the words it is meant to evoke come closest, and does the answer depend on how
the model attends to context and on how it was trained? The question is posed
through Codenames Duet turns, each of which pairs a human-produced hint with a
small, closed pool of candidate words labelled as targets the hint should
evoke, avoid words it should suppress, and neutrals. At every layer, the
separation between the mean hint-to-target and mean hint-to-non-target cosine
similarities is measured and adjusted for anisotropy so it is comparable
across layers and architectures.

The headline result of the thesis is that the peak separation is not uniformly
at the final layer: its location tracks each model's family. T5's encoder
concentrates it at the output, BERT spreads it across the stack, the causal
decoders peak in early-to-middle layers, and the local-windowed encoder is
strongest at its embedding layer. Random-weight controls show that any
hint–target separation requires training, while the causal decoders' bias
toward earlier-listed candidates is already present at initialisation.

## The model suite

The seven models populate a grid over two architectural properties, attention
pattern and positional encoding, with randomly initialised controls in both
the bidirectional and the causal families. Pretraining objective is recorded
as metadata but is not isolated by the design.

| # | Model | Notebook | Prefix | Attention | PE | Notes |
|---|-------|----------|--------|-----------|------|-------|
| 1 | Mistral-7B-Instruct-v0.2 | `notebooks/01_mistral.ipynb` | `mistral` | Causal | RoPE | Generation enabled |
| 2 | Qwen2.5-7B-Instruct | `notebooks/02_qwen.ipynb` | `qwen` | Causal | RoPE | Generation enabled |
| 3 | Qwen2.5-7B Random Baseline | `notebooks/03_qwen_random.ipynb` | `random_qwen` | Causal | RoPE | Random init, no generation |
| 4 | BERT-base-uncased | `notebooks/04_bert.ipynb` | `bert` | Bidirectional | APE | |
| 5 | BERT-base Random Baseline | `notebooks/05_bert_random.ipynb` | `random_bert` | Bidirectional | APE | Random init |
| 6 | T5-base (encoder only) | `notebooks/06_t5.ipynb` | `t5` | Bidirectional | RPE | |
| 7 | ModernBERT-base | `notebooks/07_modernbert.ipynb` | `modernbert` | Local + global | RoPE | |

## Repository layout

```
codenames-interpretability/
├── codenames/                      # The Python package — the methodology
│   ├── contract.py                 # Frozen experimental parameters
│   ├── data.py                     # Dataset loading and turn sampling
│   ├── prompts.py                  # Prompt construction (three chat-template strategies)
│   ├── spans.py                    # Token-span detection and pooling
│   ├── extraction.py               # Per-board forward pass and metrics
│   ├── loop.py                     # Main extraction loop, both conditions
│   ├── generation.py               # Free-form generation, causal models only
│   ├── sanity.py                   # SC1–SC7 diagnostic functions
│   ├── checkpoint.py, canon_cache.py   # Resume and cross-size reuse
│   ├── comparison.py, diagnostics.py, depcheck.py
│   ├── persistence.py              # File I/O for CSV / parquet / NPZ
│   ├── models/                     # One file per model: load_<name>()
│   ├── analysis/                   # Cross-model tables and the figure set
│   └── viz/                        # Per-board heatmaps and 2D projections
├── notebooks/                      # Thin per-model orchestration shells (01–07)
├── pyproject.toml
├── LICENSE
└── README.md
```

The package contains the methodology; each notebook is a short shell that
imports it, loads one model, runs the extraction loop with per-cell visibility,
and writes results. All seven notebooks consume the same package.

## Methodology in brief

The full methodology, with motivations and definitions, is in Chapters 4 and 5
of the thesis. In operational terms:

For each clue-giving turn drawn from the CULTURAL CODES dataset
\[Shaikh et al., 2023\], a prompt is built containing a task instruction, the
hint, and the candidate list, and a forward pass of the model is intercepted
to retain the hidden state at every layer. The thesis runs over all 7,703
turns the dataset supplies; the CLI defaults to a 2,000-board sample and takes
`--full` for the complete set, while the notebooks run the full dataset. Each
word's span is pooled into a single vector by two procedures reported in
parallel (mean and maximum-norm).

The cosine similarity between the hint vector and each candidate vector is
computed at every layer, and a per-layer separation margin is derived as the
difference between the mean hint-to-target and mean hint-to-non-target cosine.
The margin is reported alongside an anisotropy-adjusted form that divides by
the layer-local standard deviation of pairwise cosines.

Because candidates are presented alphabetically, a candidate's position is
correlated with its identity. To separate the two, each turn is also run under
two random candidate orderings, and the cosine variance is decomposed into a
positional component and an identity component (the per-layer semantic signal
ratio). A per-layer Spearman correlation between candidate position and
cosine-to-hint gives the direction of the confound.

Three further metric families complement the geometry: cosine-rank metrics
(top-1 accuracy, Hit@*k*, mean reciprocal rank) treat the final-layer cosine
ordering as a prediction of the target labels; a generation–geometry
concordance procedure compares that prediction against the free-form
continuation of the two instruction-tuned causal models; and projection
trustworthiness scores the 2D UMAP layouts used in the figures.

Two conditions differ only in the presence of a demographic preamble
describing the clue-giver (`no_social` vs `with_social`); the prompt is
otherwise byte-identical between conditions for a given turn.

## Running the experiments

The extraction is built for Google Colab with a GPU; the 7B causal models need
at least 16 GB of GPU memory, while the encoders run on the free tier. The
post-hoc aggregation and figures run locally.

### Quick start (Colab)

1. Open one of the model notebooks under `notebooks/` in Colab.
2. Run the first cells, which clone and install the package and mount Drive.
3. Set `DATASET_PATH` to your copy of the CULTURAL CODES `clue_generation.csv`.
4. Run the remaining cells in order; each sanity check is its own cell.

### Required data

The experiment depends on the CULTURAL CODES dataset
\[Shaikh et al., 2023\], not redistributed here. It is available at
<https://github.com/SALT-NLP/codenames> under CC BY-SA 4.0.

### Command-line interface

Installing the package exposes `codenames-experiment`:

| Subcommand | Purpose |
|---|---|
| `run` | Run the full experiment for one model |
| `compare` | Run the reference path against an accelerated path and report deltas |
| `validate` | Compare a small run against an existing run directory |
| `preflight` | Pre-flight diagnostic for the random-init models |
| `sanity` | Re-run the SC checks on already-extracted results |
| `doctor` | Verify installed dependencies match the pinned set |
| `visualize` | Render per-board heatmaps and 2D projections (`[viz]` extra) |
| `aggregate` | Build the cross-model tables and figure set (`[viz]` extra) |
| `lens-extract` | Dump per-layer hidden states at the generating position (GPU) |
| `lens-tune` | Train tuned-lens translators on generic text (GPU, `[lens]` extra) |
| `lens-apply` | Score board candidates through the raw/tuned lens (offline) |
| `lens-analyze` | Pre-registered lens trajectory analysis + overlay figures |

A run checkpoints incrementally so a dead Colab runtime can be continued:

```bash
codenames-experiment run \
    --model mistral \
    --dataset /content/drive/MyDrive/TCC/clue_generation.csv \
    --output-dir /content/drive/MyDrive/TCC/mistral_outputs \
    --full --resume
```

With `--resume`, the run skips boards already committed to the checkpoint
directory and continues from the last shard; the result is identical to an
uninterrupted run, because board sampling, shuffle seeds, and the vector
subsample are all re-derived deterministically from the contract. Without it,
stale checkpoints are wiped before a fresh run.

`--reuse-canonical` lets a later, larger run reuse the canonical
(`permutation_id=0`) records an earlier, smaller run computed for the boards
they share, skipping those forward passes without changing the output.

Three opt-in acceleration flags trade exact numerical agreement with the
reference path for speed; each is quantified against the reference path by
`compare` before use:

| Flag | What it does |
|---|---|
| `--vectorize-anisotropy` | Replace the O(n²) candidate-pair loop with a single matrix product |
| `--flash-attn` | Load Mistral or Qwen with FlashAttention 2 (requires `flash_attn`) |
| `--batch-size N` | Run N boards through one padded forward pass (default 1 = reference) |

### Lens study (post-thesis extension)

The lens pipeline (`docs/specs/lens_spec.md`) reads the model's own next-token channel
per layer at the generating position, restricted to the board candidates, to
adjudicate whether the thesis's cosine–generation gap is metric blindness or
representational decay. It is scoped to the causal decoders (`mistral`,
`qwen`, `qwen_random`). The GPU stages run from `notebooks/08_lens.ipynb` on
Colab (extraction ≈ 1.5–3 h per model-condition on an A100, tuned-lens
training ≈ 1–2 h per model); `lens-apply` and `lens-analyze` then run locally
on the downloaded dump:

```
 COLAB A100 (notebooks/08_lens.ipynb)                LOCAL (CLI)
 ┌──────────────────────────────────┐             ┌───────────────────────────────┐
 │ lens-extract: 1 forward/board    │   Drive     │ lens-apply: raw + tuned       │
 │  ─► hidden dump [N, layers+1, d] │  ────────►  │  scoring ─► scores parquet    │
 │  ─► readout weights (LN + head)  │  download   │ lens-analyze: curves, rules,  │
 │ lens-tune: per-layer translators │             │  controls, gate, figures      │
 └──────────────────────────────────┘             └───────────────────────────────┘
```

```bash
codenames-experiment lens-apply   --model mistral --dataset data/clue_generation.csv \
    --output-dir output/mistral_outputs --full --conditions no_social
codenames-experiment lens-analyze --output-root output \
    --models mistral,qwen --random-model random_qwen --condition no_social
```

`lens-analyze` writes curves, the pre-registered decision table, controls,
and per-model overlay figures under `output/lens_analysis/` and
`visualization/lens/`. Interpret nothing before the calibration gate and the
random-init null pass (see the notebook's closing notes).

### Aggregation and figures

After the per-model runs have written their `<prefix>_outputs/` folders,
`aggregate` builds the cross-model metric tables (MRR, Hit@*k*, top-1
accuracy, margins, paired social-preamble effects with bootstrap CIs,
generation–geometry concordance, and the UMAP trustworthiness sweep) and
renders the figures. `visualize` renders the per-board heatmaps and 2D
projections. Both require the optional plotting stack: `pip install -e ".[viz]"`.

## Mapping the thesis to the code

| Thesis section | Implemented in |
|---|---|
| Ch. 4 — The Codenames Duet task and per-turn unit | `data.py` |
| §5.3 — Turn sampling and prompt construction | `data.py`, `prompts.py` |
| §5.4 — Hidden state extraction and pooling | `extraction.py`, `spans.py` |
| §5.5 — Geometric metrics (margin, adjusted margin) | `extraction.py`, `sanity.py` |
| §5.6.1 — Cosine-rank metrics | `extraction.py`, `sanity.py` |
| §5.6.2 — Positional confound diagnostics | `extraction.py`, `sanity.py` |
| §5.6.3 — Between-condition diagnostics | `loop.py` |
| §5.6.4 — Projection trustworthiness | `analysis/trust.py`, `viz/` |
| §5.7 — Generation–geometry concordance | `generation.py`, `sanity.py` |
| §5.2 — The model suite | `models/` |

## Environment

Dependencies are pinned in `pyproject.toml` to the versions used for the
reported runs (`torch==2.5.1`, `transformers==4.48.0`, `numpy==2.0.2`, and the
rest of the scientific stack), chosen to coexist with Colab's environment. The
plotting stack is an optional `[viz]` extra so the extraction path stays light.
The reported runs used Python 3.10 on a single NVIDIA A100.

```bash
pip install -e .          # extraction + analysis tables
pip install -e ".[viz]"   # plus figures and projections
```

## License

MIT, see `LICENSE`. The CULTURAL CODES dataset \[Shaikh et al., 2023\] is not
redistributed here and is subject to its own licensing terms.
