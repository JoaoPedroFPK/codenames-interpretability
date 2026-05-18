# Runtime model — Colab-from-GitHub workflow

This package is designed for one execution environment: a Google Colab GPU
container that pulls source code from a GitHub repository. It does not have
a local-dev workflow. This document explains how the pieces fit together.

---

## The constraint

Colab provides a Linux container with a GPU attached, but the container is
ephemeral: anything outside `/content/drive/` evaporates at runtime
shutdown. The user edits Python files on their laptop with a text editor
(no Python runtime locally), pushes to GitHub, then pulls from inside
Colab.

The package therefore needs to:

- Install cleanly via `pip install -e .` against Colab's existing pinned
  versions of torch, transformers, numpy, pandas, tqdm, scipy.
- Avoid pinning any dependency version that could trigger a reinstall.
- Be re-runnable cell by cell so the user can verify behavior at each step
  in a notebook.

---

## The intended workflow

For every experiment run, the user:

1. Edits package code locally with their text editor.
2. Commits and pushes to GitHub.
3. Opens the target notebook in Colab (notebooks/ are openable via Colab's
   "GitHub" file dialog).
4. Runs cell 1, which clones or `git pull`s the repo into `/content/`.
5. Runs cell 2, which `pip install -e`'s the package.
6. Runs cell 3, which enables `%autoreload 2` and mounts Drive.
7. Runs either Path A (one CLI cell) or Path B (cell-by-cell), reading
   intermediate output as it streams.

When the user wants to re-run with updated code, they push the new code,
re-run cell 1 (which `git pull`s the new commits), and the next cell that
imports from the package picks up the new code thanks to `autoreload`.

---

## Why the notebook boilerplate is duplicated

The first three cells of every notebook in `notebooks/` are byte-identical.
They are intentionally **not** factored out into a shared script:

- Each notebook must be runnable as a standalone artifact opened directly
  from GitHub by Colab.
- There is no shared "setup" file the notebooks all source — each notebook
  is self-contained from the user's perspective, so they can hand a
  collaborator a single `.ipynb` URL.

The trade-off is a small amount of duplication for the property that
nothing breaks across notebooks even if some are run in isolation.

---

## How outputs flow to Drive

Each notebook writes outputs to a per-model directory under
`/content/drive/MyDrive/TCC/`. Filenames are constructed from the model's
`prefix` (defined in its loader's metadata) and the condition name:

- `{prefix}_general_{mode}.csv` — per-board behavioral results
- `{prefix}_metrics_{mode}.parquet` — per-(board, layer, word, permutation)
  scalar metrics, sharded during the run, concatenated at the end
- `{prefix}_vectors_subsample_index_{mode}.csv` — index for the
  per-condition vector subsample
- `{prefix}_vectors_subsample_{mode}_f16.npz` — float16 vector matrix
  paired with the index above
- `{prefix}_generation_{mode}.csv` — causal-only generation output
- `{prefix}_errors_{mode}.csv` — error log (only written if a board failed)
- `{prefix}_layer_margins_{pm}_{mode}.csv` — SC5 aggregate
- `{prefix}_position_confound_by_layer.csv` — SC6 aggregate
- `{prefix}_shuffle_decomposition_by_layer.csv` — SC7 aggregate

The synthesis notebook that consumes these files to produce the figures in
the thesis Results chapter is maintained separately from this repository.

---

## Validating against existing runs

After any non-trivial change to the package, before trusting it with a new
run, run `notebooks/00_validation.ipynb`:

1. Set `MODEL` to a model whose N=2000 run is already in Drive.
2. Set `AGAINST_DIR` to that run's output directory.
3. Run the cells.

The validation notebook samples the first 50 boards under the contract
seed, runs the refactored pipeline to a temp directory, then diffs the
resulting `*_general_{mode}.csv` rows against the existing N=2000 outputs
filtered to the same 50 row_ids. It reports `Validation PASSED` or lists
the differing columns.

CONTEXT.md Section 6 enumerates the most common drift sources (random-state
consumption, hot-loop operation order, dictionary iteration order, fp16
conversion timing, tokenizer init args). When validation fails, that is
the place to look first.
