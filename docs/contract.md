# Experimental contract (CONTRACT_v1.0)

The values below are fixed for the duration of this study. They are codified
in `codenames_interpretability/contract.py` as a frozen dataclass
(`Contract`) with a single canonical instance (`CONTRACT_V1`). Do not mutate
the contract during a run; if a parameter needs to change, increment the
version (`v1.1`, `v2`, etc.) and document the change here.

The full rationale for each value is in Chapter 4 (Methodology) of the
thesis. This file is a quick reference; the thesis is the source of record.

---

## Fields

### `sample_size: int = 2000`

Number of Codenames Duet turns drawn from CULTURAL CODES per run, with
`random_state=random_seed`. Identical across all seven models, so every
model sees the same boards.

### `candidate_order: str = "fixed"`

Canonical (permutation_id=0) ordering of candidate words within a board:
alphabetical. Shuffles (permutation_id=1..N_SHUFFLES) draw random
permutations under per-board seeds derived from the master seed.

### `pooling_methods: Tuple[str, ...] = ("mean", "max_norm")`

Span-pooling procedures run in parallel for every span at every layer:

- **`mean`** — average of the span's token hidden states.
- **`max_norm`** — the single token in the span with the highest L2 norm.

Both methods produce a vector per (board, layer, word). Vectors are stored
as `np.float16` after pooling.

### `vector_subsample_size: int = 100`

Number of boards (canonical ordering only) whose raw layer vectors are
retained for the per-model `*_vectors_subsample_{mode}_f16.npz` matrix.
A subset is retained so that downstream analyses (e.g., visualisation of
the cosine geometry) have raw vectors without the full N=2000 cost in
storage.

### `n_shuffles: int = 2`

Number of random candidate-order permutations per board, in addition to the
canonical (permutation_id=0) ordering. Used in SC7 for the within-vs-between
word variance decomposition.

### `generation_max_tokens: int = 30`

Maximum new tokens for the causal-LM generation phase. Encoder-only models
ignore this field.

### `shard_boards: int = 200`

Stream A flush size: the per-condition metrics buffer is written to a
parquet shard every K boards, then concatenated at end of condition. Bounds
peak CPU memory during a run.

### `random_seed: int = 42`

Single seed governing: Python / numpy / torch RNG, dataset sampling,
shuffle seed generation, and (for random-init models) weight
initialisation. The seed is fixed once at the start of a run; it is not
re-applied between conditions.

### `max_seq_len: int = 512`

Tokenizer truncation length for encoder models with a hard sequence-length
limit (BERT, T5, BERT Random, ModernBERT). The three causal models in this
suite do not truncate. The field lives in the contract for transparency;
whether to apply truncation is a per-model decision captured in each model
loader's `use_truncation` metadata flag.

---

## Why a frozen contract

The bit-identity requirement (CONTEXT.md Section 6) says that the
refactored code must produce outputs interchangeable with the existing
N=2000 runs. Any drift in these parameters — even silently — would break
that contract. The frozen dataclass ensures that a parameter change is
visible at code-review time, not at output-diff time.
