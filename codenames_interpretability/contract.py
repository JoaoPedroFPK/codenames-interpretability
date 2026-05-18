"""Frozen experimental parameters (Contract v1.0) and acceleration flags.

These values are fixed by the methodology chapter of the thesis. The default
instance ``CONTRACT_V1`` is the canonical contract used by all seven reference
notebooks; do not mutate it. The dataclass is frozen for that reason.

Model-specific values (MODEL_NAME, MODEL_PREFIX, BASE_DIR, MAX_SEQ_LEN where
relevant) are NOT in the contract — they live in the model loaders.

``Acceleration`` is a separate, frozen dataclass that holds *implementation*
choices that trade bit-identity with the reference path for wall-clock
speedup. These are not part of the methodology and are documented in the
thesis appendix with the measured tolerance against the reference path.
``ACCEL_REFERENCE`` is the all-defaults instance — equivalent to the original
notebook code path.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Contract:
    """Frozen experimental parameters. See methodology chapter §4.8 for rationale.

    Attributes
    ----------
    sample_size
        Number of Codenames Duet turns drawn from CULTURAL CODES per run
        (with ``random_state=random_seed``). Methodology §4.3.
    candidate_order
        How the candidate list is ordered in the canonical (permutation_id=0)
        run. ``"fixed"`` means alphabetical within each board.
    pooling_methods
        Span-pooling procedures run in parallel for every span at every layer.
        Methodology §4.4.
    vector_subsample_size
        Number of boards (canonical ordering only) whose raw layer vectors are
        retained for the NPZ vector subsample.
    n_shuffles
        Number of random candidate-order permutations per board, in addition to
        the canonical (permutation_id=0) ordering. Methodology §4.6.2.
    generation_max_tokens
        Maximum new tokens for the causal-LM generation phase. Encoder-only
        models ignore this field.
    shard_boards
        Stream A flush size: the per-condition metrics buffer is written to a
        parquet shard every K boards, then concatenated at end of condition.
    random_seed
        Single seed governing: numpy/torch RNG, dataset sampling, shuffle
        seeds, and (for random-init models) weight initialisation.
    max_seq_len
        Tokenizer truncation length for models with a hard sequence-length
        limit (BERT, T5, ModernBERT, BERT Random). Causal models in this suite
        do not truncate. The field lives in the contract for transparency, but
        whether to apply truncation is decided by each model loader.
    """

    sample_size: int = 2000
    candidate_order: str = "fixed"
    pooling_methods: Tuple[str, ...] = ("mean", "max_norm")
    vector_subsample_size: int = 100
    n_shuffles: int = 2
    generation_max_tokens: int = 30
    shard_boards: int = 200
    random_seed: int = 42
    max_seq_len: int = 512


CONTRACT_V1 = Contract()


@dataclass(frozen=True)
class Acceleration:
    """Acceleration flags. None of these change methodology; each trades
    bit-identity with the reference path for wall-clock speedup. The
    ``compare`` CLI subcommand quantifies the per-cell delta a given
    combination introduces; document the measured tolerance in the thesis
    appendix.

    Attributes
    ----------
    vectorize_anisotropy
        Compute all-pairs candidate cosines via a single ``M @ M.T`` matrix
        product instead of the nested Python loop over pairs. Reorders ~300
        fp32 additions per cosine; expected drift on
        ``layer_mean_pairwise_cosine`` is ~1e-6.
    flash_attention_for_causal
        When the model loader supports it (Mistral, Qwen) and ``flash_attn``
        is importable, load with ``attn_implementation="flash_attention_2"``.
        Tiled-softmax attention; expected drift is ~1-ULP per layer in fp16,
        possibly accumulating across all 32 layers.
    batch_size
        Number of boards processed per forward pass. Default 1 (one board
        per pass — the reference path). Larger values batch boards through a
        single padded forward pass; the order of fp16 reductions over the
        sequence dimension changes when batch dim is added. Bucketing by
        tokenized length is applied automatically to bound the padding
        overhead.
    """

    vectorize_anisotropy: bool = False
    flash_attention_for_causal: bool = False
    batch_size: int = 1


ACCEL_REFERENCE = Acceleration()
