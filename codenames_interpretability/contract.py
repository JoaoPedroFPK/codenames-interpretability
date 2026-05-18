"""Frozen experimental parameters (Contract v1.0).

These values are fixed by the methodology chapter of the thesis. The default
instance `CONTRACT_V1` is the canonical contract used by all seven reference
notebooks; do not mutate it. The dataclass is frozen for that reason.

Model-specific values (MODEL_NAME, MODEL_PREFIX, BASE_DIR, MAX_SEQ_LEN where
relevant) are NOT in the contract — they live in the model loaders.
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
