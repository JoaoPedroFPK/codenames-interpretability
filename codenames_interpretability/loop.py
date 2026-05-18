"""Main extraction loop.

Verbatim from Cell 9 of every reference notebook with these adaptations:

- ``MODEL_PREFIX``, ``BASE_DIR``, ``SAMPLE_SIZE``, etc. come from arguments
  (via the ``Contract`` and the per-model metadata).
- The ``HAS_GENERATION`` flag becomes a keyword argument resolved by the caller
  (the notebook or the CLI) based on whether ``generation_fn`` was passed in.
- The generation call site uses the passed ``generation_fn`` instead of a
  globally-named ``generate_response``.
- All prints, all tqdm bars, all sharding logic remain.

Every print produced by the loop matches the original notebook output line
for line. The shard flush is inline (no helper function), exactly as in the
notebooks.
"""

import gc
import os
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .contract import ACCEL_REFERENCE, Acceleration, Contract
from .data import GIVER_COLS, extract_giver_features
from .extraction import run_instance
from .persistence import (
    save_general_csv,
    save_generation_csv,
    save_vector_subsample,
)
from .prompts import build_prompt


def run_extraction(
    *,
    model,
    tokenizer,
    df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    contract: Contract,
    chat_template_strategy: str,
    forward_hidden_states_mode: str,
    use_truncation: bool,
    num_layers: int,
    hidden_dim: int,
    device: Optional[str] = None,
    has_generation: bool = False,
    generation_fn: Optional[Callable] = None,
    acceleration: Acceleration = ACCEL_REFERENCE,
) -> Dict[str, Dict]:
    """Run the full extraction for both conditions, saving outputs to ``base_dir``.

    Returns a dict keyed by condition name (``no_social`` / ``with_social``)
    with sub-dicts containing ``general_df``, ``metrics_df``, ``generation_df``,
    ``error_log``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_size           = contract.sample_size
    pooling_methods       = contract.pooling_methods
    n_shuffles            = contract.n_shuffles
    generation_max_tokens = contract.generation_max_tokens
    vector_subsample_size = contract.vector_subsample_size
    shard_boards          = contract.shard_boards
    random_seed           = contract.random_seed
    max_seq_len           = contract.max_seq_len

    experiment_modes = [False, True]  # False = no_social, True = with_social

    # --- Sample ---
    df_sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=random_seed,
    ).copy().reset_index(drop=True)

    print(f"Sample size: {len(df_sample)} boards")
    print(f"Row IDs (first 10): {sorted(df_sample['row_id'].tolist())[:10]} ...")

    # --- Pre-select subsample board IDs ---
    subsample_size_eff = min(vector_subsample_size, len(df_sample))
    subsample_df = df_sample.sample(n=subsample_size_eff, random_state=random_seed)
    subsample_ids = set(subsample_df["row_id"].tolist())
    print(f"Vector subsample: {len(subsample_ids)} boards")

    # --- Pre-generate shuffle seeds for reproducibility ---
    rng_shuffles = np.random.RandomState(random_seed + 1000)
    shuffle_seeds = rng_shuffles.randint(0, 2**31, size=(len(df_sample), n_shuffles))

    os.makedirs(base_dir, exist_ok=True)

    results: Dict[str, Dict] = {}

    for mode_flag in experiment_modes:
        mode_name = "with_social" if mode_flag else "no_social"
        print(f"\n{'='*60}")
        print(f"Running condition: {mode_name}  "
              f"(canonical + {n_shuffles} shuffles per board)")
        print(f"{'='*60}")

        general_records = []
        metrics_buffer = []
        vector_records_all = []
        generation_records = []
        error_log = []

        # --- Stream A shard tracking (inline; no helper function) ---
        shard_idx = 0
        boards_in_shard = 0
        shard_paths = []

        for board_idx, (_, row) in enumerate(
            tqdm(df_sample.iterrows(), total=len(df_sample), desc=mode_name)
        ):
            row_id = int(row["row_id"])
            canonical_candidates = list(row["candidates"])  # alphabetical

            # ==============================================================
            # Canonical ordering (permutation_id = 0)
            # ==============================================================
            try:
                save_vecs = (row_id in subsample_ids)

                g, m, v = run_instance(
                    row=row,
                    giver_cols=GIVER_COLS,
                    use_social_context=mode_flag,
                    candidates_order=canonical_candidates,
                    permutation_id=0,
                    save_vectors=save_vecs,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    pooling_methods=pooling_methods,
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    chat_template_strategy=chat_template_strategy,
                    forward_hidden_states_mode=forward_hidden_states_mode,
                    use_truncation=use_truncation,
                    max_seq_len=max_seq_len,
                    acceleration=acceleration,
                )
                general_records.append(g)
                metrics_buffer.extend(m)
                if v is not None:
                    vector_records_all.extend(v)

                # --- Generation (canonical ordering only, causal-only) ---
                if has_generation and generation_fn is not None:
                    prompt_for_gen, _ = build_prompt(
                        hint=str(row["output"]),
                        candidates=canonical_candidates,
                        giver_features=(
                            extract_giver_features(row, GIVER_COLS)
                            if mode_flag else {}
                        ),
                        use_social_context=mode_flag,
                        tokenizer=tokenizer,
                        chat_template_strategy=chat_template_strategy,
                    )
                    gen_result = generation_fn(
                        prompt=prompt_for_gen,
                        candidates=canonical_candidates,
                        max_new_tokens=generation_max_tokens,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    gen_record = {
                        "row_id"                  : row_id,
                        "use_social_context"      : mode_flag,
                        "generated_text"          : gen_result["generated_text"],
                        "generated_word"          : gen_result["generated_word"],
                        "generated_in_candidates" : gen_result["generated_in_candidates"],
                        "generated_correct"       : (
                            gen_result["generated_word"] in set(row["targets"])
                            if gen_result["generated_word"] else False
                        ),
                    }
                    for pm in pooling_methods:
                        gen_record[f"concordance_{pm}"] = (
                            gen_result["generated_word"] == g[f"predicted_word_{pm}"]
                            if gen_result["generated_word"] else False
                        )
                    generation_records.append(gen_record)

            except Exception as e:
                error_log.append({"row_id": row_id, "error": str(e), "permutation_id": 0})
                print(f"  ERROR row_id={row_id} perm=0: {e}")

            # ==============================================================
            # Shuffle permutations (permutation_id = 1..K)
            # ==============================================================
            for k in range(n_shuffles):
                try:
                    perm_rng = np.random.RandomState(int(shuffle_seeds[board_idx, k]))
                    shuffled_candidates = list(canonical_candidates)
                    perm_rng.shuffle(shuffled_candidates)

                    g_shuf, m_shuf, _ = run_instance(
                        row=row,
                        giver_cols=GIVER_COLS,
                        use_social_context=mode_flag,
                        candidates_order=shuffled_candidates,
                        permutation_id=k + 1,
                        save_vectors=False,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        pooling_methods=pooling_methods,
                        num_layers=num_layers,
                        hidden_dim=hidden_dim,
                        chat_template_strategy=chat_template_strategy,
                        forward_hidden_states_mode=forward_hidden_states_mode,
                        use_truncation=use_truncation,
                        max_seq_len=max_seq_len,
                        acceleration=acceleration,
                    )
                    general_records.append(g_shuf)
                    metrics_buffer.extend(m_shuf)
                except Exception as e:
                    error_log.append({"row_id": row_id, "error": str(e), "permutation_id": k + 1})
                    print(f"  ERROR row_id={row_id} perm={k+1}: {e}")

            # --- Shard flush check (INLINE) ---
            boards_in_shard += 1
            if boards_in_shard >= shard_boards and metrics_buffer:
                shard_path = os.path.join(
                    base_dir,
                    f"{prefix}_metrics_{mode_name}_shard{shard_idx:03d}.parquet",
                )
                pd.DataFrame(metrics_buffer).to_parquet(shard_path, index=False)
                shard_paths.append(shard_path)
                metrics_buffer = []
                shard_idx += 1
                boards_in_shard = 0
                gc.collect()

        # --- Final flush of remaining buffer (INLINE) ---
        if metrics_buffer:
            shard_path = os.path.join(
                base_dir,
                f"{prefix}_metrics_{mode_name}_shard{shard_idx:03d}.parquet",
            )
            pd.DataFrame(metrics_buffer).to_parquet(shard_path, index=False)
            shard_paths.append(shard_path)
            metrics_buffer = []
            shard_idx += 1
            gc.collect()

        # ------------------------------------------------------------------
        # Concatenate shards into a single parquet file
        # ------------------------------------------------------------------
        metrics_path = os.path.join(base_dir, f"{prefix}_metrics_{mode_name}.parquet")
        if shard_paths:
            all_shards = [pd.read_parquet(p) for p in shard_paths]
            metrics_df = pd.concat(all_shards, ignore_index=True)
            metrics_df.to_parquet(metrics_path, index=False)
            for p in shard_paths:
                os.remove(p)
            del all_shards
            gc.collect()
            metrics_mb = os.path.getsize(metrics_path) / 1e6
        else:
            metrics_df = pd.DataFrame()
            metrics_mb = 0.0

        # ------------------------------------------------------------------
        # Save Stream B: Vector subsample
        # ------------------------------------------------------------------
        n_vec_records = len(vector_records_all)
        vec_mb = 0.0
        if n_vec_records > 0:
            _, vec_matrix_path = save_vector_subsample(
                vector_records=vector_records_all,
                base_dir=base_dir,
                prefix=prefix,
                mode_name=mode_name,
                hidden_dim=hidden_dim,
            )
            vec_mb = os.path.getsize(vec_matrix_path) / 1e6

        # ------------------------------------------------------------------
        # Save General + Generation
        # ------------------------------------------------------------------
        general_df = pd.DataFrame(general_records)
        save_general_csv(general_df, base_dir, prefix, mode_name)

        generation_df = pd.DataFrame(generation_records)
        if has_generation and len(generation_df) > 0:
            save_generation_csv(generation_df, base_dir, prefix, mode_name)

        print(f"\nCondition '{mode_name}' complete.")
        print(f"  Boards processed     : {len(df_sample)}")
        print(f"  Permutations/board   : 1 canonical + {n_shuffles} shuffles")
        print(f"  General records      : {len(general_df):,}")
        print(f"  Metrics rows         : {len(metrics_df):,}  ({metrics_mb:.1f} MB)")
        print(f"  Subsample vectors    : {n_vec_records:,} records  ({vec_mb:.1f} MB)")
        print(f"  Generation rows      : {len(generation_df)}")
        print(f"  Errors               : {len(error_log)}")

        results[mode_name] = {
            "general_df"    : general_df,
            "metrics_df"    : metrics_df,
            "generation_df" : generation_df,
            "error_log"     : error_log,
        }

        del vector_records_all, general_records, generation_records
        vector_records_all = []  # reset for next condition
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBoth conditions complete. Outputs in: {base_dir}")
    return results
