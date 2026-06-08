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

from . import checkpoint
from .contract import ACCEL_REFERENCE, Acceleration, Contract
from .data import GIVER_COLS, extract_giver_features
from .extraction import run_instance, run_instance_batched
from .persistence import (
    save_error_log,
    save_general_csv,
    save_generation_csv,
    save_vector_subsample,
)
from .prompts import build_prompt


def _load_condition_outputs(base_dir: str, prefix: str, mode_name: str) -> Dict:
    """Reload a completed condition's final outputs from disk into a results dict.

    Used on resume when the manifest reports a condition is already complete:
    its checkpoints are gone, but the final CSV/parquet files remain and the
    downstream sanity checks expect the same in-memory shape ``run_extraction``
    would have returned.
    """
    gpath = os.path.join(base_dir, f"{prefix}_general_{mode_name}.csv")
    mpath = os.path.join(base_dir, f"{prefix}_metrics_{mode_name}.parquet")
    genpath = os.path.join(base_dir, f"{prefix}_generation_{mode_name}.csv")
    epath = os.path.join(base_dir, f"{prefix}_errors_{mode_name}.csv")
    return {
        "general_df": pd.read_csv(gpath) if os.path.exists(gpath) else pd.DataFrame(),
        "metrics_df": pd.read_parquet(mpath) if os.path.exists(mpath) else pd.DataFrame(),
        "generation_df": pd.read_csv(genpath) if os.path.exists(genpath) else pd.DataFrame(),
        "error_log": (
            pd.read_csv(epath).to_dict("records") if os.path.exists(epath) else []
        ),
    }


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
    resume: bool = False,
) -> Dict[str, Dict]:
    """Run the full extraction for both conditions, saving outputs to ``base_dir``.

    When ``resume`` is True, an interrupted run in ``base_dir`` is continued:
    already-committed boards (per the per-condition manifest) are skipped and a
    fully-completed condition's outputs are loaded from disk. ``shuffle_seeds``
    and the vector subsample are re-derived identically from the contract, and
    skipped boards keep their original ``board_idx``, so a resumed run is
    byte-identical to an uninterrupted one. With ``resume=False`` (default) any
    stale checkpoints/manifest in ``base_dir`` are wiped before the run.

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

        # --- Resume reconciliation (single source of truth = the manifest) ---
        if resume:
            boards_done_resume, ckpt_idx, condition_complete = checkpoint.reconcile(
                base_dir, prefix, mode_name, expected_n_boards=len(df_sample))
        else:
            # Fresh run: wipe any stale checkpoints/manifest so a previous
            # aborted run cannot contaminate this one's end-of-condition concat.
            checkpoint.remove_ckpts(base_dir, prefix, mode_name)
            checkpoint.remove_manifest(base_dir, prefix, mode_name)
            boards_done_resume, ckpt_idx, condition_complete = 0, 0, False

        if condition_complete:
            print(f"  [resume] condition '{mode_name}' already complete; "
                  f"loading existing outputs.")
            results[mode_name] = _load_condition_outputs(base_dir, prefix, mode_name)
            continue

        if resume and boards_done_resume > 0:
            print(f"  [resume] skipping first {boards_done_resume} "
                  f"already-committed boards; continuing at checkpoint {ckpt_idx}.")

        # These lists are flush-and-clear *buffers*: the loop appends to them
        # and ``_flush_checkpoint`` periodically writes their contents to atomic
        # checkpoint files and empties them, so each stream survives a
        # mid-condition crash. At end of condition the full ordered lists are
        # reconstructed from the checkpoints (see assembly block below).
        general_records = []
        metrics_buffer = []
        vector_records_all = []
        generation_records = []
        error_log = []

        # --- Checkpoint tracking: one monotonic index per flush boundary,
        # shared across all streams. ``ckpt_idx`` is seeded from the manifest
        # on resume so new checkpoints continue the existing sequence. ---
        boards_in_shard = 0

        def _flush_checkpoint(boards_done_count):
            """Flush every non-empty stream buffer to an atomic checkpoint.

            Writes all stream checkpoints at the current index, bumps the shared
            index, then (data-first, manifest-second) records the committed
            state in the manifest. ``boards_done_count`` is the absolute number
            of boards processed so far — the contiguous prefix a resume skips.
            """
            nonlocal ckpt_idx, metrics_buffer, general_records
            nonlocal generation_records, vector_records_all, error_log
            wrote = False
            if metrics_buffer:
                checkpoint.write_metrics_shard(
                    pd.DataFrame(metrics_buffer), base_dir, prefix, mode_name, ckpt_idx)
                metrics_buffer = []
                wrote = True
            if general_records:
                checkpoint.write_records(
                    general_records, base_dir, prefix, "general", mode_name, ckpt_idx)
                general_records = []
                wrote = True
            if generation_records:
                checkpoint.write_records(
                    generation_records, base_dir, prefix, "generation", mode_name, ckpt_idx)
                generation_records = []
                wrote = True
            if vector_records_all:
                checkpoint.write_records(
                    vector_records_all, base_dir, prefix, "vectors", mode_name, ckpt_idx)
                vector_records_all = []
                wrote = True
            if error_log:
                checkpoint.write_records(
                    error_log, base_dir, prefix, "errors", mode_name, ckpt_idx)
                error_log = []
                wrote = True
            if wrote:
                ckpt_idx += 1
                checkpoint.write_manifest(
                    base_dir, prefix, mode_name,
                    n_boards=len(df_sample), boards_done=boards_done_count,
                    ckpt_committed=ckpt_idx, complete=False)
                gc.collect()

        if acceleration.batch_size > 1:
            # --- Batched code path (acceleration.batch_size > 1) ---
            # Group boards into chunks of `batch_size`. For each chunk:
            #   1. Canonical (permutation_id=0) for all boards in the chunk → 1 forward pass
            #   2. Generation for canonical boards (one at a time; can't easily batch)
            #   3. For each shuffle k: shuffle-k for all boards in the chunk → 1 forward pass
            # Per-board error handling: if the whole batch raises, log all
            # boards in the batch with the same error and continue.
            n_boards = len(df_sample)
            rows_list = [row for _, row in df_sample.iterrows()]
            for chunk_start in tqdm(
                range(0, n_boards, acceleration.batch_size),
                desc=f"{mode_name} (batched)",
            ):
                # Resume: skip whole chunks already committed. ``boards_done_resume``
                # always lands on a chunk boundary (flushes happen at chunk ends),
                # so this skips exactly the committed prefix.
                if chunk_start < boards_done_resume:
                    continue
                chunk_end = min(chunk_start + acceleration.batch_size, n_boards)
                chunk_rows = rows_list[chunk_start:chunk_end]
                chunk_indices = list(range(chunk_start, chunk_end))
                chunk_canonical = [list(r["candidates"]) for r in chunk_rows]
                chunk_row_ids = [int(r["row_id"]) for r in chunk_rows]
                chunk_save_vecs = [rid in subsample_ids for rid in chunk_row_ids]

                # 1. Canonical batch
                try:
                    batch_results = run_instance_batched(
                        rows=chunk_rows,
                        giver_cols=GIVER_COLS,
                        use_social_context=mode_flag,
                        candidates_orders=chunk_canonical,
                        permutation_ids=[0] * len(chunk_rows),
                        save_vectors_flags=chunk_save_vecs,
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
                    canonical_general_per_board = []
                    for board_idx_in_chunk, board_result in enumerate(batch_results):
                        if board_result is None:
                            rid = chunk_row_ids[board_idx_in_chunk]
                            error_log.append({
                                "row_id": rid,
                                "error": "excluded from batch (pre-forward check failed)",
                                "permutation_id": 0,
                            })
                            canonical_general_per_board.append(None)
                            continue
                        g, m, v = board_result
                        general_records.append(g)
                        metrics_buffer.extend(m)
                        if v is not None:
                            vector_records_all.extend(v)
                        canonical_general_per_board.append(g)
                except Exception as e:
                    print(f"  ERROR batch (canonical) row_ids={chunk_row_ids}: {e}")
                    for rid in chunk_row_ids:
                        error_log.append({"row_id": rid, "error": str(e), "permutation_id": 0})
                    canonical_general_per_board = [None] * len(chunk_rows)

                # 2. Generation (one at a time, per board)
                if has_generation and generation_fn is not None:
                    for board_idx_in_chunk, row in enumerate(chunk_rows):
                        g = canonical_general_per_board[board_idx_in_chunk]
                        if g is None:
                            continue
                        rid = chunk_row_ids[board_idx_in_chunk]
                        try:
                            prompt_for_gen, _ = build_prompt(
                                hint=str(row["output"]),
                                candidates=chunk_canonical[board_idx_in_chunk],
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
                                candidates=chunk_canonical[board_idx_in_chunk],
                                max_new_tokens=generation_max_tokens,
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                            )
                            gen_record = {
                                "row_id"                  : rid,
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
                            error_log.append({"row_id": rid, "error": f"generation: {e}", "permutation_id": 0})
                            print(f"  ERROR generation row_id={rid}: {e}")

                # 3. Shuffles
                for k in range(n_shuffles):
                    shuffled_orders = []
                    for board_idx_local, board_idx_global in enumerate(chunk_indices):
                        perm_rng = np.random.RandomState(int(shuffle_seeds[board_idx_global, k]))
                        shuffled = list(chunk_canonical[board_idx_local])
                        perm_rng.shuffle(shuffled)
                        shuffled_orders.append(shuffled)

                    try:
                        batch_results = run_instance_batched(
                            rows=chunk_rows,
                            giver_cols=GIVER_COLS,
                            use_social_context=mode_flag,
                            candidates_orders=shuffled_orders,
                            permutation_ids=[k + 1] * len(chunk_rows),
                            save_vectors_flags=[False] * len(chunk_rows),
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
                        for board_idx_in_chunk, board_result in enumerate(batch_results):
                            if board_result is None:
                                rid = chunk_row_ids[board_idx_in_chunk]
                                error_log.append({
                                    "row_id": rid,
                                    "error": "excluded from batch (pre-forward check failed)",
                                    "permutation_id": k + 1,
                                })
                                continue
                            g_shuf, m_shuf, _ = board_result
                            general_records.append(g_shuf)
                            metrics_buffer.extend(m_shuf)
                    except Exception as e:
                        print(f"  ERROR batch (perm={k+1}) row_ids={chunk_row_ids}: {e}")
                        for rid in chunk_row_ids:
                            error_log.append({"row_id": rid, "error": str(e), "permutation_id": k + 1})

                # Shard flush check — increment per board in the chunk.
                boards_in_shard += len(chunk_rows)
                if boards_in_shard >= shard_boards and metrics_buffer:
                    _flush_checkpoint(chunk_end)
                    boards_in_shard = 0

        else:
            # --- Reference per-board code path (acceleration.batch_size == 1) ---
            for board_idx, (_, row) in enumerate(
                tqdm(df_sample.iterrows(), total=len(df_sample), desc=mode_name)
            ):
                # Resume: skip the already-committed contiguous prefix. The
                # original board_idx is preserved so shuffle_seeds[board_idx]
                # stays bit-identical for the boards we do process.
                if board_idx < boards_done_resume:
                    continue
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
                    _flush_checkpoint(board_idx + 1)
                    boards_in_shard = 0

        # --- Final flush of any remaining buffers (all five streams) ---
        _flush_checkpoint(len(df_sample))

        # ------------------------------------------------------------------
        # Concatenate metrics shards into a single parquet file
        # ------------------------------------------------------------------
        # Read the parquet shards back in index (= board) order and concat,
        # exactly as the pre-checkpoint code did with its in-run shard list.
        metrics_path = os.path.join(base_dir, f"{prefix}_metrics_{mode_name}.parquet")
        metric_frames = checkpoint.load_metrics_frames(base_dir, prefix, mode_name)
        if metric_frames:
            metrics_df = pd.concat(metric_frames, ignore_index=True)
            metrics_df.to_parquet(metrics_path, index=False)
            del metric_frames
            gc.collect()
            metrics_mb = os.path.getsize(metrics_path) / 1e6
        else:
            metrics_df = pd.DataFrame()
            metrics_mb = 0.0

        # ------------------------------------------------------------------
        # Save Stream B: Vector subsample
        # ------------------------------------------------------------------
        # Reconstruct the full ordered vector-record list from checkpoints and
        # hand it to the unchanged persistence helper (byte-identical output).
        vector_records_all = checkpoint.load_records(base_dir, prefix, "vectors", mode_name)
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
        # Reconstruct the full ordered record lists from checkpoints, then
        # build the DataFrame and save exactly as before — building one
        # DataFrame from the complete dict list reproduces the original dtype
        # inference, so the CSV is byte-identical.
        general_records = checkpoint.load_records(base_dir, prefix, "general", mode_name)
        general_df = pd.DataFrame(general_records)
        save_general_csv(general_df, base_dir, prefix, mode_name)

        generation_records = checkpoint.load_records(base_dir, prefix, "generation", mode_name)
        generation_df = pd.DataFrame(generation_records)
        if has_generation and len(generation_df) > 0:
            save_generation_csv(generation_df, base_dir, prefix, mode_name)

        # Persist error log per mode so post-hoc debugging doesn't depend on
        # the in-memory results dict surviving the session. Reconstructed from
        # checkpoints so resumed runs carry the full error history.
        error_log = checkpoint.load_records(base_dir, prefix, "errors", mode_name)
        save_error_log(error_log, base_dir, prefix, mode_name)

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

        # All final outputs for this condition are on disk. Mark the manifest
        # complete (so a resume after a crash in the *next* condition knows to
        # load these rather than recompute) and drop the now-redundant
        # checkpoints. The manifest itself is removed only once the whole run
        # finishes, below.
        checkpoint.write_manifest(
            base_dir, prefix, mode_name,
            n_boards=len(df_sample), boards_done=len(df_sample),
            ckpt_committed=ckpt_idx, complete=True)
        checkpoint.remove_ckpts(base_dir, prefix, mode_name)

        del vector_records_all, general_records, generation_records
        vector_records_all = []  # reset for next condition
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # The whole run is complete: remove the manifests so the output directory
    # is byte-identical to a non-resumable run (no checkpoint/manifest debris).
    for mode_name in ("no_social", "with_social"):
        checkpoint.remove_manifest(base_dir, prefix, mode_name)

    print(f"\nBoth conditions complete. Outputs in: {base_dir}")
    return results
