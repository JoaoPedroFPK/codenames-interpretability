"""File I/O for extraction outputs.

Each helper produces files at the exact path the original notebooks produce,
with the exact filename format. Save paths follow:

- ``{base_dir}/{prefix}_general_{mode}.csv``
- ``{base_dir}/{prefix}_metrics_{mode}.parquet``
- ``{base_dir}/{prefix}_generation_{mode}.csv``
- ``{base_dir}/{prefix}_vectors_subsample_index_{mode}.csv``
- ``{base_dir}/{prefix}_vectors_subsample_{mode}_f16.npz``
- ``{base_dir}/{prefix}_errors_{mode}.csv``

Aggregate SC outputs (SC5 layer margins, SC6 position confound, SC7 shuffle
decomposition) are saved by the SC functions themselves in :mod:`sanity`.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contract import Contract


def save_general_csv(
    general_df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    mode_name: str,
) -> str:
    path = os.path.join(base_dir, f"{prefix}_general_{mode_name}.csv")
    general_df.to_csv(path, index=False)
    return path


def save_metrics_parquet(
    metrics_df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    mode_name: str,
) -> str:
    path = os.path.join(base_dir, f"{prefix}_metrics_{mode_name}.parquet")
    metrics_df.to_parquet(path, index=False)
    return path


def save_generation_csv(
    generation_df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    mode_name: str,
) -> str:
    path = os.path.join(base_dir, f"{prefix}_generation_{mode_name}.csv")
    generation_df.to_csv(path, index=False)
    return path


def save_vector_subsample(
    vector_records: List[Dict],
    base_dir: str,
    prefix: str,
    mode_name: str,
    hidden_dim: int,
) -> Tuple[str, str]:
    """Save vector subsample as a CSV index + a float16 NPZ matrix.

    Mirrors the save block at the end of Cell 9 of every reference notebook.
    Records with invalid vectors (None / wrong shape) appear in the index
    with ``vector_valid=False`` and a zero row in the matrix.
    """
    n_vec_records = len(vector_records)

    vec_index_rows = []
    vec_arrays = []
    for i, rec in enumerate(vector_records):
        vec = rec["vector"]
        valid = (
            vec is not None
            and isinstance(vec, np.ndarray)
            and vec.shape == (hidden_dim,)
        )
        vec_arrays.append(vec if valid else None)
        vec_index_rows.append({
            "record_idx"        : i,
            "row_id"            : rec["row_id"],
            "layer"             : rec["layer"],
            "word"              : rec["word"],
            "word_type"         : rec["word_type"],
            "token_count"       : rec["token_count"],
            "pooling_method"    : rec["pooling_method"],
            "use_social_context": rec["use_social_context"],
            "vector_valid"      : valid,
        })
    vec_index_df = pd.DataFrame(vec_index_rows)

    valid_mask = np.array([v is not None for v in vec_arrays])
    valid_idx = np.where(valid_mask)[0]
    matrix = np.zeros((n_vec_records, hidden_dim), dtype=np.float16)
    if len(valid_idx):
        matrix[valid_idx] = np.stack([vec_arrays[i] for i in valid_idx])

    index_path = os.path.join(
        base_dir, f"{prefix}_vectors_subsample_index_{mode_name}.csv"
    )
    vec_index_df.to_csv(index_path, index=False)

    matrix_path = os.path.join(
        base_dir, f"{prefix}_vectors_subsample_{mode_name}_f16.npz"
    )
    np.savez_compressed(matrix_path, vectors=matrix)

    # NPZ integrity check (load and verify shape) — matches the notebook.
    _v = np.load(matrix_path)
    _shape = _v["vectors"].shape
    _v.close()
    vec_mb = os.path.getsize(matrix_path) / 1e6
    print(f"  Subsample NPZ verified: shape={_shape}, {vec_mb:.1f} MB")

    return index_path, matrix_path


def save_error_log(
    error_log: List[Dict],
    base_dir: str,
    prefix: str,
    mode_name: str,
) -> Optional[str]:
    if not error_log:
        return None
    path = os.path.join(base_dir, f"{prefix}_errors_{mode_name}.csv")
    pd.DataFrame(error_log).to_csv(path, index=False)
    print(f"Saved error log: {path}  ({len(error_log)} errors)")
    return path


def print_output_summary(
    *,
    base_dir: str,
    prefix: str,
    contract: Contract,
    has_generation: bool,
    pooling_methods,
) -> None:
    """Print the "Save Outputs Summary" block from the final cell of each notebook."""
    print("\n" + "=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    print(f"Directory: {base_dir}\n")

    for mode_name in ["no_social", "with_social"]:
        print(f"  [{mode_name}]")
        candidate_files = [
            f"{prefix}_general_{mode_name}.csv",
            f"{prefix}_metrics_{mode_name}.parquet",
            f"{prefix}_vectors_subsample_index_{mode_name}.csv",
            f"{prefix}_vectors_subsample_{mode_name}_f16.npz",
        ]
        if has_generation:
            candidate_files.append(f"{prefix}_generation_{mode_name}.csv")
        for suffix in candidate_files:
            fpath = os.path.join(base_dir, suffix)
            if os.path.exists(fpath):
                print(f"    {suffix}  ({os.path.getsize(fpath)/1e6:.1f} MB)")
            else:
                print(f"    {suffix}  [NOT FOUND]")

    print("\n  Aggregate files:")
    print(f"    {prefix}_position_confound_by_layer.csv")
    if contract.n_shuffles > 0:
        print(f"    {prefix}_shuffle_decomposition_by_layer.csv")
    for pm in pooling_methods:
        for mn in ["no_social", "with_social"]:
            print(f"    {prefix}_layer_margins_{pm}_{mn}.csv")

    print("\n" + "=" * 60)
    print("LOADING PATTERN FOR DOWNSTREAM NOTEBOOKS")
    print("=" * 60)
    print(f'''
# --- Stream A: Metrics (all boards, all permutations, no vectors) ---
metrics = pd.read_parquet(".../{prefix}_metrics_no_social.parquet")
metrics_canonical = metrics[metrics["permutation_id"] == 0]

# --- Stream B: Vector subsample (canonical only) ---
index  = pd.read_csv(".../{prefix}_vectors_subsample_index_no_social.csv")
data   = np.load(".../{prefix}_vectors_subsample_no_social_f16.npz")
matrix = data["vectors"]   # shape [N, {{HIDDEN_DIM}}], dtype float16
vec    = matrix[i].astype(np.float32)

# Filter by pooling method:
idx_mean = index[index["pooling_method"] == "mean"]
idx_maxn = index[index["pooling_method"] == "max_norm"]

# --- Generation results (causal-only) ---
gen = pd.read_csv(".../{prefix}_generation_no_social.csv")

# --- Shuffle analysis ---
shuffles = pd.read_csv(".../{prefix}_shuffle_decomposition_by_layer.csv")
''')
