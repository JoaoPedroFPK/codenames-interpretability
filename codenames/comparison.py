"""Comparison harness: reference path vs accelerated path.

Runs the same extraction on the same boards twice — once with the all-defaults
``Acceleration`` instance (reference path), once with a user-supplied
``Acceleration`` (fast path). Reports per-column max-absolute and
mean-absolute deltas plus the count of exact-equal cells.

This module exists so that "is the optimization safe?" becomes a numeric
question: the harness gives you a tolerance figure to put in the thesis
appendix, not a binary verdict.

The harness does NOT load the model twice. It loads once (with the fast
``Acceleration`` if that affects load-time choices) and runs the extraction
loop twice with different acceleration values per pass. For optimizations
that only affect load time (e.g., Flash Attention 2 via the loader's
``attn_implementation`` argument), this means the harness must be invoked
once per attn_implementation choice; see the ``compare`` CLI subcommand for
how this is dispatched.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .contract import ACCEL_REFERENCE, Acceleration, Contract
from .loop import run_extraction


def compare_runs(
    *,
    model_ref,
    tokenizer_ref,
    model_fast,
    tokenizer_fast,
    df: pd.DataFrame,
    prefix: str,
    contract: Contract,
    chat_template_strategy: str,
    forward_hidden_states_mode: str,
    use_truncation: bool,
    num_layers: int,
    hidden_dim: int,
    device: str,
    has_generation: bool = False,
    generation_fn: Optional[Callable] = None,
    fast_acceleration: Acceleration = ACCEL_REFERENCE,
    tmp_base_dir: str = "/tmp/cnames_compare",
) -> Dict[str, Any]:
    """Run the extraction twice and return a per-column delta report.

    Parameters
    ----------
    model_ref, tokenizer_ref
        Reference-path model and tokenizer (loaded with default
        ``attn_implementation`` for the model class).
    model_fast, tokenizer_fast
        Fast-path model and tokenizer. May be the same object as
        ``model_ref`` / ``tokenizer_ref`` when the acceleration flags only
        affect runtime behavior (vectorize_anisotropy, batch_size). When
        Flash Attention 2 is enabled, the model is reloaded with
        ``attn_implementation="flash_attention_2"`` and passed in as
        ``model_fast``.
    df
        Sample of boards to run on (e.g., 50 rows).
    fast_acceleration
        ``Acceleration`` instance controlling the fast path's runtime
        choices (vectorize_anisotropy, batch_size). Flash Attention 2 is
        controlled at model-load time by the caller, not here.

    Returns
    -------
    report
        Dict with ``general_diff``, ``metrics_diff`` keys. Each is a
        DataFrame with one row per numeric column reporting
        ``max_abs_delta``, ``mean_abs_delta``, ``n_exact_equal``, ``n_total``.
    """
    import os
    os.makedirs(tmp_base_dir, exist_ok=True)
    ref_dir = os.path.join(tmp_base_dir, "ref")
    fast_dir = os.path.join(tmp_base_dir, "fast")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(fast_dir, exist_ok=True)

    print("\n[1/2] Reference path (Acceleration defaults)")
    ref_results = run_extraction(
        model=model_ref,
        tokenizer=tokenizer_ref,
        df=df,
        base_dir=ref_dir,
        prefix=prefix,
        contract=contract,
        chat_template_strategy=chat_template_strategy,
        forward_hidden_states_mode=forward_hidden_states_mode,
        use_truncation=use_truncation,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        device=device,
        has_generation=has_generation,
        generation_fn=generation_fn,
        acceleration=ACCEL_REFERENCE,
    )

    print(f"\n[2/2] Fast path ({fast_acceleration})")
    fast_results = run_extraction(
        model=model_fast,
        tokenizer=tokenizer_fast,
        df=df,
        base_dir=fast_dir,
        prefix=prefix,
        contract=contract,
        chat_template_strategy=chat_template_strategy,
        forward_hidden_states_mode=forward_hidden_states_mode,
        use_truncation=use_truncation,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        device=device,
        has_generation=has_generation,
        generation_fn=generation_fn,
        acceleration=fast_acceleration,
    )

    general_report = _diff_general(ref_results, fast_results)
    metrics_report = _diff_metrics(ref_results, fast_results)

    _print_summary("general_df", general_report)
    _print_summary("metrics_df", metrics_report)

    return {
        "general_diff": general_report,
        "metrics_diff": metrics_report,
        "ref_results": ref_results,
        "fast_results": fast_results,
    }


def _diff_general(ref_results: Dict, fast_results: Dict) -> pd.DataFrame:
    """Per-column delta on general_df across both conditions."""
    rows = []
    for mode_name in ["no_social", "with_social"]:
        ref_df = ref_results[mode_name]["general_df"]
        fast_df = fast_results[mode_name]["general_df"]
        if len(ref_df) == 0 or len(fast_df) == 0:
            continue
        rows.extend(_diff_dataframes(ref_df, fast_df, ["row_id", "permutation_id"], mode_name))
    return pd.DataFrame(rows)


def _diff_metrics(ref_results: Dict, fast_results: Dict) -> pd.DataFrame:
    """Per-column delta on metrics_df across both conditions."""
    rows = []
    for mode_name in ["no_social", "with_social"]:
        ref_df = ref_results[mode_name]["metrics_df"]
        fast_df = fast_results[mode_name]["metrics_df"]
        if len(ref_df) == 0 or len(fast_df) == 0:
            continue
        rows.extend(_diff_dataframes(
            ref_df, fast_df, ["row_id", "permutation_id", "layer", "word"], mode_name,
        ))
    return pd.DataFrame(rows)


def _diff_dataframes(
    ref_df: pd.DataFrame,
    fast_df: pd.DataFrame,
    key_cols,
    mode_name: str,
) -> list:
    """Inner-join on key_cols, diff every numeric column."""
    rows = []
    keys_in_both = list(set(ref_df.columns) & set(fast_df.columns) & set(key_cols))
    numeric_cols = [
        c for c in ref_df.select_dtypes(include=[np.number]).columns
        if c in fast_df.columns and c not in key_cols
    ]
    if not keys_in_both or not numeric_cols:
        return rows

    merged = ref_df[keys_in_both + numeric_cols].merge(
        fast_df[keys_in_both + numeric_cols],
        on=keys_in_both,
        suffixes=("_ref", "_fast"),
    )
    if merged.empty:
        return rows

    for col in numeric_cols:
        r = merged[f"{col}_ref"]
        f = merged[f"{col}_fast"]
        # Treat (NaN, NaN) as exact equal; (NaN, x) where x is finite is a mismatch.
        both_nan = r.isna() & f.isna()
        only_one_nan = (r.isna() ^ f.isna())
        diff = (r - f).abs()
        # NaN handling: only_one_nan -> infinite delta (count as mismatch).
        max_diff = float(diff.max(skipna=True)) if diff.notna().any() else 0.0
        mean_diff = float(diff.mean(skipna=True)) if diff.notna().any() else 0.0
        n_exact = int((both_nan | (diff == 0.0)).sum())
        n_total = int(len(merged))
        n_only_one_nan = int(only_one_nan.sum())
        rows.append({
            "mode": mode_name,
            "column": col,
            "max_abs_delta": max_diff,
            "mean_abs_delta": mean_diff,
            "n_exact_equal": n_exact,
            "n_only_one_nan": n_only_one_nan,
            "n_total": n_total,
        })
    return rows


def _print_summary(label: str, report: pd.DataFrame) -> None:
    """Print a one-line-per-column summary of the diff."""
    print(f"\n=== {label} ===")
    if report.empty:
        print("  (no numeric columns to compare)")
        return
    print(f"  {'mode':<12} {'column':<35} {'max|Δ|':>12} {'mean|Δ|':>12} {'exact':>8} {'/total':>8} {'1-NaN':>7}")
    print(f"  {'-'*12} {'-'*35} {'-'*12} {'-'*12} {'-'*8} {'-'*8} {'-'*7}")
    for _, r in report.iterrows():
        print(
            f"  {r['mode']:<12} {r['column']:<35} "
            f"{r['max_abs_delta']:>12.3e} {r['mean_abs_delta']:>12.3e} "
            f"{int(r['n_exact_equal']):>8} {int(r['n_total']):>8} "
            f"{int(r['n_only_one_nan']):>7}"
        )

    worst = report.loc[report["max_abs_delta"].idxmax()]
    print(
        f"\n  Headline: worst column is {worst['column']} ({worst['mode']}) "
        f"with max|Δ|={worst['max_abs_delta']:.3e}"
    )
