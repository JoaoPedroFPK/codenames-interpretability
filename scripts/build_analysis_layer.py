#!/usr/bin/env python3
"""Stack the seven models' per-layer summaries into one tidy analysis layer.

Each model's ``output/<prefix>_outputs/`` folder already contains per-layer
*summary* CSVs — the notebooks computed these on the GPU box from the heavy
``*_metrics_*.parquet`` / ``*_vectors_*.npz`` files. This script does the cheap
final step: it stacks those tiny summaries across all seven models into a few
long-format tables keyed by ``(model, …, layer)``, ready to pivot/plot/reason
over.

Deliberately memory-light: it reads only the small summary CSVs (a few KB to a
few MB each), never the multi-hundred-MB ``.parquet`` or ``.npz`` files. Peak
RSS is a few MB regardless of how big the raw outputs are.

Three output tables are written to ``output/analysis/``:

* ``analysis_layer_margins.{csv,parquet}`` — the main table. One row per
  ``(model, condition, pooling_method, layer)``: margin / anisotropy stats.
* ``analysis_position_confound.csv`` — one row per ``(model, layer)``:
  position-confound correlation (rho) stats.
* ``analysis_shuffle_decomposition.csv`` — one row per ``(model, layer)``:
  within/between variance and the semantic ratio.

Every table carries ``model``, ``model_family`` (base architecture), and
``is_random`` (the random-init negative-control flag), plus ``layer_frac``
(layer / num_layers) so models of different depth line up on a common x-axis.

Usage::

    python scripts/build_analysis_layer.py
    python scripts/build_analysis_layer.py --output-dir output --strict
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# prefix -> base architecture. The random-init controls share their base's
# family so a family groups the trained model with its negative control.
MODEL_FAMILY = {
    "mistral": "mistral",
    "qwen": "qwen",
    "random_qwen": "qwen",
    "bert": "bert",
    "random_bert": "bert",
    "t5": "t5",
    "modernbert": "modernbert",
}
MODEL_PREFIXES = list(MODEL_FAMILY)
CONDITIONS = ["no_social", "with_social"]
POOLINGS = ["mean", "max_norm"]


def _eprint(*a, **k) -> None:
    print(*a, file=sys.stderr, **k)


def _add_keys(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prepend the model identity columns and a normalized layer fraction."""
    df = df.copy()
    df.insert(0, "model", prefix)
    df.insert(1, "model_family", MODEL_FAMILY[prefix])
    df.insert(2, "is_random", prefix.startswith("random_"))
    if "layer" in df.columns:
        max_layer = df["layer"].max()
        df["layer_frac"] = df["layer"] / max_layer if max_layer else 0.0
    return df


def _read_csv(path: str, missing: List[str], *, strict: bool) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        missing.append(os.path.relpath(path, REPO_ROOT))
        if strict:
            raise FileNotFoundError(path)
        return None
    return pd.read_csv(path)


def build_layer_margins(output_dir: str, missing: List[str], *, strict: bool) -> pd.DataFrame:
    """Stack every model's layer_margins_{pooling}_{condition}.csv."""
    frames: List[pd.DataFrame] = []
    for prefix in MODEL_PREFIXES:
        base = os.path.join(output_dir, f"{prefix}_outputs")
        for pooling in POOLINGS:
            for cond in CONDITIONS:
                path = os.path.join(base, f"{prefix}_layer_margins_{pooling}_{cond}.csv")
                df = _read_csv(path, missing, strict=strict)
                if df is None:
                    continue
                # The file already carries `condition` and `pooling_method`;
                # trust them but ensure they are present/consistent.
                if "pooling_method" not in df.columns:
                    df["pooling_method"] = pooling
                if "condition" not in df.columns:
                    df["condition"] = cond
                frames.append(_add_keys(df, prefix))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_simple(output_dir: str, suffix: str, missing: List[str], *, strict: bool) -> pd.DataFrame:
    """Stack a per-(model, layer) summary file named ``<prefix>_<suffix>.csv``."""
    frames: List[pd.DataFrame] = []
    for prefix in MODEL_PREFIXES:
        path = os.path.join(output_dir, f"{prefix}_outputs", f"{prefix}_{suffix}.csv")
        df = _read_csv(path, missing, strict=strict)
        if df is None:
            continue
        frames.append(_add_keys(df, prefix))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "output"),
                        help="Directory holding the <prefix>_outputs folders (default: ./output)")
    parser.add_argument("--analysis-dir", default=None,
                        help="Where to write the analysis tables (default: <output-dir>/analysis)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail on the first missing summary file instead of skipping it")
    args = parser.parse_args(argv)

    analysis_dir = args.analysis_dir or os.path.join(args.output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    missing: List[str] = []

    try:
        margins = build_layer_margins(args.output_dir, missing, strict=args.strict)
        confound = build_simple(args.output_dir, "position_confound_by_layer", missing, strict=args.strict)
        shuffle = build_simple(args.output_dir, "shuffle_decomposition_by_layer", missing, strict=args.strict)
    except FileNotFoundError as exc:
        _eprint(f"[strict] missing required summary file: {exc}")
        return 1

    outputs = []
    if not margins.empty:
        p_csv = os.path.join(analysis_dir, "analysis_layer_margins.csv")
        p_pq = os.path.join(analysis_dir, "analysis_layer_margins.parquet")
        margins.to_csv(p_csv, index=False)
        margins.to_parquet(p_pq, index=False)
        outputs += [(p_csv, len(margins)), (p_pq, len(margins))]
    if not confound.empty:
        p = os.path.join(analysis_dir, "analysis_position_confound.csv")
        confound.to_csv(p, index=False)
        outputs.append((p, len(confound)))
    if not shuffle.empty:
        p = os.path.join(analysis_dir, "analysis_shuffle_decomposition.csv")
        shuffle.to_csv(p, index=False)
        outputs.append((p, len(shuffle)))

    print("ANALYSIS LAYER")
    print("=" * 60)
    models_seen = sorted(set(margins.get("model", pd.Series(dtype=str)).unique()))
    print(f"Models present: {len(models_seen)} -> {', '.join(models_seen) or '(none)'}")
    for path, n in outputs:
        print(f"  wrote {os.path.relpath(path, REPO_ROOT)}  ({n} rows)")
    if missing:
        print(f"\nMissing summary files (skipped): {len(missing)}")
        for m in missing:
            print(f"  - {m}")
    if not outputs:
        _eprint("No analysis tables written — no summary files found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
