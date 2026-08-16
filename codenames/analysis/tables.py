"""Aggregated cross-model metric tables.

Builds the behavioral side of the unified analysis surface from the per-model
output folders (``output/<prefix>_outputs/``):

- ``analysis_model_summary.csv`` — one row per (model, condition, pooling):
  MRR, Hit@1/3/5, final-layer top-1 accuracy, raw margin (mean ± SEM).
- ``analysis_social_effects.csv`` — one row per (model, pooling, metric):
  paired per-board delta (with_social − no_social) with a bootstrap 95% CI.
- ``analysis_concordance.csv`` — one row per (model, condition), generation
  models only: generation accuracy and final-layer geometry concordance.
- ``analysis_concordance_by_layer.csv`` — one row per (model, condition,
  pooling, layer): agreement between the generated word and that layer's
  geometric top-1 candidate, plus the layer's top-1 target accuracy.

All behavioral aggregates use the canonical candidate ordering only
(``permutation_id == 0``), the ordering the generation phase saw. The shuffled
orderings exist for the SC6/SC7 confound analysis and stay confined to those
tables.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# prefix -> base architecture; random-init controls share their base's family.
MODEL_FAMILY: Dict[str, str] = {
    "mistral": "mistral",
    "qwen": "qwen",
    "random_qwen": "qwen",
    "bert": "bert",
    "random_bert": "bert",
    "t5": "t5",
    "modernbert": "modernbert",
    # ICLR 2027 additions: base variants (geometry only) + a third decoder.
    "mistral_base": "mistral",
    "qwen_base": "qwen",
    "llama": "llama",
}
MODEL_PREFIXES = list(MODEL_FAMILY)
GENERATION_PREFIXES = ["mistral", "qwen", "llama"]  # trained instruct decoders only
CONDITIONS = ["no_social", "with_social"]
POOLINGS = ["mean", "max_norm"]

# Per-board columns aggregated into the summary / social-effect tables. The
# general CSV stores them per pooling as ``{name}_{pooling}``; ``correct`` is
# the final-layer top-1 accuracy flag, renamed for the output tables.
_BEHAVIORAL = ["mrr", "hit_at_1", "hit_at_3", "hit_at_5", "correct", "raw_margin"]
_RENAME = {"correct": "accuracy_top1"}


def _metric_out_name(name: str) -> str:
    return _RENAME.get(name, name)


def _model_dir(output_dir: str, prefix: str) -> str:
    return os.path.join(output_dir, f"{prefix}_outputs")


def _add_identity(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.insert(0, "model", prefix)
    df.insert(1, "model_family", MODEL_FAMILY[prefix])
    df.insert(2, "is_random", prefix.startswith("random_"))
    return df


def load_general_canonical(
    output_dir: str, prefix: str, condition: str,
) -> Optional[pd.DataFrame]:
    """The canonical-ordering rows of ``{prefix}_general_{condition}.csv``."""
    path = os.path.join(_model_dir(output_dir, prefix), f"{prefix}_general_{condition}.csv")
    if not os.path.exists(path):
        return None
    cols = ["row_id", "permutation_id"] + [
        f"{m}_{pm}" for m in _BEHAVIORAL for pm in POOLINGS
    ]
    df = pd.read_csv(path, usecols=cols)
    return df[df["permutation_id"] == 0].reset_index(drop=True)


def _semantic_ratio_scalars(output_dir: str, prefix: str) -> Dict[str, float]:
    """Final-layer and across-layer-minimum semantic ratio for one model.

    The SC7 shuffle decomposition has no condition/pooling axis, so the same
    two scalars annotate every summary row of the model. The final-layer value
    matches the table's final-layer framing; the minimum is the conservative
    bound (the worst layer's identity-driven share).
    """
    path = os.path.join(_model_dir(output_dir, prefix),
                        f"{prefix}_shuffle_decomposition_by_layer.csv")
    if not os.path.exists(path):
        return {"semantic_ratio_final": float("nan"), "semantic_ratio_min": float("nan")}
    df = pd.read_csv(path, usecols=["layer", "semantic_ratio"])
    if df.empty:
        return {"semantic_ratio_final": float("nan"), "semantic_ratio_min": float("nan")}
    return {
        "semantic_ratio_final": float(df.loc[df["layer"].idxmax(), "semantic_ratio"]),
        "semantic_ratio_min": float(df["semantic_ratio"].min()),
    }


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def build_model_summary(
    output_dir: str,
    missing: Optional[List[str]] = None,
    *,
    n_boot: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    """One row per (model, condition, pooling) with mean, SEM and a 95% CI.

    The CI is a percentile bootstrap over boards (``_bootstrap_ci``), the same
    resampling used for the social-effect table, so every reported mean in the
    analysis surface carries an interval computed the same way. The boards are
    the units averaged over, hence the units resampled.
    """
    rows: List[Dict] = []
    for prefix in MODEL_PREFIXES:
        sem_ratio = _semantic_ratio_scalars(output_dir, prefix)
        for cond in CONDITIONS:
            df = load_general_canonical(output_dir, prefix, cond)
            if df is None:
                if missing is not None:
                    missing.append(f"{prefix}_general_{cond}.csv")
                continue
            for pm in POOLINGS:
                rec: Dict = {"condition": cond, "pooling": pm, "n_boards": len(df)}
                for m in _BEHAVIORAL:
                    vals = df[f"{m}_{pm}"].astype(float).to_numpy()
                    vals = vals[~np.isnan(vals)]
                    out = _metric_out_name(m)
                    rec[out] = float(vals.mean()) if vals.size else float("nan")
                    rec[f"{out}_sem"] = (
                        float(vals.std(ddof=1) / np.sqrt(len(vals)))
                        if vals.size > 1 else float("nan")
                    )
                    if vals.size:
                        lo, hi = _bootstrap_ci(vals, n_boot=n_boot, seed=seed)
                    else:
                        lo, hi = float("nan"), float("nan")
                    rec[f"{out}_ci_low"] = lo
                    rec[f"{out}_ci_high"] = hi
                rec.update(sem_ratio)
                rows.append(_add_identity(pd.DataFrame([rec]), prefix).iloc[0].to_dict())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-model Kruskal–Wallis test
# ---------------------------------------------------------------------------

def build_model_kruskal(
    output_dir: str, missing: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Kruskal–Wallis H-test across models, per (condition, pooling, metric).

    For each behavioral metric the per-board values of every available model
    form one group; the test asks whether the models share a common
    distribution of that metric. Reported as the H statistic, its degrees of
    freedom (n_models − 1), and the p-value. The per-board values are the same
    canonical-ordering samples that ``build_model_summary`` averages, so the
    test and the summary means describe the same data.
    """
    from scipy.stats import kruskal

    rows: List[Dict] = []
    for cond in CONDITIONS:
        for pm in POOLINGS:
            groups: Dict[str, List[np.ndarray]] = {m: [] for m in _BEHAVIORAL}
            models: List[str] = []
            for prefix in MODEL_PREFIXES:
                df = load_general_canonical(output_dir, prefix, cond)
                if df is None:
                    if missing is not None and pm == POOLINGS[0]:
                        missing.append(f"{prefix}_general_{cond}.csv (kruskal)")
                    continue
                models.append(prefix)
                for m in _BEHAVIORAL:
                    v = df[f"{m}_{pm}"].astype(float).to_numpy()
                    groups[m].append(v[~np.isnan(v)])
            for m in _BEHAVIORAL:
                arrs = [g for g in groups[m] if g.size > 0]
                if len(arrs) < 2:
                    continue
                h, p = kruskal(*arrs)
                rows.append({
                    "metric": _metric_out_name(m),
                    "condition": cond,
                    "pooling": pm,
                    "n_models": len(arrs),
                    "n_boards_total": int(sum(g.size for g in arrs)),
                    "kruskal_h": float(h),
                    "dof": len(arrs) - 1,
                    "p_value": float(p),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Social-preamble effects (paired, bootstrap CI)
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    deltas: np.ndarray, n_boot: int = 5000, seed: int = 2026, alpha: float = 0.05,
    chunk: int = 500,
) -> tuple:
    """Percentile bootstrap CI for the mean of ``deltas`` (chunked resampling)."""
    rng = np.random.default_rng(seed)
    n = deltas.shape[0]
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        m = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(m, n))
        means[done:done + m] = deltas[idx].mean(axis=1)
        done += m
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def build_social_effects(
    output_dir: str,
    n_boot: int = 5000,
    seed: int = 2026,
    missing: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Per-board paired deltas (with_social − no_social), aggregated per model.

    Boards are paired on ``row_id``; the same turns run under both conditions,
    so this isolates the preamble's effect from board difficulty.
    """
    rows: List[Dict] = []
    for prefix in MODEL_PREFIXES:
        base = load_general_canonical(output_dir, prefix, "no_social")
        soc = load_general_canonical(output_dir, prefix, "with_social")
        if base is None or soc is None:
            if missing is not None:
                missing.append(f"{prefix}: general CSV pair for social effects")
            continue
        merged = base.merge(soc, on="row_id", suffixes=("_ns", "_ws"))
        for pm in POOLINGS:
            for m in _BEHAVIORAL:
                a = merged[f"{m}_{pm}_ns"].astype(float).to_numpy()
                b = merged[f"{m}_{pm}_ws"].astype(float).to_numpy()
                ok = ~(np.isnan(a) | np.isnan(b))
                deltas = (b - a)[ok]
                if deltas.size == 0:
                    continue
                lo, hi = _bootstrap_ci(deltas, n_boot=n_boot, seed=seed)
                rec = {
                    "pooling": pm,
                    "metric": _metric_out_name(m),
                    "n_boards": int(deltas.size),
                    "mean_no_social": float(a[ok].mean()),
                    "mean_with_social": float(b[ok].mean()),
                    "delta": float(deltas.mean()),
                    "ci_low": lo,
                    "ci_high": hi,
                }
                rows.append(_add_identity(pd.DataFrame([rec]), prefix).iloc[0].to_dict())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Concordance (generation models only)
# ---------------------------------------------------------------------------

def build_concordance_summary(
    output_dir: str, missing: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generation behavior per (model, condition) from the generation CSVs.

    ``generation_accuracy`` is the rate at which the *generated* word is a
    target; ``concordance_{pooling}`` is the saved final-layer agreement with
    the geometric top-1 (missing generations count as non-concordant, matching
    how the run recorded them).
    """
    rows: List[Dict] = []
    for prefix in GENERATION_PREFIXES:
        for cond in CONDITIONS:
            path = os.path.join(_model_dir(output_dir, prefix),
                                f"{prefix}_generation_{cond}.csv")
            if not os.path.exists(path):
                if missing is not None:
                    missing.append(f"{prefix}_generation_{cond}.csv")
                continue
            df = pd.read_csv(path)
            rec = {
                "condition": cond,
                "n_boards": len(df),
                "generated_word_rate": float(df["generated_word"].notna().mean()),
                "generated_in_candidates": float(df["generated_in_candidates"].astype(float).mean()),
                "generation_accuracy": float(df["generated_correct"].astype(float).mean()),
            }
            for pm in POOLINGS:
                rec[f"concordance_{pm}"] = float(df[f"concordance_{pm}"].astype(float).mean())
            rows.append(_add_identity(pd.DataFrame([rec]), prefix).iloc[0].to_dict())
    return pd.DataFrame(rows)


def build_concordance_by_layer(
    output_dir: str, missing: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Layer-wise agreement between generation and geometry.

    For every layer, the geometric top-1 candidate (rank 1 by cosine-to-hint,
    canonical ordering) is compared to the word the model actually generated:

    - ``concordance`` — fraction of boards where they coincide (boards with no
      extracted generated word count as non-concordant, matching the saved
      final-layer concordance columns);
    - ``top1_accuracy`` — fraction of boards whose geometric top-1 at that
      layer is a target (the geometry-only analogue of generation accuracy).

    Reads the heavy ``{prefix}_metrics_{condition}.parquet`` with column
    pruning and a canonical-permutation filter, so memory stays modest.
    """
    frames: List[pd.DataFrame] = []
    for prefix in GENERATION_PREFIXES:
        for cond in CONDITIONS:
            mdir = _model_dir(output_dir, prefix)
            pq_path = os.path.join(mdir, f"{prefix}_metrics_{cond}.parquet")
            gen_path = os.path.join(mdir, f"{prefix}_generation_{cond}.csv")
            if not (os.path.exists(pq_path) and os.path.exists(gen_path)):
                if missing is not None:
                    missing.append(f"{prefix} {cond}: metrics parquet + generation CSV")
                continue

            gen = pd.read_csv(gen_path, usecols=["row_id", "generated_word"])
            gen = gen.drop_duplicates("row_id")

            cols = ["row_id", "layer", "word", "word_type"] + \
                   [f"rank_{pm}" for pm in POOLINGS]
            mdf = pd.read_parquet(
                pq_path, columns=cols + ["permutation_id"],
                filters=[("permutation_id", "==", 0)],
            )
            max_layer = int(mdf["layer"].max()) if len(mdf) else 0

            for pm in POOLINGS:
                top = mdf.loc[mdf[f"rank_{pm}"] == 1, ["row_id", "layer", "word", "word_type"]]
                # rank 1 is unique per (board, layer); the stable sort + dedupe
                # only guards against degenerate duplicates.
                top = (top.sort_values(["row_id", "layer", "word"], kind="stable")
                          .drop_duplicates(["row_id", "layer"]))
                top = top.merge(gen, on="row_id", how="left")
                top["concordant"] = (
                    top["generated_word"].notna() & (top["word"] == top["generated_word"])
                )
                top["top1_is_target"] = top["word_type"] == "target"
                agg = top.groupby("layer").agg(
                    concordance=("concordant", "mean"),
                    top1_accuracy=("top1_is_target", "mean"),
                    n_boards=("row_id", "nunique"),
                ).reset_index()
                agg.insert(1, "condition", cond)
                agg.insert(2, "pooling", pm)
                agg["layer_frac"] = agg["layer"] / max_layer if max_layer else 0.0
                frames.append(_add_identity(agg, prefix))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def build_all(
    output_dir: str,
    analysis_dir: str,
    *,
    n_boot: int = 5000,
    seed: int = 2026,
    include_concordance_by_layer: bool = True,
) -> Dict[str, int]:
    """Build every table and write it under ``analysis_dir``.

    Returns ``{relative_path: n_rows}`` for the run summary. Missing per-model
    inputs are skipped and reported.
    """
    os.makedirs(analysis_dir, exist_ok=True)
    missing: List[str] = []
    written: Dict[str, int] = {}

    builders = [
        ("analysis_model_summary.csv",
         lambda: build_model_summary(output_dir, missing, n_boot=n_boot, seed=seed)),
        ("analysis_model_kruskal.csv", lambda: build_model_kruskal(output_dir, missing)),
        ("analysis_social_effects.csv",
         lambda: build_social_effects(output_dir, n_boot=n_boot, seed=seed, missing=missing)),
        ("analysis_concordance.csv", lambda: build_concordance_summary(output_dir, missing)),
    ]
    if include_concordance_by_layer:
        builders.append(("analysis_concordance_by_layer.csv",
                         lambda: build_concordance_by_layer(output_dir, missing)))

    for fname, fn in builders:
        df = fn()
        if df.empty:
            print(f"  [tables] {fname}: no input data, skipped")
            continue
        path = os.path.join(analysis_dir, fname)
        df.to_csv(path, index=False)
        written[fname] = len(df)
        print(f"  [tables] wrote {fname} ({len(df)} rows)")
    if missing:
        print(f"  [tables] missing inputs (skipped): {len(missing)}")
        for m in missing:
            print(f"    - {m}")
    return written


def random_init_seed_spread(
    output_dir: str,
    base_prefixes=("random_qwen", "random_bert"),
    seeds=(2026, 2027, 2028),
    condition: str = "no_social",
    pooling: str = "mean",
) -> pd.DataFrame:
    """Per (control, weight seed): the primacy rho profile and the margin.

    Seed 2026 is the run under the bare prefix (``random_qwen``); every other
    seed is under ``{prefix}_s{seed}`` (``run --init-seed``). Reads the SC5/SC7
    per-layer CSVs each run writes and reports the peak |rho| (with its
    layer), the final-layer rho, and the max / final mean margin, so the
    spread across seeds can be quoted where the paper says "single seed".
    Missing runs are skipped, never zero-filled.
    """
    rows = []
    for base in base_prefixes:
        for seed in seeds:
            prefix = base if seed == seeds[0] else f"{base}_s{seed}"
            d = _model_dir(output_dir, prefix)
            rho_path = os.path.join(d, f"{prefix}_position_confound_by_layer.csv")
            mg_path = os.path.join(d, f"{prefix}_layer_margins_{pooling}_{condition}.csv")
            if not (os.path.exists(rho_path) and os.path.exists(mg_path)):
                continue
            rho = pd.read_csv(rho_path).sort_values("layer")
            mg = pd.read_csv(mg_path)
            mg = mg[(mg["pooling_method"] == pooling) & (mg["condition"] == condition)] \
                .sort_values("layer")
            r = rho["mean_rho"].to_numpy(dtype=float)
            m = mg["mean_margin"].to_numpy(dtype=float)
            peak = int(np.nanargmax(np.abs(r))) if r.size else -1
            rows.append({
                "base_prefix": base, "seed": int(seed), "prefix": prefix,
                "n_layers": int(len(r)),
                "peak_abs_rho": float(np.abs(r[peak])) if r.size else np.nan,
                "peak_rho_layer": int(rho["layer"].iloc[peak]) if r.size else -1,
                "final_rho": float(r[-1]) if r.size else np.nan,
                "max_margin": float(np.nanmax(m)) if m.size else np.nan,
                "final_margin": float(m[-1]) if m.size else np.nan,
            })
    return pd.DataFrame(rows, columns=["base_prefix", "seed", "prefix", "n_layers",
                                       "peak_abs_rho", "peak_rho_layer", "final_rho",
                                       "max_margin", "final_margin"])
