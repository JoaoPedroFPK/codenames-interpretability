"""Per-board peak-depth analysis over the vector subsample.

The aggregate layer tables say *where* each model's hint–target separation
peaks on average (``analysis_layer_margins.csv``). The per-board figures
(UMAP projections, heatmaps under ``visualization/``) render individual
boards from the stored vector subsample. This module bridges the two: it
recomputes the raw separation margin per (board, layer) **from the exact
vectors the figures render**, so each rendered board can be checked against
the aggregate peak-depth signature — does this board's binding really peak
where the model-level curve says it should?

Output ``analysis_board_peaks.csv``: one row per (model, condition, board)
with the board's full-depth peak (layer, ``layer_frac``, margin), its
final-layer margin, and ``retention`` (final / peak, clipped at 0) — the
per-board analogue of the "retained at final" statistic in the results
report.

Margin definition matches ``extraction.py``: mean cos(hint, targets) −
mean cos(hint, non-targets), with non-targets = ``black`` + ``tan`` words
(giver-feature words are excluded, as in the run).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .tables import MODEL_FAMILY

_NONTARGET_TYPES = ("black", "tan")


def _board_layer_margin(types: np.ndarray, vecs: np.ndarray) -> float:
    """Raw margin for one (board, layer) from its stored word vectors."""
    hint = vecs[types == "hint"]
    tgt = vecs[types == "target"]
    non = vecs[np.isin(types, _NONTARGET_TYPES)]
    if hint.shape[0] != 1 or tgt.shape[0] == 0 or non.shape[0] == 0:
        return float("nan")
    h = hint[0].astype(np.float64)
    hn = np.linalg.norm(h)
    if hn == 0:
        return float("nan")

    def _mean_cos(block: np.ndarray) -> float:
        b = block.astype(np.float64)
        norms = np.linalg.norm(b, axis=1)
        ok = norms > 0
        if not ok.any():
            return float("nan")
        return float(np.mean((b[ok] @ h) / (norms[ok] * hn)))

    return _mean_cos(tgt) - _mean_cos(non)


def _board_layer_pairwise_std(vecs: np.ndarray) -> float:
    """Std of the off-diagonal pairwise cosines among a board's word vectors.

    This is ``layer_std_pairwise_cosine`` (sanity.py SC3/SC5) recomputed per
    board: the spread of the cosine distribution, i.e. the anisotropy-driven
    compression of the cone, used to normalise the raw margin.
    """
    X = vecs.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    ok = norms[:, 0] > 0
    if ok.sum() < 2:
        return float("nan")
    Xn = X[ok] / norms[ok]
    C = np.clip(Xn @ Xn.T, -1.0, 1.0)
    iu = np.triu_indices(C.shape[0], k=1)
    return float(np.std(C[iu]))


def _board_layer_adjusted_margin(types: np.ndarray, vecs: np.ndarray) -> float:
    """Anisotropy-adjusted margin = raw margin / std(pairwise cosines).

    Matches the aggregate ``adjusted_margin`` (sanity.py): dividing by the
    cosine-spread de-confounds the raw separation from how compressed the layer's
    cone is, so margins are comparable across depth.
    """
    raw = _board_layer_margin(types, vecs)
    if np.isnan(raw):
        return float("nan")
    std = _board_layer_pairwise_std(vecs)
    if np.isnan(std) or std <= 0:
        return float("nan")
    return raw / std


def board_margins(
    output_root: str,
    *,
    pooling: str = "mean",
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-(model, condition, board, layer) raw margins from the subsample."""
    from ..viz import loader

    records: List[Dict] = []
    discovered = loader.discover_models(output_root)
    if models:
        wanted = set(models)
        discovered = [r for r in discovered if r["prefix"] in wanted or r["name"] in wanted]

    for rec in discovered:
        prefix = rec["prefix"]
        for mode in loader.MODES:
            cond = loader.load_condition(rec["dir"], prefix, mode, pooling)
            if cond is None:
                continue
            index = cond["index"]
            num_layers = loader.num_layers(index)
            grouped = index.groupby(["row_id", "layer"], sort=True)
            for (row_id, layer), sel in grouped:
                margin = _board_layer_margin(
                    sel["word_type"].to_numpy(), cond["vectors"][sel.index.to_numpy()])
                if np.isnan(margin):
                    continue
                records.append({
                    "model": prefix,
                    "model_family": MODEL_FAMILY.get(prefix, prefix),
                    "is_random": prefix.startswith("random_"),
                    "condition": mode,
                    "row_id": int(row_id),
                    "layer": int(layer),
                    "layer_frac": (layer / num_layers) if num_layers else 0.0,
                    "margin": margin,
                })
    return pd.DataFrame(records)


def peak_table(margins: pd.DataFrame) -> pd.DataFrame:
    """Collapse the per-layer margins to one peak row per (model, condition, board)."""
    if margins.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for (model, cond, row_id), sub in margins.groupby(["model", "condition", "row_id"]):
        sub = sub.sort_values("layer")
        peak = sub.loc[sub["margin"].idxmax()]
        final = sub.iloc[-1]
        rows.append({
            "model": model,
            "model_family": sub["model_family"].iloc[0],
            "is_random": bool(sub["is_random"].iloc[0]),
            "condition": cond,
            "row_id": int(row_id),
            "n_layers": int(len(sub)),
            "peak_layer": int(peak["layer"]),
            "peak_frac": float(peak["layer_frac"]),
            "peak_margin": float(peak["margin"]),
            "final_margin": float(final["margin"]),
            "retention": (
                float(final["margin"] / peak["margin"])
                if peak["margin"] > 0 else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def run(
    output_root: str,
    analysis_dir: str,
    *,
    pooling: str = "mean",
    models: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Build and write the per-board peak table. Returns row counts."""
    os.makedirs(analysis_dir, exist_ok=True)
    margins = board_margins(output_root, pooling=pooling, models=models)
    peaks = peak_table(margins)
    written: Dict[str, int] = {}
    if peaks.empty:
        print("  [boards] no vector subsample data — nothing written")
        return written
    path = os.path.join(analysis_dir, "analysis_board_peaks.csv")
    peaks.to_csv(path, index=False)
    written["analysis_board_peaks.csv"] = len(peaks)
    print(f"  [boards] wrote analysis_board_peaks.csv ({len(peaks)} rows)")
    return written
