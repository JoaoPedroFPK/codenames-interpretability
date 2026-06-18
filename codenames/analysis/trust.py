"""Trustworthiness sweep: score the rendered UMAP projection across models.

The projection figures (``visualize`` command) reduce each board's word
vectors to 2D with UMAP(metric=cosine) and validate the layout with three
cosine-aware diagnostics (:mod:`codenames.viz.metrics`). That validation only
runs on the handful of boards the figure pipeline samples. This module runs
the SAME reducer with the SAME scoring over the full vector subsample (100
boards per model/condition), at the representative depths the figures render,
producing a per-(model, condition, layer) trustworthiness surface that is
comparable across models.

Outputs:

- ``analysis_trustworthiness_detail.csv`` — one row per
  (model, condition, layer, board): trustworthiness / continuity / Shepard of
  the board's UMAP layout.
- ``analysis_trustworthiness.csv`` — the aggregate: mean ± SEM per
  (model, condition, layer), plus ``layer_frac`` for proportional depth.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .tables import MODEL_FAMILY

# Minimum words for a meaningful 2D layout + k-NN score on one board/layer.
MIN_WORDS = 5


def sweep_detail(
    output_root: str,
    *,
    pooling: str = "mean",
    n_layers: int = 6,
    k: int = 5,
    seed: int = 2026,
    max_boards: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-board UMAP quality scores for every discovered model.

    Uses :func:`codenames.viz.embedding.reduce` (the exact reducer the
    projection figures render) and :func:`codenames.viz.metrics.all_metrics`
    (the exact scoring the per-board ``dr_quality`` audit uses), so these
    numbers describe the published figures, not a lookalike pipeline.
    """
    from ..viz import embedding, loader, metrics
    from ..viz.style import select_layers

    records: List[Dict] = []
    discovered = loader.discover_models(output_root)
    if models:
        wanted = set(models)
        discovered = [r for r in discovered if r["prefix"] in wanted or r["name"] in wanted]

    for rec in discovered:
        prefix = rec["prefix"]
        family = MODEL_FAMILY.get(prefix, prefix)
        for mode in loader.MODES:
            cond = loader.load_condition(rec["dir"], prefix, mode, pooling)
            if cond is None:
                print(f"  [trust] {prefix}/{mode}: no vector data, skipped")
                continue
            index = cond["index"]
            num_layers = loader.num_layers(index)
            layers = select_layers(loader.available_layers(index), n_layers)
            board_ids = sorted(int(x) for x in index["row_id"].unique())
            if max_boards is not None:
                board_ids = board_ids[:max_boards]

            t0 = time.time()
            n_done = 0
            for layer in layers:
                for row_id in board_ids:
                    sel = index[(index["row_id"] == row_id) & (index["layer"] == layer)]
                    if len(sel) < MIN_WORDS:
                        continue
                    vecs = cond["vectors"][sel.index.to_numpy()]
                    try:
                        emb = embedding.reduce(vecs, "umap", seed=seed)
                        scores = metrics.all_metrics(vecs, emb, k=k)
                    except Exception as exc:  # degenerate boards must not kill the sweep
                        print(f"  [trust] {prefix}/{mode} L{layer} board {row_id}: "
                              f"failed ({exc})")
                        continue
                    records.append({
                        "model": prefix,
                        "model_family": family,
                        "is_random": prefix.startswith("random_"),
                        "condition": mode,
                        "layer": int(layer),
                        "layer_frac": (layer / num_layers) if num_layers else 0.0,
                        "row_id": int(row_id),
                        "n_words": int(len(sel)),
                        **scores,
                    })
                    n_done += 1
            print(f"  [trust] {prefix}/{mode}: {n_done} (board, layer) fits "
                  f"across {len(layers)} layers in {time.time() - t0:.1f}s")
    return pd.DataFrame(records)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the detail table to (model, condition, layer) mean ± SEM."""
    if detail.empty:
        return pd.DataFrame()
    keys = ["model", "model_family", "is_random", "condition", "layer", "layer_frac"]
    aggs = {}
    for m in ("trustworthiness", "continuity", "shepard"):
        aggs[m] = (m, "mean")
        aggs[f"{m}_sem"] = (m, lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    aggs["n_boards"] = ("row_id", "nunique")
    return detail.groupby(keys, as_index=False).agg(**aggs)


def run(
    output_root: str,
    analysis_dir: str,
    *,
    pooling: str = "mean",
    n_layers: int = 6,
    k: int = 5,
    seed: int = 2026,
    max_boards: Optional[int] = None,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, int]:
    """Run the sweep and write detail + summary CSVs. Returns row counts."""
    os.makedirs(analysis_dir, exist_ok=True)
    detail = sweep_detail(
        output_root, pooling=pooling, n_layers=n_layers, k=k, seed=seed,
        max_boards=max_boards, models=models,
    )
    written: Dict[str, int] = {}
    if detail.empty:
        print("  [trust] no scores produced — nothing written")
        return written
    for fname, df in [
        ("analysis_trustworthiness_detail.csv", detail),
        ("analysis_trustworthiness.csv", summarize(detail)),
    ]:
        path = os.path.join(analysis_dir, fname)
        df.to_csv(path, index=False)
        written[fname] = len(df)
        print(f"  [trust] wrote {fname} ({len(df)} rows)")
    return written
