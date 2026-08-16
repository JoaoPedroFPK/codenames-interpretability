"""Figure builders for the ICLR 2027 paper.

Each ``fig_*`` function takes tidy DataFrames (the CSVs under ``output/analysis``
and ``output/lens_analysis``), writes one PDF and returns its ``Path``. Styling
is inherited from :mod:`codenames.analysis.figures` so every model keeps the one
colour/marker it has everywhere else. Panels are designed at the ICLR text
width; the printed point sizes are the ones set here.

Run from the repo root, e.g.::

    python -m codenames.analysis.paper_figures --fig geometry \
        --out ../Thesis-Writing/iclr2027/figures/F2_geometry.pdf
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .figures import (
    DEPTH_LABEL,
    PAPER_W,
    _letter,
    _line,
    _model_handles,
    _ordered_models,
    _paper_fig,
    _top_legend,
    model_style,
)

DECODERS: Sequence[str] = ("mistral", "qwen")
DECODER_NULL = "random_qwen"


# ---------------------------------------------------------------------------
# Curve annotation helpers
# ---------------------------------------------------------------------------

def find_humps(y: np.ndarray) -> Dict[str, int]:
    """Indices of the two largest interior local maxima and the trough between.

    A local maximum is a point strictly above both neighbours (endpoints
    excluded). The two with the largest values are returned in depth order as
    ``hump1`` and ``hump2``; ``trough`` is the argmin strictly between them.
    Curves with fewer than two interior maxima return the single maximum as
    ``hump1`` and ``None`` elsewhere, so callers can annotate what exists.
    """
    y = np.asarray(y, dtype=float)
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] > y[i + 1]]
    if len(idx) < 2:
        return {"hump1": int(np.argmax(y)), "hump2": None, "trough": None}
    top = sorted(sorted(idx, key=lambda i: y[i], reverse=True)[:2])
    h1, h2 = top
    trough = h1 + 1 + int(np.argmin(y[h1 + 1:h2])) if h2 - h1 > 1 else None
    return {"hump1": int(h1), "hump2": int(h2), "trough": trough}


def _mark_hump(ax, x: float, y: float, label: str, color: str, *, dy: float = 0.06) -> None:
    ax.plot([x], [y], marker="v", markersize=4.5, color=color, zorder=5,
            markeredgecolor="white", markeredgewidth=0.4)
    ax.annotate(label, (x, y), xytext=(0, 5), textcoords="offset points",
                ha="center", va="bottom", fontsize=5.5, color=color)


# ---------------------------------------------------------------------------
# F2 — geometry: two humps and a collapse
# ---------------------------------------------------------------------------

def fig_geometry(conc_by_layer: pd.DataFrame, margins: pd.DataFrame,
                 confound: pd.DataFrame, conc_summary: pd.DataFrame, *,
                 out_path: os.PathLike, pooling: str = "mean",
                 condition: str = "no_social") -> Path:
    """2x2: (a) decoder g(l) with humps, generation reference lines and the
    random-init null; (b) anisotropy-adjusted margin, all models; (c) mean
    pairwise-cosine anisotropy, all models; (d) positional confound rho.
    """
    plt = _paper_fig((PAPER_W, 4.6))
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_W, 4.6))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) geometric top-1 for the decoders + null --------------------------
    conc = conc_by_layer[(conc_by_layer["pooling"] == pooling)
                         & (conc_by_layer["condition"] == condition)]
    for m in (*DECODERS, DECODER_NULL):
        sub = conc[conc["model"] == m].sort_values("layer")
        if sub.empty:
            continue
        x = sub["layer_frac"].to_numpy()
        y = sub["top1_accuracy"].to_numpy()
        _line(ax_a, x, y, m)
        if m == DECODER_NULL:
            continue
        s = model_style(m)
        h = find_humps(y)
        for key in ("hump1", "hump2"):
            i = h[key]
            if i is not None:
                _mark_hump(ax_a, x[i], y[i], f"L{int(sub['layer'].iloc[i])}", s["color"])
        ref = conc_summary[(conc_summary["model"] == m)
                           & (conc_summary["condition"] == condition)]
        if not ref.empty:
            ax_a.axhline(float(ref["generation_accuracy"].iloc[0]),
                         color=s["color"], ls=":", lw=0.7, alpha=0.7, zorder=0)
    ax_a.set_ylabel("P(cosine top-1 is a target)")
    ax_a.set_ylim(0, 0.72)
    ax_a.text(0.99, 0.97, "dotted: generation accuracy", transform=ax_a.transAxes,
              ha="right", va="top", fontsize=5.5, color="#555555")

    # (b), (c) margins and anisotropy for every model ---------------------
    mg = margins[(margins["pooling_method"] == pooling)
                 & (margins["condition"] == condition)]
    models = _ordered_models(pd.concat([mg[["model"]], confound[["model"]]]))
    for m in models:
        sub = mg[mg["model"] == m].sort_values("layer_frac")
        if sub.empty:
            continue
        _line(ax_b, sub["layer_frac"].to_numpy(), sub["adjusted_margin"].to_numpy(), m)
        _line(ax_c, sub["layer_frac"].to_numpy(), sub["mean_anisotropy"].to_numpy(), m)
    ax_b.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
    ax_b.set_ylabel("Anisotropy-adjusted margin")
    ax_c.set_ylabel("Mean pairwise cosine (anisotropy)")
    ax_c.set_ylim(0, 1)

    # (d) positional confound ---------------------------------------------
    for m in models:
        sub = confound[confound["model"] == m].sort_values("layer_frac")
        if sub.empty:
            continue
        _line(ax_d, sub["layer_frac"].to_numpy(), sub["mean_rho"].to_numpy(), m)
    ax_d.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
    ax_d.set_ylabel(r"Spearman $\rho$(position, cosine)")

    for ax, letter in zip((ax_a, ax_b, ax_c, ax_d), "abcd"):
        ax.set_xlabel(DEPTH_LABEL)
        ax.set_xlim(0, 1)
        _letter(ax, letter)
    _top_legend(fig, _model_handles(models), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if os.path.exists(path) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fig", required=True, choices=("geometry",))
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis-dir", default="output/analysis")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--condition", default="no_social")
    a = ap.parse_args(argv)
    if a.fig == "geometry":
        out = fig_geometry(
            _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_layer_margins.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_position_confound.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_concordance.csv")),
            out_path=a.out, pooling=a.pooling, condition=a.condition)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
