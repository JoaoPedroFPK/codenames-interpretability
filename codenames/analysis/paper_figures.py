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


def emergence_depth(y: np.ndarray, frac: float = 0.9) -> int:
    """First layer index at which ``y`` reaches ``frac`` of its final value
    (the lens spec's emergence-depth rule)."""
    y = np.asarray(y, dtype=float)
    thr = frac * y[-1]
    hits = np.flatnonzero(y >= thr)
    return int(hits[0]) if len(hits) else int(len(y) - 1)


# ---------------------------------------------------------------------------
# F3 — lens: only the second hump decodes the answer
# ---------------------------------------------------------------------------

_LENS_LS = {"raw": "-", "tuned": "--"}


def fig_lens(curves: pd.DataFrame, shuffle: pd.DataFrame, conc_by_layer: pd.DataFrame, *,
             out_path: os.PathLike, models: Sequence[str] = DECODERS,
             null_model: str = DECODER_NULL, pooling: str = "mean",
             condition: str = "no_social") -> Path:
    """One panel per decoder, absolute layer axis: candidate-restricted lens
    top-1 at p_read (raw solid, tuned dashed, CI bands), the geometric curve
    g(l) in grey behind, the random-init and shuffled-label nulls as flat bands,
    and the raw-lens emergence depth marked.
    """
    plt = _paper_fig((PAPER_W, 2.5))
    fig, axes = plt.subplots(1, len(models), figsize=(PAPER_W, 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    conc = conc_by_layer[(conc_by_layer["pooling"] == pooling)
                         & (conc_by_layer["condition"] == condition)]
    null = curves[(curves["model"] == null_model) & (curves["lens"] == "raw")]
    for ax, m, letter in zip(axes, models, "abcdefg"):
        s = model_style(m)
        g = conc[conc["model"] == m].sort_values("layer")
        if not g.empty:
            ax.plot(g["layer"], g["top1_accuracy"], color="#9a9a9a", lw=1.0,
                    marker="o", markersize=1.8, zorder=1)
        # nulls as flat bands (both are flat within 0.06-0.08 at every layer)
        if not null.empty:
            ax.axhspan(float(null["top1"].min()), float(null["ci_hi"].max()),
                       color=s["color"], alpha=0.10, lw=0, zorder=0)
        sh = shuffle[shuffle["model"] == m]
        if not sh.empty:
            ax.axhline(float(sh["top1_p97_5"].max()), color="#444444", lw=0.6,
                       ls=":", zorder=1)
        for lens, ls in _LENS_LS.items():
            c = curves[(curves["model"] == m) & (curves["lens"] == lens)].sort_values("layer")
            if c.empty:
                continue
            x = c["layer"].to_numpy(); y = c["top1"].to_numpy()
            ax.fill_between(x, c["ci_lo"], c["ci_hi"], color=s["color"], alpha=0.18, lw=0)
            ax.plot(x, y, color=s["color"], ls=ls, lw=1.1, marker=s["marker"],
                    markersize=2.2, markevery=1, zorder=3,
                    markerfacecolor=s["color"] if lens == "raw" else "white")
            if lens == "raw":
                d = emergence_depth(y)
                ax.axvline(x[d], color=s["color"], lw=0.6, ls="-.", zorder=1)
                ax.annotate(f"emergence L{int(x[d])}", (x[d], 0.02), xytext=(3, 0),
                            textcoords="offset points", fontsize=5.5, color=s["color"],
                            ha="left", va="bottom")
        if not g.empty:
            h = find_humps(g["top1_accuracy"].to_numpy())
            for key in ("hump1", "hump2"):
                i = h[key]
                if i is not None:
                    xl = int(g["layer"].iloc[i]); yl = float(g["top1_accuracy"].iloc[i])
                    ax.plot([xl], [yl], marker="v", markersize=4, color="#9a9a9a", zorder=4,
                            markeredgecolor="white", markeredgewidth=0.4)
        ax.set_title(s["label"], fontsize=7, pad=2)
        ax.set_xlabel("Layer")
        ax.set_xlim(0, int(curves[curves["model"] == m]["layer"].max()))
        ax.set_ylim(0, 0.8)
        _letter(ax, letter)
    axes[0].set_ylabel("P(top-1 candidate is a target)")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([0], [0], color="#333333", ls="-", lw=1.1, label="raw lens at $p_{read}$"),
        Line2D([0], [0], color="#333333", ls="--", lw=1.1, label="tuned lens at $p_{read}$"),
        Line2D([0], [0], color="#9a9a9a", lw=1.0, marker="o", markersize=2, label="cosine $g(\\ell)$ (humps $\\blacktriangledown$)"),
        Patch(facecolor="#888888", alpha=0.25, label="random-init decoder"),
        Line2D([0], [0], color="#444444", ls=":", lw=0.6, label="shuffled labels, 97.5th pct"),
    ]
    _top_legend(fig, handles, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
    ap.add_argument("--fig", required=True, choices=("geometry", "lens"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis-dir", default="output/analysis")
    ap.add_argument("--lens-dir", default="output/lens_analysis")
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
    elif a.fig == "lens":
        out = fig_lens(
            _read(os.path.join(a.lens_dir, f"lens_curves_answer_{a.condition}.csv")),
            _read(os.path.join(a.lens_dir, f"lens_shuffle_control_answer_{a.condition}.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
            out_path=a.out, pooling=a.pooling, condition=a.condition)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
