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



def hump_ranges(y: np.ndarray, frac: float = 0.6) -> Dict[str, tuple]:
    """Contiguous layer ranges of the two humps: layers around each hump
    maximum at which ``y`` exceeds the trough by at least ``frac`` of that
    hump's height above the trough. Returns {'hump1': (lo, hi), 'hump2': (lo, hi)}
    (inclusive), or None entries when a hump is absent."""
    y = np.asarray(y, dtype=float)
    h = find_humps(y)
    if h["hump2"] is None or h["trough"] is None:
        return {"hump1": None, "hump2": None}
    trough = y[h["trough"]]
    out = {}
    for key in ("hump1", "hump2"):
        pk = h[key]; thr = trough + frac * (y[pk] - trough)
        lo = pk
        while lo - 1 >= 0 and y[lo - 1] >= thr:
            lo -= 1
        hi = pk
        while hi + 1 < len(y) and y[hi + 1] >= thr:
            hi += 1
        out[key] = (int(lo), int(hi))
    return out


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
                 condition: str = "no_social", panels: str = "abcd",
                 human_accuracy: Optional[float] = None,
                 panel_b_models: Optional[Sequence[str]] = None,
                 panel_a_models: Optional[Sequence[str]] = None) -> Path:
    """(a) decoder g(l) with humps, generation reference lines and the
    random-init null; (b) anisotropy-adjusted margin, all models; (c) mean
    pairwise-cosine anisotropy, all models; (d) positional confound rho.
    ``panels="ab"`` renders the one-row version, ``panels="a"`` the single
    main-text panel. ``panel_a_models`` overrides the decoders drawn in (a)
    (e.g. to add a base variant in the appendix); the null is always drawn.
    """
    two_rows = panels == "abcd"
    single = panels == "a"
    a_models = tuple(panel_a_models) if panel_a_models else DECODERS
    if two_rows:
        plt = _paper_fig((PAPER_W, 4.6))
        fig, axes = plt.subplots(2, 2, figsize=(PAPER_W, 4.6))
        ax_a, ax_b, ax_c, ax_d = axes.ravel()
    elif single:
        plt = _paper_fig((PAPER_W * 0.62, 2.05))
        fig, ax_a = plt.subplots(1, 1, figsize=(PAPER_W * 0.62, 2.05))
        ax_b = ax_c = ax_d = None
    else:
        plt = _paper_fig((PAPER_W, 2.15))
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(PAPER_W, 2.15))
        ax_c = ax_d = None

    # (a) geometric top-1 for the decoders + null --------------------------
    conc = conc_by_layer[(conc_by_layer["pooling"] == pooling)
                         & (conc_by_layer["condition"] == condition)]
    for m in (*a_models, DECODER_NULL):
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
    if human_accuracy is not None:
        ax_a.axhline(human_accuracy, color="#333333", ls="-", lw=0.6, alpha=0.6, zorder=0)
        ax_a.annotate("human guessers", (0.01, human_accuracy), xytext=(2, 2),
                      textcoords="offset points", fontsize=5.5, color="#333333", va="bottom")
    ax_a.set_ylabel("P(cosine top-1 is a target)")
    ax_a.set_ylim(0, 0.76)
    ax_a.text(0.99, 0.97, "dotted: generation accuracy", transform=ax_a.transAxes,
              ha="right", va="top", fontsize=5.5, color="#555555")

    # (b), (c) margins and anisotropy for every model ---------------------
    mg = margins[(margins["pooling_method"] == pooling)
                 & (margins["condition"] == condition)]
    models = _ordered_models(pd.concat([mg[["model"]], confound[["model"]]]))
    b_models = list(panel_b_models) if panel_b_models else models
    for m in models:
        sub = mg[mg["model"] == m].sort_values("layer_frac")
        if sub.empty:
            continue
        if ax_b is not None and m in b_models:
            _line(ax_b, sub["layer_frac"].to_numpy(), sub["adjusted_margin"].to_numpy(), m)
        if ax_c is not None:
            _line(ax_c, sub["layer_frac"].to_numpy(), sub["mean_anisotropy"].to_numpy(), m)
    if ax_b is not None:
        ax_b.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
        ax_b.set_ylabel("Anisotropy-adjusted margin")
    if ax_c is not None:
        ax_c.set_ylabel("Mean pairwise cosine (anisotropy)")
        ax_c.set_ylim(0, 1)

    # (d) positional confound ---------------------------------------------
    if ax_d is not None:
        for m in models:
            sub = confound[confound["model"] == m].sort_values("layer_frac")
            if sub.empty:
                continue
            _line(ax_d, sub["layer_frac"].to_numpy(), sub["mean_rho"].to_numpy(), m)
        ax_d.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
        ax_d.set_ylabel(r"Spearman $\rho$(position, cosine)")

    live = [ax for ax in (ax_a, ax_b, ax_c, ax_d) if ax is not None]
    for ax, letter in zip(live, "abcd"):
        ax.set_xlabel(DEPTH_LABEL)
        ax.set_xlim(0, 1)
        if len(live) > 1:        # a lone panel needs no letter
            _letter(ax, letter)
    if single:
        legend_models = [m for m in (*a_models, DECODER_NULL)
                         if not conc[conc["model"] == m].empty]
        _top_legend(fig, _model_handles(legend_models), ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.84))
    else:
        legend_models = list(models if two_rows else b_models)
        if two_rows:
            legend_models += [m for m in a_models if m not in legend_models
                              and not conc[conc["model"] == m].empty]
        _top_legend(fig, _model_handles(legend_models), ncol=4 if two_rows else 7)
        fig.tight_layout(rect=(0, 0, 1, 0.91 if two_rows else 0.86))

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
# F4 — patching: where the hint matters and where the answer forms
# ---------------------------------------------------------------------------

from ..causal.basis import ROLES  # noqa: E402  (role order of the scan grid)
from ..viz.style import OKABE_ITO  # noqa: E402

ROLE_STYLE = {
    "hint":        {"label": "hint span",           "color": OKABE_ITO["bluish_green"],   "marker": "o"},
    "cand_target": {"label": "candidate (target)",  "color": OKABE_ITO["sky_blue"],       "marker": "v"},
    "cand_donor":  {"label": "candidate (donor)",   "color": OKABE_ITO["orange"],         "marker": "s"},
    "generation":  {"label": "answer positions",    "color": OKABE_ITO["reddish_purple"], "marker": "D"},
    "scaffold":    {"label": "scaffold (before $p_{read}$)", "color": OKABE_ITO["vermillion"], "marker": "P"},
    "p_read":      {"label": "$p_{read}$ alone",    "color": "#000000",                   "marker": "X"},
    "final":       {"label": "generating position", "color": OKABE_ITO["blue"],           "marker": "^"},
}
NULL_STYLE = {"label": "random site (count-matched)", "color": "#888888"}


def confirmatory_curve(effects: pd.DataFrame, *, width: int = 1, n_boot: int = 2000,
                       seed: int = 2026) -> pd.DataFrame:
    """Per-(layer, role) mean normalised patch effect with a cluster bootstrap
    over turns (95%). Non-finite effects (turns with a degenerate denominator
    or a missing role) are dropped and the surviving count is reported."""
    rng = np.random.default_rng(seed)
    d = effects[(effects["width"] == width) & np.isfinite(effects["effect"])]
    rows = []
    for (layer, role), g in d.groupby(["layer", "role"]):
        x = g["effect"].to_numpy(dtype=float)
        n = len(x)
        idx = rng.integers(0, n, (n_boot, n))
        boot = x[idx].mean(axis=1)
        rows.append({"layer": int(layer), "role": role, "n": n, "e": float(x.mean()),
                     "median": float(np.median(x)),
                     "lo": float(np.percentile(boot, 2.5)),
                     "hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows).sort_values(["layer", "role"]).reset_index(drop=True)


def redundant_roles(cur: pd.DataFrame, tol: float = 1e-9) -> set:
    """Roles not worth drawing for a word-first model: when the ``final``
    (generating-position) curve coincides with ``p_read`` at every layer the
    readout *is* the generating position, so ``final`` duplicates ``p_read`` and
    the ``generation`` role (answer tokens) lies after the readout and is zero
    by construction. Returns the set to drop (empty otherwise)."""
    f = cur[cur["role"] == "final"].set_index("layer")["e"]
    p = cur[cur["role"] == "p_read"].set_index("layer")["e"]
    if f.empty or p.empty:
        return set()
    common = f.index.intersection(p.index)
    if len(common) and float((f.loc[common] - p.loc[common]).abs().max()) < tol:
        return {"final", "generation"}
    return set()


def grid_curve(grid: pd.DataFrame, nulls: pd.DataFrame, *, width: int = 1,
               n_boot: int = 2000, seed: int = 2026) -> tuple:
    """``(curve, null_band)`` for the role x layer grid (``causal-patch --grid``).

    ``curve`` is ``confirmatory_curve`` over every grid role; ``null_band`` is
    the per-layer mean of the matched random-site null pooled over token
    counts (one value per turn: the mean over that turn's null draws), with a
    turn-bootstrap CI. NaN cells (roles absent from a turn) are dropped and the
    surviving count ``n`` reported.
    """
    curve = confirmatory_curve(grid, width=width, n_boot=n_boot, seed=seed)
    rng = np.random.default_rng(seed)
    d = nulls[(nulls["width"] == width) & np.isfinite(nulls["effect"])]
    rows = []
    for layer, g in d.groupby("layer"):
        per_turn = g.groupby("row_id")["effect"].mean().to_numpy(dtype=float)
        n = len(per_turn)
        idx = rng.integers(0, n, (n_boot, n))
        boot = per_turn[idx].mean(axis=1)
        rows.append({"layer": int(layer), "n": n, "e": float(per_turn.mean()),
                     "lo": float(np.percentile(boot, 2.5)),
                     "hi": float(np.percentile(boot, 97.5))})
    return curve, pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def fig_causal(per_model: Dict[str, tuple], conc_by_layer: pd.DataFrame,
               emergence: Dict[str, int], *, out_path: os.PathLike,
               scan_clip: float = 1.5, pooling: str = "mean",
               condition: str = "no_social",
               grids: Optional[Dict[str, tuple]] = None,
               direction: str = "denoise", layout: str = "both") -> Path:
    """Per model, two panels: (left) the attribution scan over layer x role,
    clipped at ``scan_clip`` (screening only); (right) real-patch effects by
    role with cluster-bootstrap CIs, the two geometric humps as grey bands and
    the lens emergence depth dash-dotted.

    ``per_model`` maps model key -> (scan grid ndarray[layers, roles], effects
    DataFrame with columns layer, role, width, row_id, effect). When ``grids``
    supplies ``(grid, nulls)`` for a model, the right panel shows the FULL
    role x layer grid (every role at every layer, no untested cells) with the
    matched random-site null as a grey band, instead of the top-scan-site
    effects.
    """
    grids = grids or {}
    n = len(per_model)
    if layout == "both":
        plt = _paper_fig((PAPER_W, 2.15 * n))
        fig, axes = plt.subplots(n, 2, figsize=(PAPER_W, 2.15 * n), squeeze=False,
                                 gridspec_kw={"width_ratios": [1.0, 1.6]})
    else:
        plt = _paper_fig((PAPER_W, 2.2))
        fig, row_axes = plt.subplots(1, n, figsize=(PAPER_W, 2.2), squeeze=False,
                                     sharey=(layout == "curves"))
        # build a 2-column view so the loop below stays unchanged
        import matplotlib.pyplot as _plt  # noqa: F401
        dummy = [_plt.figure().add_subplot(111) for _ in range(n)]
        if layout == "curves":
            axes = np.array([[dummy[i], row_axes[0][i]] for i in range(n)], dtype=object)
        else:
            axes = np.array([[row_axes[0][i], dummy[i]] for i in range(n)], dtype=object)
    conc = conc_by_layer[(conc_by_layer["pooling"] == pooling)
                         & (conc_by_layer["condition"] == condition)]
    letters = iter("abcdefgh")
    for row, (m, (scan, effects)) in zip(axes, per_model.items()):
        ax_h, ax_c = row
        s = model_style(m)
        # heatmap of the screening scan ------------------------------------
        grid = np.clip(np.asarray(scan, dtype=float), 0, scan_clip)
        shown = [i for i, r in enumerate(ROLES) if r not in ("prefix",)]
        im = ax_h.imshow(grid[:, shown], aspect="auto", origin="lower",
                         cmap="Blues", vmin=0, vmax=scan_clip)
        ax_h.set_xticks(range(len(shown)))
        pretty = {"hint": "hint span", "post_hint": "post-hint", "list_scaffold": "list scaffold",
                  "cand_target": "cand. (target)", "cand_donor": "cand. (donor)",
                  "cand_other": "cand. (other)", "question": "question",
                  "final": "generating pos.", "generation": "answer pos."}
        ax_h.set_xticklabels([pretty.get(ROLES[i], ROLES[i]) for i in shown],
                             fontsize=5, rotation=90)
        ax_h.set_ylabel("Layer")
        ax_h.set_title("attribution scan (screening)", fontsize=6, pad=2, loc="right")
        cb = fig.colorbar(im, ax=ax_h, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label(f"first-order score (clipped at {scan_clip:g})", fontsize=5.5)
        _letter(ax_h, next(letters))
        # confirmatory curve ------------------------------------------------
        null_band = None
        if m in grids:
            cur, null_band = grid_curve(grids[m][0], grids[m][1], width=1)
        else:
            cur = confirmatory_curve(effects, width=1)
        g = conc[conc["model"] == m].sort_values("layer")
        if not g.empty:
            rng_ = hump_ranges(g["top1_accuracy"].to_numpy())
            for key in ("hump1", "hump2"):
                r = rng_[key]
                if r is not None:
                    lo = int(g["layer"].iloc[r[0]]); hi = int(g["layer"].iloc[r[1]])
                    ax_c.axvspan(lo - 0.5, hi + 0.5, color="#e4e4e4", lw=0, zorder=0)
                    ax_c.annotate("cosine hump", ((lo + hi) / 2, 1.02), fontsize=5, ha="center",
                                  va="bottom", color="#666666")
        if m in emergence:
            ax_c.axvline(emergence[m], color=s["color"], lw=0.7, ls="-.", zorder=1)
            ax_c.annotate(f"lens emergence L{emergence[m]}", (emergence[m], -0.02),
                          xytext=(3, 0), textcoords="offset points", fontsize=5,
                          color=s["color"], ha="left", va="bottom")
        skip = redundant_roles(cur) if m in grids else set()
        for role, st in ROLE_STYLE.items():
            c = cur[cur["role"] == role]
            if c.empty or role in skip:
                continue
            ax_c.errorbar(c["layer"], c["e"], yerr=[c["e"] - c["lo"], c["hi"] - c["e"]],
                          fmt=st["marker"], color=st["color"], markersize=3, lw=0.9,
                          capsize=1.5, label=st["label"], zorder=3)
            # connect only consecutive tested layers; gaps stay gaps
            xs = c["layer"].to_numpy(); ys = c["e"].to_numpy()
            for i in range(len(xs) - 1):
                if xs[i + 1] - xs[i] == 1:
                    ax_c.plot(xs[i:i + 2], ys[i:i + 2], color=st["color"], lw=0.8,
                              alpha=0.6, zorder=2)
        if null_band is not None and not null_band.empty:
            ax_c.fill_between(null_band["layer"], null_band["lo"], null_band["hi"],
                              color=NULL_STYLE["color"], alpha=0.25, lw=0, zorder=1)
            ax_c.plot(null_band["layer"], null_band["e"], color=NULL_STYLE["color"],
                      lw=0.8, ls="--", label=NULL_STYLE["label"], zorder=2)
        ax_c.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
        ax_c.axhline(1, lw=0.5, color="#bbbbbb", ls=":", zorder=0)
        ax_c.set_xlabel("Layer")
        ax_c.set_ylabel("Share of answer destroyed $e_{nec}$" if direction == "noise"
                        else "Normalised patch effect $e$")
        ax_c.set_ylim(-0.1, 1.1)
        ax_c.set_xlim(-0.5, int(cur["layer"].max()) + 0.5)
        what = "necessity (corrupt$\\to$clean)" if direction == "noise" else "real patches"
        title = (f"{what}, every role at every layer" if m in grids
                 else "real patches at the top-scan site per layer")
        ax_c.set_title(f"{s['label']}: {title}", fontsize=6, pad=2, loc="right")
        # The grid legend has 8 entries; the only empty region of a grid
        # panel is the right-centre band (between the saturated answer
        # curves above and the inert candidate curves below).
        _letter(ax_c, next(letters))
    # One legend row for the whole figure: the grid panels carry eight
    # series and no in-panel placement stays clear of the curves.
    seen, handles = set(), []
    for row in axes:
        for h, lab in zip(*row[1].get_legend_handles_labels()):
            if lab not in seen:
                seen.add(lab); handles.append(h)
    _top_legend(fig, handles, ncol=4)
    if layout != "both":
        for d in dummy:
            plt.close(d.figure)
        used = axes[:, 1] if layout == "curves" else axes[:, 0]
        for i, ax in enumerate(used):
            _letter(ax, "abcdefgh"[i])
        if layout == "curves":
            for ax in axes[1:, 1]:
                ax.set_ylabel("")
    fig.tight_layout(rect=(0, 0, 1, 0.93 if layout == "both" and n > 1 else 0.86))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F5 — triangulation overlay
# ---------------------------------------------------------------------------

def fig_triangulation(conc_by_layer: pd.DataFrame, lens_curves: pd.DataFrame,
                      effects_by_model: Dict[str, pd.DataFrame], *,
                      out_path: os.PathLike, models: Sequence[str] = DECODERS,
                      pooling: str = "mean", condition: str = "no_social",
                      shuffle: Optional[pd.DataFrame] = None,
                      null_model: str = DECODER_NULL, show_tuned: bool = True) -> Path:
    """One panel per decoder on an absolute layer axis: cosine g(l) (grey), raw
    lens L(l) at p_read (model colour), and the real patch effects for the hint
    span and the answer positions (role colours). Hump 1, trough, hump 2 and the
    output are annotated from g(l). Models without patching data show the two
    correlational curves only.
    """
    plt = _paper_fig((PAPER_W, 2.15))
    fig, axes = plt.subplots(1, len(models), figsize=(PAPER_W, 2.15), sharey=True)
    axes = np.atleast_1d(axes)
    conc = conc_by_layer[(conc_by_layer["pooling"] == pooling)
                         & (conc_by_layer["condition"] == condition)]
    for ax, m, letter in zip(axes, models, "abcdefg"):
        s = model_style(m)
        g = conc[conc["model"] == m].sort_values("layer")
        if not g.empty:
            gx = g["layer"].to_numpy(); gy = g["top1_accuracy"].to_numpy()
            ax.plot(gx, gy, color="#9a9a9a", lw=1.1, marker="o", markersize=1.8,
                    label="cosine $g(\\ell)$", zorder=2)
            h = find_humps(gy)
            marks = [("hump 1", h["hump1"]), ("trough", h["trough"]),
                     ("hump 2", h["hump2"]), ("output", len(gy) - 1)]
            for name, i in marks:
                if i is None:
                    continue
                ax.annotate(name, (gx[i], gy[i]), xytext=(0, 6),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=5.2, color="#666666", zorder=6)
                ax.plot([gx[i]], [gy[i]], marker="v", markersize=3.5, color="#9a9a9a",
                        markeredgecolor="white", markeredgewidth=0.4, zorder=4)
        null = lens_curves[(lens_curves["model"] == null_model) & (lens_curves["lens"] == "raw")]
        if not null.empty:
            ax.axhspan(float(null["top1"].min()), float(null["ci_hi"].max()),
                       color="#888888", alpha=0.18, lw=0, zorder=0,
                       label="random-init lens" if m == models[0] else None)
        if shuffle is not None:
            sh = shuffle[shuffle["model"] == m]
            if not sh.empty:
                ax.axhline(float(sh["top1_p97_5"].max()), color="#444444", lw=0.6, ls=":",
                           zorder=1, label="shuffled labels (97.5th pct)" if m == models[0] else None)
        L = lens_curves[(lens_curves["model"] == m) & (lens_curves["lens"] == "raw")].sort_values("layer")
        if not L.empty:
            ax.fill_between(L["layer"], L["ci_lo"], L["ci_hi"], color=s["color"], alpha=0.18, lw=0)
            ax.plot(L["layer"], L["top1"], color=s["color"], lw=1.2, marker=s["marker"],
                    markersize=2.0, label="raw lens $L(\\ell)$ at $p_{read}$", zorder=3)
            d = emergence_depth(L["top1"].to_numpy())
            ax.axvline(int(L["layer"].iloc[d]), color=s["color"], lw=0.6, ls="-.", zorder=1)
        T = lens_curves[(lens_curves["model"] == m) & (lens_curves["lens"] == "tuned")].sort_values("layer")
        if show_tuned and not T.empty:
            ax.plot(T["layer"], T["top1"], color=s["color"], lw=0.9, ls="--", marker=s["marker"],
                    markersize=1.8, markerfacecolor="white", label="tuned lens", zorder=3)
        eff = effects_by_model.get(m)
        if eff is not None:
            cur = confirmatory_curve(eff, width=1)
            if "p_read" in set(cur["role"]):
                # Full grid available: the hint span, a single candidate span
                # and p_read alone, at every layer.
                roles = ("hint", "cand_target", "p_read")
            else:
                # Per-layer-locus run: `final` is the generating position,
                # which for a word-first decoder (Qwen) IS p_read.
                roles = ("hint", "cand_donor", "generation", "final")
            for role in roles:
                c = cur[cur["role"] == role]
                if c.empty:
                    continue
                st = ROLE_STYLE[role]
                ax.errorbar(c["layer"], c["e"], yerr=[c["e"] - c["lo"], c["hi"] - c["e"]],
                            fmt=st["marker"], color=st["color"], markersize=2.4, lw=0.7,
                            capsize=1.0, label=f"patch $e$: {st['label']}", zorder=3)
        else:
            ax.text(0.5, 0.92, "no patching run", transform=ax.transAxes, ha="center",
                    fontsize=6, color="#888888")
        ax.set_title(s["label"], fontsize=7, pad=2)
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, (int(g["layer"].max()) if not g.empty else 32) + 0.5)
        if len(models) > 1:          # a lone panel needs no letter
            _letter(ax, letter)
    axes[0].set_ylabel("top-1 fraction / patch effect $e$")
    # Legend over ALL panels: a role tested in one decoder only (Qwen's
    # generating position) must still be named.
    seen, handles = set(), []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in seen:
                seen.add(lab); handles.append(h)
    _top_legend(fig, handles, ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
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
    ap.add_argument("--fig", required=True, choices=("geometry", "lens", "causal", "triangulation"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--analysis-dir", default="output/analysis")
    ap.add_argument("--lens-dir", default="output/lens_analysis")
    ap.add_argument("--models", default="mistral",
                    help="comma-separated; --fig causal: which models to load and draw; --fig triangulation: which decoders get a panel")
    ap.add_argument("--panels", default="abcd", help="geometry panels: abcd, ab or a")
    ap.add_argument("--layout", default="both", choices=("both", "curves", "scan"),
                    help="causal figure layout: both (scan+curve per model), curves only, or scan only")
    ap.add_argument("--human-accuracy", type=float, default=None)
    ap.add_argument("--panel-b-models", default=None, help="comma-separated subset for panel (b)")
    ap.add_argument("--panel-a-models", default=None,
                    help="comma-separated decoders drawn in panel (a) (default: mistral,qwen)")
    ap.add_argument("--emergence", default="mistral=20,qwen=25",
                    help="model=layer pairs for the lens emergence marker")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--condition", default="no_social")
    ap.add_argument("--no-grid", action="store_true",
                    help="--fig causal: ignore the role x layer grid even if present")
    ap.add_argument("--grid-direction", default="denoise", choices=("denoise", "noise"),
                    help="--fig causal: which grid to draw (sufficiency / necessity)")
    a = ap.parse_args(argv)
    if a.fig == "geometry":
        out = fig_geometry(
            _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_layer_margins.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_position_confound.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_concordance.csv")),
            out_path=a.out, pooling=a.pooling, condition=a.condition, panels=a.panels,
            human_accuracy=a.human_accuracy,
            panel_b_models=a.panel_b_models.split(",") if a.panel_b_models else None,
            panel_a_models=a.panel_a_models.split(",") if a.panel_a_models else None)
        print(f"wrote {out}")
    elif a.fig == "lens":
        out = fig_lens(
            _read(os.path.join(a.lens_dir, f"lens_curves_answer_{a.condition}.csv")),
            _read(os.path.join(a.lens_dir, f"lens_shuffle_control_answer_{a.condition}.csv")),
            _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
            out_path=a.out, pooling=a.pooling, condition=a.condition)
        print(f"wrote {out}")
    elif a.fig == "causal":
        per_model, grids = {}, {}
        for m in a.models.split(","):
            base = os.path.join("output", f"{m}_outputs")
            scan = np.load(os.path.join(base, f"{m}_causal_scan_counterfactual_{a.condition}.npy"))
            eff = pd.read_parquet(os.path.join(
                base, f"{m}_causal_effects_counterfactual_{a.condition}.parquet"))
            per_model[m] = (scan, eff)
            dsuf = "" if a.grid_direction == "denoise" else f"_{a.grid_direction}"
            g = os.path.join(base, f"{m}_causal_effects_grid_counterfactual_{a.condition}{dsuf}.parquet")
            nl = os.path.join(base, f"{m}_causal_nulls_counterfactual_{a.condition}{dsuf}.parquet")
            if not a.no_grid and os.path.exists(g) and os.path.exists(nl):
                grids[m] = (pd.read_parquet(g), pd.read_parquet(nl))
        emergence = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.emergence.split(",") if kv}
        out = fig_causal(per_model,
                         _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
                         emergence, out_path=a.out, pooling=a.pooling, condition=a.condition,
                         grids=grids, direction=a.grid_direction, layout=a.layout)
        print(f"wrote {out}")
    elif a.fig == "triangulation":
        effects = {}
        for m in a.models.split(","):
            base = os.path.join("output", f"{m}_outputs")
            grid_pq = os.path.join(base, f"{m}_causal_effects_grid_counterfactual_{a.condition}.parquet")
            pq = os.path.join(base, f"{m}_causal_effects_counterfactual_{a.condition}.parquet")
            if os.path.exists(grid_pq) and not a.no_grid:
                effects[m] = pd.read_parquet(grid_pq)
            elif os.path.exists(pq):
                effects[m] = pd.read_parquet(pq)
        out = fig_triangulation(
            _read(os.path.join(a.analysis_dir, "analysis_concordance_by_layer.csv")),
            _read(os.path.join(a.lens_dir, f"lens_curves_answer_{a.condition}.csv")),
            effects, out_path=a.out, models=tuple(a.models.split(",")),
            pooling=a.pooling, condition=a.condition,
            shuffle=_read(os.path.join(a.lens_dir, f"lens_shuffle_control_answer_{a.condition}.csv")))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
