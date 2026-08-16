"""NeurIPS-format figure set for the cross-model analysis surface.

Renders the aggregated tables (``output/analysis/``) into a small set of
caption-ready COMPOSITE figures sized for a NeurIPS submission (5.5 in text
width), replacing the earlier one-metric-per-image set:

- **no in-figure titles or methodology footnotes** — that text belongs in the
  LaTeX caption; panels carry only a bold letter ``(a)``/``(b)`` and axis
  labels;
- **one shared legend per figure**, never one per panel;
- fonts are sized for the exact final render width (7 pt labels / 6 pt ticks
  at 5.5 in), so nothing falls below legibility when included at
  ``\\linewidth``;
- the random-init controls draw thin and dashed with open markers — present
  as reference lines, not protagonists;
- bounded quality metrics (trustworthiness etc.) are zoomed to the occupied
  range, stated in the caption, instead of wasting half the panel on [0, 1].

Main-text figures (``--figures-dir``):

1. ``fig1_behavioral`` — MRR | Hit@1 bars per model × condition.
2. ``fig2_social_effects`` — paired-delta forest: ΔMRR | ΔHit@1 | ΔMargin.
3. ``fig3_layer_geometry`` — 2×2: raw margin, adjusted margin, anisotropy,
   positional confound vs proportional depth (no_social).
4. ``fig4_concordance`` — generation–geometry concordance | top-1 accuracy.
5. ``fig5_trust`` — UMAP trustworthiness vs depth, no_social | with_social.

Appendix figures (``appendix_*``): Hit@3/5 bars, the with_social layer
geometry, semantic signal ratio, continuity/Shepard grid.

Per-board exemplars (``exemplar_*``): a 3-panel UMAP strip (embedding /
peak / final layer) and a hint-column-annotated heatmap, in the same
sans-serif house style as the aggregate charts (the thesis-style per-board
figures under ``visualization/<model>_outputs/`` are unchanged).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..viz.style import (
    OKABE_ITO,
    apply_publication_style,
    save_figure,
)

# === Model styling: ONE colour per model everywhere ======================
# Families take Okabe-Ito hues; the random-init negative controls share the
# family hue but render dashed/thin with open markers so they read as
# reference lines and survive grayscale.
MODEL_STYLE: Dict[str, Dict] = {
    "mistral":     {"label": "Mistral",       "color": OKABE_ITO["blue"],           "ls": "-",  "marker": "o", "order": 0},
    "qwen":        {"label": "Qwen",          "color": OKABE_ITO["vermillion"],     "ls": "-",  "marker": "s", "order": 1},
    "random_qwen": {"label": "Qwen (rand.)",  "color": OKABE_ITO["vermillion"],     "ls": "--", "marker": "s", "order": 2},
    "mistral_base": {"label": "Mistral (base)", "color": OKABE_ITO["blue"],          "ls": ":",  "marker": "o", "order": 0.5},
    "qwen_base":   {"label": "Qwen (base)",   "color": OKABE_ITO["vermillion"],     "ls": ":",  "marker": "s", "order": 1.5},
    "llama":       {"label": "Llama-3.1",     "color": OKABE_ITO["sky_blue"],         "ls": "-",  "marker": "D", "order": 1.7},
    "bert":        {"label": "BERT",          "color": OKABE_ITO["bluish_green"],   "ls": "-",  "marker": "^", "order": 3},
    "random_bert": {"label": "BERT (rand.)",  "color": OKABE_ITO["bluish_green"],   "ls": "--", "marker": "^", "order": 4},
    "t5":          {"label": "T5",            "color": OKABE_ITO["reddish_purple"], "ls": "-",  "marker": "D", "order": 5},
    "modernbert":  {"label": "ModernBERT",    "color": OKABE_ITO["orange"],         "ls": "-",  "marker": "v", "order": 6},
}
_FALLBACK_STYLE = {"label": None, "color": "#999999", "ls": "-", "marker": "o", "order": 99}

CONDITION_LABEL = {"no_social": "No social context", "with_social": "With social context"}
DEPTH_LABEL = "Proportional depth"

# === Paper geometry / typography =========================================
# NeurIPS text width. Figures are designed AT their final render width so the
# point sizes below are the printed sizes.
PAPER_W = 5.5
_FS = {"label": 7, "tick": 6, "legend": 6.5, "letter": 8, "annot": 5.5}

_PAPER_RC = {
    "font.size": _FS["label"],
    "axes.labelsize": _FS["label"],
    "axes.titlesize": _FS["letter"],
    "xtick.labelsize": _FS["tick"],
    "ytick.labelsize": _FS["tick"],
    "legend.fontsize": _FS["legend"],
}

_LW = 1.1          # trained-model lines
_LW_CTRL = 0.9     # random-control reference lines
_MS = 2.4


def model_style(model: str) -> Dict:
    s = MODEL_STYLE.get(model, dict(_FALLBACK_STYLE))
    if s["label"] is None:
        s = {**s, "label": model}
    return s


def _is_control(model: str) -> bool:
    return model.startswith("random_")


def _ordered_models(df: pd.DataFrame) -> List[str]:
    present = list(pd.unique(df["model"]))
    return sorted(present, key=lambda m: model_style(m)["order"])


def _letter(ax, letter: str) -> None:
    """Bold panel letter, set as a left-aligned title so it never collides."""
    ax.set_title(f"({letter})", loc="left", fontweight="bold",
                 fontsize=_FS["letter"], pad=4)


def _model_handles(models: Sequence[str]):
    from matplotlib.lines import Line2D

    handles = []
    for m in models:
        s = model_style(m)
        ctrl = _is_control(m)
        handles.append(Line2D(
            [0], [0], color=s["color"], ls=s["ls"],
            lw=_LW_CTRL if ctrl else _LW,
            marker=s["marker"], markersize=3.2, label=s["label"],
            markerfacecolor="white" if ctrl else s["color"],
        ))
    return handles


def _top_legend(fig, handles, *, ncol: Optional[int] = None) -> None:
    """One compact legend row above the panels."""
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=ncol or len(handles), frameon=False,
               fontsize=_FS["legend"], handlelength=1.9,
               columnspacing=0.9, handletextpad=0.4, borderaxespad=0.1)


def _line(ax, x, y, model: str, *, ls: Optional[str] = None) -> None:
    """One model's layer curve with a marker at every collected layer.

    A marker is drawn on every sampled point (``markevery=1``) so the figure
    distinguishes measurement from interpolation: each marker is a real
    per-layer value, and the connecting segments are the only interpolated part.
    Earlier thinning (a marker every ~n/7 points) made the regularly spaced
    markers read as decorative ticks unrelated to the collected layers.
    """
    s = model_style(model)
    ctrl = _is_control(model)
    ax.plot(x, y, color=s["color"], ls=ls or s["ls"],
            lw=_LW_CTRL if ctrl else _LW,
            marker=s["marker"], markersize=_MS, markevery=1,
            markerfacecolor="white" if ctrl else s["color"],
            zorder=2 if ctrl else 3)


def _paper_fig(figsize: Tuple[float, float]):
    import matplotlib.pyplot as plt

    apply_publication_style()
    plt.rcParams.update(_PAPER_RC)
    return plt


# ---------------------------------------------------------------------------
# Reusable panels
# ---------------------------------------------------------------------------

def _bars_panel(ax, summary: pd.DataFrame, metric: str, ylabel: str,
                models: Sequence[str]) -> None:
    """Paired condition bars for one metric (summary pre-filtered to pooling)."""
    x = np.arange(len(models))
    width = 0.38
    for off, cond, filled in ((-width / 2, "no_social", True),
                              (+width / 2, "with_social", False)):
        sub = summary[summary["condition"] == cond].set_index("model")
        vals = [sub[metric].get(m, np.nan) for m in models]
        errs = [1.96 * sub[f"{metric}_sem"].get(m, np.nan) for m in models]
        colors = [model_style(m)["color"] for m in models]
        ax.bar(x + off, vals, width,
               color=colors if filled else "white",
               edgecolor=colors, linewidth=0.6,
               hatch=None if filled else "////",
               yerr=errs, error_kw=dict(lw=0.5, capsize=1.2, ecolor="#555555"))
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([model_style(m)["label"] for m in models],
                       rotation=40, ha="right")
    ax.set_ylim(bottom=0)


def _forest_panel(ax, effects: pd.DataFrame, metric: str, xlabel: str,
                  models: Sequence[str], *, show_names: bool) -> None:
    """Per-model paired delta with 95% CI against a zero line."""
    sub = effects[effects["metric"] == metric].set_index("model")
    ypos = np.arange(len(models))[::-1]
    for m, y in zip(models, ypos):
        if m not in sub.index:
            continue
        r = sub.loc[m]
        s = model_style(m)
        ax.errorbar(
            r["delta"], y,
            xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
            fmt=s["marker"], color=s["color"], markersize=3.4,
            elinewidth=0.9, capsize=1.8,
            markerfacecolor="white" if _is_control(m) else s["color"],
        )
    ax.axvline(0, lw=0.6, color="#444444", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_yticks(ypos)
    ax.set_ylim(-0.6, len(models) - 0.4)
    # With sharey, labels render on the first panel only; setting [] on a
    # later panel would blank the shared labels everywhere.
    if show_names:
        ax.set_yticklabels([model_style(m)["label"] for m in models])
    ax.margins(x=0.12)


def _curve_panel(ax, df: pd.DataFrame, col: str, ylabel: str,
                 models: Sequence[str], *, zero_line: bool = False,
                 ylim: Optional[tuple] = None) -> None:
    for m in models:
        sub = df[df["model"] == m].sort_values("layer_frac")
        if sub.empty or col not in sub.columns:
            continue
        _line(ax, sub["layer_frac"].to_numpy(), sub[col].to_numpy(), m)
    if zero_line:
        ax.axhline(0, lw=0.5, color="#bbbbbb", zorder=0)
    ax.set_xlabel(DEPTH_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    if ylim:
        ax.set_ylim(*ylim)


# ---------------------------------------------------------------------------
# Main-text figures
# ---------------------------------------------------------------------------

def fig_behavioral(summary: pd.DataFrame, *, pooling: str = "mean",
                   metrics: Sequence[Tuple[str, str]] = (("mrr", "MRR"),
                                                         ("hit_at_1", "Hit@1"))):
    """Paired condition bars, one panel per metric (default MRR | Hit@1)."""
    from matplotlib.patches import Patch

    plt = _paper_fig((PAPER_W, 2.6))
    df = summary[summary["pooling"] == pooling]
    models = _ordered_models(df)
    fig, axes = plt.subplots(1, len(metrics), figsize=(PAPER_W, 2.6))
    axes = np.atleast_1d(axes)
    for ax, letter, (metric, ylabel) in zip(axes, "abcd", metrics):
        _bars_panel(ax, df, metric, ylabel, models)
        _letter(ax, letter)
    handles = [
        Patch(facecolor="#777777", edgecolor="#777777", label=CONDITION_LABEL["no_social"]),
        Patch(facecolor="white", edgecolor="#777777", hatch="////",
              label=CONDITION_LABEL["with_social"]),
    ]
    _top_legend(fig, handles)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


_SOCIAL_PANELS = [
    ("mrr", r"$\Delta$ MRR"),
    ("hit_at_1", r"$\Delta$ Hit@1"),
    ("raw_margin", r"$\Delta$ Raw margin"),
]


def fig_social_effects(effects: pd.DataFrame, *, pooling: str = "mean"):
    """Forest plot of paired per-board deltas, three metric panels."""
    plt = _paper_fig((PAPER_W, 2.2))
    df = effects[effects["pooling"] == pooling]
    models = _ordered_models(df)
    fig, axes = plt.subplots(1, len(_SOCIAL_PANELS), figsize=(PAPER_W, 2.2),
                             sharey=True)
    for i, (ax, (metric, xlabel)) in enumerate(zip(axes, _SOCIAL_PANELS)):
        _forest_panel(ax, df, metric, xlabel, models, show_names=(i == 0))
        _letter(ax, "abc"[i])
    fig.tight_layout()
    return fig


_GEOMETRY_PANELS = [
    ("margins", "mean_margin", "Raw margin", True, None),
    ("margins", "adjusted_margin", "Adjusted margin", True, None),
    ("margins", "mean_anisotropy", "Anisotropy", False, None),
    ("confound", "mean_rho", "Positional confound (ρ)", True, None),
]


def fig_layer_geometry(margins: pd.DataFrame, confound: pd.DataFrame, *,
                       pooling: str = "mean", condition: str = "no_social"):
    """2×2 layer-wise geometry: margins, adjusted, anisotropy, confound."""
    plt = _paper_fig((PAPER_W, 4.6))
    mg = margins[(margins["pooling_method"] == pooling)
                 & (margins["condition"] == condition)]
    models = _ordered_models(pd.concat([mg[["model"]], confound[["model"]]]))
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_W, 4.6))
    sources = {"margins": mg, "confound": confound}
    for ax, letter, (src, col, ylabel, zero, ylim) in zip(
            axes.ravel(), "abcd", _GEOMETRY_PANELS):
        _curve_panel(ax, sources[src], col, ylabel, models,
                     zero_line=zero, ylim=ylim)
        _letter(ax, letter)
    _top_legend(fig, _model_handles(models), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


_COND_LS = {"no_social": "-", "with_social": ":"}


def fig_concordance(by_layer: pd.DataFrame, summary: pd.DataFrame, *,
                    pooling: str = "mean"):
    """Generation–geometry agreement | geometric top-1 accuracy, per layer."""
    from matplotlib.lines import Line2D

    plt = _paper_fig((PAPER_W, 2.7))
    df = by_layer[by_layer["pooling"] == pooling]
    models = _ordered_models(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PAPER_W, 2.7))
    for m in models:
        s = model_style(m)
        for cond, ls in _COND_LS.items():
            sub = df[(df["model"] == m) & (df["condition"] == cond)].sort_values("layer_frac")
            if sub.empty:
                continue
            _line(ax1, sub["layer_frac"].to_numpy(), sub["concordance"].to_numpy(), m, ls=ls)
            _line(ax2, sub["layer_frac"].to_numpy(), sub["top1_accuracy"].to_numpy(), m, ls=ls)
            ref = summary[(summary["model"] == m) & (summary["condition"] == cond)]
            if not ref.empty:
                ax2.axhline(float(ref["generation_accuracy"].iloc[0]),
                            color=s["color"], ls=ls, lw=0.5, alpha=0.4, zorder=0)
    ax1.set_ylabel("P(generated = geometric top-1)")
    ax2.set_ylabel("P(geometric top-1 is a target)")
    for ax, letter in ((ax1, "a"), (ax2, "b")):
        ax.set_xlabel(DEPTH_LABEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        _letter(ax, letter)
    handles = _model_handles(models)
    handles += [Line2D([0], [0], color="#555555", ls=ls, lw=_LW,
                       label=CONDITION_LABEL[cond]) for cond, ls in _COND_LS.items()]
    _top_legend(fig, handles, ncol=5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def fig_trust(trust: pd.DataFrame, *, metric: str = "trustworthiness",
              ylim: tuple = (0.5, 1.0)):
    """UMAP projection quality vs depth, one panel per condition, zoomed."""
    plt = _paper_fig((PAPER_W, 2.5))
    models = _ordered_models(trust)
    conditions = [c for c in ("no_social", "with_social")
                  if c in set(trust["condition"])]
    fig, axes = plt.subplots(1, max(2, len(conditions)), figsize=(PAPER_W, 2.5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, letter, cond in zip(axes, "ab", conditions):
        sub_c = trust[trust["condition"] == cond]
        for m in models:
            sub = sub_c[sub_c["model"] == m].sort_values("layer_frac")
            if sub.empty:
                continue
            s = model_style(m)
            _line(ax, sub["layer_frac"].to_numpy(), sub[metric].to_numpy(), m)
            ci = 1.96 * sub[f"{metric}_sem"]
            ax.fill_between(sub["layer_frac"], sub[metric] - ci, sub[metric] + ci,
                            color=s["color"], alpha=0.13, linewidth=0)
        ax.set_xlabel(DEPTH_LABEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        _letter(ax, letter)
    axes[0].set_ylabel(metric.capitalize() if metric != "shepard"
                       else "Shepard correlation")
    _top_legend(fig, _model_handles(models), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------

def fig_hits_appendix(summary: pd.DataFrame, *, pooling: str = "mean"):
    return fig_behavioral(summary, pooling=pooling,
                          metrics=(("hit_at_3", "Hit@3"), ("hit_at_5", "Hit@5")))


def fig_semantic_ratio(shuffle: pd.DataFrame):
    """Semantic signal ratio vs depth (single panel, zoomed to [0.5, 1.02])."""
    plt = _paper_fig((PAPER_W * 0.62, 2.5))
    models = _ordered_models(shuffle)
    fig, ax = plt.subplots(figsize=(PAPER_W * 0.62, 2.5))
    _curve_panel(ax, shuffle, "semantic_ratio", "Semantic signal ratio",
                 models, ylim=(0.5, 1.02))
    _top_legend(fig, _model_handles(models), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return fig


def fig_trust_quality_appendix(trust: pd.DataFrame):
    """2×2: continuity and Shepard, no_social | with_social, zoomed."""
    plt = _paper_fig((PAPER_W, 4.4))
    models = _ordered_models(trust)
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_W, 4.4), sharex=True)
    panels = [("continuity", "no_social", (0.5, 1.0)),
              ("continuity", "with_social", (0.5, 1.0)),
              ("shepard", "no_social", (0.0, 0.9)),
              ("shepard", "with_social", (0.0, 0.9))]
    for ax, letter, (metric, cond, ylim) in zip(axes.ravel(), "abcd", panels):
        sub_c = trust[trust["condition"] == cond]
        for m in models:
            sub = sub_c[sub_c["model"] == m].sort_values("layer_frac")
            if sub.empty:
                continue
            s = model_style(m)
            _line(ax, sub["layer_frac"].to_numpy(), sub[metric].to_numpy(), m)
            ci = 1.96 * sub[f"{metric}_sem"]
            ax.fill_between(sub["layer_frac"], sub[metric] - ci, sub[metric] + ci,
                            color=s["color"], alpha=0.13, linewidth=0)
        ax.set_ylim(*ylim)
        ax.set_xlim(0, 1)
        name = "Continuity" if metric == "continuity" else "Shepard corr."
        ax.set_ylabel(f"{name}")
        _letter(ax, letter)
    for ax in axes[1]:
        ax.set_xlabel(DEPTH_LABEL)
    _top_legend(fig, _model_handles(models), ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Per-board exemplars (paper style: sans-serif, white background)
# ---------------------------------------------------------------------------

def _exemplar_layers(index: pd.DataFrame, vectors: np.ndarray, row_id: int,
                     num_layers: int) -> List[int]:
    """Embedding layer, the board's peak adjusted-margin layer, and the final layer.

    Peak is by the anisotropy-adjusted margin (raw / pairwise-cosine std), so
    the highlighted panel is the layer where the de-confounded separation is
    strongest.
    """
    from .boards import _board_layer_adjusted_margin

    margins = {}
    for layer in sorted(int(x) for x in index["layer"].unique()):
        sel = index[(index["row_id"] == row_id) & (index["layer"] == layer)]
        if len(sel) < 3:
            continue
        m = _board_layer_adjusted_margin(sel["word_type"].to_numpy(),
                                         vectors[sel.index.to_numpy()])
        if not np.isnan(m):
            margins[layer] = m
    if not margins:
        return []
    peak = max(margins, key=margins.get)
    chosen = [0, peak, num_layers]
    if len(set(chosen)) < 3:  # peak at an endpoint: insert a mid layer
        chosen = sorted(set(chosen) | {num_layers // 2})
    return sorted(set(chosen))


def _joint_layer_embedding(per_layer_vecs: Sequence[np.ndarray], *,
                           seed: int = 2026) -> np.ndarray:
    """Joint 2-D embedding of several layers of one board in a single frame.

    Each layer's vectors are L2-normalised and then centred on that layer's own
    mean before a single UMAP is fit across the stacked result, so the panels
    share one coordinate system: a word keeps its position relative to its
    neighbours across depth, and common neighbours land in the same region of
    every panel. Independent per-panel fits did not — they placed the same word
    in unrelated screen regions, so the panels were not comparable.

    The per-layer centring is what makes the shared frame robust. For deep
    causal decoders the representation shifts so much from one layer to the next
    that an un-centred joint fit spends its two dimensions separating the layers
    into disjoint blobs and collapses the within-layer structure; removing each
    layer's mean strips that offset and lets the fit use its space for the
    arrangement of words within each layer, which is what the panel shows.
    """
    from ..viz.embedding import _normalize
    import umap

    blocks = [(_normalize(v) - _normalize(v).mean(axis=0, keepdims=True))
              for v in per_layer_vecs]
    stack = np.vstack(blocks)
    n = stack.shape[0]
    nn = max(2, min(15, n - 1))
    reducer = umap.UMAP(n_components=2, metric="euclidean", n_neighbors=nn,
                        min_dist=0.1, random_state=seed)
    return reducer.fit_transform(stack)


def fig_board_exemplar(
    output_root: str, model: str, row_id: int, *,
    pooling: str = "mean", condition: str = "no_social",
    layers: Optional[Sequence[int]] = None, seed: int = 2026, k: int = 5,
    cond: Optional[Dict] = None, rec: Optional[Dict] = None,
    general: Optional[pd.DataFrame] = None,
):
    """3-panel UMAP strip (embedding / peak / final) in the paper style.

    Same reducer as the thesis projection figures (UMAP, cosine), same
    word-type palette, but: sans-serif on white, no in-figure title (the
    caption carries model/condition/board identity), and the panel's
    anisotropy-adjusted margin annotated bottom-left so the picture is
    quantitatively anchored to that measure (raw margin /
    pairwise-cosine std). Social-context points are labelled with their true
    value (``single``) rather than the bare tag (``giver.marriage``).

    ``cond``/``rec``/``general`` may be passed in to reuse already-loaded data
    (the vector NPZ is large; the batch renderer loads it once per
    model/condition).
    """
    from ..viz import embedding, loader
    from ..viz.style import depth_label, label_color, style_for
    from .boards import _board_layer_adjusted_margin

    plt = _paper_fig((PAPER_W, 2.4))
    if rec is None:
        rec = loader.resolve_model(output_root, model)
    if cond is None:
        cond = loader.load_condition(rec["dir"], rec["prefix"], condition, pooling)
    if cond is None:
        return None
    index = cond["index"]
    num_layers = loader.num_layers(index)
    sel_layers = list(layers) if layers else _exemplar_layers(
        index, cond["vectors"], row_id, num_layers)
    if not sel_layers:
        return None

    if general is None:
        general = loader.load_general(rec["dir"], rec["prefix"], condition)
    value_map = loader.giver_value_map(general, row_id)

    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    fig, axes = plt.subplots(1, len(sel_layers), figsize=(PAPER_W, 2.4))
    axes = np.atleast_1d(axes)
    types_present: set = set()

    # Joint dimensionality reduction across all selected layers (see
    # _joint_layer_embedding): one shared coordinate frame and shared axis
    # limits, so a word and its neighbours keep their position across depth and
    # the panels become comparable. The margin annotation is still computed in
    # the full hidden space, per panel.
    panel_data = []
    for layer in sel_layers:
        sel = index[(index["row_id"] == row_id) & (index["layer"] == layer)]
        types = sel["word_type"].astype(str).tolist()
        words = [value_map.get(w, w) if t == "giver_feature" else w
                 for w, t in zip(sel["word"].astype(str), types)]
        vecs = cond["vectors"][sel.index.to_numpy()]
        panel_data.append({"layer": layer, "types": types, "words": words, "vecs": vecs})
    counts = [len(p["vecs"]) for p in panel_data]
    joint = _joint_layer_embedding([p["vecs"] for p in panel_data], seed=seed)
    starts = np.cumsum([0] + counts)
    for i, p in enumerate(panel_data):
        p["emb"] = joint[starts[i]:starts[i + 1]]
    _pad = 0.08 * (joint.max(axis=0) - joint.min(axis=0) + 1e-9)
    xlim = (joint[:, 0].min() - _pad[0], joint[:, 0].max() + _pad[0])
    ylim = (joint[:, 1].min() - _pad[1], joint[:, 1].max() + _pad[1])

    for ax, letter, p in zip(axes, "abcdef", panel_data):
        layer = p["layer"]
        types = p["types"]
        words = p["words"]
        vecs = p["vecs"]
        emb = p["emb"]
        types_present.update(types)
        margin = _board_layer_adjusted_margin(np.asarray(types), vecs)

        hint_xy = None
        target_xy = []
        for i, wt in enumerate(types):
            s = style_for(wt)
            ax.scatter(emb[i, 0], emb[i, 1], c=s["color"], marker=s["marker"],
                       s=s["size"] * 0.55, linewidths=0.4,
                       edgecolors="white" if wt in ("hint", "black") else "none",
                       zorder=3)
            if wt == "hint":
                hint_xy = emb[i]
            elif wt == "target":
                target_xy.append(emb[i])
        if hint_xy is not None:
            for txy in target_xy:
                ax.annotate("", xy=tuple(txy), xytext=tuple(hint_xy),
                            arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                            color=style_for("target")["color"],
                                            alpha=0.7, mutation_scale=8),
                            zorder=2)
        texts = [ax.text(emb[i, 0], emb[i, 1], words[i],
                         fontsize=_FS["annot"],
                         fontweight="bold" if types[i] in ("hint", "target") else "normal",
                         color=label_color(types[i]), zorder=4)
                 for i in range(len(words))]
        if adjust_text is not None and texts:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", lw=0.3, color="#bbbbbb"),
                        expand=(1.2, 1.4))
        if not np.isnan(margin):
            ax.text(0.02, 0.02, f"adj. margin {margin:+.2f}",
                    transform=ax.transAxes, fontsize=_FS["tick"],
                    color="#666666", va="bottom")
        ax.set_title(f"({letter}) L{layer} — {depth_label(layer, num_layers)}",
                     loc="left", fontweight="bold", fontsize=_FS["letter"], pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    from ..viz.style import legend_handles
    handles = legend_handles(sorted(types_present))
    # No in-figure title: model/condition/board identity lives in the caption.
    # The word-type legend sits alone at the top.
    if handles:
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                   ncol=len(handles), frameon=False, fontsize=_FS["legend"],
                   handletextpad=0.3, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def fig_board_heatmap(
    output_root: str, model: str, row_id: int, layer: int, *,
    pooling: str = "mean", condition: str = "no_social",
    cond: Optional[Dict] = None, rec: Optional[Dict] = None,
    general: Optional[pd.DataFrame] = None,
):
    """Single-condition heatmap with every cell annotated.

    Uses the SAME fixed 0..1 cosine colour scale as the per-board pipeline
    heatmaps so the two are directly comparable, and sizes to the word count so
    the per-cell numbers stay legible. Tick labels darken the pale word-type
    hues so they stay legible at small point sizes. No in-figure title — the
    caption carries model/layer/condition. Social-context rows/columns are
    labelled with their true value (``single``) rather than the bare tag
    (``giver.marriage``).

    ``cond``/``rec``/``general`` may be passed in to reuse already-loaded data
    (the vector NPZ is large; the batch renderer loads it once per
    model/condition).
    """
    import seaborn as sns

    from ..viz import heatmap as hm
    from ..viz import loader
    from ..viz.style import label_color

    plt = _paper_fig((PAPER_W * 0.62, PAPER_W * 0.62))
    if rec is None:
        rec = loader.resolve_model(output_root, model)
    if cond is None:
        cond = loader.load_condition(rec["dir"], rec["prefix"], condition, pooling)
    if cond is None:
        return None
    words, types, vecs = loader.board_layer_words(cond, row_id, layer)
    if len(words) < 3:
        return None
    if general is None:
        general = loader.load_general(rec["dir"], rec["prefix"], condition)
    value_map = loader.giver_value_map(general, row_id)
    words = [value_map.get(w, w) if t == "giver_feature" else w
             for w, t in zip(words, types)]
    order = hm.order_words(words, types)
    words = [words[i] for i in order]
    types = [types[i] for i in order]
    mat = hm.cosine_matrix(vecs[order])
    n = len(words)
    # Fixed 0..1 scale, matching the per-board renderer (see heatmap.py): every
    # heatmap shares one scale so absolute cosine levels are comparable.
    vmax = 1.0

    # Size to the word count so the annotated numbers stay readable (a dense
    # ~18-word matrix is naturally wider than a paper column).
    side = max(PAPER_W * 0.62, n * 0.26)
    fig, ax = plt.subplots(figsize=(side, side))
    sub = mat[1:, :n - 1]
    mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
    sns.heatmap(sub, mask=mask, ax=ax, cmap="Reds", vmin=0.0, vmax=vmax,
                square=True, annot=True, fmt=".2f",
                annot_kws={"size": _FS["annot"]},
                linewidths=0.4, linecolor="white",
                xticklabels=words[:-1], yticklabels=words[1:],
                cbar_kws={"label": "Cosine similarity", "shrink": 0.7})
    # Contrast-aware annotation colour: white text on the darkest cells.
    for t in ax.texts:
        try:
            val = float(t.get_text())
        except ValueError:
            continue
        t.set_color("white" if val >= 0.62 * vmax else "#333333")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=_FS["tick"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=_FS["tick"])
    for lbl, wt in zip(ax.get_xticklabels(), types[:-1]):
        lbl.set_color(label_color(wt))
    for lbl, wt in zip(ax.get_yticklabels(), types[1:]):
        lbl.set_color(label_color(wt))
    # No in-figure title: model/layer/condition identity lives in the caption.
    fig.tight_layout()
    return fig


# Exemplar boards rendered by default in the thesis figures.
EXEMPLARS = [
    ("mistral", 3641),
    ("t5", 3641),
    ("modernbert", 5096),
]


# ---------------------------------------------------------------------------
# Renderer: tables in, figure files out
# ---------------------------------------------------------------------------

def _read(analysis_dir: str, name: str) -> Optional[pd.DataFrame]:
    path = os.path.join(analysis_dir, name)
    if not os.path.exists(path):
        print(f"  [figures] missing {name} — figure(s) skipped")
        return None
    return pd.read_csv(path)


def render_all(
    analysis_dir: str,
    figures_dir: str,
    *,
    pooling: str = "mean",
    formats: Sequence[str] = ("pdf", "png"),
    output_root: str = "output",
) -> List[str]:
    """Render the paper figure set. Returns written paths."""
    import matplotlib.pyplot as plt

    written: List[str] = []

    def _save(fig, stem: str):
        if fig is None:
            return
        paths = save_figure(fig, os.path.join(figures_dir, stem),
                            formats=tuple(formats))
        plt.close(fig)
        written.extend(paths)
        print(f"  [figures] wrote {stem}.{{{','.join(formats)}}}")

    summary = _read(analysis_dir, "analysis_model_summary.csv")
    effects = _read(analysis_dir, "analysis_social_effects.csv")
    margins = _read(analysis_dir, "analysis_layer_margins.csv")
    confound = _read(analysis_dir, "analysis_position_confound.csv")
    shuffle = _read(analysis_dir, "analysis_shuffle_decomposition.csv")
    conc_layer = _read(analysis_dir, "analysis_concordance_by_layer.csv")
    conc_sum = _read(analysis_dir, "analysis_concordance.csv")
    trust = _read(analysis_dir, "analysis_trustworthiness.csv")

    if summary is not None and not summary.empty:
        _save(fig_behavioral(summary, pooling=pooling), "fig1_behavioral")
        _save(fig_hits_appendix(summary, pooling=pooling), "appendix_hits_3_5")
    if effects is not None and not effects.empty:
        _save(fig_social_effects(effects, pooling=pooling), "fig2_social_effects")
    if margins is not None and confound is not None and not margins.empty:
        _save(fig_layer_geometry(margins, confound, pooling=pooling,
                                 condition="no_social"), "fig3_layer_geometry")
        _save(fig_layer_geometry(margins, confound, pooling=pooling,
                                 condition="with_social"),
              "appendix_layer_geometry_with_social")
    if shuffle is not None and not shuffle.empty:
        _save(fig_semantic_ratio(shuffle), "appendix_semantic_ratio")
    if conc_layer is not None and conc_sum is not None and not conc_layer.empty:
        _save(fig_concordance(conc_layer, conc_sum, pooling=pooling),
              "fig4_concordance")
    if trust is not None and not trust.empty:
        _save(fig_trust(trust), "fig5_trust")
        _save(fig_trust_quality_appendix(trust), "appendix_trust_quality")

    peaks = _read(analysis_dir, "analysis_board_peaks.csv")
    for model, row_id in EXEMPLARS:
        try:
            _save(fig_board_exemplar(output_root, model, row_id,
                                     pooling=pooling),
                  f"exemplar_{model}_{row_id}_umap")
        except Exception as exc:
            print(f"  [figures] exemplar {model}/{row_id} UMAP failed: {exc}")
        if peaks is None:
            continue
        row = peaks[(peaks["model"] == model) & (peaks["row_id"] == row_id)
                    & (peaks["condition"] == "no_social")]
        if row.empty:
            continue
        peak_layer = int(row["peak_layer"].iloc[0])
        try:
            _save(fig_board_heatmap(output_root, model, row_id, peak_layer,
                                    pooling=pooling),
                  f"exemplar_{model}_{row_id}_heatmap_L{peak_layer:02d}")
        except Exception as exc:
            print(f"  [figures] exemplar {model}/{row_id} heatmap failed: {exc}")

    return written


def render_examples(
    output_root: str,
    figures_dir: str,
    *,
    boards: Optional[Sequence[int]] = None,
    n_boards: int = 10,
    pooling: str = "mean",
    seed: int = 2026,
    formats: Sequence[str] = ("pdf", "png"),
) -> List[str]:
    """Paper-style example LIBRARY for every model, organised under ``figures_dir``.

    For each model × condition × board it renders, in the aggregate exemplar
    style (white background, sans-serif, fixed 0..1 heatmap scale):

    - one 3-panel UMAP strip (embedding / peak / final) ->
      ``umaps/{model}_{board}_{condition}.{pdf,png}``
    - one fully-annotated heatmap at EACH of those same three layers ->
      ``heatmaps/{model}_{board}_L{xx}_{condition}.{pdf,png}``

    Boards default to the cross-model shared sample (same ``n_boards`` for every
    model, so the library is directly comparable). The large vector NPZ is loaded
    once per (model, condition) and reused across all of that model's boards.
    """
    import matplotlib.pyplot as plt

    from ..viz import loader
    from ..viz.pipeline import shared_board_sample

    if boards is None:
        boards, info = shared_board_sample(output_root, n_boards, pooling=pooling, seed=seed)
        print(f"[examples] shared boards ({info['shared']} available): {boards}")
    boards = [int(b) for b in boards]

    umap_dir = os.path.join(figures_dir, "umaps")
    heat_dir = os.path.join(figures_dir, "heatmaps")
    written: List[str] = []

    def _save(fig, directory: str, stem: str):
        if fig is None:
            return
        paths = save_figure(fig, os.path.join(directory, stem), formats=tuple(formats))
        plt.close(fig)
        written.extend(paths)

    for rec in loader.discover_models(output_root):
        prefix = rec["prefix"]
        for condition in ("no_social", "with_social"):
            cond = loader.load_condition(rec["dir"], prefix, condition, pooling)
            if cond is None:
                continue
            general = loader.load_general(rec["dir"], prefix, condition)
            num_layers = loader.num_layers(cond["index"])
            present = set(int(x) for x in cond["index"]["row_id"].unique())
            n_board_fig = 0
            for row_id in boards:
                if row_id not in present:
                    continue
                sel_layers = _exemplar_layers(
                    cond["index"], cond["vectors"], row_id, num_layers)
                if not sel_layers:
                    continue
                # 3-panel UMAP strip (shares the layer set with the heatmaps).
                try:
                    _save(fig_board_exemplar(
                        output_root, prefix, row_id, condition=condition,
                        pooling=pooling, layers=sel_layers, seed=seed,
                        cond=cond, rec=rec, general=general),
                        umap_dir, f"{prefix}_{row_id}_{condition}")
                except Exception as exc:
                    print(f"  [examples] {prefix}/{row_id}/{condition} UMAP failed: {exc}")
                # One heatmap per selected layer.
                for layer in sel_layers:
                    try:
                        _save(fig_board_heatmap(
                            output_root, prefix, row_id, layer, condition=condition,
                            pooling=pooling, cond=cond, rec=rec, general=general),
                            heat_dir, f"{prefix}_{row_id}_L{layer:02d}_{condition}")
                    except Exception as exc:
                        print(f"  [examples] {prefix}/{row_id}/{condition} "
                              f"heatmap L{layer} failed: {exc}")
                n_board_fig += 1
            print(f"  [examples] {prefix} · {condition}: {n_board_fig} boards rendered")

    print(f"[examples] done -> {umap_dir} / {heat_dir} ({len(written)} files)")
    return written
