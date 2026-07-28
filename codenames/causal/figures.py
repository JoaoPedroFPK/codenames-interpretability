"""Causal-tier figures (causal_spec.md §6).

Three exhibits:

1. ``patching_heatmap`` -- the (layer, position) grid, with FDR-surviving loci
   ringed and everything else muted. The caption must state whether the grid is
   attribution or real patches; presenting an attribution scan as a patching
   result would misreport a screen as evidence (§3.4).
2. ``dose_response`` -- steering effect against alpha for every arm, with
   parse-rate on a twin axis so off-distribution collapse is visible rather
   than inferred (§8).
3. ``triangulation`` -- the paper's central exhibit: the thesis margin, the
   lens curve, and the causal-effect curve on one depth axis, with BOTH humps
   of the geometric curve marked. Read at trajectory level, never as a
   single-peak match.

Colours follow codenames.analysis.figures.MODEL_STYLE so the paper reads as
one system.
"""

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..analysis.figures import _FALLBACK_STYLE, MODEL_STYLE
from ..viz.style import apply_publication_style, save_figure

_GEO_COLOR = "#999999"
_CLAIM_COLOR = "#d55e00"
_DOSE_ARMS = {
    "primary": {"color": "#0072B2", "ls": "-"},
    "random_direction": {"color": "#999999", "ls": "--"},
    "shuffled_label": {"color": "#cc79a7", "ls": ":"},
    "counterfactual_target": {"color": "#009e73", "ls": "-."},
}


def _style(model: Optional[str]) -> dict:
    return MODEL_STYLE.get(model, _FALLBACK_STYLE) if model else _FALLBACK_STYLE


def patching_heatmap(
    grid: np.ndarray,
    out_path: str,
    *,
    claimed_sites: Sequence[Tuple[int, int]] = (),
    title: str = "",
    metric_label: str = "normalized effect e",
) -> List[str]:
    """Heatmap over (layer, position); claimed loci ringed."""
    import matplotlib.pyplot as plt

    values = np.asarray(grid, dtype=float)
    n_layers, n_positions = values.shape
    for layer, position in claimed_sites:
        if not (0 <= layer < n_layers and 0 <= position < n_positions):
            raise ValueError(
                f"claimed site ({layer}, {position}) is outside the "
                f"{n_layers}x{n_positions} grid"
            )

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    limit = float(np.nanmax(np.abs(values))) or 1.0
    mesh = ax.imshow(
        values, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-limit, vmax=limit, interpolation="nearest",
    )
    for layer, position in claimed_sites:
        ax.add_patch(plt.Rectangle(
            (position - 0.5, layer - 0.5), 1, 1,
            fill=False, edgecolor=_CLAIM_COLOR, linewidth=1.6,
        ))
    ax.set_xlabel("token position")
    ax.set_ylabel("layer")
    if title:
        ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label=metric_label)
    fig.tight_layout()
    written = save_figure(fig, out_path)
    plt.close(fig)
    return written


def dose_response(curves: pd.DataFrame, out_path: str, *, title: str = "") -> List[str]:
    """Steering effect vs alpha per arm, with parse-rate on a twin axis."""
    import matplotlib.pyplot as plt

    required = {"alpha", "effect", "arm"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(f"dose_response needs columns {sorted(required)}; missing {sorted(missing)}")

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, block in curves.groupby("arm"):
        style = _DOSE_ARMS.get(str(arm), {"color": "#666666", "ls": "-"})
        block = block.sort_values("alpha")
        ax.plot(block["alpha"], block["effect"], marker="o", markersize=3,
                label=str(arm), color=style["color"], linestyle=style["ls"])

    ax.axhline(0.0, color=_GEO_COLOR, linewidth=0.8)
    ax.axvline(0.0, color=_GEO_COLOR, linewidth=0.8)
    ax.set_xlabel(r"steering strength $\alpha$ (x median residual norm)")
    ax.set_ylabel("change in target selection")
    if title:
        ax.set_title(title)

    if "parse_rate" in curves.columns:
        twin = ax.twinx()
        collapsed = curves.groupby("alpha")["parse_rate"].mean().sort_index()
        twin.plot(collapsed.index, collapsed.values, color="#b0b0b0",
                  linewidth=0.9, linestyle=":")
        twin.set_ylabel("parse rate", color="#8a8a8a")
        twin.set_ylim(0, 1.02)

    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    written = save_figure(fig, out_path)
    plt.close(fig)
    return written


def triangulation(
    *,
    margin: np.ndarray,
    lens: np.ndarray,
    causal: np.ndarray,
    out_path: str,
    humps: Sequence[int] = (),
    model: Optional[str] = None,
    title: str = "",
) -> List[str]:
    """The central exhibit: geometry || lens || causal effect across depth."""
    import matplotlib.pyplot as plt

    curves = [np.asarray(c, dtype=float) for c in (margin, lens, causal)]
    if len({c.shape[0] for c in curves}) != 1:
        raise ValueError("margin, lens and causal must have the same length")

    margin_c, lens_c, causal_c = curves
    layers = np.arange(margin_c.shape[0])
    style = _style(model)

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(layers, margin_c, "--", color=_GEO_COLOR, label="geometry (thesis margin)")
    ax.plot(layers, lens_c, ":", color=style["color"], label="lens decodability")
    ax.plot(layers, causal_c, "-", color=style["color"], marker=style["marker"],
            markersize=3, label="causal effect")

    for hump in humps:
        ax.axvline(hump, color=_GEO_COLOR, linewidth=0.8, alpha=0.6)
        ax.annotate("hump", xy=(hump, ax.get_ylim()[1]), fontsize=7,
                    color=_GEO_COLOR, ha="center", va="top")

    ax.set_xlabel("layer")
    ax.set_ylabel("normalised value")
    ax.set_title(title or (style.get("label") or ""))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    written = save_figure(fig, out_path)
    plt.close(fig)
    return written
