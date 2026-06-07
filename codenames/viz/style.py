"""Shared figure template for the Results chapter — ONE house style.

Every Results figure imports this module so the whole chapter has a single
visual grammar: one palette (word-type -> colour/marker, used identically for
markers and axis-label colours), one typography stack, one legend treatment, and
one export path (vector PDF at thesis text width + 300 DPI PNG fallback).

It follows the bundled ``scientific-visualization`` skill (despined axes,
sans-serif, vector PDF) but is self-contained so the package works anywhere.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

# Batch / headless rendering — these figures are produced by a CLI pipeline,
# never in an interactive session. Set before pyplot is imported anywhere.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# === Palette =============================================================
# Okabe-Ito colorblind-safe palette.
OKABE_ITO: Dict[str, str] = {
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermillion":     "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":          "#000000",
}

# Canonical word-type styling — the SINGLE source of truth for colour + marker +
# legend label + ordering, used identically by markers (projection) and
# axis-label colours (heatmap). ``order`` fixes legend order and z-order.
WORD_TYPE_STYLE: Dict[str, Dict] = {
    "hint":          {"label": "Hint",          "color": OKABE_ITO["vermillion"],   "marker": "D", "order": 5},
    "target":        {"label": "Target",        "color": OKABE_ITO["bluish_green"], "marker": "o", "order": 4},
    "black":         {"label": "Assassin",      "color": OKABE_ITO["black"],        "marker": "X", "order": 3},
    "tan":           {"label": "Neutral",       "color": OKABE_ITO["orange"],       "marker": "o", "order": 1},
    "giver_feature": {"label": "Giver feature", "color": OKABE_ITO["sky_blue"],     "marker": "^", "order": 2},
}

# Fallback for any unexpected word_type so the pipeline never crashes.
_UNKNOWN_STYLE = {"label": "Other", "color": "#999999", "marker": "o", "order": 0}


def style_for(word_type: str) -> Dict:
    """Return the plotting style dict for a ``word_type`` (never raises)."""
    return WORD_TYPE_STYLE.get(word_type, _UNKNOWN_STYLE)


# === Typography ==========================================================
FONT_STACK: List[str] = ["Arial", "Helvetica", "DejaVu Sans"]
FOOTNOTE_COLOR = "#8a8a8a"   # light grey; footnotes recede

# One size scale (pt at final render size), referenced everywhere by name.
FS: Dict[str, float] = {
    "suptitle":     11,
    "panel_title":   9,
    "axis_label":    7,
    "tick_label":    6,
    "annot":         5,     # small in-panel / in-cell annotations
    "word_label":    5.5,   # projection point labels
    "footnote":      6,
    "legend":        7,
    "legend_title":  8,
}


# === Thesis geometry =====================================================
# LaTeX \textwidth for a typical single-column thesis (~16 cm). Figures render at
# this physical size so fonts are correct when included with \includegraphics at
# \textwidth; vector PDF then scales losslessly.
TEXTWIDTH_IN = 6.3
COLWIDTH_IN = TEXTWIDTH_IN / 2.0


def grid_size(
    n_cols: int,
    n_rows: int,
    *,
    width_in: float = TEXTWIDTH_IN,
    panel_aspect: float = 1.0,
    header_in: float = 0.55,
    footer_in: float = 0.65,
) -> Tuple[float, float]:
    """Figure size (inches) for an ``n_rows`` x ``n_cols`` panel grid at a target
    width, reserving space for the suptitle (header) and legend/footnote (footer).
    """
    panel_w = width_in / max(1, n_cols)
    panel_h = panel_w * panel_aspect
    return (width_in, n_rows * panel_h + header_in + footer_in)


# === Global rcParams =====================================================
def apply_publication_style() -> None:
    """Apply the house matplotlib rcParams (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": FS["axis_label"],
        "axes.titlesize": FS["panel_title"],
        "axes.labelsize": FS["axis_label"],
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": FS["tick_label"],
        "ytick.labelsize": FS["tick_label"],
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": FS["legend"],
        "legend.frameon": False,
        "figure.titlesize": FS["suptitle"],
        "axes.grid": False,
        "pdf.fonttype": 42,   # embed TrueType (editable text in the PDF)
        "ps.fonttype": 42,
    })


# === Legend (one treatment, reused everywhere) ===========================
def legend_handles(word_types: Sequence[str]):
    """Proxy legend handles (colour + marker shape) for the given word types,
    ordered by the canonical ``order`` field. Only types present are shown.
    """
    from matplotlib.lines import Line2D

    present = [wt for wt in WORD_TYPE_STYLE if wt in set(word_types)]
    present.sort(key=lambda wt: -WORD_TYPE_STYLE[wt]["order"])
    handles = []
    for wt in present:
        s = WORD_TYPE_STYLE[wt]
        handles.append(
            Line2D([0], [0], marker=s["marker"], color="none",
                   markerfacecolor=s["color"], markeredgecolor=s["color"],
                   markersize=6, label=s["label"], linestyle="none")
        )
    return handles


def add_word_type_legend(fig, word_types: Sequence[str], *, y: float = 0.01):
    """Place the single house word-type legend (marker swatches, canonical order)
    at the bottom centre of ``fig``. No-op if no known types are present.
    """
    handles = legend_handles(word_types)
    if not handles:
        return None
    return fig.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        bbox_to_anchor=(0.5, y), title="Word type",
        title_fontsize=FS["legend_title"], fontsize=FS["legend"],
    )


def footnote(fig, text: str, *, y: float = 0.005) -> None:
    """Add a single light-grey, receding footnote at the bottom of ``fig``."""
    fig.text(0.5, y, text, ha="center", va="bottom",
             fontsize=FS["footnote"], color=FOOTNOTE_COLOR)


# === Layer helpers =======================================================
def depth_label(layer: int, num_layers: int) -> str:
    """Human-readable depth band for a layer index (``num_layers`` = max layer)."""
    if num_layers <= 0:
        return f"Layer {layer}"
    if layer == 0:
        return "Embeddings"
    if layer == num_layers:
        return "Final"
    frac = layer / num_layers
    if frac <= 0.25:
        return "Early"
    if frac <= 0.5:
        return "Mid"
    if frac <= 0.75:
        return "Mid-Deep"
    return "Deep"


def select_layers(all_layers: Sequence[int], k: int = 6) -> List[int]:
    """Pick ``k`` layers spread evenly across depth, including first and last."""
    layers = sorted(set(int(x) for x in all_layers))
    if not layers:
        return []
    if len(layers) <= k:
        return layers
    idx = [round(i * (len(layers) - 1) / (k - 1)) for i in range(k)]
    return sorted({layers[i] for i in idx})


# === Export (one path: vector PDF + PNG fallback, deterministic names) ===
def save_figure(
    fig,
    path_base: str,
    formats: Tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> List[str]:
    """Save ``fig`` as ``{path_base}.{ext}`` for each format (PDF first, vector).

    ``path_base`` has no extension; parent directories are created as needed.
    Filenames are deterministic (caller supplies the stem).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_base)), exist_ok=True)
    written: List[str] = []
    for ext in formats:
        out = f"{path_base}.{ext}"
        fig.savefig(out, dpi=dpi, facecolor="white")
        written.append(out)
    return written
