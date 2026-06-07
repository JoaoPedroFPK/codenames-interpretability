"""Publication styling for the visualization figures.

Self-contained on purpose: it *follows* the conventions of the bundled
``scientific-visualization`` skill (sans-serif fonts, despined axes, the
Okabe-Ito colorblind-safe categorical palette, multi-format export at 300 DPI)
but does not import from the skill checkout, so the package works anywhere.

The module also owns the canonical word-type -> colour/marker mapping shared by
both the heatmap and the projection figures, so the two figure families stay
visually consistent.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib

# Batch / headless rendering — these figures are produced by a CLI pipeline,
# never in an interactive session. Set before pyplot is imported anywhere.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# --- Okabe-Ito colorblind-safe palette ------------------------------------
OKABE_ITO: Dict[str, str] = {
    "orange":        "#E69F00",
    "sky_blue":      "#56B4E9",
    "bluish_green":  "#009E73",
    "yellow":        "#F0E442",
    "blue":          "#0072B2",
    "vermillion":    "#D55E00",
    "reddish_purple": "#CC79A7",
    "black":         "#000000",
}


# --- Canonical word-type styling ------------------------------------------
# Keyed by the ``word_type`` values stored in the vector index. ``order`` fixes
# the legend ordering and z-order (higher draws on top).
WORD_TYPE_STYLE: Dict[str, Dict] = {
    "hint":          {"label": "Hint",         "color": OKABE_ITO["vermillion"],   "marker": "D", "order": 5},
    "target":        {"label": "Target",       "color": OKABE_ITO["bluish_green"], "marker": "o", "order": 4},
    "black":         {"label": "Assassin",     "color": OKABE_ITO["black"],        "marker": "X", "order": 3},
    "tan":           {"label": "Neutral",      "color": OKABE_ITO["orange"],       "marker": "o", "order": 1},
    "giver_feature": {"label": "Giver feature", "color": OKABE_ITO["sky_blue"],    "marker": "^", "order": 2},
}

# Fallback for any unexpected word_type so the pipeline never crashes on a new
# label; renders as a neutral grey circle.
_UNKNOWN_STYLE = {"label": "Other", "color": "#999999", "marker": "o", "order": 0}


def style_for(word_type: str) -> Dict:
    """Return the plotting style dict for a ``word_type`` (never raises)."""
    return WORD_TYPE_STYLE.get(word_type, _UNKNOWN_STYLE)


def apply_publication_style() -> None:
    """Apply publication-quality matplotlib rcParams (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.titlesize": 11,
        "axes.grid": False,
        "pdf.fonttype": 42,   # embed TrueType (editable text in the PDF)
        "ps.fonttype": 42,
    })


def depth_label(layer: int, num_layers: int) -> str:
    """Human-readable depth band for a layer index (``num_layers`` = max layer).

    Layer 0 is the embedding layer; ``num_layers`` is the final hidden layer.
    """
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
    """Pick ``k`` representative layers spread across depth, always including
    the first and last available layer. Returns a sorted, de-duplicated list.
    """
    layers = sorted(set(int(x) for x in all_layers))
    if not layers:
        return []
    if len(layers) <= k:
        return layers
    # Even spacing across the index range, inclusive of both endpoints.
    idx = [round(i * (len(layers) - 1) / (k - 1)) for i in range(k)]
    chosen = sorted({layers[i] for i in idx})
    return chosen


def save_figure(
    fig,
    path_base: str,
    formats: Tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> List[str]:
    """Save ``fig`` as ``{path_base}.{ext}`` for each format. Returns paths.

    ``path_base`` has no extension. Parent directories are created as needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path_base)), exist_ok=True)
    written: List[str] = []
    for ext in formats:
        out = f"{path_base}.{ext}"
        fig.savefig(out, dpi=dpi, facecolor="white")
        written.append(out)
    return written


def legend_handles(word_types: Sequence[str]):
    """Build matplotlib proxy legend handles for the given word types, ordered
    by the canonical ``order`` field. Only types present are shown.
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
