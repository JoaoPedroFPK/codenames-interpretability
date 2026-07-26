"""Overlay figure: lens bracket vs cosine geometry vs generation.

One axes per model. Solid line = tuned lens (with bootstrap CI band),
dotted = raw lens, grey dashed = the thesis geometric top-1 curve g(l)
with its two humps marked, horizontal line = generation accuracy, optional
thin dashed line = the random-init null. Colours follow
codenames.analysis.figures.MODEL_STYLE so the paper reads as one system.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd

from ..analysis.figures import MODEL_STYLE, _FALLBACK_STYLE
from ..viz.style import apply_publication_style, save_figure

_GEO_COLOR = "#999999"


def _local_maxima(y: np.ndarray) -> np.ndarray:
    """Indices of strict-or-plateau local maxima by neighbour comparison."""
    idx = []
    for i in range(len(y)):
        left = y[i - 1] if i > 0 else -np.inf
        right = y[i + 1] if i < len(y) - 1 else -np.inf
        if y[i] >= left and y[i] >= right:
            idx.append(i)
    return np.asarray(idx, dtype=int)


def lens_overlay_figure(
    curves: pd.DataFrame,
    model_key: str,
    concordance_csv: str,
    generation_acc: float,
    out_path: str,
    condition: str = "no_social",
    pooling: str = "mean",
    random_curves: Optional[pd.DataFrame] = None,
) -> str:
    import matplotlib.pyplot as plt

    apply_publication_style()
    style = MODEL_STYLE.get(model_key, _FALLBACK_STYLE)
    color = style["color"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for lens, ls, alpha in (("tuned", "-", 1.0), ("raw", ":", 0.9)):
        c = curves[curves["lens"] == lens].sort_values("layer")
        if not len(c):
            continue
        ax.plot(c["layer"], c["top1"], ls, color=color, alpha=alpha,
                marker=style.get("marker", "o"), markersize=3,
                label=f"{lens.capitalize()} lens")
        if lens == "tuned":
            ax.fill_between(c["layer"], c["ci_lo"], c["ci_hi"],
                            color=color, alpha=0.15, linewidth=0)

    g = pd.read_csv(concordance_csv)
    g = g[(g["model"] == model_key) & (g["condition"] == condition)
          & (g["pooling"] == pooling)].sort_values("layer")
    if len(g):
        y = g["top1_accuracy"].to_numpy()
        ax.plot(g["layer"], y, "--", color=_GEO_COLOR,
                label="Cosine geometry $g(\\ell)$")
        peaks = _local_maxima(y)
        if len(peaks) > 2:  # keep the two dominant humps only
            peaks = peaks[np.argsort(y[peaks])[-2:]]
        ax.plot(g["layer"].to_numpy()[peaks], y[peaks], "^",
                color=_GEO_COLOR, markersize=5, linestyle="none")

    if random_curves is not None and len(random_curves):
        r = random_curves.sort_values("layer")
        lens_r = "tuned" if (r["lens"] == "tuned").any() else r["lens"].iloc[0]
        r = r[r["lens"] == lens_r]
        ax.plot(r["layer"], r["top1"], "--", color=color, linewidth=0.8,
                alpha=0.6, label="Random-init null")

    ax.axhline(generation_acc, color=color, linewidth=0.8, alpha=0.5,
               linestyle=(0, (1, 1)))
    ax.text(0.99, generation_acc, "generation", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=color, alpha=0.8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Candidate-restricted top-1 accuracy")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{style.get('label', model_key)} — vocabulary channel "
                 f"vs cosine geometry ({condition})", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()

    base, ext = os.path.splitext(out_path)
    save_figure(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    return out_path if ext == ".png" else base + ".png"
