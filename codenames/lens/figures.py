"""Overlay figure: lens bracket vs cosine geometry vs generation.

Two panels sharing the layer axis. Top: solid line = tuned lens (with
bootstrap CI band), dotted = raw lens, grey dashed = the thesis geometric
top-1 curve g(l) with its two humps marked, horizontal line = generation
accuracy, optional thin dashed line = the random-init null. Bottom:
per-layer divergence L(l) - g(l) for both lenses with the bracket between
them filled, and sign-consistent dissociation regions shaded ("geometry
without decodability" / "decodability without geometry"). The bottom panel
is skipped when the model has no geometric curve. Colours follow
codenames.analysis.figures.MODEL_STYLE so the paper reads as one system.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analysis.figures import MODEL_STYLE, _FALLBACK_STYLE
from ..viz.style import apply_publication_style, save_figure

_GEO_COLOR = "#999999"
_BELOW_TINT = "#e9dfd5"   # geometry without decodability (early hump)
_ABOVE_TINT = "#d8e6dd"   # decodability without geometry (late stack)


def _local_maxima(y: np.ndarray) -> np.ndarray:
    """Indices of strict-or-plateau local maxima by neighbour comparison."""
    idx = []
    for i in range(len(y)):
        left = y[i - 1] if i > 0 else -np.inf
        right = y[i + 1] if i < len(y) - 1 else -np.inf
        if y[i] >= left and y[i] >= right:
            idx.append(i)
    return np.asarray(idx, dtype=int)


def _runs(mask: np.ndarray, min_run: int) -> List[Tuple[int, int]]:
    """Inclusive (start, end) index runs of True at least min_run long."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_run:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(mask) - start >= min_run:
        runs.append((start, len(mask) - 1))
    return runs


def _dissociation_regions(diff_raw: np.ndarray, diff_tuned: np.ndarray,
                          min_run: int = 3):
    """Layer-index runs where BOTH lenses sit on the same side of g(l).

    Returns (below, above): below = geometry without decodability,
    above = decodability without geometry. Runs shorter than min_run are
    dropped; layers where the bracket straddles zero belong to neither.
    """
    below = _runs((np.maximum(diff_raw, diff_tuned) < 0), min_run)
    above = _runs((np.minimum(diff_raw, diff_tuned) > 0), min_run)
    return below, above


def build_lens_overlay_figure(
    curves: pd.DataFrame,
    model_key: str,
    concordance_csv: str,
    generation_acc: float,
    condition: str = "no_social",
    pooling: str = "mean",
    random_curves: Optional[pd.DataFrame] = None,
):
    """Build the (1- or 2-panel) overlay figure and return it unsaved."""
    import matplotlib.pyplot as plt

    apply_publication_style()
    style = MODEL_STYLE.get(model_key, _FALLBACK_STYLE)
    color = style["color"]

    g = pd.read_csv(concordance_csv)
    g = g[(g["model"] == model_key) & (g["condition"] == condition)
          & (g["pooling"] == pooling)].sort_values("layer")

    if len(g):
        fig, (ax, ax_d) = plt.subplots(
            2, 1, figsize=(5.5, 4.6), sharex=True,
            gridspec_kw={"height_ratios": [1.0, 0.6]})
    else:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax_d = None

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

    ax.set_ylabel("Candidate-restricted top-1 accuracy")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{style.get('label', model_key)} — vocabulary channel "
                 f"vs cosine geometry ({condition})", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    if ax_d is not None:
        geo = g.set_index("layer")["top1_accuracy"]
        diffs = {}
        for lens in ("raw", "tuned"):
            c = curves[curves["lens"] == lens].sort_values("layer")
            if len(c):
                d = c.set_index("layer")["top1"] - geo
                diffs[lens] = d.dropna()
        common = None
        for d in diffs.values():
            common = d.index if common is None else common.intersection(d.index)
        if diffs and len(common):
            layers = common.to_numpy()
            d_raw = diffs.get("raw", diffs.get("tuned")).loc[common].to_numpy()
            d_tuned = diffs.get("tuned", diffs.get("raw")).loc[common].to_numpy()

            below, above = _dissociation_regions(d_raw, d_tuned)
            for runs, tint, label, va, ypos in (
                    (below, _BELOW_TINT, "geometry without\ndecodability",
                     "bottom", 0.05),
                    (above, _ABOVE_TINT, "decodability without\ngeometry",
                     "top", 0.95)):
                for start, end in runs:
                    ax_d.axvspan(layers[start] - 0.5, layers[end] + 0.5,
                                 color=tint, alpha=0.6, linewidth=0)
                if runs:
                    start, end = max(runs, key=lambda r: r[1] - r[0])
                    ax_d.text((layers[start] + layers[end]) / 2.0, ypos, label,
                              transform=ax_d.get_xaxis_transform(),
                              ha="center", va=va, fontsize=6.5,
                              color="#666666")

            ax_d.fill_between(layers, d_raw, d_tuned, color=color,
                              alpha=0.15, linewidth=0)
            ax_d.plot(layers, d_tuned, "-", color=color, linewidth=1.0)
            ax_d.plot(layers, d_raw, ":", color=color, linewidth=1.0)
            ax_d.axhline(0.0, color=_GEO_COLOR, linewidth=0.8)
        ax_d.set_ylabel("$L(\\ell) - g(\\ell)$", fontsize=8)
        ax_d.set_xlabel("Layer")
    else:
        ax.set_xlabel("Layer")

    fig.tight_layout()
    return fig


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

    fig = build_lens_overlay_figure(
        curves, model_key, concordance_csv, generation_acc,
        condition=condition, pooling=pooling, random_curves=random_curves)
    base, ext = os.path.splitext(out_path)
    save_figure(fig, base, formats=("pdf", "png"))
    plt.close(fig)
    return out_path if ext == ".png" else base + ".png"
