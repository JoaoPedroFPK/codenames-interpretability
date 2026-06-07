"""Word x word cosine-similarity heatmap at a fixed board and layer.

The similarity matrix is symmetric, so only the lower triangle is shown (the
diagonal — trivially 1 — is masked too). The ``no_social`` and ``with_social``
conditions are drawn side by side; the social panel additionally contains the
giver's demographic feature words. A colorblind-safe diverging colormap centred
at cosine = 0 makes positive vs negative association directly readable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .style import apply_publication_style, style_for

# Canonical block ordering so related word types cluster on the axes.
_TYPE_ORDER = {"hint": 0, "target": 1, "black": 2, "tan": 3, "giver_feature": 4}


def order_words(words: Sequence[str], word_types: Sequence[str]) -> List[int]:
    """Return row indices ordered by word-type block, then alphabetically."""
    idx = list(range(len(words)))
    idx.sort(key=lambda i: (_TYPE_ORDER.get(word_types[i], 99), words[i].lower()))
    return idx


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Symmetric word x word cosine-similarity matrix (L2-normalised)."""
    Xn = vectors.astype(np.float64)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xn / norms
    return np.clip(Xn @ Xn.T, -1.0, 1.0)


def _symmetric_limit(matrices: Sequence[np.ndarray]) -> float:
    """Largest absolute off-diagonal value across panels (>=0.1), for vmin/vmax."""
    vmax = 0.0
    for m in matrices:
        if m.size == 0:
            continue
        off = m.copy()
        np.fill_diagonal(off, 0.0)
        vmax = max(vmax, float(np.abs(off).max()))
    return max(vmax, 0.1)


def plot_heatmap_pair(
    panels: Dict[str, Dict],
    *,
    layer: int,
    title: str,
    annotate_max_words: int = 28,
) -> Tuple["object", Dict]:
    """Draw the ``no_social`` / ``with_social`` heatmap pair for one layer.

    ``panels`` maps mode -> ``{"words", "word_types", "vectors"}``. Modes with
    no data are rendered as an empty annotated panel. Returns ``(figure, info)``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_publication_style()

    modes = ["no_social", "with_social"]
    prepared: Dict[str, Dict] = {}
    matrices: List[np.ndarray] = []
    for mode in modes:
        p = panels.get(mode)
        if not p or len(p.get("words", [])) < 2:
            prepared[mode] = None
            continue
        order = order_words(p["words"], p["word_types"])
        words = [p["words"][i] for i in order]
        types = [p["word_types"][i] for i in order]
        mat = cosine_matrix(p["vectors"][order])
        prepared[mode] = {"words": words, "word_types": types, "matrix": mat}
        matrices.append(mat)

    vmax = _symmetric_limit(matrices)
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.64])

    for ax_i, mode in enumerate(modes):
        ax = axes[ax_i]
        prep = prepared[mode]
        nice = "No social" if mode == "no_social" else "With social"
        if prep is None:
            ax.text(0.5, 0.5, f"{nice}: no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=9)
            ax.axis("off")
            continue

        mat = prep["matrix"]
        words = prep["words"]
        types = prep["word_types"]
        n = len(words)
        # Mask upper triangle incl. diagonal -> strictly lower triangle shown.
        mask = np.triu(np.ones_like(mat, dtype=bool), k=0)
        annot = n <= annotate_max_words

        sns.heatmap(
            mat, mask=mask, ax=ax, cmap="RdBu_r", center=0.0,
            vmin=vmin, vmax=vmax, square=True,
            annot=annot, fmt=".2f", annot_kws={"size": 4.5},
            linewidths=0.4, linecolor="white",
            xticklabels=words, yticklabels=words,
            cbar=(ax_i == 0), cbar_ax=(cbar_ax if ax_i == 0 else None),
            cbar_kws={"label": "Cosine similarity"},
        )
        ax.set_title(f"{nice}  (n={n} words)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5.5)
        # Colour the tick labels by word type.
        for lbl, wt in zip(ax.get_xticklabels(), types):
            lbl.set_color(style_for(wt)["color"])
        for lbl, wt in zip(ax.get_yticklabels(), types):
            lbl.set_color(style_for(wt)["color"])

    fig.suptitle(title, y=0.99, fontsize=13, fontweight="bold")

    # Word-type legend (colours match the axis-label colours), shown once.
    types_present = set()
    for mode in modes:
        if prepared[mode]:
            types_present.update(prepared[mode]["word_types"])
    handles = _legend_handles(types_present)
    if handles:
        fig.legend(
            handles=handles, loc="lower center", ncol=len(handles),
            bbox_to_anchor=(0.5, 0.005), title="Word type (axis-label colour)",
            title_fontsize=8, fontsize=7,
        )
    fig.text(
        0.5, 0.065,
        "Lower triangle only (matrix is symmetric; the unit diagonal is omitted). "
        "Colorblind-safe diverging map centred at cosine = 0.",
        ha="center", va="bottom", fontsize=6.5, color="#666666",
    )
    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.16, wspace=0.25)
    info = {"layer": layer, "vmax": vmax}
    return fig, info


def _legend_handles(types_present):
    """Proxy legend handles (coloured squares) for word types in canonical order."""
    from matplotlib.patches import Patch
    present = [t for t in _TYPE_ORDER if t in set(types_present)]
    present.sort(key=lambda t: _TYPE_ORDER[t])
    return [Patch(facecolor=style_for(t)["color"], edgecolor="none",
                  label=style_for(t)["label"]) for t in present]
