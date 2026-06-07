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

from .style import (
    FS,
    add_word_type_legend,
    apply_publication_style,
    footnote,
    grid_size,
    style_for,
)

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

    # --- Single shared ordering across BOTH panels (for cell-by-cell diffing) ---
    # Build one canonical word sequence over the union of words: type blocks
    # (hint < target < assassin < neutral), alphabetical within, with giver
    # features last. The shared board words therefore line up identically in both
    # panels; the extra giver-feature rows/cols only append at the end of the
    # with-social panel (it has them, no-social does not).
    word_type_map: Dict[str, str] = {}
    for mode in modes:
        p = panels.get(mode)
        if p:
            for w, t in zip(p["words"], p["word_types"]):
                word_type_map.setdefault(w, t)
    global_order = sorted(
        word_type_map,
        key=lambda w: (_TYPE_ORDER.get(word_type_map[w], 99), w.lower()),
    )

    prepared: Dict[str, Dict] = {}
    matrices: List[np.ndarray] = []
    for mode in modes:
        p = panels.get(mode)
        if not p or len(p.get("words", [])) < 2:
            prepared[mode] = None
            continue
        present = set(p["words"])
        words = [w for w in global_order if w in present]
        idx_of = {w: i for i, w in enumerate(p["words"])}
        sel = [idx_of[w] for w in words]
        types = [word_type_map[w] for w in words]
        mat = cosine_matrix(p["vectors"][sel])
        prepared[mode] = {"words": words, "word_types": types, "matrix": mat}
        matrices.append(mat)

    vmax = _symmetric_limit(matrices)
    vmin = -vmax

    fig, axes = plt.subplots(
        1, 2, figsize=grid_size(2, 1, panel_aspect=1.0, footer_in=1.2),
    )
    cbar_ax = fig.add_axes([0.92, 0.30, 0.015, 0.52])

    for ax_i, mode in enumerate(modes):
        ax = axes[ax_i]
        prep = prepared[mode]
        nice = "No social" if mode == "no_social" else "With social"
        if prep is None:
            ax.text(0.5, 0.5, f"{nice}: no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey", fontsize=FS["panel_title"])
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
            annot=annot, fmt=".2f", annot_kws={"size": FS["annot"]},
            linewidths=0.4, linecolor="white",
            xticklabels=words, yticklabels=words,
            cbar=(ax_i == 0), cbar_ax=(cbar_ax if ax_i == 0 else None),
            cbar_kws={"label": "Cosine similarity"},
        )
        ax.set_title(f"{nice}  (n={n} words)", fontsize=FS["panel_title"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=FS["tick_label"])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FS["tick_label"])
        # Colour the tick labels by word type.
        for lbl, wt in zip(ax.get_xticklabels(), types):
            lbl.set_color(style_for(wt)["color"])
        for lbl, wt in zip(ax.get_yticklabels(), types):
            lbl.set_color(style_for(wt)["color"])

    fig.suptitle(title, y=0.99, fontsize=FS["suptitle"], fontweight="bold")

    # Single house legend (word-type taxonomy; colours match the axis labels).
    types_present = set()
    for mode in modes:
        if prepared[mode]:
            types_present.update(prepared[mode]["word_types"])
    add_word_type_legend(fig, types_present, y=0.02)
    footnote(
        fig,
        "Lower triangle only (matrix is symmetric; the unit diagonal is omitted). "
        "Colorblind-safe diverging map centred at cosine = 0. Both panels share an "
        "identical word ordering, so shared cells can be diffed directly; the "
        "giver-feature rows/cols append only in the with-social panel.",
        y=0.11,
    )
    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.24, wspace=0.25)
    info = {"layer": layer, "vmax": vmax}
    return fig, info
