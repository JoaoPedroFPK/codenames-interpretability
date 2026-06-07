"""Word x word cosine-similarity heatmap at a fixed board and layer.

The similarity matrix is symmetric, so only the lower triangle is shown; the
self-similarity diagonal and the resulting empty first row / last column are
trimmed. The ``no_social`` and ``with_social`` conditions are drawn side by side
with identical word ordering (the social panel appends the giver's demographic
feature words). A sequential Reds colormap (darker = higher cosine) and per-cell
values match the thesis example figures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .style import (
    FS,
    add_word_type_legend,
    apply_publication_style,
    footnote,
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


def _max_offdiag(matrices: Sequence[np.ndarray]) -> float:
    """Largest off-diagonal cosine across panels (>=0.1), for the Reds vmax."""
    vmax = 0.0
    for m in matrices:
        if m.size == 0:
            continue
        off = m.copy()
        np.fill_diagonal(off, -np.inf)
        vmax = max(vmax, float(np.nanmax(off)))
    return max(vmax, 0.1)


def plot_heatmap_pair(
    panels: Dict[str, Dict],
    *,
    layer: int,
    title: str,
    annotate_max_words: int = 40,
    label_map: Optional[Dict[str, str]] = None,
) -> Tuple["object", Dict]:
    """Draw the ``no_social`` / ``with_social`` heatmap pair for one layer.

    ``panels`` maps mode -> ``{"words", "word_types", "vectors"}``. Modes with
    no data are rendered as an empty annotated panel. Returns ``(figure, info)``.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_publication_style()
    lmap = label_map or {}

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

    # Sequential Reds: cream (low) -> dark red (high), shared 0..vmax scale so the
    # two panels are directly comparable.
    vmax = _max_offdiag(matrices)
    vmin = 0.0

    # Size the figure to the word count so cells (and their numbers) stay legible
    # (a dense ~20x20 matrix is naturally a wide figure, not a text-width one).
    n_max = max((len(prepared[m]["words"]) for m in modes if prepared[m]), default=10)
    panel_w = max(3.0, n_max * 0.32)        # inches per square panel
    fig_w = 2 * panel_w + 2.4               # + axis labels, gap, colorbar
    fig_h = panel_w + 1.7                    # + suptitle, legend, footnote
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
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

        # Strictly-lower-triangle layout. The first word has no cells on its row
        # and the last word has none in its column, so they would dangle with an
        # empty axis tick. Trim them: show rows = words[1:], cols = words[:-1].
        sub = mat[1:, :n - 1]
        sub_mask = np.triu(np.ones_like(sub, dtype=bool), k=1)
        y_words, y_types = words[1:], types[1:]
        x_words, x_types = words[:-1], types[:-1]
        annot = (n - 1) <= annotate_max_words

        sns.heatmap(
            sub, mask=sub_mask, ax=ax, cmap="Reds",
            vmin=vmin, vmax=vmax, square=True,
            annot=annot, fmt=".2f", annot_kws={"size": FS["annot"]},
            linewidths=0.5, linecolor="white",
            xticklabels=[lmap.get(w, w) for w in x_words],
            yticklabels=[lmap.get(w, w) for w in y_words],
            cbar=(ax_i == 0), cbar_ax=(cbar_ax if ax_i == 0 else None),
            cbar_kws={"label": "Cosine similarity"},
        )
        # Contrast-aware annotation colour: white text on the darkest cells.
        if annot:
            for t in ax.texts:
                try:
                    val = float(t.get_text())
                except ValueError:
                    continue
                t.set_color("white" if val >= 0.62 * vmax else "#333333")
        ax.set_title(f"{nice}  (n={n} words)", fontsize=FS["panel_title"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=FS["tick_label"])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FS["tick_label"])
        # Colour the tick labels by word type (y uses words[1:], x uses words[:-1]).
        for lbl, wt in zip(ax.get_xticklabels(), x_types):
            lbl.set_color(style_for(wt)["color"])
        for lbl, wt in zip(ax.get_yticklabels(), y_types):
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
        "Lower triangle only (matrix is symmetric; the self-similarity diagonal "
        "and the empty first row / last column are trimmed). Sequential Reds: "
        "darker = higher cosine. Both panels share an identical word ordering, so "
        "shared cells diff directly; giver-feature rows/cols append only in the "
        "with-social panel.",
        y=0.11,
    )
    fig.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.24, wspace=0.25)
    info = {"layer": layer, "vmax": vmax}
    return fig, info
