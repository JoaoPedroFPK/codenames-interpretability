"""Word x word cosine-similarity heatmap at a fixed board and layer.

The similarity matrix is symmetric, so only the lower triangle is shown; the
self-similarity diagonal and the resulting empty first row / last column are
trimmed. The ``no_social`` and ``with_social`` conditions are rendered as
SEPARATE single-panel figures (one file per condition, so each fits a report
page) but remain directly comparable: both are prepared together, sharing one
canonical word ordering (the social figure appends the giver's demographic
feature words at the end) and one 0..vmax colour scale. A sequential Reds
colormap (darker = higher cosine) and per-cell values match the thesis
example figures.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .style import (
    FS,
    add_word_type_legend,
    apply_publication_style,
    fig_header,
    label_color,
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


def prepare_conditions(panels: Dict[str, Dict]) -> Tuple[Dict[str, Optional[Dict]], float]:
    """Shared preparation for BOTH conditions' heatmaps of one (board, layer).

    Builds one canonical word sequence over the union of words: type blocks
    (hint < target < assassin < neutral), alphabetical within, giver features
    last. The shared board words therefore line up identically in both
    figures; the extra giver-feature rows/cols only append at the end of the
    with-social one. Also computes the single 0..vmax colour scale, so the two
    separately-saved figures remain directly comparable cell by cell.

    Returns ``(prepared, vmax)`` where ``prepared`` maps mode ->
    ``{"words", "word_types", "matrix"}`` (or ``None`` if that mode lacks data).
    """
    modes = ["no_social", "with_social"]
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

    prepared: Dict[str, Optional[Dict]] = {}
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

    return prepared, _max_offdiag(matrices)


def plot_heatmap_condition(
    panels: Dict[str, Dict],
    mode: str,
    *,
    layer: int,
    annotate_max_words: int = 40,
    label_map: Optional[Dict[str, str]] = None,
    header: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> Tuple[Optional["object"], Dict]:
    """Draw ONE condition's heatmap for one layer as a standalone figure.

    ``panels`` maps mode -> ``{"words", "word_types", "vectors"}`` and should
    carry BOTH conditions when available — the word ordering and colour scale
    are computed jointly (see :func:`prepare_conditions`) so the per-condition
    files stay comparable. Returns ``(figure, info)``; the figure is ``None``
    when ``mode`` has no usable data.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_publication_style()
    lmap = label_map or {}

    prepared, data_vmax = prepare_conditions(panels)
    prep = prepared.get(mode)
    if prep is None:
        return None, {"layer": layer, "vmax": 1.0, "data_vmax": data_vmax}
    # FIXED 0..1 cosine scale (NOT a per-figure adaptive max): every heatmap —
    # across layers, conditions and models — shares one colour scale, so
    # absolute cosine levels are directly comparable between figures (e.g. the
    # rising anisotropy that uniformly darkens deep layers). ``data_vmax``
    # (the actual off-diagonal max) is still returned for reference.
    vmin, vmax = 0.0, 1.0

    mat = prep["matrix"]
    words = prep["words"]
    types = prep["word_types"]
    n = len(words)

    # Size the figure to the word count so cells (and their numbers) stay
    # legible (a dense ~20x20 matrix is naturally larger than text width).
    # The lower-triangle matrix leaves its upper-right half empty, so the
    # colorbar and legend live INSIDE that dead zone instead of widening the
    # canvas with an extra margin column / bottom strip.
    panel_w = max(3.0, n * 0.32)            # inches for the square matrix
    margins = dict(left=0.13, right=0.97, top=0.90, bottom=0.16)
    fig_w = panel_w / (margins["right"] - margins["left"])
    fig_h = panel_w / (margins["top"] - margins["bottom"])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(**margins)
    # The legend (top right) and a horizontal colorbar right beneath it form
    # ONE right-aligned block inside the empty triangle, instead of elements
    # floating at unrelated positions. At this x/y the matrix's diagonal stays
    # below/left of the bar for any n (positions are figure-relative, so they
    # hold across word counts; with_social's 5-row legend still clears it).
    cbar_ax = fig.add_axes([0.63, 0.655, 0.33, 0.025])

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
        cbar=True, cbar_ax=cbar_ax,
        cbar_kws={"label": "Cosine similarity", "orientation": "horizontal"},
    )
    # Quiet the colorbar chrome: no outline box, no tick dashes.
    cbar = ax.collections[0].colorbar
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=FS["tick_label"])
    # Contrast-aware annotation colour: white text on the darkest cells.
    if annot:
        for t in ax.texts:
            try:
                val = float(t.get_text())
            except ValueError:
                continue
            t.set_color("white" if val >= 0.62 * vmax else "#333333")
    # Same header mechanism as the projection figure: bold human title (model
    # + condition, no word counts or debug-ish text) over a smaller grey
    # provenance line carrying the board id and layer.
    nice = "No social context" if mode == "no_social" else "With social context"
    title = f"{header} — {nice}" if header else nice
    fig_header(fig, title, subtitle)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=FS["tick_label"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FS["tick_label"])
    ax.tick_params(length=0)   # word labels only — no tick dashes
    # Colour the tick labels by word type (y uses words[1:], x uses words[:-1]);
    # text uses the darkened neutral hue so beige labels stay readable.
    for lbl, wt in zip(ax.get_xticklabels(), x_types):
        lbl.set_color(label_color(wt))
    for lbl, wt in zip(ax.get_yticklabels(), y_types):
        lbl.set_color(label_color(wt))

    # Word-type legend in the empty triangle's top-right corner (the compact
    # framed variant), next to the colorbar — no bottom strip. The methodology
    # note (lower triangle, Reds, matched ordering) stays in the figure caption.
    add_word_type_legend(fig, set(types), corner=True)
    info = {"layer": layer, "vmax": vmax, "data_vmax": data_vmax, "mode": mode}
    return fig, info
