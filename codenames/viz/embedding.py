"""Cosine-aware 2D projection of word vectors and the per-board panel figure.

Three reducers are run for every (board, layer): UMAP and t-SNE with
``metric="cosine"``, and PCA on L2-normalised vectors (where Euclidean geometry
coincides with cosine geometry). The reducer that best preserves the original
cosine neighbourhood structure — scored by :mod:`codenames.viz.metrics` — is
selected per layer, and its score is annotated on the panel so the figure is
self-documenting and defensible.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import metrics as _m
from .style import (
    FS,
    apply_publication_style,
    depth_label,
)

METHODS = ("umap", "tsne", "pca")

# The reducer used to render the projection. Standardised on UMAP(cosine) for a
# consistent picture across boards/layers. All three reducers (UMAP, t-SNE, PCA)
# are still fit and scored per panel for the dr_quality_*.csv audit, so the
# trustworthiness of the rendered UMAP layout can always be checked.
PREFERRED_METHOD = "umap"

# === Reference aesthetic (distance_map_2d_*.png) =========================
# Warm paper background, serif type, visible light-grey spines + ticks,
# saturated category colours, hint -> target arrows, framed corner legend and
# an italic disclaimer footnote. This deliberately diverges from the house
# sans-serif style in style.py; it is applied ONLY to the projection figure
# (via rc_context in plot_layer_panels) so the heatmap is unaffected.
_BG_COLOR = "#FAFAF8"
_SPINE_COLOR = "#CCCCCC"
_LEADER_COLOR = "#999999"
_FOOTNOTE_COLOR = "#888888"

_REF_COLOR = {
    "hint":          "#C0392B",
    "target":        "#1A6B5A",
    "black":         "#1C1C1C",
    "tan":           "#C8A86B",
    "giver_feature": "#6E8FA6",
}
_REF_MARKER = {
    "hint": "D", "target": "^", "black": "X", "tan": "s", "giver_feature": "P",
}
_REF_SIZE = {
    "hint": 140, "target": 100, "black": 60, "tan": 55, "giver_feature": 70,
}
_REF_ALPHA = {
    "hint": 1.0, "target": 1.0, "black": 0.8, "tan": 0.7, "giver_feature": 0.9,
}
_REF_LEGEND_LABEL = {
    "hint": "Hint", "target": "Target [T]", "black": "Assassin",
    "tan": "Neutral", "giver_feature": "Giver feature",
}
# Legend / z-order priority (drawn high to low), matching the reference.
_REF_ORDER = ["hint", "target", "black", "tan", "giver_feature"]

# In-panel font sizes (pt at the reference render size).
_REF_FS = {"title": 10, "axis": 8, "tick": 7, "label": 8, "suptitle": 12,
           "legend": 8, "footnote": 7}

_REF_DISCLAIMER = (
    "Coordinates produced by UMAP projection (n_components=2, metric=cosine, "
    "random_state=42). No quantitative metric is derived from projected "
    "coordinates; all reported results use the original high-dimensional space."
)

_REF_RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "serif"],
    "figure.facecolor": _BG_COLOR,
    "axes.facecolor": _BG_COLOR,
    "savefig.facecolor": _BG_COLOR,
    "axes.spines.top": True,
    "axes.spines.right": True,
}


def _ref_color(wt: str) -> str:
    return _REF_COLOR.get(wt, _LEADER_COLOR)


def _normalize(X: np.ndarray) -> np.ndarray:
    Xn = X.astype(np.float64)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Xn / norms


def reduce(vectors: np.ndarray, method: str, seed: int = 42) -> np.ndarray:
    """Project ``vectors`` [N, D] to [N, 2] with a cosine-aware reducer."""
    n = vectors.shape[0]
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(_normalize(vectors))
    if method == "tsne":
        from sklearn.manifold import TSNE
        # perplexity must be < n; keep it modest for the small per-board sets.
        perplexity = max(2.0, min(30.0, (n - 1) / 3.0))
        tsne = TSNE(
            n_components=2, metric="cosine", init="pca",
            perplexity=perplexity, random_state=seed,
        )
        return tsne.fit_transform(_normalize(vectors))
    if method == "umap":
        import umap
        n_neighbors = max(2, min(15, n - 1))
        reducer = umap.UMAP(
            n_components=2, metric="cosine", n_neighbors=n_neighbors,
            min_dist=0.1, random_state=seed,
        )
        return reducer.fit_transform(_normalize(vectors))
    raise ValueError(f"Unknown reduction method: {method!r}")


def _combined_score(scores: Dict[str, float]) -> float:
    """Single comparable score: mean of trustworthiness, continuity, and the
    non-negative part of the Shepard correlation. NaNs count as 0.
    """
    t = scores.get("trustworthiness", float("nan"))
    c = scores.get("continuity", float("nan"))
    s = scores.get("shepard", float("nan"))
    vals = [
        0.0 if np.isnan(t) else t,
        0.0 if np.isnan(c) else c,
        0.0 if np.isnan(s) else max(0.0, s),
    ]
    return float(np.mean(vals))


def embed_all(
    vectors: np.ndarray, k: int = 5, seed: int = 42,
    methods: Sequence[str] = METHODS,
) -> Dict:
    """Fit every reducer once, scoring each. Returns ``{"results", "table"}``.

    ``results`` maps method -> ``{embedding, scores, combined}`` (only successful
    reducers); ``table`` is one row per attempted method for the CSV audit.
    """
    results: Dict[str, Dict] = {}
    table: List[Dict] = []
    for method in methods:
        try:
            emb = reduce(vectors, method, seed=seed)
            sc = _m.all_metrics(vectors, emb, k=k)
            combined = _combined_score(sc)
        except Exception as exc:  # a reducer can fail on degenerate inputs
            table.append({"method": method, "trustworthiness": float("nan"),
                          "continuity": float("nan"), "shepard": float("nan"),
                          "combined": float("nan"), "error": str(exc)})
            continue
        results[method] = {"embedding": emb, "scores": sc, "combined": combined}
        table.append({"method": method, **sc, "combined": combined, "error": ""})
    return {"results": results, "table": table}


def best_embedding(
    vectors: np.ndarray, k: int = 5, seed: int = 42,
    methods: Sequence[str] = METHODS,
) -> Dict:
    """Run all reducers, score each, return the winner plus the comparison.

    Returns a dict with: ``method`` (winner), ``embedding`` [N,2],
    ``scores`` (winner's metric dict), ``combined`` (winner's score), and
    ``table`` (list of per-method dicts for the dr_quality.csv export).
    """
    allr = embed_all(vectors, k=k, seed=seed, methods=methods)
    results = allr["results"]
    if not results:
        raise RuntimeError("All reduction methods failed for this board/layer.")
    winner = max(results, key=lambda m: results[m]["combined"])
    b = results[winner]
    return {"method": winner, "embedding": b["embedding"], "scores": b["scores"],
            "combined": b["combined"], "table": allr["table"]}


def _draw_panel(ax, emb, words, word_types, vectors, *, layer, num_layers,
                method_name, scores, label_map=None):
    """Render one projection panel in the reference aesthetic.

    Saturated category markers, an arrow from the hint to each target, every
    word labelled (hint/target bold), repelled off the points with thin leader
    lines, and visible light-grey spines + numeric ticks.
    """
    lmap = label_map or {}
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    ax.set_facecolor(_BG_COLOR)

    # Markers: per-type colour/shape/size; white edge on hint and assassin so
    # they read against neighbours and in grayscale print.
    hint_idx = None
    target_indices = []
    for i, wt in enumerate(word_types):
        edged = wt in ("hint", "black")
        ax.scatter(
            emb[i, 0], emb[i, 1], c=_ref_color(wt),
            marker=_REF_MARKER.get(wt, "o"), s=_REF_SIZE.get(wt, 40),
            alpha=_REF_ALPHA.get(wt, 0.9),
            edgecolors="white" if edged else "none",
            linewidths=0.8 if edged else 0.5, zorder=3,
        )
        if wt == "hint":
            hint_idx = i
        elif wt == "target":
            target_indices.append(i)

    # Arrows from the hint to each target — teal, with an arrowhead, drawn under
    # the markers (matches the reference figure).
    if hint_idx is not None:
        hx, hy = emb[hint_idx]
        for ti in target_indices:
            tx, ty = emb[ti]
            ax.annotate(
                "", xy=(tx, ty), xytext=(hx, hy),
                arrowprops=dict(arrowstyle="-|>", color=_REF_COLOR["target"],
                                lw=1.5, alpha=0.7, mutation_scale=12),
                zorder=2,
            )

    # Labels: every word; hint and target bold; target tagged "[T]".
    texts = []
    for i, wt in enumerate(word_types):
        base = lmap.get(words[i], words[i])
        label = f"{base} [T]" if wt == "target" else base
        weight = "bold" if wt in ("hint", "target") else "normal"
        texts.append(ax.text(
            emb[i, 0], emb[i, 1], label, fontsize=_REF_FS["label"],
            fontweight=weight, color=_ref_color(wt), zorder=5,
        ))
    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", lw=0.4, color=_LEADER_COLOR),
            expand=(1.3, 1.5), force_text=(0.5, 0.8),
        )

    ax.set_title(f"Layer {layer} — {depth_label(layer, num_layers)}",
                 fontsize=_REF_FS["title"], fontweight="bold")
    ax.set_xlabel(f"{method_name} 1", fontsize=_REF_FS["axis"])
    ax.set_ylabel(f"{method_name} 2", fontsize=_REF_FS["axis"])
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors=_SPINE_COLOR, labelsize=_REF_FS["tick"])
    ax.margins(0.16)


def _ref_legend(fig, present_types):
    """Compact framed corner legend in the reference aesthetic."""
    from matplotlib.lines import Line2D

    handles = []
    for wt in _REF_ORDER:
        if wt not in present_types:
            continue
        edged = wt in ("hint", "black")
        handles.append(Line2D(
            [0], [0], marker=_REF_MARKER[wt], color="w",
            markerfacecolor=_REF_COLOR[wt], markersize=9 if wt == "hint" else 8,
            label=_REF_LEGEND_LABEL[wt],
            markeredgecolor="white" if edged else _REF_COLOR[wt],
            markeredgewidth=0.8 if edged else 0.0, linestyle="none",
        ))
    if not handles:
        return None
    leg = fig.legend(
        handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93),
        fontsize=_REF_FS["legend"], framealpha=0.9, edgecolor=_SPINE_COLOR,
        frameon=True,
    )
    leg.get_frame().set_linewidth(0.6)
    return leg


def plot_layer_panels(
    layer_data: List[Dict],
    *,
    num_layers: int,
    title: str,
    k: int = 5,
    seed: int = 42,
    n_cols: int = 3,
    method: str = PREFERRED_METHOD,
    label_map: Optional[Dict[str, str]] = None,
) -> Tuple["object", List[Dict]]:
    """Multi-panel projection across layers for one board/condition.

    ``layer_data`` is a list of dicts (one per layer) with keys ``layer``,
    ``words`` (list[str]), ``word_types`` (list[str]), ``vectors`` ([W, D]).
    The panel is rendered with a fixed reducer (``method``, default UMAP); all
    reducers are still scored into ``records`` for the audit.

    Returns ``(figure, records)`` where ``records`` is a flat list of per-layer
    metric rows for CSV export.
    """
    import matplotlib.pyplot as plt

    # House defaults (DPI, export settings) — then the reference overrides
    # (serif, paper background, visible spines/ticks) are applied via rc_context
    # so they are scoped to THIS figure only and never leak to the heatmap.
    apply_publication_style()

    n = len(layer_data)
    n_cols = min(n_cols, max(1, n))
    n_rows = int(np.ceil(n / n_cols))

    # A larger canvas (≈5 in panels) so every word label breathes, matching the
    # reference figure's proportions rather than the compact thesis-width grid.
    panel_w, panel_h = 5.0, 4.4
    figsize = (n_cols * panel_w, n_rows * panel_h + 1.4)

    records: List[Dict] = []
    all_types_present: set = set()

    with plt.rc_context(_REF_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

        for panel_i, ld in enumerate(layer_data):
            r, c = divmod(panel_i, n_cols)
            ax = axes[r][c]
            layer = int(ld["layer"])
            words = ld["words"]
            word_types = ld["word_types"]
            vectors = ld["vectors"]
            all_types_present.update(word_types)

            if vectors.shape[0] < 3:
                ax.text(0.5, 0.5, "too few words", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_REF_FS["axis"],
                        color="grey")
                ax.set_title(f"Layer {layer} — {depth_label(layer, num_layers)}",
                             fontsize=_REF_FS["title"], fontweight="bold")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            allr = embed_all(vectors, k=k, seed=seed)
            results = allr["results"]
            chosen = method if method in results else (
                max(results, key=lambda m: results[m]["combined"]) if results else None)
            if chosen is None:
                ax.text(0.5, 0.5, "reduction failed", ha="center", va="center",
                        transform=ax.transAxes, fontsize=_REF_FS["axis"],
                        color="grey")
                ax.set_xticks([]); ax.set_yticks([])
                continue

            emb = np.asarray(results[chosen]["embedding"])
            _draw_panel(
                ax, emb, words, word_types, vectors,
                layer=layer, num_layers=num_layers,
                method_name=chosen.upper(), scores=results[chosen]["scores"],
                label_map=label_map,
            )
            for row in allr["table"]:
                records.append({"layer": layer, "selected": (row["method"] == chosen), **row})

        # Blank any unused panels.
        for panel_i in range(n, n_rows * n_cols):
            r, c = divmod(panel_i, n_cols)
            axes[r][c].axis("off")

        # Bold suptitle (board/condition identity), framed corner legend and an
        # italic disclaimer footnote — all per the reference figure.
        if title:
            fig.suptitle(title, fontsize=_REF_FS["suptitle"], fontweight="bold",
                         y=0.99)
        _ref_legend(fig, all_types_present)
        fig.text(0.5, 0.012, _REF_DISCLAIMER, ha="center", fontsize=_REF_FS["footnote"],
                 fontstyle="italic", color=_FOOTNOTE_COLOR, wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    return fig, records
