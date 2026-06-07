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
    add_word_type_legend,
    apply_publication_style,
    depth_label,
    footnote,
    grid_size,
    style_for,
)

METHODS = ("umap", "tsne", "pca")

# The reducer used to render the projection. Standardised on UMAP(cosine) for a
# consistent picture across boards/layers. All three reducers (UMAP, t-SNE, PCA)
# are still fit and scored per panel for the dr_quality_*.csv audit, so the
# trustworthiness of the rendered UMAP layout can always be checked.
PREFERRED_METHOD = "umap"


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


def _nearest_to_hint(vectors: np.ndarray, word_types: Sequence[str]) -> Optional[int]:
    """Index of the word closest (max cosine) to the hint, excluding the hint."""
    hint_idx = [i for i, t in enumerate(word_types) if t == "hint"]
    if not hint_idx:
        return None
    h = hint_idx[0]
    Xn = _normalize(vectors)
    sims = Xn @ Xn[h]
    sims[h] = -np.inf
    j = int(np.argmax(sims))
    return j if np.isfinite(sims[j]) else None


def _draw_panel(ax, emb, words, word_types, vectors, *, layer, num_layers,
                method_name, scores):
    """Render one projection panel with separated markers and repelled labels."""
    import matplotlib.pyplot as plt  # noqa: F401
    try:
        from adjustText import adjust_text
    except ImportError:
        adjust_text = None

    # Markers: thin white edge separates overlapping points; hint larger + dark.
    for i, wt in enumerate(word_types):
        s = style_for(wt)
        is_hint = (wt == "hint")
        ax.scatter(
            emb[i, 0], emb[i, 1], c=s["color"], marker=s["marker"],
            s=110 if is_hint else 46,
            edgecolors="black" if is_hint else "white",
            linewidths=0.7 if is_hint else 0.4,
            zorder=s["order"] + 3, alpha=0.95,
        )

    # Connector: hint -> nearest word in cosine space (drawn under everything).
    j = _nearest_to_hint(vectors, word_types)
    hint_idx = [i for i, t in enumerate(word_types) if t == "hint"]
    if j is not None and hint_idx:
        h = hint_idx[0]
        ax.annotate(
            "", xy=(emb[j, 0], emb[j, 1]), xytext=(emb[h, 0], emb[h, 1]),
            arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=0.8,
                            alpha=0.8, shrinkA=4, shrinkB=4),
            zorder=2,
        )

    # Labels: repelled off the points with thin leader lines.
    texts = []
    for i, wt in enumerate(word_types):
        label = f"{words[i]} [T]" if wt == "target" else words[i]
        weight = "bold" if wt in ("hint", "target") else "normal"
        texts.append(ax.text(
            emb[i, 0], emb[i, 1], label, fontsize=FS["word_label"], color="#222222",
            fontweight=weight, zorder=11,
        ))
    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#bdbdbd", lw=0.3),
            expand=(1.15, 1.3), force_text=(0.4, 0.6),
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        )

    ax.set_title(f"Layer {layer} — {depth_label(layer, num_layers)}",
                 fontsize=FS["panel_title"], fontweight="bold")
    ax.set_xlabel(f"{method_name} 1", fontsize=FS["axis_label"])
    ax.set_ylabel(f"{method_name} 2", fontsize=FS["axis_label"])
    ax.set_xticks([]); ax.set_yticks([])
    ax.margins(0.16)
    ax.text(
        0.02, 0.02,
        f"{method_name}  T={scores['trustworthiness']:.2f}"
        f"  C={scores['continuity']:.2f}  rho={scores['shepard']:.2f}",
        transform=ax.transAxes, fontsize=FS["annot"], va="bottom", ha="left",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
    )


def plot_layer_panels(
    layer_data: List[Dict],
    *,
    num_layers: int,
    title: str,
    k: int = 5,
    seed: int = 42,
    n_cols: int = 3,
    method: str = PREFERRED_METHOD,
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

    apply_publication_style()

    n = len(layer_data)
    n_cols = min(n_cols, max(1, n))
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=grid_size(n_cols, n_rows, panel_aspect=1.0,
                          header_in=0.5, footer_in=1.3),
        squeeze=False,
    )

    records: List[Dict] = []
    all_types_present: set = set()

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
                    transform=ax.transAxes, fontsize=FS["annot"], color="grey")
            ax.set_title(f"Layer {layer} — {depth_label(layer, num_layers)}",
                         fontsize=FS["panel_title"], fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        allr = embed_all(vectors, k=k, seed=seed)
        results = allr["results"]
        chosen = method if method in results else (
            max(results, key=lambda m: results[m]["combined"]) if results else None)
        if chosen is None:
            ax.text(0.5, 0.5, "reduction failed", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FS["annot"], color="grey")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        emb = np.asarray(results[chosen]["embedding"])
        _draw_panel(
            ax, emb, words, word_types, vectors,
            layer=layer, num_layers=num_layers,
            method_name=chosen.upper(), scores=results[chosen]["scores"],
        )
        for row in allr["table"]:
            records.append({"layer": layer, "selected": (row["method"] == chosen), **row})

    # Blank any unused panels.
    for panel_i in range(n, n_rows * n_cols):
        r, c = divmod(panel_i, n_cols)
        axes[r][c].axis("off")

    # Shared house legend (word-type taxonomy) + light footnote, with the
    # footnote clearly above the legend so they never collide.
    add_word_type_legend(fig, sorted(all_types_present), y=0.02)
    fig.suptitle(title, y=0.99, fontsize=FS["suptitle"], fontweight="bold")
    footnote(
        fig,
        f"Projection: {method.upper()} (cosine), fixed across panels. "
        "Per-panel T = trustworthiness, C = continuity, rho = Shepard "
        "correlation (all reducers audited in dr_quality_*.csv). Hint = diamond; "
        "targets tagged [T]; arrow = hint to nearest word in cosine space.",
        y=0.10,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    return fig, records
