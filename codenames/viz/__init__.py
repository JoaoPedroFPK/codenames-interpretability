"""Local, post-hoc visualization of the layer-wise word-vector geometry.

Reads the experiment outputs under ``output/{model}/`` and renders, for a small
sampled set of boards per model, two publication-formatted figure families:

- a **word x word cosine heatmap** (no_social vs with_social) at fixed layers;
- a **cosine-aware 2D projection** of the word vectors across layers, with the
  reduction validated by trustworthiness / continuity / Shepard metrics.

This subpackage depends on the optional ``[viz]`` dependency group
(``pip install -e ".[viz]"``); it is never imported by the experiment run path,
so the core Colab install does not require matplotlib/umap/scikit-learn.
"""

from . import embedding, heatmap, loader, metrics, pipeline, style  # noqa: F401

__all__ = ["embedding", "heatmap", "loader", "metrics", "pipeline", "style"]
