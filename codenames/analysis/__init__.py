"""Cross-model aggregation of the per-model experiment outputs.

Each model run writes its own ``output/<prefix>_outputs/`` folder. This
package reads those folders and builds the cross-model tables and figures:

- :mod:`codenames.analysis.tables` — aggregated behavioral metrics (MRR,
  Hit@K, top-1 accuracy, margins), paired social-preamble effects with
  bootstrap CIs, and concordance.
- :mod:`codenames.analysis.trust` — the trustworthiness sweep over the
  rendered UMAP(cosine) projection per (model, condition, layer).
- :mod:`codenames.analysis.figures` — the chart set.

This is post-hoc analysis: it only reads files under ``output/`` and never
loads a model. Heavy plotting/reduction imports stay inside the modules.
"""
