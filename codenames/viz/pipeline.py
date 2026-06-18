"""Orchestration: read outputs, sample boards, render the figure set.

Produces, per model and per sampled board:

- ``heatmaps/L{layer}_{condition}.{pdf,png}`` — one cosine heatmap per
  condition at each representative layer (separate files so each fits a
  report page; all heatmaps share a fixed 0..1 colour scale so any two are
  directly comparable);
- ``umaps/{condition}.{pdf,png}`` — multi-panel cosine-aware projection
  across representative layers, for each condition that has the board;
- ``umaps/dr_quality_{condition}.csv`` — UMAP/t-SNE/PCA comparison scores per layer.

Figures are written under ``{viz_dir}/{model}/board_{row_id}/`` with the two
figure families split into ``heatmaps/`` and ``umaps/`` subfolders; the now
redundant type prefix is dropped from the filename (the condition is kept).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from . import embedding, heatmap, loader
from .style import depth_label, display_name, save_figure, select_layers


# Title separator: a mid-dot framed by thin spaces (U+2009) so the dots get a
# little air and do not render cramped against the surrounding words.
_SEP = "  ·  "


def shared_board_sample(
    output_root: str,
    n_boards: int,
    *,
    pooling: str = "mean",
    seed: int = 2026,
) -> tuple:
    """Deterministic board ids present in EVERY discovered model's subsample.

    Computes the intersection of the per-model subsample board sets (index CSVs
    only — no vectors loaded) and reproducibly samples ``n_boards`` from it. The
    selection is a pure function of (shared id-set, seed), so it is identical no
    matter which model — or how many — you render. This is what makes the same
    boards line up across models for direct comparison.

    Returns ``(board_ids, info)`` where ``info`` carries ``per_model`` counts and
    the ``shared`` intersection size. ``board_ids`` is empty if the models share
    no boards (e.g. different run sizes/seeds).
    """
    models = loader.discover_models(output_root)
    per_model = {
        rec["name"]: loader.subsample_board_ids(rec["dir"], rec["prefix"], pooling)
        for rec in models
    }
    nonempty = [s for s in per_model.values() if s]
    shared = set.intersection(*nonempty) if nonempty else set()
    board_ids = loader.sample_ids(shared, n_boards, seed) if shared else []
    info = {"per_model": {k: len(v) for k, v in per_model.items()}, "shared": len(shared)}
    return board_ids, info


def _board_header(
    prefix: str, row_id: int, meta: Dict, targets: Sequence[str], mode: str,
) -> Tuple[str, str]:
    """Human title + de-emphasized provenance line for a board figure.

    Title carries the semantics a reader cares about — the model's display
    name and the actual clue → target words (never internal identifiers or a
    bare target count). Provenance (board id, condition) moves to a smaller
    grey subtitle.
    """
    bits = [display_name(prefix)]
    hint = meta.get("hint")
    if hint is not None and not pd.isna(hint):
        clue = f"clue “{hint}”"
        if targets:
            clue += f" → {', '.join(targets)}"
        bits.append(clue)
    title = _SEP.join(bits)
    nice = "no social context" if mode == "no_social" else "with social context"
    subtitle = f"board {row_id}{_SEP}{nice}"
    return title, subtitle


def _layer_data_for_board(cond: Optional[Dict], row_id: int, layers: Sequence[int]) -> List[Dict]:
    out: List[Dict] = []
    if cond is None:
        return out
    for layer in layers:
        words, types, vecs = loader.board_layer_words(cond, row_id, layer)
        out.append({"layer": layer, "words": words, "word_types": types, "vectors": vecs})
    return out


def run(
    model: str,
    *,
    output_root: str = "output",
    viz_dir: str = "visualization",
    n_boards: int = 5,
    pooling: str = "mean",
    layers: Optional[Sequence[int]] = None,
    seed: int = 2026,
    k: int = 5,
    formats: Sequence[str] = ("pdf", "png"),
    boards: Optional[Sequence[int]] = None,
    cross_model: bool = True,
    skip_projection: bool = False,
) -> Dict:
    """Generate the figure set for one model. Returns a small summary dict.

    Board selection, in priority order:

    1. If ``boards`` is given, those exact row_ids are used (any not available
       for this model are skipped with a warning).
    2. Else if ``cross_model`` (the default), ``n_boards`` are sampled from the
       intersection of subsample boards across ALL discovered models — so a
       single-model run picks the SAME boards another model's run would, making
       them directly comparable without ``--all``.
    3. Else (or if the models share no boards), ``n_boards`` are sampled from
       this model's own subsample.
    """
    import matplotlib.pyplot as plt

    rec = loader.resolve_model(output_root, model)
    name, model_dir, prefix = rec["name"], rec["dir"], rec["prefix"]
    print(f"[viz] model='{name}'  dir='{model_dir}'  prefix='{prefix}'  pooling={pooling}")

    conds: Dict[str, Optional[Dict]] = {}
    generals: Dict[str, pd.DataFrame] = {}
    for mode in loader.MODES:
        conds[mode] = load = loader.load_condition(model_dir, prefix, mode, pooling)
        generals[mode] = loader.load_general(model_dir, prefix, mode)
        status = "missing" if load is None else f"{len(load['index'])} vector rows"
        print(f"[viz]   {mode}: {status}")

    present = {m: c for m, c in conds.items() if c is not None}
    if not present:
        raise SystemExit(f"No usable vector data for model '{name}' (pooling={pooling}).")

    # Derive layer set and board sample from whichever condition has data.
    ref_cond = present.get("no_social") or next(iter(present.values()))
    nlayers = loader.num_layers(ref_cond["index"])
    avail = loader.available_layers(ref_cond["index"])
    sel_layers = sorted(set(int(x) for x in layers)) if layers else select_layers(avail, 6)
    sel_layers = [L for L in sel_layers if L in set(avail)]

    # Boards available for this model (valid in all present conditions).
    id_sets = [set(c["index"]["row_id"].unique()) for c in present.values()]
    common = set.intersection(*id_sets) if id_sets else set()
    pool_index = ref_cond["index"]
    if common:
        pool_index = pool_index[pool_index["row_id"].isin(common)]

    if boards is not None:
        requested = [int(b) for b in boards]
        avail = set(int(x) for x in pool_index["row_id"].unique())
        board_ids = [b for b in requested if b in avail]
        missing = [b for b in requested if b not in avail]
        if missing:
            print(f"[viz]   WARNING: {len(missing)} requested board(s) not available "
                  f"for '{name}' (skipped): {missing}")
        if not board_ids:
            print(f"[viz]   no requested boards available for '{name}'; skipping.")
            return {"model": name, "boards": [], "layers": sel_layers, "figures": 0}
    elif cross_model:
        shared_ids, info = shared_board_sample(
            output_root, n_boards, pooling=pooling, seed=seed)
        avail = set(int(x) for x in pool_index["row_id"].unique())
        board_ids = [b for b in shared_ids if b in avail]
        if board_ids:
            print(f"[viz]   cross-model shared boards "
                  f"(intersection of {len(info['per_model'])} models, "
                  f"{info['shared']} shared): {board_ids}")
        else:
            print("[viz]   no cross-model shared boards available; "
                  "falling back to this model's own sample.")
            board_ids = loader.sample_boards(pool_index, n_boards, seed)
    else:
        board_ids = loader.sample_boards(pool_index, n_boards, seed)
    print(f"[viz]   layers={sel_layers}  boards={board_ids}")

    n_fig = 0
    for row_id in board_ids:
        # Per board, the two figure families live in their own subfolders, and
        # the now-redundant type prefix is dropped from the filename:
        #   board_{id}/heatmaps/L{layer}_{condition}.{pdf,png}
        #   board_{id}/umaps/{condition}.{pdf,png}  (+ dr_quality_{condition}.csv)
        board_dir = os.path.join(viz_dir, name, f"board_{row_id}")
        heatmap_dir = os.path.join(board_dir, "heatmaps")
        umap_dir = os.path.join(board_dir, "umaps")

        # Display labels: show the giver-feature VALUE (e.g. "united states")
        # instead of the key ("giver.country"). Built from the board's
        # giver_features dict; non-giver words keep their own text.
        board_meta_ws = loader.board_meta(
            generals.get("with_social", pd.DataFrame()), row_id) \
            or loader.board_meta(generals.get("no_social", pd.DataFrame()), row_id)
        giver_vals = board_meta_ws.get("giver_features") or {}
        label_map = {str(k): str(v) for k, v in giver_vals.items()}

        # --- One heatmap per condition per representative layer ---
        # Both conditions are passed to the renderer even though each file
        # shows one: the word ordering and colour scale are computed jointly
        # so the two files remain directly comparable.
        for layer in sel_layers:
            panels: Dict[str, Dict] = {}
            for mode, cond in present.items():
                words, types, vecs = loader.board_layer_words(cond, row_id, layer)
                if len(words) >= 2:
                    panels[mode] = {"words": words, "word_types": types, "vectors": vecs}
            if not panels:
                continue
            # Same provenance treatment as the projection subtitle: board id
            # and the layer (with its depth band) under the bold title.
            subtitle = (f"board {row_id}{_SEP}layer {layer} "
                        f"({depth_label(layer, nlayers)})")
            for mode in panels:
                fig, _info = heatmap.plot_heatmap_condition(
                    panels, mode, layer=layer, label_map=label_map,
                    header=display_name(prefix), subtitle=subtitle)
                if fig is None:
                    continue
                save_figure(fig, os.path.join(heatmap_dir, f"L{layer:02d}_{mode}"),
                            formats=tuple(formats))
                plt.close(fig)
                n_fig += 1

        # --- Projection multi-panel per condition + dr_quality export ---
        # The projection (UMAP) fits are the slow step; skip them when only the
        # heatmaps changed (the existing umap_*/dr_quality_* files stay valid).
        for mode, cond in ({} if skip_projection else present).items():
            meta = loader.board_meta(generals.get(mode, pd.DataFrame()), row_id)
            layer_data = _layer_data_for_board(cond, row_id, sel_layers)
            if not any(ld["vectors"].shape[0] >= 3 for ld in layer_data):
                continue
            # Actual target words (same in both conditions) for the header.
            words0, types0, _ = loader.board_layer_words(cond, row_id, sel_layers[0])
            targets = [w for w, t in zip(words0, types0) if t == "target"]
            title, subtitle = _board_header(prefix, row_id, meta, targets, mode)
            method = embedding.PREFERRED_METHOD  # rendered reducer (default UMAP)
            fig, records = embedding.plot_layer_panels(
                layer_data, num_layers=nlayers, title=title, k=k, seed=seed,
                method=method, label_map=label_map, subtitle=subtitle,
            )
            # The folder (umaps/) already names the family; the file is just the
            # condition. The reducer actually used is recorded in dr_quality's
            # ``selected`` column, so it is never lost if the method changes.
            save_figure(fig, os.path.join(umap_dir, f"{mode}"),
                        formats=tuple(formats))
            plt.close(fig)
            n_fig += 1
            if records:
                pd.DataFrame(records).to_csv(
                    os.path.join(umap_dir, f"dr_quality_{mode}.csv"), index=False
                )

        print(f"[viz]   board {row_id}: figures written to {board_dir}")

    summary = {"model": name, "boards": board_ids, "layers": sel_layers, "figures": n_fig}
    print(f"[viz] done: {n_fig} figures across {len(board_ids)} boards -> {viz_dir}/{name}/")
    return summary


def run_all(
    output_root: str = "output",
    *,
    n_boards: int = 5,
    pooling: str = "mean",
    seed: int = 2026,
    boards: Optional[Sequence[int]] = None,
    **kwargs,
) -> List[Dict]:
    """Generate figures for every model discovered under ``output_root``.

    For direct cross-model comparison, the SAME boards are used for every model:
    unless an explicit ``boards`` list is given, ``n_boards`` are sampled from the
    **intersection** of the boards available across all discovered models (so the
    chosen boards are guaranteed present everywhere and identical across models).
    """
    models = loader.discover_models(output_root)
    if not models:
        raise SystemExit(f"No model outputs found under '{output_root}'.")

    if boards is None:
        boards, info = shared_board_sample(output_root, n_boards, pooling=pooling, seed=seed)
        print(f"[viz] models: {list(info['per_model'])}")
        print(f"[viz] shared boards across all models: {info['shared']}")
        if boards:
            print(f"[viz] using SHARED boards for all models: {boards}")
        else:
            print("[viz] WARNING: models have NO common subsample boards "
                  f"(per-model counts: {info['per_model']}). This usually means they "
                  "were run with different run sizes/seeds. Falling back to per-model "
                  "sampling — boards will NOT be comparable across models.")

    summaries = []
    for rec in models:
        summaries.append(run(
            rec["name"], output_root=output_root,
            n_boards=n_boards, pooling=pooling, seed=seed, boards=boards, **kwargs,
        ))
    return summaries
