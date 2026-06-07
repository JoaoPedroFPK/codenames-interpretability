"""Orchestration: read outputs, sample boards, render the figure set.

Produces, per model and per sampled board:

- ``heatmap_L{layer}.{pdf,png}`` — cosine heatmap pair (no_social vs with_social)
  at each representative layer;
- ``umap_{condition}_layers.{pdf,png}`` — multi-panel cosine-aware projection
  across representative layers, for each condition that has the board;
- ``dr_quality_{condition}.csv`` — the UMAP/t-SNE/PCA comparison scores per layer.

Figures are written under ``{viz_dir}/{model}/board_{row_id}/``. The heatmap is a
two-condition pair, so figures are grouped by board with the condition encoded in
the filename rather than nested under a single-condition folder.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import pandas as pd

from . import embedding, heatmap, loader
from .style import save_figure, select_layers


# Title separator: a mid-dot framed by thin spaces (U+2009) so the dots get a
# little air and do not render cramped against the surrounding words.
_SEP = "  ·  "


def _board_title(model: str, row_id: int, meta: Dict, suffix: str = "") -> str:
    hint = meta.get("hint")
    nt = meta.get("n_targets")
    bits = [f"{model}", f"board {row_id}"]
    if hint is not None and not pd.isna(hint):
        bits.append(f"hint = '{hint}'")
    if nt is not None and not pd.isna(nt):
        bits.append(f"{int(nt)} target(s)")
    head = _SEP.join(bits)
    return f"{head}{suffix}"


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
    seed: int = 42,
    k: int = 5,
    formats: Sequence[str] = ("pdf", "png"),
    boards: Optional[Sequence[int]] = None,
) -> Dict:
    """Generate the figure set for one model. Returns a small summary dict.

    If ``boards`` is given, those exact row_ids are used (any not available for
    this model are skipped with a warning) — this is how the same boards are
    visualised across models for direct comparison. Otherwise ``n_boards`` are
    sampled reproducibly from this model's own subsample.
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
    else:
        board_ids = loader.sample_boards(pool_index, n_boards, seed)
    print(f"[viz]   layers={sel_layers}  boards={board_ids}")

    n_fig = 0
    for row_id in board_ids:
        board_dir = os.path.join(viz_dir, name, f"board_{row_id}")

        # Display labels: show the giver-feature VALUE (e.g. "united states")
        # instead of the key ("giver.country"). Built from the board's
        # giver_features dict; non-giver words keep their own text.
        board_meta_ws = loader.board_meta(
            generals.get("with_social", pd.DataFrame()), row_id) \
            or loader.board_meta(generals.get("no_social", pd.DataFrame()), row_id)
        giver_vals = board_meta_ws.get("giver_features") or {}
        label_map = {str(k): str(v) for k, v in giver_vals.items()}

        # --- Heatmap pair per representative layer ---
        for layer in sel_layers:
            panels: Dict[str, Dict] = {}
            for mode, cond in present.items():
                words, types, vecs = loader.board_layer_words(cond, row_id, layer)
                if len(words) >= 2:
                    panels[mode] = {"words": words, "word_types": types, "vectors": vecs}
            if not panels:
                continue
            title = _board_title(name, row_id, board_meta_ws, suffix=f"{_SEP}layer {layer}")
            fig, _info = heatmap.plot_heatmap_pair(
                panels, layer=layer, title=title, label_map=label_map)
            paths = save_figure(fig, os.path.join(board_dir, f"heatmap_L{layer:02d}"),
                                formats=tuple(formats))
            plt.close(fig)
            n_fig += 1

        # --- Projection multi-panel per condition + dr_quality export ---
        for mode, cond in present.items():
            meta = loader.board_meta(generals.get(mode, pd.DataFrame()), row_id)
            layer_data = _layer_data_for_board(cond, row_id, sel_layers)
            if not any(ld["vectors"].shape[0] >= 3 for ld in layer_data):
                continue
            nice = "no social" if mode == "no_social" else "with social"
            title = _board_title(name, row_id, meta, suffix=f"{_SEP}{nice}")
            method = embedding.PREFERRED_METHOD  # rendered reducer (default UMAP)
            fig, records = embedding.plot_layer_panels(
                layer_data, num_layers=nlayers, title=title, k=k, seed=seed,
                method=method, label_map=label_map,
            )
            # Name the file after the reducer actually used, so the artefact is
            # never mislabelled if the preferred method changes.
            save_figure(fig, os.path.join(board_dir, f"{method}_{mode}_layers"),
                        formats=tuple(formats))
            plt.close(fig)
            n_fig += 1
            if records:
                pd.DataFrame(records).to_csv(
                    os.path.join(board_dir, f"dr_quality_{mode}.csv"), index=False
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
    seed: int = 42,
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
        per_model = {
            rec["name"]: loader.subsample_board_ids(rec["dir"], rec["prefix"], pooling)
            for rec in models
        }
        shared = set.intersection(*per_model.values()) if per_model else set()
        print(f"[viz] models: {list(per_model)}")
        print(f"[viz] shared boards across all models: {len(shared)}")
        if shared:
            boards = loader.sample_ids(shared, n_boards, seed)
            print(f"[viz] using SHARED boards for all models: {boards}")
        else:
            sizes = {m: len(s) for m, s in per_model.items()}
            print("[viz] WARNING: models have NO common subsample boards "
                  f"(per-model counts: {sizes}). This usually means they were run "
                  "with different run sizes/seeds. Falling back to per-model "
                  "sampling — boards will NOT be comparable across models.")

    summaries = []
    for rec in models:
        summaries.append(run(
            rec["name"], output_root=output_root,
            n_boards=n_boards, pooling=pooling, seed=seed, boards=boards, **kwargs,
        ))
    return summaries
