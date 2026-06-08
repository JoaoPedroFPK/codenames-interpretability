"""Size-independent canonical cache for cross-run reuse.

The canonical (``permutation_id == 0``) results for a board depend only on that
board's content, not on the sample it was drawn in or its position — *provided*
the forward pass is the per-board reference path (``batch_size == 1``). So the
canonical metrics / general / generation records for a ``row_id`` computed in
one run are byte-identical to those a later run would compute, and can be
reused.

This module persists those canonical records keyed by ``row_id`` so a later,
larger run reuses the per-board canonical + generation work for any ``row_id``
it shares with an earlier run. Shuffle permutations (``permutation_id >= 1``)
are *not* cached: their seeds are position-keyed and sample-size-dependent, so
they are always recomputed for the target run — keeping outputs bit-identical
to the reference notebooks (see ``docs`` / the resume design notes).

Per condition (``mode``) the cache is three files in the output directory:

    {prefix}_canoncache_metrics_{mode}.parquet   # flat canonical metric rows
    {prefix}_canoncache_general_{mode}.pkl        # {row_id: general record}
    {prefix}_canoncache_generation_{mode}.pkl     # {row_id: generation record}

Metrics are flat scalar rows, so parquet is compact and round-trips to
byte-identical output. General/generation records carry nested fields
(``giver_features`` dict, ``missing_span_words`` list), so they are pickled to
guarantee an exact round-trip. The cache is **persistent** (never auto-deleted)
and only ever grows: ``update`` adds row_ids not already present, so reuse is
idempotent across repeated runs.
"""

import os
import pickle
from typing import Dict, List, Optional

import pandas as pd


def _metrics_path(base_dir: str, prefix: str, mode: str) -> str:
    return os.path.join(base_dir, f"{prefix}_canoncache_metrics_{mode}.parquet")


def _general_path(base_dir: str, prefix: str, mode: str) -> str:
    return os.path.join(base_dir, f"{prefix}_canoncache_general_{mode}.pkl")


def _generation_path(base_dir: str, prefix: str, mode: str) -> str:
    return os.path.join(base_dir, f"{prefix}_canoncache_generation_{mode}.pkl")


def _atomic_write(path: str, write_fn) -> None:
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def _read_pickle(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


class CanonCache:
    """In-memory view of the canonical cache for one condition.

    Lookups are by ``row_id``. ``has`` reflects the general stream (every cached
    board has exactly one canonical general record); ``has_generation`` is
    separate because encoder models produce no generation records.
    """

    def __init__(self, general: Dict[int, dict], metrics: Dict[int, List[dict]],
                 generation: Dict[int, dict]):
        self._general = general
        self._metrics = metrics
        self._generation = generation

    def has(self, row_id: int) -> bool:
        return int(row_id) in self._general

    def general(self, row_id: int) -> dict:
        return self._general[int(row_id)]

    def metrics(self, row_id: int) -> List[dict]:
        return self._metrics.get(int(row_id), [])

    def has_generation(self, row_id: int) -> bool:
        return int(row_id) in self._generation

    def generation(self, row_id: int) -> dict:
        return self._generation[int(row_id)]

    def __len__(self) -> int:
        return len(self._general)

    @property
    def row_ids(self):
        return set(self._general.keys())


def load(base_dir: str, prefix: str, mode: str) -> CanonCache:
    """Load the canonical cache for one condition (empty if none exists)."""
    general = {int(k): v for k, v in _read_pickle(_general_path(base_dir, prefix, mode)).items()}
    generation = {int(k): v for k, v in _read_pickle(_generation_path(base_dir, prefix, mode)).items()}

    metrics: Dict[int, List[dict]] = {}
    mpath = _metrics_path(base_dir, prefix, mode)
    if os.path.exists(mpath):
        mdf = pd.read_parquet(mpath)
        # groupby(sort=False) preserves each board's original row order
        # (layer 0..L, hint then candidates), which the reuse path relies on.
        for rid, sub in mdf.groupby("row_id", sort=False):
            metrics[int(rid)] = sub.to_dict("records")

    return CanonCache(general, metrics, generation)


def update(base_dir: str, prefix: str, mode: str, *,
           general_df: pd.DataFrame, metrics_df: pd.DataFrame,
           generation_df: Optional[pd.DataFrame]) -> int:
    """Fold a completed condition's canonical results into the cache.

    Adds only ``row_id``s not already cached (idempotent). Returns the number
    of newly-cached boards. ``*_df`` are the assembled per-condition outputs;
    the canonical subset (``permutation_id == 0``) is extracted here.
    """
    if general_df is None or len(general_df) == 0:
        return 0

    existing_general = _read_pickle(_general_path(base_dir, prefix, mode))
    existing_generation = _read_pickle(_generation_path(base_dir, prefix, mode))
    cached_ids = {int(k) for k in existing_general}

    # --- General (canonical only) ---
    canon_general = general_df[general_df["permutation_id"] == 0]
    new_general_records = [
        rec for rec in canon_general.to_dict("records")
        if int(rec["row_id"]) not in cached_ids
    ]
    new_ids = {int(rec["row_id"]) for rec in new_general_records}
    if not new_ids:
        return 0  # nothing new to add

    for rec in new_general_records:
        existing_general[int(rec["row_id"])] = rec
    _atomic_write(_general_path(base_dir, prefix, mode),
                  lambda tmp: _dump_pickle(existing_general, tmp))

    # --- Generation (canonical-only stream; no permutation_id column) ---
    if generation_df is not None and len(generation_df) > 0:
        for rec in generation_df.to_dict("records"):
            rid = int(rec["row_id"])
            if rid in new_ids:
                existing_generation[rid] = rec
        _atomic_write(_generation_path(base_dir, prefix, mode),
                      lambda tmp: _dump_pickle(existing_generation, tmp))

    # --- Metrics (canonical only), appended to the parquet store ---
    if metrics_df is not None and len(metrics_df) > 0:
        canon_metrics = metrics_df[
            (metrics_df["permutation_id"] == 0)
            & (metrics_df["row_id"].isin(new_ids))
        ]
        if len(canon_metrics):
            mpath = _metrics_path(base_dir, prefix, mode)
            frames = [canon_metrics]
            if os.path.exists(mpath):
                frames.insert(0, pd.read_parquet(mpath))
            merged = pd.concat(frames, ignore_index=True)
            _atomic_write(mpath, lambda tmp: merged.to_parquet(tmp, index=False))

    return len(new_ids)


def _dump_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
