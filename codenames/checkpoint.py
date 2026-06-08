"""Atomic, crash-safe checkpoint I/O for the extraction loop.

Each per-condition stream is flushed to disk at shard boundaries as a
sequence of checkpoint files:

    {prefix}_{stream}_{mode}_ckpt{idx:04d}.{ext}

where ``stream`` is one of ``metrics`` / ``general`` / ``generation`` /
``vectors`` and ``idx`` increases monotonically with board order.

Two invariants make this safe to resume from (sub-step 1.2):

1. **Atomic writes.** Every checkpoint is written to a ``.tmp`` sibling and
   then ``os.replace``-d into place, so a checkpoint file that exists on disk
   is always complete — a crash mid-write leaves only the ``.tmp``.

2. **Order-preserving, lossless assembly.** Metrics are stored as parquet
   shards (matching the pre-checkpoint code, which concatenated parquet
   shards). General / generation / vectors are stored as pickled *raw record
   lists* — not reassembled DataFrames — so the final assembly reconstructs
   the exact ordered ``list[dict]`` the loop would have held in memory and
   then runs the unchanged persistence helpers. The final outputs are
   therefore byte-identical to the pre-checkpoint loop.

``records`` streams (general/generation/vectors) carry plain dicts (the
vectors stream's dicts contain small ``numpy`` arrays); pickle round-trips
them to equal objects, so ``pd.DataFrame(load_records(...))`` equals
``pd.DataFrame(in_memory_records)``.
"""

import json
import os
import pickle
import re
from typing import List, Optional, Tuple

import pandas as pd

# Streams that are checkpointed as pickled raw-record lists. ``errors`` is
# included so the per-condition error log survives a crash and resumes
# byte-identically alongside the data streams.
RECORD_STREAMS = ("general", "generation", "vectors", "errors")
# The metrics stream is checkpointed as parquet shards.
METRICS_STREAM = "metrics"

_EXT = {"metrics": "parquet", "general": "pkl", "generation": "pkl",
        "vectors": "pkl", "errors": "pkl"}


def ckpt_path(base_dir: str, prefix: str, stream: str, mode: str, idx: int) -> str:
    ext = _EXT[stream]
    return os.path.join(base_dir, f"{prefix}_{stream}_{mode}_ckpt{idx:04d}.{ext}")


def _ckpt_glob_re(prefix: str, stream: str, mode: str) -> "re.Pattern[str]":
    ext = _EXT[stream]
    return re.compile(
        rf"^{re.escape(prefix)}_{re.escape(stream)}_{re.escape(mode)}_ckpt(\d+)\.{ext}$"
    )


def list_ckpts(base_dir: str, prefix: str, stream: str, mode: str) -> List[Tuple[int, str]]:
    """Return ``(idx, path)`` for every committed checkpoint, sorted by idx.

    ``.tmp`` files (incomplete writes from a crash) are ignored by the regex.
    """
    if not os.path.isdir(base_dir):
        return []
    pat = _ckpt_glob_re(prefix, stream, mode)
    out = []
    for name in os.listdir(base_dir):
        m = pat.match(name)
        if m:
            out.append((int(m.group(1)), os.path.join(base_dir, name)))
    out.sort(key=lambda t: t[0])
    return out


def _atomic_write(path: str, write_fn) -> None:
    """Write via a ``.tmp`` sibling then atomically replace ``path``."""
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def write_records(records: list, base_dir: str, prefix: str, stream: str,
                  mode: str, idx: int) -> str:
    """Pickle a raw-record list atomically to its checkpoint path."""
    assert stream in RECORD_STREAMS, stream
    path = ckpt_path(base_dir, prefix, stream, mode, idx)

    def _w(tmp):
        with open(tmp, "wb") as f:
            pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)

    _atomic_write(path, _w)
    return path


def write_metrics_shard(df: pd.DataFrame, base_dir: str, prefix: str,
                        mode: str, idx: int) -> str:
    """Write a metrics parquet shard atomically."""
    path = ckpt_path(base_dir, prefix, METRICS_STREAM, mode, idx)
    _atomic_write(path, lambda tmp: df.to_parquet(tmp, index=False))
    return path


def load_records(base_dir: str, prefix: str, stream: str, mode: str) -> list:
    """Concatenate all committed record checkpoints in index order."""
    assert stream in RECORD_STREAMS, stream
    out: list = []
    for _, path in list_ckpts(base_dir, prefix, stream, mode):
        with open(path, "rb") as f:
            out.extend(pickle.load(f))
    return out


def load_metrics_frames(base_dir: str, prefix: str, mode: str) -> List[pd.DataFrame]:
    """Read all committed metrics parquet shards in index order."""
    return [pd.read_parquet(path)
            for _, path in list_ckpts(base_dir, prefix, METRICS_STREAM, mode)]


def remove_ckpts(base_dir: str, prefix: str, mode: str,
                 streams=("metrics",) + RECORD_STREAMS) -> None:
    """Delete committed checkpoints (and stray ``.tmp`` files) for a condition."""
    for stream in streams:
        for _, path in list_ckpts(base_dir, prefix, stream, mode):
            if os.path.exists(path):
                os.remove(path)
            tmp = path + ".tmp"
            if os.path.exists(tmp):
                os.remove(tmp)


def remove_tmp(base_dir: str, prefix: str, mode: str,
               streams=("metrics",) + RECORD_STREAMS) -> None:
    """Delete only stray ``.tmp`` files (incomplete writes), keeping committed
    checkpoints intact. Used during resume reconciliation."""
    if not os.path.isdir(base_dir):
        return
    for stream in streams:
        ext = _EXT[stream]
        suffix = f"_{stream}_{mode}_"
        for name in os.listdir(base_dir):
            if name.startswith(f"{prefix}{suffix}") and name.endswith(f".{ext}.tmp"):
                os.remove(os.path.join(base_dir, name))


# ---------------------------------------------------------------------------
# Manifest — the single source of truth for what has been committed.
#
# Written *after* a flush's checkpoint files are on disk (data-first,
# manifest-second), so a crash between the two leaves only orphan checkpoints
# with ``idx >= ckpt_committed`` that ``reconcile`` discards. The manifest is
# retained only while a run is incomplete; a clean run deletes it at the end,
# leaving the output directory byte-identical to a non-resumable run.
# ---------------------------------------------------------------------------

def manifest_path(base_dir: str, prefix: str, mode: str) -> str:
    return os.path.join(base_dir, f"{prefix}_{mode}_manifest.json")


def read_manifest(base_dir: str, prefix: str, mode: str) -> Optional[dict]:
    path = manifest_path(base_dir, prefix, mode)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_manifest(base_dir: str, prefix: str, mode: str, *,
                   n_boards: int, boards_done: int, ckpt_committed: int,
                   complete: bool) -> str:
    path = manifest_path(base_dir, prefix, mode)
    obj = {
        "n_boards": int(n_boards),
        "boards_done": int(boards_done),
        "ckpt_committed": int(ckpt_committed),
        "complete": bool(complete),
    }

    def _w(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, sort_keys=True)

    _atomic_write(path, _w)
    return path


class ResumeSizeMismatch(RuntimeError):
    """Raised when a --resume target's manifest was written for a different
    run size than the current run, which would mix non-corresponding boards."""


def remove_manifest(base_dir: str, prefix: str, mode: str) -> None:
    path = manifest_path(base_dir, prefix, mode)
    for q in (path, path + ".tmp"):
        if os.path.exists(q):
            os.remove(q)


def reconcile(base_dir: str, prefix: str, mode: str,
              expected_n_boards: int) -> Tuple[int, int, bool]:
    """Bring the checkpoint dir into a consistent, resumable state.

    Returns ``(boards_done, ckpt_committed, complete)``:

    * No manifest → the checkpoints cannot be trusted (a crash before the
      first commit, or leftovers from an aborted non-resumable run). Wipe all
      checkpoints + ``.tmp`` and report a fresh start ``(0, 0, False)``.
    * Manifest present → delete orphan checkpoints with ``idx >=
      ckpt_committed`` (written but not committed) and all ``.tmp`` files,
      then report the committed state to resume from.

    Raises :class:`ResumeSizeMismatch` if the manifest was written for a
    different run size than ``expected_n_boards`` — resuming across run sizes
    would skip the wrong (non-corresponding) boards, since ``df_sample``
    membership and order depend on the sample size.
    """
    m = read_manifest(base_dir, prefix, mode)
    if m is None:
        remove_ckpts(base_dir, prefix, mode)
        remove_tmp(base_dir, prefix, mode)
        return 0, 0, False

    manifest_n = int(m.get("n_boards", -1))
    if manifest_n != int(expected_n_boards):
        raise ResumeSizeMismatch(
            f"--resume target '{base_dir}' has a {mode} manifest for "
            f"n_boards={manifest_n}, but this run has n_boards={expected_n_boards}. "
            f"Re-run without --resume to start fresh, or use the matching run size."
        )

    committed = int(m["ckpt_committed"])
    for stream in ("metrics",) + RECORD_STREAMS:
        for idx, path in list_ckpts(base_dir, prefix, stream, mode):
            if idx >= committed and os.path.exists(path):
                os.remove(path)
    remove_tmp(base_dir, prefix, mode)
    return int(m["boards_done"]), committed, bool(m["complete"])
