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

import os
import pickle
import re
from typing import List, Tuple

import pandas as pd

# Streams that are checkpointed as pickled raw-record lists.
RECORD_STREAMS = ("general", "generation", "vectors")
# The metrics stream is checkpointed as parquet shards.
METRICS_STREAM = "metrics"

_EXT = {"metrics": "parquet", "general": "pkl", "generation": "pkl", "vectors": "pkl"}


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
