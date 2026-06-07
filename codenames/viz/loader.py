"""Read experiment outputs and assemble per-board, per-layer word vectors.

Each model writes, per condition (``no_social`` / ``with_social``):

- ``{prefix}_vectors_subsample_index_{mode}.csv`` — one row per stored vector,
  with ``record_idx`` giving its row in the matrix below;
- ``{prefix}_vectors_subsample_{mode}_f16.npz`` — key ``vectors``, ``[N, D]`` f16;
- ``{prefix}_general_{mode}.csv`` — board-level metadata (hint, targets, etc.).

Only the vector *subsample* boards have raw vectors, so all board sampling here
draws from the index, never the full metrics table. The model directory and the
file prefix can differ (e.g. dir ``bert_random`` / prefix ``random_bert``), so
the prefix is auto-detected from the directory contents.
"""

from __future__ import annotations

import ast
import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

MODES = ("no_social", "with_social")

# Fallback model -> prefix map (mirrors codenames/cli.py). Auto-detection from
# directory contents takes precedence; this only helps resolve a --model name to
# a directory when the directory is named after the model.
MODEL_PREFIXES: Dict[str, str] = {
    "mistral": "mistral",
    "qwen": "qwen",
    "qwen_random": "random_qwen",
    "bert": "bert",
    "bert_random": "random_bert",
    "t5": "t5",
    "modernbert": "modernbert",
}

_INDEX_SUFFIX = "_vectors_subsample_index_"


def detect_prefix(model_dir: str) -> Optional[str]:
    """Infer the file prefix from an index filename in ``model_dir``."""
    for mode in MODES:
        hits = glob.glob(os.path.join(model_dir, f"*{_INDEX_SUFFIX}{mode}.csv"))
        if hits:
            base = os.path.basename(hits[0])
            return base.split(_INDEX_SUFFIX)[0]
    return None


def discover_models(output_root: str) -> List[Dict]:
    """List model output directories that contain a vector index file.

    Returns dicts with ``name`` (directory name), ``dir`` (path), ``prefix``.
    """
    found: List[Dict] = []
    if not os.path.isdir(output_root):
        return found
    for name in sorted(os.listdir(output_root)):
        d = os.path.join(output_root, name)
        if not os.path.isdir(d):
            continue
        prefix = detect_prefix(d)
        if prefix:
            found.append({"name": name, "dir": d, "prefix": prefix})
    return found


def resolve_model(output_root: str, model: str) -> Dict:
    """Resolve a --model name to a ``{name, dir, prefix}`` record or raise."""
    d = os.path.join(output_root, model)
    if os.path.isdir(d):
        prefix = detect_prefix(d) or MODEL_PREFIXES.get(model)
        if prefix:
            return {"name": model, "dir": d, "prefix": prefix}
    # Fall back to scanning, in case the directory is named after the prefix.
    for rec in discover_models(output_root):
        if rec["name"] == model or rec["prefix"] == MODEL_PREFIXES.get(model, model):
            return rec
    available = [r["name"] for r in discover_models(output_root)]
    raise SystemExit(
        f"No vector outputs for model '{model}' under '{output_root}'. "
        f"Available: {', '.join(available) if available else '(none)'}."
    )


def load_condition(
    model_dir: str, prefix: str, mode: str, pooling: str = "mean",
) -> Optional[Dict]:
    """Load one condition: tidy index frame + aligned f32 vector matrix.

    Returns ``{"index": DataFrame, "vectors": ndarray[M, D]}`` filtered to valid
    vectors of the requested pooling method, or ``None`` if files are missing.
    Vectors are upcast to float32 (downstream code L2-normalises as needed).
    """
    idx_path = os.path.join(model_dir, f"{prefix}{_INDEX_SUFFIX}{mode}.csv")
    npz_path = os.path.join(model_dir, f"{prefix}_vectors_subsample_{mode}_f16.npz")
    if not (os.path.exists(idx_path) and os.path.exists(npz_path)):
        return None

    index = pd.read_csv(idx_path)
    with np.load(npz_path) as data:
        matrix = data["vectors"]

    mask = (index["pooling_method"] == pooling) & (index["vector_valid"])
    sub = index[mask].copy()
    if sub.empty:
        return None
    vectors = matrix[sub["record_idx"].to_numpy()].astype(np.float32)
    sub = sub.reset_index(drop=True)
    return {"index": sub, "vectors": vectors}


def num_layers(index: pd.DataFrame) -> int:
    """Maximum layer index present (the final hidden layer)."""
    return int(index["layer"].max()) if len(index) else 0


def available_layers(index: pd.DataFrame) -> List[int]:
    return sorted(int(x) for x in index["layer"].unique())


def sample_boards(index: pd.DataFrame, n: int, seed: int = 42) -> List[int]:
    """Reproducibly sample ``n`` board (row_id) values from the subsample."""
    ids = np.sort(index["row_id"].unique())
    if len(ids) <= n:
        return [int(x) for x in ids]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(ids, size=n, replace=False)
    return sorted(int(x) for x in chosen)


def board_layer_words(
    cond: Dict, row_id: int, layer: int,
) -> Tuple[List[str], List[str], np.ndarray]:
    """Words, word types, and vectors for one board at one layer."""
    index = cond["index"]
    sel = index[(index["row_id"] == row_id) & (index["layer"] == layer)]
    words = sel["word"].astype(str).tolist()
    types = sel["word_type"].astype(str).tolist()
    vecs = cond["vectors"][sel.index.to_numpy()]
    return words, types, vecs


def load_general(model_dir: str, prefix: str, mode: str) -> pd.DataFrame:
    path = os.path.join(model_dir, f"{prefix}_general_{mode}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def board_meta(general: pd.DataFrame, row_id: int) -> Dict:
    """Extract hint / n_targets / giver_features for titles and captions."""
    if general.empty or "row_id" not in general.columns:
        return {}
    rows = general[general["row_id"] == row_id]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    meta: Dict = {
        "hint": row.get("hint"),
        "n_targets": row.get("n_targets"),
    }
    gf = row.get("giver_features")
    if isinstance(gf, str) and gf.strip().startswith("{"):
        try:
            meta["giver_features"] = ast.literal_eval(gf)
        except (ValueError, SyntaxError):
            meta["giver_features"] = None
    return meta
