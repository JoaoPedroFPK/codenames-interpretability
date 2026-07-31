"""Offline lens scoring: hidden dump + readout weights -> scores parquet.

Never loads the 7B model. For each layer: fp32 (optionally translated)
states -> RMSNorm -> logits over the union of candidate first-subword ids
only (a [N, U] matmul, U ~ a few hundred) -> per-turn max-over-variants
scores and ranks (readout.py rule).
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .readout import build_token_table, rank_from_scores
from .tuned import TunedLens


@dataclass
class Readout:
    norm_weight: np.ndarray   # [d] fp32
    lm_head: np.ndarray       # [V, d] fp32
    eps: float


def load_readout(path: str) -> Readout:
    z = np.load(path, allow_pickle=False)
    return Readout(norm_weight=z["norm_weight"].astype(np.float32),
                   lm_head=z["lm_head_weight"].astype(np.float32),
                   eps=float(z["eps"]))


def rmsnorm_np(H: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    H = H.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(H * H, axis=-1, keepdims=True) + eps)
    return H / rms * w


def _word_type(word, targets, black):
    if word in targets:
        return "target"
    if word in black:
        return "black"
    return "tan"


def compute_scores(
    hidden_path: str,
    index_path: str,
    df_sample: pd.DataFrame,
    tokenizer,
    readout: Readout,
    lens_name: str,
    translators: Optional[TunedLens] = None,
) -> pd.DataFrame:
    mm = np.load(hidden_path, mmap_mode="r")
    n_boards, n_states, _ = mm.shape
    # The generating channel hands a CSV path; the answer channel hands a
    # frame whose ``ok`` is derived from ``p_star_missing`` so boards without
    # a resolved p* are skipped rather than scored on NaN states.
    index = (index_path if isinstance(index_path, pd.DataFrame)
             else pd.read_csv(index_path))
    by_row_id = df_sample.set_index("row_id")

    # Per-board candidate token tables + the union of variant ids.
    boards = []
    union: List[int] = []
    seen = set()
    for rec in index.to_dict("records"):
        if not rec["ok"]:
            boards.append(None)
            continue
        row = by_row_id.loc[int(rec["row_id"])]
        table = build_token_table(tokenizer, list(row["candidates"]))
        boards.append({"row_id": int(rec["row_id"]), "table": table,
                       "targets": set(row["targets"]),
                       "black": set(row["black"])})
        for ids in table.values():
            for tid in ids:
                if tid not in seen:
                    seen.add(tid)
                    union.append(tid)
    col_of = {tid: j for j, tid in enumerate(union)}
    W = readout.lm_head[union]                     # [U, d]

    frames = []
    for layer in range(n_states):
        H = np.asarray(mm[:len(boards), layer, :], dtype=np.float32)
        if translators is not None:
            H = translators.translate(H, layer)
        logits = rmsnorm_np(H, readout.norm_weight, readout.eps) @ W.T
        recs = []
        for i, board in enumerate(boards):
            if board is None:
                continue
            scores = {
                w: (float(np.max(logits[i, [col_of[t] for t in ids]]))
                    if ids else float("nan"))
                for w, ids in board["table"].items()
            }
            ranks = rank_from_scores(scores)
            for w, s in scores.items():
                recs.append({
                    "row_id": board["row_id"], "layer": layer, "word": w,
                    "word_type": _word_type(w, board["targets"],
                                            board["black"]),
                    "lens": lens_name, "score": s, "rank": ranks[w],
                })
        frames.append(pd.DataFrame(recs))
    return pd.concat(frames, ignore_index=True)


def save_scores(frames: List[pd.DataFrame], base_dir: str, prefix: str,
                mode_name: str) -> str:
    path = os.path.join(base_dir, f"{prefix}_lens_scores_{mode_name}.parquet")
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path
