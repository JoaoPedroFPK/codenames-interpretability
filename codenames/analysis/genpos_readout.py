"""Generating-position cosine readout (venue plan A3; thesis §6.8/§7.4).

The thesis measures hint→candidate cosine geometry and explains the 40–45pp
generation–geometry gap with a conjecture: generation reads the relation
between the GENERATING position and the candidates, not between the hint
position and the candidates. This module measures that relation directly —
per layer, cosine between the generating-position residual state (the lens
dump) and each candidate's pooled span vector, top-1 against the human
targets — turning the thesis's most load-bearing explanatory sentence into a
measured curve.

Offline: joins the ``{prefix}_lens_hidden_{mode}_f16.npy`` dump (full corpus)
to candidate vectors. Wave 1 runs on the thesis's 100-board pooled vector
subsample; the candidate-span dump (causal subsample, lens_spec.md §5) scales
it to n = 1,500 once extracted.
"""

from typing import Optional

import numpy as np
import pandas as pd

_TARGET_TYPES = {"target", "targets"}


def genpos_readout_curve(
    gen_hidden,
    gen_index: pd.DataFrame,
    vec_matrix,
    vec_index: pd.DataFrame,
    *,
    pooling: str = "mean",
) -> pd.DataFrame:
    """Per-layer top-1 accuracy of the generating-position→candidate cosine.

    Parameters
    ----------
    gen_hidden
        ``(n_boards, n_layers+1, hidden)`` array or memmap — the lens dump.
    gen_index
        Its index frame (``board_idx``, ``row_id``, ``ok``).
    vec_matrix / vec_index
        The vector-subsample matrix and its index frame (``record_idx``,
        ``row_id``, ``layer``, ``word``, ``word_type``, ``pooling_method``,
        ``vector_valid``). Hint rows are excluded — the readout ranks the
        candidate pool only, mirroring the thesis's cosine-rank metrics.

    Returns
    -------
    DataFrame with ``layer``, ``top1``, ``n_boards``.
    """
    ok = gen_index[gen_index["ok"].astype(bool)]
    board_of = dict(zip(ok["row_id"].astype(int), ok["board_idx"].astype(int)))

    cand = vec_index[
        (vec_index["pooling_method"] == pooling)
        & vec_index["vector_valid"].astype(bool)
        & ~vec_index["word_type"].isin({"hint"})
    ]
    cand = cand[cand["row_id"].astype(int).isin(board_of)]

    rows = []
    for layer, frame in cand.groupby("layer"):
        hits, n = 0, 0
        for row_id, board in frame.groupby("row_id"):
            b = board_of[int(row_id)]
            state = np.asarray(gen_hidden[b, int(layer)], dtype=np.float32)
            norm = np.linalg.norm(state)
            if norm == 0.0 or len(board) < 2:
                continue
            vecs = np.asarray(
                vec_matrix[board["record_idx"].to_numpy()], dtype=np.float32)
            vnorms = np.linalg.norm(vecs, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cos = np.where(
                    vnorms > 0, vecs @ state / (vnorms * norm), 0.0)
            best = board.iloc[int(np.argmax(cos))]
            hits += best["word_type"] in _TARGET_TYPES
            n += 1
        if n:
            rows.append({"layer": int(layer), "top1": hits / n, "n_boards": n})
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)
