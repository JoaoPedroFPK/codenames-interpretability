"""Distance-preservation metrics for the 2D projections.

The projection figures reduce high-dimensional, cosine-compared word vectors to
2D. Before such a picture can be cited, we must show the reduction preserved the
original (cosine) neighbourhood structure. This module provides three
complementary, cosine-aware diagnostics:

- **trustworthiness** — penalises points that are close in 2D but were far in the
  original cosine space (false neighbours introduced by the projection);
- **continuity** — the dual: penalises points that were close in cosine space but
  were pushed apart in 2D (true neighbours lost);
- **Shepard correlation** — global Spearman correlation between all pairwise
  cosine distances (high-dim) and Euclidean distances (2D).

All three are in ``[0, 1]`` (Shepard in ``[-1, 1]`` but ~1 when distances are
preserved), higher is better. A perfect isometric embedding scores ~1.0.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def _safe_k(n_points: int, k: int) -> int:
    """Clamp the neighbourhood size so the rank-based metrics are well defined."""
    if n_points <= 2:
        return 1
    return max(1, min(k, (n_points - 1) // 2))


def _cosine_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances (1 - cosine similarity), L2-normalising first."""
    Xn = X.astype(np.float64)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xn / norms
    sim = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return 1.0 - sim


def _rank_matrix(dist: np.ndarray) -> np.ndarray:
    """For each row, the rank (1 = nearest) of every other point by distance.

    The point itself (distance 0 on the diagonal) is given rank 0 and excluded
    from neighbourhoods by callers.
    """
    n = dist.shape[0]
    order = np.argsort(dist, axis=1, kind="stable")
    ranks = np.empty((n, n), dtype=int)
    arange = np.arange(n)
    for i in range(n):
        ranks[i, order[i]] = arange
    return ranks  # diagonal (self) gets rank 0


def trustworthiness_cosine(X: np.ndarray, embedding: np.ndarray, k: int = 5) -> float:
    """Cosine-space trustworthiness via scikit-learn (``metric='cosine'``).

    Uses the original vectors' cosine distances for the high-dim neighbourhoods
    and Euclidean distances for the 2D embedding (sklearn's convention).
    """
    from sklearn.manifold import trustworthiness

    n = X.shape[0]
    kk = _safe_k(n, k)
    return float(trustworthiness(X, embedding, n_neighbors=kk, metric="cosine"))


def continuity_cosine(X: np.ndarray, embedding: np.ndarray, k: int = 5) -> float:
    """Continuity: the dual of trustworthiness.

    Penalises true cosine neighbours that the projection pushed out of the 2D
    k-neighbourhood, weighted by how far out they were ranked in 2D.
    """
    n = X.shape[0]
    kk = _safe_k(n, k)

    d_high = _cosine_distance_matrix(X)
    d_low = squareform(pdist(embedding.astype(np.float64), metric="euclidean"))

    rank_low = _rank_matrix(d_low)
    order_high = np.argsort(d_high, axis=1, kind="stable")

    total = 0.0
    for i in range(n):
        high_nn = order_high[i, 1:kk + 1]          # true neighbours (exclude self)
        low_nn = set(np.argsort(d_low[i], kind="stable")[1:kk + 1].tolist())
        for j in high_nn:
            if j not in low_nn:
                # rank in 2D minus k (how far it was demoted)
                total += (rank_low[i, j] - kk)

    norm = n * kk * (2 * n - 3 * kk - 1)
    if norm <= 0:
        return float("nan")
    return float(1.0 - (2.0 / norm) * total)


def shepard_corr(X: np.ndarray, embedding: np.ndarray) -> float:
    """Spearman correlation between high-dim cosine distances and 2D Euclidean
    distances over all unique pairs (the Shepard-diagram statistic).
    """
    d_high = pdist(_normalize(X), metric="cosine")
    d_low = pdist(embedding.astype(np.float64), metric="euclidean")
    if d_high.size < 2 or np.allclose(d_high, d_high[0]):
        return float("nan")
    rho, _ = spearmanr(d_high, d_low)
    return float(rho)


def _normalize(X: np.ndarray) -> np.ndarray:
    Xn = X.astype(np.float64)
    norms = np.linalg.norm(Xn, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Xn / norms


def all_metrics(X: np.ndarray, embedding: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Compute all three diagnostics for one (X, embedding) pair."""
    return {
        "trustworthiness": trustworthiness_cosine(X, embedding, k),
        "continuity": continuity_cosine(X, embedding, k),
        "shepard": shepard_corr(X, embedding),
    }
