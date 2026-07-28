"""Statistical plan (causal_spec.md §3.4).

Three commitments this module encodes, each replacing a weaker practice:

1. The resampling unit is the **turn**, not the observation. Turns share boards
   and hints, so observation-level resampling understates variance.
2. Control comparisons are **paired bootstrap contrasts**, not CI overlap.
   Non-overlapping CIs is a conservative proxy and overlapping CIs do not imply
   a null; since treatment and control run on the same turns, the paired
   difference is available and strictly better.
3. Multiplicity is **two-stage**. The attribution scan is screening only and
   carries no inferential claim; stage 2 is BH-FDR at q = 0.05 with a
   permutation max-statistic cross-check. Uncorrected per-cell thresholding
   over a ~4,000-cell grid paints a locus out of noise.
"""

from typing import Dict, Sequence, Tuple

import numpy as np


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Step-up FDR control. Returns a boolean rejection mask."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresholds = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    rejected = np.zeros(n, dtype=bool)
    if passed.any():
        cutoff = int(np.max(np.where(passed)[0]))
        rejected[order[: cutoff + 1]] = True
    return rejected


def cluster_bootstrap_ci(
    values: Sequence[float],
    turn_ids: Sequence[int],
    *,
    n_boot: int = 10000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile CI for the mean, resampling whole turns with replacement."""
    v = np.asarray(values, dtype=float)
    t = np.asarray(turn_ids)
    turns = np.unique(t)
    grouped = [v[t == turn] for turn in turns]

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.integers(0, len(turns), size=len(turns))
        means[i] = np.concatenate([grouped[j] for j in picked]).mean()
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def paired_contrast(
    treatment: Sequence[float],
    control: Sequence[float],
    turn_ids: Sequence[int],
    *,
    n_boot: int = 10000,
    seed: int = 2026,
) -> Dict[str, float]:
    """Cluster-bootstrapped CI on the per-turn treatment-minus-control difference."""
    difference = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    low, high = cluster_bootstrap_ci(difference, turn_ids, n_boot=n_boot, seed=seed)
    return {
        "mean_difference": float(difference.mean()),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(low > 0 or high < 0),
    }


def permutation_max_null(
    effects_by_site: np.ndarray,
    turn_ids: Sequence[int],
    *,
    n_perm: int = 1000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> float:
    """Family-wise threshold: the (1-alpha) quantile of max |mean| across sites.

    ``effects_by_site`` is (n_turns, n_sites). Signs are flipped per turn, which
    is the exchangeable null for a paired clean/corrupted design and preserves
    the spatial correlation across the grid that BH ignores.
    """
    effects = np.asarray(effects_by_site, dtype=float)
    rng = np.random.default_rng(seed)
    maxima = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=effects.shape[0])[:, None]
        maxima[i] = np.abs(np.nanmean(effects * signs, axis=0)).max()
    return float(np.quantile(maxima, 1 - alpha))


def claim_gate(locus: Dict[str, bool]) -> bool:
    """All four §3.4 conditions must hold before a locus is claimed.

    Missing evidence counts as absent, never as satisfied.
    """
    return bool(
        locus.get("survives_fdr", False)
        and locus.get("beats_random_site", False)
        and locus.get("stable_across_orderings", False)
        and locus.get("confirmed_real_patch", False)
    )
