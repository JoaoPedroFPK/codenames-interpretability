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


def bootstrap_means(values: Sequence[float], *, n_boot: int = 10000,
                    seed: int = 2026) -> np.ndarray:
    """Bootstrap distribution of the mean when each turn contributes ONE value.

    That is the shape of the role x layer grid (one row per turn per cell), so
    resampling values IS resampling turns and the loop in
    ``cluster_bootstrap_ci`` can be vectorised.
    """
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    return v[idx].mean(axis=1)


def bootstrap_p_two_sided(boot_means: np.ndarray) -> float:
    """Percentile-bootstrap two-sided p-value for H0: mean == 0."""
    b = np.asarray(boot_means, dtype=float)
    if b.size == 0:
        return float("nan")
    lower = float(np.mean(b <= 0.0))
    upper = float(np.mean(b >= 0.0))
    return float(min(1.0, 2.0 * min(lower, upper)))


def analyze_grid(grid, nulls, *, n_boot: int = 10000, n_perm: int = 1000,
                 q: float = 0.05, seed: int = 2026, alpha: float = 0.05):
    """Per-cell table for the role x layer grid (§3.4).

    Columns: mean effect with a turn-bootstrap CI and a REAL two-sided
    bootstrap p-value; BH across all cells; the paired contrast against the
    random-site null matched on token count (join on ``row_id`` and
    ``n_positions == matched_n``); and the sign-flip permutation
    max-statistic threshold per width (family-wise, conservative cross-check).
    """
    import pandas as pd

    grid = grid.copy()
    nulls = nulls.copy()
    key = ["row_id", "layer", "width"]
    null_lookup = nulls.rename(columns={"effect": "null_effect"})[
        key + ["matched_n", "null_effect"]]
    merged = grid.merge(null_lookup, how="left", left_on=key + ["n_positions"],
                        right_on=key + ["matched_n"])

    rows = []
    for (layer, role, width), block in merged.groupby(["layer", "role", "width"]):
        e = block["effect"].to_numpy(dtype=float)
        finite = np.isfinite(e)
        if not finite.any():
            rows.append({"layer": int(layer), "role": str(role), "width": int(width),
                         "n_turns": 0, "mean_effect": np.nan, "ci_low": np.nan,
                         "ci_high": np.nan, "p_boot": np.nan,
                         "n_paired": 0, "random_site_mean": np.nan, "paired_diff": np.nan,
                         "paired_ci_low": np.nan, "paired_ci_high": np.nan,
                         "beats_random_site": False})
            continue
        boot = bootstrap_means(e[finite], n_boot=n_boot, seed=seed)
        row = {"layer": int(layer), "role": str(role), "width": int(width),
               "n_turns": int(finite.sum()), "mean_effect": float(e[finite].mean()),
               "ci_low": float(np.quantile(boot, alpha / 2)),
               "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
               "p_boot": bootstrap_p_two_sided(boot)}
        null_e = block["null_effect"].to_numpy(dtype=float)
        paired = finite & np.isfinite(null_e)
        if paired.any():
            diff = e[paired] - null_e[paired]
            pboot = bootstrap_means(diff, n_boot=n_boot, seed=seed)
            lo, hi = float(np.quantile(pboot, alpha / 2)), float(np.quantile(pboot, 1 - alpha / 2))
            row.update({"n_paired": int(paired.sum()),
                        "random_site_mean": float(null_e[paired].mean()),
                        "paired_diff": float(diff.mean()),
                        "paired_ci_low": lo, "paired_ci_high": hi,
                        "beats_random_site": bool(lo > 0)})
        else:
            row.update({"n_paired": 0, "random_site_mean": np.nan, "paired_diff": np.nan,
                        "paired_ci_low": np.nan, "paired_ci_high": np.nan,
                        "beats_random_site": False})
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table

    tested = table["p_boot"].notna()
    table["survives_fdr"] = False
    table.loc[tested, "survives_fdr"] = benjamini_hochberg(table.loc[tested, "p_boot"], q=q)

    # Permutation max-statistic per width over the (turn x cell) matrix.
    table["perm_threshold"] = np.nan
    for width, block in grid.groupby("width"):
        wide = block.pivot_table(index="row_id", columns=["layer", "role"],
                                 values="effect", aggfunc="first")
        thr = permutation_max_null(wide.to_numpy(dtype=float), wide.index.to_numpy(),
                                   n_perm=n_perm, seed=seed, alpha=alpha)
        table.loc[table["width"] == width, "perm_threshold"] = thr
    table["exceeds_perm_threshold"] = (
        table["mean_effect"].abs() > table["perm_threshold"]).fillna(False)
    return table.sort_values(["width", "layer", "role"]).reset_index(drop=True)
