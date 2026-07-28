"""Patching metrics (causal_spec.md §3.2).

LD = logit(clean target first-subword) - logit(counterfactual target
first-subword), both read at p* under the §5 readout rule (max over surface
variants). The scoring call is delegated to codenames.lens.readout so the
readout rule cannot drift between the lens and causal studies.

e = (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt): 0 is no restoration,
1 is full restoration. Under the symmetric counterfactual the denominator is
well conditioned -- measured on the completed Mistral run, LD_clean has mean
+9.32 (SD 7.70, n = 6,212) so the symmetric denominator is about 18.6 logits.

e is deliberately NOT clipped to [0, 1]: values outside that range are real
signal (a patch overshooting the clean run, or moving away from it), and
clipping would silently hide a pipeline fault.
"""

from typing import Dict, List, Sequence

import numpy as np

from ..lens.readout import score_candidates

# Below this the denominator carries no information and e is undefined.
_DEGENERATE = 1e-6


def logit_difference(
    logits: np.ndarray,
    table: Dict[str, List[int]],
    clean_target: str,
    donor_target: str,
) -> float:
    """Clean-target minus donor-target score at one position."""
    scores = score_candidates(np.asarray(logits), table)
    clean = scores.get(clean_target, float("nan"))
    donor = scores.get(donor_target, float("nan"))
    if np.isnan(clean) or np.isnan(donor):
        return float("nan")
    return float(clean - donor)


def normalized_effect(*, ld_patched: float, ld_clean: float, ld_corrupt: float) -> float:
    """Restoration fraction. NaN when the clean/corrupt contrast is degenerate."""
    denominator = ld_clean - ld_corrupt
    if not np.isfinite(denominator) or abs(denominator) < _DEGENERATE:
        return float("nan")
    if not np.isfinite(ld_patched):
        return float("nan")
    return float((ld_patched - ld_corrupt) / denominator)


def effect_series(
    ld_patched: Sequence[float], ld_clean: Sequence[float], ld_corrupt: Sequence[float]
) -> np.ndarray:
    """Vectorised normalized_effect over paired per-turn arrays."""
    patched = np.asarray(ld_patched, dtype=float)
    clean = np.asarray(ld_clean, dtype=float)
    corrupt = np.asarray(ld_corrupt, dtype=float)
    denominator = clean - corrupt
    out = np.full(patched.shape, np.nan, dtype=float)
    usable = np.isfinite(denominator) & (np.abs(denominator) >= _DEGENERATE)
    usable &= np.isfinite(patched)
    out[usable] = (patched[usable] - corrupt[usable]) / denominator[usable]
    return out
