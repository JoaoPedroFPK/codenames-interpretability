"""Pilot gate P1-P8 (causal_spec.md §12.5).

These are MACHINERY checks, not hypothesis tests. They exist because every
number in §3-§4 is a design commitment and none of them establishes that the
implementation does what the design says: a mis-indexed p*, a corruption that
does not corrupt, a hook writing the wrong tensor, or an attribution scan
uncorrelated with the real patches it screens for would all survive
pre-registration and silently produce confident nonsense.

**Anti-peeking rule (binding, §12.5).** Only two classes of pilot output may
inform the confirmatory run: nuisance parameters (throughput, storage,
per-turn variance for re-calibrating the power target) and the binary verdicts
below. The pilot may NOT narrow the confirmatory grid, choose which layers to
test, set K, or fix the direction of any hypothesis. Pilot effect LOCATIONS
are discarded; the confirmatory run re-discovers them or it does not.

**Outcome routing.** P1-P4 and P7 are blocking. P6 failing does not block --
it converts RQ2 into a pre-registered bounded negative (§2.1 row 4). P5
failing does not block either, but it removes the attribution shortcut and so
materially raises the budget, which is re-costed before proceeding.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

PILOT_THRESHOLDS = {
    "P1": 0.99,                        # p* reproduces the recorded generated token
    "P2_lo": 0.98, "P2_hi": 1.02,      # full-stack patch identity: e == 1
    "P3": 0.02,                        # null patch identity: |e| == 0
    "P4_flip": 0.60, "P4_sign": 0.80,  # corruption actually corrupts
    "P5_rho": 0.5, "P5_fnr": 0.20,     # attribution tracks real patches
    "P6_change": 0.10, "P6_parse": 0.90,  # steering is not inert
}

_BLOCKING = ("P1", "P2", "P3", "P4", "P7")


def pilot_verdict(results: Dict[str, float]) -> Dict[str, object]:
    """Route P1-P8 observations into the §12.5 launch decision."""
    t = PILOT_THRESHOLDS
    failed: List[str] = []

    if float(results.get("P1", 0.0)) < t["P1"]:
        failed.append("P1")
    if not (t["P2_lo"] <= float(results.get("P2", 0.0)) <= t["P2_hi"]):
        failed.append("P2")
    if abs(float(results.get("P3", 1.0))) > t["P3"]:
        failed.append("P3")
    if (float(results.get("P4_flip", 0.0)) < t["P4_flip"]
            or float(results.get("P4_sign", 0.0)) < t["P4_sign"]):
        failed.append("P4")
    if not bool(results.get("P7_finite", False)):
        failed.append("P7")

    attribution_lost = (
        float(results.get("P5_rho", 0.0)) < t["P5_rho"]
        or float(results.get("P5_fnr", 1.0)) > t["P5_fnr"]
    )
    rq2_bounded = (
        float(results.get("P6_change", 0.0)) < t["P6_change"]
        or float(results.get("P6_parse", 0.0)) < t["P6_parse"]
    )

    return {
        "launch_full_run": len(failed) == 0,
        "blocking_failures": failed,
        "attribution_shortcut_lost": bool(attribution_lost),
        "rq2_bounded_negative": bool(rq2_bounded),
    }


def pilot_report(results: Dict[str, float]) -> pd.DataFrame:
    """The P1-P8 table handed to the resourcing conversation (§12.6)."""
    t = PILOT_THRESHOLDS
    verdict = pilot_verdict(results)
    failed = set(verdict["blocking_failures"])

    rows = [
        {"check": "P1", "what": "p* reproduces the generated token",
         "observed": results.get("P1"), "threshold": f">= {t['P1']}",
         "passed": "P1" not in failed, "blocking": True},
        {"check": "P2", "what": "full-stack patch identity (e == 1)",
         "observed": results.get("P2"),
         "threshold": f"[{t['P2_lo']}, {t['P2_hi']}]",
         "passed": "P2" not in failed, "blocking": True},
        {"check": "P3", "what": "null patch identity (e == 0)",
         "observed": results.get("P3"), "threshold": f"|e| <= {t['P3']}",
         "passed": "P3" not in failed, "blocking": True},
        {"check": "P4", "what": "corruption flips the answer",
         "observed": results.get("P4_flip"),
         "threshold": f">= {t['P4_flip']} flip, >= {t['P4_sign']} sign",
         "passed": "P4" not in failed, "blocking": True},
        {"check": "P5", "what": "attribution tracks real patches",
         "observed": results.get("P5_rho"),
         "threshold": f"rho >= {t['P5_rho']}, FNR <= {t['P5_fnr']}",
         "passed": not verdict["attribution_shortcut_lost"], "blocking": False},
        {"check": "P6", "what": "steering is not inert",
         "observed": results.get("P6_change"),
         "threshold": f">= {t['P6_change']} at parse >= {t['P6_parse']}",
         "passed": not verdict["rq2_bounded_negative"], "blocking": False},
        {"check": "P7", "what": "random-init numerics are usable",
         "observed": results.get("P7_finite"), "threshold": "finite, no NaNs",
         "passed": "P7" not in failed, "blocking": True},
        {"check": "P8", "what": "throughput and storage",
         "observed": results.get("P8_fwd_per_s"), "threshold": "recorded",
         "passed": True, "blocking": False},
    ]
    return pd.DataFrame(rows)


def measure_p1(answer_index: pd.DataFrame) -> float:
    """Fraction of turns with a resolvable p*, among those with a parsed word."""
    if answer_index.empty:
        return 0.0
    return float((~answer_index["p_star_missing"].astype(bool)).mean())


def measure_p4(ld_corrupt: np.ndarray, flipped: np.ndarray) -> Dict[str, float]:
    """Manipulation check: the corruption must move the answer to the donor."""
    ld = np.asarray(ld_corrupt, dtype=float)
    finite = np.isfinite(ld)
    return {
        "P4_flip": float(np.mean(np.asarray(flipped, dtype=bool))) if flipped.size else 0.0,
        "P4_sign": float(np.mean(ld[finite] < 0)) if finite.any() else 0.0,
    }
