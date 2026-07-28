import numpy as np
import pytest

from codenames.causal.analysis import (
    benjamini_hochberg,
    claim_gate,
    cluster_bootstrap_ci,
    paired_contrast,
    permutation_max_null,
)


def test_bh_rejects_the_expected_set():
    """Hand-checked: n=5, q=0.05 -> thresholds 0.01/0.02/0.03/0.04/0.05.

    0.001<=0.01 and 0.008<=0.02 pass; 0.039>0.03 and 0.041>0.04 do not, so the
    largest passing rank is 2 and exactly the two smallest p-values reject.
    """
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    assert benjamini_hochberg(p, q=0.05).tolist() == [True, True, False, False, False]


def test_bh_step_up_rescues_a_ranked_value_below_the_cutoff():
    """Step-up: a mid p-value above its own threshold still rejects if a later
    one passes. This is what distinguishes BH from per-cell thresholding."""
    p = np.array([0.001, 0.04, 0.03])
    assert benjamini_hochberg(p, q=0.05).tolist() == [True, True, True]


def test_bh_is_never_more_liberal_than_uncorrected():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=500)
    assert benjamini_hochberg(p, q=0.05).sum() <= (p < 0.05).sum()


def test_bh_rejects_nothing_under_a_pure_null():
    rng = np.random.default_rng(2026)
    assert benjamini_hochberg(rng.uniform(size=1000), q=0.05).sum() <= 50


def test_bh_handles_the_empty_case():
    assert benjamini_hochberg(np.array([]), q=0.05).tolist() == []


def test_cluster_bootstrap_resamples_turns_not_rows():
    """Two turns, 50 rows each, very different means.

    Resampling ROWS would give a narrow CI; resampling TURNS must give a wide
    one. Turns share boards and hints, so the turn is the correct unit (§3.4).
    """
    values = np.concatenate([np.zeros(50), np.ones(50)])
    turns = np.array([1] * 50 + [2] * 50)
    lo, hi = cluster_bootstrap_ci(values, turns, n_boot=500, seed=2026)
    assert hi - lo > 0.5


def test_cluster_bootstrap_is_seeded():
    rng = np.random.default_rng(0)
    values, turns = rng.normal(size=100), np.arange(100)
    assert cluster_bootstrap_ci(values, turns, n_boot=200, seed=2026) == \
           cluster_bootstrap_ci(values, turns, n_boot=200, seed=2026)


def test_paired_contrast_detects_a_constant_shift():
    rng = np.random.default_rng(2026)
    turns = np.arange(200)
    control = rng.normal(size=200)
    treatment = control + 0.5
    out = paired_contrast(treatment, control, turns, n_boot=500, seed=2026)
    assert out["mean_difference"] == pytest.approx(0.5, abs=0.05)
    assert out["ci_low"] > 0


def test_paired_contrast_finds_no_effect_when_there_is_none():
    rng = np.random.default_rng(7)
    turns = np.arange(300)
    control = rng.normal(size=300)
    out = paired_contrast(control.copy(), control, turns, n_boot=500, seed=2026)
    assert out["mean_difference"] == pytest.approx(0.0, abs=1e-9)
    assert out["ci_low"] <= 0 <= out["ci_high"]


def test_permutation_max_null_is_positive_and_seeded():
    rng = np.random.default_rng(0)
    effects = rng.normal(size=(50, 20))
    a = permutation_max_null(effects, np.arange(50), n_perm=200, seed=2026)
    b = permutation_max_null(effects, np.arange(50), n_perm=200, seed=2026)
    assert a > 0 and a == b


def test_claim_gate_requires_all_four_conditions():
    ok = dict(survives_fdr=True, beats_random_site=True,
              stable_across_orderings=True, confirmed_real_patch=True)
    assert claim_gate(ok) is True
    for key in ok:
        assert claim_gate(dict(ok, **{key: False})) is False


def test_claim_gate_treats_missing_evidence_as_absent():
    assert claim_gate({}) is False
