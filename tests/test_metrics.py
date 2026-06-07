"""Tests for the contract and the metric helpers that don't require a model.

The full extraction pipeline (run_instance) is not unit-tested here — it
requires a model and tokenizer, and the bit-identity verification happens
in `notebooks/00_validation.ipynb`. These tests cover the pieces that can
be exercised without GPU.
"""

import math

from codenames.contract import CONTRACT_V1, Contract
from codenames.sanity import _wilson_confidence_interval


def test_contract_is_frozen():
    """Contract is frozen; mutating a field raises."""
    import dataclasses
    try:
        CONTRACT_V1.sample_size = 9999  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    except AttributeError:
        return  # some Python versions raise AttributeError on frozen mutation
    raise AssertionError("Contract should be frozen but accepted a mutation.")


def test_contract_v1_default_values():
    """CONTRACT_V1 matches its frozen, canonical parameter values."""
    c = CONTRACT_V1
    assert c.sample_size == 2000
    assert c.candidate_order == "fixed"
    assert c.pooling_methods == ("mean", "max_norm")
    assert c.vector_subsample_size == 100
    assert c.n_shuffles == 2
    assert c.generation_max_tokens == 30
    assert c.shard_boards == 200
    assert c.random_seed == 42
    assert c.max_seq_len == 512


def test_contract_construct_with_override():
    """A new Contract instance with an override is independent of CONTRACT_V1."""
    c = Contract(sample_size=50)
    assert c.sample_size == 50
    assert CONTRACT_V1.sample_size == 2000


def test_wilson_zero_n():
    """N=0 yields a degenerate [0, 0] interval."""
    assert _wilson_confidence_interval(0, 0) == (0.0, 0.0)


def test_wilson_perfect_score():
    """N successes out of N gives an upper end at or just below 1.0."""
    lo, hi = _wilson_confidence_interval(10, 10)
    assert 0 < lo < 1.0
    assert math.isclose(hi, 1.0, rel_tol=1e-6) or hi <= 1.0
    assert hi <= 1.0


def test_wilson_half():
    """50% accuracy is symmetric about 0.5 (to first order)."""
    lo, hi = _wilson_confidence_interval(50, 100)
    midpoint = (lo + hi) / 2
    assert abs(midpoint - 0.5) < 0.02
