import numpy as np
import pytest

from codenames.causal.metrics import logit_difference, normalized_effect


def test_logit_difference_uses_max_over_surface_variants():
    logits = np.zeros(10, dtype=np.float32)
    logits[3] = 5.0     # a "sea" variant
    logits[7] = 2.0     # a "moon" variant
    table = {"sea": [3, 4], "moon": [7]}
    assert logit_difference(logits, table, "sea", "moon") == pytest.approx(3.0)


def test_logit_difference_sign_flips_with_argument_order():
    logits = np.zeros(10, dtype=np.float32)
    logits[3], logits[7] = 5.0, 2.0
    table = {"sea": [3], "moon": [7]}
    a = logit_difference(logits, table, "sea", "moon")
    b = logit_difference(logits, table, "moon", "sea")
    assert a == pytest.approx(-b)


def test_normalized_effect_identities():
    # Patch restores nothing -> 0 ; restores fully -> 1. These are exact.
    assert normalized_effect(ld_patched=-9.0, ld_clean=9.0, ld_corrupt=-9.0) == pytest.approx(0.0)
    assert normalized_effect(ld_patched=9.0, ld_clean=9.0, ld_corrupt=-9.0) == pytest.approx(1.0)
    assert normalized_effect(ld_patched=0.0, ld_clean=9.0, ld_corrupt=-9.0) == pytest.approx(0.5)


def test_normalized_effect_is_nan_when_denominator_degenerate():
    assert np.isnan(normalized_effect(ld_patched=1.0, ld_clean=2.0, ld_corrupt=2.0))


def test_missing_token_ids_yield_nan_not_zero():
    logits = np.zeros(10, dtype=np.float32)
    table = {"sea": [], "moon": [7]}
    assert np.isnan(logit_difference(logits, table, "sea", "moon"))


def test_normalized_effect_can_exceed_the_unit_interval():
    """Overshoot is real signal, not an error: patching can push past clean."""
    e = normalized_effect(ld_patched=12.0, ld_clean=9.0, ld_corrupt=-9.0)
    assert e > 1.0
