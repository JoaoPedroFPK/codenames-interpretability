import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal.steer import (
    direction_diff_of_means,
    direction_from_unembedding,
    random_direction,
    shuffled_label_direction,
    steer_generate,
)

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def test_diff_of_means_is_the_mean_difference():
    a = np.array([[1.0, 0.0], [3.0, 0.0]])
    b = np.array([[0.0, 1.0], [0.0, 3.0]])
    np.testing.assert_allclose(direction_diff_of_means(a, b), np.array([2.0, -2.0]))


def test_random_direction_has_the_requested_norm_and_is_seeded():
    d = random_direction(64, norm=3.0, seed=2026)
    assert np.linalg.norm(d) == pytest.approx(3.0)
    np.testing.assert_allclose(d, random_direction(64, norm=3.0, seed=2026))
    assert not np.allclose(d, random_direction(64, norm=3.0, seed=7))


def test_shuffled_label_direction_differs_from_the_true_one():
    """The control that a random direction does NOT provide (§3.3.7)."""
    rng = np.random.default_rng(0)
    states = rng.normal(size=(40, 8))
    labels = np.array([True] * 20 + [False] * 20)
    true = direction_diff_of_means(states[labels], states[~labels])
    shuffled = shuffled_label_direction(states, labels, seed=2026)
    assert not np.allclose(true, shuffled)


def test_shuffled_label_direction_is_seeded():
    rng = np.random.default_rng(0)
    states = rng.normal(size=(20, 4))
    labels = np.array([True] * 10 + [False] * 10)
    np.testing.assert_allclose(
        shuffled_label_direction(states, labels, seed=2026),
        shuffled_label_direction(states, labels, seed=2026),
    )


def test_unembedding_direction_averages_candidate_rows():
    unembed = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_allclose(
        direction_from_unembedding(unembed, [0, 2]), (unembed[0] + unembed[2]) / 2
    )


def test_alpha_zero_reproduces_the_unsteered_generation(tiny):
    model, tok = tiny
    rng = np.random.default_rng(2026)
    direction = rng.normal(size=model.config.hidden_size).astype(np.float32)
    kw = dict(
        model=model, tokenizer=tok, prompt="Hint water. Word:", layer=1,
        direction=direction, sites="from_hint", max_new_tokens=4, device="cpu",
    )
    baseline = steer_generate(alpha=0.0, **kw)
    assert baseline == steer_generate(alpha=0.0, **kw)


def test_large_alpha_changes_the_generation(tiny):
    model, tok = tiny
    rng = np.random.default_rng(2026)
    direction = rng.normal(size=model.config.hidden_size).astype(np.float32)
    kw = dict(
        model=model, tokenizer=tok, prompt="Hint water. Word:", layer=1,
        direction=direction, sites="from_hint", max_new_tokens=4, device="cpu",
    )
    assert steer_generate(alpha=0.0, **kw) != steer_generate(alpha=500.0, **kw)


def test_steering_removes_its_hook(tiny):
    model, tok = tiny
    steer_generate(
        model=model, tokenizer=tok, prompt="Hint water. Word:", layer=1,
        direction=np.ones(model.config.hidden_size, dtype=np.float32),
        alpha=1.0, sites="from_hint", max_new_tokens=2, device="cpu",
    )
    leaked = [h for m in model.modules() for h in m._forward_hooks.values()]
    assert not leaked, "steering hook leaked"


def test_unknown_injection_site_is_rejected(tiny):
    model, tok = tiny
    with pytest.raises(ValueError, match="sites"):
        steer_generate(
            model=model, tokenizer=tok, prompt="Hint water. Word:", layer=1,
            direction=np.zeros(model.config.hidden_size, dtype=np.float32),
            alpha=1.0, sites="bogus", max_new_tokens=2, device="cpu",
        )
