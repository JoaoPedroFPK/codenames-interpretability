import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal.attribution import attribution_scan, top_sites
from codenames.lens.readout import build_token_table

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

CLEAN_PROMPT = "Hint water. Words: sea ship. Answer:"
CORRUPT_PROMPT = "Hint rocket. Words: sea ship. Answer:"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@pytest.fixture(scope="module")
def setup(tiny):
    model, tok = tiny
    ids = tok(CLEAN_PROMPT, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    cache = [h.detach().clone() for h in out.hidden_states]
    return model, tok, cache, build_token_table(tok, ["sea", "ship"])


def _scan(setup):
    model, tok, cache, table = setup
    return attribution_scan(
        model=model, tokenizer=tok, clean_cache=cache,
        corrupt_prompt=CORRUPT_PROMPT, readout_table=table,
        clean_target="sea", donor_target="ship", p_star=-1, device="cpu",
    )


def test_scan_shape_matches_the_grid(setup, tiny):
    model, tok, cache, _ = setup
    grid = _scan(setup)
    n_pos = tok(CORRUPT_PROMPT, return_tensors="pt")["input_ids"].shape[1]
    assert grid.shape == (len(cache), n_pos)
    assert np.isfinite(grid).all()


def test_scan_is_deterministic(setup):
    np.testing.assert_allclose(_scan(setup), _scan(setup), rtol=0, atol=0)


def test_scan_leaves_no_hooks_and_no_grads(setup):
    model, _, _, _ = setup
    _scan(setup)
    leaked = [h for m in model.modules() for h in m._forward_hooks.values()]
    assert not leaked, "attribution hooks leaked"
    assert all(p.grad is None for p in model.parameters()), "parameter grads left dirty"


def test_scan_is_not_all_zero(setup):
    """A silently-zero grid would screen out every site."""
    assert np.abs(_scan(setup)).max() > 0


def test_top_sites_returns_the_largest_by_absolute_score():
    grid = np.array([[0.0, -5.0], [2.0, 1.0]])
    assert top_sites(grid, k=2) == [(0, 1), (1, 0)]


def test_top_sites_is_bounded_by_grid_size():
    grid = np.zeros((2, 2))
    assert len(top_sites(grid, k=99)) == 4
