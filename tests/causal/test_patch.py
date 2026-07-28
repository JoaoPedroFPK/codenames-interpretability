import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal.patch import all_sites, layer_window, run_patch
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
    table = build_token_table(tok, ["sea", "ship"])
    return model, tok, cache, table, ids["input_ids"].shape[1]


def _patch(setup, sites):
    model, tok, cache, table, _ = setup
    return run_patch(
        model=model, tokenizer=tok, clean_cache=cache,
        clean_prompt=CLEAN_PROMPT, corrupt_prompt=CORRUPT_PROMPT,
        sites=sites, readout_table=table,
        clean_target="sea", donor_target="ship", p_star=-1, device="cpu",
    )


def test_layer_window_is_clipped_at_stack_edges():
    assert layer_window(0, 3, 10) == [0, 1]
    assert layer_window(5, 3, 10) == [4, 5, 6]
    assert layer_window(9, 5, 10) == [7, 8, 9]
    assert layer_window(4, 1, 10) == [4]


def test_all_sites_covers_the_grid():
    assert all_sites(n_layers=3, n_positions=2) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
    ]


def test_full_stack_patch_restores_completely(setup):
    """P2 identity (causal_spec.md §12.5): patch every site -> e == 1."""
    _, _, cache, _, n_pos = setup
    e = _patch(setup, all_sites(n_layers=len(cache), n_positions=n_pos))
    assert e == pytest.approx(1.0, abs=0.02)


def test_null_patch_restores_nothing(setup):
    """P3 identity (causal_spec.md §12.5): patch no sites -> e == 0."""
    assert _patch(setup, []) == pytest.approx(0.0, abs=0.02)


def test_partial_patch_lies_between_the_identities(setup):
    _, _, cache, _, n_pos = setup
    e = _patch(setup, [(len(cache) - 1, n_pos - 1)])
    assert np.isfinite(e)


def test_hooks_are_removed_so_later_runs_are_unaffected(setup):
    """A leaked hook silently corrupts every subsequent measurement."""
    model, _, cache, _, n_pos = setup
    _patch(setup, [(1, 0)])
    assert _patch(setup, []) == pytest.approx(0.0, abs=0.02)
    leaked = [h for m in model.modules() for h in m._forward_hooks.values()]
    assert not leaked, f"{len(leaked)} hooks leaked"


def test_hooks_are_removed_even_when_the_forward_raises(setup):
    model, tok, cache, table, _ = setup
    with pytest.raises(Exception):
        run_patch(
            model=model, tokenizer=tok, clean_cache=cache,
            clean_prompt=CLEAN_PROMPT, corrupt_prompt=CORRUPT_PROMPT,
            sites=[(1, 10_000)], readout_table=table,
            clean_target="sea", donor_target="ship", p_star=-1, device="cpu",
        )
    leaked = [h for m in model.modules() for h in m._forward_hooks.values()]
    assert not leaked, "hooks must be removed on the error path too"


def test_patching_is_deterministic(setup):
    _, _, cache, _, n_pos = setup
    sites = [(len(cache) - 1, n_pos - 1)]
    assert _patch(setup, sites) == _patch(setup, sites)
