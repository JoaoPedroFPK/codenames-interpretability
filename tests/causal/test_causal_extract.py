import numpy as np
import pandas as pd
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal.extract import run_corrupted_extraction

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


# Both single-token under the tiny Qwen BPE vocabulary, so the two prompts
# tokenise to identical lengths. Verified by
# test_fixture_hints_are_genuinely_length_matched below.
CLEAN_HINT, DONOR_HINT = "water", "rocket"


def _fixtures():
    df = pd.DataFrame([
        {"row_id": 1, "output": CLEAN_HINT, "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
        {"row_id": 2, "output": DONOR_HINT, "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
    ])
    pairs = pd.DataFrame([
        {"row_id": 1, "donor_row_id": 2, "clean_target": "sea",
         "donor_target": "moon", "hint": CLEAN_HINT, "donor_hint": DONOR_HINT,
         "n_donors": 1},
    ])
    return df, pairs


def test_fixture_hints_are_genuinely_length_matched(tiny):
    """Guards the fixture itself; a mismatched pair would mask real bugs."""
    _, tok = tiny
    assert len(tok.encode(CLEAN_HINT, add_special_tokens=False)) == \
           len(tok.encode(DONOR_HINT, add_special_tokens=False))


def _run(tiny, base_dir, scheme="counterfactual", seed=2026):
    model, tok = tiny
    df, pairs = _fixtures()
    return run_corrupted_extraction(
        model=model, tokenizer=tok, df_sample=df, pair_table=pairs,
        base_dir=str(base_dir), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, device="cpu", scheme=scheme, seed=seed,
    )


def test_counterfactual_scheme_swaps_the_hint_only(tiny, tmp_path):
    out = _run(tiny, tmp_path)
    idx = pd.read_csv(out["index"])
    assert idx.loc[0, "corrupt_hint"] == DONOR_HINT
    assert idx.loc[0, "clean_hint"] == CLEAN_HINT


def test_length_matched_pairs_preserve_token_count(tiny, tmp_path):
    out = _run(tiny, tmp_path)
    idx = pd.read_csv(out["index"])
    assert idx.loc[0, "clean_n_tokens"] == idx.loc[0, "corrupt_n_tokens"], (
        "position alignment broken - patching (layer,pos) requires identical indexing"
    )


def test_noise_scheme_is_deterministic_under_seed(tiny, tmp_path):
    a = _run(tiny, tmp_path / "a", scheme="noise")
    b = _run(tiny, tmp_path / "b", scheme="noise")
    np.testing.assert_allclose(
        np.load(a["hidden"]).astype(np.float32),
        np.load(b["hidden"]).astype(np.float32), rtol=0, atol=0,
    )


def test_noise_scheme_actually_perturbs_relative_to_counterfactual(tiny, tmp_path):
    cf = np.load(_run(tiny, tmp_path / "c")["hidden"]).astype(np.float32)
    nz = np.load(_run(tiny, tmp_path / "n", scheme="noise")["hidden"]).astype(np.float32)
    assert not np.allclose(cf, nz), "the two schemes must not coincide"


def test_clean_cache_is_written_for_the_patching_stage(tiny, tmp_path):
    out = _run(tiny, tmp_path)
    clean = np.load(out["clean_hidden"]).astype(np.float32)
    corrupt = np.load(out["hidden"]).astype(np.float32)
    assert clean.shape == corrupt.shape
    assert not np.allclose(clean, corrupt), "clean and corrupted runs must differ"


def test_unknown_scheme_is_rejected(tiny, tmp_path):
    with pytest.raises(ValueError, match="scheme"):
        _run(tiny, tmp_path, scheme="bogus")


def test_misaligned_pairs_are_dropped_and_logged_not_silently_patched(tiny, tmp_path):
    """Equal standalone hint tokens do NOT guarantee equal prompt tokens.

    A pair whose prompts differ in length must never reach the patch stage:
    patching (layer, position) across different-length sequences silently reads
    the wrong position.
    """
    model, tok = tiny
    df, pairs = _fixtures()
    # "ocean" is 2 tokens here while "water" is 1 -> prompts differ by one.
    df.loc[df.row_id == 2, "output"] = "ocean"
    pairs.loc[0, "donor_hint"] = "ocean"

    out = run_corrupted_extraction(
        model=model, tokenizer=tok, df_sample=df, pair_table=pairs,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, device="cpu",
    )
    assert len(pd.read_csv(out["index"])) == 0
    dropped = pd.read_csv(out["dropped"])
    assert len(dropped) == 1
    assert dropped.loc[0, "reason"] == "prompt_length_mismatch"
    assert np.load(out["hidden"]).shape[0] == 0
