import numpy as np
import pandas as pd
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal.stages import run_patch_stage, run_scan_stage, run_steer_stage

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _frame():
    return pd.DataFrame([
        {"row_id": 1, "output": "water", "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
        {"row_id": 2, "output": "rocket", "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
    ])


def _common(tiny):
    model, tok = tiny
    return dict(
        model=model, tokenizer=tok, df_sample=_frame(),
        chat_template_strategy="raw", mode="no_social", seed=2026,
    )


# --- scan ------------------------------------------------------------------

def test_scan_returns_a_grid_and_ranked_loci(tiny, tmp_path):
    grid, loci = run_scan_stage(**_common(tiny), top_k=4)
    assert grid.ndim == 2 and np.isfinite(grid).all()
    assert list(loci.columns) >= ["layer", "position", "score"]
    assert len(loci) == 4
    # ranked strongest first
    assert (loci["score"].abs().diff().dropna() <= 1e-9).all()


def test_scan_top_k_is_bounded_by_the_grid(tiny):
    grid, loci = run_scan_stage(**_common(tiny), top_k=10_000)
    assert len(loci) == grid.size


# --- patch -----------------------------------------------------------------

def test_patch_stage_emits_per_turn_effects(tiny):
    _, loci = run_scan_stage(**_common(tiny), top_k=2)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,))
    assert {"layer", "position", "row_id", "effect", "width"} <= set(out.columns)
    assert len(out) > 0
    # one row per (locus, width, measured pair)
    assert out["row_id"].nunique() >= 1


def test_patch_stage_records_every_requested_window_width(tiny):
    _, loci = run_scan_stage(**_common(tiny), top_k=1)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1, 3))
    assert set(out["width"]) == {1, 3}


def test_patch_stage_effects_are_finite_or_explicitly_nan(tiny):
    _, loci = run_scan_stage(**_common(tiny), top_k=1)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,))
    assert out["effect"].notna().any(), "every effect was NaN"


# --- steer -----------------------------------------------------------------

def test_steer_stage_covers_every_arm_and_alpha(tiny):
    alphas = (-1.0, 1.0)
    out = run_steer_stage(**_common(tiny), layer=1, alphas=alphas, max_new_tokens=3)
    assert {"arm", "alpha", "row_id", "generated", "changed"} <= set(out.columns)
    assert set(out["alpha"]) == set(alphas)
    # the four pre-registered arms of §3.3
    assert {"primary", "random_direction", "shuffled_label",
            "counterfactual_target"} <= set(out["arm"])


def test_steer_stage_is_deterministic(tiny):
    kw = dict(**_common(tiny), layer=1, alphas=(1.0,), max_new_tokens=3)
    a = run_steer_stage(**kw)
    b = run_steer_stage(**kw)
    pd.testing.assert_frame_equal(a, b)
