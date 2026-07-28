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


# --- resume (causal-patch is ~94% of the compute budget) --------------------

def test_patch_stage_writes_checkpoints(tiny, tmp_path):
    _, loci = run_scan_stage(**_common(tiny), top_k=1)
    run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,),
                    checkpoint_dir=str(tmp_path), prefix="tiny",
                    flush_every=1)
    from codenames import checkpoint
    assert checkpoint.list_ckpts(str(tmp_path), "tiny_patch", "patch", "no_social")


def test_resume_is_byte_identical_to_an_uninterrupted_run(tiny, tmp_path):
    """The project's standing byte-identity rule for --resume."""
    _, loci = run_scan_stage(**_common(tiny), top_k=2)
    kw = dict(**_common(tiny), loci=loci, window_widths=(1,), prefix="tiny")

    full = run_patch_stage(**kw)

    # Interrupt after the first pair, then resume into the same directory.
    partial_dir = str(tmp_path / "ck")
    run_patch_stage(**kw, checkpoint_dir=partial_dir, flush_every=1,
                    stop_after_pairs=1)
    resumed = run_patch_stage(**kw, checkpoint_dir=partial_dir, flush_every=1,
                              resume=True)

    pd.testing.assert_frame_equal(
        full.sort_values(["row_id", "layer", "position", "width"]).reset_index(drop=True),
        resumed.sort_values(["row_id", "layer", "position", "width"]).reset_index(drop=True),
    )


def test_resume_without_a_checkpoint_starts_clean(tiny, tmp_path):
    _, loci = run_scan_stage(**_common(tiny), top_k=1)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,),
                          checkpoint_dir=str(tmp_path), prefix="tiny", resume=True)
    assert len(out) > 0


def test_resume_skips_work_already_done(tiny, tmp_path):
    _, loci = run_scan_stage(**_common(tiny), top_k=1)
    kw = dict(**_common(tiny), loci=loci, window_widths=(1,), prefix="tiny",
              checkpoint_dir=str(tmp_path), flush_every=1)
    run_patch_stage(**kw, stop_after_pairs=1)

    seen = []
    run_patch_stage(**kw, resume=True, progress=lambda: seen.append(1))
    # the first pair was already checkpointed, so fewer pairs are re-measured
    assert len(seen) < len(run_patch_stage(**_common(tiny), loci=loci,
                                           window_widths=(1,))["row_id"].unique()) + 1


def test_every_steering_arm_shares_the_same_norm_relative_scale(tiny):
    """Spec §5B: alpha multiplies the median residual norm at the injection
    layer. Raw unembedding rows and difference-of-means vectors have unrelated
    scales, so without rescaling the arms are not comparable to each other or
    across models - which is what made Qwen appear inert."""
    out = run_steer_stage(**_common(tiny), layer=1, alphas=(1.0,), max_new_tokens=3)
    assert "residual_scale" in out.columns
    assert (out["residual_scale"] > 0).all()
    # one scale per turn, shared by every arm at that turn
    per_turn = out.groupby("row_id")["residual_scale"].nunique()
    assert (per_turn == 1).all()
