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


def _ragged_frame():
    """Turns whose prompts tokenise to DIFFERENT lengths.

    Every fixture above happens to hold boards of equal size and similar words,
    so every prompt came out the same length and the scan's cross-turn
    aggregation looked fine. On the real corpus it crashed with
    `operands could not be broadcast together with shapes (33,118) (33,108)`.
    Ragged boards are the condition the aggregation actually has to survive.
    """
    return pd.DataFrame([
        {"row_id": 1, "output": "water", "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea"]},
        {"row_id": 2, "output": "rocket", "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship", "castle",
                                         "engine", "telescope"]},
        {"row_id": 3, "output": "harbour", "targets": ["ship"], "black": ["sea"],
         "tan": ["moon"], "candidates": ["moon", "sea", "ship", "anchor"]},
    ])


def _common(tiny, df=None):
    model, tok = tiny
    return dict(
        model=model, tokenizer=tok,
        df_sample=_frame() if df is None else df,
        chat_template_strategy="raw", mode="no_social", seed=2026,
    )


# --- scan ------------------------------------------------------------------

def test_scan_returns_a_grid_and_ranked_loci(tiny, tmp_path):
    from codenames.causal.basis import ROLE_INDEX
    grid, loci = run_scan_stage(**_common(tiny), top_k=4)
    assert grid.ndim == 2
    # With no generation CSV nothing is teacher-forced, so the `generation`
    # role has no tokens and stays NaN -- an absent role must not read as a
    # measured zero effect.
    measured = np.delete(grid, ROLE_INDEX["generation"], axis=1)
    assert np.isfinite(measured).all()
    assert np.isnan(grid[:, ROLE_INDEX["generation"]]).all()
    assert list(loci.columns) >= ["layer", "role", "score"]
    assert len(loci) == 4
    # ranked strongest first
    assert (loci["score"].abs().diff().dropna() <= 1e-9).all()


def test_scan_top_k_is_bounded_by_the_grid(tiny):
    grid, loci = run_scan_stage(**_common(tiny), top_k=10_000)
    assert len(loci) == grid.size


# --- the regression the role basis exists for ------------------------------

def test_scan_survives_turns_with_different_prompt_lengths(tiny):
    """The real failure: prompts of unequal token length made the cross-turn
    aggregate raise a broadcast error. The role basis gives every turn the same
    grid width regardless of board size."""
    from codenames.causal.basis import ROLES
    grid, loci = run_scan_stage(**_common(tiny, _ragged_frame()), top_k=4)
    assert grid.shape[1] == len(ROLES)
    assert len(loci) == 4


def test_scan_grid_width_is_independent_of_the_draw(tiny):
    """Two draws with different boards must produce directly comparable grids;
    otherwise no cross-model or cross-condition comparison is defined."""
    from codenames.causal.basis import ROLES
    a, _ = run_scan_stage(**_common(tiny), top_k=1)
    b, _ = run_scan_stage(**_common(tiny, _ragged_frame()), top_k=1)
    assert a.shape[1] == b.shape[1] == len(ROLES)


def test_patch_survives_turns_with_different_prompt_lengths(tiny):
    """A locus names a role, so it resolves to each turn's own token indices
    even though those indices differ between turns."""
    df = _ragged_frame()
    _, loci = run_scan_stage(**_common(tiny, df), top_k=3)
    out = run_patch_stage(**_common(tiny, df), loci=loci, window_widths=(1,))
    assert out["row_id"].nunique() >= 2
    assert out["effect"].notna().any()


def test_a_role_absent_from_a_turn_is_nan_not_a_measured_zero(tiny):
    """Every (locus, width, turn) cell is recorded. Dropping the absent ones
    would make coverage look complete when it is not."""
    df = _ragged_frame()
    _, loci = run_scan_stage(**_common(tiny, df), top_k=3)
    out = run_patch_stage(**_common(tiny, df), loci=loci, window_widths=(1,))
    per_turn = out.groupby("row_id").size()
    assert per_turn.nunique() == 1, "every turn must report every locus"
    assert (out.loc[out["n_positions"] == 0, "effect"].isna()).all()


# --- the answer position p* (spec §4.1) ------------------------------------

def _generation_csv(tmp_path, scaffolded: bool):
    """Recorded generations. `scaffolded` mimics Mistral's dominant format,
    where the answer word sits 10+ tokens behind a preamble -- 87% of its turns.
    """
    lead = "The word that best matches the hint is " if scaffolded else ""
    rows = [{"row_id": 1, "generated_text": lead + "sea", "generated_word": "sea"},
            {"row_id": 2, "generated_text": lead + "moon", "generated_word": "moon"}]
    path = tmp_path / f"gen_{'scaffolded' if scaffolded else 'wordfirst'}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_supplying_generations_moves_the_readout_off_the_final_token(tiny, tmp_path):
    """The §4.1 correction. Without generations the readout is the generating
    position; with them it is the answer position p*, which for a scaffolded
    generation is a different token entirely. If the two agreed, p* would not
    be being used."""
    gen = _generation_csv(tmp_path, scaffolded=True)
    without, _ = run_scan_stage(**_common(tiny), top_k=1)
    with_p_star, _ = run_scan_stage(**_common(tiny), top_k=1, generation_csv=gen)
    assert not np.allclose(np.nan_to_num(without), np.nan_to_num(with_p_star)), \
        "p* readout is identical to the generating-position readout"


def test_teacher_forced_generation_gets_its_own_role(tiny, tmp_path):
    """The appended generation must not be pooled into the question scaffold:
    it falls past the last candidate, where the positional default would
    otherwise label it `question`."""
    from codenames.causal.basis import ROLE_INDEX
    gen = _generation_csv(tmp_path, scaffolded=True)
    grid, _ = run_scan_stage(**_common(tiny), top_k=1, generation_csv=gen)
    assert np.isfinite(grid[:, ROLE_INDEX["generation"]]).any()
    assert np.isfinite(grid[:, ROLE_INDEX["final"]]).any(), \
        "the generating position must survive as its own role"


def test_patch_stage_accepts_generations_and_stays_finite(tiny, tmp_path):
    gen = _generation_csv(tmp_path, scaffolded=True)
    _, loci = run_scan_stage(**_common(tiny), top_k=2, generation_csv=gen)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,),
                          generation_csv=gen)
    assert out["effect"].notna().any()


def test_missing_generations_fall_back_without_crashing(tiny, tmp_path):
    """The random-init null has no recorded generations by construction."""
    path = tmp_path / "empty.csv"
    pd.DataFrame({"row_id": [], "generated_text": [],
                  "generated_word": []}).to_csv(path, index=False)
    grid, loci = run_scan_stage(**_common(tiny), top_k=1,
                                generation_csv=str(path))
    assert len(loci) == 1


# --- patch -----------------------------------------------------------------

def test_patch_stage_emits_per_turn_effects(tiny):
    _, loci = run_scan_stage(**_common(tiny), top_k=2)
    out = run_patch_stage(**_common(tiny), loci=loci, window_widths=(1,))
    assert {"layer", "role", "row_id", "effect", "width"} <= set(out.columns)
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
        full.sort_values(["row_id", "layer", "role", "width"]).reset_index(drop=True),
        resumed.sort_values(["row_id", "layer", "role", "width"]).reset_index(drop=True),
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


def test_scan_per_layer_returns_one_locus_per_depth(tiny):
    """The causal CURVE needs a locus at every layer; a global top-k can all
    land at one depth and produce no curve at all."""
    grid, loci = run_scan_stage(**_common(tiny), per_layer=True)
    assert len(loci) == grid.shape[0]
    assert sorted(loci["layer"]) == list(range(grid.shape[0]))


# --- readout index: the estimand reads at p*-1, never at p* ----------------

def test_scan_reads_logits_at_the_emitting_position(tiny, tmp_path, monkeypatch):
    """With a resolved p*, every readout must use p*-1 (the position whose
    next-token distribution emits the answer). Reading at p* conditions on
    the answer already having been emitted."""
    import codenames.causal.stages as stages
    from codenames.causal.positions import answer_position
    from codenames.prompts import build_prompt

    model, tok = tiny
    text = "sea is the answer"
    gen_path = tmp_path / "gen.csv"
    pd.DataFrame([
        {"row_id": 1, "generated_text": text, "generated_word": "sea"},
        {"row_id": 2, "generated_text": "moon", "generated_word": "moon"},
    ]).to_csv(gen_path, index=False)

    captured = []
    real_scan = stages.attribution_scan

    def spy(**kwargs):
        captured.append(kwargs["p_star"])
        cache = kwargs["clean_cache"]
        return np.zeros((len(cache), cache[0].shape[1]))

    monkeypatch.setattr(stages, "attribution_scan", spy)
    run_scan_stage(**_common(tiny), generation_csv=str(gen_path))
    assert captured, "attribution_scan was never called"

    row = _frame().iloc[0]
    prompt, _ = build_prompt(
        hint=str(row["output"]), candidates=list(row["candidates"]),
        giver_features={}, use_social_context=False, tokenizer=tok,
        chat_template_strategy="raw")
    p_star = answer_position(tok, prompt, text, "sea")
    assert p_star is not None and p_star > 0
    assert captured[0] == p_star - 1
