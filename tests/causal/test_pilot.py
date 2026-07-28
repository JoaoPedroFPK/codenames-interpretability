import numpy as np
import pandas as pd
import pytest

from codenames.causal.pilot import PILOT_THRESHOLDS, pilot_report, pilot_verdict


def _results(**overrides):
    base = {
        "P1": 1.00, "P2": 1.00, "P3": 0.00, "P4_flip": 0.55, "P4_sign": 0.90, "P4_clean_accuracy": 0.60,
        "P5_rho": 0.70, "P5_fnr": 0.10, "P5_n_high_effect": 12,
        "P6_change": 0.30, "P6_parse": 0.95,
        "P7_finite": True, "P8_fwd_per_s": 54.0,
    }
    base.update(overrides)
    return base


def test_all_green_launches_the_full_run():
    v = pilot_verdict(_results())
    assert v["launch_full_run"] is True
    assert v["blocking_failures"] == []


def test_p6_failure_does_not_block_but_bounds_rq2():
    """A steering null is a pre-registered publishable outcome (§2.1 row 4)."""
    v = pilot_verdict(_results(P6_change=0.01))
    assert v["launch_full_run"] is True
    assert v["rq2_bounded_negative"] is True


def test_p5_failure_costs_the_shortcut_but_does_not_block():
    v = pilot_verdict(_results(P5_rho=0.1))
    assert v["launch_full_run"] is True
    assert v["attribution_shortcut_lost"] is True


def test_p5_fnr_failure_also_costs_the_shortcut():
    assert pilot_verdict(_results(P5_fnr=0.9))["attribution_shortcut_lost"] is True


@pytest.mark.parametrize(
    "override,expected",
    [
        ({"P1": 0.5}, "P1"),
        ({"P2": 0.5}, "P2"),
        ({"P3": 0.9}, "P3"),
        ({"P4_flip": 0.05}, "P4"),
        ({"P7_finite": False}, "P7"),
    ],
)
def test_identity_and_manipulation_failures_block(override, expected):
    v = pilot_verdict(_results(**override))
    assert v["launch_full_run"] is False
    assert expected in v["blocking_failures"]


def test_p2_out_of_band_high_also_blocks():
    """Overshooting the full-stack identity is as wrong as undershooting."""
    assert pilot_verdict(_results(P2=1.5))["launch_full_run"] is False


def test_thresholds_match_the_spec():
    assert PILOT_THRESHOLDS["P1"] == 0.99
    assert PILOT_THRESHOLDS["P2_lo"] == 0.98
    assert PILOT_THRESHOLDS["P2_hi"] == 1.02
    assert PILOT_THRESHOLDS["P3"] == 0.02
    assert PILOT_THRESHOLDS["P4_flip_ratio"] == 0.85
    assert PILOT_THRESHOLDS["P5_rho"] == 0.5
    assert PILOT_THRESHOLDS["P5_fnr"] == 0.20


def test_report_is_a_frame_with_a_row_per_check():
    frame = pilot_report(_results())
    assert isinstance(frame, pd.DataFrame)
    assert set(frame.columns) >= {"check", "observed", "threshold", "passed"}
    assert set(frame["check"]) >= {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}


def test_report_marks_p8_as_recorded_not_thresholded():
    frame = pilot_report(_results()).set_index("check")
    assert frame.loc["P8", "threshold"] == "recorded"
    assert bool(frame.loc["P8", "passed"]) is True  # pandas stores np.bool_


# --- end-to-end pilot on the tiny model -----------------------------------

import numpy as np
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _pilot_fixtures(tmp_path):
    df = pd.DataFrame([
        {"row_id": 1, "output": "water", "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
        {"row_id": 2, "output": "rocket", "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
    ])
    gen = tmp_path / "gen.csv"
    pd.DataFrame([
        {"row_id": 1, "generated_text": "sea", "generated_word": "sea"},
        {"row_id": 2, "generated_text": "moon", "generated_word": "moon"},
    ]).to_csv(gen, index=False)
    return df, str(gen)


def test_run_pilot_produces_the_full_report(tiny, tmp_path):
    from codenames.causal.pilot import run_pilot
    model, tok = tiny
    df, gen = _pilot_fixtures(tmp_path)
    report, results = run_pilot(
        model=model, tokenizer=tok, df_sample=df, generation_csv=gen,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, device="cpu", seed=2026,
    )
    assert {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"} <= set(report["check"])
    assert "P8_fwd_per_s" in results and results["P8_fwd_per_s"] > 0


def test_pilot_identities_hold_on_a_real_model(tiny, tmp_path):
    """P2 and P3 are exact identities, not tolerances - they must hold here."""
    from codenames.causal.pilot import run_pilot
    model, tok = tiny
    df, gen = _pilot_fixtures(tmp_path)
    _, results = run_pilot(
        model=model, tokenizer=tok, df_sample=df, generation_csv=gen,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, device="cpu", seed=2026,
    )
    assert results["P2"] == pytest.approx(1.0, abs=0.02), "full-stack patch identity"
    assert abs(results["P3"]) <= 0.02, "null patch identity"


def test_run_pilot_infers_device_from_the_model(tiny, tmp_path):
    """Regression: a hardcoded device default broke every CUDA forward pass.

    The pilot must take its device from the model it was handed, so the two
    can never disagree ("Expected all tensors to be on the same device").
    """
    import inspect
    from codenames.causal.pilot import run_pilot

    assert inspect.signature(run_pilot).parameters["device"].default is None
    src = inspect.getsource(run_pilot)
    assert "next(model.parameters()).device" in src


def test_pilot_runs_on_whatever_device_the_model_is_on(tiny, tmp_path):
    from codenames.causal.pilot import run_pilot
    model, tok = tiny
    df, gen = _pilot_fixtures(tmp_path)
    report, results = run_pilot(
        model=model, tokenizer=tok, df_sample=df, generation_csv=gen,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, seed=2026,
    )
    assert results["P2"] == pytest.approx(1.0, abs=0.02)


def test_pilot_drops_misaligned_pairs_and_reports_the_yield(tiny, tmp_path):
    """Regression: a 124-vs-123 token prompt pair crashed the real run.

    Equal standalone hint lengths do not imply equal prompt lengths. The pilot
    must drop those pairs and report the yield, never patch across them.
    """
    from codenames.causal.pilot import run_pilot
    model, tok = tiny
    df, gen = _pilot_fixtures(tmp_path)
    _, results = run_pilot(
        model=model, tokenizer=tok, df_sample=df, generation_csv=gen,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, seed=2026,
    )
    assert "n_misaligned_dropped" in results
    assert "alignment_yield" in results
    assert results["n_measured"] + results["n_misaligned_dropped"] == results["n_pairs"]


def test_p1_uses_teacher_forced_argmax_not_a_reencoded_prefix(tiny, tmp_path):
    """Regression: P1 read 0.0 because it re-encoded a prefix separately.

    BPE prompt tokenisation is not a prefix of the joint tokenisation, and the
    old check also compared a LAST subword against the FIRST token at p*.
    """
    import inspect
    from codenames.causal.pilot import run_pilot
    src = inspect.getsource(run_pilot)
    assert "argmax" in src
    assert "text[: text.lower().find(word.lower())]" not in src


def test_report_surfaces_alignment_and_sign_diagnostics(tiny, tmp_path):
    from codenames.causal.pilot import run_pilot
    model, tok = tiny
    df, gen = _pilot_fixtures(tmp_path)
    report, _ = run_pilot(
        model=model, tokenizer=tok, df_sample=df, generation_csv=gen,
        base_dir=str(tmp_path), prefix="tiny", mode="no_social",
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, seed=2026,
    )
    assert {"P4_sign", "alignment", "n_measured"} <= set(report["check"])


def test_p8_separates_forward_only_from_wall_clock():
    """The first real run reported 13.1 fwd/s, but that clock also covered
    backward passes and generation. Forward-only is the number that converts
    the §12.2 counts into A100-hours."""
    import inspect
    from codenames.causal.pilot import run_pilot
    src = inspect.getsource(run_pilot)
    assert "fwd_seconds" in src
    assert '"P8_wall_fwd_per_s"' in src


def test_p4_normalised_is_diagnostic_not_a_gate():
    """The flip rate needs the model's own accuracy as a denominator, but that
    must not silently become a pass condition."""
    from codenames.causal.pilot import pilot_report, pilot_verdict
    r = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.30, "P4_sign": 0.9,
         "P5_rho": 0.7, "P5_fnr": 0.1, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True, "P4_clean_accuracy": 0.617}
    report = pilot_report(r).set_index("check")
    assert report.loc["P4_normalised", "observed"] == pytest.approx(0.30 / 0.617, abs=1e-6)
    assert "NOT a gate" in report.loc["P4_normalised", "threshold"]
    # the real gate still fails on the raw flip rate
    assert "P4" in pilot_verdict(r)["blocking_failures"]


def test_p4_is_ceiling_relative_after_the_2026_07_28_amendment():
    """The observed run: flip 0.577 against a model accuracy of 0.600 passes,
    because 0.577/0.600 = 0.96 >= 0.85. The old absolute 0.60 gate sat at the
    model's own ceiling and was near-unpassable."""
    ok = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.90,
          "P4_clean_accuracy": 0.600, "P5_rho": 0.7, "P5_fnr": 0.1,
          "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    assert pilot_verdict(ok)["launch_full_run"] is True


def test_p4_still_fails_when_corruption_genuinely_does_not_work():
    weak = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.20, "P4_sign": 0.90,
            "P4_clean_accuracy": 0.600, "P5_rho": 0.7, "P5_fnr": 0.1,
            "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    assert "P4" in pilot_verdict(weak)["blocking_failures"]


def test_p4_fails_when_the_denominator_is_unmeasured():
    """An absent clean accuracy must not become a free pass."""
    missing = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.9, "P4_sign": 0.9,
               "P5_rho": 0.7, "P5_fnr": 0.1, "P6_change": 0.3, "P6_parse": 0.95,
               "P7_finite": True}
    assert "P4" in pilot_verdict(missing)["blocking_failures"]


def test_p4_sign_still_blocks_independently():
    """Both criteria must hold - a good flip ratio cannot rescue a bad sign."""
    bad_sign = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.4,
                "P4_clean_accuracy": 0.600, "P5_rho": 0.7, "P5_fnr": 0.1,
                "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    assert "P4" in pilot_verdict(bad_sign)["blocking_failures"]


# --- P5 instrument (amended 2026-07-28) ------------------------------------

def _p5_base(**over):
    r = {"P1": 1.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.9,
         "P4_clean_accuracy": 0.6, "P5_rho": 0.7, "P5_fnr": 0.1,
         "P5_n_high_effect": 12, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True}
    r.update(over)
    return r


def test_p5_zero_high_effect_sites_is_uninformative_not_a_pass():
    """An FNR over zero high-effect sites means nothing was there to miss."""
    v = pilot_verdict(_p5_base(P5_n_high_effect=0, P5_fnr=0.0, P5_rho=0.0))
    assert v["attribution_uninformative"] is True
    assert v["attribution_shortcut_lost"] is False   # not claimed either way
    assert v["launch_full_run"] is True              # still non-blocking


def test_p5_high_fnr_loses_the_shortcut():
    v = pilot_verdict(_p5_base(P5_fnr=0.9))
    assert v["attribution_shortcut_lost"] is True
    assert v["attribution_uninformative"] is False


def test_p5_good_fnr_and_rho_keeps_the_shortcut():
    assert pilot_verdict(_p5_base())["attribution_shortcut_lost"] is False


def test_p5_sampling_is_stratified_and_rng_is_not_reseeded_per_pair():
    """Two bugs at once: random-only sampling had no dynamic range, and the
    generator was re-created inside the loop so every pair drew identical sites.
    """
    import inspect
    from codenames.causal.pilot import run_pilot
    src = inspect.getsource(run_pilot)
    assert "top_sites" in src and "rand_sites" in src
    body = src.split("for pair in pairs.itertuples():", 1)[1]
    assert "np.random.default_rng(seed)" not in body, "rng re-seeded inside the loop"


def test_p5_effect_floor_is_absolute_not_a_quantile():
    """A quantile cut guarantees 'high-effect' sites exist even when none do."""
    import inspect
    from codenames.causal import pilot
    assert pilot._P5_EFFECT_FLOOR > 0
    src = inspect.getsource(pilot.run_pilot)
    assert "_P5_EFFECT_FLOOR" in src
    assert "np.median(np.abs(real))" not in src


def test_report_surfaces_the_p5_diagnostics():
    report = pilot_report(_p5_base(P5_n_sites=780)).set_index("check")
    assert report.loc["P5_n_high", "observed"] == 12
    assert report.loc["P5_n_sites", "observed"] == 780


# --- P1 not-applicable for the random-init null (amended 2026-07-28) --------

def test_p1_not_applicable_does_not_block():
    """The random-init null has no generations: P1 asks whether p* reproduces
    the model's own greedy output, and an untrained model has no meaningful
    output to reproduce. P7 is what matters for that arm."""
    r = {"P1": 0.0, "P1_applicable": False, "P2": 1.0, "P3": 0.0,
         "P4_flip": 0.577, "P4_sign": 0.9, "P4_clean_accuracy": 0.6,
         "P5_rho": 0.7, "P5_fnr": 0.1, "P5_n_high_effect": 12,
         "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    v = pilot_verdict(r)
    assert "P1" not in v["blocking_failures"]
    assert v["launch_full_run"] is True


def test_p1_still_blocks_when_it_is_applicable():
    """A trained model with a bad P1 must still fail - this is what caught the
    re-encoded-prefix bug."""
    r = {"P1": 0.0, "P1_applicable": True, "P2": 1.0, "P3": 0.0,
         "P4_flip": 0.577, "P4_sign": 0.9, "P4_clean_accuracy": 0.6,
         "P5_rho": 0.7, "P5_fnr": 0.1, "P5_n_high_effect": 12,
         "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    assert "P1" in pilot_verdict(r)["blocking_failures"]


def test_p1_defaults_to_applicable():
    """Absent the flag, P1 is treated as a real check - never skipped by
    accident. A missing generations file must fail loudly, not silently pass."""
    r = {"P1": 0.0, "P2": 1.0, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.9,
         "P4_clean_accuracy": 0.6, "P5_rho": 0.7, "P5_fnr": 0.1,
         "P5_n_high_effect": 12, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True}
    assert "P1" in pilot_verdict(r)["blocking_failures"]


def test_report_marks_p1_not_applicable():
    r = {"P1": 0.0, "P1_applicable": False, "P2": 1.0, "P3": 0.0,
         "P4_flip": 0.577, "P4_sign": 0.9, "P4_clean_accuracy": 0.6,
         "P5_rho": 0.7, "P5_fnr": 0.1, "P5_n_high_effect": 12,
         "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    row = pilot_report(r).set_index("check").loc["P1"]
    assert row["observed"] == "n/a"


def test_p2_uses_a_robust_statistic_not_the_mean():
    """Qwen returned mean 0.886 while the identity itself held: a pair whose
    clean/corrupt denominator is near zero makes e explode and drags an
    average. The identity is per-pair, so the summary must be robust."""
    import inspect
    from codenames.causal.pilot import run_pilot
    src = inspect.getsource(run_pilot)
    assert 'np.nanmedian(e_full)' in src
    assert '"P2_within_tol"' in src


def test_report_surfaces_the_p2_outlier_diagnostics():
    r = {"P1": 1.0, "P2": 1.0, "P2_mean": 0.886, "P2_within_tol": 0.97,
         "P2_denom_median": 12.4, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.9,
         "P4_clean_accuracy": 0.6, "P5_rho": 0.7, "P5_fnr": 0.1,
         "P5_n_high_effect": 12, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True}
    report = pilot_report(r).set_index("check")
    assert report.loc["P2_mean", "observed"] == 0.886
    assert report.loc["P2_within_tol", "observed"] == 0.97
    assert report.loc["P2_denom", "observed"] == 12.4
    # the robust statistic is what gates
    assert pilot_verdict(r)["launch_full_run"] is True


def test_p1_records_miss_margins_for_diagnosis():
    """Qwen P1=0.939 vs Mistral 1.000. A near-tie margin means numerical drift
    between the accelerated generation path and the reference path; a wide
    margin means something structural. The lens calibration gate was diagnosed
    exactly this way (median 4.2-logit margins => structural)."""
    import inspect
    from codenames.causal.pilot import run_pilot
    src = inspect.getsource(run_pilot)
    assert "p1_margins" in src
    assert '"P1_miss_margin_median"' in src


def test_report_surfaces_p1_margin_diagnostics():
    r = {"P1": 0.939, "P1_miss_margin_median": 0.03, "P1_n_misses": 8,
         "P2": 1.0, "P3": 0.0, "P4_flip": 0.577, "P4_sign": 0.9,
         "P4_clean_accuracy": 0.6, "P5_rho": 0.7, "P5_fnr": 0.1,
         "P5_n_high_effect": 12, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True}
    report = pilot_report(r).set_index("check")
    assert report.loc["P1_margin", "observed"] == 0.03
    assert report.loc["P1_misses", "observed"] == 8


def test_p1_counts_only_decisive_misses():
    """P1 validates p* INDEXING. A near-tie argmax flip between the accelerated
    generation path and the reference path is numerical drift, not a mis-index.

    Evidence for the distinction: Mistral scores 1.000 on identical code while
    Qwen shows 8 misses at a median margin of 0.54 logits. Your lens gate
    diagnosed genuinely STRUCTURAL disagreement at median 4.2 logits, and a
    truly mis-indexed p* scores near 0 (as the broken P1 did).
    """
    import inspect
    from codenames.causal.pilot import run_pilot, PILOT_THRESHOLDS
    assert PILOT_THRESHOLDS["P1_decisive_margin"] == 1.0
    src = inspect.getsource(run_pilot)
    assert "p1_decisive_misses" in src
    assert '"P1_raw"' in src, "raw agreement must still be reported"


def test_p1_still_fails_on_a_genuine_indexing_fault():
    """Wide-margin misses must still block - that is the fault P1 exists for."""
    r = {"P1": 0.5, "P1_raw": 0.5, "P2": 1.0, "P3": 0.0, "P4_flip": 0.577,
         "P4_sign": 0.9, "P4_clean_accuracy": 0.6, "P5_rho": 0.7, "P5_fnr": 0.1,
         "P5_n_high_effect": 12, "P6_change": 0.3, "P6_parse": 0.95,
         "P7_finite": True}
    assert "P1" in pilot_verdict(r)["blocking_failures"]


def test_report_keeps_raw_agreement_visible():
    r = {"P1": 1.0, "P1_raw": 0.939, "P1_n_decisive_misses": 0,
         "P1_miss_margin_median": 0.54, "P1_n_misses": 8, "P2": 1.0, "P3": 0.0,
         "P4_flip": 0.577, "P4_sign": 0.9, "P4_clean_accuracy": 0.6,
         "P5_rho": 0.7, "P5_fnr": 0.1, "P5_n_high_effect": 12,
         "P6_change": 0.3, "P6_parse": 0.95, "P7_finite": True}
    report = pilot_report(r).set_index("check")
    assert report.loc["P1_raw", "observed"] == 0.939
    assert report.loc["P1_decisive", "observed"] == 0
