import numpy as np
import pandas as pd
import pytest

from codenames.causal.pilot import PILOT_THRESHOLDS, pilot_report, pilot_verdict


def _results(**overrides):
    base = {
        "P1": 1.00, "P2": 1.00, "P3": 0.00, "P4_flip": 0.75, "P4_sign": 0.90,
        "P5_rho": 0.70, "P5_fnr": 0.10, "P6_change": 0.30, "P6_parse": 0.95,
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
        ({"P4_flip": 0.1}, "P4"),
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
    assert PILOT_THRESHOLDS["P4_flip"] == 0.60
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
    assert set(report["check"]) == {"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"}
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
