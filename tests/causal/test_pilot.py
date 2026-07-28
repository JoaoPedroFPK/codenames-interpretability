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
