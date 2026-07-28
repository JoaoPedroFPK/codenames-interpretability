import os

import numpy as np
import pandas as pd
import pytest

from codenames.causal.figures import dose_response, patching_heatmap, triangulation


def test_heatmap_is_written(tmp_path):
    grid = np.random.default_rng(0).normal(size=(8, 20))
    written = patching_heatmap(
        grid, str(tmp_path / "h"), claimed_sites=[(3, 5)], title="tiny",
    )
    assert written and all(os.path.exists(p) for p in written)
    assert any(os.path.getsize(p) > 1000 for p in written)


def test_heatmap_accepts_an_empty_claimed_set(tmp_path):
    grid = np.zeros((4, 4))
    written = patching_heatmap(grid, str(tmp_path / "empty"), claimed_sites=[])
    assert written


def test_heatmap_rejects_out_of_range_sites(tmp_path):
    """A mis-specified locus must fail loudly, not be silently clipped."""
    grid = np.zeros((4, 4))
    with pytest.raises(ValueError, match="outside"):
        patching_heatmap(grid, str(tmp_path / "bad"), claimed_sites=[(9, 9)])


def test_dose_response_is_written(tmp_path):
    curves = pd.DataFrame({
        "alpha": [-2, -1, 0, 1, 2] * 2,
        "effect": [-0.2, -0.1, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.01, 0.0],
        "arm": ["primary"] * 5 + ["random_direction"] * 5,
        "parse_rate": [0.95] * 10,
    })
    written = dose_response(curves, str(tmp_path / "d"))
    assert written and all(os.path.exists(p) for p in written)


def test_dose_response_requires_the_expected_columns(tmp_path):
    with pytest.raises(ValueError, match="columns"):
        dose_response(pd.DataFrame({"alpha": [0]}), str(tmp_path / "x"))


def test_triangulation_marks_both_humps(tmp_path):
    layers = np.arange(12)
    written = triangulation(
        margin=np.sin(layers / 3), lens=np.cos(layers / 3), causal=np.zeros(12),
        out_path=str(tmp_path / "t"), humps=(2, 9), model="mistral",
    )
    assert written and all(os.path.exists(p) for p in written)


def test_triangulation_rejects_mismatched_curve_lengths(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        triangulation(
            margin=np.zeros(5), lens=np.zeros(4), causal=np.zeros(5),
            out_path=str(tmp_path / "t2"), humps=(1,), model="mistral",
        )
