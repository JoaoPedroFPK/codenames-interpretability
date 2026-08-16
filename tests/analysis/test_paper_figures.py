"""Smoke tests for the ICLR paper figure builders (synthetic frames)."""
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


def _conc(models=("mistral", "qwen", "random_qwen"), n_layers=9):
    rows = []
    for m in models:
        for l in range(n_layers):
            base = 0.07 if m.startswith("random_") else 0.2 + 0.3 * np.sin(np.pi * l / (n_layers - 1)) ** 2
            rows.append({"model": m, "layer": l, "layer_frac": l / (n_layers - 1),
                         "top1_accuracy": base, "concordance": base * 0.8,
                         "pooling": "mean", "condition": "no_social"})
    return pd.DataFrame(rows)


def _margins(models=("mistral", "qwen", "random_qwen", "bert"), n_layers=9):
    rows = []
    for m in models:
        for l in range(n_layers):
            rows.append({"model": m, "layer": l, "layer_frac": l / (n_layers - 1),
                         "pooling_method": "mean", "condition": "no_social",
                         "mean_margin": 0.05, "adjusted_margin": 0.5,
                         "mean_anisotropy": 0.3 + 0.05 * l})
    return pd.DataFrame(rows)


def _confound(models=("mistral", "qwen", "random_qwen", "bert"), n_layers=9):
    return pd.DataFrame([{"model": m, "layer": l, "layer_frac": l / (n_layers - 1),
                          "mean_rho": -0.3} for m in models for l in range(n_layers)])


def _summary():
    return pd.DataFrame([{"model": "mistral", "condition": "no_social", "generation_accuracy": 0.617},
                         {"model": "qwen", "condition": "no_social", "generation_accuracy": 0.595}])


def test_find_humps_returns_two_maxima_and_trough():
    from codenames.analysis.paper_figures import find_humps
    y = np.array([0.2, 0.5, 0.3, 0.1, 0.2, 0.45, 0.4, 0.15])
    h = find_humps(y)
    assert h["hump1"] == 1 and h["hump2"] == 5 and h["trough"] == 3


def test_fig_geometry_runs(tmp_path):
    from codenames.analysis.paper_figures import fig_geometry
    out = fig_geometry(_conc(), _margins(), _confound(), _summary(),
                       out_path=tmp_path / "F2_geometry.pdf")
    assert out.exists() and out.stat().st_size > 0


def _lens_curves(models=("mistral", "qwen", "random_qwen"), n_layers=9):
    rows = []
    for m in models:
        for lens in ("raw", "tuned"):
            if m.startswith("random_") and lens == "tuned":
                continue
            for l in range(n_layers):
                y = 0.07 if (m.startswith("random_") or l < 5) else 0.7
                rows.append({"lens": lens, "layer": l, "top1": y, "mrr": y, "n": 100,
                             "ci_lo": y - 0.01, "ci_hi": y + 0.01, "model": m})
    return pd.DataFrame(rows)


def _shuffle(models=("mistral", "qwen"), n_layers=9):
    return pd.DataFrame([{"lens": lens, "layer": l, "top1": 0.072, "top1_p97_5": 0.078, "model": m}
                         for m in models for lens in ("raw", "tuned") for l in range(n_layers)])


def test_fig_lens_runs(tmp_path):
    from codenames.analysis.paper_figures import fig_lens
    out = fig_lens(_lens_curves(), _shuffle(), _conc(), out_path=tmp_path / "F3_lens.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_emergence_depth_is_first_layer_at_90pct_of_final():
    from codenames.analysis.paper_figures import emergence_depth
    y = np.array([0.1, 0.1, 0.2, 0.5, 0.66, 0.7, 0.7])
    assert emergence_depth(y) == 4
