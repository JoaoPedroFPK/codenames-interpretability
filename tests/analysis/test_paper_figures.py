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


def test_fig_geometry_single_panel_and_extra_panel_a_models(tmp_path):
    """``panels="a"`` renders one axis; ``panel_a_models`` adds a base decoder
    to panel (a) when its curve is present (appendix version)."""
    from codenames.analysis.paper_figures import fig_geometry
    conc = _conc(models=("mistral", "qwen", "random_qwen", "mistral_base"))
    out = fig_geometry(conc, _margins(), _confound(), _summary(),
                       out_path=tmp_path / "F2a.pdf", panels="a")
    assert out.exists() and out.stat().st_size > 0
    out2 = fig_geometry(conc, _margins(), _confound(), _summary(),
                        out_path=tmp_path / "A_geo.pdf", panels="abcd",
                        panel_a_models=("mistral", "qwen", "mistral_base"))
    assert out2.exists() and out2.stat().st_size > 0


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


def _effects(n_layers=9, n_turns=20):
    rows = []
    roles = ["hint"] * 4 + ["cand_donor"] * 2 + ["generation"] * 3
    for l in range(n_layers):
        for t in range(n_turns):
            for w in (1, 3, 5):
                rows.append({"layer": l, "role": roles[l], "n_positions": 2, "width": w,
                             "row_id": t, "effect": 0.5 + 0.01 * t})
    return pd.DataFrame(rows)


def test_confirmatory_curve_has_ci_and_drops_nonfinite():
    from codenames.analysis.paper_figures import confirmatory_curve
    eff = _effects()
    eff.loc[0, "effect"] = np.nan
    cur = confirmatory_curve(eff, width=1, n_boot=50, seed=1)
    assert set(cur.columns) >= {"layer", "role", "e", "lo", "hi", "n"}
    assert len(cur) == 9 and (cur["lo"] <= cur["e"]).all() and (cur["e"] <= cur["hi"]).all()


def test_fig_causal_runs(tmp_path):
    from codenames.analysis.paper_figures import fig_causal
    scan = np.random.default_rng(0).random((9, 10))
    out = fig_causal({"mistral": (scan, _effects())}, _conc(), {"mistral": 6},
                     out_path=tmp_path / "F4.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_fig_triangulation_runs(tmp_path):
    from codenames.analysis.paper_figures import fig_triangulation
    out = fig_triangulation(_conc(), _lens_curves(), {"mistral": _effects()},
                            out_path=tmp_path / "F5.pdf")
    assert out.exists() and out.stat().st_size > 0


def test_hump_ranges_uses_60pct_rule():
    from codenames.analysis.paper_figures import hump_ranges
    y = np.array([0.2, 0.3, 0.45, 0.5, 0.45, 0.3, 0.1, 0.3, 0.45, 0.5, 0.45, 0.3, 0.2])
    r = hump_ranges(y)
    # trough 0.1 at index 6; threshold 0.1+0.6*0.4=0.34 -> indices with y>=0.34
    assert r["hump1"] == (2, 4) and r["hump2"] == (8, 10)


def _grid(n_layers=9, n_turns=20):
    """Role x layer grid + matched random-site nulls (causal-patch --grid)."""
    from codenames.causal.grid import GRID_ROLES
    rows, nulls = [], []
    for l in range(n_layers):
        for t in range(n_turns):
            for role in GRID_ROLES:
                k = {"hint": 2, "cand_target": 1, "cand_donor": 1, "final": 1,
                     "generation": 5, "p_read": 1, "scaffold": 4}[role]
                e = np.nan if (role == "scaffold" and t % 5 == 0) else 0.3 + 0.02 * l + 0.01 * t
                rows.append({"layer": l, "role": role, "n_positions": 0 if np.isnan(e) else k,
                             "width": 1, "row_id": t, "effect": e})
            for k in (1, 2, 4, 5):
                nulls.append({"layer": l, "role": "random_site", "n_positions": k,
                              "matched_n": k, "matched_roles": "x", "width": 1,
                              "row_id": t, "effect": 0.02 + 0.001 * t})
    return pd.DataFrame(rows), pd.DataFrame(nulls)


def test_grid_curve_has_every_role_and_a_null_band():
    from codenames.analysis.paper_figures import grid_curve
    from codenames.causal.grid import GRID_ROLES
    grid, nulls = _grid()
    cur, null = grid_curve(grid, nulls, width=1, n_boot=50, seed=1)
    assert set(cur["role"]) == set(GRID_ROLES)
    assert len(cur) == 9 * len(GRID_ROLES)
    assert set(null.columns) >= {"layer", "e", "lo", "hi"} and len(null) == 9
    sc = cur[cur["role"] == "scaffold"]
    assert (sc["n"] == 16).all()   # NaN cells dropped, count reported honestly


def test_fig_causal_with_grid_runs(tmp_path):
    from codenames.analysis.paper_figures import fig_causal
    scan = np.random.default_rng(0).random((9, 10))
    grid, nulls = _grid()
    out = fig_causal({"mistral": (scan, _effects())}, _conc(), {"mistral": 6},
                     out_path=tmp_path / "F4g.pdf", grids={"mistral": (grid, nulls)})
    assert out.exists() and out.stat().st_size > 0


def test_redundant_roles_for_word_first_models():
    """When the generating position is p_read (word-first answers), the
    'final' curve duplicates 'p_read' and 'generation' lies after the readout;
    both are dropped from the plotted curve."""
    from codenames.analysis.paper_figures import redundant_roles
    import pandas as pd
    rows = []
    for layer in range(5):
        e = layer / 4
        rows += [{"layer": layer, "role": "p_read", "e": e},
                 {"layer": layer, "role": "final", "e": e},
                 {"layer": layer, "role": "generation", "e": 0.0},
                 {"layer": layer, "role": "hint", "e": 1 - e}]
    cur = pd.DataFrame(rows)
    assert redundant_roles(cur) == {"final", "generation"}
    cur.loc[cur.role == "final", "e"] += 0.1
    assert redundant_roles(cur) == set()


def test_cli_triangulation_forwards_models(tmp_path, monkeypatch):
    """``--models`` selects which decoders get a panel, not only which effects
    files are read: a decoder run through geometry and lens but not patching
    (Llama) has to be renderable on its own.
    """
    from pathlib import Path
    from codenames.analysis import paper_figures as pf
    (tmp_path / "analysis").mkdir()
    (tmp_path / "lens").mkdir()
    _conc(models=("mistral", "qwen", "llama", "random_qwen")).to_csv(
        tmp_path / "analysis" / "analysis_concordance_by_layer.csv", index=False)
    _lens_curves(models=("mistral", "qwen", "llama", "random_qwen")).to_csv(
        tmp_path / "lens" / "lens_curves_answer_no_social.csv", index=False)
    seen = {}

    def fake(conc, lens, effects, *, out_path, models=("mistral", "qwen"), **kw):
        seen["models"] = tuple(models)
        Path(out_path).write_text("x")
        return Path(out_path)

    monkeypatch.setattr(pf, "fig_triangulation", fake)
    pf.main(["--fig", "triangulation", "--models", "llama",
             "--analysis-dir", str(tmp_path / "analysis"),
             "--lens-dir", str(tmp_path / "lens"),
             "--out", str(tmp_path / "A_llama.pdf")])
    assert seen["models"] == ("llama",)
