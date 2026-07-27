import numpy as np
import pandas as pd

from codenames.lens.analysis import (
    calibration_check,
    classify_trajectory,
    emergence_depth,
    layer_curves,
    shuffle_control,
    spearman_vs_geometric,
    trained_vs_random,
)


def _curves(top1_by_layer, lens="tuned"):
    return pd.DataFrame({
        "lens": lens, "layer": range(len(top1_by_layer)),
        "top1": top1_by_layer, "mrr": top1_by_layer,
        "n": 1000, "ci_lo": [t - 0.02 for t in top1_by_layer],
        "ci_hi": [t + 0.02 for t in top1_by_layer],
    })


def test_classify_plateau():
    c = _curves([0.1, 0.3, 0.55, 0.58, 0.57, 0.56, 0.58, 0.60])
    r = classify_trajectory(c, "tuned", n_states=8)
    assert r["label"] == "plateau"


def test_classify_dip_then_recover():
    c = _curves([0.1, 0.3, 0.55, 0.58, 0.30, 0.28, 0.45, 0.60])
    r = classify_trajectory(c, "tuned", n_states=8)
    assert r["label"] == "dip_then_recover"
    assert r["dip_value"] <= 0.30


def test_classify_late_rise():
    c = _curves([0.05, 0.06, 0.08, 0.10, 0.12, 0.20, 0.45, 0.60])
    r = classify_trajectory(c, "tuned", n_states=8)
    assert r["label"] == "late_rise"


def test_emergence_depth():
    c = _curves([0.1, 0.2, 0.54, 0.58, 0.57, 0.56, 0.58, 0.60])
    assert emergence_depth(c, "tuned") == 2   # 0.54 >= 0.9 * 0.60


def _toy_scores():
    recs = []
    rng = np.random.default_rng(0)
    for row_id in range(20):
        for layer in range(3):
            scores = rng.standard_normal(4)
            order = np.argsort(-scores)
            words = ["a", "b", "c", "d"]
            for j, w in enumerate(words):
                recs.append({
                    "row_id": row_id, "layer": layer, "word": w,
                    "word_type": "target" if w == "a" else "tan",
                    "lens": "raw", "score": float(scores[j]),
                    "rank": int(np.where(order == j)[0][0]) + 1,
                })
    return pd.DataFrame(recs)


def test_layer_curves_and_shuffle_chance_level():
    scores = _toy_scores()
    curves = layer_curves(scores, n_boot=200, seed=1)
    assert set(curves.columns) >= {"lens", "layer", "top1", "mrr", "n",
                                   "ci_lo", "ci_hi"}
    assert len(curves) == 3
    assert (curves["n"] == 20).all()
    # one target among four random candidates -> both near 0.25
    ctrl = shuffle_control(scores, n_perm=50, seed=1)
    assert abs(ctrl["top1"].mean() - 0.25) < 0.1
    assert abs(curves["top1"].mean() - 0.25) < 0.15


def test_calibration_check(tmp_path):
    scores = _toy_scores()
    final = scores[scores["layer"] == 2]
    top = final[final["rank"] == 1][["row_id", "word"]]
    gen = pd.DataFrame({
        "row_id": top["row_id"].values,
        "generated_word": top["word"].values,       # perfect agreement
        "generated_in_candidates": True,
        "generated_correct": (top["word"] == "a").values,
    })
    gcsv = str(tmp_path / "gen.csv")
    gen.to_csv(gcsv, index=False)
    r = calibration_check(scores, gcsv)
    assert r["agreement"] == 1.0
    assert r["n_parseable"] == 20


def test_spearman_vs_geometric(tmp_path):
    curves = _curves([0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4, 0.35], lens="raw")
    g = pd.DataFrame({
        "model": "mistral", "condition": "no_social", "pooling": "mean",
        "layer": range(8),
        "top1_accuracy": [0.1, 0.2, 0.3, 0.4, 0.5, 0.45, 0.4, 0.35],
    })
    gcsv = str(tmp_path / "g.csv")
    g.to_csv(gcsv, index=False)
    r = spearman_vs_geometric(curves, "raw", gcsv, "mistral", n_boot=100)
    assert r["spearman"] > 0.99   # identical ordering


def test_trained_vs_random():
    trained = _curves([0.2, 0.4, 0.6], lens="raw")
    random = _curves([0.25, 0.25, 0.25], lens="raw")
    m = trained_vs_random(trained, random)
    assert list(m["exceeds_random_ci"]) == [False, True, True]
    np.testing.assert_allclose(m["delta"], [-0.05, 0.15, 0.35])


def test_overlay_figure_writes_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from codenames.lens.figures import lens_overlay_figure

    curves = pd.concat([
        _curves([0.1, 0.3, 0.5, 0.55, 0.5, 0.45, 0.5, 0.6], lens="raw"),
        _curves([0.2, 0.4, 0.55, 0.58, 0.56, 0.55, 0.58, 0.6], lens="tuned"),
    ])
    gcsv = tmp_path / "g.csv"
    pd.DataFrame({
        "model": "mistral", "condition": "no_social", "pooling": "mean",
        "layer": range(8), "top1_accuracy": np.linspace(0.25, 0.15, 8),
    }).to_csv(gcsv, index=False)
    random_curves = _curves([0.24, 0.25, 0.26, 0.25, 0.24, 0.25, 0.26, 0.25],
                            lens="tuned")
    out = lens_overlay_figure(curves, "mistral", str(gcsv), 0.617,
                              str(tmp_path / "overlay_mistral.png"),
                              random_curves=random_curves)
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_run_analysis_without_scores_is_graceful(tmp_path):
    from codenames.lens.analysis import run_analysis
    out = run_analysis(output_root=str(tmp_path), models=("mistral",),
                       random_model=None, out_dir=str(tmp_path / "an"))
    assert out == {}
