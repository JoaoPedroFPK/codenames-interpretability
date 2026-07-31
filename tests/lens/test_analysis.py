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
    """§3 amended gate: computed on the word-first subsample only; the
    full-sample agreement is kept as a descriptive statistic."""
    scores = _toy_scores()
    final = scores[scores["layer"] == 2]
    top = final[final["rank"] == 1][["row_id", "word"]].reset_index(drop=True)
    words, texts = [], []
    for i, w in enumerate(top["word"].values):
        if i % 2 == 0:                       # word-first, agrees with lens
            words.append(w)
            texts.append(f'"{w}" is my clue.')
        else:                                # scaffolded, disagrees
            other = "a" if w != "a" else "b"
            words.append(other)
            texts.append(f'The word that best matches is "{other}".')
    gen = pd.DataFrame({
        "row_id": top["row_id"].values,
        "generated_text": texts,
        "generated_word": words,
        "generated_in_candidates": True,
        "generated_correct": [w == "a" for w in words],
    })
    gcsv = str(tmp_path / "gen.csv")
    gen.to_csv(gcsv, index=False)
    r = calibration_check(scores, gcsv)
    assert r["n_parseable"] == 20
    assert r["n_word_first"] == 10
    assert r["word_first_agreement"] == 1.0
    assert r["agreement"] == 0.5
    assert r["passes_gate"] is True          # gated on word-first only


def test_calibration_check_word_first_failure(tmp_path):
    """A word-first disagreement is a real pipeline defect -> gate fails."""
    scores = _toy_scores()
    final = scores[scores["layer"] == 2]
    top = final[final["rank"] == 1][["row_id", "word"]].reset_index(drop=True)
    wrong = ["a" if w != "a" else "b" for w in top["word"].values]
    gen = pd.DataFrame({
        "row_id": top["row_id"].values,
        "generated_text": [f"{w} is my clue." for w in wrong],
        "generated_word": wrong,
        "generated_in_candidates": True,
        "generated_correct": [w == "a" for w in wrong],
    })
    gcsv = str(tmp_path / "gen.csv")
    gen.to_csv(gcsv, index=False)
    r = calibration_check(scores, gcsv)
    assert r["n_word_first"] == 20
    assert r["word_first_agreement"] == 0.0
    assert r["passes_gate"] is False


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


def test_overlay_has_dissociation_panel_with_geometry(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from codenames.lens.figures import build_lens_overlay_figure

    curves = pd.concat([
        _curves([0.1, 0.3, 0.5, 0.55, 0.5, 0.45, 0.5, 0.6], lens="raw"),
        _curves([0.2, 0.4, 0.55, 0.58, 0.56, 0.55, 0.58, 0.6], lens="tuned"),
    ])
    gcsv = tmp_path / "g.csv"
    pd.DataFrame({
        "model": "mistral", "condition": "no_social", "pooling": "mean",
        "layer": range(8), "top1_accuracy": np.linspace(0.25, 0.15, 8),
    }).to_csv(gcsv, index=False)
    fig = build_lens_overlay_figure(curves, "mistral", str(gcsv), 0.617)
    assert len(fig.axes) == 2          # overlay + divergence panel
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_overlay_falls_back_to_single_panel_without_geometry(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from codenames.lens.figures import build_lens_overlay_figure

    curves = _curves([0.1, 0.3, 0.5, 0.55, 0.5, 0.45, 0.5, 0.6], lens="raw")
    gcsv = tmp_path / "g.csv"
    pd.DataFrame({
        "model": "qwen", "condition": "no_social", "pooling": "mean",
        "layer": range(8), "top1_accuracy": np.linspace(0.25, 0.15, 8),
    }).to_csv(gcsv, index=False)   # no rows for 'mistral'
    fig = build_lens_overlay_figure(curves, "mistral", str(gcsv), 0.617)
    assert len(fig.axes) == 1          # graceful single-panel fallback
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dissociation_regions():
    from codenames.lens.figures import _dissociation_regions
    # both lenses below geometry for 4 layers, straddle, then both above 3
    diff_raw = np.array([-0.3, -0.3, -0.2, -0.1, 0.05, 0.2, 0.25, 0.3])
    diff_tuned = np.array([-0.4, -0.35, -0.25, -0.15, -0.05, 0.1, 0.2, 0.3])
    below, above = _dissociation_regions(diff_raw, diff_tuned, min_run=3)
    assert below == [(0, 3)]
    assert above == [(5, 7)]
    # runs shorter than min_run are dropped
    below, above = _dissociation_regions(diff_raw, diff_tuned, min_run=5)
    assert below == [] and above == []


def test_run_analysis_without_scores_is_graceful(tmp_path):
    from codenames.lens.analysis import run_analysis
    out = run_analysis(output_root=str(tmp_path), models=("mistral",),
                       random_model=None, out_dir=str(tmp_path / "an"))
    assert out == {}


def _tiny_scores(rows=40):
    import numpy as np
    rng = np.random.default_rng(2026)
    recs = []
    for row_id in range(rows // 8):
        for layer in (0, 1):
            words = [("sea", "target"), ("moon", "black"),
                     ("ship", "tan"), ("castle", "tan")]
            scores = rng.standard_normal(4)
            order = (-scores).argsort().argsort() + 1
            for (w, wt), s, r in zip(words, scores, order):
                recs.append({"row_id": row_id, "layer": layer, "word": w,
                             "word_type": wt, "lens": "raw",
                             "score": float(s), "rank": int(r)})
    return pd.DataFrame(recs)


def test_run_analysis_answer_channel_reads_answer_scores_and_null_fallback(tmp_path):
    """channel='answer' reads {m}_lens_scores_answer_{mode}.parquet, writes
    answer-labelled outputs, and falls back to the null's generating-channel
    scores (a model with no behaviour has no answer channel)."""
    import os

    from codenames.lens.analysis import run_analysis

    root = tmp_path
    for m, label in [("m", "answer_no_social"), ("r", "no_social")]:
        d = root / f"{m}_outputs"
        d.mkdir()
        _tiny_scores().to_parquet(d / f"{m}_lens_scores_{label}.parquet",
                                  index=False)
    out = root / "lens_analysis"
    summary = run_analysis(output_root=str(root), models=("m",),
                           random_model="r", mode="no_social",
                           channel="answer", out_dir=str(out), n_boot=20)
    assert "m" in summary
    curves = pd.read_csv(out / "lens_curves_answer_no_social.csv")
    assert set(curves["model"]) == {"m", "r"}
