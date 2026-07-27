"""Pre-registered lens analysis (docs/specs/lens_spec.md §3, §7).

Decision-rule constants are the §3 pre-registration; do not tune them
against the data. The calibration check gates everything: if final-layer
raw-lens agreement with generation fails, downstream claims are BLOCKED.
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PLATEAU_BAND = 0.10       # §3: within 10 points of interior max to the readout
DIP_DEPTH = 0.15          # §3: >= 15-point drop below interior max
INTERIOR_FRAC = 0.75      # §3: interior = first three quarters of the stack
EMERGENCE_FRAC = 0.90     # §3: emergence = 90% of final accuracy
LATE_RISE_FRAC = 0.50     # interior max under half of final -> late rise
CALIBRATION_GATE = 0.95   # §3: minimum final-layer raw-lens vs generation


def _per_turn_hits_mrr(sl: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized per-turn top-1 hit and MRR for one (lens, layer) slice."""
    universe = pd.Index(sl["row_id"].unique())
    top = sl[sl["rank"] == 1].drop_duplicates("row_id").set_index("row_id")
    hits = (top["word_type"] == "target").reindex(universe, fill_value=False)
    tgt = sl[(sl["word_type"] == "target") & (sl["rank"] > 0)]
    min_rank = tgt.groupby("row_id")["rank"].min().reindex(universe)
    mrr = (1.0 / min_rank).to_numpy(dtype=float)
    return hits.to_numpy(dtype=float), mrr


def _boot_ci(hits: np.ndarray, n_boot: int, seed: int,
             chunk: int = 1000) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, len(hits), size=(k, len(hits)))
        means.append(hits[idx].mean(axis=1))
        done += k
    means = np.concatenate(means)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def layer_curves(scores: pd.DataFrame, n_boot: int = 5000,
                 seed: int = 2026) -> pd.DataFrame:
    rows = []
    for lens in scores["lens"].unique():
        sl_lens = scores[scores["lens"] == lens]
        for layer in sorted(sl_lens["layer"].unique()):
            sl = sl_lens[sl_lens["layer"] == layer]
            hits, mrr = _per_turn_hits_mrr(sl)
            lo, hi = _boot_ci(hits, n_boot, seed)
            rows.append({"lens": lens, "layer": int(layer),
                         "top1": float(hits.mean()),
                         "mrr": float(np.nanmean(mrr)),
                         "n": int(len(hits)), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def classify_trajectory(curves: pd.DataFrame, lens: str,
                        n_states: int) -> Dict:
    c = curves[curves["lens"] == lens].sort_values("layer")
    interior_max_layer = int(np.floor(INTERIOR_FRAC * (n_states - 1)))
    interior = c[c["layer"] <= interior_max_layer]
    l_star_row = interior.loc[interior["top1"].idxmax()]
    l_star, l_star_layer = float(l_star_row["top1"]), int(l_star_row["layer"])
    final = float(c.iloc[-1]["top1"])
    after = c[c["layer"] > l_star_layer]

    dip_layer, dip_value = None, None
    pre_final = after[after["layer"] < c["layer"].max()]
    if len(pre_final):
        dip_row = pre_final.loc[pre_final["top1"].idxmin()]
        dip_layer, dip_value = int(dip_row["layer"]), float(dip_row["top1"])

    result = {"label": "unclassified", "l_star": l_star,
              "l_star_layer": l_star_layer, "final": final,
              "dip_layer": dip_layer, "dip_value": dip_value}
    if l_star < LATE_RISE_FRAC * final:
        result["label"] = "late_rise"
    elif (dip_value is not None and (l_star - dip_value) >= DIP_DEPTH
          and final > dip_value):
        result["label"] = "dip_then_recover"
    elif len(after) == 0 or (after["top1"] >= l_star - PLATEAU_BAND).all():
        result["label"] = "plateau"
    return result


def emergence_depth(curves: pd.DataFrame, lens: str) -> int:
    c = curves[curves["lens"] == lens].sort_values("layer")
    final = float(c.iloc[-1]["top1"])
    ok = c[c["top1"] >= EMERGENCE_FRAC * final]
    return int(ok.iloc[0]["layer"]) if len(ok) else int(c.iloc[-1]["layer"])


def shuffle_control(scores: pd.DataFrame, n_perm: int = 100,
                    seed: int = 2026) -> pd.DataFrame:
    """Label-shuffle null (lens_spec §3): permute target/non-target labels
    within each turn.

    Under a within-turn label permutation, the fixed top-1 slot receives a
    uniformly random candidate's label, so the per-turn hit is exactly
    Bernoulli(n_targets / n_candidates). The permutation null is therefore
    simulated directly from those per-turn probabilities — identical in
    distribution to materialising the permutations, and tractable at
    N = 7,703 x 33 layers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for lens in scores["lens"].unique():
        sl_lens = scores[scores["lens"] == lens]
        for layer in sorted(sl_lens["layer"].unique()):
            sl = sl_lens[sl_lens["layer"] == layer]
            grp = sl.groupby("row_id")["word_type"]
            p = (grp.apply(lambda s: (s == "target").mean())
                 .to_numpy(dtype=float))
            draws = rng.random((n_perm, len(p))) < p[None, :]
            accs = draws.mean(axis=1)
            rows.append({"lens": lens, "layer": int(layer),
                         "top1": float(accs.mean()),
                         "top1_p97_5": float(np.quantile(accs, 0.975))})
    return pd.DataFrame(rows)


def calibration_check(scores: pd.DataFrame, generation_csv: str) -> Dict:
    """§3 (amended 2026-07-27): the output-channel identity only covers
    generations that open with the parsed word; scaffolded generations
    ("The word that best matches ... is X") place the word past the dumped
    position, so the gate uses the word-first subsample and the full-sample
    agreement is descriptive only."""
    gen = pd.read_csv(generation_csv)
    gen = gen[gen["generated_in_candidates"] == True]  # noqa: E712
    final_layer = int(scores["layer"].max())
    top = scores[(scores["lens"] == "raw")
                 & (scores["layer"] == final_layer)
                 & (scores["rank"] == 1)][["row_id", "word", "word_type"]]
    merged = gen.merge(top, on="row_id", how="inner")
    agree = (merged["generated_word"].str.lower()
             == merged["word"].str.lower())
    word_first = merged.apply(
        lambda r: str(r["generated_text"]).lstrip().lstrip("\"'`")
        .lower().startswith(str(r["generated_word"]).lower()), axis=1)
    wf_agreement = (float(agree[word_first].mean())
                    if word_first.any() else float("nan"))
    return {"agreement": float(agree.mean()),
            "n_parseable": int(len(merged)),
            "word_first_agreement": wf_agreement,
            "n_word_first": int(word_first.sum()),
            "lens_top1_final": float((top["word_type"] == "target").mean()),
            "generation_acc": float(gen["generated_correct"].mean()),
            "passes_gate": bool(word_first.any()
                                and wf_agreement >= CALIBRATION_GATE)}


def spearman_vs_geometric(curves: pd.DataFrame, lens: str,
                          concordance_csv: str, model_key: str,
                          condition: str = "no_social",
                          pooling: str = "mean",
                          n_boot: int = 5000, seed: int = 2026) -> Dict:
    g = pd.read_csv(concordance_csv)
    g = g[(g["model"] == model_key) & (g["condition"] == condition)
          & (g["pooling"] == pooling)][["layer", "top1_accuracy"]]
    c = curves[curves["lens"] == lens][["layer", "top1"]]
    m = c.merge(g, on="layer", how="inner")
    n_states = int(m["layer"].max()) + 1
    m = m[m["layer"] <= int(np.floor(INTERIOR_FRAC * (n_states - 1)))]
    rho = float(spearmanr(m["top1"], m["top1_accuracy"]).statistic)
    rng = np.random.default_rng(seed)
    arr = m[["top1", "top1_accuracy"]].to_numpy()
    boots = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), size=len(arr))]
        boots.append(spearmanr(s[:, 0], s[:, 1]).statistic)
    return {"spearman": rho,
            "ci_lo": float(np.nanquantile(boots, 0.025)),
            "ci_hi": float(np.nanquantile(boots, 0.975)),
            "n_layers": int(len(m))}


def trained_vs_random(curves_trained: pd.DataFrame,
                      curves_random: pd.DataFrame) -> pd.DataFrame:
    t = curves_trained.rename(columns={"top1": "top1_trained"})
    r = curves_random.rename(columns={"top1": "top1_random",
                                      "ci_hi": "random_ci_hi"})
    m = t.merge(r[["lens", "layer", "top1_random", "random_ci_hi"]],
                on=["lens", "layer"])
    m["delta"] = m["top1_trained"] - m["top1_random"]
    m["exceeds_random_ci"] = m["top1_trained"] > m["random_ci_hi"]
    return m


def run_analysis(
    output_root: str = "output",
    models: Sequence[str] = ("mistral", "qwen"),
    random_model: Optional[str] = "random_qwen",
    mode: str = "no_social",
    out_dir: str = os.path.join("output", "lens_analysis"),
    n_boot: int = 5000,
    seed: int = 2026,
) -> Dict[str, Dict]:
    """Orchestrate the full §7 analysis over saved scores parquets.

    Reads {output_root}/{m}_outputs/{m}_lens_scores_{mode}.parquet for each
    model (including the random-init null), the thesis generation CSVs, and
    the aggregate geometric curve. Writes five CSVs into ``out_dir`` and
    returns the per-model summary dict.
    """
    os.makedirs(out_dir, exist_ok=True)
    concordance_csv = os.path.join(output_root, "analysis",
                                   "analysis_concordance_by_layer.csv")

    def _scores_path(m):
        return os.path.join(output_root, f"{m}_outputs",
                            f"{m}_lens_scores_{mode}.parquet")

    curves_by_model, all_curves, decisions, calibrations, shuffles = \
        {}, [], [], [], []

    random_curves = None
    if random_model is not None and os.path.exists(_scores_path(random_model)):
        rs = pd.read_parquet(_scores_path(random_model))
        random_curves = layer_curves(rs, n_boot=n_boot, seed=seed)
        curves_by_model[random_model] = random_curves
        all_curves.append(random_curves.assign(model=random_model))
        shuffles.append(shuffle_control(rs, seed=seed)
                        .assign(model=random_model))

    summary: Dict[str, Dict] = {}
    for m in models:
        sp = _scores_path(m)
        if not os.path.exists(sp):
            print(f"  [lens-analyze] missing scores for '{m}' "
                  f"({sp}); skipping.")
            continue
        scores = pd.read_parquet(sp)
        curves = layer_curves(scores, n_boot=n_boot, seed=seed)
        curves_by_model[m] = curves
        all_curves.append(curves.assign(model=m))
        n_states = int(scores["layer"].max()) + 1

        gen_csv = os.path.join(output_root, f"{m}_outputs",
                               f"{m}_generation_{mode}.csv")
        calib = (calibration_check(scores, gen_csv)
                 if os.path.exists(gen_csv) else
                 {"agreement": np.nan, "n_parseable": 0,
                  "word_first_agreement": np.nan, "n_word_first": 0,
                  "lens_top1_final": np.nan, "generation_acc": np.nan,
                  "passes_gate": False})
        calibrations.append({"model": m, **calib})
        if not calib["passes_gate"]:
            print(f"  WARNING: calibration gate FAILED for '{m}' "
                  f"(word-first agreement="
                  f"{calib['word_first_agreement']} on "
                  f"n={calib['n_word_first']}); downstream claims "
                  f"for this model are BLOCKED (lens_spec §3).")

        for lens in scores["lens"].unique():
            cls = classify_trajectory(curves, lens, n_states)
            row = {"model": m, "lens": lens, **cls,
                   "emergence_depth": emergence_depth(curves, lens),
                   "blocked": not calib["passes_gate"]}
            if os.path.exists(concordance_csv):
                row.update({f"geo_{k}": v for k, v in
                            spearman_vs_geometric(
                                curves, lens, concordance_csv, m,
                                condition=mode, n_boot=n_boot,
                                seed=seed).items()})
            decisions.append(row)

        shuffles.append(shuffle_control(scores, seed=seed).assign(model=m))
        summary[m] = {"curves": curves, "calibration": calib}

    if not all_curves:
        print("  [lens-analyze] no lens scores found for any model — run "
              "lens-extract and lens-apply first. Nothing written.")
        return summary

    pd.concat(all_curves, ignore_index=True).to_csv(
        os.path.join(out_dir, f"lens_curves_{mode}.csv"), index=False)
    pd.DataFrame(decisions).to_csv(
        os.path.join(out_dir, f"lens_decision_table_{mode}.csv"), index=False)
    pd.DataFrame(calibrations).to_csv(
        os.path.join(out_dir, f"lens_calibration_{mode}.csv"), index=False)
    pd.concat(shuffles, ignore_index=True).to_csv(
        os.path.join(out_dir, f"lens_shuffle_control_{mode}.csv"), index=False)

    if random_curves is not None:
        deltas = []
        for m in models:
            if m in curves_by_model:
                deltas.append(trained_vs_random(
                    curves_by_model[m], random_curves).assign(model=m))
        if deltas:
            pd.concat(deltas, ignore_index=True).to_csv(
                os.path.join(out_dir, f"lens_trained_vs_random_{mode}.csv"),
                index=False)

    print(f"  lens analysis written to {out_dir}")
    return summary
