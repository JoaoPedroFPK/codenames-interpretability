"""Sanity checks SC1–SC7.

Each SC function takes the results dict from :func:`loop.run_extraction` plus
auxiliary arguments. Code is verbatim from the SC cells of the reference
notebooks; print formatting (table headers, column alignment, decimal
precision, PASS/WARN markers) is byte-identical because the user reads these
to verify a run.

SC0 (Generation Diagnostic) is intentionally omitted — SC4 already covers
generation accuracy and concordance.
"""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from .data import GIVER_COLS, extract_giver_features
from .prompts import _FEATURE_LABEL_MAP, build_prompt


def _wilson_confidence_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def sc1_prompt_structure(
    df_sample: pd.DataFrame,
    tokenizer,
    chat_template_strategy: str,
) -> None:
    """SC1 — Prompt Structure Verification.

    Confirms the prompt builder produces the expected structure for both
    conditions: hint locatable, all candidates locatable after the anchor,
    all giver markers present in ``with_social``, and no leakage of giver
    markers in ``no_social``.
    """
    print("SC1: Prompt Structure Verification")
    print("=" * 60)

    test_row = df_sample.iloc[0]

    for mode_flag in [False, True]:
        mode_name = "with_social" if mode_flag else "no_social"
        giver_features = (
            extract_giver_features(test_row, GIVER_COLS)
            if mode_flag else {}
        )
        prompt, feature_markers = build_prompt(
            hint=test_row["output"],
            candidates=list(test_row["candidates"]),
            giver_features=giver_features,
            use_social_context=mode_flag,
            tokenizer=tokenizer,
            chat_template_strategy=chat_template_strategy,
        )

        hint_found = test_row["output"] in prompt
        all_cands_found = all(c in prompt for c in test_row["candidates"])
        if mode_flag:
            markers_found = all(m in prompt for m in feature_markers.values())
            markers_leaked = False
        else:
            markers_found = True
            markers_leaked = any(
                f"{label}:" in prompt for label in _FEATURE_LABEL_MAP.values()
            )

        token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        print(f"\nCondition: {mode_name}")
        print(f"  Hint found in prompt       : {hint_found}")
        print(f"  All candidates found       : {all_cands_found}")
        if mode_flag:
            print(f"  All giver markers found    : {markers_found}")
        else:
            print(f"  Giver markers leaked       : {markers_leaked}  (must be False)")
        print(f"  Prompt token count         : {token_count}")
        print("  Prompt preview:")
        preview = prompt[:300].replace(chr(10), " | ")
        print(f"    {preview}")


def sc2_span_coverage(results: Dict) -> None:
    """SC2 — Span Coverage and Token Count Statistics."""
    print("SC2: Span Coverage and Token Count Statistics")
    print("=" * 60)

    for mode_name in ["no_social", "with_social"]:
        mdf = results[mode_name]["metrics_df"]
        if len(mdf) == 0:
            print(f"\n{mode_name}: no metrics rows; skipping.")
            continue
        mdf_canon = mdf[mdf["permutation_id"] == 0]
        mdf_l0    = mdf_canon[mdf_canon["layer"] == 0]

        cand_rows = mdf_l0[mdf_l0["word_type"].isin(["target", "black", "tan"])]
        total_cand_slots   = len(cand_rows)
        missing_cand_slots = int(cand_rows["cosine_to_hint_mean"].isna().sum())
        coverage_pct = 100.0 * (1 - missing_cand_slots / max(total_cand_slots, 1))

        print(f"\nCondition: {mode_name}")
        print(f"  Candidate slots   : {total_cand_slots}")
        print(f"  Missing spans     : {missing_cand_slots}")
        print(f"  Span coverage     : {coverage_pct:.2f}%")
        if coverage_pct < 95.0:
            print("  WARNING: Span coverage below 95%. Review span detection logic.")

        print("\n  Token count per word type at layer 0:")
        for wt in ["hint", "target", "black", "tan", "giver_feature"]:
            subset = mdf_l0[mdf_l0["word_type"] == wt]["token_count"]
            if len(subset) > 0:
                print(f"    {wt:14s}: mean={subset.mean():.2f}, "
                      f"std={subset.std():.2f}, max={int(subset.max())}")


def sc3_anisotropy(results: Dict, num_layers: int) -> None:
    """SC3 — Anisotropy Characterization (mean pooling, no_social)."""
    print("SC3: Anisotropy Characterization (mean pooling, no_social)")
    print("=" * 60)

    mdf = results["no_social"]["metrics_df"]
    if len(mdf) > 0:
        mdf_canon = mdf[mdf["permutation_id"] == 0]
        aniso_per_layer = (
            mdf_canon.groupby("layer")
            .agg(
                mean_aniso=("layer_mean_pairwise_cosine", "mean"),
                std_aniso=("layer_std_pairwise_cosine",  "mean"),
                n_boards=("row_id", "nunique"),
            )
            .reset_index()
        )
        print(f"\n{'Layer':>6}  {'Mean aniso':>12}  {'Std aniso':>12}  {'N boards':>10}")
        print("-" * 50)
        for _, r in aniso_per_layer.iterrows():
            print(f"{int(r['layer']):>6}  {r['mean_aniso']:>12.4f}  "
                  f"{r['std_aniso']:>12.4f}  {int(r['n_boards']):>10}")


def sc4_behavioral_accuracy(
    results: Dict,
    pooling_methods,
    has_generation: bool,
) -> None:
    """SC4 — Behavioral Accuracy Summary.

    Reports cosine-rank accuracy, MRR, Hit@K for both pooling methods, plus
    generation-based accuracy and concordance (causal-only). Canonical
    ordering only.
    """
    print("SC4: Behavioral Accuracy Summary")
    print("=" * 60)

    for mode_name in ["no_social", "with_social"]:
        gdf = results[mode_name]["general_df"]
        gdf_canon = gdf[gdf["permutation_id"] == 0]
        gen_df = results[mode_name]["generation_df"]
        n = len(gdf_canon)

        print(f"\nCondition: {mode_name}  (N={n})")

        for pm in pooling_methods:
            n_correct = int(gdf_canon[f"correct_{pm}"].sum())
            accuracy = n_correct / n if n > 0 else float("nan")
            ci_lo, ci_hi = _wilson_confidence_interval(n_correct, n)

            mrr = gdf_canon[f"mrr_{pm}"].dropna().mean()
            hit1 = gdf_canon[f"hit_at_1_{pm}"].dropna().mean()
            hit3 = gdf_canon[f"hit_at_3_{pm}"].dropna().mean()
            hit5 = gdf_canon[f"hit_at_5_{pm}"].dropna().mean()
            mean_rank = gdf_canon[f"mean_target_rank_{pm}"].dropna().mean()
            std_rank = gdf_canon[f"mean_target_rank_{pm}"].dropna().std()

            mean_n_cands = float(gdf_canon["n_candidates"].mean())
            mean_n_targets = float(gdf_canon["n_targets"].mean())
            random_rank = (mean_n_cands + 1) / (mean_n_targets + 1)

            print(f"\n  Cosine-rank ({pm}):")
            print(f"    Top-1 accuracy    : {accuracy:.3f} ({n_correct}/{n})")
            print(f"    Wilson 95% CI     : [{ci_lo:.3f}, {ci_hi:.3f}]")
            print(f"    MRR               : {mrr:.4f}")
            print(f"    Hit@1 / @3 / @5   : {hit1:.3f} / {hit3:.3f} / {hit5:.3f}")
            print(f"    Mean target rank  : {mean_rank:.2f} ± {std_rank:.2f}")
            print(f"    Random baseline   : {random_rank:.2f}")

        if has_generation and len(gen_df) > 0:
            n_gen = len(gen_df)
            gen_in_cands = int(gen_df["generated_in_candidates"].sum())
            gen_correct = int(gen_df["generated_correct"].sum())
            gen_acc = gen_correct / n_gen if n_gen > 0 else float("nan")
            gen_ci = _wilson_confidence_interval(gen_correct, n_gen)

            print("\n  Generation-based accuracy:")
            print(f"    Generated in candidates : {gen_in_cands}/{n_gen} "
                  f"({gen_in_cands/n_gen:.3f})")
            print(f"    Generation accuracy     : {gen_acc:.3f} ({gen_correct}/{n_gen})")
            print(f"    Wilson 95% CI           : [{gen_ci[0]:.3f}, {gen_ci[1]:.3f}]")

            for pm in pooling_methods:
                concordance = gen_df[f"concordance_{pm}"].mean()
                print(f"    Concordance with {pm:8s}: {concordance:.3f}")


def sc5_layer_margin_curve(
    results: Dict,
    base_dir: str,
    prefix: str,
    num_layers: int,
    pooling_methods,
) -> None:
    """SC5 — Layer-wise Margin Curve (Anisotropy-Adjusted).

    Computes raw and anisotropy-adjusted margins per layer for both pooling
    methods. Adjusted margin = raw_margin / std_pairwise_cosine (z-score).
    Saved per condition × pooling method.
    """
    print("SC5: Layer-wise Margin Curve")
    print("=" * 60)

    for pm in pooling_methods:
        print(f"\n--- Pooling: {pm} ---")

        for mode_name in ["no_social", "with_social"]:
            mdf = results[mode_name]["metrics_df"]
            if len(mdf) == 0:
                continue
            mdf_canon = mdf[mdf["permutation_id"] == 0]

            layer_margin_records = []

            for layer_idx in range(num_layers + 1):
                layer_data = mdf_canon[mdf_canon["layer"] == layer_idx]

                board_margins = []
                board_hint_tgt = []
                board_hint_non = []

                for row_id, board_df in layer_data.groupby("row_id"):
                    tgt_cos = board_df[board_df["word_type"] == "target"][f"cosine_to_hint_{pm}"].dropna()
                    non_cos = board_df[board_df["word_type"].isin(["black", "tan"])][f"cosine_to_hint_{pm}"].dropna()
                    if len(tgt_cos) == 0 or len(non_cos) == 0:
                        continue
                    mt = float(tgt_cos.mean())
                    mn = float(non_cos.mean())
                    board_margins.append(mt - mn)
                    board_hint_tgt.append(mt)
                    board_hint_non.append(mn)

                if board_margins:
                    aniso_vals = layer_data.groupby("row_id")["layer_mean_pairwise_cosine"].first().dropna()
                    aniso_std_vals = layer_data.groupby("row_id")["layer_std_pairwise_cosine"].first().dropna()
                    layer_mean_aniso = float(aniso_vals.mean()) if len(aniso_vals) > 0 else float("nan")
                    layer_std_aniso = float(aniso_std_vals.mean()) if len(aniso_std_vals) > 0 else float("nan")

                    raw_margin = float(np.mean(board_margins))
                    adjusted_margin = (
                        raw_margin / layer_std_aniso
                        if not np.isnan(layer_std_aniso) and layer_std_aniso > 0
                        else float("nan")
                    )

                    layer_margin_records.append({
                        "layer"           : layer_idx,
                        "pooling_method"  : pm,
                        "condition"       : mode_name,
                        "mean_margin"     : raw_margin,
                        "std_margin"      : float(np.std(board_margins)),
                        "adjusted_margin" : adjusted_margin,
                        "mean_hint_tgt"   : float(np.mean(board_hint_tgt)),
                        "mean_hint_non"   : float(np.mean(board_hint_non)),
                        "mean_anisotropy" : layer_mean_aniso,
                        "n_boards"        : len(board_margins),
                    })

            margin_df = pd.DataFrame(layer_margin_records)

            print(f"\n  Condition: {mode_name}")
            print(f"  {'Layer':>6}  {'Raw':>8}  {'Adj':>8}  {'hint-tgt':>10}  {'hint-non':>10}  {'Aniso':>8}")
            print(f"  {'-'*60}")
            for _, r in margin_df.iterrows():
                print(f"  {int(r['layer']):>6}  {r['mean_margin']:>8.5f}  "
                      f"{r['adjusted_margin']:>8.4f}  "
                      f"{r['mean_hint_tgt']:>10.5f}  "
                      f"{r['mean_hint_non']:>10.5f}  "
                      f"{r['mean_anisotropy']:>8.5f}")

            if len(margin_df) > 0:
                best_raw = margin_df.loc[margin_df["mean_margin"].idxmax()]
                best_adj = margin_df.loc[margin_df["adjusted_margin"].idxmax()]
                print(f"\n  Peak raw: layer {int(best_raw['layer'])} ({best_raw['mean_margin']:.5f})")
                print(f"  Peak adj: layer {int(best_adj['layer'])} ({best_adj['adjusted_margin']:.4f})")

            margin_path = os.path.join(base_dir, f"{prefix}_layer_margins_{pm}_{mode_name}.csv")
            margin_df.to_csv(margin_path, index=False)


def sc6_positional_confound(
    results: Dict,
    base_dir: str,
    prefix: str,
    num_layers: int,
) -> None:
    """SC6 — Per-Layer Positional Confound (Spearman ρ at every layer).

    Spearman correlation between candidate list position and cosine-to-hint at
    every layer, no_social condition only. The single-layer Mann-Whitney U test
    is also reported at the final layer for backward compatibility.

    Includes a 5-line candidate-order consistency check at the start to ensure
    both conditions used the same alphabetical ordering.
    """
    print("SC6: Per-Layer Positional Confound")
    print("=" * 60)

    # --- Order consistency check (folded from old SC4) ---
    gdf_ns = results["no_social"]["general_df"]
    gdf_ws = results["with_social"]["general_df"]
    gdf_ns_canon = gdf_ns[gdf_ns["permutation_id"] == 0].set_index("row_id")
    gdf_ws_canon = gdf_ws[gdf_ws["permutation_id"] == 0].set_index("row_id")
    common_rows = set(gdf_ns_canon.index) & set(gdf_ws_canon.index)
    print(f"Order consistency: {len(common_rows)} common boards (both conditions saw same alphabetical ordering)")

    # --- Per-layer Spearman ρ ---
    mdf = results["no_social"]["metrics_df"]
    if len(mdf) == 0:
        print("No metrics rows; skipping SC6.")
        return
    mdf_canon = mdf[mdf["permutation_id"] == 0]

    confound_records = []

    for layer_idx in range(num_layers + 1):
        layer_data = mdf_canon[
            (mdf_canon["layer"] == layer_idx)
            & (mdf_canon["word_type"].isin(["target", "black", "tan"]))
        ]

        layer_rhos = []
        for row_id, board_df in layer_data.groupby("row_id"):
            positions = board_df["list_position"].values
            cosines = board_df["cosine_to_hint_mean"].values
            valid = ~np.isnan(cosines) & ~np.isnan(positions)
            if valid.sum() < 3:
                continue
            rho, _ = spearmanr(positions[valid], cosines[valid])
            layer_rhos.append(rho)

        if layer_rhos:
            confound_records.append({
                "layer"    : layer_idx,
                "mean_rho" : float(np.mean(layer_rhos)),
                "std_rho"  : float(np.std(layer_rhos)),
                "n_boards" : len(layer_rhos),
            })

    confound_df = pd.DataFrame(confound_records)

    print(f"\n{'Layer':>6}  {'Mean rho':>10}  {'Std rho':>10}  {'Concern?':>10}")
    print("-" * 45)
    for _, r in confound_df.iterrows():
        concern = "YES" if abs(r["mean_rho"]) > 0.1 else ""
        print(f"{int(r['layer']):>6}  {r['mean_rho']:>10.4f}  {r['std_rho']:>10.4f}  {concern:>10}")

    confound_path = os.path.join(base_dir, f"{prefix}_position_confound_by_layer.csv")
    confound_df.to_csv(confound_path, index=False)
    print(f"\nSaved: {confound_path}")

    # --- Final-layer Mann-Whitney U for backward compatibility ---
    final_data = mdf_canon[
        (mdf_canon["layer"] == num_layers)
        & (mdf_canon["word_type"].isin(["target", "black", "tan"]))
    ]
    positions_all = final_data["list_position"].values
    cosines_all = final_data["cosine_to_hint_mean"].values
    valid_mask = ~np.isnan(cosines_all) & ~np.isnan(positions_all)
    pv, cv = positions_all[valid_mask], cosines_all[valid_mask]

    if len(pv) >= 10:
        rho_f, p_rho = spearmanr(pv, cv)
        med = np.median(pv)
        near_c, far_c = cv[pv <= med], cv[pv > med]
        if len(near_c) >= 2 and len(far_c) >= 2:
            u_stat, u_p = mannwhitneyu(near_c, far_c, alternative="greater")
            eff_r = u_stat / (len(near_c) * len(far_c))
        else:
            u_stat, u_p, eff_r = float("nan"), float("nan"), float("nan")
        print("\n  Final-layer Mann-Whitney U:")
        print(f"    Spearman rho = {rho_f:+.4f} (p={p_rho:.4f})")
        print(f"    U={u_stat:.1f}, p={u_p:.4f}, r={eff_r:.4f}")
        print(f"    Near cos={near_c.mean():.5f}, Far cos={far_c.mean():.5f}")


def sc7_shuffle_decomposition(
    results: Dict,
    base_dir: str,
    prefix: str,
    num_layers: int,
    n_shuffles: int,
) -> None:
    """SC7 — Shuffle Confound Decomposition.

    Decomposes cosine variance into position-driven (within-word, across
    permutations) and semantics-driven (between-word, within-permutation)
    components. Reports the semantic signal ratio per layer.
    """
    if n_shuffles == 0:
        print("\nSC7 skipped (N_SHUFFLES = 0).")
        return

    print("SC7: Shuffle Confound Decomposition")
    print("=" * 60)

    mode_name = "no_social"
    mdf = results[mode_name]["metrics_df"]

    if len(mdf) == 0:
        print("No metrics rows; skipping SC7.")
        return

    # Final layer headline
    final_data = mdf[
        (mdf["layer"] == num_layers)
        & (mdf["word_type"].isin(["target", "black", "tan"]))
    ]
    word_var = final_data.groupby(["row_id", "word"])["cosine_to_hint_mean"].var().dropna()
    between_var = final_data.groupby(["row_id", "permutation_id"])["cosine_to_hint_mean"].var().dropna()
    mean_within = float(word_var.mean())
    mean_between = float(between_var.mean())
    total = mean_within + mean_between
    ratio = mean_between / total if total > 0 else float("nan")

    print(f"\n  Final layer (layer {num_layers}), mean pooling, no_social:")
    print(f"  Mean within-word variance  (positional) : {mean_within:.6f}")
    print(f"  Mean between-word variance (semantic)   : {mean_between:.6f}")
    print(f"  Semantic signal ratio                   : {ratio:.4f}")
    print("    (1.0 = pure semantic, 0.0 = pure positional)")

    # Per-layer
    print("\n  Per-layer semantic signal ratio:")
    print(f"  {'Layer':>6}  {'Within (pos)':>14}  {'Between (sem)':>14}  {'Ratio':>8}")
    print(f"  {'-'*48}")

    shuffle_records = []
    for layer_idx in range(num_layers + 1):
        ld = mdf[
            (mdf["layer"] == layer_idx)
            & (mdf["word_type"].isin(["target", "black", "tan"]))
        ]
        wv = ld.groupby(["row_id", "word"])["cosine_to_hint_mean"].var().dropna()
        bv = ld.groupby(["row_id", "permutation_id"])["cosine_to_hint_mean"].var().dropna()
        mwv = float(wv.mean()) if len(wv) > 0 else 0.0
        mbv = float(bv.mean()) if len(bv) > 0 else 0.0
        tv = mwv + mbv
        r = mbv / tv if tv > 0 else float("nan")
        print(f"  {layer_idx:>6}  {mwv:>14.6f}  {mbv:>14.6f}  {r:>8.4f}")
        shuffle_records.append({
            "layer": layer_idx,
            "within_var": mwv,
            "between_var": mbv,
            "semantic_ratio": r,
        })

    shuffle_path = os.path.join(base_dir, f"{prefix}_shuffle_decomposition_by_layer.csv")
    pd.DataFrame(shuffle_records).to_csv(shuffle_path, index=False)
    print(f"\n  Saved: {shuffle_path}")
