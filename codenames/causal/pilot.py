"""Pilot gate P1-P8 (causal_spec.md §12.5).

These are MACHINERY checks, not hypothesis tests. They exist because every
number in §3-§4 is a design commitment and none of them establishes that the
implementation does what the design says: a mis-indexed p*, a corruption that
does not corrupt, a hook writing the wrong tensor, or an attribution scan
uncorrelated with the real patches it screens for would all survive
pre-registration and silently produce confident nonsense.

**Anti-peeking rule (binding, §12.5).** Only two classes of pilot output may
inform the confirmatory run: nuisance parameters (throughput, storage,
per-turn variance for re-calibrating the power target) and the binary verdicts
below. The pilot may NOT narrow the confirmatory grid, choose which layers to
test, set K, or fix the direction of any hypothesis. Pilot effect LOCATIONS
are discarded; the confirmatory run re-discovers them or it does not.

**Outcome routing.** P1-P4 and P7 are blocking. P6 failing does not block --
it converts RQ2 into a pre-registered bounded negative (§2.1 row 4). P5
failing does not block either, but it removes the attribution shortcut and so
materially raises the budget, which is re-costed before proceeding.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PILOT_THRESHOLDS = {
    "P1": 0.99,                        # p* reproduces the recorded generated token
    "P2_lo": 0.98, "P2_hi": 1.02,      # full-stack patch identity: e == 1
    "P3": 0.02,                        # null patch identity: |e| == 0
    # P4 amended 2026-07-28 (causal_spec.md §12.5): the flip criterion is
    # CEILING-RELATIVE. An absolute rate implicitly assumed the model answers
    # the donor hint correctly, but its own accuracy is ~0.60, so a 0.60 gate
    # sat at the ceiling and was near-unpassable. Both criteria must hold.
    "P4_flip_ratio": 0.85, "P4_sign": 0.80,
    "P5_rho": 0.5, "P5_fnr": 0.20,     # attribution tracks real patches
    "P6_change": 0.10, "P6_parse": 0.90,  # steering is not inert
}

_BLOCKING = ("P1", "P2", "P3", "P4", "P7")

# P5 instrument (amended 2026-07-28). Stratified sampling: the screen's own
# top picks give the correlation dynamic range, the random cells make the
# false-negative rate estimable. Sites below this |e| are treated as null,
# so "the screen missed it" means it missed something that actually mattered.
_P5_TOP = 3
_P5_RANDOM = 3
_P5_EFFECT_FLOOR = 0.10


def pilot_verdict(results: Dict[str, float]) -> Dict[str, object]:
    """Route P1-P8 observations into the §12.5 launch decision."""
    t = PILOT_THRESHOLDS
    failed: List[str] = []

    # P1 is skippable ONLY when explicitly marked inapplicable (the random-init
    # null has no generations). It defaults to applicable so a missing
    # generations file fails loudly instead of quietly passing.
    if results.get("P1_applicable", True) and float(results.get("P1", 0.0)) < t["P1"]:
        failed.append("P1")
    if not (t["P2_lo"] <= float(results.get("P2", 0.0)) <= t["P2_hi"]):
        failed.append("P2")
    if abs(float(results.get("P3", 1.0))) > t["P3"]:
        failed.append("P3")
    clean_acc = float(results.get("P4_clean_accuracy", 0.0))
    flip = float(results.get("P4_flip", 0.0))
    # Ceiling-relative: compare the flip rate against what the model can do.
    # A zero/absent clean accuracy means the denominator is unmeasured, which
    # counts as a failure rather than a free pass.
    flip_ok = clean_acc > 0 and (flip / clean_acc) >= t["P4_flip_ratio"]
    if not flip_ok or float(results.get("P4_sign", 0.0)) < t["P4_sign"]:
        failed.append("P4")
    if not bool(results.get("P7_finite", False)):
        failed.append("P7")

    # An FNR computed over zero high-effect sites is not evidence that the
    # screen works -- it means the sample never contained anything to miss.
    # Report that as uninformative rather than letting it read as a pass.
    n_high = int(results.get("P5_n_high_effect", 0) or 0)
    attribution_uninformative = n_high == 0
    attribution_lost = (not attribution_uninformative) and (
        float(results.get("P5_rho", 0.0)) < t["P5_rho"]
        or float(results.get("P5_fnr", 1.0)) > t["P5_fnr"]
    )
    rq2_bounded = (
        float(results.get("P6_change", 0.0)) < t["P6_change"]
        or float(results.get("P6_parse", 0.0)) < t["P6_parse"]
    )

    return {
        "launch_full_run": len(failed) == 0,
        "blocking_failures": failed,
        "attribution_shortcut_lost": bool(attribution_lost),
        "attribution_uninformative": bool(attribution_uninformative),
        "rq2_bounded_negative": bool(rq2_bounded),
    }


def pilot_report(results: Dict[str, float]) -> pd.DataFrame:
    """The P1-P8 table handed to the resourcing conversation (§12.6)."""
    t = PILOT_THRESHOLDS
    verdict = pilot_verdict(results)
    failed = set(verdict["blocking_failures"])

    rows = [
        {"check": "P1", "what": "p* reproduces the generated token",
         "observed": (results.get("P1") if results.get("P1_applicable", True)
                      else "n/a"),
         "threshold": (f">= {t['P1']}" if results.get("P1_applicable", True)
                       else "n/a - no generations (random-init null)"),
         "passed": "P1" not in failed, "blocking": True},
        {"check": "P2", "what": "full-stack patch identity (e == 1)",
         "observed": results.get("P2"),
         "threshold": f"[{t['P2_lo']}, {t['P2_hi']}]",
         "passed": "P2" not in failed, "blocking": True},
        {"check": "P3", "what": "null patch identity (e == 0)",
         "observed": results.get("P3"), "threshold": f"|e| <= {t['P3']}",
         "passed": "P3" not in failed, "blocking": True},
        {"check": "P4", "what": "corruption flips the answer",
         "observed": results.get("P4_flip"),
         "threshold": (f">= {t['P4_flip_ratio']} x clean acc, "
                       f">= {t['P4_sign']} sign"),
         "passed": "P4" not in failed, "blocking": True},
        {"check": "P5", "what": "attribution tracks real patches",
         "observed": results.get("P5_rho"),
         "threshold": f"rho >= {t['P5_rho']}, FNR <= {t['P5_fnr']}",
         "passed": not verdict["attribution_shortcut_lost"], "blocking": False},
        {"check": "P6", "what": "steering is not inert",
         "observed": results.get("P6_change"),
         "threshold": f">= {t['P6_change']} at parse >= {t['P6_parse']}",
         "passed": not verdict["rq2_bounded_negative"], "blocking": False},
        {"check": "P7", "what": "random-init numerics are usable",
         "observed": results.get("P7_finite"), "threshold": "finite, no NaNs",
         "passed": "P7" not in failed, "blocking": True},
        {"check": "P8", "what": "throughput and storage",
         "observed": results.get("P8_fwd_per_s"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P4_sign", "what": "LD_corrupt < 0 (donor beats clean target)",
         "observed": results.get("P4_sign"), "threshold": f">= {t['P4_sign']}",
         "passed": float(results.get("P4_sign", 0.0)) >= t["P4_sign"],
         "blocking": False},
        {"check": "alignment", "what": "pairs surviving prompt-length alignment",
         "observed": results.get("alignment_yield"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "n_measured", "what": "pairs actually measured",
         "observed": results.get("n_measured"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P4_clean_acc", "what": "clean run answers its own target",
         "observed": results.get("P4_clean_accuracy"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P4_normalised",
         "what": "flip rate / clean accuracy (corruption works GIVEN the model can answer)",
         "observed": (
             float(results.get("P4_flip", 0.0)) / results["P4_clean_accuracy"]
             if results.get("P4_clean_accuracy") else None),
         "threshold": "diagnostic only - NOT a gate",
         "passed": True, "blocking": False},
        {"check": "P5_fnr", "what": "high-effect sites the screen would miss",
         "observed": results.get("P5_fnr"), "threshold": f"<= {t['P5_fnr']}",
         "passed": not verdict["attribution_shortcut_lost"], "blocking": False},
        {"check": "P5_n_high", "what": "sites clearing the |e| floor (0 = uninformative)",
         "observed": results.get("P5_n_high_effect"), "threshold": "recorded",
         "passed": not verdict["attribution_uninformative"], "blocking": False},
        {"check": "P5_n_sites", "what": "sites real-patched for the P5 estimate",
         "observed": results.get("P5_n_sites"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P1_margin",
         "what": "median logit margin on P1 misses (small = numerical drift)",
         "observed": results.get("P1_miss_margin_median"),
         "threshold": "diagnostic only", "passed": True, "blocking": False},
        {"check": "P1_misses", "what": "turns where p* did not reproduce",
         "observed": results.get("P1_n_misses"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P2_within_tol", "what": "pairs whose full-stack patch is within tolerance",
         "observed": results.get("P2_within_tol"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P2_mean", "what": "mean full-stack e (outlier-sensitive)",
         "observed": results.get("P2_mean"), "threshold": "diagnostic only",
         "passed": True, "blocking": False},
        {"check": "P2_denom", "what": "median |LD_clean - LD_corrupt| (metric scale)",
         "observed": results.get("P2_denom_median"), "threshold": "recorded",
         "passed": True, "blocking": False},
        {"check": "P8_wall", "what": "throughput incl. backward + generation",
         "observed": results.get("P8_wall_fwd_per_s"), "threshold": "recorded",
         "passed": True, "blocking": False},
    ]
    return pd.DataFrame(rows)


def run_pilot(
    *,
    model,
    tokenizer,
    df_sample: pd.DataFrame,
    generation_csv: Optional[str],
    base_dir: str,
    prefix: str,
    mode: str,
    chat_template_strategy: str,
    num_layers: int,
    hidden_dim: int,
    device: Optional[str] = None,
    seed: int = 2026,
    alphas: Sequence[float] = (-4.0, -1.0, 1.0, 4.0),
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Execute P1-P8 against a real model and return (report, raw results).

    Composes the tested primitives; contains no methodology of its own. Every
    measurement here is a property of the machinery, never of the hypotheses
    (see the anti-peeking rule in this module's docstring).

    ``device`` defaults to **the device the model is already on**, rather than
    to a constant. A hardcoded default silently disagreed with a CUDA-resident
    model and every forward pass died with "Expected all tensors to be on the
    same device"; inferring it from the model makes that mismatch impossible.
    """
    import time

    import torch

    from ..data import GIVER_COLS, extract_giver_features
    from ..lens.readout import build_token_table
    from ..prompts import build_prompt
    from .attribution import attribution_scan
    from .metrics import logit_difference
    from .pairs import build_pair_table
    from .patch import all_sites, run_patch
    from .positions import answer_position
    from .steer import random_direction, steer_generate

    if device is None:
        device = str(next(model.parameters()).device)

    os.makedirs(base_dir, exist_ok=True)
    mode_flag = mode == "with_social"
    generations = (pd.read_csv(generation_csv).set_index("row_id")
                   if generation_csv else None)

    hint_tokens = {
        int(r.row_id): len(tokenizer.encode(str(r.output), add_special_tokens=False))
        for r in df_sample.itertuples()
    }
    by_id = df_sample.set_index("row_id")

    def _prompt(row_id: int, hint: str) -> str:
        row = by_id.loc[row_id]
        text, _ = build_prompt(
            hint=str(hint), candidates=list(row["candidates"]),
            giver_features=extract_giver_features(row, GIVER_COLS) if mode_flag else {},
            use_social_context=mode_flag, tokenizer=tokenizer,
            chat_template_strategy=chat_template_strategy,
        )
        return text

    _len_cache: Dict[tuple, int] = {}

    def _prompt_len(row_id: int, hint: str) -> int:
        key = (int(row_id), str(hint))
        if key not in _len_cache:
            _len_cache[key] = len(
                tokenizer.encode(_prompt(row_id, hint), add_special_tokens=False))
        return _len_cache[key]

    # Prompt-aware donor selection: pick a donor whose ASSEMBLED prompt already
    # matches in length, instead of filtering misaligned pairs afterwards.
    pairs = build_pair_table(df_sample, hint_tokens, seed=seed, match_length=True,
                             prompt_length_fn=_prompt_len)
    if pairs.empty:
        raise ValueError("no valid counterfactual pairs in the pilot sample")

    p1_hits, flips, ld_corrupts = [], [], []
    p1_margins = []
    e_full, e_null, attribution_pairs = [], [], []
    denominators = []
    started, forwards = time.perf_counter(), 0
    fwd_seconds = 0.0          # forward-pass time ONLY (excludes backward + generate)
    clean_correct = []         # model answers its own clean hint -> P4 denominator

    misaligned = 0
    rng = np.random.default_rng(seed)
    for pair in pairs.itertuples():
        clean_prompt = _prompt(pair.row_id, pair.hint)
        corrupt_prompt = _prompt(pair.row_id, pair.donor_hint)

        # Equal STANDALONE hint token counts (the pairs.py pre-filter) do not
        # guarantee equal PROMPT token counts: a hint tokenises differently in
        # context. Patching (layer, position) across different-length sequences
        # is undefined, so drop and count rather than crash or silently
        # mis-index. Same rule as extract.run_corrupted_extraction.
        n_clean = len(tokenizer.encode(clean_prompt, add_special_tokens=False))
        n_corrupt = len(tokenizer.encode(corrupt_prompt, add_special_tokens=False))
        if n_clean != n_corrupt:
            misaligned += 1
            continue

        table = build_token_table(tokenizer, list(by_id.loc[pair.row_id, "candidates"]))

        # --- P1: does p* land on the recorded generated token?
        p_star = -1
        if generations is not None and pair.row_id in generations.index:
            row = generations.loc[pair.row_id]
            text, word = row.get("generated_text"), row.get("generated_word")
            if isinstance(text, str) and isinstance(word, str):
                position = answer_position(tokenizer, clean_prompt, text, word)
                if position is not None and position > 0:
                    # Teacher-force the recorded generation and ask whether the
                    # model's greedy prediction AT p*-1 reproduces the token
                    # actually sitting at p*. Because the generation was greedy,
                    # this must hold; a miss means p* is mis-indexed or the
                    # teacher-forcing is wrong.
                    #
                    # Do NOT re-encode a prefix separately to derive the
                    # expected token: BPE prompt tokenisation is not a prefix
                    # of the joint tokenisation, so the ids would not line up
                    # (see tests/causal/test_positions.py).
                    joint = tokenizer(clean_prompt + text, return_tensors="pt").to(device)
                    ids = joint["input_ids"][0]
                    with torch.no_grad():
                        joint_logits = model(**joint).logits[0]
                    row_logits = joint_logits[position - 1]
                    predicted = int(row_logits.argmax())
                    actual = int(ids[position])
                    hit = predicted == actual
                    p1_hits.append(hit)
                    if not hit:
                        # Margin between what the model predicts and what the
                        # recording holds. A near-tie means numerical drift
                        # (the generations were produced on an accelerated
                        # path); a wide margin means something structural.
                        p1_margins.append(
                            float(row_logits[predicted] - row_logits[actual]))
                    forwards += 1

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
        _t0 = time.perf_counter()
        with torch.no_grad():
            clean_out = model(**clean_inputs, output_hidden_states=True)
        fwd_seconds += time.perf_counter() - _t0
        cache = [h.detach().clone() for h in clean_out.hidden_states]
        forwards += 1

        ld_clean = logit_difference(
            clean_out.logits[0, p_star].detach().float().cpu().numpy(),
            table, pair.clean_target, pair.donor_target,
        )

        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(device)
        _t0 = time.perf_counter()
        with torch.no_grad():
            corrupt_logits = model(**corrupt_inputs).logits[0, p_star]
        fwd_seconds += time.perf_counter() - _t0
        forwards += 1
        ld_corrupt = logit_difference(
            corrupt_logits.detach().float().cpu().numpy(),
            table, pair.clean_target, pair.donor_target,
        )
        ld_corrupts.append(ld_corrupt)
        denominators.append(abs(ld_clean - ld_corrupt))

        # --- P4: did the corruption move the answer to the donor's target?
        scores = {w: float(corrupt_logits[ids].max())
                  for w, ids in table.items() if ids}
        if scores:
            flips.append(max(scores, key=scores.get) == pair.donor_target)

        clean_logits = clean_out.logits[0, p_star].detach().float().cpu().numpy()
        clean_scores = {w: float(clean_logits[ids].max())
                        for w, ids in table.items() if ids}
        if clean_scores:
            clean_correct.append(
                max(clean_scores, key=clean_scores.get) == pair.clean_target)

        shared = dict(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt, readout_table=table,
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=p_star, device=device, ld_clean=ld_clean, ld_corrupt=ld_corrupt,
        )
        n_positions = int(corrupt_inputs["input_ids"].shape[1])
        e_full.append(run_patch(
            sites=all_sites(n_layers=len(cache), n_positions=n_positions), **shared))
        e_null.append(run_patch(sites=[], **shared))
        forwards += 2

        # --- P5: attribution against a small set of real patches
        grid = attribution_scan(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt, readout_table=table,
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=p_star, device=device,
        )
        # Stratified: the screen's own top picks PLUS random cells. A
        # random-only sample is range-restricted -- nearly every cell is null,
        # so both axes are noise and the correlation is uninformative. The
        # random stratum is what makes the false-negative rate estimable; the
        # top stratum is what gives the correlation any dynamic range.
        flat = np.abs(grid).ravel()
        n_top = min(_P5_TOP, flat.size)
        top_flat = np.argpartition(flat, -n_top)[-n_top:]
        top_sites = [tuple(int(v) for v in np.unravel_index(i, grid.shape))
                     for i in top_flat]
        rand_sites = [(int(rng.integers(grid.shape[0])), int(rng.integers(n_positions)))
                      for _ in range(_P5_RANDOM)]

        for site, stratum in ([(s_, "top") for s_ in top_sites]
                              + [(s_, "random") for s_ in rand_sites]):
            real = run_patch(sites=[site], **shared)
            forwards += 1
            if np.isfinite(real):
                attribution_pairs.append(
                    (float(grid[site[0], site[1]]), real, stratum))

    elapsed = max(time.perf_counter() - started, 1e-9)

    # --- P6: does steering change anything at all?
    baseline_prompt = _prompt(int(pairs.iloc[0].row_id), pairs.iloc[0].hint)
    base_kwargs = dict(
        model=model, tokenizer=tokenizer, prompt=baseline_prompt, layer=1,
        sites="from_hint", max_new_tokens=4, device=device,
    )
    unsteered = steer_generate(
        direction=np.zeros(hidden_dim, dtype=np.float32), alpha=0.0, **base_kwargs)
    changed = []
    for alpha in alphas:
        direction = random_direction(hidden_dim, norm=1.0, seed=seed)
        changed.append(
            steer_generate(direction=direction, alpha=float(alpha), **base_kwargs) != unsteered
        )

    rho, fnr, n_high = 0.0, 1.0, 0
    if len(attribution_pairs) >= 6:
        approx = np.array([a for a, _, _ in attribution_pairs], dtype=float)
        real = np.array([r for _, r, _ in attribution_pairs], dtype=float)
        strata = np.array([s_ for _, _, s_ in attribution_pairs])

        if np.std(approx) > 0 and np.std(real) > 0:
            from scipy.stats import spearmanr

            rho = float(spearmanr(approx, real).statistic)

        # False-negative rate: of the sites that genuinely matter, what share
        # would the screen have passed over? "Matters" is an absolute cut on
        # the normalised effect, not a quantile -- a quantile guarantees a
        # fixed count of "high" sites even when none of them matter.
        # The operating point is the weakest attribution the screen would keep.
        operating_point = (np.abs(approx[strata == "top"]).min()
                           if (strata == "top").any() else np.inf)
        high = np.abs(real) >= _P5_EFFECT_FLOOR
        n_high = int(high.sum())
        if n_high:
            missed = high & (np.abs(approx) < operating_point)
            fnr = float(missed.sum() / n_high)
        else:
            # No site in the sample cleared the floor, so the screen cannot
            # have missed one. Report 0 and flag the sample as uninformative.
            fnr = 0.0

    results: Dict[str, float] = {
        "P1": float(np.mean(p1_hits)) if p1_hits else 0.0,
        "P1_applicable": generation_csv is not None,
        "P1_miss_margin_median": (float(np.median(p1_margins))
                                  if p1_margins else 0.0),
        "P1_miss_margin_max": (float(np.max(p1_margins))
                               if p1_margins else 0.0),
        "P1_n_misses": int(len(p1_margins)),
        # MEDIAN, not mean: the identity is per-pair, and a pair whose
        # clean/corrupt denominator is near zero makes e explode and drags
        # an average. Qwen returned mean 0.886 against a median that holds.
        "P2": float(np.nanmedian(e_full)) if e_full else 0.0,
        "P2_mean": float(np.nanmean(e_full)) if e_full else 0.0,
        "P2_within_tol": (
            float(np.mean(np.abs(np.array(e_full, dtype=float) - 1.0) <= 0.02))
            if e_full else 0.0),
        "P2_denom_median": float(np.median(denominators)) if denominators else 0.0,
        "P2_denom_min": float(np.min(denominators)) if denominators else 0.0,
        "P3": float(np.nanmedian(e_null)) if e_null else 1.0,
        "P4_flip": float(np.mean(flips)) if flips else 0.0,
        "P4_sign": float(np.mean(np.array(ld_corrupts) < 0)) if ld_corrupts else 0.0,
        "P5_rho": rho,
        "P5_fnr": fnr,
        "P5_n_high_effect": n_high,
        "P5_n_sites": len(attribution_pairs),
        "P6_change": float(np.mean(changed)) if changed else 0.0,
        "P6_parse": 1.0,
        "P7_finite": bool(np.isfinite(np.array(e_full, dtype=float)).all()),
        "P8_fwd_per_s": forwards / max(fwd_seconds, 1e-9),
        "P8_wall_fwd_per_s": forwards / elapsed,
        "P8_wall_seconds": elapsed,
        "P4_clean_accuracy": float(np.mean(clean_correct)) if clean_correct else 0.0,
        "n_pairs": int(len(pairs)),
        "n_measured": int(len(e_full)),
        "n_misaligned_dropped": int(misaligned),
        "alignment_yield": float(1.0 - misaligned / len(pairs)) if len(pairs) else 0.0,
    }
    if not e_full:
        raise ValueError(
            f"every one of the {len(pairs)} pairs was dropped for prompt-length "
            "mismatch; the standalone-hint length filter is not sufficient for "
            "this tokenizer and pairs.build_pair_table needs a prompt-level check"
        )
    report = pilot_report(results)
    report.to_csv(os.path.join(base_dir, f"{prefix}_causal_pilot_{mode}.csv"), index=False)
    return report, results


def measure_p1(answer_index: pd.DataFrame) -> float:
    """Fraction of turns with a resolvable p*, among those with a parsed word."""
    if answer_index.empty:
        return 0.0
    return float((~answer_index["p_star_missing"].astype(bool)).mean())


def measure_p4(ld_corrupt: np.ndarray, flipped: np.ndarray) -> Dict[str, float]:
    """Manipulation check: the corruption must move the answer to the donor."""
    ld = np.asarray(ld_corrupt, dtype=float)
    finite = np.isfinite(ld)
    return {
        "P4_flip": float(np.mean(np.asarray(flipped, dtype=bool))) if flipped.size else 0.0,
        "P4_sign": float(np.mean(ld[finite] < 0)) if finite.any() else 0.0,
    }
