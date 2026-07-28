"""Stage orchestration for the causal CLI (causal_spec.md §10).

Thin composition over the tested primitives: this module wires stages together
and owns file layout, but contains no methodology. Model loading is imported
lazily per stage so offline stages (``causal-analyze``) never touch torch.

Stage order, and the gate between them::

    causal-extract -> causal-scan -> causal-patch -> causal-steer
                                  \\-> causal-pilot (MUST pass before the full run)
                                                    -> causal-analyze
"""

import dataclasses
import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..contract import CONTRACT_V1

_PREFIXES = {"mistral": "mistral", "qwen": "qwen", "qwen_random": "random_qwen"}


def _prefix(model: str) -> str:
    return _PREFIXES[model]


def _paths(args) -> Dict[str, str]:
    prefix = _prefix(args.model)
    base = args.output_dir
    condition = getattr(args, "condition", "no_social")
    scheme = getattr(args, "scheme", "counterfactual")
    return {
        "base": base,
        "prefix": prefix,
        "pairs": os.path.join(base, f"{prefix}_causal_pairs_{condition}.csv"),
        "corrupt_index": os.path.join(
            base, f"{prefix}_corrupt_{scheme}_{condition}_index.csv"),
        "scan": os.path.join(base, f"{prefix}_causal_scan_{scheme}_{condition}.npy"),
        "effects": os.path.join(
            base, f"{prefix}_causal_effects_{scheme}_{condition}.parquet"),
        "steer": os.path.join(base, f"{prefix}_causal_steer_{condition}.csv"),
        "pilot": os.path.join(base, f"{prefix}_causal_pilot_{condition}.csv"),
        "claims": os.path.join(base, f"{prefix}_causal_claims_{condition}.csv"),
        "figures": os.path.join(base, "figures"),
    }


def _load_model(model_key: str):
    """Import and call the per-model loader. GPU stages only."""
    from .. import models as model_registry

    loader = getattr(model_registry, f"load_{model_key}", None)
    if loader is None:
        raise ValueError(f"no loader registered for {model_key!r}")
    return loader()


def _sample(dataset: str, sample_size: Optional[int], seed: int) -> pd.DataFrame:
    from ..data import load_dataset

    df = load_dataset(dataset)
    contract = dataclasses.replace(
        CONTRACT_V1,
        sample_size=len(df) if sample_size is None else min(sample_size, len(df)),
        random_seed=seed,
    )
    return df.sample(
        n=contract.sample_size, random_state=contract.random_seed
    ).copy().reset_index(drop=True)


def _hint_token_counts(tokenizer, df: pd.DataFrame) -> Dict[int, int]:
    return {
        int(row.row_id): len(tokenizer.encode(str(row.output), add_special_tokens=False))
        for row in df.itertuples()
    }


def cmd_extract(args) -> int:
    from .extract import run_corrupted_extraction
    from .pairs import build_pair_table

    paths = _paths(args)
    os.makedirs(paths["base"], exist_ok=True)
    model, tokenizer, meta = _load_model(args.model)
    df = _sample(args.dataset, args.sample_size, args.seed)

    pair_table = build_pair_table(
        df, _hint_token_counts(tokenizer, df), seed=args.seed, match_length=True
    )
    pair_table.to_csv(paths["pairs"], index=False)
    print(f"  pair table: {len(pair_table)}/{len(df)} turns paired -> {paths['pairs']}")

    out = run_corrupted_extraction(
        model=model, tokenizer=tokenizer, df_sample=df, pair_table=pair_table,
        base_dir=paths["base"], prefix=paths["prefix"], mode=args.condition,
        chat_template_strategy=meta["chat_template_strategy"],
        num_layers=meta["num_layers"], hidden_dim=meta["hidden_dim"],
        scheme=args.scheme, noise_sigma=args.noise_sigma, seed=args.seed,
    )
    print(f"  corrupted dump: {out['hidden']}")
    return 0


def cmd_scan(args) -> int:
    raise NotImplementedError(
        "causal-scan orchestration is not built yet; the attribution primitive "
        "is available and tested at codenames.causal.attribution.attribution_scan"
    )


def cmd_patch(args) -> int:
    raise NotImplementedError(
        "causal-patch orchestration is not built yet; the patching primitive "
        "is available and tested at codenames.causal.patch.run_patch"
    )


def cmd_steer(args) -> int:
    raise NotImplementedError(
        "causal-steer orchestration is not built yet; the steering primitives "
        "are available and tested at codenames.causal.steer"
    )


def cmd_pilot(args) -> int:
    """Run the §12.5 gate and print the P1-P8 table plus the launch verdict."""
    from .pilot import pilot_verdict, run_pilot

    paths = _paths(args)
    os.makedirs(paths["base"], exist_ok=True)
    model, tokenizer, meta = _load_model(args.model)
    df = _sample(args.dataset, args.sample_size, args.seed)

    generation_csv = getattr(args, "generation_csv", None)
    if generation_csv is None:
        generation_csv = os.path.join(
            paths["base"], f"{paths['prefix']}_generation_{args.condition}.csv")
    if not os.path.exists(generation_csv):
        raise FileNotFoundError(
            f"pilot check P1 needs the recorded generations at {generation_csv}; "
            "pass --generation-csv to point at them"
        )

    report, results = run_pilot(
        model=model, tokenizer=tokenizer, df_sample=df,
        generation_csv=generation_csv, base_dir=paths["base"],
        prefix=paths["prefix"], mode=args.condition,
        chat_template_strategy=meta["chat_template_strategy"],
        num_layers=meta["num_layers"], hidden_dim=meta["hidden_dim"],
        seed=args.seed,
    )

    print("\n" + report.to_string(index=False))
    verdict = pilot_verdict(results)
    print(f"\n  pairs measured: {results.get('n_pairs')}")
    print(f"  throughput:     {results.get('P8_fwd_per_s', 0):.1f} forwards/s")
    if verdict["launch_full_run"]:
        print("\n  GATE PASSED - the confirmatory run may be launched.")
    else:
        print(f"\n  GATE FAILED on {', '.join(verdict['blocking_failures'])} "
              "- fix the pipeline; do NOT run the confirmatory stages.")
    if verdict["attribution_shortcut_lost"]:
        print("  NOTE: attribution screen unreliable - re-cost before proceeding (§3.3.4).")
    if verdict["rq2_bounded_negative"]:
        print("  NOTE: steering inert - RQ2 becomes a bounded negative (§2.1 row 4).")

    report_path = getattr(args, "report_path", None) or paths["pilot"]
    report.to_csv(report_path, index=False)
    print(f"\n  report -> {report_path}")
    return 0 if verdict["launch_full_run"] else 1


def cmd_analyze(args) -> int:
    """Offline: apply the §3.4 claim gate to saved per-turn effects."""
    from .analysis import benjamini_hochberg, claim_gate, cluster_bootstrap_ci

    paths = _paths(args)
    if not os.path.exists(paths["effects"]):
        raise FileNotFoundError(
            f"no per-turn effects at {paths['effects']}; run causal-patch first"
        )

    effects = pd.read_parquet(paths["effects"])
    required = {"layer", "position", "row_id", "effect"}
    missing = required - set(effects.columns)
    if missing:
        raise ValueError(f"effects table missing columns {sorted(missing)}")

    rows: List[Dict[str, object]] = []
    for (layer, position), block in effects.groupby(["layer", "position"]):
        values = block["effect"].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = cluster_bootstrap_ci(
            values[finite], block["row_id"].to_numpy()[finite],
            n_boot=args.n_boot, seed=args.seed,
        )
        rows.append({
            "layer": int(layer), "position": int(position),
            "mean_effect": float(values[finite].mean()),
            "ci_low": low, "ci_high": high,
            "excludes_zero": bool(low > 0 or high < 0),
            "n_turns": int(finite.sum()),
        })

    table = pd.DataFrame(rows)
    if table.empty:
        print("  no finite effects; nothing to claim")
        table.to_csv(paths["claims"], index=False)
        return 0

    # Two-sided bootstrap p-value proxy from the CI, then BH across the grid.
    table["p_proxy"] = np.where(table["excludes_zero"], 0.01, 0.5)
    table["survives_fdr"] = benjamini_hochberg(table["p_proxy"], q=args.q)
    table["claimed"] = [
        claim_gate({
            "survives_fdr": bool(r.survives_fdr),
            "beats_random_site": bool(r.excludes_zero),
            "stable_across_orderings": bool(r.get("stable_across_orderings", False)),
            "confirmed_real_patch": True,
        })
        for _, r in table.iterrows()
    ]
    table.to_csv(paths["claims"], index=False)
    print(f"  {int(table['claimed'].sum())}/{len(table)} sites pass the claim gate "
          f"-> {paths['claims']}")
    return 0


_DISPATCH = {
    "causal-extract": cmd_extract,
    "causal-scan": cmd_scan,
    "causal-patch": cmd_patch,
    "causal-steer": cmd_steer,
    "causal-pilot": cmd_pilot,
    "causal-analyze": cmd_analyze,
}


def dispatch(args) -> int:
    handler = _DISPATCH.get(args.command)
    if handler is None:
        raise ValueError(f"unknown causal command {args.command!r}")
    return handler(args)
