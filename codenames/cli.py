"""Command-line interface for the experiment.

The CLI is the **batch-run** interface; the notebooks are the **interactive
verification** interface. Both call the same underlying package functions;
this module is a thin orchestration layer.

Subcommands:
- ``run``:        full experiment for one model
- ``preflight``:  random-init pre-flight diagnostic
- ``validate``:   bit-identity check against existing outputs
- ``sanity``:     re-run SC functions on already-extracted results
- ``compare``:    reference path vs accelerated path, with per-column deltas
- ``visualize``:  per-board heatmap / projection figures
- ``aggregate``:  cross-model metric tables + publication figure set
- ``lens-extract``: generating-position hidden-state dump (GPU; docs/specs/lens_spec.md)
- ``lens-tune``:    tuned-lens translator training (GPU)
- ``lens-apply``:   raw/tuned candidate scoring from the dump (offline)
- ``lens-analyze``: pre-registered trajectory analysis + overlay figures
- ``job-submit``: enqueue a GPU job for the Colab runner
- ``job-status``: runner health plus recent job states
- ``job-logs``:   print a job's captured output
- ``job-cancel``: request cancellation of a running job
- ``job-sync``:   download run artifacts from Drive into output/
- ``job-runner``: the agent poll loop itself (runs inside Colab)

Output of each subcommand is identical to running the corresponding cells of
the model notebook in order; no extra logging, no progress suppression. All
arg parsing uses ``argparse`` from the standard library — no third-party
CLI framework.
"""

import argparse
import importlib
import os
import sys
import tempfile
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd


MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    "mistral":     ("codenames.models.mistral",     "load_mistral_instruct"),
    "qwen":        ("codenames.models.qwen",        "load_qwen_instruct"),
    "qwen_random": ("codenames.models.qwen_random", "load_qwen_random"),
    "bert":        ("codenames.models.bert",        "load_bert_base"),
    "bert_random": ("codenames.models.bert_random", "load_bert_random"),
    "t5":          ("codenames.models.t5",          "load_t5_encoder"),
    "modernbert":  ("codenames.models.modernbert",  "load_modernbert"),
}


def _resolve_loader(model_name: str) -> Callable:
    """Lazy-import the chosen model loader.

    Loading is deferred until the model is actually invoked so that CLI
    startup doesn't pull in all seven model libraries.
    """
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(
            f"Unknown --model '{model_name}'. "
            f"Choose one of: {', '.join(MODEL_REGISTRY)}."
        )
    module_path, attr = MODEL_REGISTRY[model_name]
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# ---------------------------------------------------------------------------
# Subcommand parsers
# ---------------------------------------------------------------------------

def _make_run_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "run",
        help="Run the full experiment for one model.",
        description=(
            "Run the full extraction pipeline (both conditions) plus all "
            "seven sanity checks. Produces the same outputs as running the "
            "corresponding notebook end-to-end."
        ),
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--output-dir", required=True,
                   help="Base output directory (BASE_DIR) for the final result files only.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory for intermediate checkpoint/manifest files and the "
                        "canonical reuse cache. Keeps --output-dir limited to the final "
                        "outputs. Defaults to a 'checkpoints' subfolder of --output-dir.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Override CONTRACT_V1.sample_size (default N=2000) for this run.")
    p.add_argument("--full", action="store_true",
                   help="Process the ENTIRE dataset (all 7704 rows), overriding "
                        "--sample-size and the contract's N=2000 default.")
    p.add_argument("--conditions", default="no_social,with_social",
                   help="Comma-separated conditions to run (default: both).")
    p.add_argument("--skip-sanity-checks", action="store_true",
                   help="Skip SC1-SC7 after extraction.")
    p.add_argument("--no-generation", action="store_true",
                   help="Disable generation phase for causal models (no effect on encoders).")
    p.add_argument("--resume", action="store_true",
                   help="Resume an interrupted run in --output-dir: skip boards already "
                        "committed to checkpoints (per the manifest) and reuse a completed "
                        "condition's outputs. Byte-identical to an uninterrupted run. "
                        "Without this flag, stale checkpoints in --output-dir are wiped.")
    p.add_argument("--reuse-canonical", action="store_true",
                   help="Reuse per-board canonical (permutation_id=0) results from a "
                        "persistent row_id cache in --output-dir, and write newly-computed "
                        "canonicals back to it. Lets a later, larger run skip the canonical "
                        "forward pass for boards it shares with an earlier run (shuffles "
                        "always recomputed). Byte-identical to a non-reusing run. Ignored "
                        "when --batch-size > 1. Pass it on both runs to populate then reuse.")
    # --- Acceleration flags (default off; characterise tolerance with `compare` first) ---
    p.add_argument("--vectorize-anisotropy", action="store_true",
                   help="Use vectorized M @ M.T for all-pairs anisotropy. ~1e-6 drift on aniso aggregates.")
    p.add_argument("--flash-attn", action="store_true",
                   help="Load causal models (Mistral, Qwen) with attn_implementation='flash_attention_2'.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Boards per forward pass (default: 1 = reference). Higher = faster, slight fp16 drift.")
    return p


def _make_doctor_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "doctor",
        help="Verify installed dependencies match the pinned, reproducible set.",
        description=(
            "Read-only dependency check: confirms the installed package "
            "versions match the pins in pyproject.toml, that the installed "
            "transformers exposes each model's classes, and reports Python / "
            "CUDA / flash_attn status. Downloads no weights and changes "
            "nothing. Exits non-zero on any hard failure."
        ),
    )
    p.add_argument("--model", default=None, choices=list(MODEL_REGISTRY),
                   help="Only check this model's transformers classes (default: all).")
    p.add_argument("--allow-drift", action="store_true",
                   help="Treat a version mismatch against the pin as a warning, not a failure.")
    p.add_argument("--require-cuda", action="store_true",
                   help="Fail if a CUDA device is not available.")
    return p


def _make_preflight_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "preflight",
        help="Run the pre-flight diagnostic (random-init models only).",
        description=(
            "Runs preflight_random_init on 5 boards: NaN/Inf detection, "
            "hidden-state norm growth, L0 anisotropy. For non-random-init "
            "models this command prints a message and exits cleanly."
        ),
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    return p


def _make_validate_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "validate",
        help="Bit-identity check against an existing run.",
        description=(
            "Runs the pipeline on a small subsample and compares the outputs "
            "against the corresponding rows of an existing run directory."
        ),
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--against", required=True,
                   help="Path to existing output directory to compare against.")
    p.add_argument("-n", "--n", type=int, default=50,
                   help="Number of boards to validate on (default: 50).")
    p.add_argument("--tolerance", type=float, default=1e-6,
                   help="Numeric tolerance for CSV comparisons (default: 1e-6).")
    return p


def _make_compare_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "compare",
        help="Run reference path vs accelerated path and report deltas.",
        description=(
            "Runs the same extraction twice on a small subsample: once with "
            "Acceleration defaults (reference path) and once with the "
            "requested acceleration flags. Prints per-column max-abs / "
            "mean-abs delta, used to quantify the tolerance introduced by "
            "each optimization for the thesis appendix."
        ),
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("-n", "--n", type=int, default=50,
                   help="Number of boards to compare on (default: 50).")
    p.add_argument("--vectorize-anisotropy", action="store_true",
                   help="Enable vectorized all-pairs anisotropy.")
    p.add_argument("--flash-attn", action="store_true",
                   help="Enable Flash Attention 2 (Mistral/Qwen only).")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Boards per forward pass (default: 1 = reference).")
    return p


def _make_sanity_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "sanity",
        help="Re-run SC functions on already-extracted results.",
        description=(
            "Loads previously-extracted results from --results-dir and runs "
            "the specified sanity checks against them. Useful for re-running "
            "SC output formatting without re-running the (expensive) extraction."
        ),
    )
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    p.add_argument("--results-dir", required=True,
                   help="Directory containing the previously-extracted output files.")
    p.add_argument("--checks", default="sc1,sc2,sc3,sc4,sc5,sc6,sc7",
                   help="Comma-separated subset of sanity checks to run.")
    return p


def _make_visualize_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "visualize",
        help="Render heatmap + 2D-projection figures from extracted outputs.",
        description=(
            "Local, post-hoc visualization. Reads {prefix}_vectors_subsample_* "
            "files from --output-dir/<model>/, samples boards, and writes "
            "publication-formatted figures (cosine heatmaps + cosine-aware UMAP/"
            "t-SNE/PCA projections, validated by trustworthiness/continuity/"
            "Shepard metrics) under --viz-dir. Requires the optional [viz] "
            "dependency group: pip install -e \".[viz]\"."
        ),
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", choices=list(MODEL_REGISTRY),
                   help="Single model to visualize (looks under <output-dir>/<model>/).")
    g.add_argument("--all", action="store_true",
                   help="Visualize every model directory discovered under --output-dir.")
    p.add_argument("--output-dir", default="output",
                   help="Root directory holding per-model output folders (default: output).")
    p.add_argument("--viz-dir", default="visualization",
                   help="Directory to write figures into (default: visualization).")
    p.add_argument("--n-boards", type=int, default=5,
                   help="Number of boards to sample per model/condition (default: 5).")
    p.add_argument("--pooling", default="mean", choices=["mean", "max_norm"],
                   help="Span-pooling method to visualize (default: mean).")
    p.add_argument("--seed", type=int, default=2026,
                   help="Random seed for board sampling and reducers (default: 2026).")
    p.add_argument("--layers", default=None,
                   help="Comma-separated explicit layer indices; default picks ~6 by depth.")
    p.add_argument("--boards", default=None,
                   help="Comma-separated explicit board row_ids to visualize. Pins the "
                        "SAME boards across models for direct comparison. Overrides "
                        "--n-boards. When omitted, boards are sampled from the "
                        "intersection available across ALL discovered models (for both "
                        "single-model and --all runs), so the selection is identical "
                        "across models by default.")
    p.add_argument("--heatmaps-only", action="store_true",
                   help="Render only the heatmaps, reusing the existing UMAP / "
                        "dr_quality files (the slow projection fits are skipped). Use "
                        "when only the heatmap style changed.")
    return p


_AGGREGATE_STEPS = ("tables", "concordance", "boards", "trust", "figures", "examples")


def _make_aggregate_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "aggregate",
        help="Build cross-model metric tables and the publication figure set.",
        description=(
            "Local, post-hoc aggregation across all models' outputs. Builds "
            "the unified analysis tables under --analysis-dir (behavioral "
            "summary, paired social-preamble effects with bootstrap CIs, "
            "generation/geometry concordance per layer, UMAP trustworthiness "
            "sweep) and renders the publication figures under --figures-dir. "
            "The per-layer geometry tables (margins, confound, semantic "
            "ratio) written by each run are read from --analysis-dir. The "
            "trust and figures steps require the optional [viz] group: "
            'pip install -e ".[viz]".'
        ),
    )
    p.add_argument("--output-dir", default="output",
                   help="Root directory holding per-model output folders (default: output).")
    p.add_argument("--analysis-dir", default=None,
                   help="Where the analysis tables live / are written "
                        "(default: <output-dir>/analysis).")
    p.add_argument("--figures-dir", default=os.path.join("visualization", "aggregate"),
                   help="Where figures are written (default: visualization/aggregate).")
    p.add_argument("--steps", default=",".join(_AGGREGATE_STEPS),
                   help="Comma-separated subset of steps to run, of: "
                        f"{', '.join(_AGGREGATE_STEPS)} (default: all). "
                        "'concordance' scans the heavy metrics parquets; "
                        "'boards' recomputes per-board peak depths from the vector "
                        "subsample; 'trust' fits one UMAP per (board, layer) and is "
                        "the slow step; 'examples' renders the per-board paper-style "
                        "UMAP+heatmap library under <figures-dir>/{umaps,heatmaps}/.")
    p.add_argument("--pooling", default="mean", choices=["mean", "max_norm"],
                   help="Pooling method for the trust sweep and figures (default: mean). "
                        "Tables always cover both poolings.")
    p.add_argument("--seed", type=int, default=2026,
                   help="Seed for the bootstrap and the UMAP reducer (default: 2026).")
    p.add_argument("--n-boot", type=int, default=5000,
                   help="Bootstrap resamples for the social-effect CIs (default: 5000).")
    p.add_argument("--trust-layers", type=int, default=6,
                   help="Representative layers per model for the trust sweep (default: 6).")
    p.add_argument("--trust-boards", type=int, default=None,
                   help="Cap on subsample boards per model/condition in the trust sweep "
                        "(default: all, typically 100).")
    p.add_argument("--trust-k", type=int, default=5,
                   help="Neighbourhood size k for trustworthiness/continuity (default: 5).")
    p.add_argument("--example-boards", type=int, default=10,
                   help="Boards per model for the 'examples' library (the cross-model "
                        "shared sample; default: 10).")
    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _cmd_doctor(args: argparse.Namespace) -> int:
    from . import depcheck

    models = [args.model] if args.model else None
    ok = depcheck.run(
        models=models,
        allow_drift=args.allow_drift,
        require_cuda=args.require_cuda,
    )
    return 0 if ok else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from .contract import Acceleration, CONTRACT_V1, Contract
    from .data import load_dataset, sample_turns
    from .generation import generate_response
    from .loop import run_extraction
    from .persistence import print_output_summary
    from . import sanity as sc

    import dataclasses

    loader = _resolve_loader(args.model)

    # Load the dataset first so --full can resolve against the real row count.
    df = load_dataset(args.dataset)

    # Resolve how many boards to run: --full > --sample-size > contract default.
    if args.full:
        requested_n = len(df)
        print(f"Run size: FULL dataset ({requested_n} boards)")
    elif args.sample_size is not None:
        requested_n = args.sample_size
        print(f"Run size: {requested_n} boards (--sample-size)")
    else:
        requested_n = CONTRACT_V1.sample_size
        print(f"Run size: {requested_n} boards (contract default)")

    # Reuse the frozen contract unless the size differs from its N=2000 baseline.
    if requested_n != CONTRACT_V1.sample_size:
        contract = dataclasses.replace(CONTRACT_V1, sample_size=requested_n)
    else:
        contract = CONTRACT_V1

    acceleration = Acceleration(
        vectorize_anisotropy=args.vectorize_anisotropy,
        flash_attention_for_causal=args.flash_attn,
        batch_size=args.batch_size,
    )

    # Load the model with FA2 if requested and supported.
    if acceleration.flash_attention_for_causal and args.model in ("mistral", "qwen"):
        print(f"Loading model with attn_implementation='flash_attention_2'")
        model, tokenizer, meta = loader(attn_implementation="flash_attention_2")
    else:
        model, tokenizer, meta = loader()

    df_sample = sample_turns(df, n=contract.sample_size, seed=contract.random_seed)

    has_generation = meta["supports_generation"] and not args.no_generation
    generation_fn = generate_response if has_generation else None

    results = run_extraction(
        model=model,
        tokenizer=tokenizer,
        df=df_sample,
        base_dir=args.output_dir,
        prefix=meta["prefix"],
        contract=contract,
        chat_template_strategy=meta["chat_template_strategy"],
        forward_hidden_states_mode=meta["forward_hidden_states_mode"],
        use_truncation=meta["use_truncation"],
        num_layers=meta["num_layers"],
        hidden_dim=meta["hidden_dim"],
        device=meta["device"],
        has_generation=has_generation,
        generation_fn=generation_fn,
        acceleration=acceleration,
        resume=args.resume,
        reuse_canonical=args.reuse_canonical,
        checkpoint_dir=args.checkpoint_dir,
    )

    if not args.skip_sanity_checks:
        sc.sc1_prompt_structure(df_sample, tokenizer, meta["chat_template_strategy"])
        sc.sc2_span_coverage(results)
        sc.sc3_anisotropy(results, num_layers=meta["num_layers"])
        sc.sc4_behavioral_accuracy(
            results,
            pooling_methods=contract.pooling_methods,
            has_generation=has_generation,
        )
        sc.sc5_layer_margin_curve(
            results,
            base_dir=args.output_dir,
            prefix=meta["prefix"],
            num_layers=meta["num_layers"],
            pooling_methods=contract.pooling_methods,
        )
        sc.sc6_positional_confound(
            results,
            base_dir=args.output_dir,
            prefix=meta["prefix"],
            num_layers=meta["num_layers"],
        )
        sc.sc7_shuffle_decomposition(
            results,
            base_dir=args.output_dir,
            prefix=meta["prefix"],
            num_layers=meta["num_layers"],
            n_shuffles=contract.n_shuffles,
        )

    print_output_summary(
        base_dir=args.output_dir,
        prefix=meta["prefix"],
        contract=contract,
        has_generation=has_generation,
        pooling_methods=contract.pooling_methods,
    )
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    if args.model != "qwen_random":
        print(
            f"Pre-flight is only defined for random-init models. "
            f"--model {args.model} has no preflight; exiting cleanly."
        )
        return 0

    from .contract import CONTRACT_V1
    from .data import load_dataset, sample_turns
    from .diagnostics import preflight_random_init

    loader = _resolve_loader(args.model)
    model, tokenizer, meta = loader()
    df = load_dataset(args.dataset)
    df_sample = sample_turns(df, n=CONTRACT_V1.sample_size, seed=CONTRACT_V1.random_seed)

    preflight_random_init(
        df=df_sample,
        model=model,
        tokenizer=tokenizer,
        device=meta["device"],
        num_layers=meta["num_layers"],
        chat_template_strategy=meta["chat_template_strategy"],
        n_boards=5,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Run a small extraction and diff against an existing run.

    Compares ``{prefix}_general_{mode}.csv`` and ``{prefix}_metrics_{mode}.parquet``
    by row_id with the rows present in ``--against``.
    """
    from .contract import CONTRACT_V1, Contract
    from .data import load_dataset, sample_turns
    from .generation import generate_response
    from .loop import run_extraction

    loader = _resolve_loader(args.model)

    contract = Contract(
        sample_size=args.n,
        candidate_order=CONTRACT_V1.candidate_order,
        pooling_methods=CONTRACT_V1.pooling_methods,
        vector_subsample_size=CONTRACT_V1.vector_subsample_size,
        n_shuffles=CONTRACT_V1.n_shuffles,
        generation_max_tokens=CONTRACT_V1.generation_max_tokens,
        shard_boards=CONTRACT_V1.shard_boards,
        random_seed=CONTRACT_V1.random_seed,
        max_seq_len=CONTRACT_V1.max_seq_len,
    )

    model, tokenizer, meta = loader()
    df = load_dataset(args.dataset)
    df_sample = sample_turns(df, n=args.n, seed=CONTRACT_V1.random_seed)

    with tempfile.TemporaryDirectory(prefix="cnames_validate_") as tmpdir:
        results = run_extraction(
            model=model,
            tokenizer=tokenizer,
            df=df_sample,
            base_dir=tmpdir,
            prefix=meta["prefix"],
            contract=contract,
            chat_template_strategy=meta["chat_template_strategy"],
            forward_hidden_states_mode=meta["forward_hidden_states_mode"],
            use_truncation=meta["use_truncation"],
            num_layers=meta["num_layers"],
            hidden_dim=meta["hidden_dim"],
            device=meta["device"],
            has_generation=meta["supports_generation"],
            generation_fn=generate_response if meta["supports_generation"] else None,
        )

        row_ids = sorted(df_sample["row_id"].tolist())
        failures = []

        for mode_name in ["no_social", "with_social"]:
            # General CSV
            new_general = results[mode_name]["general_df"]
            old_general_path = os.path.join(
                args.against, f"{meta['prefix']}_general_{mode_name}.csv"
            )
            if not os.path.exists(old_general_path):
                failures.append(f"Missing reference file: {old_general_path}")
                continue
            old_general = pd.read_csv(old_general_path)
            old_general = old_general[old_general["row_id"].isin(row_ids)]
            for col in new_general.select_dtypes(include=[np.number]).columns:
                if col not in old_general.columns:
                    continue
                merged = new_general[["row_id", "permutation_id", col]].merge(
                    old_general[["row_id", "permutation_id", col]],
                    on=["row_id", "permutation_id"],
                    suffixes=("_new", "_old"),
                )
                if merged.empty:
                    continue
                diff = (merged[f"{col}_new"] - merged[f"{col}_old"]).abs()
                max_diff = float(diff.max(skipna=True))
                if max_diff > args.tolerance:
                    failures.append(
                        f"general[{mode_name}][{col}]: max diff {max_diff:.2e} "
                        f"(tolerance {args.tolerance:.0e})"
                    )

        if failures:
            print("\nValidation FAILED:")
            for line in failures:
                print(f"  - {line}")
            return 1

        print("\nValidation PASSED")
        return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    """Reference path vs accelerated path on a small subsample.

    When ``--flash-attn`` is requested for a causal model (mistral, qwen),
    the two passes use different model loads (eager vs FA2). To avoid
    holding two 7B models simultaneously, the reference model is freed
    before loading the fast model. The two passes write results to disk
    independently and are diffed at the end.

    Otherwise (vectorize_anisotropy and/or batch_size > 1 only), a single
    model is loaded and used for both passes.
    """
    import gc

    import torch

    from .comparison import _diff_general, _diff_metrics, _print_summary
    from .contract import ACCEL_REFERENCE, Acceleration, CONTRACT_V1, Contract
    from .comparison import compare_runs
    from .data import load_dataset, sample_turns
    from .generation import generate_response
    from .loop import run_extraction

    loader = _resolve_loader(args.model)

    contract = Contract(
        sample_size=args.n,
        candidate_order=CONTRACT_V1.candidate_order,
        pooling_methods=CONTRACT_V1.pooling_methods,
        vector_subsample_size=CONTRACT_V1.vector_subsample_size,
        n_shuffles=CONTRACT_V1.n_shuffles,
        generation_max_tokens=CONTRACT_V1.generation_max_tokens,
        shard_boards=CONTRACT_V1.shard_boards,
        random_seed=CONTRACT_V1.random_seed,
        max_seq_len=CONTRACT_V1.max_seq_len,
    )

    fast_accel = Acceleration(
        vectorize_anisotropy=args.vectorize_anisotropy,
        flash_attention_for_causal=args.flash_attn,
        batch_size=args.batch_size,
    )

    wants_fa2_reload = args.flash_attn and args.model in ("mistral", "qwen")

    df = load_dataset(args.dataset)
    df_sample = sample_turns(df, n=args.n, seed=CONTRACT_V1.random_seed)

    if not wants_fa2_reload:
        # Single-model path: vectorize_anisotropy and/or batch_size only.
        print(f"Loading model (single load for both passes): {args.model}")
        model, tokenizer, meta = loader()
        compare_runs(
            model_ref=model,
            tokenizer_ref=tokenizer,
            model_fast=model,
            tokenizer_fast=tokenizer,
            df=df_sample,
            prefix=meta["prefix"],
            contract=contract,
            chat_template_strategy=meta["chat_template_strategy"],
            forward_hidden_states_mode=meta["forward_hidden_states_mode"],
            use_truncation=meta["use_truncation"],
            num_layers=meta["num_layers"],
            hidden_dim=meta["hidden_dim"],
            device=meta["device"],
            has_generation=meta["supports_generation"],
            generation_fn=generate_response if meta["supports_generation"] else None,
            fast_acceleration=fast_accel,
        )
        return 0

    # --- FA2 path: two separate model loads, free between passes ---
    import os
    import tempfile

    tmp_base_dir = tempfile.mkdtemp(prefix="cnames_compare_fa2_")
    ref_dir = os.path.join(tmp_base_dir, "ref")
    fast_dir = os.path.join(tmp_base_dir, "fast")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(fast_dir, exist_ok=True)

    print(f"[1/2] Loading reference model (eager attention): {args.model}")
    model, tokenizer, meta = loader()
    print("[1/2] Running reference path on N={} boards".format(args.n))
    ref_results = run_extraction(
        model=model,
        tokenizer=tokenizer,
        df=df_sample,
        base_dir=ref_dir,
        prefix=meta["prefix"],
        contract=contract,
        chat_template_strategy=meta["chat_template_strategy"],
        forward_hidden_states_mode=meta["forward_hidden_states_mode"],
        use_truncation=meta["use_truncation"],
        num_layers=meta["num_layers"],
        hidden_dim=meta["hidden_dim"],
        device=meta["device"],
        has_generation=meta["supports_generation"],
        generation_fn=generate_response if meta["supports_generation"] else None,
        acceleration=ACCEL_REFERENCE,
    )
    print("[1/2] Reference pass complete. Freeing reference model before FA2 load.")
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[2/2] Loading fast model (flash_attention_2): {args.model}")
    model_fast, tokenizer_fast, meta_fast = loader(attn_implementation="flash_attention_2")
    print("[2/2] Running fast path on N={} boards".format(args.n))
    fast_results = run_extraction(
        model=model_fast,
        tokenizer=tokenizer_fast,
        df=df_sample,
        base_dir=fast_dir,
        prefix=meta_fast["prefix"],
        contract=contract,
        chat_template_strategy=meta_fast["chat_template_strategy"],
        forward_hidden_states_mode=meta_fast["forward_hidden_states_mode"],
        use_truncation=meta_fast["use_truncation"],
        num_layers=meta_fast["num_layers"],
        hidden_dim=meta_fast["hidden_dim"],
        device=meta_fast["device"],
        has_generation=meta_fast["supports_generation"],
        generation_fn=generate_response if meta_fast["supports_generation"] else None,
        acceleration=fast_accel,
    )

    general_report = _diff_general(ref_results, fast_results)
    metrics_report = _diff_metrics(ref_results, fast_results)
    _print_summary("general_df", general_report)
    _print_summary("metrics_df", metrics_report)
    return 0


def _cmd_sanity(args: argparse.Namespace) -> int:
    """Re-run SC functions on already-extracted output files."""
    from .contract import CONTRACT_V1
    from . import sanity as sc

    loader = _resolve_loader(args.model)
    requested = [c.strip().lower() for c in args.checks.split(",") if c.strip()]

    # SC1 needs a tokenizer; load only that (not the full model) when requested.
    needs_tokenizer = "sc1" in requested

    prefix_module_path, _ = MODEL_REGISTRY[args.model]
    module = importlib.import_module(prefix_module_path)
    # The prefix is documented in metadata, but we don't want to load the
    # full model just to read it. Map by --model name -> prefix here.
    PREFIX_BY_MODEL = {
        "mistral": "mistral",
        "qwen": "qwen",
        "qwen_random": "random_qwen",
        "bert": "bert",
        "bert_random": "random_bert",
        "t5": "t5",
        "modernbert": "modernbert",
    }
    CHAT_BY_MODEL = {
        "mistral": "mistral_inst",
        "qwen": "chatml",
        "qwen_random": "chatml",
        "bert": "raw",
        "bert_random": "raw",
        "t5": "raw",
        "modernbert": "raw",
    }
    HAS_GEN_BY_MODEL = {
        "mistral": True, "qwen": True, "qwen_random": False,
        "bert": False, "bert_random": False, "t5": False, "modernbert": False,
    }
    prefix = PREFIX_BY_MODEL[args.model]

    results: Dict[str, Dict] = {}
    for mode_name in ["no_social", "with_social"]:
        results[mode_name] = {}
        gpath = os.path.join(args.results_dir, f"{prefix}_general_{mode_name}.csv")
        mpath = os.path.join(args.results_dir, f"{prefix}_metrics_{mode_name}.parquet")
        results[mode_name]["general_df"] = pd.read_csv(gpath) if os.path.exists(gpath) else pd.DataFrame()
        results[mode_name]["metrics_df"] = pd.read_parquet(mpath) if os.path.exists(mpath) else pd.DataFrame()
        genpath = os.path.join(args.results_dir, f"{prefix}_generation_{mode_name}.csv")
        results[mode_name]["generation_df"] = (
            pd.read_csv(genpath) if os.path.exists(genpath) else pd.DataFrame()
        )

    # Heuristic for num_layers: read from a metrics row if available.
    num_layers = 0
    for mode_name in ["no_social", "with_social"]:
        mdf = results[mode_name]["metrics_df"]
        if len(mdf):
            num_layers = int(mdf["layer"].max())
            break

    tokenizer = None
    if needs_tokenizer:
        # Load only the tokenizer, not the full model.
        # All seven loaders construct a tokenizer; importing the module and
        # picking it off is faster than running the full loader, but the
        # simple thing is to just run the loader and discard the model.
        # Memory cost is acceptable for SC1's single forward-pass-less call.
        model, tokenizer, _meta = loader()
        del model

    for check in requested:
        if check == "sc1":
            if tokenizer is None:
                print("SC1 requires a tokenizer; could not be loaded. Skipping.")
                continue
            # SC1 needs df_sample; reconstruct from the canonical permutation
            # of the metrics dataframe.
            sample_rows = []
            mdf = results["no_social"]["metrics_df"]
            if len(mdf):
                first_row_id = int(mdf["row_id"].iloc[0])
                # Cannot reconstruct the full row_id-keyed dataframe from
                # metrics alone (we don't have targets/black/tan stored
                # separately). SC1 in --sanity mode is best run with the
                # original dataset loaded, but we keep the call shape
                # consistent.
                print(
                    "SC1 in `sanity` subcommand requires the original dataset; "
                    "pass --dataset to a future revision or run SC1 in a notebook."
                )
                continue
        elif check == "sc2":
            sc.sc2_span_coverage(results)
        elif check == "sc3":
            sc.sc3_anisotropy(results, num_layers=num_layers)
        elif check == "sc4":
            sc.sc4_behavioral_accuracy(
                results,
                pooling_methods=CONTRACT_V1.pooling_methods,
                has_generation=HAS_GEN_BY_MODEL[args.model],
            )
        elif check == "sc5":
            sc.sc5_layer_margin_curve(
                results,
                base_dir=args.results_dir,
                prefix=prefix,
                num_layers=num_layers,
                pooling_methods=CONTRACT_V1.pooling_methods,
            )
        elif check == "sc6":
            sc.sc6_positional_confound(
                results,
                base_dir=args.results_dir,
                prefix=prefix,
                num_layers=num_layers,
            )
        elif check == "sc7":
            sc.sc7_shuffle_decomposition(
                results,
                base_dir=args.results_dir,
                prefix=prefix,
                num_layers=num_layers,
                n_shuffles=CONTRACT_V1.n_shuffles,
            )
        else:
            print(f"Unknown check: {check}. Expected sc1..sc7.")

    # Silence linter for the lazy-imported module reference
    _ = module
    return 0


def _cmd_visualize(args: argparse.Namespace) -> int:
    """Render figures from already-extracted outputs.

    The heavy plotting/reduction libraries live in the optional ``[viz]`` group
    and are imported lazily here, so ``run``/``doctor`` never require them.
    """
    try:
        from .viz import pipeline as viz_pipeline
    except ImportError as exc:
        raise SystemExit(
            "Visualization dependencies are not installed. "
            'Install them with: pip install -e ".[viz]"\n'
            f"(import error: {exc})"
        )

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    boards = None
    if args.boards:
        boards = [int(x) for x in args.boards.split(",") if x.strip()]

    common = dict(
        output_root=args.output_dir, viz_dir=args.viz_dir,
        n_boards=args.n_boards, pooling=args.pooling, layers=layers, seed=args.seed,
        boards=boards, skip_projection=args.heatmaps_only,
    )
    if args.all:
        viz_pipeline.run_all(**common)
    else:
        viz_pipeline.run(args.model, **common)
    return 0


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Build the cross-model tables and figures from existing outputs."""
    steps = [s.strip().lower() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in _AGGREGATE_STEPS]
    if unknown:
        raise SystemExit(
            f"Unknown aggregate step(s): {', '.join(unknown)}. "
            f"Expected a subset of: {', '.join(_AGGREGATE_STEPS)}."
        )
    analysis_dir = args.analysis_dir or os.path.join(args.output_dir, "analysis")

    if "tables" in steps or "concordance" in steps:
        from .analysis import tables as agg_tables

        print("[aggregate] building metric tables")
        agg_tables.build_all(
            args.output_dir, analysis_dir,
            n_boot=args.n_boot, seed=args.seed,
            include_concordance_by_layer=("concordance" in steps),
        )

    if "boards" in steps:
        from .analysis import boards as agg_boards

        print("[aggregate] computing per-board peak depths from the vector subsample")
        agg_boards.run(args.output_dir, analysis_dir, pooling=args.pooling)

    if "trust" in steps:
        try:
            from .analysis import trust as agg_trust
        except ImportError as exc:
            raise SystemExit(
                "The trust step needs the [viz] dependencies. "
                'Install them with: pip install -e ".[viz]"\n'
                f"(import error: {exc})"
            )
        print("[aggregate] running trustworthiness sweep (UMAP per board x layer)")
        agg_trust.run(
            args.output_dir, analysis_dir,
            pooling=args.pooling, n_layers=args.trust_layers, k=args.trust_k,
            seed=args.seed, max_boards=args.trust_boards,
        )

    if "figures" in steps:
        try:
            from .analysis import figures as agg_figures
        except ImportError as exc:
            raise SystemExit(
                "The figures step needs the [viz] dependencies. "
                'Install them with: pip install -e ".[viz]"\n'
                f"(import error: {exc})"
            )
        print("[aggregate] rendering publication figures")
        agg_figures.render_all(analysis_dir, args.figures_dir,
                               pooling=args.pooling, output_root=args.output_dir)

    if "examples" in steps:
        try:
            from .analysis import figures as agg_figures
        except ImportError as exc:
            raise SystemExit(
                "The examples step needs the [viz] dependencies. "
                'Install them with: pip install -e ".[viz]"\n'
                f"(import error: {exc})"
            )
        print("[aggregate] rendering the per-board example library (umaps/ + heatmaps/)")
        agg_figures.render_examples(
            args.output_dir, args.figures_dir,
            n_boards=args.example_boards, pooling=args.pooling, seed=args.seed)

    print(f"[aggregate] done. tables: {analysis_dir}  figures: {args.figures_dir}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Lens subcommands (docs/specs/lens_spec.md Draft v2)
# ---------------------------------------------------------------------------

# Registry keys of the causal decoders the lens instrument is defined for.
_LENS_MODELS = ("mistral", "qwen", "qwen_random")
# Registry key -> output prefix (matches each loader's metadata["prefix"]).
_LENS_PREFIXES = {"mistral": "mistral", "qwen": "qwen",
                  "qwen_random": "random_qwen"}
# Registry key -> HF id whose tokenizer the offline stages load. lens-apply
# never loads the 7B weights; the random-init decoder shares Qwen's tokenizer.
_LENS_TOKENIZERS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "qwen_random": "Qwen/Qwen2.5-7B-Instruct",
}
# Registry key -> chat template strategy (matches loader metadata; kept here
# so the offline stages don't need the loader).
_LENS_CHAT_TEMPLATES = {"mistral": "mistral_inst", "qwen": "chatml",
                        "qwen_random": "chatml"}


def _require_lens_model(model: str) -> None:
    if model not in _LENS_MODELS:
        raise SystemExit(
            f"lens commands are scoped to the causal decoders "
            f"{', '.join(_LENS_MODELS)}; got --model {model!r}. Encoders "
            f"have no next-token unembedding (docs/specs/lens_spec.md §10)."
        )


def _lens_resolve_contract(args, df) -> "object":
    """--full > --sample-size > contract default, as in the run command."""
    import dataclasses
    from .contract import CONTRACT_V1

    if args.full:
        requested_n = len(df)
        print(f"Run size: FULL dataset ({requested_n} boards)")
    elif args.sample_size is not None:
        requested_n = args.sample_size
        print(f"Run size: {requested_n} boards (--sample-size)")
    else:
        requested_n = CONTRACT_V1.sample_size
        print(f"Run size: {requested_n} boards (contract default)")
    if requested_n != CONTRACT_V1.sample_size:
        return dataclasses.replace(CONTRACT_V1, sample_size=requested_n)
    return CONTRACT_V1


def _make_lens_extract_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "lens-extract",
        help="Dump per-layer hidden states at the generating position (GPU).",
        description=(
            "One forward pass per board (canonical ordering only). Writes a "
            "fp16 memmap [N, layers+1, d], an index CSV, and the model's "
            "readout weights. Resumable. docs/specs/lens_spec.md §5."
        ),
    )
    p.add_argument("--model", required=True, choices=list(_LENS_MODELS))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--output-dir", required=True,
                   help="Model output dir, e.g. output/mistral_outputs.")
    p.add_argument("--sample-size", type=int, default=None,
                   help="Number of boards (default: contract N=2000).")
    p.add_argument("--full", action="store_true",
                   help="Run every board in the dataset.")
    p.add_argument("--conditions", default="no_social,with_social",
                   help="Comma-separated subset of: no_social,with_social.")
    p.add_argument("--resume", action="store_true",
                   help="Continue an interrupted lens extraction.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Manifest dir (default: <output-dir>/checkpoints).")
    p.add_argument("--flash-attn", action="store_true",
                   help="Load trained models with flash_attention_2.")
    return p


def _make_lens_tune_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "lens-tune",
        help="Train tuned-lens translators on generic text (GPU).",
        description=(
            "Per-layer affine trained to match the model's own final logits "
            "(never the human labels). Saves {prefix}_lens_translators.npz. "
            "docs/specs/lens_spec.md §6."
        ),
    )
    p.add_argument("--model", required=True, choices=list(_LENS_MODELS))
    p.add_argument("--output-dir", required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--train-text", default=None,
                     help="Plain-text file to train on.")
    src.add_argument("--hf-dataset", default="wikitext/wikitext-103-raw-v1",
                     help="HF dataset as name/config (needs the [lens] extra).")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-chars", type=int, default=20_000_000,
                   help="Character budget drawn from the training corpus.")
    return p


def _make_lens_apply_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "lens-apply",
        help="Score candidates through raw + tuned lenses (offline, no GPU).",
        description=(
            "Reads the hidden dump + readout weights (+ translators if "
            "present) and writes {prefix}_lens_scores_{mode}.parquet."
        ),
    )
    p.add_argument("--model", required=True, choices=list(_LENS_MODELS))
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--full", action="store_true")
    p.add_argument("--conditions", default="no_social")
    p.add_argument("--lenses", default="raw,tuned",
                   help="Comma-separated subset of: raw,tuned.")
    return p


def _make_lens_analyze_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "lens-analyze",
        help="Curves, decision rules, controls, overlay figures (offline).",
        description=(
            "Applies the pre-registered docs/specs/lens_spec.md §3 rules to saved "
            "scores. NOTE: --models/--random-model take output PREFIXES "
            "(mistral, qwen, random_qwen), not registry keys."
        ),
    )
    p.add_argument("--output-root", default="output")
    p.add_argument("--models", default="mistral,qwen")
    p.add_argument("--random-model", default="random_qwen",
                   help="Prefix of the random-init null, or 'none'.")
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--out-dir", default=os.path.join("output", "lens_analysis"))
    p.add_argument("--figures-dir", default=os.path.join("visualization", "lens"))
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2026)
    return p


def _cmd_lens_extract(args: argparse.Namespace) -> int:
    from .data import load_dataset, sample_turns
    from .lens.extract import run_lens_extraction

    _require_lens_model(args.model)
    df = load_dataset(args.dataset)
    contract = _lens_resolve_contract(args, df)

    loader = _resolve_loader(args.model)
    if args.flash_attn and args.model in ("mistral", "qwen"):
        print("Loading model with attn_implementation='flash_attention_2'")
        model, tokenizer, meta = loader(attn_implementation="flash_attention_2")
    else:
        model, tokenizer, meta = loader()

    df_sample = sample_turns(df, n=contract.sample_size,
                             seed=contract.random_seed)
    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())

    run_lens_extraction(
        model=model,
        tokenizer=tokenizer,
        df=df_sample,
        base_dir=args.output_dir,
        prefix=meta["prefix"],
        contract=contract,
        chat_template_strategy=meta["chat_template_strategy"],
        num_layers=meta["num_layers"],
        hidden_dim=meta["hidden_dim"],
        conditions=conditions,
        device=meta["device"],
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )
    return 0


def _cmd_lens_tune(args: argparse.Namespace) -> int:
    from .lens.tuned import TunedLensConfig, train_tuned_lens

    _require_lens_model(args.model)

    if args.train_text:
        with open(args.train_text, "r", encoding="utf-8") as f:
            texts = [f.read()[: args.max_chars]]
    else:
        try:
            from datasets import load_dataset as hf_load_dataset
        except ImportError as e:
            raise SystemExit(
                "lens-tune needs the [lens] extra for --hf-dataset "
                "(pip install -e '.[lens]') or pass --train-text FILE."
            ) from e
        name, _, config = args.hf_dataset.partition("/")
        ds = hf_load_dataset(name, config or None, split="train")
        texts, total = [], 0
        for rec in ds:
            t = rec.get("text", "")
            if t.strip():
                texts.append(t)
                total += len(t)
            if total >= args.max_chars:
                break

    loader = _resolve_loader(args.model)
    model, tokenizer, meta = loader()

    cfg = TunedLensConfig(seq_len=args.seq_len, n_steps=args.steps)
    lens = train_tuned_lens(model, tokenizer, texts, cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir,
                       f"{meta['prefix']}_lens_translators.npz")
    lens.save(out)
    val_msg = (f", final val KL {lens.val_history[-1]:.4f}"
               if lens.val_history else "")
    print(f"Tuned-lens translators saved: {out} "
          f"(final loss {lens.history[-1]:.4f}{val_msg})")
    return 0


def _cmd_lens_apply(args: argparse.Namespace) -> int:
    from transformers import AutoTokenizer

    from .data import load_dataset, sample_turns
    from .lens.apply import compute_scores, load_readout, save_scores
    from .lens.tuned import TunedLens

    _require_lens_model(args.model)
    prefix = _LENS_PREFIXES[args.model]
    df = load_dataset(args.dataset)
    contract = _lens_resolve_contract(args, df)
    df_sample = sample_turns(df, n=contract.sample_size,
                             seed=contract.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(_LENS_TOKENIZERS[args.model])
    readout = load_readout(os.path.join(
        args.output_dir, f"{prefix}_lens_readout_f16.npz"))

    lenses = [x.strip() for x in args.lenses.split(",") if x.strip()]
    translators = None
    if "tuned" in lenses:
        tpath = os.path.join(args.output_dir,
                             f"{prefix}_lens_translators.npz")
        if os.path.exists(tpath):
            translators = TunedLens.load(tpath)
        else:
            print(f"  WARNING: {tpath} not found; skipping tuned lens "
                  f"(run lens-tune first).")
            lenses = [x for x in lenses if x != "tuned"]

    for mode_name in (c.strip() for c in args.conditions.split(",")):
        hidden = os.path.join(args.output_dir,
                              f"{prefix}_lens_hidden_{mode_name}_f16.npy")
        index = os.path.join(args.output_dir,
                             f"{prefix}_lens_index_{mode_name}.csv")
        if not os.path.exists(hidden):
            print(f"  WARNING: no hidden dump for '{mode_name}' "
                  f"({hidden}); skipping.")
            continue
        frames = []
        if "raw" in lenses:
            frames.append(compute_scores(hidden, index, df_sample,
                                         tokenizer, readout, "raw"))
        if translators is not None:
            frames.append(compute_scores(hidden, index, df_sample,
                                         tokenizer, readout, "tuned",
                                         translators=translators))
        if frames:
            out = save_scores(frames, args.output_dir, prefix, mode_name)
            print(f"  Scores saved: {out}")
    return 0


def _cmd_lens_analyze(args: argparse.Namespace) -> int:
    from .lens.analysis import run_analysis
    from .lens.figures import lens_overlay_figure

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    random_model = None if args.random_model == "none" else args.random_model

    summary = run_analysis(
        output_root=args.output_root,
        models=models,
        random_model=random_model,
        mode=args.condition,
        out_dir=args.out_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    concordance_csv = os.path.join(args.output_root, "analysis",
                                   "analysis_concordance_by_layer.csv")
    curves_csv = os.path.join(args.out_dir,
                              f"lens_curves_{args.condition}.csv")
    if os.path.exists(curves_csv) and os.path.exists(concordance_csv):
        all_curves = pd.read_csv(curves_csv)
        random_curves = (
            all_curves[all_curves["model"] == random_model]
            if random_model is not None else None)
        for m in models:
            if m not in summary:
                continue
            gen_acc = summary[m]["calibration"]["generation_acc"]
            out_path = os.path.join(args.figures_dir,
                                    f"lens_overlay_{m}_{args.condition}.png")
            lens_overlay_figure(
                all_curves[all_curves["model"] == m], m, concordance_csv,
                gen_acc, out_path, condition=args.condition,
                random_curves=random_curves)
            print(f"  Figure saved: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Remote job subcommands
# (docs/superpowers/specs/2026-07-27-colab-remote-execution-design.md)
# ---------------------------------------------------------------------------

def _add_store_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--jobs-dir",
        default=None,
        help=(
            "Path to a local jobs tree. Given, the command talks to the "
            "filesystem instead of Drive — used by the Colab runner (which "
            "sees Drive as a mount) and by tests."
        ),
    )
    p.add_argument("--drive-root", default="Codenames-Research",
                   help="Drive folder holding the run outputs and the _jobs tree.")
    p.add_argument("--token", default="token.json",
                   help="OAuth token cache path.")
    p.add_argument("--client-secret", default="client_secret.json",
                   help="OAuth installed-app client secret path.")


def _open_remote_store(args):
    """Return a job store: filesystem when --jobs-dir is given, else Drive."""
    from codenames.remote.store import LocalDirStore

    if args.jobs_dir:
        store = LocalDirStore(args.jobs_dir)
        store.ensure_layout()
        return store

    from pathlib import Path

    from codenames.remote.drive import (
        DriveApiStore,
        build_drive_service,
        load_credentials,
        resolve_folder,
    )

    creds = load_credentials(Path(args.token), Path(args.client_secret))
    service = build_drive_service(creds)
    root = resolve_folder(service, [args.drive_root, "_jobs"], create=True)
    store = DriveApiStore(service, root)
    store.ensure_layout()
    return store


def _make_job_submit_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "job-submit",
        help="Enqueue a GPU job for the Colab runner.",
        description=(
            "Write a job document into the queue. The subcommand after -- is "
            "run verbatim by the runner and must be on its whitelist."
        ),
    )
    _add_store_args(p)
    p.add_argument("--git-ref", default="probing",
                   help="Git ref the runner checks out before running the job.")
    p.add_argument("--timeout", type=int, default=6 * 3600,
                   help="Job timeout in seconds.")
    p.add_argument("--expect-gpu", default=None,
                   help="Refuse the job unless the session GPU name contains this.")
    p.add_argument("job_argv", nargs=argparse.REMAINDER,
                   help="After --, the codenames-experiment subcommand and its flags.")
    return p


def _cmd_job_submit(args) -> int:
    from codenames.remote import client

    argv = [a for a in args.job_argv if a != "--"]
    store = _open_remote_store(args)
    try:
        job = client.submit(store, argv, git_ref=args.git_ref,
                            timeout_s=args.timeout, expect_gpu=args.expect_gpu)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Queued {job.job_id}")
    print(f"  argv:    {' '.join(job.argv)}")
    print(f"  git ref: {job.git_ref}")
    return 0


def _make_job_status_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser("job-status", help="Runner health plus recent job states.")
    _add_store_args(p)
    p.add_argument("--job", default=None, help="Show only this job.")
    p.add_argument("--limit", type=int, default=10,
                   help="How many recent jobs to list.")
    p.add_argument("--watch", action="store_true",
                   help=("Poll until nothing is queued or running, holding macOS "
                         "awake meanwhile so the Colab tab is not reaped."))
    p.add_argument("--interval", type=float, default=30.0,
                   help="Seconds between polls when --watch is set.")
    return p


def _print_job_status(store, args) -> int:
    from codenames.remote import client

    health = client.runner_health(store)
    if health.heartbeat is None:
        print("Runner: no runner has ever reported in this tree.")
    elif health.alive:
        hb = health.heartbeat
        print(f"Runner: ALIVE ({hb.runner_id}, {health.age_s:.0f}s ago) "
              f"gpu={hb.gpu_name} job={hb.current_job or '-'}")
    else:
        print(f"Runner: STALE — last heartbeat {health.age_s:.0f}s ago. "
              f"The Colab session is probably dead; re-run the runner cell.")

    if args.job:
        status = store.read_status(args.job)
        if status is None:
            print(f"No status for {args.job}", file=sys.stderr)
            return 1
        print(client.format_status_line(status))
        return 0

    for status in store.list_statuses(limit=args.limit):
        print(client.format_status_line(status))
    return 0


def _cmd_job_status(args) -> int:
    import time

    from codenames.remote import client

    store = _open_remote_store(args)
    if not args.watch:
        return _print_job_status(store, args)

    with client.KeepAwake():
        while True:
            rc = _print_job_status(store, args)
            if not client.has_unfinished_work(store):
                return rc
            time.sleep(args.interval)
            print("-" * 70)


def _make_job_logs_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser("job-logs", help="Print a job's captured output.")
    _add_store_args(p)
    p.add_argument("--job", required=True)
    p.add_argument("--tail", type=int, default=None, help="Only the last N lines.")
    return p


def _cmd_job_logs(args) -> int:
    from codenames.remote import client

    store = _open_remote_store(args)
    text = client.logs(store, args.job, tail=args.tail)
    if not text:
        print(f"No log for {args.job}", file=sys.stderr)
        return 1
    print(text)
    return 0


def _make_job_cancel_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser("job-cancel", help="Request cancellation of a running job.")
    _add_store_args(p)
    p.add_argument("--job", required=True)
    return p


def _cmd_job_cancel(args) -> int:
    from codenames.remote import client

    store = _open_remote_store(args)
    if not client.cancel(store, args.job):
        print(f"{args.job} has already finished; nothing to cancel.", file=sys.stderr)
        return 1
    print(f"Cancellation requested for {args.job}.")
    return 0


def _make_job_sync_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser("job-sync",
                      help="Download run artifacts from Drive into output/.")
    _add_store_args(p)
    p.add_argument("--models", default=",".join(MODEL_REGISTRY),
                   help="Comma-separated model prefixes to sync.")
    p.add_argument("--output-dir", default="output")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--skip-vectors", action="store_true",
                       help="Skip the multi-GB .npz/.npy artifacts (iteration only).")
    group.add_argument("--only-vectors", action="store_true",
                       help="Fetch only the multi-GB .npz/.npy artifacts.")
    p.add_argument("--dry-run", action="store_true")
    return p


def _cmd_job_sync(args) -> int:
    from pathlib import Path

    from codenames.remote.drive import (
        build_drive_service,
        load_credentials,
        resolve_folder,
    )
    from codenames.remote.sync import sync_model_outputs

    creds = load_credentials(Path(args.token), Path(args.client_secret))
    service = build_drive_service(creds)
    root = resolve_folder(service, [args.drive_root])
    if root is None:
        print(f"Drive folder '{args.drive_root}' not found.", file=sys.stderr)
        return 1
    report = sync_model_outputs(
        service, root, Path(args.output_dir), args.models.split(","),
        skip_heavy=args.skip_vectors, only_heavy=args.only_vectors,
        dry_run=args.dry_run,
    )
    verb = "Would download" if args.dry_run else "Downloaded"
    print(f"{verb} {len(report.downloaded)} file(s), "
          f"{report.bytes_downloaded / 1e9:.2f} GB; skipped {len(report.skipped)}.")
    for name in report.downloaded:
        print(f"  {name}")
    return 0


def _make_job_runner_parser(sp: "argparse._SubParsersAction") -> argparse.ArgumentParser:
    p = sp.add_parser(
        "job-runner",
        help="Run the agent poll loop (this is what the Colab notebook executes).",
        description=(
            "Claim and execute queued jobs until the idle budget or the session "
            "limit runs out. Exiting when idle matters: a poller holding an A100 "
            "keeps burning Colab compute units for nothing."
        ),
    )
    _add_store_args(p)
    p.add_argument("--repo-dir", default="/content/codenames-interpretability")
    p.add_argument("--log-dir", default="/content/logs")
    p.add_argument("--idle-shutdown-minutes", type=float, default=30.0)
    p.add_argument("--max-session-hours", type=float, default=11.0)
    p.add_argument("--skip-git", action="store_true",
                   help="Do not fetch/reset the repo before each job.")
    return p


def _cmd_job_runner(args) -> int:
    from pathlib import Path

    from codenames.remote.runner import RunnerConfig, run_agent_loop

    store = _open_remote_store(args)
    cfg = RunnerConfig(
        repo_dir=Path(args.repo_dir),
        jobs_root=Path(args.jobs_dir) if args.jobs_dir else Path("."),
        local_log_dir=Path(args.log_dir),
        idle_shutdown_minutes=args.idle_shutdown_minutes,
        max_session_hours=args.max_session_hours,
        skip_git=args.skip_git,
    )
    return run_agent_loop(cfg, store=store)


# --------------------------------------------------------------------------
# Causal tier (docs/specs/causal_spec.md v3). Handlers import codenames.causal
# lazily so `--help` never pays for torch.
# --------------------------------------------------------------------------

_CAUSAL_MODELS = ("mistral", "qwen", "qwen_random")
_CAUSAL_SCHEMES = ("counterfactual", "noise")
_PILOT_N = 150         # §12.5 pilot draw
_CONFIRMATORY_N = 1500  # §4.2 committed sample size


def _add_causal_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, choices=list(_CAUSAL_MODELS))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=2026,
                   help="CONTRACT_V1.random_seed; do not change.")


def _make_causal_extract_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-extract",
        help="Paired clean/corrupted forwards over the seeded subsample.",
        description=(
            "Builds the symmetric-counterfactual pair table (§5A) and caches "
            "clean and corrupted per-layer states. Pairs whose prompts differ "
            "in token count are dropped and logged: patching (layer, position) "
            "across misaligned sequences reads the wrong position."
        ),
    )
    _add_causal_common(p)
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--scheme", default="counterfactual", choices=list(_CAUSAL_SCHEMES))
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--sample-size", type=int, default=_CONFIRMATORY_N)
    p.add_argument("--noise-sigma", type=float, default=3.0,
                   help="Multiples of the embedding component SD (ROME setting).")
    return p


def _make_causal_scan_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-scan",
        help="Attribution-patching screen over the (layer, position) grid.",
        description=(
            "Stage 1 of the two-stage design. SCREENING ONLY - carries no "
            "inferential claim (§3.4). Every locus it surfaces must be "
            "confirmed with a real patch."
        ),
    )
    _add_causal_common(p)
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--sample-size", type=int, default=_CONFIRMATORY_N)
    p.add_argument("--scheme", default="counterfactual", choices=list(_CAUSAL_SCHEMES))
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--top-k", type=int, default=200,
                   help="Loci carried into stage 2 (budget ceiling, not a prediction).")
    return p


def _make_causal_patch_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-patch",
        help="Real patches on candidate loci (resumable; the expensive stage).",
    )
    _add_causal_common(p)
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--scheme", default="counterfactual", choices=list(_CAUSAL_SCHEMES))
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--sample-size", type=int, default=_CONFIRMATORY_N)
    p.add_argument("--window-widths", default="1,3,5",
                   help="Contiguous layer-band widths; 1 is single-site.")
    p.add_argument("--validation-fraction", type=float, default=0.10,
                   help="Random grid share patched regardless of attribution score.")
    p.add_argument("--resume", action="store_true")
    return p


def _make_causal_steer_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-steer",
        help="Dose-response steering with the four pre-registered controls.",
    )
    _add_causal_common(p)
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--layer", type=int, required=True,
                   help="Injection layer; sweep by submitting one job per layer.")
    p.add_argument("--direction", default="lens", choices=["lens", "dom"],
                   help="lens = label-free (primary); dom = label-fitted (robustness).")
    p.add_argument("--sites", default="from_hint",
                   choices=["from_hint", "hint_only", "generating"])
    p.add_argument("--alphas", default="-8,-4,-2,-1,-0.5,0.5,1,2,4,8")
    p.add_argument("--sample-size", type=int, default=_CONFIRMATORY_N)
    return p


def _make_causal_pilot_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-pilot",
        help="Run the P1-P8 method-validation gate (required before the full run).",
        description=(
            "Validates the machinery on a seeded 150-turn draw from the "
            "secondary condition, leaving the confirmatory sample untouched. "
            "P1-P4 and P7 are blocking. Anti-peeking rule applies: only "
            "nuisance parameters and pass/fail verdicts may inform the "
            "confirmatory run - never effect locations."
        ),
    )
    _add_causal_common(p)
    p.add_argument("--dataset", required=True, help="Path to clue_generation.csv.")
    p.add_argument("--sample-size", type=int, default=_PILOT_N)
    p.add_argument("--condition", default="with_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--generation-csv", default=None,
                   help="Recorded generations, needed by check P1. Defaults to "
                        "{output-dir}/{prefix}_generation_{condition}.csv.")
    p.add_argument("--no-generations", action="store_true",
                   help="Skip check P1 because this model has no recorded "
                        "generations (the random-init null). Must be explicit: "
                        "a merely missing file is an error, not a free skip.")
    p.add_argument("--report-path", default=None,
                   help="Where to write the P1-P8 table (default: output dir).")
    return p


def _make_causal_analyze_parser(sp) -> argparse.ArgumentParser:
    p = sp.add_parser(
        "causal-analyze",
        help="FDR, cluster bootstrap, claim gate and figures (offline, no GPU).",
    )
    _add_causal_common(p)
    p.add_argument("--condition", default="no_social",
                   choices=["no_social", "with_social"])
    p.add_argument("--q", type=float, default=0.05, help="BH-FDR level.")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--figures", action="store_true")
    return p


def _cmd_causal(args) -> int:
    """Dispatch the causal subcommands, importing torch-backed code lazily."""
    from .causal import runner  # noqa: PLC0415  (lazy by design)

    return runner.dispatch(args)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full CLI parser.

    Split out of ``main`` so the subcommand surface can be asserted in tests
    without executing anything.
    """
    parser = argparse.ArgumentParser(
        prog="codenames-experiment",
        description=(
            "Layer-wise word representation geometry across transformer "
            "architectures, evaluated on Codenames Duet."
        ),
    )
    sp = parser.add_subparsers(dest="command", required=True)
    _make_doctor_parser(sp)
    _make_run_parser(sp)
    _make_preflight_parser(sp)
    _make_validate_parser(sp)
    _make_compare_parser(sp)
    _make_sanity_parser(sp)
    _make_visualize_parser(sp)
    _make_aggregate_parser(sp)
    _make_lens_extract_parser(sp)
    _make_lens_tune_parser(sp)
    _make_lens_apply_parser(sp)
    _make_lens_analyze_parser(sp)
    _make_job_submit_parser(sp)
    _make_job_status_parser(sp)
    _make_job_logs_parser(sp)
    _make_job_cancel_parser(sp)
    _make_job_sync_parser(sp)
    _make_job_runner_parser(sp)
    _make_causal_extract_parser(sp)
    _make_causal_scan_parser(sp)
    _make_causal_patch_parser(sp)
    _make_causal_steer_parser(sp)
    _make_causal_pilot_parser(sp)
    _make_causal_analyze_parser(sp)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "preflight":
        return _cmd_preflight(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "sanity":
        return _cmd_sanity(args)
    if args.command == "visualize":
        return _cmd_visualize(args)
    if args.command == "aggregate":
        return _cmd_aggregate(args)
    if args.command == "lens-extract":
        return _cmd_lens_extract(args)
    if args.command == "lens-tune":
        return _cmd_lens_tune(args)
    if args.command == "lens-apply":
        return _cmd_lens_apply(args)
    if args.command == "lens-analyze":
        return _cmd_lens_analyze(args)
    if args.command == "job-submit":
        return _cmd_job_submit(args)
    if args.command == "job-status":
        return _cmd_job_status(args)
    if args.command == "job-logs":
        return _cmd_job_logs(args)
    if args.command == "job-cancel":
        return _cmd_job_cancel(args)
    if args.command == "job-sync":
        return _cmd_job_sync(args)
    if args.command == "job-runner":
        return _cmd_job_runner(args)
    if args.command.startswith("causal-"):
        return _cmd_causal(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
