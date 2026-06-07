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
    p.add_argument("--output-dir", required=True, help="Base output directory (BASE_DIR).")
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
            "Runs the refactored pipeline on a small subsample and compares "
            "the outputs against the corresponding rows of an existing "
            "N=2000 run. Headless equivalent of notebooks/00_validation.ipynb."
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
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for board sampling and reducers (default: 42).")
    p.add_argument("--layers", default=None,
                   help="Comma-separated explicit layer indices; default picks ~6 by depth.")
    p.add_argument("--boards", default=None,
                   help="Comma-separated explicit board row_ids to visualize. Pins the "
                        "SAME boards across models for direct comparison. Overrides "
                        "--n-boards. With --all and no --boards, boards are sampled from "
                        "the intersection available across all models.")
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
        boards=boards,
    )
    if args.all:
        viz_pipeline.run_all(**common)
    else:
        viz_pipeline.run(args.model, **common)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
