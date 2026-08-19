import pytest

from codenames.cli import build_parser

CAUSAL_COMMANDS = [
    "causal-extract", "causal-scan", "causal-patch",
    "causal-steer", "causal-pilot", "causal-analyze",
]


def _choices(parser):
    for action in parser._subparsers._group_actions:
        if action.choices:
            return action.choices
    raise AssertionError("no subparsers found")


@pytest.mark.parametrize("name", CAUSAL_COMMANDS)
def test_subcommand_is_registered(name):
    assert name in _choices(build_parser())


def test_existing_commands_still_registered():
    """The refactor must not drop any pre-existing subcommand."""
    choices = _choices(build_parser())
    for name in ("run", "doctor", "aggregate", "lens-extract", "lens-analyze"):
        assert name in choices


def test_pilot_defaults_match_the_spec():
    args = build_parser().parse_args(["causal-pilot", "--model", "mistral",
                                      "--output-dir", "/tmp/x",
                                      "--dataset", "/tmp/d.csv"])
    assert args.sample_size == 150
    assert args.condition == "with_social"
    assert args.seed == 2026


def test_patch_defaults_to_the_confirmatory_sample_size():
    args = build_parser().parse_args(["causal-patch", "--model", "mistral",
                                      "--output-dir", "/tmp/x",
                                      "--dataset", "/tmp/d.csv"])
    assert args.sample_size == 1500
    assert args.seed == 2026


def test_extract_exposes_both_corruption_schemes():
    args = build_parser().parse_args([
        "causal-extract", "--model", "qwen", "--output-dir", "/tmp/x",
        "--dataset", "/tmp/d.csv", "--scheme", "noise",
    ])
    assert args.scheme == "noise"
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "causal-extract", "--model", "qwen", "--output-dir", "/tmp/x",
            "--dataset", "/tmp/d.csv", "--scheme", "bogus",
        ])


def test_causal_models_are_the_three_decoders():
    parser = build_parser()
    for model in ("mistral", "qwen", "qwen_random"):
        args = parser.parse_args(["causal-pilot", "--model", model,
                                  "--output-dir", "/tmp/x",
                                  "--dataset", "/tmp/d.csv"])
        assert args.model == model
    with pytest.raises(SystemExit):
        parser.parse_args(["causal-pilot", "--model", "bert",
                           "--output-dir", "/tmp/x", "--dataset", "/tmp/d.csv"])


def test_dispatch_routes_every_causal_command():
    from codenames.causal import runner
    for name in CAUSAL_COMMANDS:
        assert name in runner._DISPATCH


def test_every_stage_is_built_and_routable():
    """Stages whose orchestration is not built must say so explicitly."""
    from codenames.causal import runner
    import argparse
    # All stages are now built; dispatch must route each one.
    for name in CAUSAL_COMMANDS:
        assert callable(runner._DISPATCH[name])


def test_help_does_not_import_torch():
    """--help must stay fast; torch is imported lazily per stage."""
    import subprocess, sys
    code = (
        "import sys; from codenames.cli import build_parser; build_parser(); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_causal_loader_resolves_for_every_supported_model():
    """Regression: the loaders are named per model, not load_<key>.

    Caught by an end-to-end queue run that failed with
    "no loader registered for 'mistral'" before reaching any real work.
    """
    from codenames.cli import MODEL_REGISTRY
    for model in ("mistral", "qwen", "qwen_random"):
        assert model in MODEL_REGISTRY
        module_path, attr = MODEL_REGISTRY[model]
        import importlib
        mod = importlib.import_module(module_path)
        assert callable(getattr(mod, attr)), f"{module_path}.{attr} is not callable"


def test_causal_runner_uses_the_shared_registry():
    import inspect
    from codenames.causal import runner
    assert "_resolve_loader" in inspect.getsource(runner._load_model)


def test_patch_passes_resume_through_to_the_stage():
    """causal-patch is ~94% of the budget; a dropped session must not lose it."""
    import inspect
    from codenames.causal import runner
    src = inspect.getsource(runner.cmd_patch)
    assert "checkpoint_dir" in src and "resume=" in src


def test_patch_exposes_a_resume_flag():
    args = build_parser().parse_args(["causal-patch", "--model", "mistral",
                                      "--output-dir", "/tmp/x",
                                      "--dataset", "/tmp/d.csv", "--resume"])
    assert args.resume is True


def test_pilot_exposes_an_explicit_no_generations_optout():
    args = build_parser().parse_args(["causal-pilot", "--model", "qwen_random",
                                      "--output-dir", "/tmp/x",
                                      "--dataset", "/tmp/d.csv", "--no-generations"])
    assert args.no_generations is True


def test_no_generations_defaults_false_so_a_missing_file_still_errors():
    args = build_parser().parse_args(["causal-pilot", "--model", "mistral",
                                      "--output-dir", "/tmp/x",
                                      "--dataset", "/tmp/d.csv"])
    assert args.no_generations is False


# --- T3d: the geometric intervention rides on causal-steer ------------------

def test_steer_exposes_the_equalise_intervention():
    """A new subcommand would have to be added to the runner whitelist, which
    widens what a file dropped in a Drive folder can execute. The geometric
    intervention is a mode of the already-whitelisted causal-steer instead."""
    args = build_parser().parse_args(
        ["causal-steer", "--model", "mistral", "--output-dir", "/tmp/x",
         "--dataset", "/tmp/d.csv", "--intervention", "equalise",
         "--layers", "0,5,23"])
    assert args.intervention == "equalise"
    assert args.layers == "0,5,23"


def test_steer_still_defaults_to_the_additive_intervention():
    args = build_parser().parse_args(
        ["causal-steer", "--model", "mistral", "--output-dir", "/tmp/x",
         "--dataset", "/tmp/d.csv", "--layer", "5"])
    assert args.intervention == "additive"


def test_equalise_accepts_a_full_layer_sweep_in_one_job():
    """The spec replaced a chosen null layer with the whole sweep, so the sweep
    has to be one job rather than 33."""
    args = build_parser().parse_args(
        ["causal-steer", "--model", "mistral", "--output-dir", "/tmp/x",
         "--dataset", "/tmp/d.csv", "--intervention", "equalise",
         "--layers", "all"])
    assert args.layers == "all"


def test_steer_routes_the_equalise_mode_to_its_own_stage():
    import inspect
    from codenames.causal import runner
    src = inspect.getsource(runner.cmd_steer)
    assert "run_equalise_stage" in src and "equalise" in src


def test_equalise_resolves_the_answer_position_like_every_other_stage():
    """The readout must sit at p_read. Passing no generation CSV silently
    downgrades it to the generating position, which on Mistral is the wrong
    token on 87% of turns — the defect spec amendment (k) exists to prevent."""
    import inspect
    from codenames.causal import runner
    src = inspect.getsource(runner.cmd_steer)
    assert "_generation_csv(args, paths)" in src
    assert 'paths.get("generation")' not in src


def test_steer_exposes_the_generation_flags_the_readout_needs():
    args = build_parser().parse_args(
        ["causal-steer", "--model", "mistral", "--output-dir", "/tmp/x",
         "--dataset", "/tmp/d.csv", "--intervention", "equalise",
         "--generation-csv", "/tmp/g.csv"])
    assert args.generation_csv == "/tmp/g.csv"
    assert args.no_generations is False


def test_layer_sweep_spec_is_bounded_by_the_models_depth():
    from codenames.causal.runner import _requested_layers
    assert _requested_layers("all", 33) == list(range(33))
    assert _requested_layers("0,5,23", 33) == [0, 5, 23]
    # a layer the model does not have is dropped, not silently clamped onto
    # another layer, which would mislabel the depth axis
    assert _requested_layers("5,99,-1", 33) == [5]
