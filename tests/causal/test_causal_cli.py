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
