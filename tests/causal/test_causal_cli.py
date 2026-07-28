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
                                      "--output-dir", "/tmp/x"])
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


def test_unbuilt_stages_raise_not_implemented_rather_than_failing_oddly():
    """Stages whose orchestration is not built must say so explicitly."""
    from codenames.causal import runner
    import argparse
    for name in ("causal-scan", "causal-patch", "causal-steer"):
        args = argparse.Namespace(command=name, model="mistral",
                                  output_dir="/tmp/x", seed=2026)
        with pytest.raises(NotImplementedError, match="not built yet"):
            runner.dispatch(args)


def test_help_does_not_import_torch():
    """--help must stay fast; torch is imported lazily per stage."""
    import subprocess, sys
    code = (
        "import sys; from codenames.cli import build_parser; build_parser(); "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
