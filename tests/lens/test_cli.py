import pandas as pd
import pytest

from codenames.cli import build_parser

BASE = ["lens-extract", "--model", "mistral",
        "--dataset", "/tmp/d.csv", "--output-dir", "/tmp/x"]


def test_lens_extract_position_flags_default_off():
    args = build_parser().parse_args(BASE)
    assert args.dump_answer_position is False
    assert args.generation_csv is None
    assert args.candidate_span_csv is None


def test_lens_extract_accepts_position_flags():
    args = build_parser().parse_args(BASE + [
        "--dump-answer-position", "--generation-csv", "/tmp/gen.csv",
        "--candidate-span-csv", "/tmp/pairs.csv",
    ])
    assert args.dump_answer_position is True
    assert args.generation_csv == "/tmp/gen.csv"
    assert args.candidate_span_csv == "/tmp/pairs.csv"


def test_lens_extract_forwards_position_kwargs(tmp_path, monkeypatch):
    """The CLI must actually pass the specs' §5 dump options through —
    the pre-fix parser silently produced a generating-position-only dump."""
    import codenames.cli as cli
    import codenames.data
    import codenames.lens.extract

    pairs = tmp_path / "pairs.csv"
    pd.DataFrame({"row_id": [3, 5], "donor_row_id": [5, 3]}).to_csv(
        pairs, index=False)

    df = pd.DataFrame({"row_id": range(10), "output": ["h"] * 10})
    captured = {}

    monkeypatch.setattr(codenames.data, "load_dataset", lambda p: df)
    monkeypatch.setattr(codenames.data, "sample_turns",
                        lambda d, n, seed: d.head(n))
    monkeypatch.setattr(
        cli, "_resolve_loader",
        lambda name: lambda **kw: (object(), object(), {
            "prefix": "mistral", "chat_template_strategy": "mistral_inst",
            "num_layers": 2, "hidden_dim": 8, "device": "cpu"}))
    monkeypatch.setattr(codenames.lens.extract, "run_lens_extraction",
                        lambda **kw: captured.update(kw) or {})

    args = build_parser().parse_args(BASE + [
        "--sample-size", "5",
        "--dump-answer-position", "--generation-csv", "/tmp/gen.csv",
        "--candidate-span-csv", str(pairs),
    ])
    assert cli._cmd_lens_extract(args) == 0
    assert captured["dump_answer_position"] is True
    assert captured["generation_csv"] == "/tmp/gen.csv"
    assert captured["candidate_span_row_ids"] == {3, 5}


def test_lens_apply_channel_flag():
    base = ["lens-apply", "--model", "qwen", "--dataset", "/tmp/d.csv",
            "--output-dir", "/tmp/x"]
    assert build_parser().parse_args(base).channel == "generating"
    assert build_parser().parse_args(
        base + ["--channel", "answer"]).channel == "answer"
    with pytest.raises(SystemExit):
        build_parser().parse_args(base + ["--channel", "bogus"])
