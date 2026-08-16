"""Random-init controls at extra seeds (pre-submission T8)."""

import numpy as np
import pandas as pd
import pytest

from codenames.analysis.tables import random_init_seed_spread
from codenames.cli import build_parser


def test_run_accepts_init_seed_only_for_random_models():
    base = ["run", "--dataset", "/tmp/d.csv", "--output-dir", "/tmp/o"]
    a = build_parser().parse_args(base + ["--model", "qwen_random", "--init-seed", "2027"])
    assert a.init_seed == 2027
    assert build_parser().parse_args(base + ["--model", "qwen_random"]).init_seed is None


def test_init_seed_rejected_for_trained_models():
    import codenames.cli as cli
    args = build_parser().parse_args(["run", "--dataset", "/tmp/d.csv", "--output-dir",
                                      "/tmp/o", "--model", "mistral", "--init-seed", "2027"])
    with pytest.raises(SystemExit, match="init-seed"):
        cli._cmd_run(args)


def test_init_seed_suffixes_the_prefix_and_reaches_the_loader(tmp_path, monkeypatch):
    """Outputs for seed 2027 must not overwrite the seed-2026 run, and the seed
    must be the WEIGHT seed only (contract seed 2026 still governs sampling)."""
    import codenames.cli as cli
    import codenames.data
    import codenames.loop

    captured = {}
    df = pd.DataFrame({"row_id": range(4), "output": ["h"] * 4})
    monkeypatch.setattr(codenames.data, "load_dataset", lambda p: df)
    monkeypatch.setattr(codenames.data, "sample_turns", lambda d, n, seed: d.head(n))

    def fake_loader(random_seed=2026, **kw):
        captured["random_seed"] = random_seed
        return object(), object(), {"prefix": "random_qwen", "chat_template_strategy": "chatml",
                                    "supports_generation": False,
                                    "forward_hidden_states_mode": "causal",
                                    "use_truncation": False, "num_layers": 2,
                                    "hidden_dim": 8, "device": "cpu"}
    monkeypatch.setattr(cli, "_resolve_loader", lambda name: fake_loader)
    monkeypatch.setattr(codenames.loop, "run_extraction",
                        lambda **kw: captured.update(kw) or {})
    args = build_parser().parse_args(["run", "--dataset", "/tmp/d.csv", "--output-dir",
                                      str(tmp_path), "--model", "qwen_random",
                                      "--init-seed", "2027", "--sample-size", "2",
                                      "--skip-sanity-checks"])
    assert cli._cmd_run(args) == 0
    assert captured["random_seed"] == 2027
    assert captured["prefix"] == "random_qwen_s2027"
    assert captured["contract"].random_seed == 2026


def _write_seed_outputs(root, prefix, rho, margin):
    d = root / f"{prefix}_outputs"
    d.mkdir(parents=True)
    pd.DataFrame({"layer": [0, 1, 2], "mean_rho": [0.0, rho, rho * 0.9],
                  "std_rho": [0.2] * 3, "n_boards": [10] * 3}).to_csv(
        d / f"{prefix}_position_confound_by_layer.csv", index=False)
    pd.DataFrame({"layer": [0, 1, 2], "pooling_method": ["mean"] * 3,
                  "condition": ["no_social"] * 3, "mean_margin": [margin] * 3,
                  "adjusted_margin": [margin] * 3, "n_boards": [10] * 3}).to_csv(
        d / f"{prefix}_layer_margins_mean_no_social.csv", index=False)


def test_seed_spread_reports_min_max_over_seeds(tmp_path):
    _write_seed_outputs(tmp_path, "random_qwen", -0.40, 0.001)
    _write_seed_outputs(tmp_path, "random_qwen_s2027", -0.44, 0.002)
    _write_seed_outputs(tmp_path, "random_qwen_s2028", -0.38, 0.000)
    out = random_init_seed_spread(str(tmp_path), base_prefixes=("random_qwen", "random_bert"),
                                  seeds=(2026, 2027, 2028))
    q = out[out["base_prefix"] == "random_qwen"]
    assert set(q["seed"]) == {2026, 2027, 2028}
    assert q["peak_abs_rho"].max() == pytest.approx(0.44)
    assert q["max_margin"].max() == pytest.approx(0.002)
    assert "random_bert" not in set(out["base_prefix"])   # missing runs are skipped, not zeros
    assert {"seed", "prefix", "peak_abs_rho", "peak_rho_layer", "final_rho",
            "max_margin", "final_margin", "n_layers"} <= set(out.columns)
