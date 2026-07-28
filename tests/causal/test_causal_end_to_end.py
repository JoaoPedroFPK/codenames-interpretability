"""End-to-end walk of scan -> patch -> analyze through the real CLI.

The unit tests exercise each stage in isolation; this covers the wiring that
sits between them, which is where the first real run actually broke. The
fixtures deliberately reproduce the corpus's awkward properties rather than a
tidy synthetic board:

* **ragged boards** (11-21 candidates, as CULTURAL CODES has) so prompts
  tokenise to different lengths -- 49 distinct lengths across the real 7,703
  turns, which is why a grid indexed by absolute token position could not be
  aggregated;
* **candidate words that contain one another** (ICE / ICE CREAM, NEW YORK /
  YORK), which a bare substring search mislabels silently;
* **scaffolded generations** ("The word that best matches the hint is X"),
  which is 87% of Mistral's turns and the reason the answer position p* is the
  primary readout rather than the generating position.
"""

import numpy as np
import pandas as pd
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal import runner
from codenames.cli import build_parser

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

BOARDS = [
    (["ICE CREAM", "ICE"], ["NEW YORK"], ["YORK", "MOON", "SEA", "SHIP", "CASTLE"]),
    (["MOON"], ["SEA"], ["SHIP", "ENGINE", "TELESCOPE", "ANCHOR", "ROBIN",
                         "PLATE", "SCHOOL", "GLASS"]),
    (["SHIP", "ANCHOR"], ["MOON"], ["SEA"]),
    (["GLASS"], ["PLATE"], ["SCHOOL", "NIGHT", "STRING", "ROBIN", "CASTLE",
                            "ENGINE", "MOON", "SEA", "SHIP", "ICE"]),
    (["NIGHT"], ["MOON"], ["STRING", "SEA", "SHIP"]),
    (["ENGINE", "TRAIN"], ["SHIP"], ["SEA", "MOON", "CASTLE", "GLASS"]),
]
HINTS = ["frozen", "orbit", "harbour", "fragile", "dark", "steam"]


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rows = []
    for (targets, black, tan), hint in zip(BOARDS, HINTS):
        rows.append({"output": hint, "targets": str(targets),
                     "black": str(black), "tan": str(tan)})
    path = tmp_path_factory.mktemp("data") / "clue_generation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture
def wired(monkeypatch, tiny):
    model, tok = tiny
    monkeypatch.setattr(runner, "_load_model", lambda key: (
        model, tok, {"chat_template_strategy": "mistral_inst",
                     "num_layers": model.config.num_hidden_layers,
                     "hidden_dim": model.config.hidden_size}))
    return model, tok


def _args(argv):
    return build_parser().parse_args(argv)


def _write_generations(base, prefix, condition, n):
    """Scaffolded generations in the recorded schema (Mistral's 87% format)."""
    rows = [{"row_id": i,
             "generated_text": f"The word that best matches the hint is MOON.",
             "generated_word": "MOON"} for i in range(n)]
    path = base / f"{prefix}_generation_{condition}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_prompts_in_the_fixture_really_are_ragged(tiny, dataset):
    """Guard on the fixture itself: if every prompt came out the same length
    this file would silently stop testing the thing it exists to test."""
    from codenames.data import load_dataset
    from codenames.prompts import build_prompt
    _, tok = tiny
    df = load_dataset(dataset)
    lengths = set()
    for row in df.itertuples():
        text, _ = build_prompt(hint=str(row.output), candidates=list(row.candidates),
                               giver_features={}, use_social_context=False,
                               tokenizer=tok, chat_template_strategy="mistral_inst")
        lengths.add(len(tok(text)["input_ids"]))
    assert len(lengths) > 1, f"fixture prompts are all length {lengths}"


def test_scan_then_patch_then_analyze(wired, dataset, tmp_path):
    """The whole chain the runner executes, on data shaped like the corpus."""
    from codenames.causal.basis import ROLES

    out = tmp_path / "causal"
    out.mkdir()
    common = ["--model", "mistral", "--output-dir", str(out),
              "--dataset", dataset, "--condition", "no_social"]

    assert runner.cmd_scan(_args(["causal-scan", *common, "--per-layer"])) == 0
    grid = np.load(out / "mistral_causal_scan_counterfactual_no_social.npy")
    assert grid.shape[1] == len(ROLES), "grid must be on the role basis"
    loci = pd.read_csv(out / "mistral_causal_scan_counterfactual_no_social_loci.csv")
    assert set(loci["role"]) <= set(ROLES)
    assert len(loci) == grid.shape[0], "--per-layer gives one locus per layer"

    assert runner.cmd_patch(_args(["causal-patch", *common,
                                   "--window-widths", "1,3"])) == 0
    effects = pd.read_parquet(
        out / "mistral_causal_effects_counterfactual_no_social.parquet")
    assert {"layer", "role", "width", "row_id", "effect"} <= set(effects.columns)
    assert effects["effect"].notna().any()
    assert set(effects["width"]) == {1, 3}

    assert runner.cmd_analyze(_args([
        "causal-analyze", "--model", "mistral", "--output-dir", str(out),
        "--condition", "no_social", "--n-boot", "50"])) == 0
    claims = pd.read_csv(out / "mistral_causal_claims_no_social.csv")
    assert "role" in claims.columns and len(claims) > 0


def test_the_chain_uses_p_star_when_generations_are_present(wired, dataset, tmp_path):
    """The conventional generation path is picked up without a flag, and it
    changes the readout -- otherwise §4.1's correction is inert in the stages
    even though it is implemented in the pilot."""
    from codenames.data import load_dataset

    n = len(load_dataset(dataset))
    common = ["--model", "mistral", "--dataset", dataset,
              "--condition", "no_social", "--per-layer"]

    bare = tmp_path / "bare"
    bare.mkdir()
    runner.cmd_scan(_args(["causal-scan", *common, "--output-dir", str(bare)]))
    without = np.load(bare / "mistral_causal_scan_counterfactual_no_social.npy")

    withgen = tmp_path / "withgen"
    withgen.mkdir()
    _write_generations(withgen, "mistral", "no_social", n)
    runner.cmd_scan(_args(["causal-scan", *common, "--output-dir", str(withgen)]))
    with_p = np.load(withgen / "mistral_causal_scan_counterfactual_no_social.npy")

    assert not np.allclose(np.nan_to_num(without), np.nan_to_num(with_p)), \
        "generations present but the readout did not move to p*"


def test_patch_resume_is_byte_identical_through_the_cli(wired, dataset, tmp_path):
    """--resume is the project's standing byte-identity guarantee, and patch is
    ~94% of the compute budget, so an interrupted A100 session must continue
    into exactly the run it would have produced uninterrupted."""
    out = tmp_path / "r"
    out.mkdir()
    common = ["--model", "mistral", "--output-dir", str(out),
              "--dataset", dataset, "--condition", "no_social"]
    runner.cmd_scan(_args(["causal-scan", *common, "--per-layer"]))
    patch_args = ["causal-patch", *common, "--window-widths", "1"]
    runner.cmd_patch(_args(patch_args))
    first = pd.read_parquet(
        out / "mistral_causal_effects_counterfactual_no_social.parquet")

    runner.cmd_patch(_args([*patch_args, "--resume"]))
    resumed = pd.read_parquet(
        out / "mistral_causal_effects_counterfactual_no_social.parquet")

    key = ["row_id", "layer", "role", "width"]
    pd.testing.assert_frame_equal(
        first.sort_values(key).reset_index(drop=True),
        resumed.sort_values(key).reset_index(drop=True),
    )
