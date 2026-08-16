"""Full role x layer real-patch grid with matched nulls (pre-submission T1/T3b).

The confirmatory patch stage patches only the loci the attribution scan
surfaced (one role per layer under ``--per-layer``), so answer positions were
never patched at L4-15 and the hint span never beyond L9. ``--grid`` patches
EVERY requested role at EVERY layer, independent of the scan, and runs the
random-site baseline (spec §3.3.1) on the same turns so the paired contrast of
§3.4 is available per cell.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.causal import runner
from codenames.causal.basis import ROLES
from codenames.causal.grid import (
    DERIVED_ROLES,
    GRID_ROLES,
    grid_role_positions,
    parse_layers,
    parse_roles,
    random_site_positions,
    run_grid_stage,
)
from codenames.causal.patch import run_patch, run_patch_many
from codenames.cli import build_parser
from codenames.lens.readout import build_token_table

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _ragged_frame():
    return pd.DataFrame([
        {"row_id": 1, "output": "water", "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea"]},
        {"row_id": 2, "output": "rocket", "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship", "castle",
                                         "engine", "telescope"]},
        {"row_id": 3, "output": "harbour", "targets": ["ship"], "black": ["sea"],
         "tan": ["moon"], "candidates": ["moon", "sea", "ship", "anchor"]},
    ])


def _generations(path, row_ids):
    """Scaffolded generations so p_read sits inside the teacher-forced span."""
    rows = [{"row_id": i, "generated_text": "The word that best matches the hint is moon.",
             "generated_word": "moon"} for i in row_ids]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _common(tiny, df=None, **kw):
    model, tok = tiny
    return dict(model=model, tokenizer=tok,
                df_sample=_frame_or(df), chat_template_strategy="raw",
                mode="no_social", seed=2026, **kw)


def _frame_or(df):
    return _ragged_frame() if df is None else df


# --- role parsing -----------------------------------------------------------

def test_roles_all_is_the_pre_registered_grid_set():
    assert parse_roles("all") == GRID_ROLES
    assert set(GRID_ROLES) >= {"hint", "cand_target", "cand_donor", "final", "generation"}
    assert set(DERIVED_ROLES) == {"p_read", "scaffold"}
    assert set(GRID_ROLES) >= set(DERIVED_ROLES)


def test_roles_accepts_any_basis_or_derived_role_and_rejects_others():
    assert parse_roles("hint,cand_other,p_read") == ("hint", "cand_other", "p_read")
    with pytest.raises(ValueError, match="bogus"):
        parse_roles("hint,bogus")
    with pytest.raises(ValueError, match="random_site"):
        parse_roles("random_site")   # reserved for the null, never a treatment


def test_layers_all_or_explicit_list():
    assert parse_layers("all", 5) == [0, 1, 2, 3, 4]
    assert parse_layers("0,2,4", 5) == [0, 2, 4]
    with pytest.raises(ValueError):
        parse_layers("0,9", 5)


# --- derived roles ----------------------------------------------------------

def test_derived_roles_split_the_answer_span_around_p_read():
    by_role = {"hint": [3, 4], "final": [10], "generation": [11, 12, 13, 14, 15]}
    out = grid_role_positions(by_role, p_read=14, roles=("hint", "generation", "p_read", "scaffold"))
    assert out["p_read"] == [14]
    # scaffold = teacher-forced tokens BEFORE the readout; tokens after it
    # cannot reach a causal readout and must not pad the token count.
    assert out["scaffold"] == [11, 12, 13]
    assert out["generation"] == [11, 12, 13, 14, 15]
    assert out["hint"] == [3, 4]


def test_word_first_turn_has_p_read_at_final_and_no_scaffold():
    by_role = {"hint": [3, 4], "final": [10], "generation": [11, 12]}
    out = grid_role_positions(by_role, p_read=10, roles=("final", "p_read", "scaffold"))
    assert out["p_read"] == [10]
    assert "scaffold" not in out          # absent -> recorded as NaN downstream


# --- random-site null -------------------------------------------------------

def test_random_site_matches_count_and_avoids_tested_roles():
    by_role = {"prefix": [0, 1, 2], "hint": [3, 4], "post_hint": [5, 6],
               "list_scaffold": [7, 8, 9], "cand_target": [10], "cand_other": [11, 12, 13],
               "question": [14, 15, 16], "final": [17], "generation": [18, 19, 20]}
    tested = ("hint", "cand_target", "final", "generation")
    rng = np.random.default_rng(0)
    picked = random_site_positions(by_role, p_read=19, tested_roles=tested, k=2, rng=rng)
    assert len(picked) == 2
    forbidden = {3, 4, 10, 17, 18, 19, 20}
    assert not set(picked) & forbidden
    assert all(p < 19 for p in picked)


def test_random_site_is_deterministic_for_a_seed():
    by_role = {"prefix": list(range(0, 8)), "hint": [8], "cand_other": list(range(9, 20)),
               "final": [20]}
    a = random_site_positions(by_role, p_read=20, tested_roles=("hint",), k=3,
                              rng=np.random.default_rng(2026))
    b = random_site_positions(by_role, p_read=20, tested_roles=("hint",), k=3,
                              rng=np.random.default_rng(2026))
    assert a == b


# --- batched patching -------------------------------------------------------

def test_run_patch_many_matches_sequential_run_patch(tiny):
    model, tok = tiny
    clean, corrupt = "Hint water. Words: sea ship. Answer:", "Hint rocket. Words: sea ship. Answer:"
    ids = tok(clean, return_tensors="pt")
    with torch.no_grad():
        out = model(**ids, output_hidden_states=True)
    cache = [h.detach().clone() for h in out.hidden_states]
    table = build_token_table(tok, ["sea", "ship"])
    n_pos = ids["input_ids"].shape[1]
    site_sets = [[(1, 0)], [(2, 1), (2, 2)], [(len(cache) - 1, n_pos - 1)], []]
    kw = dict(model=model, tokenizer=tok, clean_cache=cache, corrupt_prompt=corrupt,
              readout_table=table, clean_target="sea", donor_target="ship",
              p_star=-1, device="cpu")
    seq = [run_patch(sites=s, clean_prompt=clean, **kw) for s in site_sets]
    # ld_clean/ld_corrupt exactly as the stage supplies them
    from codenames.causal.metrics import logit_difference
    ld_clean = logit_difference(out.logits[0, -1].detach().numpy(), table, "sea", "ship")
    with torch.no_grad():
        ld_corrupt = logit_difference(
            model(**tok(corrupt, return_tensors="pt")).logits[0, -1].numpy(), table, "sea", "ship")
    many = run_patch_many(site_sets=site_sets, ld_clean=ld_clean, ld_corrupt=ld_corrupt,
                          batch_size=3, **kw)
    assert len(many) == len(site_sets)
    np.testing.assert_allclose(many, seq, atol=1e-4)


# --- the grid stage ---------------------------------------------------------

def test_grid_has_one_row_per_layer_role_width_turn(tiny, tmp_path):
    df = _ragged_frame()
    gen = _generations(tmp_path / "gen.csv", df["row_id"])
    grid, nulls = run_grid_stage(**_common(tiny, df), roles=("hint", "cand_target",
                                                            "generation", "p_read", "scaffold"),
                                 layers="all", window_widths=(1,), generation_csv=gen,
                                 batch_size=4)
    n_layers = tiny[0].config.num_hidden_layers + 1
    assert set(grid.columns) >= {"layer", "role", "width", "row_id", "effect", "n_positions"}
    assert set(grid["role"]) == {"hint", "cand_target", "generation", "p_read", "scaffold"}
    assert set(grid["layer"]) == set(range(n_layers))
    per_turn = grid.groupby("row_id").size()
    assert per_turn.nunique() == 1 and per_turn.iloc[0] == n_layers * 5
    # cand_target is present on every turn here; every one is a real patch
    assert grid[grid["role"] == "cand_target"]["effect"].notna().all()


def test_absent_role_is_nan_not_dropped(tiny, tmp_path):
    """Word-first turns have no scaffold; the row exists with NaN."""
    df = _ragged_frame()
    # no generations -> p_read at the generating position -> scaffold empty
    grid, _ = run_grid_stage(**_common(tiny, df), roles=("scaffold", "p_read"),
                             layers="0,1", window_widths=(1,), batch_size=2)
    sc = grid[grid["role"] == "scaffold"]
    assert grid["row_id"].nunique() >= 2      # the pair table drops unpaired turns
    assert len(sc) == 2 * grid["row_id"].nunique()
    assert sc["effect"].isna().all() and (sc["n_positions"] == 0).all()
    pr = grid[grid["role"] == "p_read"]
    assert pr["effect"].notna().all() and (pr["n_positions"] == 1).all()


def test_nulls_are_matched_random_sites_at_every_layer(tiny, tmp_path):
    df = _ragged_frame()
    gen = _generations(tmp_path / "gen.csv", df["row_id"])
    grid, nulls = run_grid_stage(**_common(tiny, df), roles=("hint", "generation"),
                                 layers="all", window_widths=(1,), generation_csv=gen,
                                 batch_size=4)
    assert (nulls["role"] == "random_site").all()
    assert set(nulls["layer"]) == set(grid["layer"])
    # one null per (turn, layer, width, distinct token count among tested roles)
    for rid, g in grid.groupby("row_id"):
        counts = set(g.loc[g["n_positions"] > 0, "n_positions"])
        got = set(nulls.loc[nulls["row_id"] == rid, "matched_n"])
        assert got == counts, (rid, got, counts)
    assert "matched_roles" in nulls.columns
    assert nulls["effect"].notna().any()


def test_grid_is_independent_of_the_scan_loci(tiny, tmp_path):
    """No loci file is needed: the grid is defined by roles x layers alone."""
    df = _ragged_frame()
    grid, _ = run_grid_stage(**_common(tiny, df), roles=("hint",), layers="1,2",
                             window_widths=(1, 3), batch_size=4)
    assert set(grid["layer"]) == {1, 2} and set(grid["width"]) == {1, 3}


# --- CLI wiring -------------------------------------------------------------

def test_patch_parser_accepts_grid_flags():
    args = build_parser().parse_args([
        "causal-patch", "--model", "mistral", "--output-dir", "/tmp/x",
        "--dataset", "/tmp/d.csv", "--grid", "--roles", "all", "--layers", "all",
        "--batch-size", "8", "--window-widths", "1"])
    assert args.grid and args.roles == "all" and args.layers == "all"
    assert args.batch_size == 8
    default = build_parser().parse_args([
        "causal-patch", "--model", "mistral", "--output-dir", "/tmp/x",
        "--dataset", "/tmp/d.csv"])
    assert not default.grid and default.batch_size == 1


def test_analyze_parser_accepts_grid_flag():
    args = build_parser().parse_args(["causal-analyze", "--model", "mistral",
                                      "--output-dir", "/tmp/x", "--grid"])
    assert args.grid


@pytest.fixture
def wired(monkeypatch, tiny):
    model, tok = tiny
    monkeypatch.setattr(runner, "_load_model", lambda key: (
        model, tok, {"chat_template_strategy": "raw",
                     "num_layers": model.config.num_hidden_layers,
                     "hidden_dim": model.config.hidden_size}))
    return model, tok


@pytest.fixture
def dataset(tmp_path_factory):
    boards = [
        (["MOON"], ["SEA"], ["SHIP", "ENGINE", "TELESCOPE", "ANCHOR"]),
        (["SHIP", "ANCHOR"], ["MOON"], ["SEA"]),
        (["GLASS"], ["PLATE"], ["SCHOOL", "NIGHT", "STRING", "ROBIN", "MOON", "SEA"]),
        (["NIGHT"], ["MOON"], ["STRING", "SEA", "SHIP"]),
    ]
    hints = ["orbit", "harbour", "fragile", "dark"]
    rows = [{"output": h, "targets": str(t), "black": str(b), "tan": str(n)}
            for (t, b, n), h in zip(boards, hints)]
    path = tmp_path_factory.mktemp("data") / "clue_generation.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _args(argv):
    return build_parser().parse_args(argv)


def test_grid_cli_writes_grid_and_nulls_without_touching_effects(wired, dataset, tmp_path):
    out = tmp_path / "causal"
    out.mkdir()
    common = ["--model", "mistral", "--output-dir", str(out), "--dataset", dataset,
              "--condition", "no_social", "--pilot-n", "0"]
    rc = runner.cmd_patch(_args(["causal-patch", *common, "--grid", "--roles", "hint,final",
                                 "--layers", "all", "--window-widths", "1",
                                 "--batch-size", "4"]))
    assert rc == 0
    assert (out / "mistral_causal_effects_grid_counterfactual_no_social.parquet").exists()
    assert (out / "mistral_causal_nulls_counterfactual_no_social.parquet").exists()
    assert not (out / "mistral_causal_effects_counterfactual_no_social.parquet").exists()
    grid = pd.read_parquet(out / "mistral_causal_effects_grid_counterfactual_no_social.parquet")
    assert set(grid["role"]) == {"hint", "final"}
    nulls = pd.read_parquet(out / "mistral_causal_nulls_counterfactual_no_social.parquet")
    assert (nulls["role"] == "random_site").all()

    # offline analysis of the grid: real cluster-bootstrap p, BH, paired contrast
    rc = runner.cmd_analyze(_args(["causal-analyze", "--model", "mistral",
                                   "--output-dir", str(out), "--condition", "no_social",
                                   "--grid", "--n-boot", "50", "--n-perm", "20"]))
    assert rc == 0
    claims = pd.read_csv(out / "mistral_causal_grid_claims_counterfactual_no_social.csv")
    assert {"layer", "role", "width", "mean_effect", "ci_low", "ci_high", "n_turns",
            "p_boot", "survives_fdr", "random_site_mean", "paired_diff",
            "paired_ci_low", "paired_ci_high", "beats_random_site",
            "perm_threshold", "exceeds_perm_threshold"} <= set(claims.columns)
    assert set(claims["role"]) == {"hint", "final"}


def test_grid_resume_is_byte_identical(wired, dataset, tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    common = ["--model", "mistral", "--output-dir", str(out), "--dataset", dataset,
              "--condition", "no_social", "--pilot-n", "0", "--grid", "--roles", "hint",
              "--layers", "0,1", "--window-widths", "1", "--batch-size", "2"]
    runner.cmd_patch(_args(["causal-patch", *common]))
    first = pd.read_parquet(out / "mistral_causal_effects_grid_counterfactual_no_social.parquet")
    first_nulls = pd.read_parquet(out / "mistral_causal_nulls_counterfactual_no_social.parquet")
    runner.cmd_patch(_args(["causal-patch", *common, "--resume"]))
    again = pd.read_parquet(out / "mistral_causal_effects_grid_counterfactual_no_social.parquet")
    again_nulls = pd.read_parquet(out / "mistral_causal_nulls_counterfactual_no_social.parquet")
    key = ["row_id", "layer", "role", "width"]
    pd.testing.assert_frame_equal(first.sort_values(key).reset_index(drop=True),
                                  again.sort_values(key).reset_index(drop=True))
    key_n = ["row_id", "layer", "width", "n_positions"]
    pd.testing.assert_frame_equal(first_nulls.sort_values(key_n).reset_index(drop=True),
                                  again_nulls.sort_values(key_n).reset_index(drop=True))
