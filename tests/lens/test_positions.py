"""Matched-position lens dumps + attention from p_read (pre-submission T3a/T3c).

The answer-channel lens reads p_read; the geometry tier reads the hint and
candidate spans. ``run_position_extraction`` dumps mean-pooled per-layer
states at the hint span and at the target-candidate span, and the attention
mass from p_read onto each role, from ONE joint (prompt + teacher-forced
generation) pass per board -- prompt-position states in a causal decoder do
not depend on the appended generation, so no second pass is needed. It writes
NEW files only and never touches the generating/answer dumps.
"""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.cli import build_parser
from codenames.contract import Contract
from codenames.lens.positions import (
    ATTENTION_COLS,
    POSITION_CHANNELS,
    run_position_extraction,
)

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
pytestmark = pytest.mark.network


def _toy_df():
    rows = [
        {"output": "season", "targets": ["spring"], "black": ["bond"],
         "tan": ["cycle", "point"]},
        {"output": "spy", "targets": ["bond"], "black": ["spring"],
         "tan": ["press", "ghost"]},
        {"output": "time", "targets": ["cycle", "point"], "black": ["ghost"],
         "tan": ["bond", "spring"]},
    ]
    df = pd.DataFrame(rows)
    df["candidates"] = df.apply(
        lambda r: sorted(list(r["targets"]) + list(r["black"]) + list(r["tan"])),
        axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    return df


def _generations(path, df):
    rows = [{"row_id": int(r.row_id),
             "generated_text": f"The word that best matches the hint is {r.targets[0]}.",
             "generated_word": r.targets[0]} for r in df.itertuples()]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY, attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    return model, tok


def _run(tiny, tmp_path, gen=None, **kw):
    model, tok = tiny
    df = _toy_df()
    return df, run_position_extraction(
        model=model, tokenizer=tok, df=df, base_dir=str(tmp_path), prefix="tiny",
        contract=Contract(sample_size=3), chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers, hidden_dim=model.config.hidden_size,
        mode_name="no_social", device="cpu", generation_csv=gen, **kw)


def test_position_channels_are_hint_and_cand_target():
    assert POSITION_CHANNELS == ("hint", "cand_target")


def test_dumps_have_lens_shape_and_index_marks_spans(tiny, tmp_path):
    model, _ = tiny
    df, out = _run(tiny, tmp_path)
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    for ch in POSITION_CHANNELS:
        mm = np.load(out[ch], mmap_mode="r")
        assert mm.shape == (3, L + 1, D) and mm.dtype == np.float16
        assert np.isfinite(np.asarray(mm, dtype=np.float32)).all()
    idx = pd.read_csv(out["index"])
    assert {"board_idx", "row_id", "hint_ok", "cand_target_ok", "target_word",
            "n_targets", "hint_n_tokens", "cand_target_n_tokens",
            "p_star", "p_read"} <= set(idx.columns)
    assert idx["hint_ok"].all() and idx["cand_target_ok"].all()
    assert (idx["hint_n_tokens"] >= 1).all()
    # multi-target turn: the first listed target is used and the count kept
    assert idx["n_targets"].max() == 2


def test_hint_state_is_the_mean_over_the_hint_span(tiny, tmp_path):
    """Pooling rule stated and checked: mean over the span's tokens (the
    geometry tier's `mean` pooling), so the comparison is like for like."""
    from codenames.causal.basis import role_of_each_token, role_positions
    from codenames.prompts import build_prompt
    model, tok = tiny
    df, out = _run(tiny, tmp_path)
    L = model.config.num_hidden_layers
    row = df.sample(n=3, random_state=Contract(sample_size=3).random_seed) \
            .reset_index(drop=True).iloc[0]
    prompt, _ = build_prompt(hint=str(row["output"]), candidates=list(row["candidates"]),
                             giver_features={}, use_social_context=False,
                             tokenizer=tok, chat_template_strategy="raw")
    roles = role_positions(role_of_each_token(
        tok, prompt, hint=str(row["output"]), candidates=list(row["candidates"]),
        clean_target=str(row["targets"][0]), donor_target=""))
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        hs = model(**inputs, output_hidden_states=True).hidden_states
    ref = hs[L][0, roles["hint"]].mean(0).numpy()
    got = np.asarray(np.load(out["hint"], mmap_mode="r")[0, L], dtype=np.float32)
    np.testing.assert_allclose(got, ref, rtol=2e-2, atol=2e-2)


def test_attention_rows_sum_to_one_and_need_a_resolved_p_read(tiny, tmp_path):
    gen = _generations(tmp_path / "gen.csv", _toy_df())
    df, out = _run(tiny, tmp_path, gen=gen)
    att = pd.read_csv(out["attention"])
    assert {"board_idx", "row_id", "layer", *ATTENTION_COLS} <= set(att.columns)
    idx = pd.read_csv(out["index"])
    assert (idx["p_read"] > 0).all(), "generations were supplied; p_read must resolve"
    n_layers = tiny[0].config.num_hidden_layers
    assert len(att) == 3 * n_layers          # one row per board per block
    mass = att[list(ATTENTION_COLS)].sum(axis=1)
    np.testing.assert_allclose(mass, 1.0, atol=1e-3)
    assert (att["to_hint"] >= 0).all()


def test_without_generations_positions_are_dumped_and_attention_is_empty(tiny, tmp_path):
    df, out = _run(tiny, tmp_path, gen=None)
    att = pd.read_csv(out["attention"])
    assert att.empty
    idx = pd.read_csv(out["index"])
    assert (idx["p_read"] == -1).all() and idx["hint_ok"].all()


def test_existing_lens_dumps_are_left_alone(tiny, tmp_path):
    marker = tmp_path / "tiny_lens_hidden_no_social_f16.npy"
    marker.write_bytes(b"do not touch")
    _run(tiny, tmp_path)
    assert marker.read_bytes() == b"do not touch"


def test_resume_is_byte_identical(tiny, tmp_path):
    gen = _generations(tmp_path / "gen.csv", _toy_df())
    a = tmp_path / "a"; b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    model, tok = tiny
    kw = dict(model=model, tokenizer=tok, df=_toy_df(), prefix="tiny",
              contract=Contract(sample_size=3), chat_template_strategy="raw",
              num_layers=model.config.num_hidden_layers, hidden_dim=model.config.hidden_size,
              mode_name="no_social", device="cpu", generation_csv=gen)
    full = run_position_extraction(base_dir=str(a), **kw)
    run_position_extraction(base_dir=str(b), stop_after=2, **kw)
    part = run_position_extraction(base_dir=str(b), resume=True, **kw)
    for ch in POSITION_CHANNELS:
        np.testing.assert_array_equal(np.load(full[ch]), np.load(part[ch]))
    pd.testing.assert_frame_equal(pd.read_csv(full["index"]), pd.read_csv(part["index"]))
    pd.testing.assert_frame_equal(pd.read_csv(full["attention"]), pd.read_csv(part["attention"]))


# --- CLI ------------------------------------------------------------------

def test_lens_extract_accepts_dump_positions_and_attention():
    base = ["lens-extract", "--model", "mistral", "--dataset", "/tmp/d.csv",
            "--output-dir", "/tmp/x"]
    args = build_parser().parse_args(base)
    assert args.dump_positions is None and args.dump_attention is False
    args = build_parser().parse_args(base + ["--dump-positions", "hint,cand_target",
                                             "--dump-attention",
                                             "--generation-csv", "/tmp/g.csv"])
    assert args.dump_positions == "hint,cand_target" and args.dump_attention


def test_lens_apply_and_analyze_accept_position_channels():
    for ch in POSITION_CHANNELS:
        a = build_parser().parse_args(["lens-apply", "--model", "qwen", "--dataset", "/tmp/d.csv",
                                       "--output-dir", "/tmp/x", "--channel", ch])
        assert a.channel == ch
        b = build_parser().parse_args(["lens-analyze", "--channel", ch])
        assert b.channel == ch


# --- seam: dump -> apply -> analyze on the hint channel ---------------------

def test_hint_channel_scores_and_analysis(tiny, tmp_path):
    from codenames.lens.analysis import run_analysis
    from codenames.lens.apply import compute_scores, load_readout, save_scores
    from codenames.lens.raw import dump_readout_weights

    model, tok = tiny
    root = tmp_path / "output"
    base = root / "tiny_outputs"
    base.mkdir(parents=True)
    df = _toy_df()
    out = run_position_extraction(
        model=model, tokenizer=tok, df=df, base_dir=str(base), prefix="tiny",
        contract=Contract(sample_size=3), chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers, hidden_dim=model.config.hidden_size,
        mode_name="no_social", device="cpu")
    readout_path = base / "tiny_lens_readout_f16.npz"
    dump_readout_weights(model, str(readout_path))
    index = pd.read_csv(out["index"])
    index["ok"] = index["ok"].astype(bool) & index["hint_ok"].astype(bool)
    sample = df.sample(n=3, random_state=2026).reset_index(drop=True)
    scores = compute_scores(out["hint"], index, sample, tok, load_readout(str(readout_path)), "raw")
    assert set(scores["layer"]) == set(range(model.config.num_hidden_layers + 1))
    save_scores([scores], str(base), "tiny", "hint_no_social")

    run_analysis(output_root=str(root), models=("tiny",), random_model=None,
                 mode="no_social", channel="hint", out_dir=str(root / "lens_analysis"),
                 n_boot=20, seed=1)
    curves = pd.read_csv(root / "lens_analysis" / "lens_curves_hint_no_social.csv")
    assert {"lens", "layer", "top1", "mrr", "ci_lo", "ci_hi", "model"} <= set(curves.columns)
    calib = pd.read_csv(root / "lens_analysis" / "lens_calibration_hint_no_social.csv")
    assert bool(calib["gate_applicable"].iloc[0]) is False
    assert bool(calib["passes_gate"].iloc[0]) is True     # not blocked: gate n/a
