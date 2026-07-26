"""End-to-end seam test: extract -> tune -> apply -> analyze on the tiny
fixture. Verifies the on-disk contracts between stages (npz keys, memmap
layout, index columns, scores schema) that the unit tests cover only in
isolation.
"""

import os

import numpy as np
import pandas as pd
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.contract import Contract
from codenames.lens.analysis import classify_trajectory, layer_curves
from codenames.lens.apply import compute_scores, load_readout, save_scores
from codenames.lens.extract import run_lens_extraction
from codenames.lens.tuned import TunedLens, TunedLensConfig, train_tuned_lens

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
pytestmark = pytest.mark.network


def _toy_df():
    rows = [
        {"output": "season", "targets": ["spring"], "black": ["bond"],
         "tan": ["cycle", "point"]},
        {"output": "spy", "targets": ["bond"], "black": ["spring"],
         "tan": ["press", "ghost"]},
        {"output": "time", "targets": ["cycle"], "black": ["ghost"],
         "tan": ["bond", "spring"]},
    ]
    df = pd.DataFrame(rows)
    df["candidates"] = df.apply(
        lambda r: sorted(list(r["targets"]) + list(r["black"]) + list(r["tan"])),
        axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    return df


def test_full_pipeline_on_tiny_model(tmp_path):
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    df = _toy_df()
    contract = Contract(sample_size=3)
    L = model.config.num_hidden_layers

    # 1. Extract
    out = run_lens_extraction(
        model=model, tokenizer=tok, df=df, base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=L, hidden_dim=model.config.hidden_size,
        conditions=("no_social",), device="cpu")

    # 2. Tune (tiny budget)
    lens = train_tuned_lens(
        model, tok, ["the quick brown fox jumps over the lazy dog. " * 60],
        TunedLensConfig(seq_len=16, n_steps=4, seqs_per_step=2,
                        positions_per_seq=4))
    tpath = str(tmp_path / "tiny_lens_translators.npz")
    lens.save(tpath)

    # 3. Apply (raw + tuned) through the on-disk artefacts only
    df_sample = df.sample(n=3, random_state=contract.random_seed) \
                  .copy().reset_index(drop=True)
    readout = load_readout(str(tmp_path / "tiny_lens_readout_f16.npz"))
    frames = [
        compute_scores(out["no_social"]["hidden"], out["no_social"]["index"],
                       df_sample, tok, readout, "raw"),
        compute_scores(out["no_social"]["hidden"], out["no_social"]["index"],
                       df_sample, tok, readout, "tuned",
                       translators=TunedLens.load(tpath)),
    ]
    spath = save_scores(frames, str(tmp_path), "tiny", "no_social")
    scores = pd.read_parquet(spath)
    assert set(scores["lens"]) == {"raw", "tuned"}
    # 3 turns x (L+1) layers x 4 candidates x 2 lenses
    assert len(scores) == 3 * (L + 1) * 4 * 2

    # 4. Analyze
    curves = layer_curves(scores, n_boot=100, seed=1)
    assert len(curves) == 2 * (L + 1)
    cls = classify_trajectory(curves, "raw", n_states=L + 1)
    assert cls["label"] in {"plateau", "dip_then_recover", "late_rise",
                            "unclassified"}

    # Raw lens at the FINAL layer must reproduce the model's own next-token
    # preference: candidate-restricted argmax of a real forward equals the
    # stored rank-1 word (calibration identity end to end).
    import torch
    from codenames.lens.readout import build_token_table, score_candidates
    from codenames.prompts import build_prompt
    row = df_sample.iloc[0]
    prompt, _ = build_prompt(hint=str(row["output"]),
                             candidates=list(row["candidates"]),
                             giver_features={}, use_social_context=False,
                             tokenizer=tok, chat_template_strategy="raw")
    with torch.no_grad():
        logits = model(**tok(prompt, return_tensors="pt")).logits[0, -1]
    table = build_token_table(tok, list(row["candidates"]))
    direct = score_candidates(logits.float().numpy(), table)
    expected = max(direct, key=direct.get)
    got = scores[(scores["lens"] == "raw") & (scores["layer"] == L)
                 & (scores["row_id"] == int(row["row_id"]))
                 & (scores["rank"] == 1)]["word"].iloc[0]
    assert got == expected
