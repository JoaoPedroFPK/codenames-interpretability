import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.contract import Contract
from codenames.lens.extract import run_lens_extraction
from codenames.lens.raw import RawLens

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


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    return model, tok


def test_extraction_shapes_and_calibration(tiny, tmp_path):
    model, tok = tiny
    df = _toy_df()
    contract = Contract(sample_size=3)
    out = run_lens_extraction(
        model=model, tokenizer=tok, df=df, base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size,
        conditions=("no_social",), device="cpu",
    )
    mm = np.load(out["no_social"]["hidden"], mmap_mode="r")
    L, D = model.config.num_hidden_layers, model.config.hidden_size
    assert mm.shape == (3, L + 1, D) and mm.dtype == np.float16
    idx = pd.read_csv(out["no_social"]["index"])
    assert idx["ok"].all()

    # Readout weights written alongside.
    import os
    assert os.path.exists(str(tmp_path / "tiny_lens_readout_f16.npz"))

    # Calibration identity through the dump: final-layer row of board 0 ->
    # RawLens == a fresh forward's last-token logits (fp16 dump tolerance).
    from codenames.prompts import build_prompt
    row = df.sample(n=3, random_state=contract.random_seed) \
            .reset_index(drop=True).iloc[0]
    prompt, _ = build_prompt(hint=str(row["output"]),
                             candidates=list(row["candidates"]),
                             giver_features={}, use_social_context=False,
                             tokenizer=tok, chat_template_strategy="raw")
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        ref = model(**inputs).logits[0, -1]
    lens = RawLens(model)
    got = lens.logits(torch.tensor(np.asarray(mm[0, L]), dtype=torch.float32))
    assert torch.allclose(got, ref.float(), atol=0.05)


def test_resume_skips_done_boards(tiny, tmp_path):
    model, tok = tiny
    df = _toy_df()
    contract = Contract(sample_size=3)
    kwargs = dict(model=model, tokenizer=tok, df=df, base_dir=str(tmp_path),
                  prefix="tiny", contract=contract, chat_template_strategy="raw",
                  num_layers=model.config.num_hidden_layers,
                  hidden_dim=model.config.hidden_size,
                  conditions=("no_social",), device="cpu")
    out1 = run_lens_extraction(**kwargs)
    mm1 = np.array(np.load(out1["no_social"]["hidden"], mmap_mode="r"))
    out2 = run_lens_extraction(resume=True, **kwargs)     # complete -> no-op
    mm2 = np.array(np.load(out2["no_social"]["hidden"], mmap_mode="r"))
    assert np.array_equal(mm1, mm2)


def test_resume_continues_partial_run(tiny, tmp_path):
    """Simulate a mid-run interruption: manifest says 2/3 boards done."""
    from codenames import checkpoint

    model, tok = tiny
    df = _toy_df()
    contract = Contract(sample_size=3)
    kwargs = dict(model=model, tokenizer=tok, df=df, base_dir=str(tmp_path),
                  prefix="tiny", contract=contract, chat_template_strategy="raw",
                  num_layers=model.config.num_hidden_layers,
                  hidden_dim=model.config.hidden_size,
                  conditions=("no_social",), device="cpu")
    out1 = run_lens_extraction(**kwargs)
    full = np.array(np.load(out1["no_social"]["hidden"], mmap_mode="r"))

    # Rewind: mark only 2 boards committed, zero the third board's rows.
    ckpt_dir = str(tmp_path / "checkpoints")
    checkpoint.write_manifest(ckpt_dir, "tiny_lens", "no_social",
                              n_boards=3, boards_done=2, ckpt_committed=0,
                              complete=False)
    mm = np.lib.format.open_memmap(out1["no_social"]["hidden"], mode="r+")
    mm[2] = 0
    mm.flush()
    del mm

    out2 = run_lens_extraction(resume=True, **kwargs)
    resumed = np.array(np.load(out2["no_social"]["hidden"], mmap_mode="r"))
    assert np.array_equal(resumed, full)          # board 2 recomputed
    idx = pd.read_csv(out2["no_social"]["index"])
    assert len(idx) == 3 and idx["ok"].all()
