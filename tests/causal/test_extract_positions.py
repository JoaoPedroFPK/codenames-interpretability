import dataclasses

import numpy as np
import pandas as pd
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from codenames.contract import CONTRACT_V1
from codenames.lens.extract import run_lens_extraction

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tiny():
    model = AutoModelForCausalLM.from_pretrained(TINY)
    model.eval()
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _frame():
    """Real schema: hint lives in `output`; candidates is a list."""
    return pd.DataFrame([
        {"row_id": 0, "output": "ocean", "targets": ["sea"], "black": ["moon"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
        {"row_id": 1, "output": "rocket", "targets": ["moon"], "black": ["sea"],
         "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
    ])


def _generation_csv(tmp_path, rows):
    path = tmp_path / "gen.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def _run(tiny, tmp_path, gen_csv, n=2):
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=n)
    return run_lens_extraction(
        model=model, tokenizer=tok, df=_frame().head(n), base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",),
        device="cpu", dump_answer_position=True, generation_csv=gen_csv,
    )


def test_answer_position_array_is_written_and_shaped(tiny, tmp_path):
    model, _ = tiny
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea is the answer", "generated_word": "sea"},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    out = _run(tiny, tmp_path, gen)
    arr = np.load(out["no_social"]["answer_hidden"], mmap_mode="r")
    assert arr.shape == (2, model.config.num_hidden_layers + 1, model.config.hidden_size)


def test_unparseable_generation_is_recorded_as_missing_not_zero(tiny, tmp_path):
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "no candidate here", "generated_word": None},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    out = _run(tiny, tmp_path, gen)
    index = pd.read_csv(out["no_social"]["answer_index"])
    row0 = index[index.row_id == 0].iloc[0]
    assert bool(row0["p_star_missing"]) is True
    arr = np.load(out["no_social"]["answer_hidden"], mmap_mode="r")
    # Missing must be NaN, never a silent zero vector.
    assert np.isnan(np.asarray(arr[0])).all()


def test_word_first_flag_is_recorded(tiny, tmp_path):
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea is the answer", "generated_word": "sea"},
        {"row_id": 1, "generated_text": "The best match is moon", "generated_word": "moon"},
    ])
    out = _run(tiny, tmp_path, gen)
    index = pd.read_csv(out["no_social"]["answer_index"]).set_index("row_id")
    assert bool(index.loc[0, "word_first"]) is True
    assert bool(index.loc[1, "word_first"]) is False


def test_generating_position_dump_is_unchanged_by_the_extension(tiny, tmp_path):
    """The p* addition must not perturb the existing lens dump."""
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    base = dict(
        model=model, tokenizer=tok, df=_frame(), prefix="tiny", contract=contract,
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",), device="cpu",
    )
    without = run_lens_extraction(base_dir=str(tmp_path / "a"), **base)
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea is the answer", "generated_word": "sea"},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    with_ps = run_lens_extraction(
        base_dir=str(tmp_path / "b"), dump_answer_position=True,
        generation_csv=gen, **base
    )
    np.testing.assert_array_equal(
        np.load(without["no_social"]["hidden"]), np.load(with_ps["no_social"]["hidden"])
    )


def test_p_star_is_not_derived_from_prompt_token_count(tiny, tmp_path):
    """Guards the Task 2 finding: p* comes from the JOINT tokenization."""
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea is the answer", "generated_word": "sea"},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    out = _run(tiny, tmp_path, gen)
    index = pd.read_csv(out["no_social"]["answer_index"]).set_index("row_id")
    lens_index = pd.read_csv(out["no_social"]["index"]).set_index("row_id")
    # Word-first turn: p* must land before the prompt's own token count, because
    # the prompt's trailing whitespace merges with the first generated token.
    assert index.loc[0, "p_star"] < lens_index.loc[0, "prompt_token_count"]
