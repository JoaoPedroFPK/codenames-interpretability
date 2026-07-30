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


def _run_candspan(tiny, tmp_path, row_ids):
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    return run_lens_extraction(
        model=model, tokenizer=tok, df=_frame(), base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",),
        device="cpu", candidate_span_row_ids=row_ids,
    )


def test_candspan_npz_written_only_for_requested_rows(tiny, tmp_path):
    """causal_spec.md §12.4: the dump is scoped to the causal subsample."""
    out = _run_candspan(tiny, tmp_path, {1})
    with np.load(out["no_social"]["candspan"]) as npz:
        keys = sorted(npz.files)
    assert keys == ["r1__moon", "r1__sea", "r1__ship"]
    index = pd.read_csv(out["no_social"]["candspan_index"])
    assert set(index["row_id"]) == {1}
    assert sorted(index["candidate"]) == ["moon", "sea", "ship"]
    assert index["ok"].all()


def test_candspan_states_match_a_fresh_forward(tiny, tmp_path):
    """Dumped span states must be the model's own hidden states, fp16-cast."""
    import torch

    from codenames.prompts import build_prompt

    model, tok = tiny
    out = _run_candspan(tiny, tmp_path, {1})
    index = pd.read_csv(out["no_social"]["candspan_index"]).set_index("candidate")
    row = _frame().iloc[1]
    prompt, _ = build_prompt(
        hint=str(row["output"]), candidates=list(row["candidates"]),
        giver_features={}, use_social_context=False, tokenizer=tok,
        chat_template_strategy="raw")
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        ref = model(**inputs, output_hidden_states=True, return_dict=True)
    s = int(index.loc["sea", "token_start"])
    e = int(index.loc["sea", "token_end"])
    expected = np.stack([
        ref.hidden_states[layer][0, s:e].float().numpy().astype(np.float16)
        for layer in range(model.config.num_hidden_layers + 1)
    ])
    with np.load(out["no_social"]["candspan"]) as npz:
        got = npz["r1__sea"]
    assert got.dtype == np.float16
    assert got.shape == expected.shape          # (L+1, span_len, hidden)
    np.testing.assert_array_equal(got, expected)


def test_candspan_token_bounds_decode_to_the_candidate(tiny, tmp_path):
    from codenames.prompts import build_prompt

    _, tok = tiny
    out = _run_candspan(tiny, tmp_path, {0})
    index = pd.read_csv(out["no_social"]["candspan_index"]).set_index("candidate")
    row = _frame().iloc[0]
    prompt, _ = build_prompt(
        hint=str(row["output"]), candidates=list(row["candidates"]),
        giver_features={}, use_social_context=False, tokenizer=tok,
        chat_template_strategy="raw")
    ids = tok(prompt)["input_ids"]
    for word in row["candidates"]:
        s = int(index.loc[word, "token_start"])
        e = int(index.loc[word, "token_end"])
        assert word in tok.decode(ids[s:e]).lower()


def test_main_dump_unchanged_by_candspan_extension(tiny, tmp_path):
    """The candidate-span addition must not perturb the existing lens dump."""
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    base = dict(
        model=model, tokenizer=tok, df=_frame(), prefix="tiny", contract=contract,
        chat_template_strategy="raw", num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",), device="cpu",
    )
    without = run_lens_extraction(base_dir=str(tmp_path / "a"), **base)
    with_spans = run_lens_extraction(
        base_dir=str(tmp_path / "b"), candidate_span_row_ids={0, 1}, **base)
    np.testing.assert_array_equal(
        np.load(without["no_social"]["hidden"]),
        np.load(with_spans["no_social"]["hidden"]))


def _rewind_manifest(tmp_path, boards_done):
    from codenames import checkpoint
    checkpoint.write_manifest(
        str(tmp_path / "checkpoints"), "tiny_lens", "no_social",
        n_boards=2, boards_done=boards_done, ckpt_committed=0, complete=False)


def test_answer_dump_survives_resume(tiny, tmp_path):
    """Resume must not wipe already-computed p* states (w+ reopen bug)."""
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea is the answer", "generated_word": "sea"},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    out1 = _run(tiny, tmp_path, gen)
    full = np.array(np.load(out1["no_social"]["answer_hidden"]))
    assert not np.isnan(full).all()

    _rewind_manifest(tmp_path, 1)
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    out2 = run_lens_extraction(
        model=model, tokenizer=tok, df=_frame(), base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",),
        device="cpu", dump_answer_position=True, generation_csv=gen,
        resume=True,
    )
    resumed = np.array(np.load(out2["no_social"]["answer_hidden"]))
    np.testing.assert_array_equal(resumed, full)   # board 0 kept, board 1 redone
    index = pd.read_csv(out2["no_social"]["answer_index"])
    assert sorted(index["row_id"]) == [0, 1]       # index keeps both rows


def test_candspan_dump_survives_resume(tiny, tmp_path):
    out1 = _run_candspan(tiny, tmp_path, {0, 1})
    with np.load(out1["no_social"]["candspan"]) as npz:
        keys1 = sorted(npz.files)

    _rewind_manifest(tmp_path, 1)
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    out2 = run_lens_extraction(
        model=model, tokenizer=tok, df=_frame(), base_dir=str(tmp_path),
        prefix="tiny", contract=contract, chat_template_strategy="raw",
        num_layers=model.config.num_hidden_layers,
        hidden_dim=model.config.hidden_size, conditions=("no_social",),
        device="cpu", candidate_span_row_ids={0, 1}, resume=True,
    )
    with np.load(out2["no_social"]["candspan"]) as npz:
        assert sorted(npz.files) == keys1          # nothing lost, no duplicates
    index = pd.read_csv(out2["no_social"]["candspan_index"])
    assert sorted(set(index["row_id"])) == [0, 1]


def test_answer_position_rejects_multiple_conditions(tiny, tmp_path):
    """One generation CSV belongs to one condition; teacher-forcing another
    condition's prompts with it would silently dump wrong states."""
    model, tok = tiny
    contract = dataclasses.replace(CONTRACT_V1, sample_size=2)
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": "sea", "generated_word": "sea"},
    ])
    with pytest.raises(ValueError, match="single condition"):
        run_lens_extraction(
            model=model, tokenizer=tok, df=_frame(), base_dir=str(tmp_path),
            prefix="tiny", contract=contract, chat_template_strategy="raw",
            num_layers=model.config.num_hidden_layers,
            hidden_dim=model.config.hidden_size,
            conditions=("no_social", "with_social"),
            device="cpu", dump_answer_position=True, generation_csv=gen,
        )


def test_answer_dump_states_are_at_the_emitting_position(tiny, tmp_path):
    """The dumped state must be hidden[p*-1] — the position whose output
    channel emits the answer — never hidden[p*], whose layer-0 state IS the
    answer token's embedding (a lens there decodes the answer trivially)."""
    import torch

    from codenames.prompts import build_prompt

    model, tok = tiny
    text = "sea is the answer"
    gen = _generation_csv(tmp_path, [
        {"row_id": 0, "generated_text": text, "generated_word": "sea"},
        {"row_id": 1, "generated_text": "I think moon", "generated_word": "moon"},
    ])
    out = _run(tiny, tmp_path, gen)
    index = pd.read_csv(out["no_social"]["answer_index"]).set_index("row_id")
    p_star = int(index.loc[0, "p_star"])
    p_read = int(index.loc[0, "p_read"])
    assert p_read == p_star - 1

    row = _frame().iloc[0]
    prompt, _ = build_prompt(
        hint=str(row["output"]), candidates=list(row["candidates"]),
        giver_features={}, use_social_context=False, tokenizer=tok,
        chat_template_strategy="raw")
    joint = tok(prompt + text, return_tensors="pt")
    with torch.no_grad():
        ref = model(**joint, output_hidden_states=True, return_dict=True)
    arr = np.load(out["no_social"]["answer_hidden"], mmap_mode="r")
    for layer in (0, model.config.num_hidden_layers):
        expected = ref.hidden_states[layer][0, p_read].float().numpy() \
            .astype(np.float16)
        np.testing.assert_array_equal(np.asarray(arr[0, layer]), expected)
