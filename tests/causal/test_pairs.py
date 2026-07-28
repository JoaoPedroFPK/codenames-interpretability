import pandas as pd

from codenames.causal.pairs import build_donor_index, build_pair_table, donors_for_turn


def _frame():
    """Matches the real schema from codenames.data.load_dataset:
    row_id:int, hint:str, targets/black/tan:list, candidates:list (alphabetical).
    """
    return pd.DataFrame(
        [
            {"row_id": 1, "hint": "ocean", "targets": ["sea"],
             "black": ["moon"], "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
            {"row_id": 2, "hint": "rocket", "targets": ["moon"],
             "black": ["sea"], "tan": ["ship"], "candidates": ["moon", "sea", "ship"]},
            {"row_id": 3, "hint": "a b c", "targets": ["ship"],
             "black": ["sea"], "tan": ["moon"], "candidates": ["moon", "sea", "ship"]},
        ]
    )


# hint_tokens is NOT a dataset column - the caller computes it with a tokenizer
# and passes it in. These fixtures use stand-in counts.
HINT_TOKENS = {1: 1, 2: 1, 3: 3}


def test_donor_index_maps_target_word_to_turns():
    df = _frame()
    idx = build_donor_index(dict(zip(df.row_id, df.targets)))
    assert idx["sea"] == [1]
    assert idx["moon"] == [2]


def test_donor_must_target_a_word_on_this_board_and_not_our_target():
    df = _frame()
    idx = build_donor_index(dict(zip(df.row_id, df.targets)))
    got = donors_for_turn(
        row_id=1, candidates=["moon", "sea", "ship"], targets=["sea"],
        donor_index=idx, hint_tokens=HINT_TOKENS,
        donor_targets=dict(zip(df.row_id, df.targets)), match_length=False,
    )
    assert set(got) == {2, 3}          # both target a non-"sea" board word
    assert 1 not in got                # never itself


def test_length_matching_filters_donors():
    df = _frame()
    idx = build_donor_index(dict(zip(df.row_id, df.targets)))
    got = donors_for_turn(
        row_id=1, candidates=["moon", "sea", "ship"], targets=["sea"],
        donor_index=idx, hint_tokens=HINT_TOKENS,
        donor_targets=dict(zip(df.row_id, df.targets)), match_length=True,
    )
    assert got == [2]                  # turn 3's hint is 3 tokens, turn 1's is 1


def test_pair_table_is_deterministic_under_seed():
    df = _frame()
    a = build_pair_table(df, HINT_TOKENS, seed=2026, match_length=False)
    b = build_pair_table(df, HINT_TOKENS, seed=2026, match_length=False)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.columns) >= {"row_id", "donor_row_id", "donor_target", "clean_target"}
    assert (a.row_id != a.donor_row_id).all()


def test_turns_without_a_donor_are_dropped_not_silently_mispaired():
    df = pd.DataFrame([
        {"row_id": 9, "hint": "solo", "targets": ["alpha"],
         "black": [], "tan": [], "candidates": ["alpha"]},
    ])
    out = build_pair_table(df, {9: 1}, seed=2026, match_length=False)
    assert len(out) == 0


def test_pair_table_accepts_the_real_dataset_hint_column():
    """CULTURAL CODES stores the clue in `output`, not `hint`."""
    df = _frame().rename(columns={"hint": "output"})
    out = build_pair_table(df, HINT_TOKENS, seed=2026, match_length=False)
    assert len(out) > 0
    assert set(out["hint"]) <= {"ocean", "rocket", "a b c"}


def test_missing_hint_column_is_a_clear_error():
    from codenames.causal.pairs import hint_column
    import pytest
    with pytest.raises(KeyError, match="hint.*output"):
        hint_column(_frame().drop(columns=["hint"]))


def test_prompt_aware_donor_selection_picks_an_aligned_donor():
    """Real corpus: filtering misaligned pairs afterwards left only ~47%.

    Selecting a donor whose assembled prompt already matches restores the
    yield without inflating the sample size.
    """
    df = _frame()
    # "a b c" assembles to a longer prompt; the selector must avoid it.
    lengths = {"ocean": 10, "rocket": 10, "a b c": 14}

    def prompt_len(row_id, hint):
        return lengths[hint]

    out = build_pair_table(df, {1: 1, 2: 1, 3: 1}, seed=2026,
                           match_length=False, prompt_length_fn=prompt_len)
    assert len(out) > 0
    for r in out.itertuples():
        assert prompt_len(r.row_id, r.hint) == prompt_len(r.row_id, r.donor_hint)


def test_turns_with_no_aligned_donor_are_dropped():
    df = _frame()
    lengths = {"ocean": 10, "rocket": 99, "a b c": 98}   # nothing matches turn 1

    def prompt_len(row_id, hint):
        return lengths[hint]

    out = build_pair_table(df, {1: 1, 2: 1, 3: 1}, seed=2026,
                           match_length=False, prompt_length_fn=prompt_len)
    assert 1 not in set(out["row_id"])
