import numpy as np

from codenames.lens.readout import (
    build_token_table,
    first_token_ids,
    rank_from_scores,
    score_candidates,
    surface_variants,
)


class FakeTokenizer:
    """Deterministic offline tokenizer: one id per known prefix."""

    _vocab = {"spring": 7, "Spring": 8, " spring": 9, " Spring": 10,
              "bond": 3, " bond": 4, "Bond": 5, " Bond": 6}

    def encode(self, text, add_special_tokens=False):
        for prefix in sorted(self._vocab, key=len, reverse=True):
            if text.startswith(prefix):
                return [self._vocab[prefix], 999]  # 999 = dummy continuation
        return [0, 999]


def test_surface_variants():
    assert surface_variants("spring") == ["spring", "Spring", " spring", " Spring"]
    assert surface_variants("Bond") == ["Bond", " Bond"]  # no duplicates


def test_first_token_ids_dedup():
    tok = FakeTokenizer()
    assert first_token_ids(tok, "spring") == [7, 8, 9, 10]


def test_build_token_table():
    tok = FakeTokenizer()
    table = build_token_table(tok, ["spring", "bond"])
    assert table == {"spring": [7, 8, 9, 10], "bond": [3, 5, 4, 6]}


def test_score_is_max_over_variants():
    logits = np.zeros(1000, dtype=np.float32)
    logits[7], logits[8], logits[9], logits[10] = 1.0, 5.0, 2.0, 0.5
    logits[3] = 3.0
    table = {"spring": [7, 8, 9, 10], "bond": [3, 4, 5, 6]}
    scores = score_candidates(logits, table)
    assert scores["spring"] == 5.0
    assert scores["bond"] == 3.0


def test_rank_from_scores():
    ranks = rank_from_scores({"a": 3.0, "b": 5.0, "c": float("nan")})
    assert ranks == {"b": 1, "a": 2, "c": -1}
