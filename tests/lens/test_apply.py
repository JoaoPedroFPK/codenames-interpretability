import numpy as np
import pandas as pd

from codenames.lens.apply import Readout, compute_scores, rmsnorm_np
from codenames.lens.tuned import TunedLens


class FakeTokenizer:
    _vocab = {"spring": 1, "bond": 2, "cycle": 3, "point": 4}

    def encode(self, text, add_special_tokens=False):
        key = text.strip().lower()
        return [self._vocab.get(key, 0)]


def _fixture(tmp_path):
    n, L, d, V = 2, 2, 4, 6
    rng = np.random.default_rng(2026)
    hidden = rng.standard_normal((n, L + 1, d)).astype(np.float16)
    hidden_path = str(tmp_path / "hidden.npy")
    np.save(hidden_path, hidden)
    index = pd.DataFrame({"board_idx": [0, 1], "row_id": [10, 11],
                          "prompt_token_count": [5, 5], "ok": [True, True],
                          "error": ["", ""]})
    index_path = str(tmp_path / "index.csv")
    index.to_csv(index_path, index=False)
    df = pd.DataFrame({
        "row_id": [10, 11],
        "output": ["season", "spy"],
        "targets": [["spring"], ["bond"]],
        "black": [["bond"], ["spring"]],
        "tan": [["cycle", "point"], ["cycle", "point"]],
        "candidates": [["bond", "cycle", "point", "spring"]] * 2,
    })
    readout = Readout(
        norm_weight=np.ones(d, dtype=np.float32),
        lm_head=rng.standard_normal((V, d)).astype(np.float32),
        eps=1e-6,
    )
    return hidden_path, index_path, df, readout


def test_rmsnorm_matches_definition():
    H = np.array([[3.0, 4.0]], dtype=np.float32)
    w = np.array([2.0, 2.0], dtype=np.float32)
    out = rmsnorm_np(H, w, eps=0.0)
    rms = np.sqrt(np.mean(H ** 2))
    np.testing.assert_allclose(out, H / rms * w, rtol=1e-6)


def test_scores_shape_and_identity_translators(tmp_path):
    hidden_path, index_path, df, readout = _fixture(tmp_path)
    tok = FakeTokenizer()
    raw = compute_scores(hidden_path, index_path, df, tok, readout, "raw")
    # 2 turns x 3 layers x 4 candidates
    assert len(raw) == 2 * 3 * 4
    assert set(raw.columns) == {"row_id", "layer", "word", "word_type",
                                "lens", "score", "rank"}
    assert set(raw["word_type"]) == {"target", "black", "tan"}
    # per (turn, layer) ranks are a permutation of 1..4
    g = raw.groupby(["row_id", "layer"])["rank"].apply(sorted)
    assert all(v == [1, 2, 3, 4] for v in g)

    d = readout.norm_weight.shape[0]
    ident = TunedLens(A=np.stack([np.eye(d, dtype=np.float32)] * 2),
                      b=np.zeros((2, d), dtype=np.float32), history=[])
    tuned = compute_scores(hidden_path, index_path, df, tok, readout,
                           "tuned", translators=ident)
    assert set(tuned["lens"]) == {"tuned"}
    np.testing.assert_allclose(tuned["score"].values, raw["score"].values,
                               rtol=1e-5, atol=1e-5)


def test_failed_boards_are_skipped(tmp_path):
    hidden_path, index_path, df, readout = _fixture(tmp_path)
    idx = pd.read_csv(index_path)
    idx.loc[1, "ok"] = False
    idx.to_csv(index_path, index=False)
    raw = compute_scores(hidden_path, index_path, df, FakeTokenizer(),
                         readout, "raw")
    assert set(raw["row_id"]) == {10}


def test_compute_scores_accepts_an_index_frame_and_skips_not_ok(tmp_path):
    """The answer-channel index arrives as a frame (ok = ~p_star_missing);
    boards without a resolved p* must be absent, not scored on NaN states."""
    hidden_path, _, df, readout = _fixture(tmp_path)
    index = pd.DataFrame({"board_idx": [0, 1], "row_id": [10, 11],
                          "ok": [True, False]})
    out = compute_scores(hidden_path, index, df, FakeTokenizer(), readout, "raw")
    assert set(out["row_id"]) == {10}
