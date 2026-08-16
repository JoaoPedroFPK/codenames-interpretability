"""SWOW-EN forced-choice items (pre-submission T7)."""

import numpy as np
import pandas as pd
import pytest

from codenames.data import load_dataset
from codenames.data_swow import build_items, load_swow_strengths, main


def _strengths():
    """A toy SWOW strength table: cue, response, strength."""
    rng = np.random.default_rng(0)
    cues = ["water", "rocket", "harbour", "fragile", "dark", "steam", "castle", "engine",
            "night", "glass", "school", "plate", "robin", "string", "anchor", "moon"]
    vocab = ["sea", "moon", "ship", "glass", "sky", "star", "boat", "break", "light",
             "train", "tower", "car", "day", "window", "teacher", "dish", "bird",
             "rope", "chain", "lunar", "ocean", "fire", "stone", "iron"]
    rows = []
    for c in cues:
        picks = rng.choice(vocab, size=6, replace=False)
        for j, r in enumerate(picks):
            rows.append({"cue": c, "response": r, "R123.Strength": 0.4 / (j + 1)})
    df = pd.DataFrame(rows)
    # top associate that IS the cue must be skipped
    df.loc[len(df)] = {"cue": "moon", "response": "moon", "R123.Strength": 0.9}
    # a multiword / non-alphabetic top response must be skipped too
    df.loc[len(df)] = {"cue": "glass", "response": "half full", "R123.Strength": 0.9}
    return df


def test_build_items_shapes_and_constraints():
    items = build_items(_strengths(), n=5, seed=2026)
    assert len(items) == 5
    assert {"output", "targets", "black", "tan", "candidates", "row_id", "cue_strength"} \
        <= set(items.columns)
    for r in items.itertuples():
        pool = list(r.candidates)
        assert 11 <= len(pool) <= 15
        assert pool == sorted(pool)
        assert len(set(pool)) == len(pool)
        assert all(w.replace("-", "").isalpha() for w in pool)
        assert len(r.targets) == 1 and r.targets[0] in pool
        assert r.output not in pool
        assert r.black == []
        assert set(r.tan) == set(pool) - set(r.targets)


def test_distractors_are_not_associates_of_the_cue():
    """Low-strength = never given as a response to this cue in SWOW."""
    st = _strengths()
    items = build_items(st, n=8, seed=2026)
    for r in items.itertuples():
        associates = set(st.loc[st["cue"] == r.output, "response"])
        assert not (set(r.tan) & associates), (r.output, set(r.tan) & associates)


def test_build_items_is_seeded_and_skips_degenerate_cues():
    a = build_items(_strengths(), n=6, seed=2026)
    b = build_items(_strengths(), n=6, seed=2026)
    pd.testing.assert_frame_equal(a, b)
    c = build_items(_strengths(), n=6, seed=2027)
    assert not a["output"].tolist() == c["output"].tolist() or \
        not a["candidates"].tolist() == c["candidates"].tolist()
    assert "moon" not in set(a["output"]) or a[a["output"] == "moon"]["targets"].iloc[0] != ["moon"]
    for r in a.itertuples():
        assert r.targets[0] != r.output


def test_items_csv_round_trips_through_load_dataset(tmp_path):
    """The item file must be consumable by `run` unchanged: same columns as
    CULTURAL CODES (stringified lists), candidates rebuilt alphabetically."""
    st = tmp_path / "strength.csv"
    _strengths().to_csv(st, index=False)
    out = tmp_path / "swow_items.csv"
    assert main(["--swow", str(st), "--n", "5", "--seed", "2026", "--out", str(out)]) == 0
    df = load_dataset(str(out))
    assert len(df) == 5
    for r in df.itertuples():
        assert 11 <= len(r.candidates) <= 15 and r.targets[0] in r.candidates


def test_load_swow_strengths_accepts_the_official_columns(tmp_path):
    p = tmp_path / "strength.SWOW-EN.R123.csv"
    pd.DataFrame({"cue": ["a", "a"], "response": ["b", "c"], "R123": [10, 5],
                  "N": [100, 100], "R123.Strength": [0.1, 0.05]}).to_csv(p, index=False, sep="\t")
    df = load_swow_strengths(str(p))
    assert list(df.columns) == ["cue", "response", "strength"]
    assert df["strength"].tolist() == [0.1, 0.05]
