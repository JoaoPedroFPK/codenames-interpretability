"""Second corpus: forced-choice items from the Small World of Words (SWOW-EN).

De Deyne, Navarro, Perfors, Brysbaert & Storms (2019), *The "Small World of
Words" English word association norms for over 12,000 cue words*, Behavior
Research Methods 51:987-1006. Data: https://smallworldofwords.org/en/project
/research (SWOW-EN, R100 / R123 strength tables), released under
CC BY-NC-SA 4.0 -- cite the paper and the licence wherever items are used.

Each item mirrors one CULTURAL CODES clue turn so the SAME prompt body, the
same extraction loop and the same lens/patching code run unchanged
(pre-submission task T7):

* ``output`` (the hint) = the SWOW cue;
* ``targets`` = the cue's single strongest associate (single-target by
  construction);
* ``tan`` = 10-14 distractors drawn from words that are responses to OTHER
  cues and were NEVER given as a response to this cue (association strength
  0 in the norms) -- the "low-strength" pool; ``black`` is empty;
* ``candidates`` = the alphabetical pool of 11-15 words (pool size drawn
  uniformly), matching the Codenames board sizes the models saw.

Cues are skipped when the top associate is the cue itself, is not a plain
word, or has fewer than 11 eligible distractors. Written as CSV with the
stringified-list columns ``codenames.data.load_dataset`` expects, so
``codenames-experiment run --dataset swow_items.csv`` works as-is (point
``--output-dir`` at a separate folder: the per-model prefix is unchanged).
"""

import argparse
import re
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

_WORD = re.compile(r"^[A-Za-z][A-Za-z\-]*$")
_STRENGTH_COLS = ("R123.Strength", "R1.Strength", "strength", "Strength")
POOL_MIN, POOL_MAX = 11, 15


def load_swow_strengths(path: str) -> pd.DataFrame:
    """Read a SWOW strength export (tab or comma separated) to
    ``cue, response, strength``."""
    df = pd.read_csv(path, sep=None, engine="python")
    col = next((c for c in _STRENGTH_COLS if c in df.columns), None)
    if col is None or "cue" not in df.columns or "response" not in df.columns:
        raise ValueError(f"{path}: expected columns cue, response and one of {_STRENGTH_COLS}")
    out = df[["cue", "response", col]].rename(columns={col: "strength"})
    return out.dropna(subset=["cue", "response"]).reset_index(drop=True)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either the raw SWOW export columns or the loader's output."""
    if "strength" in df.columns:
        return df[["cue", "response", "strength"]].copy()
    col = next((c for c in _STRENGTH_COLS if c in df.columns), None)
    if col is None:
        raise ValueError(f"need a strength column, one of {_STRENGTH_COLS}")
    return df[["cue", "response", col]].rename(columns={col: "strength"}).copy()


def _is_word(s: object) -> bool:
    return isinstance(s, str) and bool(_WORD.match(s))


def build_items(strengths: pd.DataFrame, n: int, seed: int = 2026,
                pool_min: int = POOL_MIN, pool_max: int = POOL_MAX) -> pd.DataFrame:
    """``n`` seeded items in the CULTURAL CODES schema (see module docstring).

    Deterministic for a seed: cues are shuffled once, then walked in order
    until ``n`` valid items exist; distractors are drawn per item from the
    same generator.
    """
    st = _normalise(strengths)
    st["cue"] = st["cue"].astype(str).str.strip().str.lower()
    st["response"] = st["response"].astype(str).str.strip().str.lower()
    st = st[st["cue"].map(_is_word) & st["response"].map(_is_word)]
    st = st[st["cue"] != st["response"]]
    if st.empty:
        raise ValueError("no usable cue/response rows")

    associates = st.groupby("cue")["response"].apply(set).to_dict()
    top = (st.sort_values(["cue", "strength"], ascending=[True, False])
             .groupby("cue").head(1).set_index("cue"))
    vocab = np.array(sorted(set(st["response"])))

    rng = np.random.default_rng(seed)
    cues = np.array(sorted(top.index))
    rng.shuffle(cues)

    rows: List[dict] = []
    for cue in cues:
        if len(rows) >= n:
            break
        target = str(top.loc[cue, "response"])
        eligible = np.array([w for w in vocab
                             if w != cue and w != target and w not in associates[cue]])
        pool_size = int(rng.integers(pool_min, pool_max + 1))
        k = pool_size - 1
        if eligible.size < k:
            continue
        distractors = sorted(rng.choice(eligible, size=k, replace=False).tolist())
        rows.append({
            "output": cue, "targets": [target], "black": [], "tan": distractors,
            "candidates": sorted([target] + distractors),
            "cue_strength": float(top.loc[cue, "strength"]),
        })
    if len(rows) < n:
        raise ValueError(f"only {len(rows)} valid items could be built (asked for {n})")
    items = pd.DataFrame(rows).reset_index(drop=True)
    items["row_id"] = items.index.astype(int)
    return items


def write_items_csv(items: pd.DataFrame, path: str) -> None:
    """Stringify the list columns the way CULTURAL CODES stores them."""
    out = items[["output", "targets", "black", "tan", "cue_strength"]].copy()
    for col in ("targets", "black", "tan"):
        out[col] = out[col].map(repr)
    out.to_csv(path, index=False)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--swow", required=True, help="SWOW-EN strength table (R123 or R100).")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    items = build_items(load_swow_strengths(a.swow), n=a.n, seed=a.seed)
    write_items_csv(items, a.out)
    print(f"wrote {len(items)} items -> {a.out} (pools {items['candidates'].map(len).min()}"
          f"-{items['candidates'].map(len).max()} words)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
