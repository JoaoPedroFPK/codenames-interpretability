"""Dataset loading and turn sampling.

The CULTURAL CODES `clue_generation.csv` is shared by all seven notebooks. The
same SAMPLE_SIZE boards are drawn under ``random_state=random_seed`` in every
notebook, guaranteeing cross-model comparison is on identical boards.

This module is byte-identical-equivalent to Cell 3 and Cell 4 of every
reference notebook.
"""

import ast
from typing import Dict, List

import pandas as pd

GIVER_COLS: List[str] = [
    "giver.marriage",
    "giver.education",
    "giver.race",
    "giver.continent",
    "giver.language",
    "giver.religion",
    "giver.gender",
    "giver.country",
    "giver.political",
]


def load_dataset(path: str) -> pd.DataFrame:
    """Load CULTURAL CODES, evaluate stringified list columns, build candidates.

    Mirrors Cell 3 of every reference notebook: reads the CSV, ``ast.literal_eval``
    on ``targets``/``black``/``tan``, builds the alphabetical ``candidates``
    column, resets index and assigns ``row_id``.
    """
    df = pd.read_csv(path)

    for col in ["targets", "black", "tan"]:
        df[col] = df[col].apply(ast.literal_eval)

    df["candidates"] = df.apply(build_candidates_fixed_order, axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)
    return df


def sample_turns(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw ``n`` boards via ``df.sample(n, random_state=seed)`` and reset index.

    Identical call pattern to Cell 9 of every reference notebook. The sample
    is reproducible across models when the same seed is used.
    """
    sampled = df.sample(n=min(n, len(df)), random_state=seed).copy().reset_index(drop=True)
    return sampled


def build_candidates_fixed_order(row) -> List[str]:
    """Return all board words in stable alphabetical order.

    Verbatim from Cell 3 of every reference notebook.
    """
    all_words = list(row["targets"]) + list(row["black"]) + list(row["tan"])
    return sorted(all_words)


def extract_giver_features(row, giver_cols: List[str]) -> Dict[str, object]:
    """Extract non-null giver feature values from a dataset row.

    Verbatim from Cell 8 of every reference notebook (defined inline there;
    factored out here because both ``prompts.py`` and ``loop.py`` need it).
    """
    return {
        c: row[c]
        for c in giver_cols
        if c in row.index and not pd.isna(row[c])
    }
