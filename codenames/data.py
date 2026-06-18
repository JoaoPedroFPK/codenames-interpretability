"""Dataset loading and turn sampling.

Loads the CULTURAL CODES ``clue_generation.csv`` and draws the same
``SAMPLE_SIZE`` boards under ``random_state=random_seed`` for every model, so
cross-model comparison is on identical boards.
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

    Reads the CSV, applies ``ast.literal_eval`` to ``targets``/``black``/``tan``,
    builds the alphabetical ``candidates`` column, resets the index and assigns
    ``row_id``.
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

    Reproducible across models when the same seed is used.
    """
    sampled = df.sample(n=min(n, len(df)), random_state=seed).copy().reset_index(drop=True)
    return sampled


def build_candidates_fixed_order(row) -> List[str]:
    """Return all board words in stable alphabetical order."""
    all_words = list(row["targets"]) + list(row["black"]) + list(row["tan"])
    return sorted(all_words)


def extract_giver_features(row, giver_cols: List[str]) -> Dict[str, object]:
    """Extract non-null giver feature values from a dataset row."""
    return {
        c: row[c]
        for c in giver_cols
        if c in row.index and not pd.isna(row[c])
    }
