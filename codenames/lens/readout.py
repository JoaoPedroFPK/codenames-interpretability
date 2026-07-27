"""Candidate-restricted readout rule (docs/specs/lens_spec.md §5, pre-registered).

A candidate's score at a layer is the MAX over its surface variants of the
logit of the variant's FIRST subword token. Variants mirror the generation
parser's case-insensitivity: board form and capitalized form, each with and
without a leading space. Pure numpy; no torch dependency so analysis runs
anywhere.
"""

from typing import Dict, List

import numpy as np


def surface_variants(word: str) -> List[str]:
    out, seen = [], set()
    for v in (word, word.capitalize(), " " + word, " " + word.capitalize()):
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def first_token_ids(tokenizer, word: str) -> List[int]:
    ids, seen = [], set()
    for v in surface_variants(word):
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks and toks[0] not in seen:
            seen.add(toks[0])
            ids.append(int(toks[0]))
    return ids


def build_token_table(tokenizer, candidates: List[str]) -> Dict[str, List[int]]:
    return {c: first_token_ids(tokenizer, c) for c in candidates}


def score_candidates(
    logits: np.ndarray, table: Dict[str, List[int]]
) -> Dict[str, float]:
    return {
        c: (float(np.max(logits[ids])) if ids else float("nan"))
        for c, ids in table.items()
    }


def rank_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    valid = {w: s for w, s in scores.items() if not np.isnan(s)}
    order = sorted(valid, key=valid.get, reverse=True)
    ranks = {w: i + 1 for i, w in enumerate(order)}
    for w in scores:
        ranks.setdefault(w, -1)
    return ranks
