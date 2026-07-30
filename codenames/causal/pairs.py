"""Symmetric-counterfactual donor construction (causal_spec.md §5A).

A donor hint H' is valid for a turn on board B when H' own true target is a
word ON B that does not overlap the clean turn's targets. The corrupted run is
then a valid clean run for a DIFFERENT answer, which is what makes the logit
difference of §3.2 well defined at both ends. Donors are additionally
constrained to equal hint token count so clean and corrupted runs share a
position indexing (§5A alignment rule); measured cost is 5 turns in 7,703.

The rejected alternative, an unconstrained hint from any other turn, points at
nothing on B: the corrupted run is degenerate rather than counterfactual and
the §3.2 denominator is undefined.
"""

import re
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def substitute_hint(suffix: str, clean_hint: str, donor_hint: str) -> str:
    """Counterfactual scaffold (amendment (l), 2026-07-30).

    Teacher-forcing the CLEAN generation onto the corrupted run re-injects the
    clean hint whenever the scaffold quotes it ('The hint "death" suggests
    ...'), partially undoing the corruption — measured on the corrected
    Mistral pilot: 79/130 turns leak, sign-violation rate 29.1% on leaking
    turns vs 11.8% on clean ones, and P4 passes on the leak-free stratum.
    The corrupted run therefore teacher-forces the clean generation with every
    word-bounded mention of the clean hint replaced by the donor hint —
    the same symmetry §5A applies to the prompt, extended to the scaffold.
    Case-insensitive; compounds ("death-related") are replaced too, since
    they leak equally. Joint sequences that stop length-matching after the
    substitution are dropped and counted, mirroring the §5A alignment rule.
    """
    if not clean_hint:
        return suffix
    return re.sub(rf"\b{re.escape(clean_hint)}\b", donor_hint, suffix,
                  flags=re.IGNORECASE)


def hint_column(df: pd.DataFrame) -> str:
    """Name of the column holding the clue word.

    CULTURAL CODES stores it as ``output``; test fixtures and intermediate
    tables often call it ``hint``. Accepting both keeps callers from having to
    rename a column just to cross this boundary.
    """
    for name in ("hint", "output"):
        if name in df.columns:
            return name
    raise KeyError("expected a 'hint' or 'output' column holding the clue word")


def build_donor_index(targets: Dict[int, Sequence[str]]) -> Dict[str, List[int]]:
    """Map each target word to the turns whose true target set contains it."""
    index: Dict[str, List[int]] = defaultdict(list)
    for row_id, words in targets.items():
        for word in words:
            index[word].append(int(row_id))
    return {word: sorted(rows) for word, rows in index.items()}


def donors_for_turn(
    *,
    row_id: int,
    candidates: Sequence[str],
    targets: Sequence[str],
    donor_index: Dict[str, List[int]],
    hint_tokens: Dict[int, int],
    donor_targets: Optional[Dict[int, Sequence[str]]] = None,
    match_length: bool = True,
) -> List[int]:
    """Turns whose hint is a valid symmetric counterfactual for this turn."""
    own = set(targets)
    board = set(candidates)
    out = set()
    for word in board - own:
        for donor in donor_index.get(word, ()):
            if donor == row_id:
                continue
            if donor_targets is not None:
                donor_words = set(donor_targets.get(donor, ()))
                if not donor_words or not donor_words <= board or donor_words & own:
                    continue
            if match_length and hint_tokens.get(donor) != hint_tokens.get(row_id):
                continue
            out.add(donor)
    return sorted(out)


def build_pair_table(
    df_sample: pd.DataFrame,
    hint_tokens: Dict[int, int],
    *,
    seed: int = 2026,
    match_length: bool = True,
    prompt_length_fn: Optional[Callable[[int, str], int]] = None,
    max_donor_tries: int = 40,
) -> pd.DataFrame:
    """One (clean, donor) pair per turn. Turns with no valid donor are dropped.

    ``hint_tokens`` maps row_id to hint token count and is supplied by the
    caller, since it depends on the model's tokenizer and is not a dataset
    column.
    """
    hint_col = hint_column(df_sample)
    targets = {int(r.row_id): list(r.targets) for r in df_sample.itertuples()}
    boards = {int(r.row_id): list(r.candidates) for r in df_sample.itertuples()}
    hints = dict(zip(df_sample["row_id"].astype(int), df_sample[hint_col]))
    index = build_donor_index(targets)

    rng = np.random.default_rng(seed)
    rows = []
    for row_id in sorted(targets):
        donors = donors_for_turn(
            row_id=row_id,
            candidates=boards[row_id],
            targets=targets[row_id],
            donor_index=index,
            hint_tokens=hint_tokens,
            donor_targets=targets,
            match_length=match_length,
        )
        if not donors:
            continue

        if prompt_length_fn is None:
            donor = int(donors[rng.integers(len(donors))])
        else:
            # Standalone hint length is only a pre-filter: a hint tokenises
            # differently in context, so the ASSEMBLED prompts can still differ
            # in length. Measured on the real corpus, filtering afterwards left
            # only ~47% of pairs usable. Selecting a donor whose prompt already
            # matches restores the yield without inflating the sample size.
            target_len = prompt_length_fn(row_id, hints[row_id])
            order = rng.permutation(len(donors))[:max_donor_tries]
            donor = None
            for j in order:
                candidate = int(donors[j])
                if prompt_length_fn(row_id, hints[candidate]) == target_len:
                    donor = candidate
                    break
            if donor is None:
                continue
        rows.append(
            {
                "row_id": row_id,
                "donor_row_id": donor,
                "clean_target": targets[row_id][0],
                "donor_target": targets[donor][0],
                "hint": hints[row_id],
                "donor_hint": hints[donor],
                "n_donors": len(donors),
            }
        )

    columns = [
        "row_id", "donor_row_id", "clean_target", "donor_target",
        "hint", "donor_hint", "n_donors",
    ]
    return pd.DataFrame(rows, columns=columns).reset_index(drop=True)
