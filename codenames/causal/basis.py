"""Canonical position basis for cross-turn aggregation (causal_spec.md §5A).

The patching grid is indexed by ``(layer, position)``. Within a single pair the
position axis is well defined -- the §5A alignment rule guarantees the clean and
corrupted prompts tokenise to the same length. **Across turns it is not.**
Prompts differ in token length (Mistral 108-131 on the pilot draw) because the
hint, the candidate words and the candidate count all vary, so absolute token
index ``p`` names a different thing in every turn: position 47 is a candidate
word here and the question scaffold there. Averaging such grids is not a
meaningful aggregate even when the shapes happen to match.

The basis below replaces the absolute index with the token's **role in the
prompt**, which is comparable across turns by construction:

    prefix | hint | post_hint | list_scaffold
           | cand_target | cand_donor | cand_other | question | final

Two properties make this the right axis rather than a convenience:

* It is the coordinate the spec already reasons in elsewhere. §3.3.3 requires
  every claimed locus to be stable in *candidate-relative* coordinates, and §5B
  fixes the steering injection sites as "hint-span-only" and
  "generating-position-only" -- both are role names, not indices.
* It makes the intervention statable. "Patch the clean hint span at layer 5" is
  a coherent cross-turn intervention with a sufficiency reading; "patch position
  47" is not.

``cand_target`` and ``cand_donor`` are separated from the other candidates
because the symmetric-counterfactual contrast is precisely between those two
words, so collapsing them into one candidate role would average away the
contrast the design is built on.

The social block (``with_social`` only) falls in ``prefix``: it precedes the
hint, and no RQ in this spec is about social conditioning. Role membership is
otherwise condition-independent, so a role-indexed grid is directly comparable
between the two conditions and between models with different tokenizers.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Fixed, ordered role vocabulary. The scan grid's second axis is this tuple, so
# its width is a constant of the study rather than a property of a draw.
ROLES = (
    "prefix",
    "hint",
    "post_hint",
    "list_scaffold",
    "cand_target",
    "cand_donor",
    "cand_other",
    "question",
    "final",
    "generation",
)

ROLE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(ROLES)}

# Anchors from prompts.build_instruction_body; identical for every model and
# condition, which is what lets one basis serve all of them.
_QUESTION = "Which word best matches the hint? Only output the word."
_CANDIDATE_ANCHOR = "The possible words are:"
_HINT_ANCHOR = 'The hint is: "'


def _offsets(tokenizer, prompt: str) -> Optional[List]:
    """Character offsets per token, or None for a non-fast tokenizer.

    Tokenised exactly as the forward pass does (``add_special_tokens`` left at
    its default), so index i here is index i in the hidden-state cache. A BOS
    added by the tokenizer carries the empty offset ``(0, 0)``, which overlaps
    no anchor and therefore stays ``prefix`` -- correct, it is scaffolding.
    """
    try:
        enc = tokenizer(prompt, return_offsets_mapping=True)
    except (TypeError, NotImplementedError, ValueError):
        return None
    mapping = enc.get("offset_mapping")
    return list(mapping) if mapping is not None else None


def _char_spans(prompt: str, hint: str,
                candidates: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """Character ranges for the hint, each candidate, and the question.

    Candidates are located through their **numbered list entry** (``"3. SEA"``)
    rather than by searching for the bare word. A bare search silently matches
    inside a longer candidate -- on a board holding both ``SEA`` and
    ``SEASHORE`` the span for ``SEA`` lands on the first five characters of
    ``SEASHORE`` -- and Codenames boards really do carry such pairs (ICE / ICE
    CREAM, NEW / NEW YORK). Nothing would crash; the grid would just describe
    the wrong token.

    The hint is anchored to ``The hint is: "`` for the same reason, so a hint
    that repeats a board word cannot capture that word's span instead.
    """
    spans: Dict[str, Tuple[int, int]] = {}

    at = prompt.find(_HINT_ANCHOR)
    if at != -1 and hint:
        start = at + len(_HINT_ANCHOR)
        if prompt[start:start + len(hint)] == hint:
            spans["hint"] = (start, start + len(hint))

    list_at = prompt.find(_CANDIDATE_ANCHOR)
    search_from = 0 if list_at == -1 else list_at + len(_CANDIDATE_ANCHOR)
    for i, word in enumerate(candidates):
        word = str(word)
        entry = f"{i + 1}. {word}"
        at = prompt.find(entry, search_from)
        if at == -1:
            # Ordinal drifted from the list order; fall back to the bare word
            # inside the list region, which is still safer than a global find.
            at = prompt.find(word, search_from)
            if at == -1:
                continue
            spans[f"cand:{word}"] = (at, at + len(word))
            continue
        start = at + len(entry) - len(word)
        spans[f"cand:{word}"] = (start, start + len(word))

    at = prompt.find(_QUESTION)
    if at != -1:
        spans["question"] = (at, at + len(_QUESTION))
    return spans


def _tokens_for(mapping: Sequence, char_span: Tuple[int, int]) -> Tuple[int, int]:
    """Half-open token range covering a character range; ``(0, 0)`` if none."""
    lo, hi = char_span
    start = end = None
    for idx, off in enumerate(mapping):
        s, e = int(off[0]), int(off[1])
        if e <= s:  # special tokens carry an empty offset
            continue
        if s < hi and e > lo:
            if start is None:
                start = idx
            end = idx + 1
    return (start, end) if start is not None else (0, 0)


def role_of_each_token(
    tokenizer,
    prompt: str,
    *,
    hint: str,
    candidates: Sequence[str],
    clean_target: str,
    donor_target: str,
    n_positions: Optional[int] = None,
) -> np.ndarray:
    """Role index for every token of ``prompt``; shape ``(n_positions,)``.

    ``prompt`` is the prompt alone. ``n_positions`` is the length of the scored
    sequence actually run through the model, which under §12.3 is the prompt
    **plus the teacher-forced clean generation**; those trailing tokens get the
    ``generation`` role. Without that they would fall past the last candidate
    and be labelled ``question``, silently pooling the model's own answer with
    the instruction scaffold.

    The offsets are taken with the same ``add_special_tokens`` default the
    forward pass uses, so role index i is hidden-state index i.

    Tokens that match no anchor keep a positional default: ``prefix`` before the
    hint, ``post_hint`` between the hint and the candidate list, ``question``
    after the last candidate. Every token therefore carries a role, and the
    role counts sum to the sequence length -- a property the aggregation below
    relies on to stay honest about coverage.
    """
    mapping = _offsets(tokenizer, prompt)
    n_from_offsets = 0 if mapping is None else len(mapping)
    n = int(n_positions) if n_positions is not None else n_from_offsets
    if n <= 0:
        return np.zeros(0, dtype=np.int16)

    roles = np.full(n, ROLE_INDEX["prefix"], dtype=np.int16)
    if mapping is None:
        # No offsets (non-fast tokenizer): only the generating position is
        # locatable. Marking the rest `prefix` would assert structure that was
        # never resolved, so the grid stays honest about what it could not label.
        roles[-1] = ROLE_INDEX["final"]
        return roles

    char_spans = _char_spans(prompt, str(hint), [str(c) for c in candidates])
    tok_spans = {name: _tokens_for(mapping, span)
                 for name, span in char_spans.items()}
    tok_spans = {name: span for name, span in tok_spans.items() if span[1] > span[0]}

    def assign(span, role: str) -> None:
        lo, hi = min(n, span[0]), min(n, span[1])
        if hi > lo:
            roles[lo:hi] = ROLE_INDEX[role]

    hint_span = tok_spans.get("hint")
    cand_spans = {str(w): tok_spans[f"cand:{w}"]
                  for w in candidates if f"cand:{w}" in tok_spans}
    question_span = tok_spans.get("question")

    # Positional defaults first, so anchored spans overwrite them.
    if hint_span is not None:
        first_cand = min((s for s, _ in cand_spans.values()), default=n)
        if first_cand > hint_span[1]:
            roles[min(n, hint_span[1]):min(n, first_cand)] = ROLE_INDEX["post_hint"]
    if cand_spans:
        first_cand = min(s for s, _ in cand_spans.values())
        last_cand = max(e for _, e in cand_spans.values())
        # The ordinal markers between the words ("1. ", "\n2. ", ...). They are
        # neither preamble nor candidate content, and they are where an ordinal
        # -slot locus would sit -- the positional confound §3.3.3 requires every
        # claimed locus to be tested against. Pooling them into `prefix` (which
        # is what a default fall-through did) both mislabelled ~45 tokens per
        # prompt and hid the one signal that would diagnose that confound.
        roles[min(n, first_cand):min(n, last_cand)] = ROLE_INDEX["list_scaffold"]
        if n > last_cand:
            roles[min(n, last_cand):] = ROLE_INDEX["question"]

    for word, span in cand_spans.items():
        if word == str(clean_target):
            assign(span, "cand_target")
        elif word == str(donor_target):
            assign(span, "cand_donor")
        else:
            assign(span, "cand_other")
    if question_span is not None:
        assign(question_span, "question")
    if hint_span is not None:
        assign(hint_span, "hint")

    # Anything past the prompt is the teacher-forced generation (§12.3).
    if n > n_from_offsets:
        roles[n_from_offsets:] = ROLE_INDEX["generation"]

    # The generating position -- the LAST PROMPT token, not the last token of
    # the scored sequence. It is the output channel and the secondary readout
    # of §4.1, so it must not be pooled into the scaffold around it.
    roles[min(n, n_from_offsets) - 1] = ROLE_INDEX["final"]
    return roles


def role_positions(roles: np.ndarray) -> Dict[str, List[int]]:
    """Token indices per role name; roles absent from this turn are omitted."""
    out: Dict[str, List[int]] = {}
    arr = np.asarray(roles)
    for name, idx in ROLE_INDEX.items():
        hits = np.flatnonzero(arr == idx)
        if hits.size:
            out[name] = [int(v) for v in hits]
    return out


def collapse_grid(grid: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """Reduce a ``(n_layers, n_positions)`` grid to ``(n_layers, len(ROLES))``.

    Within a role the **signed value of largest magnitude** is kept rather than
    a sum or a mean. A sum would make a role's score scale with its token count
    -- ``cand_other`` spans ~17 words and would dominate ``hint``'s two tokens
    for reasons of arithmetic, not causation -- and a mean would dilute a sharp
    single-token effect across a wide role. Max-|.| answers the question the
    screen actually asks: is there *any* strongly attributed token of this kind
    at this depth?

    Roles with no tokens in this turn become NaN so the cross-turn aggregate can
    average over the turns where the role exists instead of counting an absent
    role as a zero effect.
    """
    grid = np.asarray(grid, dtype=np.float64)
    roles = np.asarray(roles)
    n_layers = grid.shape[0]
    out = np.full((n_layers, len(ROLES)), np.nan, dtype=np.float64)
    usable = min(grid.shape[1], roles.shape[0])
    for name, idx in ROLE_INDEX.items():
        cols = np.flatnonzero(roles[:usable] == idx)
        if cols.size == 0:
            continue
        block = grid[:, cols]
        winner = np.argmax(np.abs(block), axis=1)
        out[:, idx] = block[np.arange(n_layers), winner]
    return out
