"""Answer-position resolution (causal_spec.md §4.1, lens_spec.md §5.1).

The generating position (final prompt token) is where the answer forms ONLY
when the model emits the answer word first. Measured on the completed runs:
Mistral 13.0% of turns, Qwen 91.5%. For the remaining Mistral turns the answer
token sits 10+ positions downstream behind scaffolding ("The word that best
matches the hint ... is ..."), where no identity with the output channel holds.

p* is the token index at which the parsed answer word begins in the
teacher-forced prompt+generation sequence, and it is the primary measurement
position for both specs. Restricting to word-first turns instead would bias the
estimand: those turns are easier (Mistral 0.857 vs 0.678 generation accuracy).
"""

import re
from typing import Optional

# Leading characters a model may emit before the answer word. Kept explicit
# rather than a regex so the rule is auditable against the generation parser.
_STRIP = " \t\n\r\"'`*-–—:."


def is_word_first(generated_text: Optional[str], generated_word: Optional[str]) -> bool:
    """True when the generation opens with the parsed answer word."""
    if not isinstance(generated_text, str) or not isinstance(generated_word, str):
        return False
    return generated_text.strip(_STRIP).lower().startswith(generated_word.strip().lower())


def _locate(haystack: str, needle: str) -> int:
    """Character index of the answer word, preferring a whole-word match.

    A bare ``find`` takes the first substring occurrence, which on the dominant
    scaffolded format is inside the preamble rather than at the answer: the
    hint word ``instagram`` contains ``tag``, and ``matches`` in "The word that
    best matches the hint" contains ``match``. Measured on 1,200 real Mistral
    generations this mislocated 9 turns; requiring word boundaries fixes all of
    them. The substring search is kept as a fallback so a word that only ever
    appears glued to punctuation is still found rather than dropped.
    """
    match = re.search(rf"\b{re.escape(needle)}\b", haystack)
    if match:
        return match.start()
    return haystack.find(needle)


def answer_position(
    tokenizer, prompt: str, generated_text: str, generated_word: str
) -> Optional[int]:
    """Token index where ``generated_word`` starts within prompt+generated_text.

    Resolved through the fast tokenizer's **character offset mapping**, taken
    with the same ``add_special_tokens`` default the forward pass uses, so the
    returned index is directly a hidden-state index.

    Both of those details are load-bearing, and getting either wrong is silent:

    * *Offsets, not decoded piece lengths.* The original implementation walked
      the sequence accumulating ``len(tokenizer.decode([id]))``. SentencePiece
      decoding drops the leading space marker, so the running character count
      falls behind the true string -- measured at 134 characters of prompt
      rebuilding as 112 -- and the index drifts further the longer the text.
      Against real recorded Mistral generations the returned index landed on
      the answer's first subword in **1 of 82** resolved turns, and returned
      ``None`` on a further 41% where the drift ran off the end.
    * *The forward pass's own tokenisation.* Encoding with
      ``add_special_tokens=False`` while the model is run with the default
      ``True`` shifts every index by the BOS the tokenizer prepends.

    Neither error is visible to gate check P1, which teacher-forces the greedy
    generation and compares the argmax at ``p*-1`` against the token at ``p*``:
    under greedy decoding that identity holds at *every* index inside the
    generated span, so a constant offset passes it. P1 validates teacher
    forcing, not that ``p*`` points at the answer.

    Returns None when the word does not occur in the generation, in which case
    the turn has no p* and is excluded with a reported count (spec §4.1).
    """
    if not isinstance(generated_text, str) or not isinstance(generated_word, str):
        return None

    needle = generated_word.strip().lower()
    if not needle:
        return None
    char_idx = _locate(generated_text.lower(), needle)
    if char_idx < 0:
        return None

    target_char = len(prompt) + char_idx
    full = prompt + generated_text

    mapping = None
    try:
        mapping = tokenizer(full, return_offsets_mapping=True).get("offset_mapping")
    except (TypeError, NotImplementedError, ValueError):
        mapping = None

    if mapping:
        first_after = None
        for pos, off in enumerate(mapping):
            start, end = int(off[0]), int(off[1])
            if end <= start:  # special tokens carry an empty offset
                continue
            if start <= target_char < end:
                return pos
            if first_after is None and start >= target_char:
                first_after = pos
        # The word may begin just past a token boundary when leading
        # whitespace is absorbed into the preceding token; the first token
        # starting at or after the target is then the answer's own token.
        return first_after

    # Non-fast tokenizer: fall back to accumulating decoded pieces. Inexact for
    # SentencePiece (see above), which is why it is the last resort.
    ids = tokenizer.encode(full, add_special_tokens=False)
    running = 0
    for pos, token_id in enumerate(ids):
        piece = tokenizer.decode([token_id])
        nxt = running + len(piece)
        if running <= target_char < nxt:
            return pos
        running = nxt
    return None
