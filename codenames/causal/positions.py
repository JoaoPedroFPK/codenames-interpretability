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

from typing import Optional

# Leading characters a model may emit before the answer word. Kept explicit
# rather than a regex so the rule is auditable against the generation parser.
_STRIP = " \t\n\r\"'`*-–—:."


def is_word_first(generated_text: Optional[str], generated_word: Optional[str]) -> bool:
    """True when the generation opens with the parsed answer word."""
    if not isinstance(generated_text, str) or not isinstance(generated_word, str):
        return False
    return generated_text.strip(_STRIP).lower().startswith(generated_word.strip().lower())


def answer_position(
    tokenizer, prompt: str, generated_text: str, generated_word: str
) -> Optional[int]:
    """Token index where ``generated_word`` starts within prompt+generated_text.

    Located by character offset, then mapped to a token by accumulating decoded
    piece lengths -- robust across tokenizers that do not expose offset
    mappings. Returns None when the word does not occur in the generation, in
    which case the turn has no p* and is excluded with a reported count.
    """
    if not isinstance(generated_text, str) or not isinstance(generated_word, str):
        return None

    needle = generated_word.strip().lower()
    if not needle:
        return None
    char_idx = generated_text.lower().find(needle)
    if char_idx < 0:
        return None

    target_char = len(prompt) + char_idx
    ids = tokenizer.encode(prompt + generated_text, add_special_tokens=False)

    running = 0
    for pos, token_id in enumerate(ids):
        piece = tokenizer.decode([token_id])
        nxt = running + len(piece)
        if running <= target_char < nxt:
            return pos
        running = nxt
    return None
