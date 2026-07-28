import pytest
from transformers import AutoTokenizer

from codenames.causal.positions import answer_position, is_word_first

# Same tiny model the lens tests use: fast BPE tokenizer, no sentencepiece
# dependency, already in the local HF cache.
TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(TINY)


def test_word_first_strips_whitespace_and_quotes():
    assert is_word_first('  "soul" is the answer', "soul")
    assert is_word_first("Soul, because...", "soul")
    assert not is_word_first("The word that best matches is soul", "soul")


def test_word_first_handles_missing_values():
    assert not is_word_first(None, "soul")
    assert not is_word_first("soul", None)


def test_answer_position_is_none_when_word_absent(tok):
    assert answer_position(tok, "P:", "nothing here", "soul") is None


def test_answer_position_points_at_the_word(tok):
    prompt, gen, word = "Prompt: ", "The answer is soul.", "soul"
    pos = answer_position(tok, prompt, gen, word)
    assert pos is not None
    ids = tok.encode(prompt + gen, add_special_tokens=False)
    assert 0 <= pos < len(ids)
    # The token at pos must begin the word (allowing a leading space marker).
    piece = tok.decode([ids[pos]]).strip().lower()
    assert piece and word.startswith(piece[: len(piece)])


def test_p_star_is_not_the_prompt_token_count(tok):
    """BPE prompt tokenization is NOT a prefix of the joint tokenization.

    "Prompt: " alone ends in a standalone ' ' token, but in "Prompt: soul..."
    that space merges into ' soul'. So p* must be resolved against the JOINT
    prompt+generation sequence and can be earlier than len(encode(prompt)).
    Offsetting from the prompt's own token count silently reads the wrong
    position -- which is the whole failure mode p* exists to avoid.
    """
    prompt, gen, word = "Prompt: ", "soul is my guess", "soul"
    pos = answer_position(tok, prompt, gen, word)
    n_prompt = len(tok.encode(prompt, add_special_tokens=False))
    ids = tok.encode(prompt + gen, add_special_tokens=False)

    assert pos == 2 and n_prompt == 3, "boundary merge no longer reproduces"
    assert pos < n_prompt
    assert tok.decode([ids[pos]]).strip().lower() == word


def test_p_star_token_begins_the_answer_word(tok):
    """The invariant that must hold regardless of tokenizer boundary quirks."""
    for prompt, gen, word in [
        ("Prompt: ", "soul is my guess", "soul"),
        ("Q: ", "The answer is soul.", "soul"),
        ("Q:", "I pick moon", "moon"),
    ]:
        pos = answer_position(tok, prompt, gen, word)
        ids = tok.encode(prompt + gen, add_special_tokens=False)
        assert pos is not None
        assert word.startswith(tok.decode([ids[pos]]).strip().lower())


# --- p* must land on the ANSWER, not merely be self-consistent -------------
#
# The original resolver walked the sequence accumulating len(decode([id])).
# SentencePiece drops the leading space marker, so the character count drifted
# and p* landed on the answer's first subword in 1 of 82 real Mistral turns.
# Gate check P1 could not see it: it teacher-forces the greedy generation and
# compares argmax at p*-1 against the token at p*, an identity that holds at
# EVERY index inside a greedily generated span. These tests check the property
# P1 cannot.

def _lands_on_answer(tok, prompt, text, word):
    from codenames.causal.positions import answer_position
    p = answer_position(tok, prompt, text, word)
    if p is None:
        return False
    ids = tok(prompt + text)["input_ids"]
    if p >= len(ids):
        return False
    got = tok.decode([ids[p]]).strip().lower()
    return bool(got) and word.strip().lower().startswith(got)


def test_p_star_lands_on_the_answer_token_for_a_scaffolded_generation(tok):
    """Mistral's dominant format: the answer sits 10+ tokens behind a preamble.
    This is 87% of its turns, so a resolver that only works on word-first
    generations is a resolver that does not work."""
    prompt = '<s>[INST] Which word best matches the hint? [/INST]'
    text = 'The word that best matches the hint "frozen" is "ICE".'
    assert _lands_on_answer(tok, prompt, text, "ICE")


def test_p_star_lands_on_the_answer_token_for_a_word_first_generation(tok):
    prompt = '<s>[INST] Which word best matches the hint? [/INST]'
    assert _lands_on_answer(tok, prompt, "ICE", "ICE")


def test_p_star_indexes_the_forward_pass_tokenisation(tok):
    """p* is used directly as a hidden-state index, so it must count the BOS
    the tokenizer prepends. Encoding with add_special_tokens=False while the
    model runs with the default True shifts every index by one."""
    from codenames.causal.positions import answer_position
    prompt = '<s>[INST] hint [/INST]'
    text = 'The answer is "MOON".'
    p = answer_position(tok, prompt, text, "MOON")
    assert p is not None
    assert p < len(tok(prompt + text)["input_ids"])


def test_the_answer_word_hiding_inside_the_preamble_is_not_matched(tok):
    """"matches" contains "match"; "instagram" contains "tag". A bare substring
    search takes the preamble occurrence and reports a position in the
    scaffolding. Nine of 1,200 real Mistral turns hit this."""
    prompt = '<s>[INST] hint [/INST]'
    text = 'The word that best matches the hint "sports" is "match".'
    assert _lands_on_answer(tok, prompt, text, "match")


def test_a_word_absent_from_the_generation_still_returns_none(tok):
    """§4.1 excludes such turns and reports the count; they must not silently
    resolve to some other token."""
    from codenames.causal.positions import answer_position
    assert answer_position(tok, "<s>[INST] hi [/INST]",
                           "The answer is MOON.", "CASTLE") is None


# --- the tiny model masks this bug, so it is reproduced explicitly ---------
#
# TINY is byte-level BPE: decode([id]) keeps the leading space, so the old
# accumulate-decoded-lengths resolver happens to be exact on it and every test
# above passes on the broken implementation. SentencePiece (Mistral) drops the
# space marker, and there the same code landed on the answer in 1 of 82 real
# turns. This is the second time a tiny random test model has hidden a real bug
# by not sharing the property that triggers it (cf. the final-norm gamma).
#
# The stub below has SentencePiece's decode behaviour and a correct offset
# mapping, so it pins the fix without downloading a 7B tokenizer.

class _SentencePieceLikeTokenizer:
    """Whitespace tokenizer whose decode drops the leading space, as SP does."""

    def __init__(self):
        self.is_fast = True

    def _split(self, text):
        spans, i = [], 0
        for piece in text.split(" "):
            spans.append((i, i + len(piece)))
            i += len(piece) + 1
        return [s for s in spans if s[1] > s[0]]

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        spans = self._split(text)
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out

    def encode(self, text, add_special_tokens=True):
        return list(range(len(self._split(text))))

    def decode(self, ids):
        # The bug under test: the space that separated this piece is gone.
        return "X" * 3


def test_offsets_are_used_rather_than_decoded_piece_lengths():
    """With a tokenizer whose decode() loses characters, an accumulator drifts
    and lands on the wrong token; the offset mapping stays exact."""
    tok = _SentencePieceLikeTokenizer()
    prompt = "the word that best matches the hint is "
    text = "definitely the answer MOON here"
    p = answer_position(tok, prompt, text, "MOON")
    spans = tok(prompt + text, return_offsets_mapping=True)["offset_mapping"]
    start, end = spans[p]
    assert (prompt + text)[start:end] == "MOON"
