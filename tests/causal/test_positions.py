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
