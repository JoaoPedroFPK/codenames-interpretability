"""Tests for prompts.py — no model dependency.

These tests verify the byte-identical instruction body and the three
chat-template strategies. The chatml strategy is tested with a stub
tokenizer; the real tokenizers aren't loaded.
"""

from codenames_interpretability.prompts import (
    _FEATURE_LABEL_MAP,
    _format_feature_value,
    build_instruction_body,
    build_prompt,
)


def test_feature_label_map_has_both_political_keys():
    """Both `giver.political` and `giver.politics` map to 'Politics'."""
    assert _FEATURE_LABEL_MAP["giver.political"] == "Politics"
    assert _FEATURE_LABEL_MAP["giver.politics"] == "Politics"


def test_format_feature_value_handles_floats():
    """Trailing zeros and trailing decimal points are stripped."""
    assert _format_feature_value(1.0) == "1"
    assert _format_feature_value(1.5000) == "1.5"
    assert _format_feature_value(1.2345) == "1.2345"


def test_instruction_body_no_social_is_stable():
    """The no_social body matches the canonical Format A shape."""
    body, markers = build_instruction_body(
        hint="cat",
        candidates=["mouse", "dog", "fish"],
        giver_features=None,
        use_social_context=False,
    )
    assert markers == {}
    assert 'The hint is: "cat"' in body
    assert "1. mouse\n2. dog\n3. fish" in body
    assert body.endswith("Only output the word.")
    assert "characteristics" not in body


def test_instruction_body_with_social_includes_markers():
    """The with_social body includes the giver-features preamble and the markers map."""
    body, markers = build_instruction_body(
        hint="cat",
        candidates=["mouse"],
        giver_features={"giver.gender": "female", "giver.political": "left"},
        use_social_context=True,
    )
    assert "characteristics" in body
    assert "Gender: female" in body
    assert "Politics: left" in body
    assert markers["giver.gender"] == "Gender: female"
    assert markers["giver.political"] == "Politics: left"


def test_build_prompt_mistral_inst_strategy():
    """The mistral_inst strategy wraps the body in <s>[INST] ... [/INST]."""
    prompt, _ = build_prompt(
        hint="cat",
        candidates=["mouse"],
        giver_features=None,
        use_social_context=False,
        tokenizer=None,  # not consulted for mistral_inst
        chat_template_strategy="mistral_inst",
    )
    assert prompt.startswith("<s>[INST] ")
    assert prompt.endswith(" [/INST]")


def test_build_prompt_raw_strategy():
    """The raw strategy returns the body verbatim."""
    body, _ = build_instruction_body(
        hint="cat", candidates=["mouse"], giver_features=None, use_social_context=False,
    )
    prompt, _ = build_prompt(
        hint="cat",
        candidates=["mouse"],
        giver_features=None,
        use_social_context=False,
        tokenizer=None,
        chat_template_strategy="raw",
    )
    assert prompt == body


def test_build_prompt_chatml_strategy_uses_tokenizer():
    """The chatml strategy delegates to tokenizer.apply_chat_template."""
    captured = {}

    class StubTokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            captured["messages"] = messages
            captured["tokenize"] = tokenize
            captured["add_generation_prompt"] = add_generation_prompt
            return "<wrapped>"

    prompt, _ = build_prompt(
        hint="cat",
        candidates=["mouse"],
        giver_features=None,
        use_social_context=False,
        tokenizer=StubTokenizer(),
        chat_template_strategy="chatml",
    )
    assert prompt == "<wrapped>"
    assert captured["tokenize"] is False
    assert captured["add_generation_prompt"] is True
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == "You are a helpful assistant."
    assert captured["messages"][1]["role"] == "user"


def test_build_prompt_unknown_strategy_raises():
    """An unknown strategy raises ValueError with the expected message."""
    import pytest
    with pytest.raises(ValueError, match="Unknown chat_template_strategy"):
        build_prompt(
            hint="cat",
            candidates=["mouse"],
            giver_features=None,
            use_social_context=False,
            tokenizer=None,
            chat_template_strategy="nonsense",
        )
