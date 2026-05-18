"""Prompt construction.

The ``instruction_body`` is byte-identical across all seven reference
notebooks (Format A from CONTRACT_v1.0 Section 2). Only the per-model chat-template
wrapping differs. Three strategies are supported:

- ``"mistral_inst"`` — Mistral-Instruct scaffolding ``<s>[INST] {body} [/INST]``,
  hand-rolled (does NOT call ``apply_chat_template``).
- ``"chatml"`` — ``tokenizer.apply_chat_template`` with a system message and
  ``add_generation_prompt=True``. Used by Qwen and Random Qwen.
- ``"raw"`` — the instruction body verbatim with no wrapping. Used by BERT,
  BERT Random, T5, and ModernBERT (encoder-only).

The trailing instruction ``"Only output the word."`` is preserved for every
model including encoder-only ones — by design, an identical input string is
the experimental control.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

# Per-model system message for the ``chatml`` strategy. Qwen and Random Qwen
# both use the same string, taken verbatim from their Cell 5.
_CHATML_SYSTEM_MESSAGE = "You are a helpful assistant."


_FEATURE_LABEL_MAP: Dict[str, str] = {
    "giver.marriage":  "Marriage",
    "giver.education": "Education",
    "giver.race":      "Race",
    "giver.continent": "Continent",
    "giver.language":  "Language",
    "giver.religion":  "Religion",
    "giver.gender":    "Gender",
    "giver.country":   "Country",
    "giver.political": "Politics",
    "giver.politics":  "Politics",  # T5 dataset variant; preserved for back-compat
}


def _format_feature_value(v) -> str:
    """Format a feature value as a clean string. Returns 'NA' for missing.

    Verbatim from Cell 5 of every reference notebook.
    """
    if pd.isna(v):
        return "NA"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v)


def build_instruction_body(
    hint: str,
    candidates: List[str],
    giver_features: Optional[Dict[str, object]],
    use_social_context: bool,
) -> Tuple[str, Dict[str, str]]:
    """Construct the shared instruction body (Format A from CONTRACT_v1.0 §2).

    Byte-identical across all seven reference notebooks. Returns the body plus
    a dict mapping each giver-feature column key to the exact substring
    written into the prompt — span detection uses this to locate the feature
    spans afterwards.
    """
    words_block = "\n".join(
        [f"{i+1}. {w}" for i, w in enumerate(candidates)]
    )
    feature_markers: Dict[str, str] = {}

    if use_social_context and giver_features:
        parts = []
        for k, v in giver_features.items():
            v_str = _format_feature_value(v)
            label = _FEATURE_LABEL_MAP.get(k, k.split(".")[-1].capitalize())
            marker = f"{label}: {v_str}"
            feature_markers[k] = marker
            parts.append(marker)
        social_block = ", ".join(parts) if parts else "N/A"

        instruction_body = (
            f"You are playing the game Codenames.\n"
            f"The clue was given by a player with the following characteristics:\n"
            f"{social_block}\n"
            f'The hint is: "{hint}"\n'
            f"The possible words are:\n"
            f"{words_block}\n"
            f"Which word best matches the hint? Only output the word."
        )
    else:
        instruction_body = (
            f"You are playing the game Codenames.\n"
            f'The hint is: "{hint}"\n'
            f"The possible words are:\n"
            f"{words_block}\n"
            f"Which word best matches the hint? Only output the word."
        )

    return instruction_body, feature_markers


def build_prompt(
    hint: str,
    candidates: List[str],
    giver_features: Optional[Dict[str, object]],
    use_social_context: bool,
    tokenizer,
    chat_template_strategy: str,
) -> Tuple[str, Dict[str, str]]:
    """Build the full prompt for one (hint, candidates, condition) triple.

    Dispatches on ``chat_template_strategy``:

    - ``"mistral_inst"``: wraps the body as ``<s>[INST] {body} [/INST]`` per
      the canonical Mistral Cell 5. The tokenizer argument is unused for this
      strategy but kept in the signature for uniformity.
    - ``"chatml"``: builds a ``[{"role": "system", ...}, {"role": "user", ...}]``
      message list with the canonical Qwen system message
      (``"You are a helpful assistant."``) and calls
      ``tokenizer.apply_chat_template(..., tokenize=False,
      add_generation_prompt=True)``.
    - ``"raw"``: returns the instruction body verbatim with no wrapping.
    """
    instruction_body, feature_markers = build_instruction_body(
        hint=hint,
        candidates=candidates,
        giver_features=giver_features,
        use_social_context=use_social_context,
    )

    if chat_template_strategy == "mistral_inst":
        prompt = f"<s>[INST] {instruction_body} [/INST]"
    elif chat_template_strategy == "chatml":
        messages = [
            {"role": "system", "content": _CHATML_SYSTEM_MESSAGE},
            {"role": "user",   "content": instruction_body},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif chat_template_strategy == "raw":
        prompt = instruction_body
    else:
        raise ValueError(
            f"Unknown chat_template_strategy: {chat_template_strategy!r}. "
            "Expected one of: 'mistral_inst', 'chatml', 'raw'."
        )

    return prompt, feature_markers
