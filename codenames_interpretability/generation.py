"""Causal-only generation utility.

Verbatim from Cell 7 of the Mistral and Qwen reference notebooks. Mistral and
Qwen are byte-identical here except for an inline comment about which model
"echoes the hint"; this file uses the more general Qwen comment.

The parsing logic (whole-word matching, hint exclusion, first-by-character-position)
is methodologically committed in the thesis and must not change.
"""

import re
from typing import Dict, List


def generate_response(
    *,
    prompt: str,
    candidates: List[str],
    max_new_tokens: int,
    model,
    tokenizer,
    device: str,
) -> Dict[str, object]:
    """Generate a continuation and parse the first in-candidate word.

    Matching strategy: find ALL candidate words in the response, return the
    FIRST one that appears (by character position). The hint word itself is
    excluded from matching — instruction-tuned causal LMs typically echo the
    hint in quotes before answering.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]

    import torch
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids  = output_ids[0, prompt_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Extract hint to exclude from matching (instruction-tuned LMs often echo
    # the hint in quotes).
    hint_match = re.search(r'hint\s+is\s*:?\s*"([^"]+)"', prompt, re.IGNORECASE)
    hint_word = hint_match.group(1).lower() if hint_match else None

    candidates_lower = {c.lower(): c for c in candidates}

    occurrences = []  # list of (position, candidate)
    gen_lower = generated_text.lower()

    for cand_lower, cand_original in candidates_lower.items():
        if hint_word and cand_lower == hint_word:
            continue
        for match in re.finditer(rf"\b{re.escape(cand_lower)}\b", gen_lower):
            occurrences.append((match.start(), cand_original))

    matched_word = None
    n_candidates_in_response = 0

    if occurrences:
        occurrences.sort(key=lambda x: x[0])
        seen = set()
        unique_mentions = []
        for pos, cand in occurrences:
            if cand not in seen:
                seen.add(cand)
                unique_mentions.append((pos, cand))
        n_candidates_in_response = len(unique_mentions)
        _, matched_word = unique_mentions[0]

    return {
        "generated_text"           : generated_text,
        "generated_word"           : matched_word,
        "generated_in_candidates"  : matched_word is not None,
        "n_candidates_in_response" : n_candidates_in_response,
    }
