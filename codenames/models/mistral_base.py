"""Mistral-7B-v0.2 BASE loader (pre-submission task T5).

The base checkpoint that Mistral-7B-Instruct-v0.2 was fine-tuned from
(32k context, rope theta 1e6, no sliding window). Mistral published it as a
raw release; ``mistral-community/Mistral-7B-v0.2`` is the HF conversion of
those official weights and is the identifier used here, stated in the paper.
(``mistralai/Mistral-7B-v0.1`` is a DIFFERENT pretraining run -- rope theta
1e4, sliding window -- and would not isolate the effect of instruction
tuning.) Prompted with the raw instruction body: no chat template exists for
a base model, and the readout is geometric only (``--no-generation``).
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ._causal_loader import load_causal_decoder

MODEL_NAME = "mistral-community/Mistral-7B-v0.2"
PREFIX = "mistral_base"
CHAT_TEMPLATE_STRATEGY = "raw"


def load_mistral_base(attn_implementation: Optional[str] = None,
                      ) -> Tuple[Any, Any, Dict[str, Any]]:
    return load_causal_decoder(
        model_name=MODEL_NAME, prefix=PREFIX,
        chat_template_strategy=CHAT_TEMPLATE_STRATEGY,
        torch_dtype=torch.float16, attn_implementation=attn_implementation)
