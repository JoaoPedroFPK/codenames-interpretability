"""Llama-3.1-8B-Instruct loader (pre-submission task T6): the third
instruction-tuned decoder, run through geometry, lens and (time permitting)
patching. bf16 (the dtype it was released in); prompt through its own chat
template via the ``llama3`` strategy (frozen date string, single BOS).
Gated on the Hub: the Colab session needs an HF token with access.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ._causal_loader import load_causal_decoder

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PREFIX = "llama"
CHAT_TEMPLATE_STRATEGY = "llama3"


def load_llama_instruct(attn_implementation: Optional[str] = None,
                        ) -> Tuple[Any, Any, Dict[str, Any]]:
    return load_causal_decoder(
        model_name=MODEL_NAME, prefix=PREFIX,
        chat_template_strategy=CHAT_TEMPLATE_STRATEGY,
        torch_dtype=torch.bfloat16, attn_implementation=attn_implementation)
