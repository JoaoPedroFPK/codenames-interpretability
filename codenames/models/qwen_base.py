"""Qwen2.5-7B BASE loader (pre-submission task T5).

The pretrained checkpoint Qwen2.5-7B-Instruct was tuned from. Raw
instruction body (no chat template); geometric readout only.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from ._causal_loader import load_causal_decoder

MODEL_NAME = "Qwen/Qwen2.5-7B"
PREFIX = "qwen_base"
CHAT_TEMPLATE_STRATEGY = "raw"


def load_qwen_base(attn_implementation: Optional[str] = None,
                   ) -> Tuple[Any, Any, Dict[str, Any]]:
    return load_causal_decoder(
        model_name=MODEL_NAME, prefix=PREFIX,
        chat_template_strategy=CHAT_TEMPLATE_STRATEGY,
        torch_dtype=torch.float16, attn_implementation=attn_implementation)
