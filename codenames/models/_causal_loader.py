"""Shared loader for the additional causal decoders (base models, Llama).

Mirrors ``mistral.py`` / ``qwen.py`` exactly (dtype, no device_map, hidden
states requested at inference time, FA2 opt-in with sdpa fallback) so the
new models inherit the same extraction path; only the identifiers and the
prompt strategy differ, and those live in the per-model modules.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _flash_attn_importable() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def load_causal_decoder(
    *,
    model_name: str,
    prefix: str,
    chat_template_strategy: str,
    torch_dtype=torch.float16,
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_attn = attn_implementation
    if resolved_attn == "flash_attention_2" and not _flash_attn_importable():
        print("WARNING: attn_implementation='flash_attention_2' requested but "
              "flash_attn is not importable. Falling back to 'sdpa'.")
        resolved_attn = "sdpa"

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype, "device_map": None, "low_cpu_mem_usage": False,
    }
    if resolved_attn is not None:
        load_kwargs["attn_implementation"] = resolved_attn
        print(f"Attention implementation: {resolved_attn}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print("Model loaded successfully.")
    print(f"Number of transformer layers : {num_layers}")
    print(f"Hidden state dimensionality  : {hidden_dim}")
    print(f"Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} layers)")
    print(f"Vocabulary size              : {model.config.vocab_size}")

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": prefix,
        "chat_template_strategy": chat_template_strategy,
        "supports_generation": True,
        "forward_hidden_states_mode": "causal",
        "use_truncation": False,
        "attn_implementation": resolved_attn,
    }
    return model, tokenizer, metadata
