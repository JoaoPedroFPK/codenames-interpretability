"""Qwen2.5-7B-Instruct loader.

Mirrors Cell 2 of ``reference_notebooks/Qwen_Codenames_Layer_Extraction.ipynb``
verbatim. fp16, no device_map, output_hidden_states set at inference time
inside ``run_instance``. Uses ChatML via ``apply_chat_template`` with the
canonical system message ``"You are a helpful assistant."``.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_qwen_instruct() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load Qwen2.5-7B-Instruct."""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=False,
    ).to(device)

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
        "prefix": "qwen",
        "chat_template_strategy": "chatml",
        "supports_generation": True,
        "forward_hidden_states_mode": "causal",
        "use_truncation": False,
    }

    return model, tokenizer, metadata
