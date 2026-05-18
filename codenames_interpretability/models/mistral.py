"""Mistral-7B-Instruct-v0.2 loader.

Mirrors Cell 2 of ``reference_notebooks/Mistral_Codenames_Layer_Extraction.ipynb``
verbatim. fp16, no device_map, output_hidden_states set at inference time
inside ``run_instance``.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_mistral_instruct() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load Mistral-7B-Instruct-v0.2.

    Returns
    -------
    (model, tokenizer, metadata)
        ``metadata`` keys: ``num_layers``, ``hidden_dim``, ``device``,
        ``model_name``, ``prefix``, ``chat_template_strategy``,
        ``supports_generation``, ``forward_hidden_states_mode``,
        ``use_truncation``.
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
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

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": "mistral",
        "chat_template_strategy": "mistral_inst",
        "supports_generation": True,
        "forward_hidden_states_mode": "causal",
        "use_truncation": False,
    }

    return model, tokenizer, metadata
