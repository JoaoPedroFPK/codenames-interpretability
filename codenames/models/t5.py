"""T5-base encoder loader.

Mirrors Cell 2 of ``reference_notebooks/T5_Codenames_Layer_Extraction.ipynb``
verbatim. T5EncoderModel only (no decoder); fp32 (T5's native precision —
fp16 causes overflow in some T5 layers). ``output_hidden_states=True`` is set
at LOAD time on the model config.

Note: the T5 config uses ``num_layers`` (not ``num_hidden_layers``) and
``d_model`` (not ``hidden_size``); the metadata exposes these under the
common names so downstream code doesn't need to special-case T5.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import T5EncoderModel, T5TokenizerFast


def load_t5_encoder() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load T5-base encoder."""
    model_name = "t5-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = T5TokenizerFast.from_pretrained(model_name)

    model = T5EncoderModel.from_pretrained(
        model_name,
        output_hidden_states=True,
    ).to(device)

    model.eval()

    num_layers = model.config.num_layers
    hidden_dim = model.config.d_model

    print("Model loaded successfully.")
    print(f"Number of encoder layers     : {num_layers}")
    print(f"Hidden state dimensionality  : {hidden_dim}")
    print(f"Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} encoder layers)")
    print("Architecture                 : T5 encoder (bidirectional, RPE)")
    print(f"Total params (encoder only)  : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": "t5",
        "chat_template_strategy": "raw",
        "supports_generation": False,
        "forward_hidden_states_mode": "encoder_load_time",
        "use_truncation": True,
    }

    return model, tokenizer, metadata
