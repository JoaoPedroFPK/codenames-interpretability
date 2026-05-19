"""ModernBERT-base loader.

Mirrors Cell 2 of
``reference_notebooks/ModernBERT_Codenames_Layer_Extraction.ipynb`` verbatim.

Distinguishing features within the encoder family:
- fp16 (encoder, but the model is large enough that fp32 is wasteful)
- ``output_hidden_states`` is passed at inference time, NOT on the config —
  matches the AutoModel/causal-LM convention, different from BertModel/T5
- Alternating global / local attention; ``GLOBAL_LAYERS`` and ``LOCAL_LAYERS``
  are computed at load time and exposed in metadata for downstream analysis
  (e.g., SC6 plots coloured by attention type)
- Flash Attention 2 used when available; falls back to ``"eager"`` otherwise.
  The notebook installs ``flash-attn`` in its Cell 0; the package probes at
  runtime instead.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModel, AutoTokenizer


def _flash_attn_available() -> bool:
    """Probe whether flash_attn is importable. No side effects."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def load_modernbert() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load ModernBERT-base."""
    model_name = "answerdotai/ModernBERT-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    flash_available = _flash_attn_available()
    attn_impl = "flash_attention_2" if flash_available else "eager"
    print(f"Attention implementation: {attn_impl}")

    # AutoModel: encoder-only path. output_hidden_states is NOT set here; it
    # is passed at inference time in run_instance.
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation=attn_impl,
    ).to(device)

    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # In ModernBERT-base: global every 3rd layer (0-indexed: 0, 3, 6, ..., 21).
    # Local layers are all others (sliding window of 128 tokens).
    global_attn_every_n = getattr(model.config, "global_attn_every_n_layers", 3)
    global_layers = [i for i in range(num_layers) if i % global_attn_every_n == 0]
    local_layers = [i for i in range(num_layers) if i % global_attn_every_n != 0]
    local_window = getattr(model.config, "local_attention", 128)

    print("Model loaded successfully.")
    print(f"Number of transformer layers : {num_layers}")
    print(f"Hidden state dimensionality  : {hidden_dim}")
    print(f"Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} transformer layers)")
    print("Architecture                 : Encoder-only with alternating attention")
    print(f"  Global layers (full attn)  : {global_layers}")
    print(f"  Local layers ({local_window}-tok window): {local_layers}")

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": "modernbert",
        "chat_template_strategy": "raw",
        "supports_generation": False,
        "forward_hidden_states_mode": "encoder_inference",
        "use_truncation": True,
        "global_layers": global_layers,
        "local_layers": local_layers,
        "local_window": local_window,
        "attn_implementation": attn_impl,
    }

    return model, tokenizer, metadata
