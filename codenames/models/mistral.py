"""Mistral-7B-Instruct-v0.2 loader.

Mirrors Cell 2 of ``reference_notebooks/Mistral_Codenames_Layer_Extraction.ipynb``
verbatim. fp16, no device_map, output_hidden_states set at inference time
inside ``run_instance``.

``attn_implementation`` defaults to None, which preserves the original
notebook behavior (HuggingFace picks the default for the model class —
typically ``"eager"`` for Mistral). Pass ``"flash_attention_2"`` to opt
into FA2 when ``flash_attn`` is available; this is exercised via the
``compare`` subcommand with the ``--flash-attn`` flag.
"""

from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _flash_attn_importable() -> bool:
    """Probe whether flash_attn is importable. No side effects."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def load_mistral_instruct(
    attn_implementation: Optional[str] = None,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load Mistral-7B-Instruct-v0.2.

    Parameters
    ----------
    attn_implementation
        If ``None`` (default), uses the HuggingFace default for the model
        class — equivalent to the original notebook behavior. If
        ``"flash_attention_2"``, requires ``flash_attn`` to be importable;
        falls back to ``"eager"`` with a warning otherwise.

    Returns
    -------
    (model, tokenizer, metadata)
        ``metadata`` keys: ``num_layers``, ``hidden_dim``, ``device``,
        ``model_name``, ``prefix``, ``chat_template_strategy``,
        ``supports_generation``, ``forward_hidden_states_mode``,
        ``use_truncation``, ``attn_implementation`` (the value actually used).
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve attn_implementation: requested or default. If FA2 was
    # requested but flash_attn is not importable, fall back to eager.
    resolved_attn = attn_implementation
    if resolved_attn == "flash_attention_2" and not _flash_attn_importable():
        print(
            "WARNING: attn_implementation='flash_attention_2' requested but "
            "flash_attn is not importable. Falling back to 'eager'."
        )
        resolved_attn = "eager"

    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.float16,
        "device_map": None,
        "low_cpu_mem_usage": False,
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
        "attn_implementation": resolved_attn,
    }

    return model, tokenizer, metadata
