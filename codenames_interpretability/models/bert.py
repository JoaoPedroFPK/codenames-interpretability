"""BERT-base-uncased loader.

Mirrors Cell 2 of ``reference_notebooks/BERT_Codenames_Layer_Extraction.ipynb``
verbatim. Loaded in fp32 (BERT's native precision). ``output_hidden_states=True``
is set at LOAD time — this is the BertModel/BertConfig convention and is
different from the AutoModel/AutoModelForCausalLM convention used by Mistral,
Qwen, Random Qwen, and ModernBERT.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import BertModel, BertTokenizerFast


def load_bert_base() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load BERT-base-uncased."""
    model_name = "bert-base-uncased"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer and model: {model_name}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    model = BertModel.from_pretrained(
        model_name,
        output_hidden_states=True,
    ).to(device)

    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print("Model loaded successfully.")
    print(f"Number of transformer layers : {num_layers}")
    print(f"Hidden state dimensionality  : {hidden_dim}")
    print(f"Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} transformer layers)")
    print("Architecture                 : Bidirectional encoder")

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": "bert",
        "chat_template_strategy": "raw",
        "supports_generation": False,
        "forward_hidden_states_mode": "encoder_load_time",
        "use_truncation": True,
    }

    return model, tokenizer, metadata
