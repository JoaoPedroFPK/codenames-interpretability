"""BERT-base-uncased loader, random init (negative control).

Mirrors Cell 2 of
``reference_notebooks/Random_Baseline_Codenames_Layer_Extraction.ipynb``
verbatim. The tokenizer is loaded from the pretrained bert-base-uncased
checkpoint (tokenization is deterministic and weight-independent); the model
is constructed via ``BertModel(BertConfig(output_hidden_states=True))`` so
that no pretrained weights are loaded.

Includes an embedding-std verification block — if the embedding std looks
like a pretrained model (>= 0.1), the loader raises. This guards against
accidental pretrained loads.
"""

from typing import Any, Dict, Tuple

import torch
from transformers import BertConfig, BertModel, BertTokenizerFast


def load_bert_random(random_seed: int = 42) -> Tuple[Any, Any, Dict[str, Any]]:
    """Instantiate a randomly-initialised BERT-base-uncased."""
    tokenizer_name = "bert-base-uncased"
    model_name = "bert-base-uncased (random init)"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer: {tokenizer_name}")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # BertConfig defaults match bert-base-uncased; output_hidden_states is
    # set on the config because from_pretrained is not called.
    config = BertConfig(output_hidden_states=True)

    # Re-fix seed immediately before model construction for weight reproducibility.
    torch.manual_seed(random_seed)

    # RANDOM weights — NEVER from_pretrained.
    model = BertModel(config)
    model = model.to(device)
    model.eval()

    num_layers = config.num_hidden_layers
    hidden_dim = config.hidden_size

    print()
    print("Model initialized with RANDOM weights (no pretraining).")
    print(f"Number of transformer layers : {num_layers}")
    print(f"Hidden state dimensionality  : {hidden_dim}")
    print(f"Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} layers)")
    print(f"Total parameters             : {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Random init: word embedding weights ~ truncated_normal(std=0.02).
    # Pretrained:  word embedding weights have std ~0.3-0.5.
    emb_std = model.embeddings.word_embeddings.weight.std().item()
    print(
        f"Embedding weight std: {emb_std:.4f}  "
        f"(expect ~0.02 for random init, ~0.3-0.5 for pretrained)"
    )
    if emb_std < 0.1:
        print("CONFIRMED: Weights are randomly initialized (not pretrained).")
    else:
        raise RuntimeError(
            f"Embedding std={emb_std:.4f} is too high for random init. "
            "Pretrained weights may have been loaded by mistake. "
            "Verify BertModel(config) is used and not BertModel.from_pretrained(...)."
        )

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "tokenizer_name": tokenizer_name,
        "prefix": "random_bert",
        "chat_template_strategy": "raw",
        "supports_generation": False,
        "forward_hidden_states_mode": "encoder_load_time",
        "use_truncation": True,
    }

    return model, tokenizer, metadata
