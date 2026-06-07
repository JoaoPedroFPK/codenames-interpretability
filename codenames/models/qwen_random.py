"""Random Qwen2.5-7B loader (architecture-only, random weights).

The canonical init pattern uses
``accelerate.init_empty_weights()`` + ``to_empty(device=...)`` +
``model.apply(model._init_weights)`` plus an explicit RMSNorm re-init loop
(because Qwen2's ``_init_weights`` does not handle RMSNorm and the
constructor-set values of 1.0 are wiped by ``to_empty()``).

This loader is the spec-driven path (user-resolved decision 2026-05-18):
Random Qwen is the seventh model and has not yet been run on N=2000, so
there is no existing bit-identity constraint. The package output here is
the new ground truth.

The tokenizer is loaded from the pretrained checkpoint — tokenization is
weight-independent. The random seed is re-set immediately before model
construction so the resulting weights are deterministic.
"""

from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_qwen_random(random_seed: int = 42) -> Tuple[Any, Any, Dict[str, Any]]:
    """Instantiate a randomly-initialised Qwen2.5-7B with the trained architecture.

    Parameters
    ----------
    random_seed
        Seed re-applied immediately before weight construction. Defaults to
        the contract value (42).
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nLoading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Fetching architecture config from: {model_name}")
    config = AutoConfig.from_pretrained(model_name)

    # Defensive re-seed: ensures the weight init below is deterministic
    # regardless of any prior RNG consumption.
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print(f"Instantiating model on meta device (seed={random_seed})...")
    print(f"  initializer_range from config: {getattr(config, 'initializer_range', 'N/A')}")

    # Construct the architecture on the meta device — no memory allocation.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    # Materialise on GPU. to_empty leaves parameters uninitialised (the meta-tensor
    # storage is replaced with real allocations whose contents are arbitrary).
    model = model.to_empty(device=device)

    # Populate weights via the model's own _init_weights. For Linear and
    # Embedding this samples from N(0, initializer_range^2); LayerNorm/RMSNorm
    # weights are set by their own reset_parameters, which Qwen2's
    # _init_weights does NOT call.
    model.apply(model._init_weights)

    # Explicit RMSNorm re-init: Qwen2RMSNorm.__init__ sets weight to 1.0, but
    # to_empty() wiped that constructor value, and _init_weights does not
    # touch RMSNorm. Restore the post-init value of 1.0.
    _rmsnorm_count = 0
    for module in model.modules():
        cls_name = module.__class__.__name__
        if cls_name.endswith("RMSNorm"):
            with torch.no_grad():
                module.weight.fill_(1.0)
            _rmsnorm_count += 1

    print(f"  RMSNorm modules re-initialised : {_rmsnorm_count}")

    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    print("\nModel instantiated successfully.")
    print(f"  Number of transformer layers : {num_layers}")
    print(f"  Hidden state dimensionality  : {hidden_dim}")
    print(f"  Total hidden states per token: {num_layers + 1}  (embedding + {num_layers} layers)")
    print(f"  Vocabulary size              : {model.config.vocab_size}")
    print(f"  dtype                        : {next(model.parameters()).dtype}")
    print(f"  Device                       : {next(model.parameters()).device}")

    # Quick parameter sanity: the first Linear weight should have std close
    # to initializer_range. If it doesn't, something has gone wrong with the
    # init flow.
    _first_linear = None
    for _m in model.modules():
        if isinstance(_m, torch.nn.Linear):
            _first_linear = _m
            break
    if _first_linear is not None:
        _w = _first_linear.weight.detach().float()
        print(
            f"\n  Sanity (first Linear weight): mean={_w.mean().item():+.6f}, "
            f"std={_w.std().item():.6f}"
        )
        print(f"  Expected std ≈ {config.initializer_range} (random init)")

    metadata = {
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "device": device,
        "model_name": model_name,
        "prefix": "random_qwen",
        "chat_template_strategy": "chatml",
        "supports_generation": False,
        "forward_hidden_states_mode": "causal",
        "use_truncation": False,
    }

    return model, tokenizer, metadata
