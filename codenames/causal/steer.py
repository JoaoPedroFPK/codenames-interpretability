"""Steering directions and residual-stream injection (causal_spec.md §5B).

Two candidate directions, and they are NOT equivalent:

* ``direction_from_unembedding`` (d_lens) is built from the unembedding rows of
  the candidate tokens and is LABEL-FREE. It is PRIMARY, because a sufficiency
  result obtained with a label-free direction is what supports the
  triangulation claim.
* ``direction_diff_of_means`` (d_DoM) is fitted on human target labels, so it is
  supervised. It is the robustness arm and requires the shuffled-label control
  (§3.3.7) plus a disjoint fit/eval split (§4.2). A random direction does NOT
  control for label fitting, which is why ``shuffled_label_direction`` exists.

Sufficiency is claimed only when the effect beats BOTH the random-direction
control and the counterfactual-target specificity control (§3.3.8): with an
11-21 word candidate pool, a large enough alpha moves the model toward any
nominated word, so efficacy alone shows the intervention does something, not
that it does the specific thing.
"""

from typing import Optional, Sequence

import numpy as np
import torch

from .patch import _decoder_layers

INJECTION_SITES = ("from_hint", "hint_only", "generating")


def direction_diff_of_means(
    states_target: np.ndarray, states_distractor: np.ndarray
) -> np.ndarray:
    """Supervised direction: mean(target contexts) - mean(distractor contexts)."""
    return np.asarray(states_target).mean(0) - np.asarray(states_distractor).mean(0)


def direction_from_unembedding(
    unembed: np.ndarray, token_ids: Sequence[int]
) -> np.ndarray:
    """Label-free direction: the mean unembedding row of the candidate tokens."""
    return np.asarray(unembed)[list(token_ids)].mean(0)


def shuffled_label_direction(
    states: np.ndarray, labels: np.ndarray, *, seed: int = 2026
) -> np.ndarray:
    """Difference of means with target/non-target labels permuted within turn."""
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(np.asarray(labels).astype(bool))
    return direction_diff_of_means(np.asarray(states)[permuted], np.asarray(states)[~permuted])


def random_direction(dim: int, *, norm: float, seed: int = 2026) -> np.ndarray:
    """Norm-matched random direction (the §3.3.5 control)."""
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim)
    return (vector / np.linalg.norm(vector) * norm).astype(np.float32)


def _injection_slice(sites: str, n_tokens: int, hint_start: int) -> slice:
    if sites == "generating":
        return slice(n_tokens - 1, n_tokens)
    if sites == "hint_only":
        return slice(hint_start, hint_start + 1)
    if sites == "from_hint":
        return slice(hint_start, n_tokens)
    raise ValueError(f"unknown sites {sites!r}; expected one of {INJECTION_SITES}")


def steer_generate(
    *,
    model,
    tokenizer,
    prompt: str,
    layer: int,
    direction: np.ndarray,
    alpha: float,
    sites: str = "from_hint",
    max_new_tokens: int = 32,
    device: str = "cpu",
    hint_start: int = 0,
) -> str:
    """Greedily generate with ``alpha * direction`` added at ``layer``."""
    if sites not in INJECTION_SITES:
        raise ValueError(f"unknown sites {sites!r}; expected one of {INJECTION_SITES}")

    layers = _decoder_layers(model)
    vector = torch.tensor(
        np.asarray(direction, dtype=np.float32) * float(alpha), device=device
    )

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        span = _injection_slice(sites, hidden.shape[1], hint_start)
        hidden[:, span, :] = hidden[:, span, :] + vector.to(hidden.dtype)
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden

    module = model.get_input_embeddings() if layer == 0 else layers[layer - 1]
    handle = module.register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    return tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
