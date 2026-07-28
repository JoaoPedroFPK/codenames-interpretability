"""Residual-stream activation patching (causal_spec.md §5A).

Denoising direction: clean states are written into a corrupted run and the
restored target signal is measured at p*. Residual-stream patching yields the
TOTAL effect routed through a given depth and position; it does not separate
direct from indirect paths (path patching is a stretch goal, §9), so wording
elsewhere is scoped to "total effect through the residual stream at depth L".

Layer indexing follows ``output_hidden_states``: index 0 is the embedding
output and index i+1 is the output of decoder block i. Layer 0 is patched by
hooking the embedding module, so a site list containing only layer 0 still
patches something rather than silently doing nothing.

``ld_clean`` is measured with a real forward pass on the clean prompt, never
derived from the full-stack patch. That distinction is what keeps the P2
identity (full-stack patch => e == 1) a genuine test of the machinery instead
of a tautology.
"""

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .metrics import logit_difference, normalized_effect

Site = Tuple[int, int]

_LAYER_LIST_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")
_FINAL_NORM_PATHS = ("model.norm", "transformer.ln_f", "gpt_neox.final_layer_norm",
                     "model.decoder.final_layer_norm")


def layer_window(center: int, width: int, n_layers: int) -> List[int]:
    """Contiguous band of layers centred on ``center``, clipped to the stack."""
    half = width // 2
    return [layer for layer in range(center - half, center + half + 1) if 0 <= layer < n_layers]


def all_sites(*, n_layers: int, n_positions: int) -> List[Site]:
    """Every (layer, position) cell of the grid, row-major."""
    return [(layer, pos) for layer in range(n_layers) for pos in range(n_positions)]


def _final_norm(model):
    """The norm applied after the last block.

    ``output_hidden_states`` returns the POST-norm state as its last entry, so
    patching that index must target this module -- not the last block's output,
    which is pre-norm. Writing a post-norm value into the pre-norm slot makes
    the model apply the norm twice. The error is invisible when the norm is
    near-idempotent (gamma about 1) and was masked on both the tiny test model
    and Mistral; it cost ~14% of the full-stack identity on Qwen.
    """
    for path in _FINAL_NORM_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    return None


def _decoder_layers(model):
    """The module list whose outputs are the residual stream after each block."""
    for path in _LAYER_LIST_PATHS:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("could not locate the decoder layer list on this model")


def patch_hook(positions: Sequence[int], values: torch.Tensor) -> Callable:
    """Forward hook overwriting the residual stream at the given positions."""
    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        for slot, position in enumerate(positions):
            hidden[:, position, :] = values[slot].to(hidden.dtype).to(hidden.device)
        return (hidden,) + tuple(output[1:]) if is_tuple else hidden
    return hook


def _register(model, layers, sites: Sequence[Site], clean_cache) -> List:
    """Hook the module that produces each cached index."""
    by_layer: Dict[int, List[int]] = {}
    for layer, position in sites:
        by_layer.setdefault(layer, []).append(position)

    handles = []
    top = len(clean_cache) - 1          # index of the post-final-norm state
    final_norm = _final_norm(model)
    for layer, positions in by_layer.items():
        values = torch.stack([clean_cache[layer][0, p] for p in positions])
        if layer == 0:
            module = model.get_input_embeddings()
        elif layer == top and final_norm is not None:
            module = final_norm
        else:
            module = layers[layer - 1]
        handles.append(module.register_forward_hook(patch_hook(positions, values)))
    return handles


def run_patch(
    *,
    model,
    tokenizer,
    clean_cache: Sequence[torch.Tensor],
    corrupt_prompt: str,
    sites: Sequence[Site],
    readout_table: Dict[str, List[int]],
    clean_target: str,
    donor_target: str,
    p_star: int,
    device: str = "cpu",
    clean_prompt: Optional[str] = None,
    ld_clean: Optional[float] = None,
    ld_corrupt: Optional[float] = None,
) -> float:
    """Patch ``sites`` from the clean cache into the corrupted run; return e.

    In production ``ld_clean`` and ``ld_corrupt`` are supplied once per turn
    from the cached clean and corrupted runs, so each patch costs exactly one
    forward pass. Passing ``clean_prompt`` instead lets the function stand
    alone in tests at the cost of one extra forward.
    """
    layers = _decoder_layers(model)
    corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(device)

    def _logits(active_sites: Sequence[Site], inputs) -> np.ndarray:
        handles = _register(model, layers, active_sites, clean_cache) if active_sites else []
        try:
            with torch.no_grad():
                out = model(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        return out.logits[0, p_star].detach().float().cpu().numpy()

    def _ld(active_sites, inputs) -> float:
        return logit_difference(
            _logits(active_sites, inputs), readout_table, clean_target, donor_target
        )

    if ld_corrupt is None:
        ld_corrupt = _ld([], corrupt_inputs)
    if ld_clean is None:
        if clean_prompt is None:
            raise ValueError("run_patch needs clean_prompt or an explicit ld_clean")
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
        ld_clean = _ld([], clean_inputs)

    ld_patched = _ld(sites, corrupt_inputs) if sites else ld_corrupt
    return normalized_effect(
        ld_patched=ld_patched, ld_clean=ld_clean, ld_corrupt=ld_corrupt
    )
