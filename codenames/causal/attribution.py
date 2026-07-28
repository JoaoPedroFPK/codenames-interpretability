"""Attribution patching: a first-order screen over the full grid.

Following the attribution-patching line (Nanda 2023; Syed et al., EAP;
Kramar et al., AtP*). The gradient of the metric with respect to the corrupted
activation, dotted with (clean - corrupt), approximates the real patch effect
at every site from ONE backward pass, which is what makes an exhaustive
layer x position scan affordable.

It is a SCREEN and carries no inferential claim (causal_spec.md §3.4). Being a
first-order approximation it is least reliable exactly where effects are
largest, so every reported locus is confirmed with a real patch and a random
10% of the grid is really patched regardless of score, to bound the
false-negative rate (§3.3.4). Reporting this grid as if it were causal would
be presenting an attribution heatmap as a patching heatmap.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .patch import _decoder_layers

Site = Tuple[int, int]


def top_sites(grid: np.ndarray, *, k: int) -> List[Site]:
    """The k highest-|score| cells, strongest first."""
    flat = np.abs(np.asarray(grid)).ravel()
    k = min(k, flat.size)
    picked = np.argpartition(flat, -k)[-k:]
    picked = picked[np.argsort(-flat[picked])]
    return [tuple(int(v) for v in np.unravel_index(i, grid.shape)) for i in picked]


def attribution_scan(
    *,
    model,
    tokenizer,
    clean_cache: Sequence[torch.Tensor],
    corrupt_prompt: str,
    readout_table: Dict[str, List[int]],
    clean_target: str,
    donor_target: str,
    p_star: int,
    device: str = "cpu",
) -> np.ndarray:
    """First-order patch-effect estimate for every (layer, position) cell."""
    layers = _decoder_layers(model)
    inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(device)
    n_layers = len(clean_cache)
    n_positions = int(inputs["input_ids"].shape[1])

    # Fail with a diagnosis rather than a broadcast error deep in the loop.
    cached_positions = int(clean_cache[0].shape[1])
    if cached_positions != n_positions:
        raise ValueError(
            f"clean cache has {cached_positions} positions but the corrupted "
            f"prompt has {n_positions}; patching (layer, position) across "
            "different-length sequences is undefined. Drop the pair upstream "
            "(see causal_spec.md §5A alignment rule)."
        )

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden.retain_grad()
            captured[layer_index] = hidden
            return output
        return hook

    handles.append(model.get_input_embeddings().register_forward_hook(make_hook(0)))
    for block_index, block in enumerate(layers):
        handles.append(block.register_forward_hook(make_hook(block_index + 1)))

    grid = np.zeros((n_layers, n_positions), dtype=np.float32)
    try:
        model.zero_grad(set_to_none=True)
        out = model(**inputs)
        logits = out.logits[0, p_star]

        clean_ids = readout_table.get(clean_target) or []
        donor_ids = readout_table.get(donor_target) or []
        if not clean_ids or not donor_ids:
            raise ValueError("readout_table lacks ids for one of the targets")

        metric = logits[clean_ids].max() - logits[donor_ids].max()
        metric.backward()

        for layer_index, hidden in captured.items():
            if hidden.grad is None or layer_index >= n_layers:
                continue
            delta = clean_cache[layer_index].to(hidden.device) - hidden.detach()
            contribution = (hidden.grad[0] * delta[0]).sum(-1)
            grid[layer_index, : contribution.shape[0]] = (
                contribution.detach().float().cpu().numpy()
            )
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)

    return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
