"""Raw logit lens: Unembed(FinalLN(h)).

Supports the Llama-family causal decoders in the suite (Mistral, Qwen,
Random Qwen), whose final norm is an RMSNorm at ``model.base_model.norm``.
At the final layer the lens output is IDENTICAL to the model's logits;
the calibration check in analysis.py and test_raw.py rely on this.
"""

import numpy as np
import torch


class RawLens:
    def __init__(self, model):
        base = model.base_model
        if not hasattr(base, "norm"):
            raise AttributeError(
                "model.base_model has no final `norm`; the lens is scoped to "
                "Llama-family causal decoders (mistral, qwen, random_qwen)."
            )
        self.norm = base.norm
        self.lm_head = model.get_output_embeddings()

    @torch.no_grad()
    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h))


def dump_readout_weights(model, path: str) -> None:
    """Persist final-norm weight, lm_head weight, and eps for offline scoring.

    Written once per model at extraction time so the offline stages (apply,
    analysis) never load the 7B model — and so the random-init decoder's
    (unrepeatable-without-seed) readout is preserved exactly.
    """
    base = model.base_model
    norm_w = base.norm.weight.detach().float().cpu().numpy()
    eps = float(getattr(base.norm, "variance_epsilon", getattr(base.norm, "eps", 1e-6)))
    lm_w = model.get_output_embeddings().weight.detach().half().cpu().numpy()
    np.savez_compressed(
        path,
        norm_weight=norm_w,
        lm_head_weight=lm_w,
        eps=np.float64(eps),
        model_name=str(getattr(model.config, "_name_or_path", "unknown")),
    )
