"""Pre-flight diagnostics for random-init models.

Verbatim from Cell 8 of
``reference_notebooks/Random_Qwen_Codenames_Layer_Extraction.ipynb``.

Three checks:

1. **NaN / Inf detection** — counts non-finite values in the hidden states
   across all layers. Any non-zero count is a hard stop: fp16 has likely
   overflowed in the residual stream. Fallback: re-instantiate the model
   with ``torch.bfloat16`` and re-run this cell.
2. **Hidden state norm growth** — reports the mean L2 norm of token hidden
   states at each layer. Random transformers can exhibit exponential norm
   growth across the residual stream. Threshold: if any layer's mean norm
   exceeds 100, flag as a soft warning.
3. **Anisotropy at layer 0** — random-init transformers have an elevated L0
   pairwise cosine because random embedding vectors share more directional
   bias than trained ones. Reported as informational; no hard threshold.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from .prompts import build_prompt


def preflight_random_init(
    *,
    df: pd.DataFrame,
    model,
    tokenizer,
    device: str,
    num_layers: int,
    chat_template_strategy: str,
    n_boards: int = 5,
) -> Dict[str, Any]:
    """Run the random-init pre-flight diagnostic on ``n_boards`` boards.

    Parameters
    ----------
    df
        Dataset with ``output``, ``candidates`` columns (the standard
        post-:func:`data.load_dataset` shape).
    model, tokenizer, device, num_layers
        From the loader's metadata.
    chat_template_strategy
        Forwarded to :func:`prompts.build_prompt`.
    n_boards
        Number of boards to run through. Default 5.

    Returns
    -------
    summary
        Dict with ``nan_count``, ``inf_count``, ``norms_per_layer``,
        ``norm_max``, ``aniso_l0_mean``, ``check1_pass``, ``check2_pass``,
        and ``all_hard_checks_pass``.
    """
    print("PRE-FLIGHT DIAGNOSTIC")
    print("=" * 60)
    print(f"Running forward pass on {n_boards} boards (no_social condition)...")

    norms_per_layer: list = [[] for _ in range(num_layers + 1)]
    nan_count = 0
    inf_count = 0
    aniso_l0 = []

    n_diag = min(n_boards, len(df))
    for i in range(n_diag):
        row = df.iloc[i]
        prompt, _ = build_prompt(
            hint=str(row["output"]),
            candidates=list(row["candidates"]),
            giver_features={},
            use_social_context=False,
            tokenizer=tokenizer,
            chat_template_strategy=chat_template_strategy,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
        for layer in range(num_layers + 1):
            hs = out.hidden_states[layer][0].detach().float()
            nan_count += int(torch.isnan(hs).sum().item())
            inf_count += int(torch.isinf(hs).sum().item())
            norms = torch.norm(hs, p=2, dim=1)
            norms_per_layer[layer].append(float(norms.mean().item()))

        # Anisotropy at L0: mean pairwise cosine across all token positions.
        hs0 = out.hidden_states[0][0].detach().float()
        if hs0.shape[0] >= 2:
            hs0n = hs0 / (hs0.norm(p=2, dim=1, keepdim=True) + 1e-8)
            sim = hs0n @ hs0n.T
            mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            aniso_l0.append(float(sim[mask].mean().item()))

        del out
        if device == "cuda":
            torch.cuda.empty_cache()

    # --- Check 1: NaN / Inf ---
    print("\nCheck 1: NaN / Inf in hidden states")
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    check1_pass = (nan_count == 0 and inf_count == 0)
    print(f"  Status: {'PASS' if check1_pass else 'FAIL — HARD STOP'}")

    # --- Check 2: Norm growth ---
    print(f"\nCheck 2: Mean hidden-state L2 norm per layer (across {n_diag} boards)")
    print(f"  {'Layer':>6}  {'Mean norm':>12}")
    print(f"  {'-'*22}")
    norm_max = 0.0
    for layer, vals in enumerate(norms_per_layer):
        m = float(np.mean(vals)) if vals else 0.0
        norm_max = max(norm_max, m)
        print(f"  {layer:>6}  {m:>12.4f}")
    print(f"  Max mean norm across layers: {norm_max:.4f}")
    check2_pass = (norm_max < 100.0)
    print(f"  Status: {'PASS' if check2_pass else 'WARN — norms unusually high'}")

    # --- Check 3: Anisotropy at L0 ---
    aniso_l0_mean = float(np.mean(aniso_l0)) if aniso_l0 else float("nan")
    print("\nCheck 3: Mean pairwise cosine (anisotropy) at layer 0")
    print(f"  Mean L0 anisotropy: {aniso_l0_mean:.4f}")
    print("  Reference: Random BERT L0 anisotropy = 0.357 (interpretable, non-degenerate)")
    print("  Status: report-only (no hard threshold)")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("PRE-FLIGHT SUMMARY")
    print(f"{'='*60}")
    if check1_pass and check2_pass:
        print("  ALL HARD CHECKS PASSED. Safe to run main extraction loop.")
    elif not check1_pass:
        print("  HARD STOP: NaN/Inf detected. Re-instantiate model with bfloat16:")
        print('    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)')
        print("  Then re-run this diagnostic.")
    else:
        print("  WARN: Norms exceed soft threshold (100). Inspect per-layer norm")
        print("  growth above. Main loop may still run; anisotropy should be")
        print("  interpreted carefully in the Results chapter.")

    return {
        "nan_count": nan_count,
        "inf_count": inf_count,
        "norms_per_layer": [float(np.mean(v)) if v else 0.0 for v in norms_per_layer],
        "norm_max": norm_max,
        "aniso_l0_mean": aniso_l0_mean,
        "check1_pass": check1_pass,
        "check2_pass": check2_pass,
        "all_hard_checks_pass": check1_pass and check2_pass,
    }
