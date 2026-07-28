"""Paired clean/corrupted forwards (causal_spec.md §5A).

Two schemes, both reported.

* **Symmetric counterfactual (primary).** Keep the board, swap the hint for a
  donor whose own true target is a different word on that board. The corrupted
  run is then a valid clean run for a DIFFERENT answer, so the §3.2 denominator
  is well defined at both ends. This is the scheme Zhang & Nanda recommend.
* **Noised embeddings (secondary).** Gaussian noise on the hint token
  embeddings at sigma = 3x the empirical embedding component standard
  deviation (the ROME setting). Retained for comparability; this is the
  off-distribution scheme Zhang & Nanda warn about and it is never the sole
  basis of a claim (§3.4).

Both the clean and the corrupted per-layer states are cached, because the
patching stage needs the clean cache as its source and the corrupted run as
its destination.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..data import GIVER_COLS, extract_giver_features
from ..prompts import build_prompt

SCHEMES = ("counterfactual", "noise")


def _hint_token_span(tokenizer, prompt: str, hint: str) -> slice:
    """Character-located hint span mapped to token indices."""
    start_char = prompt.find(hint)
    if start_char < 0:
        return slice(0, 0)
    end_char = start_char + len(hint)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    running, first, last = 0, None, None
    for pos, token_id in enumerate(ids):
        piece = tokenizer.decode([token_id])
        nxt = running + len(piece)
        if first is None and running <= start_char < nxt:
            first = pos
        if running < end_char <= nxt:
            last = pos + 1
            break
        running = nxt
    if first is None:
        return slice(0, 0)
    return slice(first, last if last is not None else first + 1)


def _states_at_last_position(out, num_layers: int) -> np.ndarray:
    return np.stack([
        out.hidden_states[layer][0, -1].detach().float().cpu().numpy()
        for layer in range(num_layers + 1)
    ]).astype(np.float16)


def run_corrupted_extraction(
    *,
    model,
    tokenizer,
    df_sample: pd.DataFrame,
    pair_table: pd.DataFrame,
    base_dir: str,
    prefix: str,
    mode: str,
    chat_template_strategy: str,
    num_layers: int,
    hidden_dim: int,
    device: Optional[str] = None,
    scheme: str = "counterfactual",
    noise_sigma: float = 3.0,
    seed: int = 2026,
    require_alignment: bool = True,
) -> Dict[str, str]:
    """Cache clean and corrupted per-layer states for every pair.

    ``require_alignment`` drops pairs whose clean and corrupted prompts differ
    in token count, recording the count. This check is authoritative and the
    hint-token match in ``pairs.build_pair_table`` is only a cheap pre-filter:
    equal STANDALONE hint token counts do not guarantee equal PROMPT token
    counts, because a hint tokenises differently in context. Patching
    ``(layer, position)`` across sequences of different length silently reads
    the wrong position, so misaligned pairs must never reach the patch stage.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(base_dir, exist_ok=True)

    mode_flag = mode == "with_social"
    n = len(pair_table)
    shape = (n, num_layers + 1, hidden_dim)
    corrupt_path = os.path.join(base_dir, f"{prefix}_corrupt_{scheme}_{mode}_f16.npy")
    clean_path = os.path.join(base_dir, f"{prefix}_clean_{scheme}_{mode}_f16.npy")
    corrupt_mm = np.lib.format.open_memmap(
        corrupt_path, mode="w+", dtype=np.float16, shape=shape)
    clean_mm = np.lib.format.open_memmap(
        clean_path, mode="w+", dtype=np.float16, shape=shape)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    by_id = df_sample.set_index("row_id")
    rows = []
    dropped = []
    kept = 0

    for i, pair in enumerate(pair_table.itertuples()):
        clean_row = by_id.loc[pair.row_id]
        giver = extract_giver_features(clean_row, GIVER_COLS) if mode_flag else {}
        candidates = list(clean_row["candidates"])

        clean_prompt, _ = build_prompt(
            hint=str(pair.hint), candidates=candidates, giver_features=giver,
            use_social_context=mode_flag, tokenizer=tokenizer,
            chat_template_strategy=chat_template_strategy,
        )
        corrupt_hint = str(pair.donor_hint) if scheme == "counterfactual" else str(pair.hint)
        corrupt_prompt, _ = build_prompt(
            hint=corrupt_hint, candidates=candidates, giver_features=giver,
            use_social_context=mode_flag, tokenizer=tokenizer,
            chat_template_strategy=chat_template_strategy,
        )

        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(device)
        corrupt_inputs = tokenizer(corrupt_prompt, return_tensors="pt").to(device)

        n_clean = int(clean_inputs["input_ids"].shape[1])
        n_corrupt = int(corrupt_inputs["input_ids"].shape[1])
        if require_alignment and n_clean != n_corrupt:
            dropped.append({
                "row_id": int(pair.row_id), "donor_row_id": int(pair.donor_row_id),
                "clean_n_tokens": n_clean, "corrupt_n_tokens": n_corrupt,
                "reason": "prompt_length_mismatch",
            })
            continue

        with torch.no_grad():
            clean_out = model(
                input_ids=clean_inputs["input_ids"],
                attention_mask=clean_inputs["attention_mask"],
                output_hidden_states=True, return_dict=True,
            )
            if scheme == "noise":
                span = _hint_token_span(tokenizer, corrupt_prompt, corrupt_hint)
                embeddings = model.get_input_embeddings()(corrupt_inputs["input_ids"])
                std = float(embeddings.detach().float().std())
                block = embeddings[:, span]
                noise = torch.randn(
                    block.shape, generator=generator, dtype=torch.float32
                ).to(block.device) * (noise_sigma * std)
                embeddings = embeddings.clone()
                embeddings[:, span] = block + noise.to(block.dtype)
                corrupt_out = model(
                    inputs_embeds=embeddings,
                    attention_mask=corrupt_inputs["attention_mask"],
                    output_hidden_states=True, return_dict=True,
                )
            else:
                corrupt_out = model(
                    input_ids=corrupt_inputs["input_ids"],
                    attention_mask=corrupt_inputs["attention_mask"],
                    output_hidden_states=True, return_dict=True,
                )

        clean_mm[kept] = _states_at_last_position(clean_out, num_layers)
        corrupt_mm[kept] = _states_at_last_position(corrupt_out, num_layers)
        rows.append({
            "pair_idx": kept,
            "row_id": int(pair.row_id),
            "donor_row_id": int(pair.donor_row_id),
            "clean_hint": str(pair.hint),
            "corrupt_hint": corrupt_hint,
            "clean_target": pair.clean_target,
            "donor_target": pair.donor_target,
            "clean_n_tokens": n_clean,
            "corrupt_n_tokens": n_corrupt,
            "scheme": scheme,
        })
        kept += 1
        del clean_out, corrupt_out

    clean_mm.flush()
    corrupt_mm.flush()
    del clean_mm, corrupt_mm

    if kept != n:  # shrink to the rows actually written
        for path in (clean_path, corrupt_path):
            full = np.load(path, mmap_mode="r")
            trimmed = np.array(full[:kept])
            del full
            np.save(path, trimmed)

    # Explicit columns so an all-dropped run still writes a parseable header.
    index_cols = [
        "pair_idx", "row_id", "donor_row_id", "clean_hint", "corrupt_hint",
        "clean_target", "donor_target", "clean_n_tokens", "corrupt_n_tokens", "scheme",
    ]
    dropped_cols = [
        "row_id", "donor_row_id", "clean_n_tokens", "corrupt_n_tokens", "reason",
    ]
    index_path = os.path.join(base_dir, f"{prefix}_corrupt_{scheme}_{mode}_index.csv")
    pd.DataFrame(rows, columns=index_cols).to_csv(index_path, index=False)
    dropped_path = os.path.join(base_dir, f"{prefix}_corrupt_{scheme}_{mode}_dropped.csv")
    pd.DataFrame(dropped, columns=dropped_cols).to_csv(dropped_path, index=False)
    if dropped:
        print(f"  {len(dropped)}/{n} pairs dropped for prompt-length mismatch")
    return {
        "hidden": corrupt_path, "clean_hidden": clean_path,
        "index": index_path, "dropped": dropped_path,
    }
