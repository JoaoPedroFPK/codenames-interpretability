"""Position-level GPU extraction (docs/specs/lens_spec.md §5).

One forward pass per board (canonical candidate ordering only — the lens
study needs no shuffles and no generation). For each board, the per-layer
hidden states at the GENERATING POSITION (final prompt token) are written
into a preallocated fp16 memmap; row order equals the seeded board sample
order used by loop.run_extraction, so lens rows align 1:1 with the thesis
outputs. Resumable via the same manifest machinery as the main loop
(prefix suffixed "_lens" so the two never collide).

ANSWER POSITION (added 2026-07-27, lens_spec.md §5.1 / causal_spec.md §4.1).
The generating position is behaviourally valid only when the model emits the
answer word first — measured: Mistral 13.0% of turns, Qwen 91.5%. With
``dump_answer_position=True`` a second memmap records the per-layer states at
``p*``, the token index where the parsed answer word begins in the
teacher-forced prompt+generation sequence. This is the PRIMARY readout
position for both specs; the generating position is retained as a secondary
channel. It costs one extra forward pass per board — the joint sequence must
be tokenised together, because BPE prompt tokenisation is not a prefix of the
joint tokenisation (a trailing prompt space merges with the first generated
word), so p* can never be derived by offsetting from the prompt's own token
count. The extra pass is ~15k forwards against a ~2.55M budget.
"""

import os
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .. import checkpoint
from ..causal.positions import answer_position, is_word_first
from ..contract import Contract
from ..data import GIVER_COLS, extract_giver_features
from ..prompts import build_prompt
from .raw import dump_readout_weights

_FLUSH_EVERY = 200  # boards between manifest commits (mirrors shard_boards)


def _dump_answer_position(
    *, model, tokenizer, prompt, generations, row_id, board_idx,
    num_layers, device, answer_mm,
) -> Dict[str, object]:
    """Teacher-force this turn's recorded generation and cache states at p*.

    Returns the index row. When the generation is unparseable or the word is
    not locatable the memmap row is left as NaN and ``p_star_missing`` is set,
    so downstream code can never mistake a missing state for a zero vector.
    """
    record = {
        "board_idx": board_idx, "row_id": row_id,
        "p_star": -1, "p_star_missing": True, "word_first": False,
    }
    if generations is None or row_id not in generations.index:
        return record

    gen_row = generations.loc[row_id]
    text = gen_row.get("generated_text")
    word = gen_row.get("generated_word")
    if not isinstance(text, str) or not isinstance(word, str):
        return record

    record["word_first"] = is_word_first(text, word)
    position = answer_position(tokenizer, prompt, text, word)
    if position is None:
        return record

    joint = tokenizer(prompt + text, return_tensors="pt").to(device)
    if position >= joint["input_ids"].shape[1]:
        return record

    with torch.no_grad():
        out = model(
            input_ids=joint["input_ids"],
            attention_mask=joint["attention_mask"],
            output_hidden_states=True, return_dict=True,
        )
    for layer in range(num_layers + 1):
        answer_mm[board_idx, layer] = (
            out.hidden_states[layer][0, position].detach()
            .float().cpu().numpy().astype(np.float16))
    del out

    record["p_star"] = int(position)
    record["p_star_missing"] = False
    return record


def run_lens_extraction(
    *,
    model,
    tokenizer,
    df: pd.DataFrame,
    base_dir: str,
    prefix: str,
    contract: Contract,
    chat_template_strategy: str,
    num_layers: int,
    hidden_dim: int,
    conditions: Tuple[str, ...] = ("no_social", "with_social"),
    device: Optional[str] = None,
    resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    dump_answer_position: bool = False,
    generation_csv: Optional[str] = None,
    candidate_span_row_ids: Optional[Set[int]] = None,
) -> Dict[str, Dict[str, str]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dump_answer_position and generation_csv is None:
        raise ValueError("dump_answer_position=True requires generation_csv")
    generations = None
    if dump_answer_position:
        generations = pd.read_csv(generation_csv).set_index("row_id")

    # Board sample: byte-identical to loop.run_extraction's draw.
    df_sample = df.sample(
        n=min(contract.sample_size, len(df)),
        random_state=contract.random_seed,
    ).copy().reset_index(drop=True)
    n_boards = len(df_sample)

    os.makedirs(base_dir, exist_ok=True)
    ckpt_dir = checkpoint_dir or os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    lens_prefix = f"{prefix}_lens"

    # Readout weights: written once per model (idempotent).
    readout_path = os.path.join(base_dir, f"{prefix}_lens_readout_f16.npz")
    if not os.path.exists(readout_path):
        dump_readout_weights(model, readout_path)
        print(f"  Readout weights saved: {readout_path}")

    results: Dict[str, Dict[str, str]] = {}
    for mode_name in conditions:
        mode_flag = mode_name == "with_social"
        hidden_path = os.path.join(
            base_dir, f"{prefix}_lens_hidden_{mode_name}_f16.npy")
        index_path = os.path.join(
            base_dir, f"{prefix}_lens_index_{mode_name}.csv")
        results[mode_name] = {"hidden": hidden_path, "index": index_path}

        answer_mm = None
        answer_rows = []
        if dump_answer_position:
            answer_path = os.path.join(
                base_dir, f"{prefix}_lens_answer_{mode_name}_f16.npy")
            answer_index_path = os.path.join(
                base_dir, f"{prefix}_lens_answer_index_{mode_name}.csv")
            results[mode_name]["answer_hidden"] = answer_path
            results[mode_name]["answer_index"] = answer_index_path
            answer_mm = np.lib.format.open_memmap(
                answer_path, mode="w+", dtype=np.float16,
                shape=(n_boards, num_layers + 1, hidden_dim))
            # Missing p* is NaN, never a silent zero vector.
            answer_mm[:] = np.nan

        boards_done = 0
        index_rows = []
        if resume:
            man = checkpoint.read_manifest(ckpt_dir, lens_prefix, mode_name)
            if man is not None:
                boards_done = int(man.get("boards_done", 0))
                if man.get("complete") and os.path.exists(hidden_path):
                    print(f"  [resume] lens condition '{mode_name}' already "
                          f"complete; skipping.")
                    continue
                if os.path.exists(index_path):
                    index_rows = pd.read_csv(index_path) \
                        .to_dict("records")[:boards_done]
        else:
            checkpoint.remove_manifest(ckpt_dir, lens_prefix, mode_name)

        if resume and os.path.exists(hidden_path) and boards_done > 0:
            mm = np.lib.format.open_memmap(hidden_path, mode="r+")
            if mm.shape != (n_boards, num_layers + 1, hidden_dim):
                raise ValueError(
                    f"Existing lens dump {hidden_path} has shape {mm.shape}, "
                    f"expected {(n_boards, num_layers + 1, hidden_dim)}; "
                    "refusing to resume into a mismatched file.")
            print(f"  [resume] lens '{mode_name}': continuing at board "
                  f"{boards_done}/{n_boards}.")
        else:
            boards_done = 0
            index_rows = []
            mm = np.lib.format.open_memmap(
                hidden_path, mode="w+", dtype=np.float16,
                shape=(n_boards, num_layers + 1, hidden_dim))

        def _commit(done):
            pd.DataFrame(index_rows).to_csv(index_path, index=False)
            mm.flush()
            if answer_mm is not None:
                pd.DataFrame(answer_rows).to_csv(answer_index_path, index=False)
                answer_mm.flush()
            checkpoint.write_manifest(
                ckpt_dir, lens_prefix, mode_name,
                n_boards=n_boards, boards_done=done,
                ckpt_committed=0, complete=(done == n_boards))

        for board_idx, (_, row) in enumerate(
            tqdm(df_sample.iterrows(), total=n_boards,
                 desc=f"lens {mode_name}")
        ):
            if board_idx < boards_done:
                continue
            row_id = int(row["row_id"])
            try:
                prompt, _ = build_prompt(
                    hint=str(row["output"]),
                    candidates=list(row["candidates"]),
                    giver_features=(extract_giver_features(row, GIVER_COLS)
                                    if mode_flag else {}),
                    use_social_context=mode_flag,
                    tokenizer=tokenizer,
                    chat_template_strategy=chat_template_strategy,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        output_hidden_states=True, return_dict=True,
                    )
                for layer in range(num_layers + 1):
                    mm[board_idx, layer] = (
                        out.hidden_states[layer][0, -1].detach()
                        .float().cpu().numpy().astype(np.float16))
                index_rows.append({
                    "board_idx": board_idx, "row_id": row_id,
                    "prompt_token_count": int(inputs["input_ids"].shape[1]),
                    "ok": True, "error": "",
                })
                del out

                if answer_mm is not None:
                    answer_rows.append(_dump_answer_position(
                        model=model, tokenizer=tokenizer, prompt=prompt,
                        generations=generations, row_id=row_id,
                        board_idx=board_idx, num_layers=num_layers,
                        device=device, answer_mm=answer_mm,
                    ))
            except Exception as e:  # zero row + logged error; never abort
                mm[board_idx] = 0
                index_rows.append({
                    "board_idx": board_idx, "row_id": row_id,
                    "prompt_token_count": 0, "ok": False, "error": str(e),
                })
                print(f"  ERROR lens row_id={row_id}: {e}")
            if device == "cuda" and (board_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if (board_idx + 1) % _FLUSH_EVERY == 0:
                _commit(board_idx + 1)

        _commit(n_boards)
        n_ok = sum(1 for r in index_rows if r["ok"])
        print(f"  lens '{mode_name}': {n_ok}/{n_boards} boards ok; "
              f"{os.path.getsize(hidden_path) / 1e9:.2f} GB")

    return results
