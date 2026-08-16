"""Matched-position lens dumps and attention from p_read (lens_spec.md §5;
pre-submission tasks T3a / T3c).

The answer channel reads the residual at ``p_read = p*-1``; the geometry tier
reads the hint span and the candidate spans. To compare the two like for
like, this module dumps per-layer states at the SAME positions the geometry
tier pools over -- the hint span and the target-candidate span, **mean over
the span's tokens** (the geometry tier's ``mean`` pooling) -- so
``lens-apply --channel hint|cand_target`` scores them with the model's own
unembedding exactly as the answer channel is scored.

From the same forward pass it also records where ``p_read`` attends: the
mean over heads of the attention mass from ``p_read`` onto each role
(hint, target candidate, other candidates, prompt scaffold, and the
teacher-forced generation up to and including itself), per block. That is
the trough account of paper §6: does the answer position read the hint
directly, or the candidate positions?

One joint pass (prompt + teacher-forced generation) per board suffices. In a
causal decoder the states at prompt positions do not depend on the tokens
appended after them, so the hint/candidate states from the joint pass equal
those of a prompt-only pass; the pass is joint only so that ``p_read`` exists
for the attention readout. Boards without a resolvable ``p*`` (unparseable
generation, or no generation CSV) still get their span dumps and no
attention row.

Attention requires eager or SDPA attention (``output_attentions=True`` is
undefined under flash-attention), so this stage is run WITHOUT
``--flash-attn``. That is why it writes NEW files only and never rewrites
the generating-position / answer dumps of ``extract.py``: those were produced
under flash-attention and re-deriving them here would introduce kernel-level
drift into artefacts already cited.
"""

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .. import checkpoint
from ..causal.basis import ROLE_INDEX, role_of_each_token
from ..causal.positions import answer_position, readout_index
from ..contract import Contract
from ..data import GIVER_COLS, extract_giver_features
from ..prompts import build_prompt

POSITION_CHANNELS = ("hint", "cand_target")
ATTENTION_COLS = ("to_hint", "to_cand_target", "to_cand_other",
                  "to_scaffold", "to_generation")
_SCAFFOLD_ROLES = ("prefix", "post_hint", "list_scaffold", "question", "final")
_FLUSH_EVERY = 200

_INDEX_COLS = ["board_idx", "row_id", "hint_ok", "hint_n_tokens",
               "cand_target_ok", "cand_target_n_tokens", "target_word",
               "n_targets", "p_star", "p_read", "ok", "error"]
_ATT_COLS = ["board_idx", "row_id", "layer", *ATTENTION_COLS]


def _paths(base_dir: str, prefix: str, mode_name: str) -> Dict[str, str]:
    out = {ch: os.path.join(base_dir, f"{prefix}_lens_pos_{ch}_{mode_name}_f16.npy")
           for ch in POSITION_CHANNELS}
    out["index"] = os.path.join(base_dir, f"{prefix}_lens_pos_index_{mode_name}.csv")
    out["attention"] = os.path.join(
        base_dir, f"{prefix}_lens_attention_pread_{mode_name}.csv")
    return out


def _attention_row(attn: torch.Tensor, roles: np.ndarray, p_read: int) -> Dict[str, float]:
    """Mean-over-heads attention mass from ``p_read`` onto each role group.

    ``attn`` is one block's ``(heads, q, k)`` tensor for the single sequence.
    Keys after ``p_read`` are masked in a causal model, so the groups
    partition ``[0, p_read]`` and their masses sum to one.
    """
    mass = attn[:, p_read, :p_read + 1].float().mean(dim=0).cpu().numpy()
    r = roles[:p_read + 1]

    def total(names: Sequence[str]) -> float:
        idx = [ROLE_INDEX[n] for n in names]
        return float(mass[np.isin(r, idx)].sum())

    return {
        "to_hint": total(["hint"]),
        "to_cand_target": total(["cand_target"]),
        "to_cand_other": total(["cand_other", "cand_donor"]),
        "to_scaffold": total(_SCAFFOLD_ROLES),
        "to_generation": total(["generation"]),
    }


def run_position_extraction(
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
    mode_name: str = "no_social",
    device: Optional[str] = None,
    generation_csv: Optional[str] = None,
    channels: Sequence[str] = POSITION_CHANNELS,
    dump_attention: bool = True,
    resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    stop_after: Optional[int] = None,
) -> Dict[str, str]:
    """Dump mean-pooled per-layer states at the hint span and the target
    candidate span, plus attention from ``p_read`` per role.

    Board order is the same seeded draw as ``extract.run_lens_extraction``,
    so ``board_idx`` lines up with the generating/answer dumps and every join
    still goes through ``row_id``. Resumable; byte-identical to an
    uninterrupted run. ``stop_after`` exists to exercise the resume path.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    channels = tuple(channels)
    unknown = set(channels) - set(POSITION_CHANNELS)
    if unknown:
        raise ValueError(f"unknown position channels {sorted(unknown)}; "
                         f"choose from {POSITION_CHANNELS}")
    generations = None
    if generation_csv is not None:
        generations = pd.read_csv(generation_csv).set_index("row_id")

    df_sample = df.sample(n=min(contract.sample_size, len(df)),
                          random_state=contract.random_seed).copy().reset_index(drop=True)
    n_boards = len(df_sample)
    mode_flag = mode_name == "with_social"

    os.makedirs(base_dir, exist_ok=True)
    ckpt_dir = checkpoint_dir or os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix = f"{prefix}_lenspos"
    paths = _paths(base_dir, prefix, mode_name)
    shape = (n_boards, num_layers + 1, hidden_dim)

    boards_done = 0
    index_rows: List[Dict[str, object]] = []
    att_rows: List[Dict[str, object]] = []
    if resume:
        man = checkpoint.read_manifest(ckpt_dir, ckpt_prefix, mode_name)
        if man is not None:
            boards_done = int(man.get("boards_done", 0))
            if os.path.exists(paths["index"]):
                index_rows = pd.read_csv(paths["index"]).to_dict("records")[:boards_done]
            if os.path.exists(paths["attention"]):
                att_rows = [r for r in pd.read_csv(paths["attention"]).to_dict("records")
                            if int(r["board_idx"]) < boards_done]
    else:
        checkpoint.remove_manifest(ckpt_dir, ckpt_prefix, mode_name)

    resuming = resume and boards_done > 0 and all(os.path.exists(paths[c]) for c in channels)
    mms: Dict[str, np.memmap] = {}
    for ch in channels:
        if resuming:
            mm = np.lib.format.open_memmap(paths[ch], mode="r+")
            if mm.shape != shape:
                raise ValueError(f"existing dump {paths[ch]} has shape {mm.shape}, "
                                 f"expected {shape}; refusing to resume into it")
        else:
            mm = np.lib.format.open_memmap(paths[ch], mode="w+", dtype=np.float16, shape=shape)
            mm[:] = np.nan          # a missing span is NaN, never a silent zero
        mms[ch] = mm
    if not resuming:
        boards_done, index_rows, att_rows = 0, [], []

    def _commit(done: int) -> None:
        pd.DataFrame(index_rows, columns=_INDEX_COLS).to_csv(paths["index"], index=False)
        pd.DataFrame(att_rows, columns=_ATT_COLS).to_csv(paths["attention"], index=False)
        for mm in mms.values():
            mm.flush()
        checkpoint.write_manifest(ckpt_dir, ckpt_prefix, mode_name, n_boards=n_boards,
                                  boards_done=done, ckpt_committed=0,
                                  complete=(done == n_boards))

    last = boards_done
    for board_idx, (_, row) in enumerate(
            tqdm(df_sample.iterrows(), total=n_boards, desc=f"lens positions {mode_name}")):
        if board_idx < boards_done:
            continue
        if stop_after is not None and board_idx >= boards_done + stop_after:
            break
        row_id = int(row["row_id"])
        targets = list(row["targets"])
        target = str(targets[0]) if targets else ""
        rec: Dict[str, object] = {
            "board_idx": board_idx, "row_id": row_id,
            "hint_ok": False, "hint_n_tokens": 0,
            "cand_target_ok": False, "cand_target_n_tokens": 0,
            "target_word": target, "n_targets": len(targets),
            "p_star": -1, "p_read": -1, "ok": True, "error": "",
        }
        try:
            prompt, _ = build_prompt(
                hint=str(row["output"]), candidates=list(row["candidates"]),
                giver_features=(extract_giver_features(row, GIVER_COLS) if mode_flag else {}),
                use_social_context=mode_flag, tokenizer=tokenizer,
                chat_template_strategy=chat_template_strategy)
            text = ""
            p_star = None
            if generations is not None and row_id in generations.index:
                g = generations.loc[row_id]
                if isinstance(g, pd.DataFrame):
                    g = g.iloc[0]
                gt, gw = g.get("generated_text"), g.get("generated_word")
                if isinstance(gt, str) and isinstance(gw, str):
                    p_star = answer_position(tokenizer, prompt, gt, gw)
                    if p_star is not None and p_star > 0:
                        text = gt
                    else:
                        p_star = None
            joint = tokenizer(prompt + text, return_tensors="pt").to(device)
            n_pos = int(joint["input_ids"].shape[1])
            want_attention = dump_attention and p_star is not None and p_star < n_pos
            with torch.no_grad():
                out = model(input_ids=joint["input_ids"],
                            attention_mask=joint["attention_mask"],
                            output_hidden_states=True,
                            output_attentions=want_attention, return_dict=True)
            roles = role_of_each_token(
                tokenizer, prompt, hint=str(row["output"]),
                candidates=list(row["candidates"]), clean_target=target,
                donor_target="", n_positions=n_pos)
            for ch in channels:
                idx = np.flatnonzero(roles == ROLE_INDEX[ch])
                rec[f"{ch}_n_tokens"] = int(idx.size)
                if idx.size == 0:
                    continue
                for layer in range(num_layers + 1):
                    mms[ch][board_idx, layer] = (
                        out.hidden_states[layer][0, idx].float().mean(dim=0)
                        .cpu().numpy().astype(np.float16))
                rec[f"{ch}_ok"] = True
            if want_attention:
                p_read = readout_index(int(p_star))
                rec["p_star"], rec["p_read"] = int(p_star), int(p_read)
                atts = out.attentions
                if atts is None or atts[0] is None:
                    raise RuntimeError(
                        "output_attentions returned nothing; load the model "
                        "with eager/sdpa attention (not flash-attention)")
                for layer, attn in enumerate(atts):
                    att_rows.append({"board_idx": board_idx, "row_id": row_id,
                                     "layer": layer + 1,
                                     **_attention_row(attn[0], roles, p_read)})
            del out
        except Exception as e:  # noqa: BLE001  logged, never aborts the run
            rec["ok"] = False
            rec["error"] = str(e)
            print(f"  ERROR lens positions row_id={row_id}: {e}")
        index_rows.append(rec)
        last = board_idx + 1
        if device == "cuda" and last % 50 == 0:
            torch.cuda.empty_cache()
        if last % _FLUSH_EVERY == 0:
            _commit(last)
    _commit(last)
    ok = sum(1 for r in index_rows if r["ok"])
    print(f"  lens positions '{mode_name}': {ok}/{last} boards ok; "
          f"{sum(1 for r in index_rows if r['p_read'] >= 0)} with attention rows")
    return paths
