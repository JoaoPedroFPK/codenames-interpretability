"""Position-level GPU extraction (docs/specs/lens_spec.md §5).

One forward pass per board (canonical candidate ordering only — the lens
study needs no shuffles and no generation). For each board, the per-layer
hidden states at the GENERATING POSITION (final prompt token) are written
into a preallocated fp16 memmap; rows follow this function's own seeded
draw over the ``df`` it receives, so joins against other artifacts must go
through ``row_id``, never positional order (the CLI hands in an already
sampled frame, which permutes absolute order relative to the thesis
outputs). Resumable via the same manifest machinery as the main loop
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

CANDIDATE SPANS (lens_spec.md §5 / causal_spec.md §12.4). With
``candidate_span_row_ids`` the per-layer states over each candidate's token
span are additionally written, restricted to the causal subsample (full
corpus would be ≈169 GB across the two decoders). The spans live in the
prompt, so they are sliced from the SAME forward pass as the generating
position — no extra GPU cost. Storage is an incrementally-written,
``np.load``-compatible npz keyed ``r{row_id}__{candidate}``, each entry
``(num_layers+1, span_len, hidden)`` fp16, with a companion index CSV.
Entries carry a fixed zip timestamp so a resumed archive stays
byte-identical to an uninterrupted one.
"""

import os
import zipfile
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from .. import checkpoint
from ..causal.positions import answer_position, is_word_first
from ..contract import Contract
from ..data import GIVER_COLS, extract_giver_features
from ..prompts import build_prompt
from ..spans import find_token_spans
from .raw import dump_readout_weights

_FLUSH_EVERY = 200  # boards between manifest commits (mirrors shard_boards)

_SPAN_INDEX_COLS = ["board_idx", "row_id", "candidate",
                    "token_start", "token_end", "ok"]


class _SpanZipBuffer:
    """Incremental npz writer with commit-granularity crash safety.

    Entries buffer in memory and are appended to the archive only at
    ``flush()`` (called from the manifest commit), so the file on disk is
    always a valid zip that reflects exactly the committed boards; a crash
    loses only the uncommitted window, which the resume path recomputes.
    A fixed entry timestamp keeps the archive deterministic.
    """

    def __init__(self, path: str, resume: bool):
        self.path = path
        self._pending: Dict[str, np.ndarray] = {}
        self._names: Set[str] = set()
        if not resume and os.path.exists(path):
            os.remove(path)
        if resume and os.path.exists(path):
            try:
                with zipfile.ZipFile(path) as zf:
                    self._names = set(zf.namelist())
            except zipfile.BadZipFile:
                raise ValueError(
                    f"{path} is corrupt (run interrupted before a commit); "
                    "delete it and re-run this condition without --resume "
                    "so every board's spans are recomputed.")

    def add(self, name: str, arr: np.ndarray) -> None:
        entry = name + ".npy"
        if entry in self._names or entry in self._pending:
            return  # resume recomputes a window; states are deterministic
        self._pending[entry] = np.ascontiguousarray(arr)

    def flush(self) -> None:
        if not self._pending:
            return
        with zipfile.ZipFile(self.path, mode="a",
                             compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            for entry, arr in self._pending.items():
                info = zipfile.ZipInfo(entry, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                with zf.open(info, "w", force_zip64=True) as f:
                    np.lib.format.write_array(f, arr, allow_pickle=False)
        self._names |= set(self._pending)
        self._pending.clear()


def _dump_candidate_spans(
    *, tokenizer, prompt, candidates, hidden_states, num_layers,
    row_id, board_idx, writer,
) -> List[Dict[str, object]]:
    """Slice each candidate's span states out of the prompt forward pass.

    A candidate whose span cannot be located gets an ``ok=False`` index row
    and no archive entry, so a missing span is never mistaken for data.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "candidate-span dumping needs a fast tokenizer "
            "(offset mappings are unavailable on Python tokenizers)")
    enc = tokenizer(prompt, return_offsets_mapping=True)
    spans = find_token_spans(
        prompt, enc["offset_mapping"],
        {f"cand:{word}": word for word in candidates})
    rows: List[Dict[str, object]] = []
    for word in candidates:
        span = spans.get(f"cand:{word}")
        rec = {"board_idx": board_idx, "row_id": row_id, "candidate": word,
               "token_start": -1, "token_end": -1, "ok": False}
        if span is not None:
            s, e = span
            arr = np.stack([
                hidden_states[layer][0, s:e].detach().float().cpu().numpy()
                for layer in range(num_layers + 1)
            ]).astype(np.float16)
            writer.add(f"r{row_id}__{word}", arr)
            rec.update(token_start=s, token_end=e, ok=True)
        rows.append(rec)
    return rows


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
    """Extract per-layer states at the generating position, optionally at p*
    and over the candidate spans (causal subsample only).

    ``candidate_span_row_ids`` restricts the candidate-span dump to the causal
    subsample (causal_spec.md §12.4: full corpus would be ~169 GB across both
    decoders). ``dump_answer_position`` requires a single condition per call,
    because a generation CSV records one condition's generations.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dump_answer_position and generation_csv is None:
        raise ValueError("dump_answer_position=True requires generation_csv")
    if dump_answer_position and len(conditions) > 1:
        raise ValueError(
            "dump_answer_position requires a single condition per call: a "
            "generation CSV records one condition's generations, and teacher-"
            "forcing another condition's prompts with it would silently dump "
            "wrong states. Run once per condition with its own CSV.")
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

        answer_path = answer_index_path = None
        if dump_answer_position:
            answer_path = os.path.join(
                base_dir, f"{prefix}_lens_answer_{mode_name}_f16.npy")
            answer_index_path = os.path.join(
                base_dir, f"{prefix}_lens_answer_index_{mode_name}.csv")
            results[mode_name]["answer_hidden"] = answer_path
            results[mode_name]["answer_index"] = answer_index_path

        candspan_path = candspan_index_path = None
        if candidate_span_row_ids is not None:
            candspan_path = os.path.join(
                base_dir, f"{prefix}_lens_candspan_{mode_name}_f16.npz")
            candspan_index_path = os.path.join(
                base_dir, f"{prefix}_lens_candspan_index_{mode_name}.csv")
            results[mode_name]["candspan"] = candspan_path
            results[mode_name]["candspan_index"] = candspan_index_path

        boards_done = 0
        index_rows = []
        answer_rows = []
        span_rows = []
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
                if answer_index_path and os.path.exists(answer_index_path):
                    answer_rows = pd.read_csv(answer_index_path) \
                        .to_dict("records")[:boards_done]
                if candspan_index_path and os.path.exists(candspan_index_path):
                    span_rows = [
                        r for r in pd.read_csv(candspan_index_path)
                                     .to_dict("records")
                        if int(r["board_idx"]) < boards_done]
        else:
            checkpoint.remove_manifest(ckpt_dir, lens_prefix, mode_name)

        resuming = resume and os.path.exists(hidden_path) and boards_done > 0
        if resuming:
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
            answer_rows = []
            span_rows = []
            mm = np.lib.format.open_memmap(
                hidden_path, mode="w+", dtype=np.float16,
                shape=(n_boards, num_layers + 1, hidden_dim))

        answer_mm = None
        if dump_answer_position:
            if resuming:
                if not os.path.exists(answer_path):
                    raise ValueError(
                        f"Resuming at board {boards_done} but {answer_path} "
                        "does not exist: the p* dump cannot be retrofitted "
                        "onto a run started without it. Re-run without "
                        "--resume.")
                answer_mm = np.lib.format.open_memmap(answer_path, mode="r+")
                if answer_mm.shape != (n_boards, num_layers + 1, hidden_dim):
                    raise ValueError(
                        f"Existing p* dump {answer_path} has shape "
                        f"{answer_mm.shape}, expected "
                        f"{(n_boards, num_layers + 1, hidden_dim)}; refusing "
                        "to resume into a mismatched file.")
            else:
                answer_mm = np.lib.format.open_memmap(
                    answer_path, mode="w+", dtype=np.float16,
                    shape=(n_boards, num_layers + 1, hidden_dim))
                # Missing p* is NaN, never a silent zero vector.
                answer_mm[:] = np.nan

        span_writer = None
        if candidate_span_row_ids is not None:
            if resuming and not os.path.exists(candspan_path):
                raise ValueError(
                    f"Resuming at board {boards_done} but {candspan_path} "
                    "does not exist: the candidate-span dump cannot be "
                    "retrofitted onto a run started without it. Re-run "
                    "without --resume.")
            span_writer = _SpanZipBuffer(candspan_path, resume=resuming)

        def _commit(done):
            pd.DataFrame(index_rows).to_csv(index_path, index=False)
            mm.flush()
            if answer_mm is not None:
                pd.DataFrame(answer_rows).to_csv(answer_index_path, index=False)
                answer_mm.flush()
            if span_writer is not None:
                span_writer.flush()
                pd.DataFrame(span_rows, columns=_SPAN_INDEX_COLS) \
                    .to_csv(candspan_index_path, index=False)
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
                if (span_writer is not None
                        and row_id in candidate_span_row_ids):
                    span_rows.extend(_dump_candidate_spans(
                        tokenizer=tokenizer, prompt=prompt,
                        candidates=list(row["candidates"]),
                        hidden_states=out.hidden_states,
                        num_layers=num_layers, row_id=row_id,
                        board_idx=board_idx, writer=span_writer))
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
