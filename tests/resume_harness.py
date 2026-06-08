"""Deterministic, GPU-free harness for exercising ``loop.run_extraction``.

Step 1 of the checkpoint/resume work changes *only* the flush/merge/resume
orchestration in :mod:`codenames.loop`. It does not touch
:func:`codenames.extraction.run_instance` (the model forward pass). So the
correct isolation boundary for verifying that orchestration is to stub
``run_instance`` / ``run_instance_batched`` with deterministic fakes that
return records of the same shape, as a pure function of
``(row_id, permutation_id, use_social_context, layer, word)``.

This lets the entire two-condition loop run on CPU with no weights, fully
reproducibly, so we can assert:

* **P1** — a killed-and-resumed run is byte-identical to an uninterrupted run.
* **P2** — the refactored loop's final outputs match the pre-refactor loop's.

``build_prompt`` is only reached on the generation path, and with the
``"mistral_inst"`` strategy it never touches the tokenizer, so the harness
passes ``tokenizer=None`` and a stub ``generation_fn``.
"""

import dataclasses
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from codenames.contract import ACCEL_REFERENCE, Contract
from codenames.data import GIVER_COLS

# Small, fixed geometry — big enough to exercise multiple words/layers, small
# enough to stay fast. NOT the real model dims; the stubs don't run a model.
FAKE_NUM_LAYERS = 3          # hidden states = FAKE_NUM_LAYERS + 1
FAKE_HIDDEN_DIM = 4
FAKE_POOLING = ("mean", "max_norm")


def make_fake_dataset(n: int) -> pd.DataFrame:
    """Build a dataset frame shaped like :func:`codenames.data.load_dataset`.

    Each board has a deterministic hint, 2 targets / 1 black / 1 tan, an
    alphabetical ``candidates`` list, and giver demographic columns (some
    null, to exercise ``extract_giver_features``). ``row_id`` is the index.
    """
    rows = []
    for i in range(n):
        targets = [f"tgt{i}a", f"tgt{i}b"]
        black = [f"blk{i}"]
        tan = [f"tan{i}"]
        candidates = sorted(targets + black + tan)
        row = {
            "row_id": i,
            "output": f"hint{i}",
            "targets": targets,
            "black": black,
            "tan": tan,
            "candidates": candidates,
        }
        # Populate a deterministic subset of giver columns; leave some NaN.
        for j, col in enumerate(GIVER_COLS):
            row[col] = f"val{i}_{j}" if (i + j) % 3 != 0 else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df["row_id"] = df["row_id"].astype(int)
    return df


def _det(*parts: int) -> float:
    """Deterministic pseudo-value in [0, 1) from integer parts (no RNG)."""
    acc = 0
    for p in parts:
        acc = (acc * 1000003 + int(p)) % 2_147_483_647
    return (acc % 100000) / 100000.0


def _word_index(word: str, candidates: List[str]) -> int:
    return candidates.index(word) if word in candidates else -1


def _pm_code(pm: str) -> int:
    """Stable per-pooling-method integer.

    Must NOT use the builtin ``hash()``: string hashing is salted per process
    (PYTHONHASHSEED), which would make digests differ across processes and
    silently break the golden/resume byte-identity claims.
    """
    return sum(ord(c) for c in pm)


def fake_run_instance(
    *,
    row,
    candidates_order,
    permutation_id,
    save_vectors,
    pooling_methods,
    num_layers,
    hidden_dim,
    use_truncation,
    use_social_context,
    **_ignored,
) -> Tuple[Dict, List[Dict], Optional[List[Dict]]]:
    """Deterministic stand-in for :func:`extraction.run_instance`.

    Mirrors the real return contract: ``(general_record, metrics_records,
    vector_records_or_None)`` with the same key set the loop and persistence
    layer rely on. Values are deterministic functions of identifiers, so the
    same board+permutation always yields identical records.
    """
    rid = int(row["row_id"])
    hint = str(row["output"])
    candidates = list(candidates_order)
    targets = set(row["targets"])
    black = set(row["black"])
    tan = set(row["tan"])

    def wtype(w: str) -> str:
        if w in targets:
            return "target"
        if w in black:
            return "black"
        if w in tan:
            return "tan"
        return "unknown"

    metrics: List[Dict] = []
    vectors: Optional[List[Dict]] = [] if save_vectors else None

    for layer in range(num_layers + 1):
        aniso_mean = _det(rid, permutation_id, layer, 11)
        aniso_std = _det(rid, permutation_id, layer, 13)

        # hint row
        hint_rec = {
            "row_id": rid, "layer": layer, "word": hint, "word_type": "hint",
            "token_count": 1, "list_position": -1,
            "use_social_context": use_social_context,
            "permutation_id": permutation_id,
            "layer_mean_pairwise_cosine": aniso_mean,
            "layer_std_pairwise_cosine": aniso_std,
        }
        for pm in pooling_methods:
            hint_rec[f"cosine_to_hint_{pm}"] = float("nan")
            hint_rec[f"rank_{pm}"] = float("nan")
            hint_rec[f"reciprocal_rank_{pm}"] = float("nan")
        metrics.append(hint_rec)

        # candidate rows
        for w in candidates:
            widx = _word_index(w, candidates)
            rec = {
                "row_id": rid, "layer": layer, "word": w, "word_type": wtype(w),
                "token_count": 1, "list_position": widx,
                "use_social_context": use_social_context,
                "permutation_id": permutation_id,
                "layer_mean_pairwise_cosine": aniso_mean,
                "layer_std_pairwise_cosine": aniso_std,
            }
            for pm in pooling_methods:
                cos = _det(rid, permutation_id, layer, widx, _pm_code(pm))
                rank = float(widx + 1)
                rec[f"cosine_to_hint_{pm}"] = cos
                rec[f"rank_{pm}"] = rank
                rec[f"reciprocal_rank_{pm}"] = 1.0 / rank
            metrics.append(rec)

        if save_vectors:
            for w in [hint] + candidates:
                for pm in pooling_methods:
                    val = np.float16(_det(rid, layer, _word_index(w, candidates), _pm_code(pm)))
                    vectors.append({
                        "row_id": rid, "layer": layer, "word": w,
                        "word_type": ("hint" if w == hint else wtype(w)),
                        "token_count": 1, "pooling_method": pm,
                        "use_social_context": use_social_context,
                        "vector": np.full(hidden_dim, val, dtype=np.float16),
                    })

    # Behavioral prediction: pick the highest-cosine candidate at last layer.
    general: Dict = {
        "row_id": rid, "hint": hint,
        "n_targets": len(targets), "n_candidates": len(candidates),
        "n_missing_spans": 0, "missing_span_words": [],
        "prompt_token_count": 10 + rid % 5,
    }
    if use_truncation:
        general["truncated"] = False
    general["use_social_context"] = use_social_context
    general["permutation_id"] = permutation_id
    general["giver_features"] = {} if not use_social_context else {"giver.gender": "x"}
    for pm in pooling_methods:
        # deterministic "predicted word": candidate with max _det score
        scores = {w: _det(rid, permutation_id, num_layers, _word_index(w, candidates), _pm_code(pm))
                  for w in candidates}
        pw = max(scores, key=scores.get)
        general[f"predicted_word_{pm}"] = pw
        general[f"correct_{pm}"] = pw in targets
        general[f"mean_target_rank_{pm}"] = _det(rid, permutation_id, 5, _pm_code(pm))
        general[f"raw_margin_{pm}"] = _det(rid, permutation_id, 7, _pm_code(pm))

    return general, metrics, vectors


def fake_run_instance_batched(
    *,
    rows,
    candidates_orders,
    permutation_ids,
    save_vectors_flags,
    pooling_methods,
    num_layers,
    hidden_dim,
    use_truncation,
    use_social_context,
    **_ignored,
) -> List[Optional[Tuple[Dict, List[Dict], Optional[List[Dict]]]]]:
    """Batched stand-in: per-row delegation to :func:`fake_run_instance`.

    Returns a list aligned with ``rows`` (no exclusions in the harness).
    """
    out = []
    for row, cand, pid, sv in zip(rows, candidates_orders, permutation_ids, save_vectors_flags):
        out.append(fake_run_instance(
            row=row, candidates_order=cand, permutation_id=pid, save_vectors=sv,
            pooling_methods=pooling_methods, num_layers=num_layers,
            hidden_dim=hidden_dim, use_truncation=use_truncation,
            use_social_context=use_social_context,
        ))
    return out


def fake_generation_fn(*, prompt, candidates, max_new_tokens, model, tokenizer, device):
    """Deterministic stand-in for :func:`generation.generate_response`."""
    # Pick a stable "generated word" from the prompt-independent candidate list.
    gw = candidates[0] if candidates else None
    return {
        "generated_text": f"<gen:{gw}>",
        "generated_word": gw,
        "generated_in_candidates": gw in candidates if gw else False,
    }


def make_contract(sample_size: int, *, shard_boards: int = 3,
                  vector_subsample_size: int = 2, n_shuffles: int = 2) -> Contract:
    """A Contract tuned for fast tests that still cross shard boundaries."""
    return dataclasses.replace(
        Contract(),
        sample_size=sample_size,
        shard_boards=shard_boards,
        vector_subsample_size=vector_subsample_size,
        n_shuffles=n_shuffles,
        pooling_methods=FAKE_POOLING,
    )


def install_fakes(monkeypatch) -> None:
    """Patch the loop's view of run_instance / run_instance_batched."""
    import codenames.loop as loop
    monkeypatch.setattr(loop, "run_instance", _adapt_run_instance)
    monkeypatch.setattr(loop, "run_instance_batched", _adapt_run_instance_batched)


def _adapt_run_instance(**kw):
    """Adapter: the loop passes use_social_context indirectly via the row/flag.

    The real run_instance takes ``use_social_context`` as a keyword; the loop
    supplies it. We forward everything through.
    """
    return fake_run_instance(**kw)


def _adapt_run_instance_batched(**kw):
    return fake_run_instance_batched(**kw)


def run_harness(
    base_dir: str,
    *,
    sample_size: int,
    has_generation: bool,
    use_truncation: bool,
    batch_size: int = 1,
    resume: bool = False,
    reuse_canonical: bool = False,
    checkpoint_dir: str = None,
    monkeypatch=None,
) -> Dict:
    """Run ``run_extraction`` with the fakes installed, into ``base_dir``.

    Returns the results dict. Caller is responsible for installing fakes via
    ``install_fakes`` (kept separate so the same monkeypatch governs both the
    uninterrupted and the killed runs).
    """
    from codenames.loop import run_extraction

    contract = make_contract(sample_size)
    accel = dataclasses.replace(ACCEL_REFERENCE, batch_size=batch_size)

    chat_strategy = "mistral_inst" if has_generation else "raw"

    return run_extraction(
        model=None,
        tokenizer=None,
        df=make_fake_dataset(sample_size),
        base_dir=base_dir,
        prefix="fake",
        contract=contract,
        chat_template_strategy=chat_strategy,
        forward_hidden_states_mode="causal" if has_generation else "encoder_load_time",
        use_truncation=use_truncation,
        num_layers=FAKE_NUM_LAYERS,
        hidden_dim=FAKE_HIDDEN_DIM,
        device="cpu",
        has_generation=has_generation,
        generation_fn=fake_generation_fn if has_generation else None,
        acceleration=accel,
        resume=resume,
        reuse_canonical=reuse_canonical,
        checkpoint_dir=checkpoint_dir,
    )
