"""Per-board extraction.

``run_instance`` is the hottest piece of code in the package. The function
body is taken verbatim from Cell 8 of the reference notebooks. The only
permitted modifications are:

1. Module-globals (``model``, ``tokenizer``, ``DEVICE``, ``POOLING_METHODS``,
   ``NUM_LAYERS``, ``HIDDEN_DIM``) become keyword-only arguments.
2. ``build_prompt`` takes ``chat_template_strategy``.
3. A 3-way dispatch on ``forward_hidden_states_mode`` covers the three
   notebook variants: ``"causal"`` (Mistral, Qwen, Random_Qwen — explicit
   ``input_ids`` / ``attention_mask`` arguments, ``output_hidden_states=True``
   passed at inference), ``"encoder_load_time"`` (BERT, T5, BERT-Random —
   ``output_hidden_states`` on the config, plain ``**inputs_for_model``),
   ``"encoder_inference"`` (ModernBERT — ``**inputs_for_model`` with
   ``output_hidden_states=True`` and ``return_dict=True`` at inference).
4. A ``use_truncation`` flag toggles the tokenizer's ``max_length`` /
   ``truncation`` arguments and the ``"truncated"`` field in the general
   record. Encoder models (BERT, T5, BERT-Random, ModernBERT) use truncation;
   the three causal models do not.

No other changes. No "cleanup". No "more Pythonic". The hot loop matches
the canonical notebooks byte-for-byte.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .contract import ACCEL_REFERENCE, Acceleration
from .data import extract_giver_features
from .prompts import build_prompt
from .spans import cosine_similarity_np, find_token_spans, pool_span


def run_instance(
    row: pd.Series,
    giver_cols: List[str],
    use_social_context: bool,
    candidates_order: List[str],
    permutation_id: int = 0,
    save_vectors: bool = False,
    *,
    model,
    tokenizer,
    device: str,
    pooling_methods: Tuple[str, ...],
    num_layers: int,
    hidden_dim: int,
    chat_template_strategy: str,
    forward_hidden_states_mode: str,
    use_truncation: bool,
    max_seq_len: int = 512,
    acceleration: Acceleration = ACCEL_REFERENCE,
) -> Tuple[Dict, List[Dict], Optional[List[Dict]]]:
    """Process a single board under a given candidate ordering.

    Parameters
    ----------
    row
        One row from the sampled dataset.
    giver_cols
        Column names for giver demographic features.
    use_social_context
        If True, include giver features in the prompt.
    candidates_order
        The candidate words in the desired order. For the canonical run this
        is alphabetical; for shuffles it is a random permutation.
    permutation_id
        0 = canonical (alphabetical) ordering. 1..K = shuffles.
    save_vectors
        If True, raw vectors are retained for the subsample.
    model, tokenizer, device, pooling_methods, num_layers, hidden_dim
        Module-globals in the original notebooks; passed in here.
    chat_template_strategy
        One of ``"mistral_inst"``, ``"chatml"``, ``"raw"``. Forwarded to
        :func:`prompts.build_prompt`.
    forward_hidden_states_mode
        One of ``"causal"``, ``"encoder_load_time"``, ``"encoder_inference"``.
        Selects how the forward pass is invoked and where
        ``output_hidden_states=True`` is set.
    use_truncation
        If True, tokenize with ``max_length=max_seq_len, truncation=True`` and
        emit a ``"truncated"`` flag in the general record.
    max_seq_len
        Truncation length when ``use_truncation`` is True. Default 512.
    acceleration
        Implementation-detail flags (vectorized anisotropy, FA2, batch_size).
        Defaults to ``ACCEL_REFERENCE`` (all flags off = original code path).
        Individual flags are honoured at their respective code sites later
        in the function and in :func:`loop.run_extraction`.
    """
    # Mark the parameter as consumed by the caller via downstream sites.
    # No optimizations are honoured at this revision — they're added in
    # subsequent commits, each with comparison-harness validation.
    _ = acceleration
    row_id     = int(row["row_id"])
    hint       = str(row["output"])
    candidates = list(candidates_order)
    targets    = set(row["targets"])
    black      = set(row["black"])
    tan        = set(row["tan"])

    giver_features = (
        extract_giver_features(row, giver_cols)
        if use_social_context else {}
    )

    prompt, feature_markers = build_prompt(
        hint=hint,
        candidates=candidates,
        giver_features=giver_features,
        use_social_context=use_social_context,
        tokenizer=tokenizer,
        chat_template_strategy=chat_template_strategy,
    )

    # --- Tokenization (with truncation for encoder models with a hard limit) ---
    if use_truncation:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
            max_length=max_seq_len,
            truncation=True,
        ).to(device)
    else:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            return_offsets_mapping=True,
        ).to(device)

    offset_mapping     = inputs["offset_mapping"][0].tolist()
    prompt_token_count = inputs["input_ids"].shape[1]
    truncated          = use_truncation and (prompt_token_count >= max_seq_len)

    # --- Build span targets ---
    spans_to_find = {"hint": hint}
    for c in candidates:
        spans_to_find[f"cand:{c}"] = c
    if use_social_context:
        for k, marker in feature_markers.items():
            spans_to_find[f"giver:{k}"] = marker

    spans = find_token_spans(prompt, offset_mapping, spans_to_find)

    if "hint" not in spans:
        raise ValueError(
            f"Hint span not found for row_id={row_id}, hint='{hint}'."
        )

    candidate_position_map = {w: i for i, w in enumerate(candidates)}

    # --- Forward pass (3-way dispatch on forward_hidden_states_mode) ---
    if forward_hidden_states_mode == "causal":
        # Causal LMs (Mistral, Qwen, Random_Qwen): explicit input_ids /
        # attention_mask, output_hidden_states passed at inference.
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
                return_dict=True,
            )
    elif forward_hidden_states_mode == "encoder_load_time":
        # BertModel / T5EncoderModel: output_hidden_states already on config.
        # Strip offset_mapping (these classes don't accept it).
        inputs_for_model = {k: v for k, v in inputs.items() if k != "offset_mapping"}
        with torch.no_grad():
            outputs = model(**inputs_for_model)
    elif forward_hidden_states_mode == "encoder_inference":
        # AutoModel (ModernBERT): output_hidden_states must be passed at
        # inference time, NOT on the config — the AutoModel convention.
        inputs_for_model = {k: v for k, v in inputs.items() if k != "offset_mapping"}
        with torch.no_grad():
            outputs = model(
                **inputs_for_model,
                output_hidden_states=True,
                return_dict=True,
            )
    else:
        raise ValueError(
            f"Unknown forward_hidden_states_mode: {forward_hidden_states_mode!r}. "
            "Expected one of: 'causal', 'encoder_load_time', 'encoder_inference'."
        )

    hidden_states = outputs.hidden_states

    # ================================================================
    # Compute metrics across ALL layers, for ALL pooling methods
    # ================================================================
    metrics_records: List[Dict] = []
    vector_records: Optional[List[Dict]] = [] if save_vectors else None

    for layer_idx in range(num_layers + 1):
        layer_hs = hidden_states[layer_idx][0]

        # --- Pool hint vector ---
        hint_vecs = {pm: pool_span(layer_hs, spans["hint"], method=pm)
                     for pm in pooling_methods}
        hint_token_count = spans["hint"][1] - spans["hint"][0]

        # --- Pool candidate vectors ---
        cand_vecs = {}
        cand_meta = {}
        for c in candidates:
            if c in targets:
                c_type = "target"
            elif c in black:
                c_type = "black"
            elif c in tan:
                c_type = "tan"
            else:
                c_type = "unknown"

            ck = f"cand:{c}"
            cand_vecs[c] = {}
            if ck in spans:
                c_token_count = spans[ck][1] - spans[ck][0]
                for pm in pooling_methods:
                    cand_vecs[c][pm] = pool_span(layer_hs, spans[ck], method=pm)
            else:
                c_token_count = 0
                for pm in pooling_methods:
                    cand_vecs[c][pm] = None
            cand_meta[c] = {"word_type": c_type, "token_count": c_token_count}

        # --- Pool giver feature vectors (with_social only) ---
        giver_vecs = {}
        giver_token_counts = {}
        if use_social_context:
            for feat_name, marker in feature_markers.items():
                fk = f"giver:{feat_name}"
                giver_vecs[feat_name] = {}
                if fk in spans:
                    giver_token_counts[feat_name] = spans[fk][1] - spans[fk][0]
                    for pm in pooling_methods:
                        giver_vecs[feat_name][pm] = pool_span(layer_hs, spans[fk], method=pm)
                else:
                    giver_token_counts[feat_name] = 0
                    for pm in pooling_methods:
                        giver_vecs[feat_name][pm] = None

        # --- Cosines: hint -> each candidate, per pooling method ---
        cosines_per_method = {}
        for pm in pooling_methods:
            cosines_per_method[pm] = {}
            h_vec = hint_vecs[pm]
            if h_vec is None:
                for c in candidates:
                    cosines_per_method[pm][c] = float("nan")
                continue
            h_vec_f32 = h_vec.astype(np.float32)
            for c in candidates:
                c_vec = cand_vecs[c][pm]
                if c_vec is not None:
                    cosines_per_method[pm][c] = cosine_similarity_np(
                        h_vec_f32, c_vec.astype(np.float32)
                    )
                else:
                    cosines_per_method[pm][c] = float("nan")

        # --- Ranks per pooling method ---
        ranks_per_method = {}
        for pm in pooling_methods:
            valid_cosines = {
                w: v for w, v in cosines_per_method[pm].items()
                if not np.isnan(v)
            }
            sorted_words = sorted(
                valid_cosines.keys(),
                key=lambda w: valid_cosines[w],
                reverse=True,
            )
            ranks_per_method[pm] = {}
            for rank_pos, w in enumerate(sorted_words, start=1):
                ranks_per_method[pm][w] = rank_pos
            for c in candidates:
                if c not in ranks_per_method[pm]:
                    ranks_per_method[pm][c] = float("nan")

        # --- All-pairs candidate cosines for anisotropy (mean pooling) ---
        all_pair_cosines_layer = []
        valid_cand_vecs_mean = []
        for c in candidates:
            v = cand_vecs[c]["mean"]
            if v is not None:
                valid_cand_vecs_mean.append(v.astype(np.float32))
        n_valid = len(valid_cand_vecs_mean)
        if n_valid >= 2:
            for i in range(n_valid):
                for j in range(i + 1, n_valid):
                    all_pair_cosines_layer.append(
                        cosine_similarity_np(
                            valid_cand_vecs_mean[i],
                            valid_cand_vecs_mean[j],
                        )
                    )

        if all_pair_cosines_layer:
            layer_aniso_mean = float(np.mean(all_pair_cosines_layer))
            layer_aniso_std  = float(np.std(all_pair_cosines_layer))
        else:
            layer_aniso_mean = float("nan")
            layer_aniso_std  = float("nan")

        # --- Build metric records: hint ---
        hint_metric = {
            "row_id"                     : row_id,
            "layer"                      : layer_idx,
            "word"                       : hint,
            "word_type"                  : "hint",
            "token_count"                : hint_token_count,
            "list_position"              : -1,
            "use_social_context"         : use_social_context,
            "permutation_id"             : permutation_id,
            "layer_mean_pairwise_cosine" : layer_aniso_mean,
            "layer_std_pairwise_cosine"  : layer_aniso_std,
        }
        for pm in pooling_methods:
            hint_metric[f"cosine_to_hint_{pm}"]  = float("nan")
            hint_metric[f"rank_{pm}"]            = float("nan")
            hint_metric[f"reciprocal_rank_{pm}"] = float("nan")
        metrics_records.append(hint_metric)

        # --- Build metric records: candidates ---
        for c in candidates:
            c_metric = {
                "row_id"                     : row_id,
                "layer"                      : layer_idx,
                "word"                       : c,
                "word_type"                  : cand_meta[c]["word_type"],
                "token_count"                : cand_meta[c]["token_count"],
                "list_position"              : candidate_position_map[c],
                "use_social_context"         : use_social_context,
                "permutation_id"             : permutation_id,
                "layer_mean_pairwise_cosine" : layer_aniso_mean,
                "layer_std_pairwise_cosine"  : layer_aniso_std,
            }
            for pm in pooling_methods:
                cos_val  = cosines_per_method[pm][c]
                rank_val = ranks_per_method[pm][c]
                c_metric[f"cosine_to_hint_{pm}"]  = cos_val
                c_metric[f"rank_{pm}"]            = rank_val
                c_metric[f"reciprocal_rank_{pm}"] = (
                    1.0 / rank_val if not np.isnan(rank_val) else float("nan")
                )
            metrics_records.append(c_metric)

        # --- Build metric records: giver features ---
        if use_social_context:
            for feat_name in feature_markers:
                gf_metric = {
                    "row_id"                     : row_id,
                    "layer"                      : layer_idx,
                    "word"                       : feat_name,
                    "word_type"                  : "giver_feature",
                    "token_count"                : giver_token_counts.get(feat_name, 0),
                    "list_position"              : -1,
                    "use_social_context"         : use_social_context,
                    "permutation_id"             : permutation_id,
                    "layer_mean_pairwise_cosine" : layer_aniso_mean,
                    "layer_std_pairwise_cosine"  : layer_aniso_std,
                }
                for pm in pooling_methods:
                    h_vec = hint_vecs[pm]
                    g_vec = giver_vecs[feat_name].get(pm)
                    if h_vec is not None and g_vec is not None:
                        gf_metric[f"cosine_to_hint_{pm}"] = cosine_similarity_np(
                            h_vec.astype(np.float32), g_vec.astype(np.float32)
                        )
                    else:
                        gf_metric[f"cosine_to_hint_{pm}"] = float("nan")
                    gf_metric[f"rank_{pm}"]            = float("nan")
                    gf_metric[f"reciprocal_rank_{pm}"] = float("nan")
                metrics_records.append(gf_metric)

        # --- Save vectors (subsample, canonical only) ---
        if save_vectors:
            for pm in pooling_methods:
                vector_records.append({
                    "row_id": row_id, "layer": layer_idx,
                    "word": hint, "word_type": "hint",
                    "token_count": hint_token_count,
                    "pooling_method": pm,
                    "use_social_context": use_social_context,
                    "vector": hint_vecs[pm],
                })
            for c in candidates:
                for pm in pooling_methods:
                    vector_records.append({
                        "row_id": row_id, "layer": layer_idx,
                        "word": c, "word_type": cand_meta[c]["word_type"],
                        "token_count": cand_meta[c]["token_count"],
                        "pooling_method": pm,
                        "use_social_context": use_social_context,
                        "vector": cand_vecs[c][pm],
                    })
            if use_social_context:
                for feat_name in feature_markers:
                    for pm in pooling_methods:
                        vector_records.append({
                            "row_id": row_id, "layer": layer_idx,
                            "word": feat_name, "word_type": "giver_feature",
                            "token_count": giver_token_counts.get(feat_name, 0),
                            "pooling_method": pm,
                            "use_social_context": use_social_context,
                            "vector": giver_vecs[feat_name].get(pm),
                        })

    # ================================================================
    # Behavioral prediction at final layer (per pooling method)
    # ================================================================
    # cosines_per_method and ranks_per_method now hold final-layer values.

    predicted_words = {}
    correct_flags   = {}
    for pm in pooling_methods:
        valid_scores = {
            w: cosines_per_method[pm][w]
            for w in candidates
            if not np.isnan(cosines_per_method[pm].get(w, float("nan")))
        }
        pw = max(valid_scores, key=valid_scores.get) if valid_scores else None
        predicted_words[pm] = pw
        correct_flags[pm]   = (pw in targets) if pw else False

    # Rank aggregation metrics
    rank_metrics = {}
    for pm in pooling_methods:
        target_ranks = [
            ranks_per_method[pm][w]
            for w in candidates
            if cand_meta[w]["word_type"] == "target"
            and not np.isnan(ranks_per_method[pm].get(w, float("nan")))
        ]
        if target_ranks:
            rank_metrics[f"mean_target_rank_{pm}"] = float(np.mean(target_ranks))
            rank_metrics[f"min_target_rank_{pm}"]  = float(np.min(target_ranks))
            rank_metrics[f"max_target_rank_{pm}"]  = float(np.max(target_ranks))
            rank_metrics[f"mrr_{pm}"]              = float(1.0 / np.min(target_ranks))
            rank_metrics[f"hit_at_1_{pm}"]         = float(np.min(target_ranks) == 1)
            rank_metrics[f"hit_at_3_{pm}"]         = float(np.min(target_ranks) <= 3)
            rank_metrics[f"hit_at_5_{pm}"]         = float(np.min(target_ranks) <= 5)
        else:
            for suffix in ["mean_target_rank", "min_target_rank",
                           "max_target_rank", "mrr",
                           "hit_at_1", "hit_at_3", "hit_at_5"]:
                rank_metrics[f"{suffix}_{pm}"] = float("nan")

    # Distance metrics
    distance_metrics = {}
    for pm in pooling_methods:
        tgt_cos = [
            cosines_per_method[pm][w] for w in candidates
            if cand_meta[w]["word_type"] == "target"
            and not np.isnan(cosines_per_method[pm].get(w, float("nan")))
        ]
        non_cos = [
            cosines_per_method[pm][w] for w in candidates
            if cand_meta[w]["word_type"] in ("black", "tan")
            and not np.isnan(cosines_per_method[pm].get(w, float("nan")))
        ]
        distance_metrics[f"mean_cos_hint_targets_{pm}"]    = float(np.mean(tgt_cos)) if tgt_cos else float("nan")
        distance_metrics[f"mean_cos_hint_nontargets_{pm}"] = float(np.mean(non_cos)) if non_cos else float("nan")
        distance_metrics[f"raw_margin_{pm}"] = (
            (float(np.mean(tgt_cos)) - float(np.mean(non_cos)))
            if tgt_cos and non_cos else float("nan")
        )
        valid_scores = {
            w: cosines_per_method[pm][w] for w in candidates
            if not np.isnan(cosines_per_method[pm].get(w, float("nan")))
        }
        sorted_scores = sorted(valid_scores.values(), reverse=True)
        distance_metrics[f"cos_gap_r1_r2_{pm}"] = (
            sorted_scores[0] - sorted_scores[1]
            if len(sorted_scores) >= 2 else float("nan")
        )

    missing_spans = [c for c in candidates if f"cand:{c}" not in spans]

    # --- Memory cleanup ---
    del hidden_states, outputs, layer_hs
    if device == "cuda":
        torch.cuda.empty_cache()

    general_record: Dict = {
        "row_id"             : row_id,
        "hint"               : hint,
        "n_targets"          : len(targets),
        "n_candidates"       : len(candidates),
        "n_missing_spans"    : len(missing_spans),
        "missing_span_words" : missing_spans,
        "prompt_token_count" : prompt_token_count,
    }
    # Only encoder models with truncation populate this column. Causal
    # general records omit it entirely, matching their reference notebooks.
    if use_truncation:
        general_record["truncated"] = truncated
    general_record.update({
        "use_social_context" : use_social_context,
        "permutation_id"     : permutation_id,
        "giver_features"     : giver_features if use_social_context else {},
    })
    for pm in pooling_methods:
        general_record[f"predicted_word_{pm}"] = predicted_words[pm]
        general_record[f"correct_{pm}"]        = correct_flags[pm]
    general_record.update(rank_metrics)
    general_record.update(distance_metrics)

    return general_record, metrics_records, vector_records
