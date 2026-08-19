"""Confirmatory stage logic: scan, patch, steer (causal_spec.md §5, §10).

Split out of ``runner`` so each stage is testable against a tiny model without
argparse or a Drive mount. The stages compose the tested primitives and own no
methodology of their own; every threshold and rule lives in the spec.

Stage 1 (scan) is a SCREEN and carries no inferential claim (§3.4). Stage 2
(patch) is what evidence rests on, and it is the ~94% of the compute budget
that must be resumable.
"""

import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import GIVER_COLS, extract_giver_features
from ..lens.readout import build_token_table
from ..prompts import build_prompt
from .attribution import attribution_scan, top_sites
from .basis import ROLES, collapse_grid, role_of_each_token, role_positions, span_positions
from .geometry import (
    cosine_to,
    displacement_matched_rotation,
    equalise_cosines,
    swap_cosines,
)
from .metrics import logit_difference
from .pairs import build_pair_table, substitute_hint
from .patch import all_sites, layer_window, run_patch
from .positions import readout_index
from .steer import (
    direction_diff_of_means,
    direction_from_unembedding,
    random_direction,
    shuffled_label_direction,
    steer_generate,
)

STEER_ARMS = ("primary", "random_direction", "shuffled_label", "counterfactual_target")

# T3d (causal_spec.md §5C). `equalise` is primary; the other three are the
# controls without which "removing the proximity changed the answer" cannot be
# told apart from "perturbing the states changed the answer".
EQUALISE_ARMS = ("equalise", "hold_target", "swap", "displacement")


class _Context:
    """Prompts, pairs and per-pair caches shared by every stage."""

    def __init__(self, *, model, tokenizer, df_sample, chat_template_strategy,
                 mode, seed, device=None, generation_csv=None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.mode_flag = mode == "with_social"
        self.strategy = chat_template_strategy
        self.seed = seed
        self.device = device or str(next(model.parameters()).device)
        self.by_id = df_sample.set_index("row_id")
        self.generations = self._load_generations(generation_csv)
        self.no_p_star = 0
        self.suffix_misaligned = 0

        hint_tokens = {
            int(r.row_id): len(tokenizer.encode(str(r.output), add_special_tokens=False))
            for r in df_sample.itertuples()
        }
        self.pairs = build_pair_table(
            df_sample, hint_tokens, seed=seed, match_length=True,
            prompt_length_fn=self.prompt_len,
        )

    @staticmethod
    def _load_generations(path):
        if not path:
            return None
        frame = pd.read_csv(path)
        return frame.set_index("row_id") if "row_id" in frame.columns else None

    def measurement(self, row_id: int, clean_prompt: str):
        """``(scored_suffix, p_star)`` for this turn under §4.1.

        The primary measurement position is the **answer position** ``p*``, not
        the generating position: on Mistral only 13.0% of turns emit the answer
        word first, so for the other 87% the final prompt token precedes 10+
        scaffolding tokens and reads a token the model is not about to emit.
        Because that deficit is model-specific (Qwen 91.5%), measuring at the
        generating position would confound RQ3's cross-architecture comparison
        with response format -- the error v3 exists to correct.

        Following §12.3, the turn's own recorded generation is teacher-forced
        onto the sequence, so the corrupted and patched runs each cost one
        forward pass rather than a generation. §5A length-matches the clean and
        donor prompts, so appending the *clean* generation to both leaves ``p*``
        at the same index in either run.

        Returns ``("", -1)`` when no generation is available, which falls back
        to the generating position -- correct for the calibration channel and
        for models with no recorded generations (the random-init null).
        """
        from .positions import answer_position

        if self.generations is None or row_id not in self.generations.index:
            return "", -1
        row = self.generations.loc[row_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        text, word = row.get("generated_text"), row.get("generated_word")
        if not isinstance(text, str) or not isinstance(word, str):
            self.no_p_star += 1
            return "", -1
        position = answer_position(self.tokenizer, clean_prompt, text, word)
        if position is None:
            self.no_p_star += 1
            return "", -1
        return text, int(position)

    def prompt(self, row_id: int, hint: str) -> str:
        row = self.by_id.loc[row_id]
        text, _ = build_prompt(
            hint=str(hint), candidates=list(row["candidates"]),
            giver_features=(extract_giver_features(row, GIVER_COLS)
                            if self.mode_flag else {}),
            use_social_context=self.mode_flag, tokenizer=self.tokenizer,
            chat_template_strategy=self.strategy,
        )
        return text

    def prompt_len(self, row_id: int, hint: str) -> int:
        return len(self.tokenizer.encode(self.prompt(row_id, hint),
                                         add_special_tokens=False))

    def joint_len(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def table(self, row_id: int) -> Dict[str, List[int]]:
        return build_token_table(self.tokenizer, list(self.by_id.loc[row_id, "candidates"]))

    def roles(self, pair, prompt: str, n_positions: int) -> np.ndarray:
        """Per-token role vector for this pair's clean prompt (basis.py).

        The corrupted prompt shares the role layout: §5A guarantees the two
        tokenise to the same length, and the donor hint occupies the same span.
        """
        return role_of_each_token(
            self.tokenizer, prompt, hint=str(pair.hint),
            candidates=list(self.by_id.loc[pair.row_id, "candidates"]),
            clean_target=str(pair.clean_target),
            donor_target=str(pair.donor_target),
            n_positions=n_positions,
        )

    def clean_cache(self, text: str, p_star: int = -1):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        cache = [h.detach().clone() for h in out.hidden_states]
        logits = out.logits[0, p_star].detach().float().cpu().numpy()
        return cache, logits, int(inputs["input_ids"].shape[1])

    def corrupt_logits(self, text: str, p_star: int = -1) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, p_star].detach().float().cpu().numpy()


def run_scan_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    top_k: int = 200, per_layer: bool = False, device: Optional[str] = None,
    generation_csv: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Attribution screen over the grid; returns the mean grid and ranked loci.

    The grid is aggregated on the **role basis** (``basis.py``), not on absolute
    token index. Prompts vary in token length across turns, so position ``p``
    names a different thing in every turn and a mean over absolute indices is
    not an aggregate of anything; on the first real run it did not even have a
    consistent shape. Each turn's ``(layer, position)`` grid is collapsed to
    ``(layer, role)`` and averaged across turns, skipping roles a turn lacks.

    SCREENING ONLY (§3.4): the returned loci carry no inferential claim and
    every one of them is confirmed with a real patch in stage 2.
    """
    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                   chat_template_strategy=chat_template_strategy, mode=mode,
                   seed=seed, device=device, generation_csv=generation_csv)

    collapsed: List[np.ndarray] = []
    for pair in ctx.pairs.itertuples():
        clean_prompt = ctx.prompt(pair.row_id, pair.hint)
        corrupt_prompt = ctx.prompt(pair.row_id, pair.donor_hint)
        suffix, p_star = ctx.measurement(pair.row_id, clean_prompt)
        # Readouts sit at p*-1, the position that EMITS the answer token —
        # reading at p* itself conditions on the answer already being present
        # (positions.readout_index; corrected 2026-07-30).
        p_read = readout_index(p_star)
        # Counterfactual scaffold (amendment (l)): the corrupted run must not
        # re-inject the clean hint through the teacher-forced generation.
        corrupt_suffix = substitute_hint(suffix, str(pair.hint),
                                         str(pair.donor_hint))
        if ctx.joint_len(clean_prompt + suffix) != \
                ctx.joint_len(corrupt_prompt + corrupt_suffix):
            ctx.suffix_misaligned += 1
            continue
        cache, _, n_positions = ctx.clean_cache(clean_prompt + suffix, p_read)
        grid = attribution_scan(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt + corrupt_suffix,
            readout_table=ctx.table(pair.row_id),
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=p_read, device=ctx.device,
        )
        collapsed.append(collapse_grid(grid, ctx.roles(pair, clean_prompt, n_positions)))

    if ctx.no_p_star:
        print(f"  [scan] {ctx.no_p_star} turns had no resolvable p*; "
              f"measured at the generating position instead (§4.1)")

    if not collapsed:
        raise ValueError("no aligned pairs; cannot run the attribution scan")

    stack = np.stack(collapsed)
    # A role absent from EVERY turn makes nanmean warn about an empty slice.
    # That is the defined outcome (the role stays NaN, see collapse_grid), not
    # a numerical problem, and on a real run the warning would repeat per call.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_grid = np.nanmean(stack, axis=0)
    # A role absent from every turn stays out of the ranking rather than
    # ranking as an exact zero, which would read as a measured null effect.
    ranked = np.nan_to_num(mean_grid, nan=0.0)

    if per_layer:
        # One locus per layer: the strongest role at each depth. This is what
        # yields a causal-effect CURVE over depth (RQ1 / the triangulation
        # figure) rather than a scatter of globally-strongest cells, which can
        # all sit at one depth.
        sites = [(int(layer), int(np.argmax(np.abs(ranked[layer]))))
                 for layer in range(ranked.shape[0])]
    else:
        sites = top_sites(ranked, k=top_k)
    loci = pd.DataFrame(
        [{"layer": l, "role": ROLES[r], "score": float(mean_grid[l, r]),
          "n_turns": int(np.isfinite(stack[:, l, r]).sum())}
         for l, r in sites]
    )
    return mean_grid, loci


def run_patch_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    loci: pd.DataFrame, window_widths: Sequence[int] = (1, 3, 5),
    device: Optional[str] = None, progress=None,
    checkpoint_dir: Optional[str] = None, prefix: str = "causal",
    resume: bool = False, flush_every: int = 25,
    stop_after_pairs: Optional[int] = None,
    generation_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Real patches on the candidate loci; one row per (locus, width, turn).

    This is the evidence stage and ~94% of the compute budget, so it is
    resumable through the same manifest machinery as the main extraction loop.
    Pairs are processed in a fixed order and each pair's rows depend only on
    that pair, so resuming and appending is byte-identical to an uninterrupted
    run -- the project's standing rule for ``--resume``.

    Effects that cannot be computed are recorded as NaN rather than dropped, so
    downstream counts stay honest.

    ``stop_after_pairs`` exists to exercise the interrupt path in tests.
    """
    from .. import checkpoint
    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                   chat_template_strategy=chat_template_strategy, mode=mode,
                   seed=seed, device=device, generation_csv=generation_csv)

    ckpt_prefix = f"{prefix}_patch"
    rows: List[Dict[str, object]] = []
    pairs_done = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if resume:
            manifest = checkpoint.read_manifest(checkpoint_dir, ckpt_prefix, mode)
            if manifest is not None:
                pairs_done = int(manifest.get("boards_done", 0))
                rows = list(checkpoint.load_records(
                    checkpoint_dir, ckpt_prefix, "patch", mode))
                print(f"  [resume] patch: continuing at pair "
                      f"{pairs_done}/{len(ctx.pairs)} ({len(rows)} rows cached)")
        else:
            checkpoint.remove_manifest(checkpoint_dir, ckpt_prefix, mode)
            checkpoint.remove_ckpts(checkpoint_dir, ckpt_prefix, mode,
                                    streams=("patch",))

    def _commit(done: int, shard: List[Dict[str, object]], idx: int) -> None:
        if checkpoint_dir is None:
            return
        checkpoint.write_records(shard, checkpoint_dir, ckpt_prefix,
                                 "patch", mode, idx)
        checkpoint.write_manifest(
            checkpoint_dir, ckpt_prefix, mode, n_boards=len(ctx.pairs),
            boards_done=done, ckpt_committed=idx,
            complete=(done == len(ctx.pairs)))

    shard: List[Dict[str, object]] = []
    for pair_idx, pair in enumerate(ctx.pairs.itertuples()):
        if pair_idx < pairs_done:
            continue
        if stop_after_pairs is not None and pair_idx >= pairs_done + stop_after_pairs:
            break
        clean_prompt = ctx.prompt(pair.row_id, pair.hint)
        corrupt_prompt = ctx.prompt(pair.row_id, pair.donor_hint)
        # Teacher-force the clean generation onto both runs so the readout sits
        # at the answer position p* rather than the generating position (§4.1,
        # §12.3). §5A length-matches the two prompts, so p* is the same index
        # in the clean and corrupted sequences.
        suffix, p_star = ctx.measurement(pair.row_id, clean_prompt)
        # p*-1 emits the answer token; p* already contains it (readout_index).
        p_read = readout_index(p_star)
        # Counterfactual scaffold (amendment (l)); see run_scan_stage.
        corrupt_suffix = substitute_hint(suffix, str(pair.hint),
                                         str(pair.donor_hint))
        if ctx.joint_len(clean_prompt + suffix) != \
                ctx.joint_len(corrupt_prompt + corrupt_suffix):
            ctx.suffix_misaligned += 1
            continue
        cache, clean_logits, n_positions = ctx.clean_cache(clean_prompt + suffix, p_read)
        table = ctx.table(pair.row_id)

        ld_clean = logit_difference(clean_logits, table,
                                    pair.clean_target, pair.donor_target)
        ld_corrupt = logit_difference(
            ctx.corrupt_logits(corrupt_prompt + corrupt_suffix, p_read), table,
            pair.clean_target, pair.donor_target)
        shared = dict(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt + corrupt_suffix, readout_table=table,
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=p_read, device=ctx.device,
            ld_clean=ld_clean, ld_corrupt=ld_corrupt,
        )

        # A locus names a (layer, role); the role resolves to THIS turn's own
        # token indices. That is what makes the intervention statable across
        # turns -- "patch the clean hint span at layer 5" has a sufficiency
        # reading, "patch position 47" does not.
        by_role = role_positions(ctx.roles(pair, clean_prompt, n_positions))

        for locus in loci.itertuples():
            positions = by_role.get(str(locus.role))
            if not positions:
                # Role absent from this turn: recorded as NaN, not dropped, so
                # downstream counts stay honest about coverage.
                for width in window_widths:
                    shard.append({
                        "layer": int(locus.layer), "role": str(locus.role),
                        "n_positions": 0, "width": int(width),
                        "row_id": int(pair.row_id), "effect": float("nan"),
                    })
                continue
            for width in window_widths:
                band = layer_window(int(locus.layer), int(width), len(cache))
                sites = [(l, p) for l in band for p in positions]
                shard.append({
                    "layer": int(locus.layer), "role": str(locus.role),
                    "n_positions": len(positions), "width": int(width),
                    "row_id": int(pair.row_id),
                    "effect": run_patch(sites=sites, **shared),
                })
        if progress is not None:
            progress()

        if checkpoint_dir is not None and (pair_idx + 1) % flush_every == 0:
            rows.extend(shard)
            _commit(pair_idx + 1, shard, pair_idx + 1)
            shard = []

    rows.extend(shard)
    if checkpoint_dir is not None and shard:
        _commit(min(pair_idx + 1, len(ctx.pairs)), shard, pair_idx + 1)

    return pd.DataFrame(rows)


def run_steer_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    layer: int, alphas: Sequence[float], sites: str = "from_hint",
    max_new_tokens: int = 24, device: Optional[str] = None,
) -> pd.DataFrame:
    """Dose-response over the four pre-registered arms (§3.3.5-8, §5B).

    ``primary`` is the label-free unembedding direction; the other three are
    the controls without which a steering result cannot be claimed.
    """
    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                  chat_template_strategy=chat_template_strategy, mode=mode,
                  seed=seed, device=device)

    unembed = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    hidden_dim = unembed.shape[1]
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(8, hidden_dim))
    labels = np.array([True] * 4 + [False] * 4)

    # Every arm is rescaled to the SAME norm-relative magnitude (spec §5B):
    # alpha multiplies the median residual norm at the injection layer. Raw
    # unembedding rows and difference-of-means vectors have unrelated scales,
    # so without this the arms are not comparable to each other or across
    # models - which is what made Qwen look inert.
    def _rescale(vec: np.ndarray, scale: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        return vec if norm == 0 else (vec / norm * scale).astype(np.float32)

    rows: List[Dict[str, object]] = []
    for pair in ctx.pairs.itertuples():
        prompt = ctx.prompt(pair.row_id, pair.hint)
        table = ctx.table(pair.row_id)
        clean_ids = table.get(pair.clean_target) or []
        donor_ids = table.get(pair.donor_target) or []
        if not clean_ids or not donor_ids:
            continue

        cache, _, _ = ctx.clean_cache(prompt)
        scale = (float(cache[layer][0].norm(dim=-1).median())
                 if layer < len(cache) else 1.0)
        directions = {
            "primary": _rescale(direction_from_unembedding(unembed, clean_ids), scale),
            "random_direction": random_direction(hidden_dim, norm=scale, seed=seed),
            "shuffled_label": _rescale(
                shuffled_label_direction(states, labels, seed=seed), scale),
            # Specificity control: aim at the DONOR's word instead (§3.3.8).
            "counterfactual_target": _rescale(
                direction_from_unembedding(unembed, donor_ids), scale),
        }

        baseline = steer_generate(
            model=model, tokenizer=tokenizer, prompt=prompt, layer=layer,
            direction=np.zeros(hidden_dim, dtype=np.float32), alpha=0.0,
            sites=sites, max_new_tokens=max_new_tokens, device=ctx.device,
        )
        for arm in STEER_ARMS:
            for alpha in alphas:
                text = steer_generate(
                    model=model, tokenizer=tokenizer, prompt=prompt, layer=layer,
                    direction=directions[arm], alpha=float(alpha), sites=sites,
                    max_new_tokens=max_new_tokens, device=ctx.device,
                )
                lowered = text.lower()
                rows.append({
                    "arm": arm, "alpha": float(alpha), "layer": int(layer),
                    "residual_scale": scale,
                    "row_id": int(pair.row_id), "generated": text,
                    "changed": text != baseline,
                    "hit_clean_target": pair.clean_target.lower() in lowered,
                    "hit_donor_target": pair.donor_target.lower() in lowered,
                    "parsed": bool(text.strip()),
                })
    return pd.DataFrame(rows)


def _pooled_spans(hidden, spans: Dict[str, Tuple[int, int]]):
    """Mean-pooled state per named span, matching what g(l) measures."""
    return {name: hidden[lo:hi].mean(0).double().cpu().numpy()
            for name, (lo, hi) in spans.items() if hi > lo}


def _rotated_sites(hidden, spans, new_vectors):
    """``(positions, values)`` writing each span's rotation back token by token.

    The transform is defined on the span's mean-pooled vector, because that is
    the vector the geometric readout ranks. Applying the same displacement to
    every token of the span moves the mean by exactly that displacement and
    leaves the within-span structure alone, which is the minimal edit with the
    intended effect on g(l).
    """
    import torch

    positions, values = [], []
    for name, (lo, hi) in spans.items():
        if name not in new_vectors or hi <= lo:
            continue
        before = hidden[lo:hi].mean(0).double()
        delta = torch.as_tensor(new_vectors[name], dtype=torch.float64,
                                device=hidden.device) - before
        for position in range(lo, hi):
            positions.append(position)
            values.append(hidden[position].double() + delta)
    return positions, values


def run_equalise_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    layers: Sequence[int], alphas: Sequence[float] = (1.0,),
    device: Optional[str] = None, generation_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Rotate the candidate states until the hint points at none of them (§5C).

    The paper's causal tier patches states; this stage intervenes on the
    *geometry* the paper is about. At each requested layer every candidate span
    is rotated so that all hint-to-candidate cosines are equal, the hint itself
    is left alone, and the answer is read at ``p_read`` exactly as in the
    patching grid, so the two live on the same axis.

    Four arms per (layer, alpha), in the order that makes the result readable:

    ``equalise``      the primary intervention, the target's advantage removed;
    ``hold_target``   everything *except* the target flattened, so the target
                      keeps its advantage — a specificity control that should
                      behave like the clean run;
    ``swap``          the target's and the donor's angles exchanged, which
                      predicts *which* word the answer should move to;
    ``displacement``  every candidate moved exactly as far as the primary arm
                      moved it, in a direction that leaves every hint-relative
                      cosine unchanged — the control that separates the
                      geometry from the size of the perturbation.

    Per-turn ``ld_clean``, ``ld_corrupt`` and ``ld_intervened`` are all stored,
    so the effect can be estimated as a ratio of sums rather than only as a mean
    of per-turn ratios.
    """
    import torch

    from .patch import _decoder_layers, module_for_layer, patch_hook

    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                   chat_template_strategy=chat_template_strategy, mode=mode,
                   seed=seed, device=device, generation_csv=generation_csv)
    decoder_layers = _decoder_layers(model)

    rows: List[Dict[str, object]] = []
    for pair in ctx.pairs.itertuples():
        clean_prompt = ctx.prompt(pair.row_id, pair.hint)
        corrupt_prompt = ctx.prompt(pair.row_id, pair.donor_hint)
        suffix, p_star = ctx.measurement(pair.row_id, clean_prompt)
        p_read = readout_index(p_star)
        clean_text = clean_prompt + suffix
        corrupt_text = corrupt_prompt + suffix

        table = ctx.table(pair.row_id)
        cache, clean_logits, n_positions = ctx.clean_cache(clean_text, p_read)
        ld_clean = logit_difference(clean_logits, table, str(pair.clean_target),
                                    str(pair.donor_target))
        ld_corrupt = logit_difference(
            ctx.corrupt_logits(corrupt_text, p_read), table,
            str(pair.clean_target), str(pair.donor_target))

        spans = span_positions(tokenizer, clean_prompt, hint=str(pair.hint),
                               candidates=list(ctx.by_id.loc[pair.row_id, "candidates"]))
        hint_span = spans.pop("hint", None)
        spans = {w: (lo, hi) for w, (lo, hi) in spans.items() if hi <= n_positions}
        if hint_span is None or len(spans) < 2:
            continue

        inputs = tokenizer(clean_text, return_tensors="pt").to(ctx.device)
        for layer in layers:
            if layer >= len(cache):
                continue
            hidden = cache[layer][0]
            hint_vec = hidden[hint_span[0]:hint_span[1]].mean(0).double().cpu().numpy()
            candidates = _pooled_spans(hidden, spans)
            before = cosine_to(candidates, hint_vec)
            pool_mean = float(np.mean(list(before.values())))
            target, donor = str(pair.clean_target), str(pair.donor_target)
            if target not in candidates or donor not in candidates:
                continue

            for alpha in alphas:
                primary = equalise_cosines(candidates, hint_vec, alpha=float(alpha))
                arms = {
                    "equalise": primary,
                    "hold_target": equalise_cosines(candidates, hint_vec,
                                                    alpha=float(alpha), hold=(target,)),
                    "swap": swap_cosines(candidates, hint_vec, target, donor,
                                         alpha=float(alpha)),
                    "displacement": displacement_matched_rotation(
                        candidates, hint_vec, reference=primary, seed=seed,
                        order=sorted(candidates)),
                }
                for arm in EQUALISE_ARMS:
                    positions, values = _rotated_sites(hidden, spans, arms[arm])
                    if not positions:
                        continue
                    module = module_for_layer(model, decoder_layers, layer,
                                              len(cache) - 1)
                    handle = module.register_forward_hook(
                        patch_hook(positions, torch.stack(values).to(hidden.dtype)))
                    try:
                        with torch.no_grad():
                            out = model(**inputs)
                        logits = out.logits[0, p_read].detach().float().cpu().numpy()
                    finally:
                        handle.remove()
                    new_vectors = arms[arm]
                    after = cosine_to(new_vectors, hint_vec)
                    rows.append({
                        "row_id": int(pair.row_id), "layer": int(layer),
                        "arm": arm, "alpha": float(alpha),
                        "n_candidates": len(candidates),
                        "ld_clean": ld_clean, "ld_corrupt": ld_corrupt,
                        "ld_intervened": logit_difference(
                            logits, table, target, donor),
                        "cos_target_before": before[target],
                        "cos_target_after": after[target],
                        "cos_donor_before": before[donor],
                        "cos_donor_after": after[donor],
                        "cos_pool_mean_before": pool_mean,
                    })
    return pd.DataFrame(rows)
