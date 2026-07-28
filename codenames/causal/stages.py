"""Confirmatory stage logic: scan, patch, steer (causal_spec.md §5, §10).

Split out of ``runner`` so each stage is testable against a tiny model without
argparse or a Drive mount. The stages compose the tested primitives and own no
methodology of their own; every threshold and rule lives in the spec.

Stage 1 (scan) is a SCREEN and carries no inferential claim (§3.4). Stage 2
(patch) is what evidence rests on, and it is the ~94% of the compute budget
that must be resumable.
"""

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data import GIVER_COLS, extract_giver_features
from ..lens.readout import build_token_table
from ..prompts import build_prompt
from .attribution import attribution_scan, top_sites
from .metrics import logit_difference
from .pairs import build_pair_table
from .patch import all_sites, layer_window, run_patch
from .steer import (
    direction_diff_of_means,
    direction_from_unembedding,
    random_direction,
    shuffled_label_direction,
    steer_generate,
)

STEER_ARMS = ("primary", "random_direction", "shuffled_label", "counterfactual_target")


class _Context:
    """Prompts, pairs and per-pair caches shared by every stage."""

    def __init__(self, *, model, tokenizer, df_sample, chat_template_strategy,
                 mode, seed, device=None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.mode_flag = mode == "with_social"
        self.strategy = chat_template_strategy
        self.seed = seed
        self.device = device or str(next(model.parameters()).device)
        self.by_id = df_sample.set_index("row_id")

        hint_tokens = {
            int(r.row_id): len(tokenizer.encode(str(r.output), add_special_tokens=False))
            for r in df_sample.itertuples()
        }
        self.pairs = build_pair_table(
            df_sample, hint_tokens, seed=seed, match_length=True,
            prompt_length_fn=self.prompt_len,
        )

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

    def table(self, row_id: int) -> Dict[str, List[int]]:
        return build_token_table(self.tokenizer, list(self.by_id.loc[row_id, "candidates"]))

    def clean_cache(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)
        cache = [h.detach().clone() for h in out.hidden_states]
        logits = out.logits[0, -1].detach().float().cpu().numpy()
        return cache, logits, int(inputs["input_ids"].shape[1])

    def corrupt_logits(self, prompt: str) -> np.ndarray:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        return out.logits[0, -1].detach().float().cpu().numpy()


def run_scan_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    top_k: int = 200, device: Optional[str] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Attribution screen over the grid; returns the mean grid and ranked loci.

    SCREENING ONLY (§3.4): the returned loci carry no inferential claim and
    every one of them is confirmed with a real patch in stage 2.
    """
    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                   chat_template_strategy=chat_template_strategy, mode=mode,
                   seed=seed, device=device)

    total: Optional[np.ndarray] = None
    counted = 0
    for pair in ctx.pairs.itertuples():
        clean_prompt = ctx.prompt(pair.row_id, pair.hint)
        corrupt_prompt = ctx.prompt(pair.row_id, pair.donor_hint)
        cache, _, _ = ctx.clean_cache(clean_prompt)
        grid = attribution_scan(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt, readout_table=ctx.table(pair.row_id),
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=-1, device=ctx.device,
        )
        total = grid.copy() if total is None else total + grid
        counted += 1

    if total is None:
        raise ValueError("no aligned pairs; cannot run the attribution scan")

    mean_grid = total / counted
    sites = top_sites(mean_grid, k=top_k)
    loci = pd.DataFrame(
        [{"layer": l, "position": p, "score": float(mean_grid[l, p])}
         for l, p in sites]
    )
    return mean_grid, loci


def run_patch_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    loci: pd.DataFrame, window_widths: Sequence[int] = (1, 3, 5),
    device: Optional[str] = None, progress=None,
    checkpoint_dir: Optional[str] = None, prefix: str = "causal",
    resume: bool = False, flush_every: int = 25,
    stop_after_pairs: Optional[int] = None,
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
                   seed=seed, device=device)

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
        cache, clean_logits, n_positions = ctx.clean_cache(clean_prompt)
        table = ctx.table(pair.row_id)

        ld_clean = logit_difference(clean_logits, table,
                                    pair.clean_target, pair.donor_target)
        ld_corrupt = logit_difference(ctx.corrupt_logits(corrupt_prompt), table,
                                      pair.clean_target, pair.donor_target)
        shared = dict(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt, readout_table=table,
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=-1, device=ctx.device,
            ld_clean=ld_clean, ld_corrupt=ld_corrupt,
        )

        for locus in loci.itertuples():
            if locus.position >= n_positions:
                continue
            for width in window_widths:
                band = layer_window(int(locus.layer), int(width), len(cache))
                sites = [(l, int(locus.position)) for l in band]
                shard.append({
                    "layer": int(locus.layer), "position": int(locus.position),
                    "width": int(width), "row_id": int(pair.row_id),
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
