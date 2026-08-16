"""Full role x layer real-patch grid with matched nulls (causal_spec.md §3.3, §5A).

The confirmatory patch stage (``stages.run_patch_stage``) patches only the
loci the attribution scan surfaced -- under ``--per-layer`` one role per
layer. That leaves most cells of the (layer, role) grid untested: answer
positions were never patched at L4-15, the hint span never beyond L9, the
target-candidate span never at all. The grid stage patches EVERY requested
role at EVERY layer on the confirmatory draw, independent of the scan, and
runs the random-site baseline (§3.3.1) on the same turns so the paired
contrast of §3.4 exists per cell.

Roles are the ``basis.ROLES`` vocabulary plus two DERIVED roles that split the
teacher-forced answer span around the readout (pre-submission task T3b):

* ``p_read`` -- the single position whose logits are read (``p*-1``);
* ``scaffold`` -- the teacher-forced tokens BEFORE ``p_read``. Tokens after
  the readout cannot reach a causal readout, so they are excluded rather than
  padding the token count; a word-first turn therefore has no scaffold and
  the cell is NaN, not a measured zero.

``generation`` (all teacher-forced tokens, incl. ``p_read``) is kept for
continuity with the confirmatory run.

Random-site null (§3.3.1): per turn, for each DISTINCT token count among the
tested roles present on that turn, draw that many positions from tokens whose
role is NOT under test and that sit before the readout, then patch that set at
every grid layer. Matching by count rather than by role means one null row
serves every role of the same width; the analysis joins on
``(row_id, n_positions)``. Positions are drawn once per (turn, count) with a
generator seeded from ``(seed, row_id)``, so a resumed run draws the same
sites -- the standing byte-identity rule.

The grid permutation null (§3.3.2) is a sign-flip max-statistic over the
saved per-turn effects and lives in ``analysis.permutation_max_null``;
swapping the clean/corrupt assignment on the GPU is NOT a null under the
symmetric counterfactual (it is the mirror-direction replicate).
"""

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .basis import ROLES, role_positions
from .metrics import logit_difference
from .pairs import substitute_hint
from .patch import layer_window, run_patch_many
from .positions import readout_index
from .stages import _Context

DERIVED_ROLES = ("p_read", "scaffold")
GRID_ROLES = ("hint", "cand_target", "cand_donor", "final", "generation",
              "p_read", "scaffold")
NULL_ROLE = "random_site"
_ALL = "all"


def parse_roles(spec: str) -> Tuple[str, ...]:
    """``all`` -> the pre-registered grid set; else a validated comma list."""
    if spec.strip() == _ALL:
        return GRID_ROLES
    allowed = set(ROLES) | set(DERIVED_ROLES)
    out = []
    for name in (s.strip() for s in spec.split(",")):
        if not name:
            continue
        if name == NULL_ROLE:
            raise ValueError(f"'{NULL_ROLE}' is the null, never a treatment role")
        if name not in allowed:
            raise ValueError(f"unknown role {name!r}; choose from "
                             f"{sorted(allowed)} or 'all'")
        out.append(name)
    if not out:
        raise ValueError("no roles requested")
    return tuple(out)


def parse_layers(spec: str, n_layers: int) -> List[int]:
    """``all`` -> every cached index (embedding + blocks); else a comma list."""
    if spec.strip() == _ALL:
        return list(range(n_layers))
    out = []
    for token in (s.strip() for s in spec.split(",")):
        if not token:
            continue
        layer = int(token)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} outside [0, {n_layers})")
        out.append(layer)
    if not out:
        raise ValueError("no layers requested")
    return out


def grid_role_positions(by_role: Dict[str, List[int]], p_read: int,
                        roles: Sequence[str]) -> Dict[str, List[int]]:
    """Token indices per requested role, adding the derived roles.

    Roles absent from the turn are omitted (recorded as NaN by the caller).
    """
    out: Dict[str, List[int]] = {}
    for role in roles:
        if role == "p_read":
            out[role] = [int(p_read)]
        elif role == "scaffold":
            before = [p for p in by_role.get("generation", []) if p < p_read]
            if before:
                out[role] = before
        else:
            positions = by_role.get(role)
            if positions:
                out[role] = list(positions)
    return out


def random_site_positions(by_role: Dict[str, List[int]], *, p_read: int,
                          tested_roles: Sequence[str], k: int,
                          rng: np.random.Generator) -> List[int]:
    """``k`` positions (fewer if not enough exist) drawn from tokens whose role
    is not under test and that precede the readout."""
    tested = set(tested_roles)
    eligible = sorted({
        int(p) for role, positions in by_role.items()
        if role not in tested and role not in DERIVED_ROLES
        for p in positions if p < p_read
    })
    if not eligible:
        return []
    take = min(int(k), len(eligible))
    picked = rng.choice(len(eligible), size=take, replace=False)
    return sorted(eligible[i] for i in picked)


def run_grid_stage(
    *, model, tokenizer, df_sample, chat_template_strategy, mode, seed,
    roles: Sequence[str] = GRID_ROLES, layers: str = _ALL,
    window_widths: Sequence[int] = (1,), batch_size: int = 1,
    device: Optional[str] = None, progress=None,
    checkpoint_dir: Optional[str] = None, prefix: str = "causal",
    resume: bool = False, flush_every: int = 10,
    generation_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Real patches at every (layer, role, width) per turn, plus matched nulls.

    Returns ``(grid, nulls)``. ``grid`` has one row per (layer, role, width,
    turn) for EVERY requested role, NaN where the role is absent; ``nulls``
    has one ``random_site`` row per (layer, width, turn, distinct token count).
    Resumable through the manifest machinery, byte-identical to an
    uninterrupted run.
    """
    from .. import checkpoint

    roles = tuple(roles)
    ctx = _Context(model=model, tokenizer=tokenizer, df_sample=df_sample,
                   chat_template_strategy=chat_template_strategy, mode=mode,
                   seed=seed, device=device, generation_csv=generation_csv)

    ckpt_prefix = f"{prefix}_patchgrid"
    grid_rows: List[Dict[str, object]] = []
    null_rows: List[Dict[str, object]] = []
    pairs_done = 0

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        if resume:
            manifest = checkpoint.read_manifest(checkpoint_dir, ckpt_prefix, mode)
            if manifest is not None:
                pairs_done = int(manifest.get("boards_done", 0))
                grid_rows = list(checkpoint.load_records(
                    checkpoint_dir, ckpt_prefix, "grid", mode))
                null_rows = list(checkpoint.load_records(
                    checkpoint_dir, ckpt_prefix, "nulls", mode))
                print(f"  [resume] grid: continuing at pair "
                      f"{pairs_done}/{len(ctx.pairs)} ({len(grid_rows)} rows cached)")
        else:
            checkpoint.remove_manifest(checkpoint_dir, ckpt_prefix, mode)
            checkpoint.remove_ckpts(checkpoint_dir, ckpt_prefix, mode,
                                    streams=("grid", "nulls"))

    def _commit(done: int, g_shard, n_shard, idx: int) -> None:
        if checkpoint_dir is None:
            return
        checkpoint.write_records(g_shard, checkpoint_dir, ckpt_prefix, "grid", mode, idx)
        checkpoint.write_records(n_shard, checkpoint_dir, ckpt_prefix, "nulls", mode, idx)
        checkpoint.write_manifest(
            checkpoint_dir, ckpt_prefix, mode, n_boards=len(ctx.pairs),
            boards_done=done, ckpt_committed=idx,
            complete=(done == len(ctx.pairs)))

    g_shard: List[Dict[str, object]] = []
    n_shard: List[Dict[str, object]] = []
    layer_list: Optional[List[int]] = None
    pair_idx = pairs_done - 1
    t0, forwards = time.time(), 0
    for pair_idx, pair in enumerate(ctx.pairs.itertuples()):
        if pair_idx < pairs_done:
            continue
        clean_prompt = ctx.prompt(pair.row_id, pair.hint)
        corrupt_prompt = ctx.prompt(pair.row_id, pair.donor_hint)
        suffix, p_star = ctx.measurement(pair.row_id, clean_prompt)
        p_read = readout_index(p_star)
        corrupt_suffix = substitute_hint(suffix, str(pair.hint), str(pair.donor_hint))
        if ctx.joint_len(clean_prompt + suffix) != \
                ctx.joint_len(corrupt_prompt + corrupt_suffix):
            ctx.suffix_misaligned += 1
            continue
        cache, clean_logits, n_positions = ctx.clean_cache(clean_prompt + suffix, p_read)
        p_abs = p_read if p_read >= 0 else n_positions - 1
        if layer_list is None:
            layer_list = parse_layers(layers, len(cache))
        table = ctx.table(pair.row_id)
        ld_clean = logit_difference(clean_logits, table, pair.clean_target, pair.donor_target)
        ld_corrupt = logit_difference(
            ctx.corrupt_logits(corrupt_prompt + corrupt_suffix, p_read), table,
            pair.clean_target, pair.donor_target)

        by_role = role_positions(ctx.roles(pair, clean_prompt, n_positions))
        positions_of = grid_role_positions(by_role, p_abs, roles)

        # Null draws: one per distinct token count, seeded per turn.
        rng = np.random.default_rng([int(seed), int(pair.row_id)])
        counts: Dict[int, List[str]] = {}
        for role in roles:
            if role in positions_of:
                counts.setdefault(len(positions_of[role]), []).append(role)
        null_sets: Dict[int, List[int]] = {
            k: random_site_positions(by_role, p_read=p_abs, tested_roles=roles,
                                     k=k, rng=rng)
            for k in sorted(counts)
        }

        # Assemble every site set for this turn, then patch in batches.
        jobs: List[Tuple[str, object, int, int, List[int]]] = []  # kind, key, layer, width, positions
        for layer in layer_list:
            for width in window_widths:
                band = layer_window(int(layer), int(width), len(cache))
                for role in roles:
                    positions = positions_of.get(role)
                    if positions:
                        jobs.append(("grid", role, layer, width, [(l, p) for l in band for p in positions]))
                for k, positions in null_sets.items():
                    if positions:
                        jobs.append(("null", k, layer, width, [(l, p) for l in band for p in positions]))
        effects = run_patch_many(
            model=model, tokenizer=tokenizer, clean_cache=cache,
            corrupt_prompt=corrupt_prompt + corrupt_suffix,
            site_sets=[j[4] for j in jobs], readout_table=table,
            clean_target=pair.clean_target, donor_target=pair.donor_target,
            p_star=p_read, ld_clean=ld_clean, ld_corrupt=ld_corrupt,
            device=ctx.device, batch_size=batch_size,
        ) if jobs else np.zeros(0)
        forwards += len(jobs) + 2
        measured: Dict[Tuple[str, object, int, int], float] = {
            (kind, key, layer, width): float(effects[i])
            for i, (kind, key, layer, width, _) in enumerate(jobs)
        }
        for layer in layer_list:
            for width in window_widths:
                for role in roles:
                    positions = positions_of.get(role, [])
                    g_shard.append({
                        "layer": int(layer), "role": str(role),
                        "n_positions": len(positions), "width": int(width),
                        "row_id": int(pair.row_id),
                        "effect": measured.get(("grid", role, layer, width), float("nan")),
                    })
                for k, positions in null_sets.items():
                    n_shard.append({
                        "layer": int(layer), "role": NULL_ROLE,
                        "n_positions": len(positions), "matched_n": int(k),
                        "matched_roles": "|".join(counts[k]),
                        "width": int(width), "row_id": int(pair.row_id),
                        "effect": measured.get(("null", k, layer, width), float("nan")),
                    })
        if progress is not None:
            progress()
        if checkpoint_dir is not None and (pair_idx + 1) % flush_every == 0:
            grid_rows.extend(g_shard)
            null_rows.extend(n_shard)
            _commit(pair_idx + 1, g_shard, n_shard, pair_idx + 1)
            g_shard, n_shard = [], []
            elapsed = max(time.time() - t0, 1e-9)
            done_now = pair_idx + 1 - pairs_done
            remaining = len(ctx.pairs) - (pair_idx + 1)
            print(f"  [grid] {pair_idx + 1}/{len(ctx.pairs)} pairs, "
                  f"{forwards / elapsed:.1f} fwd/s, batch {batch_size}; "
                  f"~{remaining * elapsed / max(done_now, 1) / 3600:.2f} h left",
                  flush=True)

    grid_rows.extend(g_shard)
    null_rows.extend(n_shard)
    if checkpoint_dir is not None and (g_shard or n_shard):
        _commit(min(pair_idx + 1, len(ctx.pairs)), g_shard, n_shard, pair_idx + 1)
    if ctx.no_p_star:
        print(f"  [grid] {ctx.no_p_star} turns had no resolvable p*; measured at "
              f"the generating position instead (§4.1)")
    if ctx.suffix_misaligned:
        print(f"  [grid] {ctx.suffix_misaligned} pairs dropped: teacher-forced "
              f"suffix misaligned")

    grid_cols = ["layer", "role", "n_positions", "width", "row_id", "effect"]
    null_cols = ["layer", "role", "n_positions", "matched_n", "matched_roles",
                 "width", "row_id", "effect"]
    return (pd.DataFrame(grid_rows, columns=grid_cols),
            pd.DataFrame(null_rows, columns=null_cols))
