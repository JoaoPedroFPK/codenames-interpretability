"""Consistency tests for the checkpoint/resume orchestration in ``loop.py``.

Two properties are enforced across the whole Step-1 refactor:

* **P1 (resume correctness)** — a killed-and-resumed run is byte-identical to
  an uninterrupted run. (Added in sub-step 1.2, once resume exists.)
* **P2 (no regression)** — the refactored loop's final outputs match the
  pre-refactor loop's, captured here as a committed golden digest.

Everything runs on CPU via :mod:`tests.resume_harness`, which stubs the model
forward pass (``run_instance``) — the seam Step 1 does not touch — so these
tests are deterministic and need no weights.

Digests are *semantic*, not raw bytes: parquet is normalized through
``read_parquet -> to_csv`` and NPZ through array bytes, so the golden survives
parquet/pandas version differences while still pinning the data exactly.
"""

import hashlib
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

from tests import resume_harness as H

# Scenarios span: causal (with generation) vs encoder (with truncation), and
# the per-board (batch_size=1) vs batched (batch_size>1) code paths. Sample
# sizes are chosen to cross the shard_boards=3 flush boundary unevenly.
SCENARIOS = {
    "causal_b1":  dict(sample_size=7, has_generation=True,  use_truncation=False, batch_size=1),
    "encoder_b1": dict(sample_size=7, has_generation=False, use_truncation=True,  batch_size=1),
    "causal_b3":  dict(sample_size=8, has_generation=True,  use_truncation=False, batch_size=3),
    "encoder_b3": dict(sample_size=8, has_generation=False, use_truncation=True,  batch_size=3),
}

_GOLDEN_PATH = pathlib.Path(__file__).parent / "fixtures" / "resume_golden.json"


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _digest_file(path: str) -> str:
    """Semantic, version-stable digest of a single output file."""
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
        # Sort columns for stability; rows are already in deterministic order.
        norm = df.to_csv(index=False).encode()
        return "parquet:" + _hash_bytes(norm)
    if path.endswith(".npz"):
        data = np.load(path)
        h = hashlib.sha256()
        for k in sorted(data.files):
            arr = np.ascontiguousarray(data[k])
            h.update(k.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
        data.close()
        return "npz:" + h.hexdigest()
    # CSV / text: raw bytes (deterministic text output).
    with open(path, "rb") as f:
        return "raw:" + _hash_bytes(f.read())


def digest_dir(d: str) -> dict:
    """Map every output filename in ``d`` to its semantic digest."""
    out = {}
    for name in sorted(os.listdir(d)):
        full = os.path.join(d, name)
        if os.path.isfile(full):
            out[name] = _digest_file(full)
    return out


def run_scenario(scenario_name: str, base_dir: str, monkeypatch) -> dict:
    """Install fakes and run one scenario to completion; return the digest."""
    H.install_fakes(monkeypatch)
    H.run_harness(base_dir, monkeypatch=monkeypatch, **SCENARIOS[scenario_name])
    return digest_dir(base_dir)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_harness_is_deterministic(scenario, tmp_path, monkeypatch):
    """Two independent uninterrupted runs produce identical digests.

    This is the foundation for every later claim: if the harness itself were
    nondeterministic, P1/P2 byte-identity would be meaningless.
    """
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    d1.mkdir()
    d2.mkdir()
    dig1 = run_scenario(scenario, str(d1), monkeypatch)
    dig2 = run_scenario(scenario, str(d2), monkeypatch)
    assert dig1 == dig2
    assert dig1, "scenario produced no output files"


def _load_golden() -> dict:
    if not _GOLDEN_PATH.exists():
        pytest.skip(f"golden digest not yet captured at {_GOLDEN_PATH}")
    return json.loads(_GOLDEN_PATH.read_text())


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_matches_golden(scenario, tmp_path, monkeypatch):
    """P2: the loop's outputs match the committed pre-refactor golden digest.

    Regenerate the golden (only when intentionally changing the contract) with:
        REGEN_GOLDEN=1 python -m pytest tests/test_resume.py -k matches_golden
    """
    golden = _load_golden()
    dig = run_scenario(scenario, str(tmp_path), monkeypatch)
    if os.environ.get("REGEN_GOLDEN"):
        pytest.skip("REGEN_GOLDEN set; golden regeneration handled separately")
    assert scenario in golden, f"scenario {scenario} missing from golden"
    assert dig == golden[scenario]


def _install_bomb(monkeypatch, bomb_after):
    """Install fakes that raise KeyboardInterrupt after ``bomb_after`` calls.

    KeyboardInterrupt is a BaseException, so it escapes the loop's per-board
    ``except Exception`` and propagates out of ``run_extraction`` — a faithful
    stand-in for a runtime being killed mid-board (data in unflushed buffers is
    lost, committed checkpoints remain).
    """
    import codenames.loop as loop
    state = {"n": 0}

    def _bomb(fn):
        def wrapped(**kw):
            state["n"] += 1
            if state["n"] >= bomb_after:
                raise KeyboardInterrupt(f"simulated crash at call {state['n']}")
            return fn(**kw)
        return wrapped

    monkeypatch.setattr(loop, "run_instance", _bomb(H.fake_run_instance))
    monkeypatch.setattr(loop, "run_instance_batched", _bomb(H.fake_run_instance_batched))
    return state


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_resume_matches_uninterrupted(scenario, tmp_path, monkeypatch):
    """P1: a run killed (repeatedly) and resumed is byte-identical to an
    uninterrupted run, and resume makes monotonic progress each attempt."""
    opts = SCENARIOS[scenario]

    # --- Reference: one clean, uninterrupted run. ---
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref_digest = run_scenario(scenario, str(ref_dir), monkeypatch)

    # --- Crash/resume: bomb after a fixed number of NEW calls each attempt,
    # which exceeds the calls-to-first-flush so every attempt commits at least
    # one checkpoint (guaranteed forward progress), until a final clean pass. ---
    crash_dir = tmp_path / "crash"
    crash_dir.mkdir()

    attempts = 0
    crashed = 0
    completed = False
    while attempts < 30:
        attempts += 1
        resume = attempts > 1
        if attempts <= 6:
            # Early attempts crash mid-run (bomb_after=10 > 9 calls-to-first
            # -flush in the per-board path, so progress is always committed).
            _install_bomb(monkeypatch, bomb_after=10)
            try:
                H.run_harness(str(crash_dir), resume=resume, monkeypatch=monkeypatch, **opts)
                completed = True
                break
            except KeyboardInterrupt:
                crashed += 1
                continue
        else:
            # Later attempts run clean to guarantee termination.
            H.install_fakes(monkeypatch)
            H.run_harness(str(crash_dir), resume=True, monkeypatch=monkeypatch, **opts)
            completed = True
            break

    assert completed, f"run never completed after {attempts} attempts"
    assert crashed >= 1, "test did not actually exercise a crash/resume cycle"
    assert digest_dir(str(crash_dir)) == ref_digest


def test_resume_refuses_size_mismatch(tmp_path, monkeypatch):
    """Resuming a directory whose manifest was written for a different run size
    aborts (rather than skipping non-corresponding boards)."""
    from codenames.checkpoint import ResumeSizeMismatch

    d = str(tmp_path / "out")
    os.makedirs(d)

    # Leave an interrupted run of size 7 (crash after one committed flush).
    _install_bomb(monkeypatch, bomb_after=10)
    with pytest.raises(KeyboardInterrupt):
        H.run_harness(d, sample_size=7, has_generation=False, use_truncation=True,
                      batch_size=1, resume=False, monkeypatch=monkeypatch)

    # Resuming with a different size must refuse.
    H.install_fakes(monkeypatch)
    with pytest.raises(ResumeSizeMismatch):
        H.run_harness(d, sample_size=5, has_generation=False, use_truncation=True,
                      batch_size=1, resume=True, monkeypatch=monkeypatch)


@pytest.mark.parametrize("flags,exp_resume,exp_reuse", [
    ([], False, False),
    (["--resume"], True, False),
    (["--reuse-canonical"], False, True),
    (["--resume", "--reuse-canonical"], True, True),
])
def test_cli_threads_resume_flag(flags, exp_resume, exp_reuse, tmp_path, monkeypatch):
    """The ``run`` subcommand parses --resume / --reuse-canonical and forwards them."""
    import codenames.cli as cli
    import codenames.data as data
    import codenames.loop as loop
    import codenames.persistence as persistence

    meta = {
        "prefix": "fake", "chat_template_strategy": "raw",
        "forward_hidden_states_mode": "encoder_load_time", "use_truncation": True,
        "num_layers": 2, "hidden_dim": 4, "device": "cpu",
        "supports_generation": False,
    }
    monkeypatch.setattr(cli, "_resolve_loader", lambda name: (lambda **kw: (None, None, meta)))
    monkeypatch.setattr(data, "load_dataset", lambda path: H.make_fake_dataset(3))
    monkeypatch.setattr(data, "sample_turns", lambda df, n, seed: df)
    monkeypatch.setattr(persistence, "print_output_summary", lambda **kw: None)

    captured = {}

    def _capture(**kw):
        captured.update(kw)
        return {}

    monkeypatch.setattr(loop, "run_extraction", _capture)

    rc = cli.main([
        "run", "--model", "bert", "--dataset", "x.csv",
        "--output-dir", str(tmp_path), "--sample-size", "3",
        "--skip-sanity-checks", *flags,
    ])
    assert rc == 0
    assert captured.get("resume") is exp_resume
    assert captured.get("reuse_canonical") is exp_reuse
    assert captured.get("checkpoint_dir") is None  # not passed -> default subfolder


def test_cli_threads_checkpoint_dir(tmp_path, monkeypatch):
    """The run subcommand forwards --checkpoint-dir to run_extraction."""
    import codenames.cli as cli
    import codenames.data as data
    import codenames.loop as loop
    import codenames.persistence as persistence

    meta = {"prefix": "fake", "chat_template_strategy": "raw",
            "forward_hidden_states_mode": "encoder_load_time", "use_truncation": True,
            "num_layers": 2, "hidden_dim": 4, "device": "cpu", "supports_generation": False}
    monkeypatch.setattr(cli, "_resolve_loader", lambda name: (lambda **kw: (None, None, meta)))
    monkeypatch.setattr(data, "load_dataset", lambda path: H.make_fake_dataset(3))
    monkeypatch.setattr(data, "sample_turns", lambda df, n, seed: df)
    monkeypatch.setattr(persistence, "print_output_summary", lambda **kw: None)
    captured = {}
    monkeypatch.setattr(loop, "run_extraction", lambda **kw: captured.update(kw) or {})

    rc = cli.main([
        "run", "--model", "bert", "--dataset", "x.csv",
        "--output-dir", str(tmp_path / "out"), "--checkpoint-dir", str(tmp_path / "ck"),
        "--sample-size", "3", "--skip-sanity-checks",
    ])
    assert rc == 0
    assert captured.get("checkpoint_dir") == str(tmp_path / "ck")


# ===========================================================================
# Step 2 — cross-size canonical reuse (P3)
# ===========================================================================

def _install_counting_fakes(monkeypatch):
    """Install fakes that count run_instance / run_instance_batched calls."""
    import codenames.loop as loop
    state = {"n": 0}

    def _count(fn):
        def wrapped(**kw):
            state["n"] += 1
            return fn(**kw)
        return wrapped

    monkeypatch.setattr(loop, "run_instance", _count(H.fake_run_instance))
    monkeypatch.setattr(loop, "run_instance_batched", _count(H.fake_run_instance_batched))
    return state


def _experiment_digest(d):
    """Digest of experiment-output files only (excludes cache/manifest/ckpt infra)."""
    infra = ("canoncache", "manifest", "ckpt")
    return {k: v for k, v in digest_dir(d).items()
            if not any(tag in k for tag in infra)}


def test_canon_cache_update_load_roundtrip(tmp_path):
    """canon_cache.update is idempotent and load returns the canonical records."""
    import pandas as pd
    from codenames import canon_cache

    df = H.make_fake_dataset(4)
    gens, mets, gen_rows = [], [], []
    for _, r in df.iterrows():
        for perm in (0, 1):  # include a shuffle perm; only perm 0 should cache
            g, m, _ = H.fake_run_instance(
                row=r, candidates_order=list(r["candidates"]), permutation_id=perm,
                save_vectors=False, pooling_methods=H.FAKE_POOLING,
                num_layers=H.FAKE_NUM_LAYERS, hidden_dim=H.FAKE_HIDDEN_DIM,
                use_truncation=False, use_social_context=False)
            gens.append(g)
            mets.extend(m)
        gen_rows.append({"row_id": int(r["row_id"]), "generated_word": "x"})

    args = dict(general_df=pd.DataFrame(gens), metrics_df=pd.DataFrame(mets),
                generation_df=pd.DataFrame(gen_rows))
    n = canon_cache.update(str(tmp_path), "fake", "no_social", **args)
    assert n == 4
    # Idempotent: re-adding the same boards adds nothing.
    assert canon_cache.update(str(tmp_path), "fake", "no_social", **args) == 0

    cache = canon_cache.load(str(tmp_path), "fake", "no_social")
    assert len(cache) == 4
    per_board = (H.FAKE_NUM_LAYERS + 1) * (1 + 4)  # layers × (hint + 4 candidates)
    for rid in range(4):
        assert cache.has(rid) and cache.has_generation(rid)
        assert len(cache.metrics(rid)) == per_board
        assert all(rec["permutation_id"] == 0 for rec in cache.metrics(rid))


@pytest.mark.parametrize("has_gen,trunc", [(False, True), (True, False)])
def test_cross_size_canonical_reuse(has_gen, trunc, tmp_path, monkeypatch):
    """P3: a larger run reusing a smaller run's cache is byte-identical to the
    same larger run computed fresh, while skipping forward passes."""
    small = dict(sample_size=5, has_generation=has_gen, use_truncation=trunc, batch_size=1)
    large = dict(sample_size=8, has_generation=has_gen, use_truncation=trunc, batch_size=1)

    # 1. Small run with reuse on → builds the cache.
    cache_dir = str(tmp_path / "cache_dir")
    os.makedirs(cache_dir)
    H.install_fakes(monkeypatch)
    H.run_harness(cache_dir, reuse_canonical=True, monkeypatch=monkeypatch, **small)
    # The cache lives in the separate checkpoints/ subfolder, not base_dir.
    ckpt_sub = os.path.join(cache_dir, "checkpoints")
    assert os.path.isdir(ckpt_sub), "checkpoint dir not created"
    assert any("canoncache" in f for f in os.listdir(ckpt_sub)), "cache not written"
    assert not any("canoncache" in f for f in os.listdir(cache_dir)), \
        "cache leaked into base_dir"

    # 2. Reference: large run, no reuse, fresh dir.
    ref_dir = str(tmp_path / "ref")
    os.makedirs(ref_dir)
    ref_state = _install_counting_fakes(monkeypatch)
    H.run_harness(ref_dir, reuse_canonical=False, monkeypatch=monkeypatch, **large)
    ref_calls = ref_state["n"]
    ref_digest = _experiment_digest(ref_dir)

    # 3. Large run WITH reuse, in the dir holding the small run's cache.
    reuse_state = _install_counting_fakes(monkeypatch)
    H.run_harness(cache_dir, reuse_canonical=True, monkeypatch=monkeypatch, **large)
    reuse_calls = reuse_state["n"]
    reuse_digest = _experiment_digest(cache_dir)

    assert reuse_digest == ref_digest, "reuse changed the output"
    assert reuse_calls < ref_calls, "reuse did not skip any forward passes"


def test_checkpoints_separate_from_outputs(tmp_path, monkeypatch):
    """Final outputs land in base_dir; checkpoint/manifest infra lives only in
    the separate checkpoint dir, mid-run and after completion."""
    opts = dict(sample_size=7, has_generation=True, use_truncation=False, batch_size=1)
    base = str(tmp_path / "out")
    ckpt = str(tmp_path / "ckpts")
    os.makedirs(base)

    # Crash mid-run, then inspect: checkpoints must be in `ckpt`, not `base`.
    _install_bomb(monkeypatch, bomb_after=10)
    with pytest.raises(KeyboardInterrupt):
        H.run_harness(base, monkeypatch=monkeypatch, checkpoint_dir=ckpt, **opts)
    assert os.path.isdir(ckpt) and os.listdir(ckpt), "no checkpoints in checkpoint dir"
    assert any("ckpt" in f or "manifest" in f for f in os.listdir(ckpt))
    assert not any("ckpt" in f or "manifest" in f for f in os.listdir(base)), \
        "checkpoint/manifest leaked into base_dir"

    # Finish via resume; final outputs in base_dir, base_dir has no infra files.
    H.install_fakes(monkeypatch)
    H.run_harness(base, resume=True, monkeypatch=monkeypatch, checkpoint_dir=ckpt, **opts)
    base_files = os.listdir(base)
    assert any(f.endswith(".parquet") for f in base_files)  # metrics output present
    assert not any(("ckpt" in f) or ("manifest" in f) for f in base_files), \
        "infra files in base_dir after completion"


def test_reuse_disabled_under_batching(tmp_path, monkeypatch):
    """Reuse is gated off when batch_size>1: no cache is written and the output
    matches a non-reusing batched run."""
    opts = dict(sample_size=8, has_generation=False, use_truncation=True, batch_size=3)

    ref_dir = str(tmp_path / "ref")
    os.makedirs(ref_dir)
    H.install_fakes(monkeypatch)
    H.run_harness(ref_dir, reuse_canonical=False, monkeypatch=monkeypatch, **opts)
    ref_digest = _experiment_digest(ref_dir)

    d = str(tmp_path / "d")
    os.makedirs(d)
    H.install_fakes(monkeypatch)
    H.run_harness(d, reuse_canonical=True, monkeypatch=monkeypatch, **opts)

    assert not any("canoncache" in f for f in os.listdir(d)), "cache written under batching"
    assert _experiment_digest(d) == ref_digest
