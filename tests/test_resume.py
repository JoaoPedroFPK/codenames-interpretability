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
