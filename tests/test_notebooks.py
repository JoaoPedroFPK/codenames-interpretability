"""Static + dynamic guards on the model notebooks' run configuration.

The notebooks are the primary way the experiment is launched on Colab, and a
misconfigured run wastes hours of GPU time before the mistake surfaces. These
tests pin the run-configuration invariants so a regression fails here instead:

* the run processes the **full dataset** (not the N=2000 contract default) — the
  exact bug that shipped once (notebook passed ``contract=CONTRACT_V1`` whose
  ``sample_size=2000`` while ``run_extraction`` re-samples to that size);
* resume and the separate checkpoint directory are wired on **both** run paths
  (the Path A CLI cell and the Path B ``run_extraction`` cell);
* the agreed flag values (RESUME on, REUSE off under batching) are set.

The dynamic test reproduces the trap end-to-end with the GPU-free harness.
"""

import dataclasses
import glob
import json
import os

import pytest

from codenames.contract import ACCEL_REFERENCE
from codenames.data import sample_turns
from tests import resume_harness as H

NB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notebooks")
MODEL_NBS = sorted(g for g in glob.glob(os.path.join(NB_DIR, "0*.ipynb"))
                   if "00_validation" not in os.path.basename(g))


def _code_cells(path):
    nb = json.loads(open(path).read())
    return [("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
            for c in nb["cells"] if c["cell_type"] == "code"]


def _first(cells, needle):
    return next(s for s in cells if needle in s)


def test_there_are_seven_model_notebooks():
    assert len(MODEL_NBS) == 7, MODEL_NBS


@pytest.mark.parametrize("nb", MODEL_NBS, ids=lambda p: os.path.basename(p))
def test_pathB_full_run_and_flags(nb):
    """Path B (direct run_extraction) runs the full dataset with the right
    contract, flags, and separated checkpoint dir."""
    cells = _code_cells(nb)
    alls = "\n".join(cells)
    lines = [ln.strip() for ln in alls.splitlines()]
    run = _first(cells, "results = run_extraction(")

    # Full-dataset run, with a size-adjusted contract (THE bug guard).
    assert "SAMPLE_SIZE = None" in lines
    assert "n_boards = len(df) if SAMPLE_SIZE is None else SAMPLE_SIZE" in alls
    assert "CONTRACT = dataclasses.replace(CONTRACT_V1, sample_size=n_boards)" in alls
    assert "df_sample = sample_turns(df, n=CONTRACT.sample_size, seed=CONTRACT.random_seed)" in alls

    # run_extraction wired correctly.
    assert "contract=CONTRACT," in run
    assert "contract=CONTRACT_V1," not in run        # the exact regression we hit
    assert "df=df_sample," in run
    assert "base_dir=BASE_DIR," in run
    assert "resume=RESUME," in run
    assert "reuse_canonical=REUSE_CANONICAL," in run
    assert "checkpoint_dir=CHECKPOINT_DIR," in run

    # Agreed flag values.
    assert "RESUME = True" in lines
    assert "REUSE_CANONICAL = False" in lines

    # Outputs and checkpoints in separate, prefix-named folders.
    assert 'BASE_DIR = f"/content/drive/MyDrive/TCC/{meta[\'prefix\']}_outputs"' in alls
    assert 'CHECKPOINT_DIR = f"/content/drive/MyDrive/TCC/{meta[\'prefix\']}_checkpoints"' in alls


@pytest.mark.parametrize("nb", MODEL_NBS, ids=lambda p: os.path.basename(p))
def test_no_second_run_path(nb):
    """There is a single run path: the CLI 'Path A' cell (and its SAMPLE_ARG)
    has been removed, so the notebook can't accidentally run a different config."""
    alls = "\n".join(_code_cells(nb))
    assert "codenames-experiment run" not in alls, "stray CLI run cell present"
    assert "SAMPLE_ARG" not in alls, "stray SAMPLE_ARG (Path A remnant) present"
    assert "results = run_extraction(" in alls, "missing the run_extraction cell"


def _extract(df, contract, base_dir, monkeypatch):
    import codenames.loop as loop
    H.install_fakes(monkeypatch)
    return loop.run_extraction(
        model=None, tokenizer=None, df=df, base_dir=base_dir, prefix="fake",
        contract=contract, chat_template_strategy="raw",
        forward_hidden_states_mode="encoder_load_time", use_truncation=True,
        num_layers=H.FAKE_NUM_LAYERS, hidden_dim=H.FAKE_HIDDEN_DIM, device="cpu",
        has_generation=False, generation_fn=None, acceleration=ACCEL_REFERENCE,
    )


def test_contract_override_processes_full_dataset(tmp_path, monkeypatch):
    """End-to-end reproduction of the trap: with a baseline contract whose
    sample_size is smaller than the dataset, run_extraction caps the run; the
    notebook's size-adjusted contract processes every board."""
    N = 12
    baseline = H.make_contract(4)            # like CONTRACT_V1's 2000 < full 7704
    df = H.make_fake_dataset(N)

    # Notebook-correct: override sample_size to the full N.
    full_contract = dataclasses.replace(baseline, sample_size=N)
    df_sample = sample_turns(df, n=full_contract.sample_size, seed=full_contract.random_seed)
    res_full = _extract(df_sample, full_contract, str(tmp_path / "full"), monkeypatch)
    assert res_full["no_social"]["general_df"]["row_id"].nunique() == N

    # Buggy: pass the small baseline contract -> only its sample_size processed.
    res_bug = _extract(df_sample, baseline, str(tmp_path / "bug"), monkeypatch)
    assert res_bug["no_social"]["general_df"]["row_id"].nunique() == 4
