"""Structural guard on the runner notebook.

The notebook must stay a thin shim: no per-model configuration, no methodology,
and an explicit branch so a stale checkout cannot silently run older code.
"""

import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "00_runner.ipynb"


@pytest.fixture(scope="module")
def cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_notebook_exists():
    assert NOTEBOOK.exists()


def test_notebook_is_valid_json_with_cells(cells):
    assert len(cells) >= 4


def test_notebook_pins_an_explicit_branch(cells):
    assert any("REPO_BRANCH" in c for c in cells)


def test_notebook_mounts_drive(cells):
    assert any("drive.mount" in c for c in cells)


def test_notebook_invokes_the_runner_verb(cells):
    assert any("job-runner" in c for c in cells)


def test_notebook_passes_an_idle_shutdown_budget(cells):
    # Without this the poller holds an A100 and burns compute units while idle.
    assert any("--idle-shutdown-minutes" in c for c in cells)


def test_notebook_reports_the_gpu_it_actually_got(cells):
    # Colab Pro does not guarantee an A100; the session must say what it has.
    assert any("get_device_name" in c for c in cells)


def test_notebook_carries_no_per_model_configuration(cells):
    joined = "\n".join(cells)
    for forbidden in ("MODEL_KEY", "SAMPLE_SIZE", "CONDITIONS", "load_dataset"):
        assert forbidden not in joined, f"{forbidden} belongs in a job, not the runner"


def test_notebook_has_no_stored_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        assert cell.get("outputs", []) == []
