"""Notebooks must stay thin orchestration shells.

Methodology lives in the package so the CLI, the notebooks and every model
inherit it identically (CLAUDE.md §7). These tests guard the config contract
each notebook exposes, not its outputs.
"""

def _causal_source():
    import json
    nb = json.load(open("notebooks/09_causal.ipynb"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_causal_notebook_is_a_thin_shell():
    src = _causal_source()
    assert "PILOT" in src and "MODEL_KEY" in src
    assert "SEED = 2026" in src
    # Methodology must live in the package, not the notebook.
    assert "def run_patch" not in src
    assert "register_forward_hook" not in src
    assert "def " not in src.replace("def _", "")


def test_causal_notebook_defaults_to_the_pilot_gate():
    src = _causal_source()
    assert "PILOT = True" in src, "the gate must be the default; full runs are opt-in"


def test_causal_notebook_pins_the_branch():
    src = _causal_source()
    assert 'REPO_BRANCH = "probing"' in src


def test_causal_notebook_pilot_uses_the_secondary_condition():
    """The pilot must not consume the confirmatory no_social sample (§12.5)."""
    src = _causal_source()
    assert '"with_social" if PILOT else "no_social"' in src
    assert "150 if PILOT else 1500" in src
