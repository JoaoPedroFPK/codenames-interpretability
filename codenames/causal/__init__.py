"""Causal localization and control (docs/specs/causal_spec.md v3).

Activation patching (necessity) and steering (sufficiency) over frozen
decoders. Pure modules (pairs, positions, metrics, analysis, figures) carry no
torch dependency so they run anywhere; the GPU modules are imported lazily by
their callers.
"""

from .pairs import build_donor_index, build_pair_table, donors_for_turn

__all__ = ["build_donor_index", "build_pair_table", "donors_for_turn"]
