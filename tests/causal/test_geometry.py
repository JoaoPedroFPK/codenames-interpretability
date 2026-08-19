"""Geometric interventions on the candidate states (causal_spec.md §5C, T3d).

The patching tier swaps whole states; this tier changes one property of them —
the angle each candidate makes with the hint — and leaves norms and everything
orthogonal to the hint alone. These tests pin the properties the claim rests
on: the norm is preserved, the requested cosine is achieved, alpha interpolates,
and the displacement-matched control moves the states just as far without
touching the hint-relative geometry at all.
"""
import numpy as np
import pytest

from codenames.causal.geometry import (
    displacement_matched_rotation,
    equalise_cosines,
    rotate_to_cosine,
    swap_cosines,
)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture
def pool():
    rng = np.random.default_rng(2026)
    hint = rng.normal(size=64)
    vectors = {name: rng.normal(size=64) * scale
               for name, scale in zip("abcde", [1.0, 2.0, 0.5, 3.0, 1.5])}
    return hint, vectors


# --- the primitive ---------------------------------------------------------

def test_rotate_to_cosine_hits_the_target_and_keeps_the_norm(pool):
    hint, vectors = pool
    v = vectors["a"]
    out = rotate_to_cosine(v, hint, 0.3)
    assert _cos(out, hint) == pytest.approx(0.3, abs=1e-10)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(v), rel=1e-12)


def test_rotate_to_cosine_only_moves_within_the_plane_of_v_and_the_hint(pool):
    """Anything orthogonal to both is untouched, so the edit is the angle and
    nothing else."""
    hint, vectors = pool
    v = vectors["b"]
    out = rotate_to_cosine(v, hint, -0.2)
    u = hint / np.linalg.norm(hint)
    w = v - np.dot(v, u) * u
    basis = np.stack([u, w / np.linalg.norm(w)])
    residual = out - basis.T @ (basis @ out)
    assert np.linalg.norm(residual) == pytest.approx(0.0, abs=1e-10)


def test_rotate_to_cosine_clamps_an_out_of_range_request(pool):
    hint, vectors = pool
    out = rotate_to_cosine(vectors["c"], hint, 1.7)
    assert _cos(out, hint) == pytest.approx(1.0, abs=1e-8)


# --- the primary arm -------------------------------------------------------

def test_equalise_sets_every_candidate_to_the_pool_mean_cosine(pool):
    hint, vectors = pool
    before = {k: _cos(v, hint) for k, v in vectors.items()}
    out = equalise_cosines(vectors, hint)
    mean = float(np.mean(list(before.values())))
    for name, v in out.items():
        assert _cos(v, hint) == pytest.approx(mean, abs=1e-10), name
        assert np.linalg.norm(v) == pytest.approx(np.linalg.norm(vectors[name]), rel=1e-12)


def test_equalise_at_alpha_zero_is_the_identity(pool):
    hint, vectors = pool
    out = equalise_cosines(vectors, hint, alpha=0.0)
    for name, v in out.items():
        np.testing.assert_allclose(v, vectors[name], atol=1e-10)


def test_equalise_alpha_interpolates_the_cosine(pool):
    hint, vectors = pool
    before = {k: _cos(v, hint) for k, v in vectors.items()}
    mean = float(np.mean(list(before.values())))
    out = equalise_cosines(vectors, hint, alpha=0.5)
    for name, v in out.items():
        expected = before[name] + 0.5 * (mean - before[name])
        assert _cos(v, hint) == pytest.approx(expected, abs=1e-10), name


def test_equalise_removes_the_targets_rank_advantage(pool):
    """The point of the intervention: after it, no candidate is nearest."""
    hint, vectors = pool
    out = equalise_cosines(vectors, hint)
    cosines = [_cos(v, hint) for v in out.values()]
    assert max(cosines) - min(cosines) < 1e-9


# --- the specificity control ----------------------------------------------

def test_holding_a_candidate_leaves_its_cosine_and_moves_the_rest(pool):
    hint, vectors = pool
    before = {k: _cos(v, hint) for k, v in vectors.items()}
    out = equalise_cosines(vectors, hint, hold=("a",))
    assert _cos(out["a"], hint) == pytest.approx(before["a"], abs=1e-12)
    others = [_cos(out[k], hint) for k in vectors if k != "a"]
    assert max(others) - min(others) < 1e-9
    assert others[0] == pytest.approx(
        float(np.mean([before[k] for k in vectors if k != "a"])), abs=1e-10)


def test_swap_exchanges_two_candidates_cosines(pool):
    hint, vectors = pool
    before = {k: _cos(v, hint) for k, v in vectors.items()}
    out = swap_cosines(vectors, hint, "a", "b")
    assert _cos(out["a"], hint) == pytest.approx(before["b"], abs=1e-10)
    assert _cos(out["b"], hint) == pytest.approx(before["a"], abs=1e-10)
    for name in "cde":
        np.testing.assert_allclose(out[name], vectors[name], atol=1e-12)


# --- the magnitude control -------------------------------------------------

def test_displacement_matched_rotation_moves_as_far_but_not_toward_the_hint(pool):
    """The control the claim needs: if the answer degrades here too, the effect
    is perturbation size, not the hint-relative angle."""
    hint, vectors = pool
    primary = equalise_cosines(vectors, hint)
    control = displacement_matched_rotation(vectors, hint, reference=primary, seed=2026)
    for name, v in vectors.items():
        moved_primary = np.linalg.norm(primary[name] - v)
        moved_control = np.linalg.norm(control[name] - v)
        assert moved_control == pytest.approx(moved_primary, rel=1e-6), name
        assert np.linalg.norm(control[name]) == pytest.approx(np.linalg.norm(v), rel=1e-10)
        assert _cos(control[name], hint) == pytest.approx(_cos(v, hint), abs=1e-10), name


def test_displacement_matched_rotation_is_seeded(pool):
    hint, vectors = pool
    primary = equalise_cosines(vectors, hint)
    a = displacement_matched_rotation(vectors, hint, reference=primary, seed=2026)
    b = displacement_matched_rotation(vectors, hint, reference=primary, seed=2026)
    c = displacement_matched_rotation(vectors, hint, reference=primary, seed=7)
    for name in vectors:
        np.testing.assert_allclose(a[name], b[name])
        assert not np.allclose(a[name], c[name])


def test_control_still_rotates_when_the_first_draw_lands_in_the_plane():
    """Regression: seeding the control's generator identically to the hint made
    its first draw *be* the hint, the orthogonalised remainder was zero, and the
    state was returned unmoved — a control arm that silently did nothing while
    still reporting as a control."""
    rng = np.random.default_rng(2026)
    hint = rng.normal(size=64)          # the generator's first draw at seed 2026
    # two candidates, so equalisation actually moves them
    vectors = {"a": rng.normal(size=64), "b": rng.normal(size=64) * 2.0}
    primary = equalise_cosines(vectors, hint)
    control = displacement_matched_rotation(vectors, hint, reference=primary, seed=2026)
    moved = np.linalg.norm(control["a"] - vectors["a"])
    assert moved == pytest.approx(np.linalg.norm(primary["a"] - vectors["a"]), rel=1e-6)
    assert moved > 0
