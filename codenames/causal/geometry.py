"""Geometric interventions on candidate states (causal_spec.md §5C, task T3d).

Every other causal stage in this package intervenes on *states*: patching swaps
a whole residual, steering adds a direction to it. Neither touches the quantity
the study is actually about, which is the **angle** between the hint's state and
each candidate's. This module edits that angle and nothing else.

The primary operation equalises the hint-to-candidate cosines: every candidate
span at layer l is rotated until it makes the same angle with the hint as every
other, so the target's geometric advantage is gone while the hint itself is
untouched. The reading it tests is the paper's own: if hump-1 proximity is a
decision, removing it should cost the answer; if hump 1 is pre-decision, the
answer should survive.

Three properties are load-bearing and are enforced here rather than hoped for:

* **Norms are preserved exactly.** A rotation that also changed magnitudes
  would confound the angle with the state's scale, and the residual norm is
  what the layer norm downstream is sensitive to.
* **Only the plane spanned by the candidate and the hint moves.** Everything
  orthogonal to both is carried through untouched, so whatever else the state
  encodes is left alone.
* **The control moves the states exactly as far.** ``displacement_matched_
  rotation`` reproduces the primary arm's per-candidate displacement norm along
  a random direction orthogonal to the hint, which leaves every hint-relative
  cosine unchanged. If the answer degrades under that arm too, the effect is
  the size of the perturbation and not the geometry, and the primary result is
  void. A random direction that was *not* orthogonalised would not bound this,
  because in high dimensions it would perturb the cosines a little as well.

Names are candidate words; ``hint`` is the mean-pooled hint span. All functions
take and return plain arrays and are pure, so they are testable without a model.
"""

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

# Below this a vector has no usable direction and is returned unchanged.
_TINY = 1e-12


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v if n < _TINY else v / n


def cosine_to(vectors: Mapping[str, np.ndarray], hint: np.ndarray) -> Dict[str, float]:
    """Hint-to-candidate cosine per candidate, the quantity g(l) ranks on."""
    u = _unit(hint)
    out: Dict[str, float] = {}
    for name, v in vectors.items():
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        out[name] = 0.0 if n < _TINY else float(np.dot(v, u) / n)
    return out


def rotate_to_cosine(v: np.ndarray, hint: np.ndarray, target_cos: float) -> np.ndarray:
    """``v`` rotated so ``cos(v, hint) == target_cos``, with ``|v|`` preserved.

    Decompose ``v = a*u + w`` with ``u`` the hint's unit vector and ``w`` the
    orthogonal remainder. Setting ``a' = target_cos * |v|`` fixes the angle, and
    rescaling ``w`` to ``sqrt(|v|^2 - a'^2)`` restores the norm. The rotation
    therefore happens entirely in the plane of ``v`` and ``hint``.
    """
    v = np.asarray(v, dtype=np.float64)
    u = _unit(hint)
    n = float(np.linalg.norm(v))
    if n < _TINY:
        return v.copy()

    target = float(np.clip(target_cos, -1.0, 1.0))
    a_new = target * n
    w = v - float(np.dot(v, u)) * u
    w_norm = float(np.linalg.norm(w))
    if w_norm < _TINY:
        # v is parallel to the hint: no orthogonal direction to rotate through.
        return v.copy()

    remainder = n * n - a_new * a_new
    scale = 0.0 if remainder <= 0 else float(np.sqrt(remainder)) / w_norm
    return a_new * u + scale * w


def equalise_cosines(
    vectors: Mapping[str, np.ndarray],
    hint: np.ndarray,
    *,
    alpha: float = 1.0,
    hold: Iterable[str] = (),
) -> Dict[str, np.ndarray]:
    """Rotate every candidate to the pool's mean hint-cosine (the primary arm).

    ``alpha`` interpolates between the observed cosine (0, the identity) and the
    equalised one (1), which is the dose axis of the sweep. ``hold`` names
    candidates that keep their own cosine and are excluded from the mean; with
    ``hold=(target,)`` this is the specificity control, in which every candidate
    except the target is flattened and the target keeps its advantage.
    """
    held = set(hold)
    observed = cosine_to(vectors, hint)
    moving = [name for name in vectors if name not in held]
    if not moving:
        return {name: np.asarray(v, dtype=np.float64).copy() for name, v in vectors.items()}

    mean = float(np.mean([observed[name] for name in moving]))
    out: Dict[str, np.ndarray] = {}
    for name, v in vectors.items():
        v = np.asarray(v, dtype=np.float64)
        if name in held:
            out[name] = v.copy()
            continue
        wanted = observed[name] + float(alpha) * (mean - observed[name])
        out[name] = rotate_to_cosine(v, hint, wanted)
    return out


def swap_cosines(
    vectors: Mapping[str, np.ndarray],
    hint: np.ndarray,
    first: str,
    second: str,
    *,
    alpha: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Give ``first`` the cosine ``second`` had and vice versa; others untouched.

    The directional control. Flattening the pool predicts only that the answer
    degrades; exchanging the target's and the donor's angles predicts *which*
    word the model should move to, which is a far harder prediction for the
    reading to survive.
    """
    observed = cosine_to(vectors, hint)
    if first not in observed or second not in observed:
        raise KeyError(f"{first!r} and {second!r} must both be candidates")

    wanted = {
        first: observed[first] + float(alpha) * (observed[second] - observed[first]),
        second: observed[second] + float(alpha) * (observed[first] - observed[second]),
    }
    out: Dict[str, np.ndarray] = {}
    for name, v in vectors.items():
        v = np.asarray(v, dtype=np.float64)
        out[name] = rotate_to_cosine(v, hint, wanted[name]) if name in wanted else v.copy()
    return out


def _orthogonal_direction(
    u: np.ndarray, v_perp_unit: np.ndarray, dim: int, rng: np.random.Generator,
) -> np.ndarray:
    """A unit vector orthogonal to both ``u`` and ``v_perp_unit``.

    Drawn at random, but never *silently* absent: a draw that happens to lie in
    the plane it is orthogonalised against would leave a zero vector, and
    returning the state unrotated at that point would turn the control arm into
    a no-op that still reports as a control. So the draw is retried, and if the
    retries also degenerate the direction is taken deterministically from the
    standard basis, which is guaranteed to work for ``dim > 2``.
    """
    floor = 1e-8 * np.sqrt(dim)
    for _ in range(8):
        r = rng.normal(size=dim)
        r = r - float(np.dot(r, u)) * u
        r = r - float(np.dot(r, v_perp_unit)) * v_perp_unit
        norm = float(np.linalg.norm(r))
        if norm > floor:
            return r / norm
    for i in range(dim):
        r = np.zeros(dim)
        r[i] = 1.0
        r = r - float(np.dot(r, u)) * u
        r = r - float(np.dot(r, v_perp_unit)) * v_perp_unit
        norm = float(np.linalg.norm(r))
        if norm > floor:
            return r / norm
    raise ValueError("no direction orthogonal to the hint and the candidate")


def displacement_matched_rotation(
    vectors: Mapping[str, np.ndarray],
    hint: np.ndarray,
    *,
    reference: Mapping[str, np.ndarray],
    seed: int = 2026,
    order: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Move each candidate as far as ``reference`` did, without changing its angle.

    For each candidate the displacement norm of the reference arm is reproduced
    by rotating within the plane of the candidate and a random direction that is
    orthogonal to the hint, so ``|v' - v|`` matches and ``cos(v', hint)`` does
    not move. This is the arm that separates "the geometry mattered" from "a
    perturbation of that size mattered".
    """
    rng = np.random.default_rng(seed)
    u = _unit(hint)
    names = list(order) if order is not None else list(vectors)

    out: Dict[str, np.ndarray] = {}
    for name in names:
        v = np.asarray(vectors[name], dtype=np.float64)
        n = float(np.linalg.norm(v))
        target = np.asarray(reference[name], dtype=np.float64)
        distance = float(np.linalg.norm(target - v))
        if n < _TINY or distance < _TINY:
            out[name] = v.copy()
            continue

        # A direction orthogonal to both the hint and this candidate: rotating
        # into it keeps cos(v, hint) fixed because the component along u never
        # changes and the norm is preserved.
        v_perp = v - float(np.dot(v, u)) * u
        p = float(np.linalg.norm(v_perp))
        if p < _TINY:
            out[name] = v.copy()
            continue
        r = _orthogonal_direction(u, v_perp / p, v.shape[0], rng)

        # Rotate the hint-orthogonal part of v toward r by the angle that puts
        # the whole vector `distance` away from where it started.
        cos_theta = float(np.clip(1.0 - distance * distance / (2.0 * p * p), -1.0, 1.0))
        sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
        rotated_perp = cos_theta * v_perp + sin_theta * p * r
        out[name] = float(np.dot(v, u)) * u + rotated_perp

    for name, v in vectors.items():
        out.setdefault(name, np.asarray(v, dtype=np.float64).copy())
    return out
