"""CPU-only contracts for the Gate 6.3 single-layer continuation.

Gate 6.3 does not search for a new meaningful controller.  This module only
constructs the architecture-matched L27 random null and exposes the exact
standardized delta calculation used by the remote runner.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np


def unit_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("vector must have finite non-zero norm")
    return vector / norm


def vector_sha256(values: np.ndarray) -> str:
    """Hash a canonical float64 unit vector."""

    return hashlib.sha256(np.asarray(values, dtype=np.float64).reshape(-1).tobytes()).hexdigest()


def single_layer_random_bank(
    meaningful: np.ndarray,
    *,
    seeds: Sequence[int],
) -> dict[str, np.ndarray]:
    """Construct deterministic mutually orthogonal single-layer random vectors.

    Gram-Schmidt is performed in the order ``meaningful, R0, R1, ...``.  The
    resulting vectors are independent of model outputs and are valid only for
    the same hidden dimension and intervention layer as ``meaningful``.
    """

    basis = [unit_vector(meaningful)]
    bank: dict[str, np.ndarray] = {}
    for index, seed in enumerate(seeds):
        candidate = np.random.default_rng(int(seed)).standard_normal(len(basis[0]))
        for previous in basis:
            candidate = candidate - float(np.dot(candidate, previous)) * previous
        vector = unit_vector(candidate)
        bank[f"R{index}"] = vector
        basis.append(vector)
    return bank


def standardized_delta(
    direction: np.ndarray,
    *,
    eta: float,
    reference_scale: float,
) -> np.ndarray:
    """Apply one frozen standardized energy to a unit direction."""

    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("eta must be finite and positive")
    if not np.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("reference_scale must be finite and positive")
    return unit_vector(direction) * (float(eta) * float(reference_scale))


def bank_geometry(
    meaningful: np.ndarray,
    bank: dict[str, np.ndarray],
) -> dict[str, object]:
    """Return auditable norm and cosine checks for a frozen random bank."""

    meaningful_unit = unit_vector(meaningful)
    names = tuple(sorted(bank))
    norms = {name: float(np.linalg.norm(bank[name])) for name in names}
    meaningful_cosines = {
        name: float(abs(np.dot(meaningful_unit, unit_vector(bank[name])))) for name in names
    }
    pairwise = {
        f"{left}__{right}": float(abs(np.dot(unit_vector(bank[left]), unit_vector(bank[right]))))
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return {
        "norms": norms,
        "meaningful_absolute_cosines": meaningful_cosines,
        "pairwise_absolute_cosines": pairwise,
        "unit_norm_pass": all(abs(value - 1.0) <= 1e-12 for value in norms.values()),
        "meaningful_orthogonality_pass": all(
            value <= 1e-6 for value in meaningful_cosines.values()
        ),
        "random_pairwise_orthogonality_pass": all(value <= 1e-6 for value in pairwise.values()),
    }


__all__ = [
    "bank_geometry",
    "single_layer_random_bank",
    "standardized_delta",
    "unit_vector",
    "vector_sha256",
]
