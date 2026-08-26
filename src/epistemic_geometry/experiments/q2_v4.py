"""Outcome-free design primitives for Q2 V4 intervention-subspace geometry.

This module is deliberately isolated from model runners, semantic parsers, and
historical outcome journals.  It contains only linear algebra, estimators, and
controller-label randomization machinery needed by the design sprint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SubspaceSummary:
    """Numerical summary of a unit-column source-direction matrix."""

    singular_values: tuple[float, ...]
    relative_singular_values: tuple[float, ...]
    exact_rank: int
    retained_rank: int
    condition_number: float
    entropy_effective_rank: float
    stable_rank: float


def orthonormal_source_subspace(
    source_directions: np.ndarray,
    *,
    relative_singular_threshold: float = 1e-6,
) -> tuple[np.ndarray, SubspaceSummary]:
    """Return a stable orthonormal basis for normalized source columns.

    Column normalization prevents historical construction norms from weighting
    the span.  Rank selection is numerical only and cannot use behavior.
    """

    values = np.asarray(source_directions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("source directions must be a nonempty column matrix")
    if not 0.0 < relative_singular_threshold < 1.0:
        raise ValueError("relative singular threshold must lie in (0, 1)")
    norms = np.linalg.norm(values, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("source directions must have finite positive norms")
    unit = values / norms[None, :]
    left, singular, _right = np.linalg.svd(unit, full_matrices=False)
    relative = singular / singular[0]
    retained = int(np.sum(relative >= relative_singular_threshold))
    exact_rank = int(np.linalg.matrix_rank(unit))
    if retained < 1:
        raise ValueError("source subspace is numerically empty")
    energy = np.square(singular)
    probabilities = energy / np.sum(energy)
    entropy_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    stable_rank = float(np.sum(energy) / energy[0])
    summary = SubspaceSummary(
        singular_values=tuple(float(value) for value in singular),
        relative_singular_values=tuple(float(value) for value in relative),
        exact_rank=exact_rank,
        retained_rank=retained,
        condition_number=float(singular[0] / singular[retained - 1]),
        entropy_effective_rank=entropy_rank,
        stable_rank=stable_rank,
    )
    return left[:, :retained], summary


def protocol_seed(namespace: str, source_commit: str) -> int:
    """Derive one stable 128-bit seed without seed shopping."""

    payload = f"{namespace}|{source_commit}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def sample_coefficient_bank(rank: int, size: int, *, seed: int) -> np.ndarray:
    """Sample one isotropic unit-sphere bank using NumPy PCG64DXSM."""

    if rank < 2 or size < rank:
        raise ValueError("bank requires size >= rank >= 2")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    coefficients = generator.standard_normal((size, rank))
    norms = np.linalg.norm(coefficients, axis=1)
    if np.any(norms <= 0.0):
        raise RuntimeError("Gaussian bank produced a zero vector")
    return coefficients / norms[:, None]


def coefficient_bank_checks(coefficients: np.ndarray) -> dict[str, float | int | bool]:
    """Apply symmetric gross-degeneracy checks in coefficient space."""

    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("coefficient bank must be a matrix")
    norms = np.linalg.norm(values, axis=1)
    gram = values @ values.T
    off_diagonal = gram[~np.eye(len(values), dtype=bool)]
    singular = np.linalg.svd(values, compute_uv=False)
    energy = np.square(singular)
    probabilities = energy / np.sum(energy)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    maximum_absolute_cosine = float(np.max(np.abs(off_diagonal)))
    condition = float(singular[0] / singular[-1])
    checks = {
        "finite": bool(np.all(np.isfinite(values))),
        "maximum_unit_norm_error": float(np.max(np.abs(norms - 1.0))),
        "matrix_rank": int(np.linalg.matrix_rank(values)),
        "effective_rank": effective_rank,
        "condition_number": condition,
        "maximum_absolute_pair_cosine": maximum_absolute_cosine,
    }
    checks["pass"] = bool(
        checks["finite"]
        and checks["maximum_unit_norm_error"] <= 1e-12
        and checks["matrix_rank"] == values.shape[1]
        and effective_rank >= 0.75 * values.shape[1]
        and condition <= 3.0
        and maximum_absolute_cosine <= 0.98
    )
    return checks


def blind_spot_shape_matrices(errors: np.ndarray) -> dict[str, np.ndarray]:
    """Estimate total and shape distances from two independent rollouts.

    ``errors`` has shape ``(controller, item, rollout)`` and exactly two
    rollouts.  ``shape_frozen_panel`` is conditionally unbiased for the
    variance under the uniform distribution on the fixed panel.  Multiplying by
    N/(N-1) gives ``shape_item_population``, unbiased when panel items are IID
    draws from a superpopulation and the target is its variance.
    """

    values = np.asarray(errors, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 2 or values.shape[1] < 2:
        raise ValueError("errors must have shape controller x item x 2")
    if np.any((values != 0.0) & (values != 1.0)):
        raise ValueError("error outcomes must be binary")
    controllers, items, _ = values.shape
    total = np.zeros((controllers, controllers), dtype=np.float64)
    mean_shift_squared = np.zeros_like(total)
    panel_shape = np.zeros_like(total)
    for left in range(controllers):
        for right in range(left + 1, controllers):
            d0 = values[left, :, 0] - values[right, :, 0]
            d1 = values[left, :, 1] - values[right, :, 1]
            total_value = float(np.mean(d0 * d1))
            mean_product = float(np.mean(d0) * np.mean(d1))
            shape_value = total_value - mean_product
            for matrix, value in (
                (total, total_value),
                (mean_shift_squared, mean_product),
                (panel_shape, shape_value),
            ):
                matrix[left, right] = matrix[right, left] = value
    population_shape = panel_shape * (items / (items - 1.0))
    return {
        "total": total,
        "mean_shift_squared": mean_shift_squared,
        "shape_frozen_panel": panel_shape,
        "shape_item_population": population_shape,
    }


def baseline_centered_js_angle(
    mean_js_with_baseline: np.ndarray,
    *,
    zero_radius_squared_tolerance: float,
) -> dict[str, np.ndarray]:
    """Recover Hilbert angles from squared JS distances to a baseline origin.

    The baseline occupies index zero.  No angle is invented for a controller
    whose squared radius is at or below the frozen numerical tolerance.
    """

    squared = np.asarray(mean_js_with_baseline, dtype=np.float64)
    if squared.ndim != 2 or squared.shape[0] != squared.shape[1] or len(squared) < 2:
        raise ValueError("mean-JS matrix must be square and include baseline")
    if np.any(~np.isfinite(squared)) or np.min(squared) < -1e-12:
        raise ValueError("mean-JS matrix must be finite and nonnegative")
    if zero_radius_squared_tolerance < 0.0:
        raise ValueError("zero-radius tolerance must be nonnegative")
    radii_squared = squared[0, 1:].copy()
    if np.any(radii_squared <= zero_radius_squared_tolerance):
        raise ValueError("M2 angle undefined for a near-zero baseline radius")
    radii = np.sqrt(radii_squared)
    pair_squared = squared[1:, 1:]
    cosine = (
        radii_squared[:, None] + radii_squared[None, :] - pair_squared
    ) / (2.0 * np.outer(radii, radii))
    if np.max(np.abs(cosine)) > 1.0 + 1e-8:
        raise ValueError("distance matrix violates the Hilbert cosine identity")
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    return {
        "radius": radii,
        "cosine": cosine,
        "angular_dissimilarity": 1.0 - cosine,
        "angular_chord": np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0)),
    }


def controller_permutations(size: int, count: int, *, seed: int) -> np.ndarray:
    """Return identity plus unique PCG64DXSM controller-label permutations."""

    if size < 3 or count < 2:
        raise ValueError("at least three controllers and two permutations required")
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    identity = tuple(range(size))
    seen = {identity}
    rows = [identity]
    while len(rows) < count:
        candidate = tuple(int(value) for value in generator.permutation(size))
        if candidate not in seen:
            seen.add(candidate)
            rows.append(candidate)
    return np.asarray(rows, dtype=np.int64)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks for a one-dimensional vector."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute Spearman correlation using deterministic average ranks."""

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def shell_coupled_qap_statistics(
    geometries: dict[str, dict[str, np.ndarray]],
    outcomes: dict[str, np.ndarray],
    permutations: np.ndarray,
) -> dict[str, np.ndarray | dict[str, float]]:
    """Compute shell-mean QAP statistics under one shared label permutation."""

    metric_names = tuple(sorted(geometries))
    shell_names = tuple(sorted(outcomes))
    if not metric_names or not shell_names:
        raise ValueError("QAP needs metrics and shells")
    size = next(iter(outcomes.values())).shape[0]
    upper = np.triu_indices(size, 1)
    null = np.zeros((len(permutations), len(metric_names)), dtype=np.float64)
    for permutation_index, permutation in enumerate(permutations):
        for metric_index, metric in enumerate(metric_names):
            shell_values = []
            for shell in shell_names:
                geometry = geometries[metric][shell]
                permuted = geometry[np.ix_(permutation, permutation)]
                shell_values.append(spearman(permuted[upper], outcomes[shell][upper]))
            null[permutation_index, metric_index] = float(np.mean(shell_values))
    observed = {
        metric: float(null[0, index]) for index, metric in enumerate(metric_names)
    }
    maximum = np.max(null, axis=1)
    observed_maximum = max(observed.values())
    denominator = float(len(permutations))
    return {
        "metric_order": np.asarray(metric_names),
        "observed": observed,
        "null": null,
        "global_p": float(np.sum(maximum >= observed_maximum) / denominator),
        "maxT_adjusted_p": {
            metric: float(np.sum(maximum >= observed[metric]) / denominator)
            for metric in metric_names
        },
    }


__all__ = [
    "SubspaceSummary",
    "average_ranks",
    "baseline_centered_js_angle",
    "blind_spot_shape_matrices",
    "coefficient_bank_checks",
    "controller_permutations",
    "orthonormal_source_subspace",
    "protocol_seed",
    "sample_coefficient_bank",
    "shell_coupled_qap_statistics",
    "spearman",
]
