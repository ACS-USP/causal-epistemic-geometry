"""Repeated-rollout error-propensity and sampling-floor metrics."""

from __future__ import annotations

import numpy as np

from .errors import error_jaccard, phi_correlation


def _matrix(values: np.ndarray | list[list[bool]]) -> np.ndarray:
    result = np.asarray(values, dtype=bool)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] == 0:
        raise ValueError("rollout error matrix must be non-empty and two-dimensional")
    return result


def error_propensity(errors: np.ndarray | list[list[bool]]) -> np.ndarray:
    """Estimate per-item error probability from repeated binary rollouts."""

    return _matrix(errors).mean(axis=1)


def propensity_correlation(p_i: np.ndarray, p_j: np.ndarray) -> float:
    left, right = np.asarray(p_i, dtype=float), np.asarray(p_j, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("propensity vectors must be non-empty and have equal shape")
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def propensity_covariance(p_i: np.ndarray, p_j: np.ndarray) -> float:
    left, right = np.asarray(p_i, dtype=float), np.asarray(p_j, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("propensity vectors must be non-empty and have equal shape")
    return float(np.mean((left - left.mean()) * (right - right.mean())))


def expected_double_fault(p_i: np.ndarray, p_j: np.ndarray) -> float:
    left, right = np.asarray(p_i, dtype=float), np.asarray(p_j, dtype=float)
    if left.shape != right.shape:
        raise ValueError("propensity vectors must have equal shape")
    return float(np.mean(left * right))


def expected_pair_oracle(p_i: np.ndarray, p_j: np.ndarray) -> float:
    return 1.0 - expected_double_fault(p_i, p_j)


def excess_pair_oracle(p0: np.ndarray, pj: np.ndarray) -> float:
    """Steered pair-oracle gain over an independent repeated-baseline pair."""

    baseline = np.asarray(p0, dtype=float)
    steered = np.asarray(pj, dtype=float)
    if baseline.shape != steered.shape:
        raise ValueError("baseline and intervention propensities must have equal shape")
    return float(np.mean(baseline**2 - baseline * steered))


def propensity_distances(p_i: np.ndarray, p_j: np.ndarray) -> dict[str, float]:
    left, right = np.asarray(p_i, dtype=float), np.asarray(p_j, dtype=float)
    if left.shape != right.shape:
        raise ValueError("propensity vectors must have equal shape")
    return {
        "mean_absolute_difference": float(np.mean(np.abs(left - right))),
        "squared_distance": float(np.mean((left - right) ** 2)),
        "correlation": propensity_correlation(left, right),
        "covariance": propensity_covariance(left, right),
        "expected_double_fault": expected_double_fault(left, right),
        "expected_pair_oracle": expected_pair_oracle(left, right),
    }


def split_half_reliability(errors: np.ndarray | list[list[bool]]) -> dict[str, float]:
    matrix = _matrix(errors)
    if matrix.shape[1] < 2 or matrix.shape[1] % 2:
        raise ValueError("split-half reliability needs an even number of rollouts >= 2")
    midpoint = matrix.shape[1] // 2
    first = error_propensity(matrix[:, :midpoint])
    second = error_propensity(matrix[:, midpoint:])
    return {
        "correlation": propensity_correlation(first, second),
        "mean_absolute_difference": float(np.mean(np.abs(first - second))),
        "squared_difference": float(np.mean((first - second) ** 2)),
    }


def hard_rollout_pair_metrics(errors_i: list[bool], errors_j: list[bool]) -> dict[str, float]:
    if len(errors_i) != len(errors_j) or not errors_i:
        raise ValueError("hard rollout error vectors must be non-empty and equal length")
    return {
        "phi": phi_correlation(errors_i, errors_j),
        "jaccard": error_jaccard(errors_i, errors_j),
        "double_fault": float(np.logical_and(errors_i, errors_j).mean()),
    }
