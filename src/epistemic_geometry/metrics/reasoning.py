"""Repeated-rollout error-propensity and sampling-floor metrics."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np

from .errors import error_jaccard, phi_correlation


class SeedRegime(StrEnum):
    """Prospective rollout coupling semantics."""

    INDEPENDENT_PRIMARY = "INDEPENDENT_PRIMARY"
    MATCHED_COUPLING_SECONDARY = "MATCHED_COUPLING_SECONDARY"


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
    """Compatibility name for steered oracle gain over a repeated baseline.

    This quantity mixes a mean-competence term and a covariance term. Prefer
    :func:`stochastic_complementarity_estimands`, which reports both pieces.
    """

    baseline = np.asarray(p0, dtype=float)
    steered = np.asarray(pj, dtype=float)
    if baseline.shape != steered.shape:
        raise ValueError("baseline and intervention propensities must have equal shape")
    return float(np.mean(baseline**2 - baseline * steered))


def stochastic_complementarity_estimands(
    p0: np.ndarray,
    pj: np.ndarray,
) -> dict[str, float]:
    """Return population-style stochastic complementarity estimands over items.

    ``p0`` and ``pj`` are per-item error propensities under independently drawn
    baseline and intervention rollouts. The competence-adjusted term removes
    the oracle gain explained solely by a shift in mean error rate.
    """

    baseline = np.asarray(p0, dtype=float).reshape(-1)
    intervention = np.asarray(pj, dtype=float).reshape(-1)
    if baseline.shape != intervention.shape or baseline.size == 0:
        raise ValueError("propensity vectors must be non-empty and have equal shape")
    if (
        np.any(~np.isfinite(baseline))
        or np.any(~np.isfinite(intervention))
        or np.any((baseline < 0) | (baseline > 1))
        or np.any((intervention < 0) | (intervention > 1))
    ):
        raise ValueError("error propensities must be finite values in [0, 1]")
    mu0 = float(baseline.mean())
    muj = float(intervention.mean())
    variance0 = float(np.mean((baseline - mu0) ** 2))
    covariance = float(np.mean((baseline - mu0) * (intervention - muj)))
    oracle_0j = float(1.0 - np.mean(baseline * intervention))
    oracle_00 = float(1.0 - np.mean(baseline**2))
    gain = oracle_0j - oracle_00
    competence_component = mu0 * (mu0 - muj)
    adjusted = variance0 - covariance
    return {
        "mu_0": mu0,
        "mu_j": muj,
        "oracle_0j": oracle_0j,
        "oracle_00": oracle_00,
        "oracle_gain_g_j": gain,
        "competence_component": competence_component,
        "baseline_propensity_variance": variance0,
        "cross_propensity_covariance": covariance,
        "competence_adjusted_complementarity_c_j": adjusted,
        "decomposition_residual": gain - competence_component - adjusted,
    }


def unbiased_two_rollout_propensity_distance(
    errors_i: np.ndarray | list[list[bool]],
    errors_j: np.ndarray | list[list[bool]],
    *,
    seed_regime: SeedRegime | str,
    item_ids_i: Sequence[str],
    item_ids_j: Sequence[str],
) -> float:
    """Estimate ``E_t[(p_ti-p_tj)^2]`` from two independent rollouts each.

    Cross-products are unbiased only when the two rollout columns and the two
    condition banks are independent draws. Matched common-random-number
    coupling is useful as a secondary causal view but invalid for this
    estimator.
    """

    regime = SeedRegime(seed_regime)
    if regime is not SeedRegime.INDEPENDENT_PRIMARY:
        raise ValueError("unbiased propensity distance requires INDEPENDENT_PRIMARY seeds")
    left = _matrix(errors_i)
    right = _matrix(errors_j)
    if left.shape != right.shape or left.shape[1] != 2:
        raise ValueError("each condition must have shape [n_items, 2] with matched items")
    ids_i = tuple(str(value) for value in item_ids_i)
    ids_j = tuple(str(value) for value in item_ids_j)
    if len(ids_i) != left.shape[0] or len(ids_j) != right.shape[0]:
        raise ValueError("item provenance length must match rollout matrix rows")
    if len(set(ids_i)) != len(ids_i) or len(set(ids_j)) != len(ids_j):
        raise ValueError("item provenance IDs must be unique within each condition")
    if ids_i != ids_j:
        raise ValueError("item provenance IDs and row order must match exactly")
    i1, i2 = left[:, 0].astype(float), left[:, 1].astype(float)
    j1, j2 = right[:, 0].astype(float), right[:, 1].astype(float)
    return float(np.mean(i1 * i2 + j1 * j2 - i1 * j2 - i2 * j1))


def propensity_correlation_from_rollouts(
    errors_i: np.ndarray | list[list[bool]],
    errors_j: np.ndarray | list[list[bool]],
    *,
    allow_two_rollout_low_resolution: bool = False,
) -> dict[str, float | int | str]:
    """Report plug-in propensity correlation with an explicit resolution status."""

    left = _matrix(errors_i)
    right = _matrix(errors_j)
    if left.shape != right.shape:
        raise ValueError("rollout matrices must have equal shape")
    n_rollouts = left.shape[1]
    if n_rollouts < 4 and not (n_rollouts == 2 and allow_two_rollout_low_resolution):
        raise ValueError(
            "propensity correlation requires >=4 rollouts, or an explicit two-rollout "
            "low-resolution opt-in"
        )
    correlation = propensity_correlation(error_propensity(left), error_propensity(right))
    return {
        "correlation": correlation,
        "n_rollouts_per_condition": n_rollouts,
        "status": (
            "LOW_RESOLUTION_TWO_ROLLOUT_PLUGIN_ATTENUATED"
            if n_rollouts == 2
            else "DESCRIPTIVE_PLUGIN_ESTIMATE"
        ),
    }


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


def split_half_reliability(
    errors: np.ndarray | list[list[bool]],
    *,
    allow_two_rollout_low_resolution: bool = False,
) -> dict[str, float | str]:
    matrix = _matrix(errors)
    if matrix.shape[1] < 2 or matrix.shape[1] % 2:
        raise ValueError("split-half reliability needs an even number of rollouts >= 2")
    if matrix.shape[1] == 2 and not allow_two_rollout_low_resolution:
        raise ValueError(
            "two one-rollout halves are binary agreement diagnostics, not smooth "
            "propensity reliability; opt in explicitly to report them"
        )
    midpoint = matrix.shape[1] // 2
    first = error_propensity(matrix[:, :midpoint])
    second = error_propensity(matrix[:, midpoint:])
    return {
        "correlation": propensity_correlation(first, second),
        "mean_absolute_difference": float(np.mean(np.abs(first - second))),
        "squared_difference": float(np.mean((first - second) ** 2)),
        "status": (
            "LOW_RESOLUTION_ONE_ROLLOUT_PER_HALF"
            if matrix.shape[1] == 2
            else "DESCRIPTIVE_SPLIT_HALF"
        ),
    }


def hard_rollout_pair_metrics(errors_i: list[bool], errors_j: list[bool]) -> dict[str, float]:
    if len(errors_i) != len(errors_j) or not errors_i:
        raise ValueError("hard rollout error vectors must be non-empty and equal length")
    return {
        "phi": phi_correlation(errors_i, errors_j),
        "jaccard": error_jaccard(errors_i, errors_j),
        "double_fault": float(np.logical_and(errors_i, errors_j).mean()),
    }
