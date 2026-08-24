"""Offline item-bootstrap power planning for the fixed-controller Q1 test.

This module operates only on binary error matrices from already-consumed
DEVELOPMENT evaluations.  It has no benchmark, prompt, reference-answer,
model-backend, or holdout dependency.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def c_sufficient_features(
    errors: Mapping[str, np.ndarray],
    *,
    baseline: str,
    conditions: Sequence[str],
) -> np.ndarray:
    """Return per-item sufficient features for exact two-rollout C estimates."""

    base = np.asarray(errors[baseline], dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 2 or len(base) < 2:
        raise ValueError("baseline errors must have shape (n_items, 2), n_items >= 2")
    if np.any((base < 0) | (base > 1)):
        raise ValueError("error matrices must be binary")
    q0 = base.mean(axis=1)
    columns = [q0, np.square(q0)]
    for name in conditions:
        current = np.asarray(errors[name], dtype=np.float64)
        if current.shape != base.shape or np.any((current < 0) | (current > 1)):
            raise ValueError(f"condition {name!r} must be a binary matrix matching baseline")
        qj = current.mean(axis=1)
        b00_minus_b0j = base[:, 0] * base[:, 1] - (
            (base[:, 0] + base[:, 1]) * (current[:, 0] + current[:, 1]) / 4.0
        )
        columns.extend((b00_minus_b0j, qj, q0 * qj))
    return np.stack(columns, axis=1)


def c_from_feature_sums(feature_sums: np.ndarray, n_items: int) -> np.ndarray:
    """Compute exact U-statistic C values from summed sufficient features."""

    values = np.asarray(feature_sums, dtype=np.float64)
    if n_items < 2 or values.shape[-1] < 5 or (values.shape[-1] - 2) % 3:
        raise ValueError("invalid C sufficient-feature array")
    q0 = values[..., 0]
    q0_squared = values[..., 1]
    denominator = n_items * (n_items - 1)
    estimates = []
    for offset in range(2, values.shape[-1], 3):
        g_sum = values[..., offset]
        qj = values[..., offset + 1]
        q0qj = values[..., offset + 2]
        estimates.append(
            g_sum / n_items
            - (q0 * q0 - q0_squared - q0 * qj + q0qj) / denominator
        )
    return np.stack(estimates, axis=-1)


def point_c(features: np.ndarray) -> np.ndarray:
    """Compute C for every encoded condition over all feature rows."""

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("features must be two-dimensional")
    return c_from_feature_sums(matrix.sum(axis=0), len(matrix))


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(vector)),
        "median": float(np.median(vector)),
        "standard_deviation": float(np.std(vector, ddof=1)),
        "q025": float(np.quantile(vector, 0.025)),
        "q05": float(np.quantile(vector, 0.05)),
        "q95": float(np.quantile(vector, 0.95)),
        "q975": float(np.quantile(vector, 0.975)),
        "minimum": float(np.min(vector)),
        "maximum": float(np.max(vector)),
    }


def nested_item_bootstrap_power(
    features: np.ndarray,
    *,
    n_items: int,
    outer_replications: int,
    inner_resamples: int,
    seed: int,
    batch_size: int = 32,
) -> dict[str, object]:
    """Estimate percentile-bootstrap power using nested item resampling.

    Column zero is the meaningful controller; remaining columns are the null
    bank.  Every outer pseudoexperiment draws ``n_items`` DEVELOPMENT items.
    Each pseudoexperiment then receives its own two-sided 95% item-percentile
    bootstrap interval.  This directly estimates the probability that the
    future pre-specified interval has a positive lower endpoint.
    """

    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("features must contain at least two DEVELOPMENT items")
    n_conditions = (matrix.shape[1] - 2) // 3
    if n_conditions < 2:
        raise ValueError("features must encode one meaningful and at least one null condition")
    if n_items < 2 or outer_replications < 1 or inner_resamples < 39:
        raise ValueError("invalid nested-bootstrap dimensions")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    seed_sequence = np.random.SeedSequence(seed)
    outer_sequence, inner_sequence = seed_sequence.spawn(2)
    outer_rng = np.random.default_rng(outer_sequence)
    inner_rng = np.random.default_rng(inner_sequence)
    probabilities = np.full(n_items, 1.0 / n_items, dtype=np.float64)

    meaningful_points = np.empty(outer_replications, dtype=np.float64)
    specificity_points = np.empty(outer_replications, dtype=np.float64)
    meaningful_lower = np.empty(outer_replications, dtype=np.float64)
    meaningful_upper = np.empty(outer_replications, dtype=np.float64)
    specificity_lower = np.empty(outer_replications, dtype=np.float64)
    specificity_upper = np.empty(outer_replications, dtype=np.float64)

    for start in range(0, outer_replications, batch_size):
        stop = min(start + batch_size, outer_replications)
        size = stop - start
        outer_indices = outer_rng.integers(0, len(matrix), size=(size, n_items))
        outer_features = matrix[outer_indices]
        outer_c = c_from_feature_sums(outer_features.sum(axis=1), n_items)
        meaningful_points[start:stop] = outer_c[:, 0]
        specificity_points[start:stop] = outer_c[:, 0] - outer_c[:, 1:].mean(axis=1)

        # Multinomial weights are exactly equivalent to resampling n_items rows
        # with replacement and are substantially more memory-efficient here.
        counts = inner_rng.multinomial(
            n_items,
            probabilities,
            size=(size, inner_resamples),
        )
        feature_sums = np.einsum(
            "bij,bjk->bik", counts, outer_features, optimize=True
        )
        bootstrap_c = c_from_feature_sums(feature_sums, n_items)
        bootstrap_meaningful = bootstrap_c[..., 0]
        bootstrap_specificity = bootstrap_c[..., 0] - bootstrap_c[..., 1:].mean(axis=-1)
        meaningful_lower[start:stop] = np.quantile(
            bootstrap_meaningful, 0.025, axis=1
        )
        meaningful_upper[start:stop] = np.quantile(
            bootstrap_meaningful, 0.975, axis=1
        )
        specificity_lower[start:stop] = np.quantile(
            bootstrap_specificity, 0.025, axis=1
        )
        specificity_upper[start:stop] = np.quantile(
            bootstrap_specificity, 0.975, axis=1
        )

    meaningful_power = float(np.mean(meaningful_lower > 0))
    specificity_power = float(np.mean(specificity_lower > 0))
    meaningful_width = meaningful_upper - meaningful_lower
    specificity_width = specificity_upper - specificity_lower
    return {
        "method": "nested_item_percentile_bootstrap_power",
        "n_items": n_items,
        "outer_replications": outer_replications,
        "inner_resamples_per_pseudoexperiment": inner_resamples,
        "confidence": 0.95,
        "seed": seed,
        "meaningful_c_power_lower_bound_gt_zero": meaningful_power,
        "meaningful_c_power_monte_carlo_se": float(
            np.sqrt(meaningful_power * (1.0 - meaningful_power) / outer_replications)
        ),
        "meaningful_c_expected_interval_width": float(np.mean(meaningful_width)),
        "meaningful_c_interval_width_distribution": _distribution_summary(meaningful_width),
        "meaningful_c_distribution": _distribution_summary(meaningful_points),
        "specificity_c_power_lower_bound_gt_zero": specificity_power,
        "specificity_c_power_monte_carlo_se": float(
            np.sqrt(specificity_power * (1.0 - specificity_power) / outer_replications)
        ),
        "specificity_c_expected_interval_width": float(np.mean(specificity_width)),
        "specificity_c_interval_width_distribution": _distribution_summary(specificity_width),
        "specificity_c_distribution": _distribution_summary(specificity_points),
    }


__all__ = [
    "c_from_feature_sums",
    "c_sufficient_features",
    "nested_item_bootstrap_power",
    "point_c",
]
