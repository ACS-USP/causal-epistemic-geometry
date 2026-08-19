"""Small rank-correlation utilities with scientifically correct tie handling."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence

import numpy as np


def average_ranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, assigning equal values equal ranks.

    Zero-based versus one-based ranks do not change correlations. Average ranks
    do: assigning sequential ranks to tied values makes Spearman correlation
    depend on arbitrary input order.
    """

    array = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=float)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def pearson_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    """Return Pearson correlation, or ``None`` for undersized/constant inputs."""

    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("correlation inputs must have equal length")
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_correlation(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> float | None:
    """Return Spearman correlation using average ranks for every tie group."""

    x = np.asarray(left, dtype=float).reshape(-1)
    y = np.asarray(right, dtype=float).reshape(-1)
    if x.size != y.size:
        raise ValueError("correlation inputs must have equal length")
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return pearson_correlation(average_ranks(x), average_ranks(y))


def label_permutation_test(
    observed_labels: Sequence[int],
    fixed_distances: Sequence[float] | np.ndarray,
    distance_vector: Callable[[Sequence[int]], np.ndarray],
    *,
    exact: bool,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> dict[str, float | int | str | None]:
    """Test a distance-matrix association by permuting concept labels.

    The unit being permuted is the concept label, not an individual pairwise
    distance. This preserves dyadic dependence. Exact mode enumerates all
    ``n!`` label permutations. Monte Carlo mode uses a frozen seed and the
    plus-one correction.
    """

    labels = tuple(int(value) for value in observed_labels)
    fixed = np.asarray(fixed_distances, dtype=float)
    observed = spearman_correlation(distance_vector(labels), fixed)
    if observed is None:
        return {
            "observed": None,
            "p_value": None,
            "permutations": 0,
            "method": "undefined",
        }
    if exact:
        permutations = itertools.permutations(labels)
        exceedances = 0
        total = 0
        for permutation in permutations:
            statistic = spearman_correlation(distance_vector(permutation), fixed)
            if statistic is not None and abs(statistic) >= abs(observed) - 1e-15:
                exceedances += 1
            total += 1
        return {
            "observed": observed,
            "p_value": exceedances / total,
            "permutations": total,
            "method": "exact_label_permutation",
        }
    if n_permutations <= 0 or seed is None:
        raise ValueError("Monte Carlo permutation requires a positive count and seed")
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        permutation = tuple(rng.permutation(labels).tolist())
        statistic = spearman_correlation(distance_vector(permutation), fixed)
        if statistic is not None and abs(statistic) >= abs(observed) - 1e-15:
            exceedances += 1
    return {
        "observed": observed,
        "p_value": (exceedances + 1) / (n_permutations + 1),
        "permutations": n_permutations,
        "method": "monte_carlo_label_permutation_plus_one",
    }

