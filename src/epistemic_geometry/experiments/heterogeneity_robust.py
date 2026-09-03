"""Controller/node-level inference primitives for Q2 sensitivity work.

All functions are model-free and operate on already constructed numeric
matrices.  Fresh controllers or symmetric controller nodes, never dyads, are
the independent resampling units.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import spearman_flat
from epistemic_geometry.experiments.q2_v4 import average_ranks


def binomial_upper_tail(successes: int, trials: int, probability: float = 0.5) -> float:
    """Exact upper-tail binomial probability."""

    if not 0 <= successes <= trials or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid binomial arguments")
    return float(
        sum(
            math.comb(trials, value)
            * probability**value
            * (1.0 - probability) ** (trials - value)
            for value in range(successes, trials + 1)
        )
    )


def row_spearman(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Equal-shell mean within-reference Spearman for every fresh row."""

    if tuple(geometry_by_shell) != tuple(outcome_by_shell):
        raise ValueError("shells or shell order differ")
    first = np.asarray(next(iter(geometry_by_shell.values())), dtype=np.float64)
    if first.ndim != 2:
        raise ValueError("cross blocks must be matrices")
    result = np.empty(first.shape[0], dtype=np.float64)
    for row in range(first.shape[0]):
        values = [
            spearman_flat(
                np.asarray(geometry_by_shell[shell])[row],
                np.asarray(outcome_by_shell[shell])[row],
            )
            for shell in geometry_by_shell
        ]
        result[row] = np.mean(values) if np.all(np.isfinite(values)) else np.nan
    return result


def exact_sign_test(values: Sequence[float]) -> dict[str, float | int | bool]:
    """One-sided exact sign test, prospectively discarding exact zeros."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    nonzero = finite[finite != 0.0]
    positives = int(np.sum(nonzero > 0.0))
    p_value = 1.0 if len(nonzero) == 0 else binomial_upper_tail(positives, len(nonzero))
    return {
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "nonzero": int(len(nonzero)),
        "positives": positives,
        "p_value": p_value,
        "reject_0_05": bool(p_value <= 0.05),
    }


def exact_positive_sign_test(values: Sequence[float]) -> dict[str, float | int | bool]:
    """Exact positive-sign test with zeros retained as non-successes.

    This is the prospective Q2 OOS V2 erratum rule.  Every finite controller
    contributes to the Binomial denominator: positive values are successes;
    zero and negative values are non-successes.  Any nonfinite row fails closed.
    """

    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.all(finite):
        return {
            "count": int(len(array)),
            "finite": int(np.sum(finite)),
            "positives": int(np.sum(array[finite] > 0.0)),
            "zeros": int(np.sum(array[finite] == 0.0)),
            "negatives": int(np.sum(array[finite] < 0.0)),
            "p_value": 1.0,
            "reject_0_05": False,
            "degenerate": True,
        }
    positives = int(np.sum(array > 0.0))
    p_value = binomial_upper_tail(positives, len(array))
    return {
        "count": int(len(array)),
        "finite": int(len(array)),
        "positives": positives,
        "zeros": int(np.sum(array == 0.0)),
        "negatives": int(np.sum(array < 0.0)),
        "p_value": p_value,
        "reject_0_05": bool(p_value <= 0.05),
        "degenerate": False,
    }


def _continued_beta(a: float, b: float, x: float) -> float:
    maximum_iterations = 300
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        twice = 2 * iteration
        coefficient = iteration * (b - iteration) * x / ((qam + twice) * (a + twice))
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        coefficient = -(a + iteration) * (qab + iteration) * x / (
            (a + twice) * (qap + twice)
        )
        d = 1.0 + coefficient * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + coefficient / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta without an optional SciPy dependency."""

    if not a > 0.0 or not b > 0.0 or not 0.0 <= x <= 1.0:
        raise ValueError("invalid incomplete-beta arguments")
    if x in (0.0, 1.0):
        return x
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta(a, b, x) / a
    return 1.0 - front * _continued_beta(b, a, 1.0 - x) / b


def student_t_survival(statistic: float, degrees_freedom: int) -> float:
    """One-sided upper-tail probability for Student's t."""

    if degrees_freedom <= 0 or not np.isfinite(statistic):
        return 1.0
    x = degrees_freedom / (degrees_freedom + statistic * statistic)
    half_tail = 0.5 * regularized_incomplete_beta(degrees_freedom / 2.0, 0.5, x)
    return float(half_tail if statistic >= 0.0 else 1.0 - half_tail)


def studentized_mean_test(values: Sequence[float]) -> dict[str, float | int | bool]:
    """One-sided one-sample t test of a positive controller mean."""

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    count = len(array)
    if count < 2:
        return {
            "count": count,
            "mean": float("nan"),
            "standard_error": float("nan"),
            "t": float("nan"),
            "p_value": 1.0,
            "reject_0_05": False,
        }
    mean = float(np.mean(array))
    standard_error = float(np.std(array, ddof=1) / math.sqrt(count))
    statistic = mean / standard_error if standard_error > 0.0 else float("nan")
    p_value = student_t_survival(statistic, count - 1)
    return {
        "count": count,
        "mean": mean,
        "standard_error": standard_error,
        "t": statistic,
        "p_value": p_value,
        "reject_0_05": bool(p_value <= 0.05),
    }


def two_way_residualize(matrix: np.ndarray) -> np.ndarray:
    """Remove fresh-row and fixed-reference additive effects."""

    values = np.asarray(matrix, dtype=np.float64)
    return (
        values
        - np.mean(values, axis=1, keepdims=True)
        - np.mean(values, axis=0, keepdims=True)
        + np.mean(values)
    )


def rank_cluster_regression(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
) -> dict[str, float | int | bool]:
    """Two-way fixed-effect rank slope with fresh-row CR1 inference."""

    if tuple(geometry_by_shell) != tuple(outcome_by_shell):
        raise ValueError("shells or shell order differ")
    x_shells: list[np.ndarray] = []
    y_shells: list[np.ndarray] = []
    for shell in geometry_by_shell:
        geometry = np.asarray(geometry_by_shell[shell], dtype=np.float64)
        outcome = np.asarray(outcome_by_shell[shell], dtype=np.float64)
        if geometry.shape != outcome.shape or geometry.ndim != 2:
            raise ValueError("rank-regression matrices differ")
        x_shells.append(
            two_way_residualize(average_ranks(geometry.reshape(-1)).reshape(geometry.shape))
        )
        y_shells.append(
            two_way_residualize(average_ranks(outcome.reshape(-1)).reshape(outcome.shape))
        )
    x = np.stack(x_shells, axis=0)
    y = np.stack(y_shells, axis=0)
    bread = float(np.sum(x * x))
    clusters = x.shape[1]
    if bread <= 0.0 or clusters < 2:
        return {
            "slope": float("nan"),
            "standard_error": float("nan"),
            "t": float("nan"),
            "p_value": 1.0,
            "reject_0_05": False,
            "clusters": clusters,
        }
    slope = float(np.sum(x * y) / bread)
    residual = y - slope * x
    scores = np.sum(x * residual, axis=(0, 2))
    variance = clusters / (clusters - 1.0) * float(np.sum(scores * scores)) / (bread * bread)
    standard_error = math.sqrt(max(variance, 0.0))
    statistic = slope / standard_error if standard_error > 0.0 else float("nan")
    p_value = student_t_survival(statistic, clusters - 1)
    return {
        "slope": slope,
        "standard_error": standard_error,
        "t": statistic,
        "p_value": p_value,
        "reject_0_05": bool(p_value <= 0.05),
        "clusters": clusters,
    }


def symmetric_shell_association(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
    keep: np.ndarray | None = None,
) -> float:
    """Equal-shell upper-triangle Spearman, optionally on retained nodes."""

    if tuple(geometry_by_shell) != tuple(outcome_by_shell):
        raise ValueError("shells or shell order differ")
    correlations = []
    for shell in geometry_by_shell:
        geometry = np.asarray(geometry_by_shell[shell], dtype=np.float64)
        outcome = np.asarray(outcome_by_shell[shell], dtype=np.float64)
        if keep is not None:
            geometry = geometry[np.ix_(keep, keep)]
            outcome = outcome[np.ix_(keep, keep)]
        upper = np.triu_indices(len(geometry), 1)
        correlations.append(spearman_flat(geometry[upper], outcome[upper]))
    return float(np.mean(correlations))


def node_jackknife_test(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Node-jackknife pseudovalue t inference for symmetric matrices."""

    count = len(np.asarray(next(iter(geometry_by_shell.values()))))
    full = symmetric_shell_association(geometry_by_shell, outcome_by_shell)
    leave_one_out = np.empty(count, dtype=np.float64)
    nodes = np.arange(count)
    for node in range(count):
        leave_one_out[node] = symmetric_shell_association(
            geometry_by_shell, outcome_by_shell, nodes[nodes != node]
        )
    pseudovalues = count * full - (count - 1) * leave_one_out
    test = studentized_mean_test(pseudovalues)
    standard_error = float(np.std(pseudovalues, ddof=1) / math.sqrt(count))
    return {
        "full_association": full,
        "leave_one_out": leave_one_out,
        "pseudovalues": pseudovalues,
        "jackknife_standard_error": standard_error,
        "t": test["t"],
        "p_value": test["p_value"],
        "reject_0_05": test["reject_0_05"],
    }
