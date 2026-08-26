"""Outcome-free statistical primitives for the four-family Q2 V3 redesign.

The module deliberately contains no model, parser, journal, or semantic-outcome
imports.  It formalizes the controller-label group and geometry-only gates that
can be fixed before a future Amendment-2 execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import permutations, product

import numpy as np

FAMILIES = (
    "CONTROL_FLOW_PATH_COVERAGE",
    "MUTATION_ALIAS_CAUSALITY",
    "LOOP_BOUNDARY_ACCOUNTING",
    "HYPOTHESIS_BRANCH_ELIMINATION",
)
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
SHELLS = ("MEDIUM", "STRONG")


def direction_index(family_index: int, location_index: int) -> int:
    """Return the canonical base-direction index."""

    return 2 * family_index + location_index


def family_qap_mappings(
    family_count: int = 4,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate ``S_F semidirect (Z_2)^F`` controller-label mappings.

    A family block is sent to another intact family block and may be swapped
    internally.  The same mapping is intended to be applied to every shell.
    The returned tuple maps each canonical target slot to its permuted source
    slot.
    """

    mappings: list[tuple[int, ...]] = []
    for family_permutation in permutations(range(family_count)):
        for swaps in product((0, 1), repeat=family_count):
            mapping = tuple(
                direction_index(family_permutation[family], location ^ swaps[family])
                for family in range(family_count)
                for location in range(2)
            )
            mappings.append(mapping)
    return tuple(mappings)


def cross_family_edges(family_count: int = 4) -> tuple[tuple[int, int], ...]:
    """Return canonical cross-family unordered base-direction pairs."""

    return tuple(
        (left, right)
        for left in range(2 * family_count)
        for right in range(left + 1, 2 * family_count)
        if left // 2 != right // 2
    )


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ranks with deterministic tie handling."""

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


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    """Pearson correlation of average ranks, matching the frozen definition."""

    left_rank = average_ranks(left)
    right_rank = average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def family_balanced_rho(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
    *,
    mapping: Sequence[int] | None = None,
    omitted_direction: int | None = None,
) -> dict[str, object]:
    """Compute the four-family analogue of the frozen family-balanced rho.

    Each family receives equal weight within a shell, and the two shells receive
    equal weight.  A QAP mapping reindexes geometry only; outcome labels and the
    conceptual family incidence structure remain fixed.
    """

    index = tuple(range(8)) if mapping is None else tuple(int(value) for value in mapping)
    if sorted(index) != list(range(8)):
        raise ValueError("mapping must be a permutation of eight directions")
    family_shell: dict[str, dict[str, float]] = {}
    shell_summary: dict[str, float] = {}
    for shell in SHELLS:
        geometry = np.asarray(geometry_by_shell[shell], dtype=np.float64)
        outcome = np.asarray(outcome_by_shell[shell], dtype=np.float64)
        if geometry.shape != (8, 8) or outcome.shape != (8, 8):
            raise ValueError("four-family matrices must have shape 8x8")
        family_shell[shell] = {}
        for family_index, family in enumerate(FAMILIES):
            incident = [
                (left, right)
                for left, right in cross_family_edges()
                if (left // 2 == family_index or right // 2 == family_index)
                and omitted_direction not in (left, right)
            ]
            geometry_values = [geometry[index[left], index[right]] for left, right in incident]
            outcome_values = [outcome[left, right] for left, right in incident]
            family_shell[shell][family] = spearman(geometry_values, outcome_values)
        shell_values = np.asarray(list(family_shell[shell].values()), dtype=np.float64)
        shell_summary[shell] = float(np.mean(shell_values))
    aggregate = float(np.mean(list(shell_summary.values())))
    family_summary = {
        family: float(np.mean([family_shell[shell][family] for shell in SHELLS]))
        for family in FAMILIES
    }
    return {
        "aggregate": aggregate,
        "shell_summary": shell_summary,
        "family_summary": family_summary,
        "family_shell": family_shell,
    }


def exact_qap(
    geometries: Mapping[str, Mapping[str, np.ndarray]],
    outcomes: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Run exhaustive single-step maxT QAP across M0/M1/M2.

    The identity is included and p-values are ``count(null >= observed)/384``.
    The global test uses the largest observed metric statistic against the
    permutation-wise maximum.  Metric attribution uses the same single-step
    maxT null, so no uncorrected post-omnibus metric claim is possible.
    """

    if set(geometries) != {"M0", "M1", "M2"}:
        raise ValueError("exact QAP requires exactly M0/M1/M2")
    mappings = family_qap_mappings()
    metric_order = ("M0", "M1", "M2")
    edges = cross_family_edges()
    edge_lookup = {edge: index for index, edge in enumerate(edges)}
    edge_maps = np.asarray(
        [
            [edge_lookup[tuple(sorted((mapping[left], mapping[right])))] for left, right in edges]
            for mapping in mappings
        ],
        dtype=np.int64,
    )
    incident = [
        np.asarray(
            [
                index
                for index, (left, right) in enumerate(edges)
                if left // 2 == family or right // 2 == family
            ],
            dtype=np.int64,
        )
        for family in range(4)
    ]
    null = np.zeros((len(mappings), 3), dtype=np.float64)
    for metric_index, metric in enumerate(metric_order):
        for shell in SHELLS:
            geometry = np.asarray(geometries[metric][shell], dtype=np.float64)
            outcome = np.asarray(outcomes[shell], dtype=np.float64)
            geometry_edges = np.asarray([geometry[left, right] for left, right in edges])
            outcome_edges = np.asarray([outcome[left, right] for left, right in edges])
            permuted_edges = geometry_edges[edge_maps]
            for indices in incident:
                geometry_ranks = _average_rank_rows(permuted_edges[:, indices])
                outcome_ranks = average_ranks(outcome_edges[indices])
                outcome_centered = outcome_ranks - np.mean(outcome_ranks)
                geometry_centered = geometry_ranks - np.mean(geometry_ranks, axis=1)[:, None]
                denominator = np.linalg.norm(geometry_centered, axis=1) * np.linalg.norm(
                    outcome_centered
                )
                correlations = np.divide(
                    geometry_centered @ outcome_centered,
                    denominator,
                    out=np.full(len(mappings), np.nan),
                    where=denominator > 0.0,
                )
                null[:, metric_index] += correlations / 8.0
    observed = {
        metric: float(null[0, metric_index])
        for metric_index, metric in enumerate(metric_order)
    }
    max_null = np.max(null, axis=1)
    observed_max = max(observed.values())
    return {
        "universe_size": len(mappings),
        "observed": observed,
        "global_max_statistic": observed_max,
        "global_p": float(np.sum(max_null >= observed_max) / len(mappings)),
        "single_step_maxT_adjusted_p": {
            metric: float(np.sum(max_null >= observed[metric]) / len(mappings))
            for metric in metric_order
        },
        "uncorrected_p": {
            metric: float(np.sum(null[:, index] >= observed[metric]) / len(mappings))
            for index, metric in enumerate(metric_order)
        },
        "null": null,
    }


def _average_rank_rows(values: np.ndarray) -> np.ndarray:
    """Average-rank each row; QAP matrices are intentionally small."""

    array = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(array, axis=1)
    if np.all(np.diff(sorted_values, axis=1) != 0.0):
        order = np.argsort(array, axis=1, kind="mergesort")
        ranks = np.empty_like(array)
        np.put_along_axis(
            ranks,
            order,
            np.broadcast_to(np.arange(1, array.shape[1] + 1), array.shape),
            axis=1,
        )
        return ranks
    return np.vstack([average_ranks(row) for row in array])


def effective_rank(vectors: np.ndarray) -> float:
    """Participation-ratio effective rank of the unit-vector Gram matrix."""

    array = np.asarray(vectors, dtype=np.float64)
    unit = array / np.linalg.norm(array, axis=1, keepdims=True)
    eigenvalues = np.clip(np.linalg.eigvalsh(unit @ unit.T), 0.0, None)
    denominator = float(np.sum(np.square(eigenvalues)))
    return float(np.square(np.sum(eigenvalues)) / denominator)


def family_leverage(values: Sequence[float]) -> dict[str, float]:
    """Return incident centered-z energy shares for four families."""

    array = np.asarray(values, dtype=np.float64)
    edges = cross_family_edges()
    if array.shape != (len(edges),):
        raise ValueError("leverage requires one value per cross-family edge")
    standard_deviation = float(np.std(array))
    if standard_deviation <= 0.0:
        raise ValueError("leverage undefined for zero-spread geometry")
    z = (array - np.mean(array)) / standard_deviation
    denominator = 2.0 * float(np.sum(np.square(z)))
    return {
        family: float(
            np.sum(
                [
                    z[index] ** 2
                    for index, (left, right) in enumerate(edges)
                    if left // 2 == family_index or right // 2 == family_index
                ]
            )
            / denominator
        )
        for family_index, family in enumerate(FAMILIES)
    }


def lodo_rhos(
    geometry_by_shell: Mapping[str, np.ndarray],
    outcome_by_shell: Mapping[str, np.ndarray],
) -> tuple[float, ...]:
    """Return the eight leave-one-base-direction-out aggregate statistics."""

    return tuple(
        float(
            family_balanced_rho(
                geometry_by_shell,
                outcome_by_shell,
                omitted_direction=direction,
            )["aggregate"]
        )
        for direction in range(8)
    )


def radial_family_sign_flip_p(family_values: Sequence[float]) -> float:
    """Historical-style exact family-block sign-flip p-value for four families."""

    values = np.asarray(family_values, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("four family values required")
    observed = float(np.mean(values))
    null = [
        float(np.mean(values * np.asarray(signs, dtype=np.float64)))
        for signs in product((-1.0, 1.0), repeat=4)
    ]
    return float(np.sum(np.asarray(null) >= observed) / len(null))


__all__ = [
    "FAMILIES",
    "LOCATIONS",
    "SHELLS",
    "average_ranks",
    "cross_family_edges",
    "direction_index",
    "effective_rank",
    "exact_qap",
    "family_balanced_rho",
    "family_leverage",
    "family_qap_mappings",
    "lodo_rhos",
    "radial_family_sign_flip_p",
    "spearman",
]
