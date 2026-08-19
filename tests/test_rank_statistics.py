from __future__ import annotations

import math

import numpy as np

from epistemic_geometry.analysis.rank_statistics import (
    average_ranks,
    label_permutation_test,
    spearman_correlation,
)


def test_average_ranks_without_ties() -> None:
    assert np.array_equal(average_ranks([30, 10, 20]), [2, 0, 1])


def test_average_ranks_assigns_tied_groups_their_mean_rank() -> None:
    assert np.array_equal(average_ranks([10, 20, 20, 40, 40, 40]), [0, 1.5, 1.5, 4, 4, 4])


def test_average_ranks_all_equal_and_spearman_undefined() -> None:
    assert np.array_equal(average_ranks([7, 7, 7]), [1, 1, 1])
    assert spearman_correlation([7, 7, 7], [1, 2, 3]) is None


def test_spearman_known_reference_with_ties() -> None:
    # Reference obtained by applying Pearson correlation to average ranks.
    value = spearman_correlation([1, 2, 2, 4, 5], [5, 4, 4, 2, 1])
    assert math.isclose(value or 0.0, -1.0, abs_tol=1e-12)


def test_spearman_is_invariant_to_joint_input_permutation() -> None:
    left = np.array([0, 0, 1, 1, 2, 2, 2], dtype=float)
    right = np.array([2, 1, 0, 1, 4, 3, 4], dtype=float)
    permutation = np.array([6, 0, 4, 2, 1, 5, 3])
    assert math.isclose(
        spearman_correlation(left, right) or 0.0,
        spearman_correlation(left[permutation], right[permutation]) or 0.0,
        abs_tol=1e-12,
    )


def test_weekday_conceptual_fixture_uses_all_seven_factorial_labelings() -> None:
    labels = tuple(range(7))

    def cyclic_distances(order: tuple[int, ...]) -> np.ndarray:
        return np.asarray(
            [
                min(abs(order[i] - order[j]), 7 - abs(order[i] - order[j]))
                for i in range(7)
                for j in range(i + 1, 7)
            ],
            dtype=float,
        )

    result = label_permutation_test(
        labels,
        cyclic_distances(labels),
        cyclic_distances,
        exact=True,
    )
    assert result["permutations"] == math.factorial(7)
    assert result["observed"] == 1.0
    assert 0 < float(result["p_value"] or 0) <= 1

