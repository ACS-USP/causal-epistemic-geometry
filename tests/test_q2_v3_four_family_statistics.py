from __future__ import annotations

import math

import numpy as np
from scripts.design_q2_v3_four_family_statistics import (
    matrix_from_edges,
    outcomes_from_contributions,
    panel_manifest,
)

from epistemic_geometry.experiments.q2_v3_four_family import (
    FAMILIES,
    cross_family_edges,
    effective_rank,
    family_balanced_rho,
    family_leverage,
    family_qap_mappings,
    lodo_rhos,
    radial_family_sign_flip_p,
    spearman,
)


def symmetric_from_scores(scores: np.ndarray) -> np.ndarray:
    matrix = np.zeros((8, 8), dtype=np.float64)
    for score, (left, right) in zip(scores, cross_family_edges(), strict=True):
        matrix[left, right] = score
        matrix[right, left] = score
    return matrix


def test_four_family_qap_is_semidirect_product_and_unique() -> None:
    mappings = family_qap_mappings()
    assert len(mappings) == math.factorial(4) * 2**4 == 384
    assert len(set(mappings)) == len(mappings)
    assert mappings[0] == tuple(range(8))
    for mapping in mappings:
        assert sorted(mapping) == list(range(8))
        for family in range(4):
            assert mapping[2 * family] // 2 == mapping[2 * family + 1] // 2


def test_historical_five_family_group_has_3840_elements() -> None:
    mappings = family_qap_mappings(5)
    assert len(mappings) == math.factorial(5) * 2**5 == 3840
    assert len(set(mappings)) == 3840


def test_four_family_pair_counts() -> None:
    edges = cross_family_edges()
    assert len(edges) == 24
    assert len({tuple(sorted(edge)) for edge in edges}) == 24
    assert all(left // 2 != right // 2 for left, right in edges)


def test_average_rank_spearman_handles_ties() -> None:
    assert math.isclose(spearman([1, 1, 2, 3], [10, 10, 20, 30]), 1.0)
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)


def test_family_balanced_rho_weights_families_and_shells_equally() -> None:
    scores = np.arange(24, dtype=np.float64)
    matrix = symmetric_from_scores(scores)
    result = family_balanced_rho(
        {"MEDIUM": matrix, "STRONG": matrix},
        {"MEDIUM": matrix, "STRONG": matrix},
    )
    assert result["aggregate"] == 1.0
    assert result["shell_summary"] == {"MEDIUM": 1.0, "STRONG": 1.0}
    assert result["family_summary"] == {family: 1.0 for family in FAMILIES}


def test_lodo_has_eight_positive_values_for_perfect_relation() -> None:
    scores = np.arange(24, dtype=np.float64)
    matrix = symmetric_from_scores(scores)
    values = lodo_rhos(
        {"MEDIUM": matrix, "STRONG": matrix},
        {"MEDIUM": matrix, "STRONG": matrix},
    )
    assert len(values) == 8
    assert all(value == 1.0 for value in values)


def test_effective_rank_and_family_leverage() -> None:
    vectors = np.eye(8)
    assert effective_rank(vectors) == 8.0
    scores = np.linspace(-1.0, 1.0, 24)
    leverage = family_leverage(scores)
    assert math.isclose(sum(leverage.values()), 1.0)
    assert all(0.0 <= value <= 1.0 for value in leverage.values())


def test_four_family_exact_sign_flip_cannot_reach_point_zero_five() -> None:
    p_value = radial_family_sign_flip_p([1.0, 1.0, 1.0, 1.0])
    assert p_value == 1.0 / 16.0
    assert p_value > 0.05


def test_canonical_two_rollout_contribution_is_preserved() -> None:
    contributions = np.zeros((4, 2, 24), dtype=np.float64)
    contributions[:, :, 0] = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    outcomes = outcomes_from_contributions(contributions, 4)
    left, right = cross_family_edges()[0]
    assert outcomes["MEDIUM"][left, right] == 0.5
    assert outcomes["STRONG"][left, right] == 0.5


def test_matrix_from_edges_is_symmetric() -> None:
    matrix = matrix_from_edges(np.arange(24, dtype=np.float64))
    assert np.array_equal(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)


def test_expanded_panel_is_outcome_free_and_disjoint() -> None:
    manifest = panel_manifest(300)
    assert manifest["eligible_count"] == 655
    assert manifest["selected_n"] == 300
    assert len(set(manifest["selected_ids"])) == 300
    assert set(manifest["selected_ids"]).isdisjoint(manifest["disjoint_excluded_ids"])
    assert manifest["correctness_values_read"] is False
    assert manifest["outcome_fields_loaded"] == []
