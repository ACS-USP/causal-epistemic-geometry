from __future__ import annotations

import numpy as np

from epistemic_geometry.analysis.q2_exploratory import (
    family_fold_edges,
    family_heldout_incremental,
    family_heldout_predictions,
    pair_feature_matrix,
    spearman,
    unbiased_error_distance,
)


def test_unbiased_error_distance_uses_cross_rollout_products() -> None:
    errors = np.asarray(
        [
            [[0, 0], [1, 1], [0, 1]],
            [[1, 1], [1, 0], [0, 0]],
        ],
        dtype=float,
    )
    result = unbiased_error_distance(errors)
    expected = np.mean(
        (errors[0, :, 0] - errors[1, :, 0])
        * (errors[0, :, 1] - errors[1, :, 1])
    )
    assert result.shape == (2, 2)
    assert result[0, 1] == result[1, 0] == expected
    assert np.allclose(np.diag(result), 0.0)


def test_family_fold_edges_match_frozen_cross_family_convention() -> None:
    names = ["a0", "a1", "b0", "b1", "c0", "c1"]
    families = {name: name[0] for name in names}
    train, test = family_fold_edges(names, families, "a")
    assert train == [2, 3, 4, 5]
    assert len(test) == 2 * 4
    assert all(left in (0, 1) and right in train for left, right in test)


def test_family_prediction_and_incremental_diagnostics_recover_signal() -> None:
    names = [f"{family}{index}" for family in "abc" for index in range(3)]
    families = {name: name[0] for name in names}
    positions = np.asarray([0.0, 0.2, 0.5, 1.0, 1.2, 1.7, 2.1, 2.7, 3.0])
    geometry = np.abs(positions[:, None] - positions[None, :])
    nuisance = pair_feature_matrix(np.ones(len(names)), "absolute_difference")
    target = 0.1 + 0.4 * geometry
    np.fill_diagonal(target, 0.0)

    result = family_heldout_predictions(target, [geometry], names, families)
    assert result["aggregate"]["mean_spearman"] > 0.99
    assert result["aggregate"]["rmse_ratio"] < 1e-10

    incremental = family_heldout_incremental(
        target, [nuisance], geometry, names, families
    )
    assert incremental["aggregate"]["augmented_to_nuisance_rmse_ratio"] < 0.1
    assert incremental["aggregate"]["mean_residual_spearman"] > 0.99


def test_pair_features_and_tied_spearman_are_deterministic() -> None:
    values = [0.25, 0.25, 0.75]
    assert np.allclose(
        pair_feature_matrix(values, "absolute_difference"),
        [[0.0, 0.0, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0]],
    )
    assert spearman(np.asarray([1, 1, 2, 3]), np.asarray([0, 0, 4, 9])) == 1.0
