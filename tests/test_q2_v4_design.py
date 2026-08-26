from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments.q2_v4 import (
    baseline_centered_js_angle,
    blind_spot_shape_matrices,
    coefficient_bank_checks,
    controller_permutations,
    orthonormal_source_subspace,
    protocol_seed,
    sample_coefficient_bank,
    shell_coupled_qap_statistics,
)


def test_source_subspace_rank_and_conditioning() -> None:
    source = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1e-10],
        ]
    )
    basis, summary = orthonormal_source_subspace(
        source,
        relative_singular_threshold=1e-6,
    )
    assert summary.exact_rank == 3
    assert summary.retained_rank == 2
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)


def test_coefficient_bank_is_deterministic_and_well_formed() -> None:
    seed = protocol_seed("Q2-V4-TEST", "abc123")
    left = sample_coefficient_bank(8, 32, seed=seed)
    right = sample_coefficient_bank(8, 32, seed=seed)
    np.testing.assert_array_equal(left, right)
    checks = coefficient_bank_checks(left)
    assert checks["pass"] is True


def test_shape_estimator_finite_panel_expectation_and_population_correction() -> None:
    # Conditional propensity differences on three fixed items.
    left = np.asarray([0.9, 0.3, 0.6])
    right = np.asarray([0.2, 0.5, 0.1])
    delta = left - right
    finite_target = float(np.mean(np.square(delta)) - np.mean(delta) ** 2)
    population_target = finite_target * 3.0 / 2.0

    # Compute the exact expectation by enumerating all Bernoulli outcomes in
    # two independent rollout blocks for the two controllers.
    expectation_panel = 0.0
    expectation_population = 0.0
    for state in range(1 << 12):
        bits = np.asarray([(state >> index) & 1 for index in range(12)])
        errors = np.empty((2, 3, 2), dtype=np.float64)
        errors[0] = bits[:6].reshape(3, 2)
        errors[1] = bits[6:].reshape(3, 2)
        probability = 1.0
        for controller, propensity in enumerate((left, right)):
            offset = 0 if controller == 0 else 6
            for item in range(3):
                for rollout in range(2):
                    outcome = bits[offset + 2 * item + rollout]
                    probability *= propensity[item] if outcome else 1.0 - propensity[item]
        estimate = blind_spot_shape_matrices(errors)
        expectation_panel += probability * estimate["shape_frozen_panel"][0, 1]
        expectation_population += probability * estimate["shape_item_population"][0, 1]
    assert expectation_panel == pytest.approx(finite_target, abs=1e-12)
    assert expectation_population == pytest.approx(population_target, abs=1e-12)


def test_baseline_centered_js_angle_recovers_euclidean_angle() -> None:
    baseline = np.asarray([0.0, 0.0])
    controllers = np.asarray([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    points = np.vstack([baseline, controllers])
    squared = np.sum(np.square(points[:, None, :] - points[None, :, :]), axis=2)
    result = baseline_centered_js_angle(
        squared,
        zero_radius_squared_tolerance=1e-12,
    )
    expected = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(result["cosine"], expected, atol=1e-12)


def test_m2_angle_refuses_zero_radius() -> None:
    with pytest.raises(ValueError, match="near-zero"):
        baseline_centered_js_angle(
            np.asarray([[0.0, 0.0], [0.0, 0.0]]),
            zero_radius_squared_tolerance=1e-12,
        )


def test_qap_uses_same_controller_permutation_across_shells() -> None:
    base = np.asarray(
        [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
    )
    permutations = controller_permutations(3, 6, seed=42)
    result = shell_coupled_qap_statistics(
        {"A0": {"MEDIUM": base, "STRONG": 2.0 * base}},
        {"MEDIUM": base, "STRONG": 4.0 * base},
        permutations,
    )
    assert result["observed"]["A0"] == pytest.approx(1.0)
    assert result["null"].shape == (6, 1)
