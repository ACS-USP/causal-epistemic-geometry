import numpy as np

from epistemic_geometry.metrics.reasoning import (
    SeedRegime,
    error_propensity,
    excess_pair_oracle,
    expected_pair_oracle,
    propensity_correlation_from_rollouts,
    propensity_distances,
    split_half_reliability,
    stochastic_complementarity_estimands,
    unbiased_two_rollout_propensity_distance,
)


def test_error_propensity_and_excess_pair_oracle() -> None:
    baseline = np.array([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=bool)
    steered = np.array([[0, 0, 1, 1], [0, 1, 0, 1]], dtype=bool)
    p0 = error_propensity(baseline)
    pj = error_propensity(steered)
    assert np.allclose(p0, [0.5, 0.0])
    assert np.allclose(pj, [0.5, 0.5])
    assert expected_pair_oracle(p0, p0) == 0.875
    assert np.isclose(excess_pair_oracle(p0, pj), 0.0)
    distances = propensity_distances(p0, pj)
    assert distances["squared_distance"] == 0.125


def test_split_half_reliability_is_exact_for_identical_halves() -> None:
    errors = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=bool)
    result = split_half_reliability(errors)
    assert result["correlation"] == 1.0
    assert result["mean_absolute_difference"] == 0.0
    assert result["squared_difference"] == 0.0
    assert result["status"] == "DESCRIPTIVE_SPLIT_HALF"


def test_stochastic_estimand_decomposition_is_exact() -> None:
    p0 = np.array([0.1, 0.3, 0.7, 0.9])
    pj = np.array([0.2, 0.4, 0.4, 0.8])
    result = stochastic_complementarity_estimands(p0, pj)
    direct = np.mean(p0**2 - p0 * pj)
    assert np.isclose(result["oracle_gain_g_j"], direct)
    assert np.isclose(
        result["oracle_gain_g_j"],
        result["competence_component"]
        + result["competence_adjusted_complementarity_c_j"],
    )
    assert abs(result["decomposition_residual"]) < 1e-12


def test_unbiased_two_rollout_distance_exact_fixture_and_seed_guard() -> None:
    left = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], dtype=bool)
    right = np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=bool)
    expected_rows = [0, 1, 0, 0]
    assert unbiased_two_rollout_propensity_distance(
        left,
        right,
        seed_regime=SeedRegime.INDEPENDENT_PRIMARY,
        item_ids_i=["a", "b", "c", "d"],
        item_ids_j=["a", "b", "c", "d"],
    ) == np.mean(expected_rows)
    with np.testing.assert_raises_regex(ValueError, "INDEPENDENT_PRIMARY"):
        unbiased_two_rollout_propensity_distance(
            left,
            right,
            seed_regime=SeedRegime.MATCHED_COUPLING_SECONDARY,
            item_ids_i=["a", "b", "c", "d"],
            item_ids_j=["a", "b", "c", "d"],
        )
    with np.testing.assert_raises_regex(ValueError, "row order"):
        unbiased_two_rollout_propensity_distance(
            left,
            right,
            seed_regime=SeedRegime.INDEPENDENT_PRIMARY,
            item_ids_i=["a", "b", "c", "d"],
            item_ids_j=["b", "a", "c", "d"],
        )


def test_unbiased_two_rollout_distance_is_unbiased_in_monte_carlo() -> None:
    rng = np.random.default_rng(123)
    p_i = np.linspace(0.05, 0.95, 200)
    p_j = np.linspace(0.9, 0.1, 200)
    estimates = []
    for _ in range(2_000):
        left = rng.random((200, 2)) < p_i[:, None]
        right = rng.random((200, 2)) < p_j[:, None]
        estimates.append(
            unbiased_two_rollout_propensity_distance(
                left,
                right,
                seed_regime=SeedRegime.INDEPENDENT_PRIMARY,
                item_ids_i=[str(index) for index in range(200)],
                item_ids_j=[str(index) for index in range(200)],
            )
        )
    assert abs(np.mean(estimates) - np.mean((p_i - p_j) ** 2)) < 0.01


def test_two_rollout_propensity_correlation_requires_explicit_low_resolution() -> None:
    left = np.array([[0, 0], [0, 1], [1, 1], [0, 1]], dtype=bool)
    right = np.array([[0, 1], [0, 1], [1, 1], [0, 0]], dtype=bool)
    with np.testing.assert_raises_regex(ValueError, ">=4 rollouts"):
        propensity_correlation_from_rollouts(left, right)
    result = propensity_correlation_from_rollouts(
        left,
        right,
        allow_two_rollout_low_resolution=True,
    )
    assert result["status"] == "LOW_RESOLUTION_TWO_ROLLOUT_PLUGIN_ATTENUATED"
