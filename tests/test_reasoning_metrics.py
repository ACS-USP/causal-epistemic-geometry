import numpy as np

from epistemic_geometry.metrics.reasoning import (
    error_propensity,
    excess_pair_oracle,
    expected_pair_oracle,
    propensity_distances,
    split_half_reliability,
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
