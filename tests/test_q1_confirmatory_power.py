from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate6_3_v3 import audit_two_rollout_estimands
from epistemic_geometry.experiments.q1_confirmatory_power import (
    c_from_feature_sums,
    c_sufficient_features,
    nested_item_bootstrap_power,
    point_c,
)


def _fixture() -> dict[str, np.ndarray]:
    return {
        "BASELINE": np.asarray([[1, 1], [1, 0], [0, 0], [1, 1], [0, 1], [0, 0]]),
        "MEANINGFUL": np.asarray([[0, 1], [0, 0], [0, 0], [1, 0], [0, 1], [0, 0]]),
        "R0": np.asarray([[1, 1], [1, 0], [0, 1], [1, 1], [0, 0], [0, 0]]),
        "R1": np.asarray([[1, 0], [1, 1], [0, 0], [1, 1], [1, 0], [0, 0]]),
    }


def test_sufficient_features_reproduce_canonical_c() -> None:
    arrays = _fixture()
    names = ("MEANINGFUL", "R0", "R1")
    features = c_sufficient_features(arrays, baseline="BASELINE", conditions=names)
    expected = np.asarray(
        [audit_two_rollout_estimands(arrays["BASELINE"], arrays[name])["C"] for name in names]
    )
    assert np.allclose(point_c(features), expected, atol=1e-12, rtol=0)
    assert np.allclose(
        c_from_feature_sums(features.sum(axis=0), len(features)),
        expected,
        atol=1e-12,
        rtol=0,
    )


def test_nested_power_is_deterministic_and_reports_both_targets() -> None:
    arrays = _fixture()
    features = c_sufficient_features(
        arrays, baseline="BASELINE", conditions=("MEANINGFUL", "R0", "R1")
    )
    first = nested_item_bootstrap_power(
        features,
        n_items=5,
        outer_replications=80,
        inner_resamples=79,
        seed=12345,
        batch_size=9,
    )
    second = nested_item_bootstrap_power(
        features,
        n_items=5,
        outer_replications=80,
        inner_resamples=79,
        seed=12345,
        batch_size=9,
    )
    assert first == second
    assert 0 <= first["meaningful_c_power_lower_bound_gt_zero"] <= 1
    assert 0 <= first["specificity_c_power_lower_bound_gt_zero"] <= 1
    assert first["meaningful_c_expected_interval_width"] >= 0
    assert first["specificity_c_expected_interval_width"] >= 0
