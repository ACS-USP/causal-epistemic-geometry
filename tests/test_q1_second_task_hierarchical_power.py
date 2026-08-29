from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments import q1_second_task_hierarchical_power as hp


def test_row_weighted_design_effect_limits() -> None:
    sizes = [2, 2, 3, 3]
    assert hp.unequal_cluster_design_effect(sizes, 0.0) == 1.0
    expected_effective = sum(sizes) ** 2 / sum(value**2 for value in sizes)
    observed_effective = sum(sizes) / hp.unequal_cluster_design_effect(sizes, 1.0)
    assert np.isclose(observed_effective, expected_effective)


def test_family_average_variance_factor_limits() -> None:
    sizes = [2, 3, 4]
    assert np.isclose(hp.family_average_variance_factor(sizes, 0.0), np.mean([0.5, 1 / 3, 0.25]))
    assert hp.family_average_variance_factor(sizes, 1.0) == 1.0


def test_one_row_design_is_invariant_to_within_family_dependence() -> None:
    result = hp.simulate_one_row_per_family(
        80, transfer_fraction=0.75, replicates=2_000, seed=3
    )
    assert result["within_family_rho"] is None
    assert result["effective_units"] == 80


def test_clustered_design_collapses_to_one_row_at_perfect_dependence() -> None:
    family = hp.simulate_family_balanced(
        [2] * 80, rho=1.0, transfer_fraction=0.75, replicates=2_000, seed=7
    )
    one = hp.simulate_one_row_per_family(
        80, transfer_fraction=0.75, replicates=2_000, seed=7
    )
    assert family["expected_c_ci_width"] == one["expected_c_ci_width"]
    assert family["joint_frozen_rule_probability"] == one["joint_frozen_rule_probability"]
