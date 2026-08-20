from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.micro_q1 import (
    bootstrap_pair_estimands,
    construct_paired_direction,
    heldout_signed_gaps,
    pair_estimands,
    random_orthogonal_direction,
)


def test_paired_direction_orients_toward_careful_and_freezes_delta() -> None:
    careful = np.array([[2.0, 0.0], [3.0, 0.0]])
    direct = np.array([[0.0, 0.0], [1.0, 0.0]])
    direction, delta, raw = construct_paired_direction(careful, direct)
    assert np.allclose(direction, [1.0, 0.0])
    assert np.allclose(raw, [2.0, 0.0])
    assert delta == 2.0
    assert np.all(heldout_signed_gaps(direction, careful, direct) > 0)


def test_random_control_is_unit_and_orthogonal() -> None:
    direction = np.array([1.0, 2.0, 3.0])
    control, cosine = random_orthogonal_direction(direction, 17)
    assert np.isclose(np.linalg.norm(control), 1.0)
    assert abs(cosine) <= 1e-12


def test_gate4_estimands_use_all_four_cross_products_and_identity() -> None:
    baseline = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    condition = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    metrics = pair_estimands(baseline, condition)
    assert metrics["B00"] == 0.25
    assert metrics["B0j"] == 0.3125
    assert np.isclose(
        metrics["rescue"] - metrics["damage"],
        metrics["accuracy_condition"] - metrics["accuracy_baseline"],
    )


def test_item_cluster_bootstrap_keeps_condition_rollouts_together() -> None:
    baseline = np.array([[1, 1], [0, 0], [1, 0]])
    conditions = {"plus": np.array([[0, 0], [1, 1], [1, 1]]), "random": baseline.copy()}
    result = bootstrap_pair_estimands(baseline, conditions, resamples=25, seed=9)
    assert set(result) == {"plus", "random"}
    assert set(result["plus"]) >= {"G", "C", "D", "rescue", "damage"}
