from __future__ import annotations

from epistemic_geometry.experiments.q1_second_task_power import (
    PlanningCell,
    scaled_se,
    simulate_cell,
)


def test_scaled_se_improves_with_n_and_rollouts() -> None:
    interval = (0.0, 0.1)
    assert scaled_se(interval, n=200, rollouts=4) < scaled_se(interval, n=100, rollouts=2)


def test_power_simulation_is_deterministic_and_null_controlled() -> None:
    cell = PlanningCell(n=100, rollouts=2, random_controls=4, transfer_fraction=0.0)
    first = simulate_cell(cell, replicates=20_000, seed=17)
    second = simulate_cell(cell, replicates=20_000, seed=17)
    assert first == second
    assert first["joint_frozen_rule_probability"] <= 0.05


def test_more_information_improves_full_transfer_planning_power() -> None:
    low = simulate_cell(
        PlanningCell(n=100, rollouts=2, random_controls=8, transfer_fraction=1.0),
        replicates=30_000,
        seed=19,
    )
    high = simulate_cell(
        PlanningCell(n=200, rollouts=4, random_controls=8, transfer_fraction=1.0),
        replicates=30_000,
        seed=19,
    )
    assert high["joint_frozen_rule_probability"] > low["joint_frozen_rule_probability"]
