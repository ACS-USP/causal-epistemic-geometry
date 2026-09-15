from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.analysis.late_exposure_ablation import (
    PlanningCell,
    estimate_operating_characteristics,
    latent_contrasts,
    paired_evalues,
    simulate_pair,
)


def test_latent_contrasts_keep_all_rollouts_in_the_denominator() -> None:
    early = np.array([[1, 0], [1, 1]])
    full = np.array([[0, 0], [1, 0]])
    assert np.array_equal(latent_contrasts(early, full), np.array([0.5, 0.5]))


def test_paired_evalues_move_in_opposite_directions() -> None:
    harmful = paired_evalues(np.ones(20), delta=0.10)
    harmless = paired_evalues(np.zeros(20), delta=0.10)
    assert harmful["support_relevant_late_harm"] > 40
    assert harmless["exclude_relevant_late_harm"] > 2


def test_simulator_is_deterministic_and_preserves_margins() -> None:
    cell = PlanningCell(20_000, 0.60, 0.15, 1.0)
    first = simulate_pair(cell, rng=np.random.default_rng(17))
    second = simulate_pair(cell, rng=np.random.default_rng(17))
    assert all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))
    early, full = first
    assert early.mean() == pytest.approx(0.60, abs=0.02)
    assert full.mean() == pytest.approx(0.45, abs=0.02)
    assert np.all(full <= early)


def test_operating_grid_is_complete_and_repeatable() -> None:
    cells = [
        PlanningCell(12, 0.6, 0.0, 0.0),
        PlanningCell(12, 0.6, 0.2, 1.0),
    ]
    first = estimate_operating_characteristics(
        cells, delta=0.1, threshold=40, replications=100, seed=3
    )
    second = estimate_operating_characteristics(
        cells, delta=0.1, threshold=40, replications=100, seed=3
    )
    assert first == second
    assert len(first) == 2
    assert first[0]["support_rate"] <= first[1]["support_rate"]
