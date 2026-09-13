from __future__ import annotations

import pytest

from epistemic_geometry.benchmarks.reasoning.families import FAMILY_CELLS
from epistemic_geometry.benchmarks.reasoning.novelty import (
    DIVERSIFICATION_COVERAGE_NAMESPACE,
    DIVERSIFICATION_EVALUATION_NAMESPACE,
    DIVERSIFICATION_QUALIFICATION_NAMESPACE,
    plan_diversification_coverage,
)
from epistemic_geometry.benchmarks.reasoning.splits import GEOMETRY_CALIBRATION

COUNTS = {
    "FSM-R": {"length_4": (2, 3), "length_8": {"qualification": 1, "evaluation": 2}},
    "MODREG-R": {"depth_4": 1},
}


def test_plan_is_deterministic_ordered_and_contains_schedule_metadata() -> None:
    first = plan_diversification_coverage(COUNTS, seed=20260913)
    second = plan_diversification_coverage(COUNTS, study_seed=20260913)

    assert first.to_record() == second.to_record()
    assert [(m.family, m.cell) for m in first.qualification_manifests] == [
        ("FSM-R", "length_4"),
        ("FSM-R", "length_8"),
        ("MODREG-R", "depth_4"),
    ]
    assert first.phase_namespaces == {
        "qualification": DIVERSIFICATION_QUALIFICATION_NAMESPACE,
        "evaluation": DIVERSIFICATION_EVALUATION_NAMESPACE,
    }
    assert DIVERSIFICATION_COVERAGE_NAMESPACE not in first.phase_namespaces.values()
    assert first.schedule_counts == {"qualification": 16, "evaluation": 36}
    assert all(first.schedule_keys.values())
    assert {key[1] for key in first.schedule_keys["qualification"]} == {"A", "T"}
    assert {key[1] for key in first.schedule_keys["evaluation"]} == {"A", "B", "T"}
    assert first.passed is True


def test_plan_excludes_history_and_keeps_phases_content_disjoint() -> None:
    seed = 71
    historical = plan_diversification_coverage({"FSM-R": {"length_4": 1}}, seed=seed)
    historical_id = historical.qualification_latent_ids[0]
    plan = plan_diversification_coverage(
        {"FSM-R": {"length_4": (2, 2)}},
        seed=seed,
        historical_excluded_latent_ids={historical_id},
    )
    qualification = set(plan.qualification_latent_ids)
    evaluation = set(plan.evaluation_latent_ids)
    assert historical_id not in qualification | evaluation
    assert qualification.isdisjoint(evaluation)
    assert plan.all_latent_ids == plan.qualification_latent_ids + plan.evaluation_latent_ids
    assert all(
        manifest.excluded_latent_ids == {historical_id}
        for manifest in plan.qualification_manifests
    )
    assert all(
        manifest.excluded_latent_ids == {historical_id} | qualification
        for manifest in plan.evaluation_manifests
    )


def test_plan_validates_family_cells_counts_and_historical_names() -> None:
    with pytest.raises(ValueError, match="unknown reasoning family"):
        plan_diversification_coverage({"UNKNOWN": {"length_4": 1}}, seed=1)
    with pytest.raises(ValueError, match="unknown reasoning family/cell"):
        plan_diversification_coverage({"FSM-R": {"depth_4": 1}}, seed=1)
    with pytest.raises(ValueError, match="positive integers"):
        plan_diversification_coverage({"FSM-R": {"length_4": (0, 1)}}, seed=1)
    with pytest.raises(ValueError, match="historical"):
        # Phase namespaces are fixed by the constructor, so historical split
        # names cannot be supplied or repurposed.
        plan_diversification_coverage(
            {GEOMETRY_CALIBRATION: {"length_4": 1}}, seed=1
        )


def test_plan_does_not_accept_manual_items() -> None:
    with pytest.raises(TypeError):
        plan_diversification_coverage(
            {"FSM-R": {"length_4": 1}}, seed=1, items=[]  # type: ignore[call-arg]
        )


def test_standard_three_family_four_cell_qualification_schedule_is_384() -> None:
    standard = {family: {cell: 8 for cell in cells} for family, cells in FAMILY_CELLS.items()}
    plan = plan_diversification_coverage(standard, seed=20260913)

    assert len(plan.qualification_latent_ids) == 3 * 4 * 8
    assert plan.schedule_counts["qualification"] == 96 * 2 * 2
    assert plan.schedule_counts["evaluation"] == 96 * 3 * 2
