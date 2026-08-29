from __future__ import annotations

from epistemic_geometry.experiments import q1_second_task_stage_a2 as a2


def reserve_fixture() -> dict:
    return {
        "families": [
            {
                "family_id": str(index),
                "family_size": 2,
                "family_program_sha256": f"family-{index}",
                "all_item_ids": [f"{index}:0", f"{index}:1"],
                "all_item_sha256": [f"hash-{index}-0", f"hash-{index}-1"],
                "selected_item": None,
                "selection_digest": None,
            }
            for index in range(20)
        ]
    }


def test_stage_a2_manifest_and_schedule_are_deterministic_and_unique() -> None:
    first = a2.selected_reserve_families(reserve_fixture())
    second = a2.selected_reserve_families(reserve_fixture())
    assert first == second
    schedule = a2.build_schedule(first)
    assert len(schedule) == 80
    assert len({a2.logical_key(row) for row in schedule}) == 80
    assert len({row["seed"] for row in schedule}) == 80
    assert {row["condition"] for row in schedule} == {"BASELINE", "TEXTUAL_CAREFUL"}


def test_stage_a2_gate_uses_twenty_equal_families_and_frozen_count_thresholds() -> None:
    schedule = a2.build_schedule(a2.selected_reserve_families(reserve_fixture()))
    parsed = []
    for row in schedule:
        family_index = int(row["family_id"])
        correct = family_index >= 2
        parsed.append(
            {
                **row,
                "commitment_valid": True,
                "semantic_evaluable": True,
                "correct": correct,
                "generated_token_count": 100
                if row["condition"] == "BASELINE"
                else 120,
            }
        )
    result = a2.stage_a2_gate(parsed)
    assert result["baseline"]["families_wrong_both_rollouts"] == 2
    assert result["baseline"]["families_correct_at_least_once"] == 18
    assert result["baseline"]["B00"] == 0.1
