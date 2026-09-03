from __future__ import annotations

import hashlib
import json
from pathlib import Path

from epistemic_geometry.experiments import q1_second_task_stage_a2 as a2

ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
    / "stage_a_failure_audit"
)


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


def test_stage_a2_principal_authorization_pins_reviewed_artifacts() -> None:
    authorization = json.loads(
        (AUDIT / "STAGE_A2_PRINCIPAL_AUTHORIZATION.json").read_text(encoding="utf-8")
    )
    names = {
        "amendment2_draft": "AMENDMENT2_LOCK_DRAFT.json",
        "stage_a2_manifest": "STAGE_A2_FAMILY_MANIFEST.json",
        "stage_a2_schedule": "STAGE_A2_SCHEDULE.json",
    }
    for key, name in names.items():
        assert hashlib.sha256((AUDIT / name).read_bytes()).hexdigest() == authorization[
            "reviewed_hashes"
        ][key]
    assert authorization["authorized"] == "STAGE_A2_ONLY"
    assert authorization["stage_a1"] == "EXECUTED_AND_FAILED_AS_FROZEN"
    assert authorization["stage_b"] == "CLOSED_NOT_AUTHORIZED"
    assert authorization["meaningful_activation_controller_on_livecodebench"] == (
        "SEALED_NOT_OPENED"
    )
    assert authorization["activation_nulls_on_livecodebench"] == "SEALED_NOT_OPENED"
    assert authorization["fresh_stage_a2_outcomes_before_authorization"] == 0
    assert authorization["fresh_stage_a2_correctness_inspected"] is False
