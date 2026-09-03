from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "review/q1_second_task_spark2_design"
REVIEW = PARENT / "amendment1_hierarchical_unit"


def _read(name: str) -> object:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_amendment_lock_pins_every_scientific_artifact() -> None:
    lock = _read("AMENDMENT_LOCK.json")
    assert isinstance(lock, dict)
    for name, expected in lock["artifact_hashes"].items():
        assert _sha256(REVIEW / name) == expected
    for name, expected in lock["inherited_lock_hashes"].items():
        assert _sha256(PARENT / name) == expected


def test_amended_manifests_are_family_disjoint() -> None:
    stage_a = _read("STAGE_A_FAMILY_MANIFEST.json")
    stage_b = _read("STAGE_B_FAMILY_MANIFEST.json")
    reserve = _read("RESERVE_FAMILY_MANIFEST.json")
    assert isinstance(stage_a, dict) and isinstance(stage_b, dict) and isinstance(reserve, dict)
    a_ids = {row["family_id"] for row in stage_a["ordered_families"]}
    b_ids = {row["family_id"] for row in stage_b["ordered_families"]}
    r_ids = {row["family_id"] for row in reserve["families"]}
    assert (len(a_ids), len(b_ids), len(r_ids)) == (32, 130, 20)
    assert not (a_ids & b_ids or a_ids & r_ids or b_ids & r_ids)


def test_amended_schedules_are_complete_and_unique() -> None:
    stage_a = _read("STAGE_A_SCHEDULE.json")
    stage_b = _read("STAGE_B_SCHEDULE.json")
    assert isinstance(stage_a, list) and isinstance(stage_b, list)
    assert (len(stage_a), len(stage_b)) == (128, 5720)
    rows = stage_a + stage_b
    keys = {
        (row["stage"], row["family_id"], row["condition"], row["rollout_index"])
        for row in rows
    }
    assert len(keys) == len(rows)
    assert len({row["seed"] for row in rows}) == len(rows)


def test_old_row_design_is_preserved_and_explicitly_superseded() -> None:
    record = _read("PREOUTCOME_SUPERSESSION.json")
    assert isinstance(record, dict)
    assert record["benchmark_outcomes_before_amendment"] == 0
    assert record["correctness_inspected"] is False
    for name, value in record["old_artifacts"].items():
        assert value["status"] == "SUPERSEDED_PRE_OUTCOME_NEVER_EXECUTED"
        assert _sha256(PARENT / name) == value["sha256"]


def test_independent_audit_preserves_scientific_firewall() -> None:
    audit = _read("INDEPENDENT_DESIGN_AUDIT.json")
    assert isinstance(audit, dict)
    assert audit["classification"] == "Q1_SECOND_TASK_HIERARCHICAL_DESIGN_FORENSIC_CLEAN"
    assert audit["scientific_inference"] == 0
    assert audit["correctness_inspected"] is False
    assert audit["q2_outputs_inspected"] is False
    assert audit["spark1_used"] is False
    assert audit["spark2_scientific_inference"] is False
