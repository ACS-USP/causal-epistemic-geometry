from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q1_second_task_spark2_design"


def _read(name: str):
    return json.loads((REVIEW / name).read_text())


def test_frozen_stage_counts_and_disjoint_question_families() -> None:
    stage_a = _read("STAGE_A_MANIFEST.json")
    stage_b = _read("STAGE_B_HOLDOUT_MANIFEST.json")
    reserve = _read("RESERVE_MANIFEST.json")
    assert [stage_a["n_items"], stage_b["n_items"], reserve["n_items"]] == [50, 150, 242]
    groups = [
        {row["question_id"] for row in manifest["ordered_records"]}
        for manifest in (stage_a, stage_b, reserve)
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_frozen_schedules_have_unique_keys_and_seeds() -> None:
    expected = {"STAGE_A_SCHEDULE.json": 200, "STAGE_B_SCHEDULE.json": 6600}
    all_seeds = []
    for filename, count in expected.items():
        rows = _read(filename)
        keys = {
            (row["stage"], row["item_id"], row["condition"], row["rollout_index"])
            for row in rows
        }
        seeds = {row["seed"] for row in rows}
        assert len(rows) == len(keys) == len(seeds) == count
        all_seeds.extend(seeds)
    assert len(all_seeds) == len(set(all_seeds)) == 6800


def test_engine_and_firewall_are_clean() -> None:
    assert _read("SPARK2_ENGINE_QUALIFICATION.json")["classification"] == (
        "SPARK2_NATIVE_ENGINE_QUALIFIED"
    )
    assert _read("SPARK2_ENGINE_QUALIFICATION.json")["scientific_benchmark_items"] == 0
    assert _read("Q2_FIREWALL_AUDIT.json")["classification"] == "Q2_FIREWALL_CLEAN"
    assert _read("EXECUTION_AND_GOVERNANCE_LOCK.json")["authorization_state"] == (
        "DESIGN_ONLY_STAGE_A_AND_STAGE_B_NOT_AUTHORIZED"
    )


def test_principal_lock_and_artifact_hashes() -> None:
    lock = _read("PRINCIPAL_REVIEW_LOCK.json")
    assert lock["classification"] == "Q1_SECOND_TASK_DESIGN_READY_FOR_PRINCIPAL_REVIEW"
    hashes = _read("artifact_hashes.json")
    assert lock["artifact_hash_manifest"]["entries"] == len(hashes)
    for relative, expected in hashes.items():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert digest == expected
