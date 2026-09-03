#!/usr/bin/env python3
"""Independent model-free audit of the Q1 second-task prospective design."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def schedule_checks(path: Path, *, items: int, conditions: int, rollouts: int) -> dict[str, Any]:
    rows = read_json(path)
    keys = {
        (
            str(row["stage"]),
            str(row["item_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in rows
    }
    seeds = {int(row["seed"]) for row in rows}
    expected = items * conditions * rollouts
    return {
        "rows": len(rows),
        "expected_rows": expected,
        "unique_logical_keys": len(keys),
        "unique_seeds": len(seeds),
        "pass": len(rows) == expected == len(keys) == len(seeds),
    }


def main() -> int:
    protocol = read_json(REVIEW / "PROTOCOL_LOCK.json")
    instrument = read_json(REVIEW / "MODEL_FREE_INSTRUMENT_AUDIT.json")
    engine = read_json(REVIEW / "SPARK2_ENGINE_QUALIFICATION.json")
    firewall = read_json(REVIEW / "Q2_FIREWALL_AUDIT.json")
    controller = read_json(REVIEW / "CONTROLLER_PROVENANCE_LOCK.json")
    random_bank = read_json(REVIEW / "RANDOM_BANK_LOCK.json")
    stage_a = read_json(REVIEW / "STAGE_A_MANIFEST.json")
    stage_b = read_json(REVIEW / "STAGE_B_HOLDOUT_MANIFEST.json")
    reserve = read_json(REVIEW / "RESERVE_MANIFEST.json")

    prelock_hashes = {
        name: sha256(REVIEW / name) == expected
        for name, expected in protocol["hashes"].items()
    }
    stage_a_questions = {row["question_id"] for row in stage_a["ordered_records"]}
    stage_b_questions = {row["question_id"] for row in stage_b["ordered_records"]}
    reserve_questions = {row["question_id"] for row in reserve["ordered_records"]}
    split_disjoint = not (
        stage_a_questions & stage_b_questions
        or stage_a_questions & reserve_questions
        or stage_b_questions & reserve_questions
    )
    item_hashes = [
        row["item_sha256"]
        for manifest in (stage_a, stage_b, reserve)
        for row in manifest["ordered_records"]
    ]
    split_complete = len(item_hashes) == 442 == len(set(item_hashes))

    fixed_path = ROOT / controller["vector_path"]
    fixed_vector = np.load(fixed_path, allow_pickle=False).astype(np.float64)
    fixed_identity = (
        sha256(fixed_path) == controller["vector_file_sha256"]
        and vector_sha256(fixed_vector) == controller["vector_hash"]
    )
    vectors = [fixed_vector]
    null_hashes = {}
    for name in q1s.RANDOM_NAMES:
        record = random_bank["records"][name]
        path = ROOT / record["vector_path"]
        value = np.load(path, allow_pickle=False).astype(np.float64)
        valid = (
            sha256(path) == record["file_sha256"]
            and vector_sha256(value) == record["canonical_float64_vector_sha256"]
        )
        null_hashes[name] = valid
        vectors.append(value)
    gram_error = float(np.max(np.abs(np.stack(vectors) @ np.stack(vectors).T - np.eye(9))))

    schedule_a = schedule_checks(
        REVIEW / "STAGE_A_SCHEDULE.json", items=50, conditions=2, rollouts=2
    )
    schedule_b = schedule_checks(
        REVIEW / "STAGE_B_SCHEDULE.json", items=150, conditions=11, rollouts=4
    )
    cross_stage_seeds = {
        int(row["seed"])
        for filename in ("STAGE_A_SCHEDULE.json", "STAGE_B_SCHEDULE.json")
        for row in read_json(REVIEW / filename)
    }
    seed_total = schedule_a["rows"] + schedule_b["rows"]

    checks = {
        "prelock_hashes": all(prelock_hashes.values()),
        "model_free_instrument": instrument["classification"]
        == "LIVECODEBENCH_OUTPUT_INSTRUMENT_MODEL_FREE_PASS",
        "spark2_engine": engine["classification"] == "SPARK2_NATIVE_ENGINE_QUALIFIED",
        "engine_zero_scientific_items": engine["scientific_benchmark_items"] == 0,
        "controller_identity": fixed_identity,
        "null_hashes": all(null_hashes.values()),
        "null_orthogonality": gram_error <= 1e-6,
        "question_split_disjoint": split_disjoint,
        "item_pool_complete": split_complete,
        "stage_a_schedule": schedule_a["pass"],
        "stage_b_schedule": schedule_b["pass"],
        "cross_stage_seed_uniqueness": len(cross_stage_seeds) == seed_total,
        "q2_firewall": firewall["classification"] == "Q2_FIREWALL_CLEAN",
        "no_stage_authorization": read_json(REVIEW / "EXECUTION_AND_GOVERNANCE_LOCK.json")[
            "authorization_state"
        ]
        == "DESIGN_ONLY_STAGE_A_AND_STAGE_B_NOT_AUTHORIZED",
    }
    passed = all(checks.values())
    result = {
        "classification": (
            "Q1_SECOND_TASK_DESIGN_FORENSIC_CLEAN"
            if passed
            else "Q1_SECOND_TASK_DESIGN_FORENSIC_CONCERN"
        ),
        "pass": passed,
        "checks": checks,
        "prelock_hash_checks": prelock_hashes,
        "null_hash_checks": null_hashes,
        "maximum_null_gram_error": gram_error,
        "stage_a_schedule": schedule_a,
        "stage_b_schedule": schedule_b,
        "total_frozen_rows": seed_total,
        "semantic_benchmark_outcomes": 0,
        "correctness_inspected": False,
        "spark1_used": False,
        "runpod_used": False,
    }
    write_json(REVIEW / "DESIGN_FORENSIC_AUDIT.json", result)
    if not passed:
        raise RuntimeError("Q1 second-task design forensic concern")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
