#!/usr/bin/env python3
"""Seal the principal-review lock for the Q1 second-task design."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    engine = read_json(REVIEW / "SPARK2_ENGINE_QUALIFICATION.json")
    audit = read_json(REVIEW / "DESIGN_FORENSIC_AUDIT.json")
    instrument = read_json(REVIEW / "MODEL_FREE_INSTRUMENT_AUDIT.json")
    if instrument["classification"] != "LIVECODEBENCH_OUTPUT_INSTRUMENT_MODEL_FREE_PASS":
        raise RuntimeError("model-free instrument is not qualified")
    if engine["classification"] != "SPARK2_NATIVE_ENGINE_QUALIFIED":
        raise RuntimeError("Spark-2 engine is not qualified")
    if audit["classification"] != "Q1_SECOND_TASK_DESIGN_FORENSIC_CLEAN":
        raise RuntimeError("independent design audit is not clean")

    review_files = sorted(
        path
        for path in REVIEW.rglob("*")
        if path.is_file()
        and path.name not in {"artifact_hashes.json", "PRINCIPAL_REVIEW_LOCK.json"}
    )
    code_files = [
        ROOT / "experiments/specs/q1_second_task_spark2_design.yaml",
        ROOT / "scripts/audit_q1_second_task_design.py",
        ROOT / "scripts/finalize_q1_second_task_design.py",
        ROOT / "scripts/prepare_q1_second_task_design.py",
        ROOT / "scripts/qualify_q1_second_task_spark2.py",
        ROOT / "scripts/simulate_q1_second_task_power.py",
        ROOT / "src/epistemic_geometry/experiments/q1_second_task.py",
        ROOT / "src/epistemic_geometry/experiments/q1_second_task_power.py",
        ROOT / "tests/test_q1_second_task.py",
        ROOT / "tests/test_q1_second_task_design_artifacts.py",
        ROOT / "tests/test_q1_second_task_power.py",
    ]
    hashes = {
        str(path.relative_to(ROOT)): sha256(path) for path in [*review_files, *code_files]
    }
    write_json(REVIEW / "artifact_hashes.json", hashes)
    lock = {
        "schema_version": 1,
        "classification": "Q1_SECOND_TASK_DESIGN_READY_FOR_PRINCIPAL_REVIEW",
        "prepared_at_parent_commit": git_commit(),
        "engine_source_commit": engine["source_commit"],
        "instrument": instrument["classification"],
        "engine": engine["classification"],
        "forensic": audit["classification"],
        "backend_claim": "SPARK2_NATIVE_CROSS_BACKEND_REPLICATION",
        "stage_a": {"n": 50, "rollouts": 2, "conditions": 2, "logical_rows": 200},
        "stage_b": {"n": 150, "rollouts": 4, "conditions": 11, "logical_rows": 6600},
        "reserve_items": 242,
        "stage_a_authorized": False,
        "stage_b_authorized": False,
        "scientific_benchmark_outcomes": 0,
        "correctness_inspected": False,
        "q2_outputs_inspected": False,
        "q2_process_modified": False,
        "q3": "NOT_RUN",
        "artifact_hash_manifest": {
            "path": "review/q1_second_task_spark2_design/artifact_hashes.json",
            "sha256": sha256(REVIEW / "artifact_hashes.json"),
            "entries": len(hashes),
        },
    }
    write_json(REVIEW / "PRINCIPAL_REVIEW_LOCK.json", lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
