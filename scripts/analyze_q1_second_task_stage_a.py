#!/usr/bin/env python3
"""Seal and analyze the complete frozen Q1 LiveCodeBench Stage-A campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_a as stage_a  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
SCHEDULE = REVIEW / "STAGE_A_SCHEDULE.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_complete(raw_rows: list[dict[str, Any]]) -> None:
    schedule = read_json(SCHEDULE)
    stage_a.validate_schedule(schedule)
    expected = {stage_a.logical_key(row): row for row in schedule}
    observed = {stage_a.logical_key(row): row for row in raw_rows}
    if len(raw_rows) != 128 or len(observed) != 128 or set(observed) != set(expected):
        raise RuntimeError("Q1_SECOND_TASK_EXECUTION_INCOMPLETE")
    for key, row in observed.items():
        locked = expected[key]
        for field in ("family_id", "item_sha256", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"raw/schedule mismatch: {field}")
        if row.get("intervention") != "NONE" or row.get("activation_hook_active"):
            raise RuntimeError("unauthorized activation intervention in Stage A")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    journal = args.raw_dir / "journal.jsonl"
    execution = args.raw_dir / "EXECUTION_COMPLETE.json"
    raw_rows = read_jsonl(journal)
    validate_complete(raw_rows)
    if read_json(execution)["classification"] != "STAGE_A_COLLECTION_COMPLETE_UNANALYZED":
        raise RuntimeError("collection-complete seal missing")

    retry_path = args.raw_dir / "retry_ledger.jsonl"
    retries = read_jsonl(retry_path) if retry_path.exists() else []
    raw_seal = {
        "classification": "Q1_SECOND_TASK_STAGE_A_RAW_SEALED",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "journal_sha256": sha256(journal),
        "journal_bytes": journal.stat().st_size,
        "journal_rows": len(raw_rows),
        "execution_complete_sha256": sha256(execution),
        "retry_ledger_sha256": sha256(retry_path) if retry_path.exists() else None,
        "retry_records": len(retries),
        "schedule_sha256": sha256(SCHEDULE),
        "duplicates": 0,
        "missing": 0,
        "unexpected": 0,
        "stage_b_rows": 0,
        "activation_controller_rows": 0,
        "activation_null_rows": 0,
        "raw_records_modified": False,
        "correctness_inspected_before_seal": False,
    }
    write_json(args.analysis_dir / "RAW_DATA_SEAL.json", raw_seal)

    parsed_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        scored = q1s.evaluate_livecodebench_output(
            row["raw_output"], row["reference_answer"], truncated=bool(row["truncated"])
        )
        parsed_rows.append(
            {
                "stage": row["stage"],
                "item_id": row["item_id"],
                "family_id": row["family_id"],
                "condition": row["condition"],
                "rollout_index": row["rollout_index"],
                "seed": row["seed"],
                "generated_token_count": row["generated_token_count"],
                "truncated": row["truncated"],
                **scored,
            }
        )
    parsed_path = args.raw_dir / "parsed_records.jsonl"
    write_jsonl(parsed_path, parsed_rows)
    result = stage_a.stage_a_gate(parsed_rows)
    result.update(
        {
            "schema_version": 1,
            "analysis_timestamp_utc": datetime.now(UTC).isoformat(),
            "raw_data_seal_sha256": sha256(args.analysis_dir / "RAW_DATA_SEAL.json"),
            "parsed_records_sha256": sha256(parsed_path),
            "parsed_records_rows": len(parsed_rows),
            "stage_a_scientific_unit": "QUESTION_FAMILY",
            "families_equally_weighted": True,
            "meaningful_controller_livecodebench_trajectories": 0,
            "activation_null_livecodebench_trajectories": 0,
            "q2_outputs_inspected": False,
        }
    )
    write_json(args.analysis_dir / "PRIMARY_STAGE_A_RESULTS.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
