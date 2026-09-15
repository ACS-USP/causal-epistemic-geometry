#!/usr/bin/env python3
"""Independently audit a sealed Q1 budget-interaction raw journal.

This audit is deliberately outcome-blind.  It verifies bytes, lock bindings,
logical schedule coverage, wrapper identity, and the minimum rehydratable
record shape.  It never scores records, parses model text, or aggregates
scientific outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class RawSealAuditError(ValueError):
    """Raised when an independently audited raw seal is inconsistent."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RawSealAuditError(f"cannot read {path}") from exc


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawSealAuditError(f"cannot parse {label}") from exc
    if not isinstance(value, dict):
        raise RawSealAuditError(f"{label} must be a JSON object")
    return value


def _key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    try:
        return (
            str(row["latent_id"]),
            int(row["cap"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RawSealAuditError("schedule row has an invalid physical key") from exc


def _require_record_shape(record: Any) -> None:
    """Check only field presence and primitive forms; never inspect outcomes."""
    if not isinstance(record, Mapping):
        raise RawSealAuditError("journal record is not an object")
    required = {
        "latent_id",
        "view_id",
        "family",
        "cell",
        "intervention_id",
        "rollout_index",
        "sampling_seed",
        "raw_text",
        "parsed_answer",
        "parse_status",
        "correct",
        "metadata",
        "generation_config",
        "generation_config_hash",
    }
    missing = sorted(required - set(record))
    if missing:
        raise RawSealAuditError(f"journal record is missing fields: {missing}")
    if not isinstance(record["raw_text"], str):
        raise RawSealAuditError("journal raw_text is not a string")
    if record["parsed_answer"] is not None and (
        isinstance(record["parsed_answer"], bool) or not isinstance(record["parsed_answer"], int)
    ):
        raise RawSealAuditError("journal parsed_answer has an invalid type")
    if not isinstance(record["parse_status"], str) or not isinstance(record["correct"], bool):
        raise RawSealAuditError("journal parse/correctness fields have invalid types")
    if not isinstance(record["metadata"], Mapping) or not isinstance(
        record["generation_config"], Mapping
    ):
        raise RawSealAuditError("journal record lacks metadata/config objects")


def audit_raw_seal(
    lock_path: str | Path,
    journal_path: str | Path,
    raw_seal_path: str | Path,
) -> dict[str, Any]:
    """Independently verify a primary raw seal without outcome analysis."""
    lock_file = Path(lock_path).expanduser().resolve()
    journal_file = Path(journal_path).expanduser().resolve()
    seal_file = Path(raw_seal_path).expanduser().resolve()
    lock = _read_object(lock_file, "lock")
    seal = _read_object(seal_file, "raw seal")
    if seal.get("status") != "RAW_JOURNAL_SEALED":
        raise RawSealAuditError("primary raw seal is not sealed")
    if seal.get("candidate_identity_hash") != lock.get("candidate_identity_hash"):
        raise RawSealAuditError("raw seal candidate identity hash mismatch")
    if seal.get("lock_sha256") != _sha256(lock_file):
        raise RawSealAuditError("raw seal lock hash mismatch")
    if seal.get("journal_sha256") != _sha256(journal_file):
        raise RawSealAuditError("raw seal journal hash mismatch")
    if seal.get("expected_logical_key_count") != 768:
        raise RawSealAuditError("raw seal does not declare 768 expected keys")

    schedule_path = lock_file.parent / "SCHEDULE.json"
    try:
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RawSealAuditError("cannot parse frozen schedule") from exc
    if (
        not isinstance(schedule, Sequence)
        or isinstance(schedule, (str, bytes))
        or len(schedule) != 768
    ):
        raise RawSealAuditError("frozen schedule is not a 768-row sequence")
    if seal.get("schedule_sha256") != _sha256(schedule_path):
        raise RawSealAuditError("raw seal schedule hash mismatch")

    expected: dict[tuple[str, int, str, int], Mapping[str, Any]] = {}
    for row in schedule:
        if not isinstance(row, Mapping):
            raise RawSealAuditError("frozen schedule contains a non-object row")
        key = _key(row)
        if key in expected:
            raise RawSealAuditError("frozen schedule has duplicate physical keys")
        expected[key] = row

    try:
        raw = journal_file.read_bytes()
    except OSError as exc:
        raise RawSealAuditError("cannot read journal") from exc
    if not raw.endswith(b"\n"):
        raise RawSealAuditError("journal has an unterminated final row")

    observed: set[tuple[str, int, str, int]] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        try:
            wrapped = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RawSealAuditError(f"journal line {line_number} is not JSON") from exc
        if not isinstance(wrapped, Mapping):
            raise RawSealAuditError(f"journal line {line_number} is not an object")
        schedule_row = wrapped.get("schedule")
        if not isinstance(schedule_row, Mapping):
            raise RawSealAuditError(f"journal line {line_number} lacks schedule")
        key = _key(schedule_row)
        expected_row = expected.get(key)
        if expected_row is None or dict(schedule_row) != dict(expected_row):
            raise RawSealAuditError(f"journal line {line_number} schedule differs from lock")
        if wrapped.get("physical_key") != list(key):
            raise RawSealAuditError(f"journal line {line_number} physical key mismatch")
        if wrapped.get("identity_hash") != seal.get("journal_identity_hash"):
            raise RawSealAuditError(f"journal line {line_number} identity hash mismatch")
        if wrapped.get("schedule_identity_hash") != schedule_row.get("schedule_identity_hash"):
            raise RawSealAuditError(f"journal line {line_number} schedule identity mismatch")
        if key in observed:
            raise RawSealAuditError("journal has duplicate physical keys")
        observed.add(key)
        _require_record_shape(wrapped.get("record"))

    if observed != set(expected):
        raise RawSealAuditError("journal physical keys do not exactly match frozen schedule")
    if seal.get("logical_key_count") != len(observed):
        raise RawSealAuditError("raw seal logical-key count mismatch")

    return {
        "schema_version": "q1-v3-budget-interaction-raw-seal-independent-audit-v1",
        "classification": "RAW_SEAL_INDEPENDENT_AUDIT_PASS",
        "primary_raw_seal_sha256": _sha256(seal_file),
        "lock_sha256": _sha256(lock_file),
        "schedule_sha256": _sha256(schedule_path),
        "journal_sha256": _sha256(journal_file),
        "logical_key_count": len(observed),
        "expected_logical_key_count": len(expected),
        "response_text_exposed": False,
        "parse_or_correctness_evaluated": False,
        "scientific_model_outcomes_computed": False,
        "aggregates_computed": False,
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RawSealAuditError(f"refusing to overwrite existing independent audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        raise RawSealAuditError("cannot write independent audit") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit a sealed raw journal")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--raw-seal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = audit_raw_seal(args.lock, args.journal, args.raw_seal)
    _write_immutable_json(args.output, payload)
    print(
        json.dumps(
            {"classification": payload["classification"], "logical_key_count": 768}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
