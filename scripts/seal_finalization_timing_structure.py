#!/usr/bin/env python3
"""Seal structural Qwen finalization timing fields without semantic scoring.

This reader intentionally extracts only schedule identity and generated token
IDs from an append-only journal. It does not access raw text, parse fields,
answer keys, or correctness.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.finalization_timing import QWEN3_THINK_CLOSE_TOKEN_ID
from epistemic_geometry.benchmarks.reasoning.finalization_timing import validate_schedule
from epistemic_geometry.benchmarks.reasoning.finalization_timing_journal import (
    identity_hash,
    physical_key,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest


class StructuralSealError(ValueError):
    """Raised when a journal cannot support structural timing analysis."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StructuralSealError(f"could not read {label}: {path}") from exc


def _restricted_time(token_ids: Sequence[Any], *, cap: int) -> tuple[int, bool]:
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise StructuralSealError("journal token_ids must be a sequence")
    if len(token_ids) > cap:
        raise StructuralSealError("journal token_ids exceeds its scheduled cap")
    for index, token in enumerate(token_ids):
        if type(token) is not int or token < 0:
            raise StructuralSealError("journal token_ids must contain non-negative integers")
        if token == QWEN3_THINK_CLOSE_TOKEN_ID:
            return index + 1, True
    return cap, False


def seal_structure(
    manifest_path: str | Path,
    schedule_path: str | Path,
    journal_path: str | Path,
    journal_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a structural-only sealed payload from one complete journal."""

    manifest_file = Path(manifest_path)
    schedule_file = Path(schedule_path)
    journal_file = Path(journal_path)
    manifest = _read_json(manifest_file, "manifest")
    schedule = _read_json(schedule_file, "schedule")
    if not isinstance(manifest, list) or not isinstance(schedule, list):
        raise StructuralSealError("manifest and schedule must be JSON arrays")
    try:
        validate_schedule(schedule, manifest)
    except (TypeError, ValueError) as exc:
        raise StructuralSealError(f"invalid frozen schedule: {exc}") from exc
    if not journal_file.is_file():
        raise StructuralSealError(f"journal is not a file: {journal_file}")
    if not isinstance(journal_identity, Mapping):
        raise StructuralSealError("journal_identity must be a mapping")
    expected_identity = dict(journal_identity)
    expected_identity_hash = identity_hash(expected_identity)
    expected = {physical_key(row): row for row in schedule}
    observed: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    try:
        lines = journal_file.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise StructuralSealError(f"could not read journal: {journal_file}") from exc
    if not lines:
        raise StructuralSealError("journal is empty")
    for index, line in enumerate(lines, start=1):
        try:
            wrapper = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StructuralSealError(f"journal line {index} is invalid JSON") from exc
        if not isinstance(wrapper, Mapping):
            raise StructuralSealError(f"journal line {index} is not an object")
        if wrapper.get("identity") != expected_identity:
            raise StructuralSealError("journal identity does not match the frozen identity")
        if wrapper.get("identity_hash") != expected_identity_hash:
            raise StructuralSealError("journal identity hash does not match the frozen identity")
        schedule_row = wrapper.get("schedule")
        record = wrapper.get("record")
        if not isinstance(schedule_row, Mapping) or not isinstance(record, Mapping):
            raise StructuralSealError("journal wrapper lacks schedule or record object")
        try:
            key = physical_key(schedule_row)
        except (TypeError, ValueError) as exc:
            raise StructuralSealError("journal schedule has invalid physical key") from exc
        if key not in expected or dict(schedule_row) != expected[key]:
            raise StructuralSealError("journal schedule differs from the frozen schedule")
        if wrapper.get("physical_key") != list(key):
            raise StructuralSealError("journal wrapper physical key mismatch")
        if wrapper.get("schedule_identity_hash") != schedule_row.get("schedule_identity_hash"):
            raise StructuralSealError("journal wrapper schedule identity mismatch")
        if record.get("latent_id") != key[0] or record.get("rollout_index") != key[3]:
            raise StructuralSealError("journal record identity does not match schedule")
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise StructuralSealError("journal record lacks structural metadata")
        for field, value in {
            "cap": key[1],
            "condition": key[2],
            "schedule_identity_hash": schedule_row["schedule_identity_hash"],
        }.items():
            if metadata.get(field) != value:
                raise StructuralSealError(f"journal structural metadata mismatch: {field}")
        if key in observed:
            raise StructuralSealError("journal contains a duplicate physical key")
        restricted_time, event_observed = _restricted_time(record.get("token_ids"), cap=key[1])
        observed[key] = {
            "family": schedule_row["family"],
            "cell": schedule_row["cell"],
            "latent": key[0],
            "condition": key[2],
            "rollout": key[3],
            "cap": key[1],
            "restricted_think_close_time": restricted_time,
            "think_close_observed": event_observed,
        }
    if set(observed) != set(expected):
        raise StructuralSealError("journal does not contain exactly the frozen schedule keys")
    records = [observed[key] for key in sorted(observed)]
    return {
        "schema_version": "q1-finalization-timing-structural-seal-v1",
        "status": "STRUCTURAL_TIMING_JOURNAL_SEALED",
        "scope": "token IDs and schedule identity only; semantic outcome fields were not accessed",
        "manifest_sha256": _sha256(manifest_file),
        "schedule_sha256": _sha256(schedule_file),
        "schedule_digest": stable_digest(
            "Q1-V3-FINALIZATION-TIMING-V1", "SCHEDULE", canonical_json(schedule)
        ),
        "journal_sha256": _sha256(journal_file),
        "journal_identity_hash": expected_identity_hash,
        "close_token_id": QWEN3_THINK_CLOSE_TOKEN_ID,
        "logical_key_count": len(records),
        "records": records,
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise StructuralSealError(f"refusing to overwrite structural seal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--journal-identity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    identity = _read_json(args.journal_identity, "journal identity")
    payload = seal_structure(args.manifest, args.schedule, args.journal, identity)
    _write_new_json(args.output, payload)
    print(
        json.dumps(
            {"status": payload["status"], "logical_key_count": payload["logical_key_count"]}
        )
    )
    return 0
