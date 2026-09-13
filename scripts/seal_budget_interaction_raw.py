#!/usr/bin/env python3
"""Seal a completed budget-interaction journal before outcome analysis.

This is deliberately a raw-persistence operation.  It validates the frozen
lock, the complete logical schedule, and every journal wrapper, but it never
parses a response, computes correctness, or reads an outcome field.  The
resulting ``RAW_SEAL.json`` contains only provenance, counts, and file hashes.
"""

from __future__ import annotations

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

from epistemic_geometry.benchmarks.reasoning.budget_interaction import (  # noqa: E402
    validate_schedule,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (  # noqa: E402
    BudgetInteractionJournal,
    physical_key,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (  # noqa: E402
    identity_hash as journal_identity_hash,
)
from epistemic_geometry.benchmarks.reasoning.rollouts import generation_config_hash  # noqa: E402
from scripts.run_budget_interaction_collection import (  # noqa: E402
    LockValidationError,
    load_and_validate_lock,
)


class RawSealError(ValueError):
    """Raised when a completed raw journal cannot be sealed safely."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise RawSealError(f"could not read journal: {path}") from exc


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RawSealError(f"refusing to overwrite existing raw seal: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise RawSealError(f"temporary raw seal path already exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as exc:
        raise RawSealError(f"could not write raw seal: {path}") from exc


def _validate_record_metadata(
    schedule_row: Mapping[str, Any],
    record: Mapping[str, Any],
    candidate_hash: str,
    expected_controller: Mapping[str, Any],
) -> None:
    """Validate execution provenance without touching scientific outcomes."""

    metadata = record.get("metadata")
    config = record.get("generation_config")
    if not isinstance(metadata, Mapping) or not isinstance(config, Mapping):
        raise RawSealError("journal record lacks required execution metadata")
    if metadata.get("candidate_identity_hash") != candidate_hash:
        raise RawSealError("journal record candidate identity hash mismatch")
    if metadata.get("seed_regime") != schedule_row.get("seed_regime"):
        raise RawSealError("journal record seed regime mismatch")
    if config.get("surface") != schedule_row.get("surface"):
        raise RawSealError("journal record generation surface mismatch")
    if config.get("condition") != schedule_row.get("condition"):
        raise RawSealError("journal record generation condition mismatch")
    if config.get("max_new_tokens") != schedule_row.get("cap"):
        raise RawSealError("journal record generation cap mismatch")
    if config.get("sampling_seed") != schedule_row.get("sampling_seed"):
        raise RawSealError("journal record sampling seed mismatch")
    if record.get("generation_config_hash") != generation_config_hash(dict(config)):
        raise RawSealError("journal record generation config hash mismatch")

    condition = schedule_row.get("condition")
    controller = metadata.get("controller_provenance")
    if condition == "D75":
        if not isinstance(controller, Mapping):
            raise RawSealError("D75 journal record lacks controller provenance")
        if dict(controller) != dict(expected_controller):
            raise RawSealError("D75 journal record controller provenance mismatch")
    elif condition == "BASELINE" and controller is not None:
        raise RawSealError("baseline journal record carries controller provenance")


def validate_completed_raw_journal(
    lock_path: str | Path,
    journal_path: str | Path,
) -> dict[str, Any]:
    """Validate the complete journal against the frozen lock and schedule.

    The returned payload is safe to serialize: it includes no response text or
    scientific outcome values.  ``load_and_validate_lock`` verifies the frozen
    artifacts, candidate identity, vector bytes, and journal identity hash.
    """

    try:
        loaded = load_and_validate_lock(lock_path)
    except (LockValidationError, OSError, ValueError, TypeError) as exc:
        raise RawSealError("frozen lock validation failed") from exc

    lock = loaded["lock"]
    manifest = loaded["manifest"]
    schedule = loaded["schedule"]
    journal_identity = loaded["journal_identity"]
    expected_calls = lock.get("design", {}).get("calls")
    if expected_calls != 768:
        raise RawSealError("frozen lock does not declare the 768-call design")
    if not isinstance(schedule, Sequence) or len(schedule) != expected_calls:
        raise RawSealError("frozen schedule call count is not 768")
    try:
        validate_schedule(schedule, manifest)
    except (TypeError, ValueError) as exc:
        raise RawSealError("frozen schedule validation failed") from exc

    journal = Path(journal_path).expanduser().resolve()
    if not journal.is_file():
        raise RawSealError(f"journal is missing: {journal}")
    before_hash = _sha256_file(journal)
    raw = journal.read_bytes()
    if not raw:
        raise RawSealError("journal is empty")
    # The journal loader intentionally recovers an unterminated final tail for
    # resume.  A seal must refuse that recovery path and preserve bytes.
    if not raw.endswith(b"\n"):
        raise RawSealError("journal has an unterminated final row")

    try:
        loaded_journal = BudgetInteractionJournal(journal, identity=journal_identity)
    except (OSError, TypeError, ValueError) as exc:
        raise RawSealError("journal structural validation failed") from exc
    after_hash = _sha256_file(journal)
    if after_hash != before_hash:
        raise RawSealError("journal changed during raw validation")
    if loaded_journal.quarantined_tail is not None:
        raise RawSealError("journal loader quarantined a tail during sealing")

    expected_keys = {physical_key(row) for row in schedule}
    observed_keys = set(loaded_journal.entries)
    if len(expected_keys) != expected_calls:
        raise RawSealError("frozen schedule does not contain 768 unique logical keys")
    if len(observed_keys) != expected_calls:
        raise RawSealError("journal does not contain 768 unique logical keys")
    missing = expected_keys - observed_keys
    extra = observed_keys - expected_keys
    if missing or extra:
        raise RawSealError("journal logical keys do not exactly match frozen schedule")

    by_key = {physical_key(row): row for row in schedule}
    for key, entry in loaded_journal.entries.items():
        expected = by_key.get(key)
        if expected is None or entry.get("schedule") != expected:
            raise RawSealError("journal schedule wrapper does not match frozen schedule")
        record = entry.get("record")
        if not isinstance(record, Mapping):
            raise RawSealError("journal record is not an object")
        _validate_record_metadata(
            expected,
            record,
            lock["candidate_identity_hash"],
            lock.get("controller_provenance", {}),
        )

    # Use the lock's identity digest as the provenance binding.  The journal
    # constructor has already checked each wrapper's actual identity hash.
    expected_identity_hash = journal_identity_hash(journal_identity)
    if expected_identity_hash != lock.get("journal_identity_hash"):
        raise RawSealError("journal identity hash does not match frozen lock")

    return {
        "schema_version": "q1-v3-budget-interaction-raw-seal-v1",
        "status": "RAW_JOURNAL_SEALED",
        "experiment_id": journal_identity.get("experiment_id"),
        "lock_sha256": _sha256_file(Path(lock_path).expanduser().resolve()),
        "manifest_sha256": journal_identity.get("manifest_sha256"),
        "schedule_sha256": journal_identity.get("schedule_sha256"),
        "schedule_digest": journal_identity.get("schedule_digest"),
        "candidate_identity_hash": lock.get("candidate_identity_hash"),
        "journal_identity_hash": lock.get("journal_identity_hash"),
        "journal_path": str(journal),
        "journal_sha256": before_hash,
        "journal_bytes": len(raw),
        "logical_key_count": len(observed_keys),
        "expected_logical_key_count": expected_calls,
        "scientific_model_outcomes_computed": False,
        "response_text_exposed": False,
        "parse_or_correctness_evaluated": False,
        "aggregates_computed": False,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seal a completed raw budget-interaction journal")
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json",
    )
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = validate_completed_raw_journal(args.lock, args.journal)
    _write_immutable_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "logical_key_count": 768}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
