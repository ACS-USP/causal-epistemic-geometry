#!/usr/bin/env python3
"""Pure validation boundary for the frozen budget-interaction lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (  # noqa: E402
    identity_hash as journal_identity_hash,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_runner import (  # noqa: E402
    candidate_identity_hash,
    validate_candidate_identity,
)
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402
from epistemic_geometry.steering.vector import vector_hash  # noqa: E402


class LockValidationError(ValueError):
    """Raised when a frozen lock or one of its inputs is not trustworthy."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise LockValidationError(f"could not read locked file: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockValidationError(f"could not read JSON lock input: {path}") from exc


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LockValidationError(f"{label} must be a JSON object")
    return value


def _vector_path(raw_path: str, lock_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    # The lock-local path supports compact synthetic fixtures.  The fallback
    # resolves the repository-relative path used by the frozen Q1 lock.
    local = lock_path.parent / path
    if local.is_file():
        return local
    return ROOT / path


def _load_vector_values(path: Path) -> np.ndarray:
    if not path.is_file():
        raise LockValidationError(f"locked vector file is missing: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            values = np.asarray(loaded, dtype=np.float64)
        else:
            try:
                if "values" not in loaded:
                    raise LockValidationError("locked vector archive lacks a values array")
                values = np.asarray(loaded["values"], dtype=np.float64)
            finally:
                loaded.close()
    except LockValidationError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise LockValidationError(f"could not load locked vector: {path}") from exc
    values = values.reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise LockValidationError("locked vector must contain finite values")
    return values


def _verify_artifacts(lock: Mapping[str, Any], lock_path: Path) -> dict[str, str]:
    expected = lock.get("artifact_sha256")
    if not isinstance(expected, Mapping) or not expected:
        raise LockValidationError("LOCK.json artifact_sha256 must be a non-empty object")
    observed: dict[str, str] = {}
    for name, digest in expected.items():
        if not isinstance(name, str) or Path(name).name != name:
            raise LockValidationError(f"invalid artifact name in lock: {name!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise LockValidationError(f"invalid artifact SHA-256 in lock: {name}")
        path = lock_path.parent / name
        if not path.is_file():
            raise LockValidationError(f"locked artifact is missing: {path}")
        observed[name] = _sha256(path)
        if observed[name] != digest:
            raise LockValidationError(f"locked artifact SHA-256 mismatch: {name}")
    return observed


def load_and_validate_lock(lock_path: str | Path) -> dict[str, Any]:
    """Load and verify one frozen lock without model/backend construction."""

    lock_file = Path(lock_path).expanduser().resolve()
    lock = _mapping(_read_json(lock_file), "LOCK.json")
    artifact_sha256 = _verify_artifacts(lock, lock_file)

    manifest = _read_json(lock_file.parent / "MANIFEST.json")
    schedule = _read_json(lock_file.parent / "SCHEDULE.json")
    provenance = _mapping(_read_json(lock_file.parent / "PROVENANCE.json"), "PROVENANCE.json")
    if not isinstance(manifest, list) or not isinstance(schedule, list):
        raise LockValidationError("MANIFEST.json and SCHEDULE.json must contain arrays")

    candidate = validate_candidate_identity(
        _mapping(lock.get("candidate_identity"), "candidate_identity")
    )
    expected_candidate_hash = lock.get("candidate_identity_hash")
    if candidate_identity_hash(candidate) != expected_candidate_hash:
        raise LockValidationError("candidate identity hash does not match LOCK.json")
    candidate_file = lock_file.parent / "CANDIDATE_IDENTITY.json"
    if candidate_file.exists() and _read_json(candidate_file) != candidate:
        raise LockValidationError("CANDIDATE_IDENTITY.json does not match LOCK.json")

    journal_identity = _mapping(lock.get("journal_identity"), "journal_identity")
    expected_journal_hash = lock.get("journal_identity_hash")
    if journal_identity_hash(journal_identity) != expected_journal_hash:
        raise LockValidationError("journal identity hash does not match LOCK.json")
    if journal_identity.get("candidate_identity_hash") != expected_candidate_hash:
        raise LockValidationError("journal identity candidate hash does not match lock")

    if journal_identity.get("manifest_sha256") != artifact_sha256.get("MANIFEST.json"):
        raise LockValidationError("journal identity manifest SHA-256 does not match lock")
    if journal_identity.get("schedule_sha256") != artifact_sha256.get("SCHEDULE.json"):
        raise LockValidationError("journal identity schedule SHA-256 does not match lock")

    if manifest and all(isinstance(row, Mapping) and "identity_hash" in row for row in manifest):
        manifest_hash = stable_digest(
            "Q1-V3-BUDGET-CAUSAL-INTERACTION-V1",
            "MANIFEST",
            *(row["identity_hash"] for row in manifest),
        )
        if manifest_hash != journal_identity.get("manifest_hash"):
            raise LockValidationError("recomputed manifest hash does not match lock")
    schedule_digest = stable_digest(
        "Q1-V3-BUDGET-CAUSAL-INTERACTION-V1", "SCHEDULE", canonical_json(schedule)
    )
    if schedule_digest != journal_identity.get("schedule_digest"):
        raise LockValidationError("recomputed schedule digest does not match lock")

    vector_path = _vector_path(candidate["vector_path"], lock_file)
    if _sha256(vector_path) != candidate["vector_file_sha256"]:
        raise LockValidationError("vector file SHA-256 does not match candidate identity")
    vector_values = _load_vector_values(vector_path)
    if vector_hash(vector_values) != candidate["vector_canonical_sha256"]:
        raise LockValidationError("vector canonical SHA-256 does not match candidate identity")

    return {
        "lock": lock,
        "manifest": manifest,
        "schedule": schedule,
        "provenance": provenance,
        "candidate_identity": candidate,
        "journal_identity": journal_identity,
        "artifact_sha256": artifact_sha256,
        "vector_path": vector_path,
        "vector_values": vector_values,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a frozen budget-interaction lock.")
    parser.add_argument(
        "lock_path",
        nargs="?",
        type=Path,
        default=ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json",
    )
    parser.add_argument("--lock", dest="lock_option", type=Path, help="LOCK.json path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_and_validate_lock(args.lock_option or args.lock_path)
    print("LOCK_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
