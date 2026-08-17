"""Crash-safe physical-generation journal for Q1 V3 Stage A.

The journal is deliberately keyed by the scientific physical unit
``latent_id x rollout_index``.  It stores the complete max-budget trajectory
and all budget-derived rows, so a process crash cannot force behavioral
retries or silently mix incompatible executions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest

JOURNAL_VERSION = "q1-v3-physical-journal-v1"


def physical_key(row: dict[str, Any]) -> tuple[str, int]:
    """Return the Stage-A resume key for one physical trajectory row."""

    latent_id = str(row.get("latent_id", ""))
    if not latent_id:
        raise ValueError("journal row is missing latent_id")
    try:
        rollout_index = int(row["rollout_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("journal row has an invalid rollout_index") from exc
    return latent_id, rollout_index


def identity_hash(identity: dict[str, Any]) -> str:
    return stable_digest("Q1-V3-PHYSICAL-JOURNAL-IDENTITY", canonical_json(identity))


class PhysicalGenerationJournal:
    """Append-only journal with conservative recovery and provenance checks."""

    def __init__(self, path: str | Path, *, identity: dict[str, Any]) -> None:
        self.path = Path(path)
        self.identity = dict(identity)
        self.identity_digest = identity_hash(self.identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[tuple[str, int], dict[str, Any]] = {}
        self.quarantined_tail: str | None = None
        self._load()

    @property
    def rows(self) -> dict[tuple[str, int], dict[str, Any]]:
        return dict(self._rows)

    def has(self, key: tuple[str, int]) -> bool:
        return key in self._rows

    def get(self, key: tuple[str, int]) -> dict[str, Any] | None:
        return self._rows.get(key)

    def append(self, row: dict[str, Any]) -> None:
        """Persist one complete physical trajectory before returning."""

        key = physical_key(row)
        wrapped = {
            "journal_version": JOURNAL_VERSION,
            "identity": self.identity,
            "identity_hash": self.identity_digest,
            "physical_key": [key[0], key[1]],
            "trajectory": row,
        }
        existing = self._rows.get(key)
        if existing is not None:
            if canonical_json(existing) != canonical_json(row):
                raise ValueError(f"conflicting duplicate physical journal key: {key}")
            return
        encoded = (json.dumps(wrapped, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[key] = dict(row)

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw:
            return
        lines = raw.splitlines(keepends=True)
        valid: list[bytes] = []
        rewrite_needed = False
        for index, line in enumerate(lines):
            complete = line.endswith(b"\n")
            try:
                wrapped = json.loads(line.decode("utf-8"))
                self._validate_wrapper(wrapped)
                trajectory = wrapped["trajectory"]
                key = physical_key(trajectory)
                if key in self._rows:
                    raise ValueError(f"duplicate physical journal key: {key}")
                self._rows[key] = trajectory
                valid.append(line)
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                if not isinstance(exc, (UnicodeDecodeError, json.JSONDecodeError)):
                    raise
                is_tail = index == len(lines) - 1
                if not is_tail:
                    raise ValueError(f"invalid non-final physical journal row {index}") from exc
                digest = hashlib.sha256(line).hexdigest()[:16]
                quarantine = self.path.with_name(f"{self.path.name}.truncated.{digest}")
                quarantine.write_bytes(line)
                self.quarantined_tail = str(quarantine)
                break
            if not complete:
                # The JSON object itself is complete; only the final newline
                # was lost.  Normalize it so the next append cannot concatenate
                # two records, without treating a valid row as corrupted.
                valid[-1] = line + b"\n"
                rewrite_needed = True
                break
        if self.quarantined_tail is not None or rewrite_needed:
            temporary = self.path.with_suffix(self.path.suffix + ".recovered")
            temporary.write_bytes(b"".join(valid))
            temporary.replace(self.path)

    def _validate_wrapper(self, wrapped: Any) -> None:
        if not isinstance(wrapped, dict):
            raise ValueError("journal row must be an object")
        if wrapped.get("journal_version") != JOURNAL_VERSION:
            raise ValueError("unsupported physical journal version")
        if wrapped.get("identity_hash") != self.identity_digest:
            raise ValueError("physical journal provenance does not match this run")
        if wrapped.get("identity") != self.identity:
            raise ValueError("physical journal identity does not match this run")
        trajectory = wrapped.get("trajectory")
        if not isinstance(trajectory, dict):
            raise ValueError("journal trajectory must be an object")
        expected = wrapped.get("physical_key")
        actual = list(physical_key(trajectory))
        if expected != actual:
            raise ValueError("physical journal key does not match trajectory")
