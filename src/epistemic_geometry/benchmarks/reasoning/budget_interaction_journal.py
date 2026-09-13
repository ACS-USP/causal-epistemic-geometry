"""Crash safe journal for the prospective budget by D75 schedule.

The Stage A physical journal in :mod:`reasoning.journal` predates this study
and is keyed only by ``latent_id`` and ``rollout_index``.  That key is unsafe
here: the same latent and rollout occur at two caps and under two conditions.
This journal therefore stores the complete parsed rollout and uses the full
logical schedule key.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest

from .budget_interaction import NAMESPACE
from .rollouts import RolloutRecord

JOURNAL_VERSION = "q1-v3-budget-interaction-journal-v1"


def identity_hash(identity: Mapping[str, Any]) -> str:
    """Hash the caller supplied identity without dropping any fields."""

    return stable_digest("Q1-V3-BUDGET-INTERACTION-JOURNAL-IDENTITY", canonical_json(identity))


def schedule_identity_hash(row: Mapping[str, Any]) -> str:
    """Return the deterministic identity for one logical schedule row."""

    return stable_digest(
        NAMESPACE,
        "SCHEDULE",
        row.get("latent_id"),
        row.get("cap"),
        row.get("condition"),
        row.get("rollout_index"),
    )


def physical_key(row: Mapping[str, Any]) -> tuple[str, int, str, int]:
    """Return the physical key including cap and condition."""

    if not isinstance(row, Mapping):
        raise TypeError("journal rows must be mappings")
    latent_id = row.get("latent_id")
    cap = row.get("cap")
    condition = row.get("condition")
    rollout_index = row.get("rollout_index")
    if not isinstance(latent_id, str) or not latent_id:
        raise ValueError("journal row is missing latent_id")
    if isinstance(cap, bool) or not isinstance(cap, int):
        raise ValueError("journal row has an invalid cap")
    if not isinstance(condition, str) or not condition:
        raise ValueError("journal row is missing condition")
    if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
        raise ValueError("journal row has an invalid rollout_index")
    return latent_id, cap, condition, rollout_index


schedule_key = physical_key


def _record_dict(record: RolloutRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, RolloutRecord):
        return record.to_record()
    if not isinstance(record, Mapping):
        raise TypeError("journal record must be a mapping or RolloutRecord")
    return copy.deepcopy(dict(record))


def _schedule_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError("schedule row must be a mapping")
    return copy.deepcopy(dict(row))


class BudgetInteractionJournal:
    """Single process, append only JSONL journal with conservative recovery."""

    def __init__(self, path: str | Path, *, identity: dict[str, Any]) -> None:
        if not isinstance(identity, dict):
            raise TypeError("journal identity must be a dict")
        self.path = Path(path)
        self.identity = copy.deepcopy(identity)
        self.identity_digest = identity_hash(self.identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[tuple[str, int, str, int], dict[str, Any]] = {}
        self.quarantined_tail: str | None = None
        self._load()

    @property
    def rows(self) -> dict[tuple[str, int, str, int], dict[str, Any]]:
        """Return complete rollout records indexed by physical key."""

        return {key: copy.deepcopy(entry["record"]) for key, entry in self._entries.items()}

    @property
    def entries(self) -> dict[tuple[str, int, str, int], dict[str, Any]]:
        return copy.deepcopy(self._entries)

    def has(self, key: tuple[str, int, str, int]) -> bool:
        return key in self._entries

    def get(self, key: tuple[str, int, str, int]) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        return copy.deepcopy(entry["record"]) if entry is not None else None

    def get_entry(self, key: tuple[str, int, str, int]) -> dict[str, Any] | None:
        entry = self._entries.get(key)
        return copy.deepcopy(entry) if entry is not None else None

    def get_for_schedule(self, schedule: Mapping[str, Any]) -> dict[str, Any] | None:
        """Look up a row after checking its schedule identity and key."""

        self._validate_schedule(schedule)
        key = physical_key(schedule)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._validate_entry(entry)
        if entry["schedule"] != dict(schedule):
            raise ValueError(f"journal schedule mismatch for physical key: {key}")
        return copy.deepcopy(entry["record"])

    def append(
        self,
        schedule: Mapping[str, Any],
        record: RolloutRecord | Mapping[str, Any] | None = None,
    ) -> None:
        """Append a complete parsed record and fsync before returning.

        For convenience, a single full row may be supplied when it contains
        the schedule fields.  The two argument form is preferred because it
        keeps the frozen schedule and model result visibly separate.
        """

        if record is None and isinstance(schedule, Mapping) and "record" in schedule:
            wrapper = schedule
            record = wrapper["record"]
            schedule = wrapper.get("schedule", schedule)
        schedule_row = _schedule_dict(schedule)
        if record is None:
            record_row = _record_dict(schedule_row)
        else:
            record_row = _record_dict(record)
        self._validate_schedule(schedule_row)
        self._validate_record(schedule_row, record_row)
        key = physical_key(schedule_row)
        existing = self._entries.get(key)
        if existing is not None:
            if (
                canonical_json(existing["schedule"]) != canonical_json(schedule_row)
                or canonical_json(existing["record"]) != canonical_json(record_row)
            ):
                raise ValueError(f"conflicting duplicate budget journal key: {key}")
            return
        wrapped = {
            "journal_version": JOURNAL_VERSION,
            "identity": copy.deepcopy(self.identity),
            "identity_hash": self.identity_digest,
            "physical_key": list(key),
            "schedule_identity_hash": schedule_row["schedule_identity_hash"],
            "schedule": schedule_row,
            "record": record_row,
        }
        encoded = (json.dumps(wrapped, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._entries[key] = {"schedule": schedule_row, "record": record_row}

    def _validate_schedule(self, schedule: Mapping[str, Any]) -> None:
        key = physical_key(schedule)
        expected = schedule_identity_hash(schedule)
        if schedule.get("schedule_identity_hash") != expected:
            raise ValueError("schedule identity hash mismatch")
        if tuple(key) != tuple(physical_key(schedule)):
            raise ValueError("schedule physical key mismatch")

    def _validate_record(self, schedule: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        key = physical_key(schedule)
        metadata = record.get("metadata")
        generation_config = record.get("generation_config")
        if not isinstance(metadata, Mapping):
            raise ValueError("journaled rollout metadata must be an object")
        if not isinstance(generation_config, Mapping):
            raise ValueError("journaled rollout generation_config must be an object")
        if record.get("latent_id") != key[0] or record.get("rollout_index") != key[3]:
            raise ValueError("journaled rollout key does not match schedule")
        if (
            record.get("family") != schedule.get("family")
            or record.get("cell") != schedule.get("cell")
        ):
            raise ValueError("journaled rollout manifest identity does not match schedule")
        if record.get("sampling_seed") != schedule.get("sampling_seed"):
            raise ValueError("journaled rollout seed does not match schedule")
        expected_metadata = {
            "cap": key[1],
            "reasoning_budget": key[1],
            "condition": key[2],
            "schedule_identity_hash": schedule["schedule_identity_hash"],
        }
        for field, value in expected_metadata.items():
            if metadata.get(field) != value:
                raise ValueError(f"journaled rollout {field} does not match schedule")
        if (
            generation_config.get("max_new_tokens") != key[1]
            or generation_config.get("condition") != key[2]
        ):
            raise ValueError("journaled rollout generation config does not match schedule")
        if record.get("physical_generation_id") is not None:
            expected_physical_id = stable_digest(
                NAMESPACE, "PHYSICAL-GENERATION", schedule["schedule_identity_hash"]
            )
            if record["physical_generation_id"] != expected_physical_id:
                raise ValueError("journaled rollout physical identity mismatch")

    def _validate_entry(self, entry: Mapping[str, Any]) -> None:
        schedule = entry.get("schedule")
        record = entry.get("record")
        if not isinstance(schedule, Mapping) or not isinstance(record, Mapping):
            raise ValueError("budget interaction journal entry is malformed")
        self._validate_schedule(schedule)
        self._validate_record(schedule, record)

    def _validate_wrapper(self, wrapped: Any) -> dict[str, Any]:
        if not isinstance(wrapped, dict):
            raise ValueError("journal row must be an object")
        if wrapped.get("journal_version") != JOURNAL_VERSION:
            raise ValueError("unsupported budget interaction journal version")
        if wrapped.get("identity_hash") != self.identity_digest:
            raise ValueError("budget interaction journal provenance does not match this run")
        if wrapped.get("identity") != self.identity:
            raise ValueError("budget interaction journal identity does not match this run")
        schedule = wrapped.get("schedule")
        record = wrapped.get("record")
        if not isinstance(schedule, dict) or not isinstance(record, dict):
            raise ValueError("budget interaction journal schedule and record must be objects")
        key = physical_key(schedule)
        if wrapped.get("physical_key") != list(key):
            raise ValueError("budget interaction journal physical key mismatch")
        if wrapped.get("schedule_identity_hash") != schedule.get("schedule_identity_hash"):
            raise ValueError("budget interaction journal schedule identity mismatch")
        self._validate_schedule(schedule)
        self._validate_record(schedule, record)
        return {"schedule": schedule, "record": record}

    def _load(self) -> None:
        if not self.path.exists() or not self.path.stat().st_size:
            return
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid: list[bytes] = []
        rewrite_needed = False
        for index, line in enumerate(lines):
            complete = line.endswith(b"\n")
            try:
                wrapped = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index != len(lines) - 1:
                    raise ValueError(f"invalid non-final budget journal row {index}") from exc
                digest = hashlib.sha256(line).hexdigest()[:16]
                quarantine = self.path.with_name(f"{self.path.name}.truncated.{digest}")
                quarantine.write_bytes(line)
                self.quarantined_tail = str(quarantine)
                break
            entry = self._validate_wrapper(wrapped)
            key = physical_key(entry["schedule"])
            if key in self._entries:
                previous = self._entries[key]
                if canonical_json(previous) != canonical_json(entry):
                    raise ValueError(f"conflicting duplicate budget journal key: {key}")
                raise ValueError(f"duplicate budget journal key: {key}")
            self._entries[key] = entry
            valid.append(line if complete else line + b"\n")
            if not complete:
                rewrite_needed = True
                break
        if self.quarantined_tail is not None or rewrite_needed:
            temporary = self.path.with_suffix(self.path.suffix + ".recovered")
            with temporary.open("wb") as handle:
                handle.write(b"".join(valid))
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.path)


BudgetInteractionRolloutJournal = BudgetInteractionJournal

__all__ = [
    "BudgetInteractionJournal",
    "BudgetInteractionRolloutJournal",
    "JOURNAL_VERSION",
    "identity_hash",
    "physical_key",
    "schedule_key",
    "schedule_identity_hash",
]
