"""Small deterministic reliability helpers for future experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from epistemic_geometry.reproducibility import canonical_json, stable_digest


class OutputContractStatus(StrEnum):
    OK = "OK"
    FINAL_INSIDE_FENCE = "FINAL_INSIDE_FENCE"
    MULTIPLE_FINAL = "MULTIPLE_FINAL"
    MISSING_FINAL = "MISSING_FINAL"
    UNCLOSED_FENCE = "UNCLOSED_FENCE"
    TRUNCATED = "TRUNCATED"


_FINAL = re.compile(r"^\s*FINAL\s*:\s*\S.*$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```+|~~~+)")


def inspect_output_contract(raw_text: str, *, truncated: bool = False) -> OutputContractStatus:
    """Inspect only formatting mechanics; never score or interpret the answer."""

    if truncated:
        return OutputContractStatus.TRUNCATED
    fence: str | None = None
    visible_finals = 0
    fenced_final = False
    for line in raw_text.splitlines():
        marker = _FENCE.match(line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token[0]
            elif token[0] == fence:
                fence = None
            continue
        if _FINAL.match(line):
            if fence is None:
                visible_finals += 1
            else:
                fenced_final = True
    if fence is not None:
        return OutputContractStatus.UNCLOSED_FENCE
    if fenced_final:
        return OutputContractStatus.FINAL_INSIDE_FENCE
    if visible_finals > 1:
        return OutputContractStatus.MULTIPLE_FINAL
    if visible_finals == 0:
        return OutputContractStatus.MISSING_FINAL
    return OutputContractStatus.OK


def condition_formatting_differs(outputs: Mapping[str, Sequence[str]]) -> bool:
    """Detect condition-dependent parser mechanics without examining answer values."""

    signatures = {
        condition: Counter(inspect_output_contract(text).value for text in rows)
        for condition, rows in outputs.items()
    }
    return len({tuple(sorted(signature.items())) for signature in signatures.values()}) > 1


@dataclass(frozen=True)
class LogicalRowReport:
    expected: int
    observed: int
    duplicate_keys: tuple[tuple[Any, ...], ...]
    missing_keys: tuple[tuple[Any, ...], ...]
    unexpected_keys: tuple[tuple[Any, ...], ...]

    @property
    def valid(self) -> bool:
        return not (self.duplicate_keys or self.missing_keys or self.unexpected_keys)


def validate_logical_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
    expected_keys: Iterable[Sequence[Any]],
) -> LogicalRowReport:
    """Report duplicate, missing, and unexpected scientific identities."""

    materialized = list(rows)
    observed = [tuple(row[field] for field in key_fields) for row in materialized]
    counts = Counter(observed)
    expected = {tuple(key) for key in expected_keys}
    observed_set = set(observed)
    return LogicalRowReport(
        expected=len(expected),
        observed=len(observed),
        duplicate_keys=tuple(sorted(key for key, count in counts.items() if count > 1)),
        missing_keys=tuple(sorted(expected - observed_set)),
        unexpected_keys=tuple(sorted(observed_set - expected)),
    )


class CrashSafeJournal:
    """Generic append-only JSONL journal with identity and tail recovery."""

    version = "research-os-jsonl-v1"

    def __init__(
        self,
        path: str | Path,
        *,
        identity: Mapping[str, Any],
        key_fields: Sequence[str],
    ) -> None:
        self.path = Path(path)
        self.identity = dict(identity)
        self.identity_hash = stable_digest("RESEARCH-OS-JOURNAL", canonical_json(self.identity))
        self.key_fields = tuple(key_fields)
        if not self.key_fields:
            raise ValueError("journal requires at least one key field")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[tuple[Any, ...], dict[str, Any]] = {}
        self.quarantined_tail: str | None = None
        self._load()

    @property
    def rows(self) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {key: dict(value) for key, value in self._rows.items()}

    def _key(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        try:
            return tuple(row[field] for field in self.key_fields)
        except KeyError as exc:
            raise ValueError(f"journal row missing key field {exc.args[0]!r}") from exc

    def append(self, row: Mapping[str, Any]) -> None:
        key = self._key(row)
        materialized = dict(row)
        existing = self._rows.get(key)
        if existing is not None:
            if canonical_json(existing) != canonical_json(materialized):
                raise ValueError(f"conflicting duplicate journal key: {key}")
            return
        wrapper = {
            "version": self.version,
            "identity": self.identity,
            "identity_hash": self.identity_hash,
            "key_fields": list(self.key_fields),
            "key": list(key),
            "row": materialized,
        }
        encoded = (json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self.path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        self._rows[key] = materialized

    def pending_conditions(
        self, item_ids: Iterable[str], conditions: Iterable[str]
    ) -> list[tuple[str, str]]:
        if self.key_fields != ("item_id", "condition"):
            raise ValueError("pending_conditions requires item_id/condition journal keys")
        return [
            (item_id, condition)
            for item_id in item_ids
            for condition in conditions
            if (item_id, condition) not in self._rows
        ]

    def _validate(self, wrapper: Any) -> dict[str, Any]:
        if not isinstance(wrapper, dict) or wrapper.get("version") != self.version:
            raise ValueError("unsupported journal wrapper")
        if (
            wrapper.get("identity") != self.identity
            or wrapper.get("identity_hash") != self.identity_hash
        ):
            raise ValueError("journal identity mismatch")
        if wrapper.get("key_fields") != list(self.key_fields):
            raise ValueError("journal key contract mismatch")
        row = wrapper.get("row")
        if not isinstance(row, dict) or wrapper.get("key") != list(self._key(row)):
            raise ValueError("journal key does not match row")
        return row

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        valid: list[bytes] = []
        rewrite = False
        for index, line in enumerate(lines):
            try:
                row = self._validate(json.loads(line.decode()))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if index != len(lines) - 1:
                    raise ValueError(f"invalid non-final journal row {index}") from exc
                digest = hashlib.sha256(line).hexdigest()[:16]
                quarantine = self.path.with_name(f"{self.path.name}.truncated.{digest}")
                quarantine.write_bytes(line)
                self.quarantined_tail = str(quarantine)
                rewrite = True
                break
            key = self._key(row)
            if key in self._rows:
                raise ValueError(f"duplicate journal key: {key}")
            self._rows[key] = row
            valid.append(line if line.endswith(b"\n") else line + b"\n")
            rewrite = rewrite or not line.endswith(b"\n")
        if rewrite:
            temporary = self.path.with_suffix(self.path.suffix + ".recovered")
            temporary.write_bytes(b"".join(valid))
            temporary.replace(self.path)


class Removable(Protocol):
    def remove(self) -> None: ...


@contextmanager
def managed_registrations(register: Iterable[Callable[[], Removable]]):
    """Register resources and guarantee reverse-order cleanup after failures."""

    handles: list[Removable] = []
    try:
        for callback in register:
            handles.append(callback())
        yield handles
    finally:
        for handle in reversed(handles):
            handle.remove()


def validate_unit_vector(values: Sequence[float], *, tolerance: float = 1e-10) -> float:
    """Return the norm or reject a non-finite/non-unit random control."""

    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError(f"random-vector norm must be 1 (observed {norm})")
    return norm
