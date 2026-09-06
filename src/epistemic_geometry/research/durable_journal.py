"""Explicit read, single-writer, offline recovery and disk-based sealing.

Additive replacement, not a change to historical CrashSafeJournal. Cooperating
writers/recovery/sealers must all use the stable sidecar lock (never unlink it).
Advisory locks cannot prevent arbitrary external privileged filesystem writes;
inode/content checks detect such changes and fail closed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest


class JournalIntegrityError(RuntimeError):
    pass


class JournalBusy(JournalIntegrityError):
    pass


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def snapshot(path: Path) -> bytes:
    """Never repair or write; require stable bytes from the same inode."""
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        after = os.fstat(handle.fileno())
        current = path.stat()
    attributes = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(s, k) for s in (after, current) for k in attributes):
        raise JournalBusy("FILE_CHANGED_DURING_READ")
    if len(raw) != before.st_size:
        raise JournalIntegrityError("SHORT_READ")
    return raw


@dataclass
class ReadView:
    raw: bytes
    rows: dict
    partial_tail: bytes

    @property
    def sha256(self):
        return digest(self.raw)


def decode(raw: bytes, identity: dict, key_fields: tuple, *, allow_partial=False) -> ReadView:
    rows: dict[tuple, dict] = {}
    identity_hash = stable_digest("RESEARCH-OS-JOURNAL", canonical_json(identity))
    lines = raw.splitlines(keepends=True)
    tail = b""
    for index, line in enumerate(lines):
        if not line.endswith(b"\n"):
            if allow_partial and index == len(lines) - 1:
                tail = line
                break
            raise JournalIntegrityError("PARTIAL_FINAL_LINE_OFFLINE_RECOVERY_REQUIRED")
        try:
            wrapper = json.loads(line)
            row = wrapper["row"]
            key = tuple(row[f] for f in key_fields)
        except (ValueError, KeyError, TypeError) as exc:
            raise JournalIntegrityError(f"MALFORMED_RECORD_{index}") from exc
        if (
            wrapper.get("version") != "research-os-jsonl-v1"
            or wrapper.get("identity") != identity
            or wrapper.get("identity_hash") != identity_hash
            or wrapper.get("key_fields") != list(key_fields)
            or wrapper.get("key") != list(key)
        ):
            raise JournalIntegrityError(f"IDENTITY_OR_KEY_MISMATCH_{index}")
        if key in rows:
            kind = "IDENTICAL" if rows[key] == row else "CONFLICTING"
            raise JournalIntegrityError(f"{kind}_DUPLICATE_RECORD_{index}")
        rows[key] = row
    return ReadView(raw, rows, tail)


def read_status(path: Path, *, identity: dict, key_fields: tuple) -> ReadView:
    """Status can observe a partial append; it never normalizes or truncates it."""
    return decode(snapshot(path), identity, key_fields, allow_partial=True)


@contextmanager
def exclusive(path: Path):
    lock_path = path.with_name(path.name + ".writer.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JournalBusy("WRITER_RECOVERY_OR_SEALER_ACTIVE") from exc
        initial = os.fstat(fd)
        if (initial.st_dev, initial.st_ino) != (lock_path.stat().st_dev, lock_path.stat().st_ino):
            raise JournalIntegrityError("LOCK_PATH_REPLACED")
        yield
        if (initial.st_dev, initial.st_ino) != (lock_path.stat().st_dev, lock_path.stat().st_ino):
            raise JournalIntegrityError("LOCK_PATH_REPLACED")
    finally:
        os.close(fd)


def fsync_directory(path: Path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_exclusive(path: Path, raw: bytes):
    """Never overwrite an existing seal/candidate. A crash leaves fail-closed evidence."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(fd, raw[offset:])
            if count <= 0:
                raise JournalIntegrityError("SHORT_WRITE")
            offset += count
        os.fsync(fd)
    finally:
        os.close(fd)
    fsync_directory(path.parent)


def verify_schedule(view: ReadView, expected_rows: list[dict], key_fields: tuple):
    expected = {tuple(r[f] for f in key_fields): (i, r) for i, r in enumerate(expected_rows)}
    if len(expected) != len(expected_rows):
        raise JournalIntegrityError("DUPLICATE_SCHEDULE")
    if view.partial_tail or set(view.rows) != set(expected):
        raise JournalIntegrityError("PERSISTED_COVERAGE_INCOMPLETE_OR_UNEXPECTED")
    for key, row in view.rows.items():
        index, planned = expected[key]
        if any(row.get(field) != value for field, value in planned.items()):
            raise JournalIntegrityError("SCHEDULE_OR_SEED_MISMATCH")
        if row.get("schedule_index") != index:
            raise JournalIntegrityError("SCHEDULE_INDEX_MISMATCH")


class SingleWriterJournal:
    """Context-managed writer; strict resume, no automatic tail recovery."""

    def __init__(self, path: Path, *, identity: dict, key_fields: tuple):
        self.path, self.identity, self.key_fields = Path(path), identity, tuple(key_fields)
        self._rows: dict[tuple, dict] = {}
        self._fd = None
        self._sealed = False

    def __enter__(self):
        self._lock = exclusive(self.path)
        self._lock.__enter__()
        try:
            self._fd = os.open(
                self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            raw = snapshot(self.path)
            self._rows = decode(raw, self.identity, self.key_fields).rows
            self._hash = hashlib.sha256(raw)
            self._stat = os.fstat(self._fd)
            return self
        except BaseException:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
            self._lock.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        return self._lock.__exit__(*exc)

    @property
    def rows(self):
        return {key: dict(row) for key, row in self._rows.items()}

    def _unchanged(self):
        if self._fd is None or self._sealed:
            raise JournalIntegrityError("WRITER_NOT_OPEN")
        fdstat, pathstat = os.fstat(self._fd), self.path.stat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(s, f) != getattr(self._stat, f) for s in (fdstat, pathstat) for f in fields):
            raise JournalIntegrityError("JOURNAL_REPLACED_OR_EXTERNALLY_MODIFIED")

    def append(self, row: dict):
        self._unchanged()
        key = tuple(row[f] for f in self.key_fields)
        if key in self._rows:
            if canonical_json(row) != canonical_json(self._rows[key]):
                raise JournalIntegrityError("CONFLICTING_DUPLICATE_APPEND")
            return
        wrapper = dict(
            version="research-os-jsonl-v1",
            identity=self.identity,
            identity_hash=stable_digest("RESEARCH-OS-JOURNAL", canonical_json(self.identity)),
            key_fields=list(self.key_fields),
            key=list(key),
            row=row,
        )
        raw = (json.dumps(wrapper, sort_keys=True, separators=(",", ":")) + "\n").encode()
        try:
            offset = 0
            while offset < len(raw):
                count = os.write(self._fd, raw[offset:])
                if count <= 0:
                    raise JournalIntegrityError("SHORT_APPEND")
                offset += count
            os.fsync(self._fd)
            current = os.fstat(self._fd)
            named = self.path.stat()
            if (current.st_dev, current.st_ino) != (
                named.st_dev,
                named.st_ino,
            ) or current.st_size != self._stat.st_size + len(raw):
                raise JournalIntegrityError("CONCURRENT_FILE_MUTATION")
            self._hash.update(raw)
            self._stat = current
            self._rows[key] = dict(row)
        except Exception as exc:
            self._sealed = True  # never allow a generation retry after uncertain persistence
            raise JournalIntegrityError("APPEND_UNCERTAIN_OFFLINE_REVIEW_REQUIRED") from exc
        except BaseException:
            self._sealed = True
            raise

    def persisted_rows(self, expected_rows: list[dict]):
        """Completeness and operational totals must come from persisted records."""
        self._unchanged()
        os.fsync(self._fd)
        raw = snapshot(self.path)
        if digest(raw) != self._hash.hexdigest():
            raise JournalIntegrityError("DISK_DIFFERS_FROM_ACKNOWLEDGED_APPEND_STREAM")
        view = decode(raw, self.identity, self.key_fields)
        verify_schedule(view, expected_rows, self.key_fields)
        return view

    def seal(self, seal_path: Path, expected_rows: list[dict], build_metadata):
        """Stop appends, audit disk and hash those same bytes under writer exclusion."""
        view = self.persisted_rows(expected_rows)
        raw = view.raw
        metadata: dict[str, Any] = build_metadata(list(view.rows.values()))
        metadata.update(
            completed=len(view.rows),
            expected=len(expected_rows),
            missing=0,
            unexpected=0,
            duplicates=0,
            journal_sha256=view.sha256,
            journal_bytes=len(raw),
            persisted_bytes_audited=True,
        )
        self._unchanged()
        write_exclusive(seal_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
        self._sealed = True
        return metadata


def offline_tail_candidate(path: Path, candidate: Path, *, identity: dict, key_fields: tuple):
    """Preserve original; removing a partial tail requires explicit offline action."""
    with exclusive(path):
        with path.open("rb") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JournalBusy("DATA_FILE_WRITER_ACTIVE") from exc
            raw = snapshot(path)
            view = decode(raw, identity, key_fields, allow_partial=True)
            if not view.partial_tail:
                raise JournalIntegrityError("NO_PARTIAL_TAIL")
            good = raw[: -len(view.partial_tail)]
            write_exclusive(candidate, good)
            if snapshot(path) != raw:
                raise JournalIntegrityError("SOURCE_CHANGED_DURING_RECOVERY")
            return dict(
                source_sha256=digest(raw),
                candidate_sha256=digest(good),
                omitted_tail_sha256=digest(view.partial_tail),
                omitted_tail_bytes=len(view.partial_tail),
                original_unchanged=True,
            )
