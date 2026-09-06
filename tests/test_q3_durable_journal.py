"""Synthetic persistence faults only: no model, benchmark, parser or GPU."""

import ast
import hashlib
import json
import multiprocessing
import os
import subprocess
import threading
from pathlib import Path

import pytest

from epistemic_geometry.reproducibility import canonical_json, stable_digest
from epistemic_geometry.research import durable_journal as d
from epistemic_geometry.research.reliability import CrashSafeJournal

IDENTITY = {"experiment": "EXCLUDED_SYNTHETIC_FIXTURE"}
FIELDS = ("item", "condition", "rollout")


def row(i):
    return dict(
        item=f"synthetic-{i}", condition="fixture", rollout=0, seed=i + 17, schedule_index=i
    )


def encode(r):
    return (
        json.dumps(
            dict(
                version="research-os-jsonl-v1",
                identity=IDENTITY,
                identity_hash=stable_digest("RESEARCH-OS-JOURNAL", canonical_json(IDENTITY)),
                key_fields=list(FIELDS),
                key=[r[f] for f in FIELDS],
                row=r,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def writer(path):
    return d.SingleWriterJournal(path, identity=IDENTITY, key_fields=FIELDS)


def status(path):
    return d.read_status(path, identity=IDENTITY, key_fields=FIELDS)


def test_memory_6000_disk_5990_cannot_seal(tmp_path):
    path = tmp_path / "journal"
    path.write_bytes(b"".join(encode(row(i)) for i in range(5990)))
    with writer(path) as j:
        for i in range(5990, 6000):
            r = row(i)
            j._rows[tuple(r[f] for f in FIELDS)] = r
        assert len(j.rows) == 6000
        with pytest.raises(d.JournalIntegrityError, match="COVERAGE"):
            j.seal(tmp_path / "seal", [row(i) for i in range(6000)], lambda rows: {})
    assert not (tmp_path / "seal").exists()


def test_status_partial_tail_is_readonly(tmp_path):
    path = tmp_path / "journal"
    path.write_bytes(encode(row(0)) + b'{"partial":')
    before = path.stat()
    raw = path.read_bytes()
    for _ in range(3):
        view = status(path)
        assert len(view.rows) == 1 and view.partial_tail
    after = path.stat()
    assert path.read_bytes() == raw
    assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(d.JournalIntegrityError, match="PARTIAL"):
        with writer(path):
            pass
    assert path.read_bytes() == raw


def test_monitor_during_append_and_recovery_exclusion(tmp_path, monkeypatch):
    path = tmp_path / "journal"
    partial, proceed = threading.Event(), threading.Event()
    real_write = d.os.write
    first = True

    def split_write(fd, data):
        nonlocal first
        if first:
            first = False
            n = real_write(fd, data[: len(data) // 2])
            partial.set()
            assert proceed.wait(5)
            return n
        return real_write(fd, data)

    with writer(path) as j:
        monkeypatch.setattr(d.os, "write", split_write)
        errors = []

        def append():
            try:
                j.append(row(0))
            except BaseException as exc:
                errors.append(exc)

        t = threading.Thread(target=append)
        t.start()
        assert partial.wait(5)
        before = path.read_bytes()
        inode = path.stat().st_ino
        assert status(path).partial_tail
        with pytest.raises(d.JournalBusy):
            d.offline_tail_candidate(
                path, tmp_path / "candidate", identity=IDENTITY, key_fields=FIELDS
            )
        with pytest.raises(d.JournalBusy):
            with writer(path):
                pass
        assert path.read_bytes() == before and path.stat().st_ino == inode
        proceed.set()
        t.join(5)
        assert not t.is_alive() and not errors
        assert len(status(path).rows) == 1


def test_replacement_while_writer_open_blocks_append_and_seal(tmp_path):
    path = tmp_path / "journal"
    with writer(path) as j:
        j.append(row(0))
        replacement = tmp_path / "replacement"
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        with pytest.raises(d.JournalIntegrityError, match="REPLACED"):
            j.append(row(1))
        with pytest.raises(d.JournalIntegrityError, match="REPLACED"):
            j.seal(tmp_path / "seal", [row(0)], lambda _: {})
    assert not (tmp_path / "seal").exists()


def _crash(path):
    with writer(Path(path)) as j:
        j.append(row(0))
        os._exit(19)


def test_actual_process_crash_resume_missing_only(tmp_path):
    path = tmp_path / "journal"
    p = multiprocessing.get_context("spawn").Process(target=_crash, args=(str(path),))
    p.start()
    p.join(10)
    assert p.exitcode == 19
    before = path.read_bytes()
    with writer(path) as j:
        executed = []
        for i in range(2):
            if tuple(row(i)[f] for f in FIELDS) not in j.rows:
                executed.append(i)
                j.append(row(i))
        assert executed == [1]
    assert path.read_bytes().startswith(before)


def test_duplicate_append_idempotent_conflict_blocks(tmp_path):
    path = tmp_path / "journal"
    with writer(path) as j:
        j.append(row(0))
        before = path.read_bytes()
        j.append(row(0))
        assert path.read_bytes() == before
        with pytest.raises(d.JournalIntegrityError, match="CONFLICTING"):
            j.append(dict(row(0), seed=5))


@pytest.mark.parametrize("conflict", [False, True])
def test_duplicate_on_disk_always_fails(tmp_path, conflict):
    path = tmp_path / "journal"
    second = dict(row(0), seed=3) if conflict else row(0)
    path.write_bytes(encode(row(0)) + encode(second))
    with pytest.raises(d.JournalIntegrityError, match="DUPLICATE"):
        status(path)


@pytest.mark.parametrize(
    "change", ["seed", "schedule_index", "identity", "wrapper_key", "unexpected"]
)
def test_seal_rejects_wrong_provenance(tmp_path, change):
    path = tmp_path / "journal"
    w = json.loads(encode(row(0)))
    if change in ("seed", "schedule_index"):
        w["row"][change] = 99
    elif change == "identity":
        w["identity"]["experiment"] = "OTHER"
    elif change == "wrapper_key":
        w["key"][0] = "OTHER"
    else:
        w = json.loads(encode(row(1)))
    path.write_bytes(json.dumps(w).encode() + b"\n")
    with pytest.raises(d.JournalIntegrityError):
        with writer(path) as j:
            j.seal(tmp_path / "seal", [row(0)], lambda _: {})
    assert not (tmp_path / "seal").exists()


def test_offline_candidate_preserves_original_and_opaque_tail(tmp_path):
    path = tmp_path / "journal"
    candidate = tmp_path / "candidate"
    raw = encode(row(0)) + b'{"synthetic_partial":'
    path.write_bytes(raw)
    result = d.offline_tail_candidate(path, candidate, identity=IDENTITY, key_fields=FIELDS)
    assert path.read_bytes() == raw and candidate.read_bytes() == encode(row(0))
    assert result["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["omitted_tail_bytes"] == len(b'{"synthetic_partial":')


def test_complete_seal_independent_reopen_and_no_append_after_seal(tmp_path):
    path = tmp_path / "journal"
    seal = tmp_path / "seal"
    with writer(path) as j:
        for i in range(3):
            j.append(row(i))
        result = j.seal(
            seal, [row(i) for i in range(3)], lambda rows: {"count_from_disk": len(rows)}
        )
        with pytest.raises(d.JournalIntegrityError):
            j.append(row(3))
    raw = path.read_bytes()
    wrappers = [json.loads(line) for line in raw.splitlines()]
    assert len(wrappers) == result["completed"] == result["count_from_disk"] == 3
    assert hashlib.sha256(raw).hexdigest() == json.loads(seal.read_bytes())["journal_sha256"]
    with writer(path) as j:
        with pytest.raises(FileExistsError):
            j.seal(seal, [row(i) for i in range(3)], lambda _: {})


def test_old_reader_can_overwrite_fsynced_concurrent_appends(tmp_path, monkeypatch):
    """Possibility reproduction, never proof that this happened in Q3.4."""
    path = tmp_path / "journal"
    old = CrashSafeJournal(path, identity=IDENTITY, key_fields=FIELDS)
    old.append(row(0))
    captured = path.read_bytes()[:-1]  # valid JSON captured before its newline arrives
    real_read = Path.read_bytes

    def raced_read(p):
        if p == path:
            old.append(row(1))
            old.append(row(2))  # flush + fsync both before rewrite
            return captured
        return real_read(p)

    monkeypatch.setattr(Path, "read_bytes", raced_read)
    observer = CrashSafeJournal(path, identity=IDENTITY, key_fields=FIELDS)
    monkeypatch.setattr(Path, "read_bytes", real_read)
    assert len(old.rows) == 3 and len(observer.rows) == 1
    assert len(path.read_bytes().splitlines()) == 1
    assert not list(tmp_path.glob("*.truncated.*"))


def test_generation_loop_ast_unchanged():
    root = Path(__file__).resolve().parents[1]
    original = subprocess.check_output(
        [
            "git",
            "show",
            "dda4f6b40d371eaa93cde575838451d98b953fc6:scripts/execute_q3_fresh_qualification.py",
        ],
        cwd=root,
        text=True,
    )
    current = (root / "scripts/execute_q3_fresh_qualification.py").read_text()

    def loop(source):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.For) and ast.unparse(node.target) == "(index, row)":
                if ast.unparse(node.iter) == "enumerate(schedule)":
                    return ast.dump(node)
        raise AssertionError("generation loop missing")

    assert loop(original) == loop(current)


def test_fsync_error_does_not_expose_retryable_generation_exception(tmp_path, monkeypatch):
    path = tmp_path / "journal"
    with writer(path) as j:

        def broken_fsync(fd):
            raise BrokenPipeError("synthetic uncertainty after write")

        monkeypatch.setattr(d.os, "fsync", broken_fsync)
        with pytest.raises(d.JournalIntegrityError, match="APPEND_UNCERTAIN") as caught:
            j.append(row(0))
        assert not isinstance(
            caught.value, (ConnectionError, TimeoutError, BrokenPipeError, EOFError)
        )
        assert len(status(path).rows) == 1
        with pytest.raises(d.JournalIntegrityError, match="NOT_OPEN"):
            j.append(row(0))


def test_nonfinal_malformed_record_is_never_recovered(tmp_path):
    path = tmp_path / "journal"
    raw = b"invalid-json\n" + encode(row(0))
    path.write_bytes(raw)
    with pytest.raises(d.JournalIntegrityError, match="MALFORMED"):
        d.offline_tail_candidate(path, tmp_path / "candidate", identity=IDENTITY, key_fields=FIELDS)
    assert path.read_bytes() == raw
    assert not (tmp_path / "candidate").exists()


def test_incident_manifest_preserves_exact_64_bit_schedule_seeds():
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "review/q3_journal_persistence_incident/READONLY_JOURNAL_AUDIT.json").read_bytes()
    )
    planned = json.loads(
        (
            root / "review/q3_fresh_instrument_qualification/Q3_FRESH_QUALIFICATION_SCHEDULE.json"
        ).read_bytes()
    )["rows"]
    for missing in report["missing"]:
        frozen = planned[missing["schedule_index"]]
        for field in ("family_id", "condition", "rollout_index", "seed"):
            assert missing[field] == frozen[field]
    assert report["missing"][0]["seed"] == 181429994752461983
