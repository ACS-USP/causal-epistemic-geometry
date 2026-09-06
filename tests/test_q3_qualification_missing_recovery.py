"""Model-free tests for the principal-authorized ten-key Q3.4 recovery."""

from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

import recover_q3_fresh_qualification_missing as recovery  # noqa: E402

from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402


def _wrapper(identity: dict, row: dict) -> bytes:
    fields = recovery.frozen.KEY_FIELDS
    value = {
        "version": "research-os-jsonl-v1",
        "identity": identity,
        "identity_hash": stable_digest("RESEARCH-OS-JOURNAL", canonical_json(identity)),
        "key_fields": list(fields),
        "key": [row[field] for field in fields],
        "row": row,
    }
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _planned(index: int) -> dict:
    return {
        "family_id": f"family-{index}",
        "condition": "policy",
        "rollout_index": 0,
        "seed": index + 100,
    }


def test_approved_recovery_set_is_exactly_the_frozen_ten() -> None:
    schedule = recovery.frozen.load_schedule()
    rows = recovery.approved_missing(schedule)
    assert len(rows) == 10
    assert [schedule.index(row) for row in rows] == [
        1380,
        1381,
        1565,
        1566,
        1855,
        3043,
        4481,
        4482,
        4678,
        5989,
    ]
    assert len({int(row["seed"]) for row in rows}) == 10


def test_recovery_source_has_no_parser_or_correctness_path() -> None:
    source = inspect.getsource(recovery)
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("semantic_v3" in name or "evaluator" in name for name in imported)
    assert "correctness" in source  # only fail-closed metadata and comments
    assert "score(" not in source


def test_imported_generation_and_routing_ast_matches_executed_collector() -> None:
    root = Path(__file__).resolve().parents[1]
    original = subprocess.check_output(
        [
            "git",
            "show",
            f"{recovery.ORIGINAL_EXECUTOR_COMMIT}:scripts/execute_q3_fresh_qualification.py",
        ],
        cwd=root,
    )
    current = (root / "scripts/execute_q3_fresh_qualification.py").read_bytes()
    assert recovery.scientific_executor_ast(current) == recovery.scientific_executor_ast(original)
    assert recovery.scientific_executor_constants(
        current
    ) == recovery.scientific_executor_constants(original)


def test_candidate_accepts_only_original_plus_authorized_provenance(tmp_path: Path) -> None:
    identity = {"experiment": "excluded-recovery-fixture"}
    schedule = [_planned(index) for index in range(3)]
    original = {**schedule[0], "schedule_index": 0, "generated_token_count": 1}
    recovered = {
        **schedule[1],
        "schedule_index": 1,
        "generated_token_count": 2,
        "persistence_provenance": recovery.PROVENANCE,
    }
    path = tmp_path / "candidate.jsonl"
    path.write_bytes(_wrapper(identity, original) + _wrapper(identity, recovered))
    approved = {
        tuple(schedule[1][field] for field in recovery.frozen.KEY_FIELDS),
        tuple(schedule[2][field] for field in recovery.frozen.KEY_FIELDS),
    }
    view, missing = recovery.audit_candidate(path, identity, schedule, approved)
    assert len(view.rows) == 2
    assert missing == {tuple(schedule[2][field] for field in recovery.frozen.KEY_FIELDS)}


def test_candidate_rejects_reexecuted_row_without_marker(tmp_path: Path) -> None:
    identity = {"experiment": "excluded-recovery-fixture"}
    schedule = [_planned(index) for index in range(2)]
    rows = [
        {**schedule[0], "schedule_index": 0},
        {**schedule[1], "schedule_index": 1},
    ]
    path = tmp_path / "candidate.jsonl"
    path.write_bytes(b"".join(_wrapper(identity, row) for row in rows))
    approved = {tuple(schedule[1][field] for field in recovery.frozen.KEY_FIELDS)}
    with pytest.raises(RuntimeError, match="PROVENANCE_MISSING"):
        recovery.audit_candidate(path, identity, schedule, approved)


def test_candidate_rejects_provenance_on_original_row(tmp_path: Path) -> None:
    identity = {"experiment": "excluded-recovery-fixture"}
    schedule = [_planned(index) for index in range(2)]
    original = {
        **schedule[0],
        "schedule_index": 0,
        "persistence_provenance": recovery.PROVENANCE,
    }
    path = tmp_path / "candidate.jsonl"
    path.write_bytes(_wrapper(identity, original))
    approved = {tuple(schedule[1][field] for field in recovery.frozen.KEY_FIELDS)}
    with pytest.raises(RuntimeError, match="ORIGINAL_ROW_MODIFIED"):
        recovery.audit_candidate(path, identity, schedule, approved)


def test_recovery_loop_iterates_only_approved_rows() -> None:
    source = inspect.getsource(recovery.collect)
    loops = [
        ast.unparse(node.iter) for node in ast.walk(ast.parse(source)) if isinstance(node, ast.For)
    ]
    assert "approved" in loops
    assert "schedule" not in loops
    assert '"persistence_provenance": PROVENANCE' in source


def test_frozen_original_hashes_match_public_incident_inventory() -> None:
    root = Path(__file__).resolve().parents[1]
    inventory = json.loads(
        (
            root / "review/q3_journal_persistence_incident/PRESERVATION_PUBLIC_HASHES.json"
        ).read_text()
    )["inventory"]
    hashes = {row["name"]: row["sha256"] for row in inventory}
    assert hashes["001-journal.jsonl"] == recovery.ORIGINAL_JOURNAL_SHA256
    assert hashes["000-COLLECTION_COMPLETE_SEAL.json"] == recovery.ORIGINAL_SEAL_SHA256
    assert hashes["002-PREOPEN_SEAL.json"] == recovery.ORIGINAL_PREOPEN_SHA256
    assert hashes["008-execute_q3_fresh_qualification.py"] == recovery.ORIGINAL_EXECUTOR_SHA256
