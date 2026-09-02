from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

import execute_q2_oos_v2_semantic as executor  # noqa: E402
from execute_q2_oos_v2_semantic import (  # noqa: E402
    EXPECTED_CONTROLLERS,
    EXPECTED_SCHEDULE_ROWS,
    validate_frozen_objects,
)
from run_q2_oos_v2_semantic import frozen_terminal_metadata  # noqa: E402

from epistemic_geometry.types import BackendOutput  # noqa: E402


def test_frozen_oos_schedule_and_objects_are_complete_without_model() -> None:
    result = validate_frozen_objects()
    assert result["schedule_count"] == EXPECTED_SCHEDULE_ROWS == 19_200
    assert result["unique_logical_keys"] == EXPECTED_SCHEDULE_ROWS
    assert result["unique_seeds"] == EXPECTED_SCHEDULE_ROWS
    assert len(result["fresh_controller_order"]) == EXPECTED_CONTROLLERS == 16
    assert len(result["conditions"]) == 32
    assert result["semantic_outcomes_before_execution"] == 0
    assert result["correctness_inspected_before_execution"] is False
    assert result["spark1_only"] is True
    assert result["spark2_used"] is False
    assert result["runpod_used"] is False


def _minimal_frozen(commit: str) -> dict[str, object]:
    return {
        "head": commit,
        "prediction_lock_parent_head": executor.PREDICTION_LOCK_PARENT_HEAD,
        "hashes": {
            str(executor.SCHEDULE.relative_to(executor.ROOT)): executor.EXPECTED_SCHEDULE_SHA256,
            str(executor.SELECTED.relative_to(executor.ROOT)): (
                executor.EXPECTED_SELECTED_BANK_SHA256
            ),
            str(executor.PANEL.relative_to(executor.ROOT)): executor.EXPECTED_PANEL_SHA256,
        },
    }


def _preopen_payload(
    frozen: dict[str, object],
    *,
    code_commit: str | None = None,
    environment: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "q2-oos-v2-semantic-preopen-seal-v1",
        "status": "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS",
        "status_contract": "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS",
        "authorization": "PRINCIPAL_AUTHORIZATION_Q2_OOS_V2_SEMANTIC_EXECUTION",
        "code_commit": code_commit or str(frozen["head"]),
        "frozen": frozen,
        "environment": environment
        or {
            "model_revision": executor.MODEL_REVISION,
            "tokenizer_revision": executor.MODEL_REVISION,
            "qualified_environment_fingerprint": executor.EXPECTED_ENVIRONMENT,
            "model_bytes": {"manifest_sha256": executor.EXPECTED_MODEL_MANIFEST_SHA256},
        },
        "model_load_performed": False,
        "journal_path_is_empty": True,
        "semantic_outcomes_before_execution": 0,
        "pre_existing_semantic_rows": 0,
        "correctness_inspected_before_execution": False,
    }


def test_collect_refuses_before_model_load_without_preopen_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(executor, "git_head", lambda: "a" * 40)
    monkeypatch.setattr(
        executor, "verify_spark1_environment", lambda _: calls.append("environment")
    )
    monkeypatch.setattr(executor, "build_backend", lambda _: calls.append("model"))
    with pytest.raises(RuntimeError, match="PREOPEN_SEAL_REQUIRED"):
        executor.collect(tmp_path, "unused", "a" * 40)
    assert calls == []


def test_collect_refuses_stale_preopen_seal_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = "a" * 40
    stale = "b" * 40
    frozen = _minimal_frozen(current)
    payload = _preopen_payload(frozen, code_commit=stale)
    (tmp_path / "PREOPEN_SEAL.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(executor, "git_head", lambda: current)
    monkeypatch.setattr(executor, "validate_frozen_objects", lambda: frozen)
    with pytest.raises(RuntimeError, match="PREOPEN_SEAL_INVALID"):
        executor.collect(tmp_path, "unused", current)


def test_preopen_seal_validates_current_identity_and_frozen_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = "a" * 40
    frozen = _minimal_frozen(current)
    (tmp_path / "PREOPEN_SEAL.json").write_text(
        json.dumps(_preopen_payload(frozen)), encoding="utf-8"
    )
    monkeypatch.setattr(executor, "git_head", lambda: current)
    monkeypatch.setattr(executor, "validate_frozen_objects", lambda: frozen)
    seal, observed = executor.validate_preopen_seal(tmp_path, current)
    assert seal["status"] == "AUTHORIZED_PREOPEN_NO_SEMANTIC_OUTPUTS"
    assert observed == frozen


def test_model_bytes_use_the_existing_pinned_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    files = {"config.json": b"{}\n", "tokenizer.json": b"tokenizer\n"}
    for name, content in files.items():
        (model_path / name).write_bytes(content)
    rows = [
        {"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in files.items()
    ]
    manifest = {
        "model": executor.MODEL,
        "revision": executor.MODEL_REVISION,
        "files": rows,
        "file_count": len(rows),
        "total_bytes": sum(len(content) for content in files.values()),
        "manifest_sha256": "inner",
    }
    manifest_path = tmp_path / "EXACT_MODEL_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(executor, "MODEL_MANIFEST", manifest_path)
    monkeypatch.setattr(
        executor,
        "EXPECTED_MODEL_MANIFEST_SHA256",
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    verified = executor.verify_qualified_model_bytes(str(model_path))
    assert verified["file_count"] == 2
    (model_path / "tokenizer.json").write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="POST_MAINTENANCE_ENVIRONMENT_DRIFT"):
        executor.verify_qualified_model_bytes(str(model_path))


def test_frozen_schedule_binds_exact_controller_order_alpha_layer_and_duration() -> None:
    result = validate_frozen_objects()
    schedule = executor.load_schedule()
    _, selected = executor.load_selected_vectors()
    assert result["rollouts"] == [0, 1]
    assert len(result["fresh_controller_order"]) == EXPECTED_CONTROLLERS == 16
    for row in schedule:
        metadata = selected["controllers"][row["condition"]]
        assert row["candidate_id"] == metadata["candidate_id"]
        assert row["shell"] == metadata["shell"]
        assert row["controller_vector_hash"] == metadata["vector_hash"]
        assert float(row["alpha"]) == float(metadata["alpha"])
        assert row["layer"] == 27
        assert row["duration"] == "sustained_current_token"


def test_crash_safe_journal_resume_skips_completed_logical_key(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    identity = {"test": "resume"}
    first = {"item_id": "i0", "condition": "c", "rollout_index": 0, "seed": 101}
    journal = executor.CrashSafeJournal(path, identity=identity, key_fields=executor.KEY_FIELDS)
    journal.append(first)
    reopened = executor.CrashSafeJournal(path, identity=identity, key_fields=executor.KEY_FIELDS)
    assert ("i0", "c", 0) in reopened.rows
    assert len(reopened.rows) == 1


def test_collect_resume_only_generates_missing_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    environment = {"env": "fixture"}
    identity = {"test": "resume-collect"}
    frozen = {"head": commit}
    preopen = {"environment": environment}
    rows = [
        {"item_id": "i0", "condition": "c", "rollout_index": 0, "seed": 101},
        {"item_id": "i1", "condition": "c", "rollout_index": 0, "seed": 202},
    ]
    journal = executor.CrashSafeJournal(
        tmp_path / "journal.jsonl", identity=identity, key_fields=executor.KEY_FIELDS
    )
    journal.append(rows[0])
    calls: list[int] = []

    class Backend:
        def generate_reasoning(
            self, _item: object, *, sampling_seed: int, **_: object
        ) -> BackendOutput:
            calls.append(sampling_seed)
            return BackendOutput(
                "fixture", {"generated_token_count": 1, "generated_token_ids": [1], "timing": {}}
            )

    monkeypatch.setattr(executor, "validate_preopen_seal", lambda *_: (preopen, frozen))
    monkeypatch.setattr(executor, "load_items", lambda: {"i0": object(), "i1": object()})
    monkeypatch.setattr(executor, "load_schedule", lambda: rows)
    monkeypatch.setattr(executor, "load_selected_vectors", lambda: ({}, {}))
    monkeypatch.setattr(executor, "verify_spark1_environment", lambda _: environment)
    monkeypatch.setattr(executor, "build_identity", lambda *_: identity)
    monkeypatch.setattr(executor, "condition_context", lambda *_: (nullcontext(), object(), {}))
    monkeypatch.setattr(executor, "build_backend", lambda _: Backend())
    monkeypatch.setattr(executor, "complete_collection_seal", lambda *args: {"completed_rows": 2})
    result = executor.collect(tmp_path, "unused", commit)
    assert result["completed_rows"] == 2
    assert calls == [202]
    resumed = executor.CrashSafeJournal(
        tmp_path / "journal.jsonl", identity=identity, key_fields=executor.KEY_FIELDS
    )
    assert len(resumed.rows) == 2


def test_identical_duplicate_is_idempotent_and_conflicting_duplicate_blocks(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    identity = {"test": "duplicates"}
    row = {"item_id": "i", "condition": "c", "rollout_index": 0, "value": 1}
    journal = executor.CrashSafeJournal(path, identity=identity, key_fields=executor.KEY_FIELDS)
    journal.append(row)
    journal.append(dict(row))
    assert len(journal.rows) == 1
    with pytest.raises(ValueError, match="conflicting duplicate"):
        journal.append({**row, "value": 2})


def test_terminal_policy_persists_repetition_and_hard_cap_as_failures() -> None:
    repetition = frozen_terminal_metadata(
        BackendOutput(
            raw_output="fixture",
            metadata={
                "generated_token_count": 256,
                "terminal_policy": {"triggered": True, "trigger_token_count": 256},
            },
        )
    )
    hard_cap = frozen_terminal_metadata(
        BackendOutput(raw_output="fixture", metadata={"generated_token_count": 4096})
    )
    for metadata in (repetition, hard_cap):
        assert metadata["truncated"] is True
        assert metadata["terminal_answer_channel_failure"] is True
        assert metadata["commitment_valid_if_terminal_failure"] is False
        assert metadata["semantic_evaluable_if_terminal_failure"] is False
        assert metadata["binary_error_e_if_terminal_failure"] == 1


def test_collection_module_has_no_parser_or_scoring_import_path() -> None:
    tree = ast.parse(Path(executor.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("parser" in name.lower() or "semantic_v3" in name.lower() for name in imported)
    source = inspect.getsource(executor.collect)
    assert "correctness" not in source.lower()
    assert "score" not in source.lower()


def test_operational_retry_preserves_key_and_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    environment = {"env": "fixture"}
    frozen = {"head": commit}
    seal = {"environment": environment}
    row = {"item_id": "i", "condition": "c", "rollout_index": 0, "seed": 777}
    calls: list[int] = []

    class Backend:
        def generate_reasoning(
            self, _item: object, *, sampling_seed: int, **_: object
        ) -> BackendOutput:
            calls.append(sampling_seed)
            if len(calls) == 1:
                raise ConnectionError("synthetic transport interruption")
            return BackendOutput(
                "fixture", {"generated_token_count": 1, "generated_token_ids": [1], "timing": {}}
            )

    monkeypatch.setattr(executor, "validate_preopen_seal", lambda *_: (seal, frozen))
    monkeypatch.setattr(executor, "load_items", lambda: {"i": object()})
    monkeypatch.setattr(executor, "load_schedule", lambda: [row])
    monkeypatch.setattr(executor, "load_selected_vectors", lambda: ({}, {}))
    monkeypatch.setattr(executor, "verify_spark1_environment", lambda _: environment)
    monkeypatch.setattr(executor, "build_identity", lambda *_: {"test": "retry"})
    monkeypatch.setattr(executor, "condition_context", lambda *_: (nullcontext(), object(), {}))
    monkeypatch.setattr(executor, "build_backend", lambda _: Backend())
    monkeypatch.setattr(executor, "complete_collection_seal", lambda *args: {"completed_rows": 1})
    result = executor.collect(tmp_path, "unused", commit)
    assert result["completed_rows"] == 1
    assert calls == [777, 777]
    saved = next(
        iter(
            executor.CrashSafeJournal(
                tmp_path / "journal.jsonl",
                identity={"test": "retry"},
                key_fields=executor.KEY_FIELDS,
            ).rows.values()
        )
    )
    assert saved["retry_count"] == 1
    assert saved["seed"] == 777


def test_incomplete_collection_cannot_be_sealed(tmp_path: Path) -> None:
    journal = executor.CrashSafeJournal(
        tmp_path / "journal.jsonl", identity={"test": "incomplete"}, key_fields=executor.KEY_FIELDS
    )
    with pytest.raises(RuntimeError, match="EXECUTION_INCOMPLETE"):
        executor.complete_collection_seal(
            journal,
            [{"item_id": "i", "condition": "c", "rollout_index": 0}],
            tmp_path,
            {},
            {},
            "a" * 40,
            "now",
            0.0,
        )
