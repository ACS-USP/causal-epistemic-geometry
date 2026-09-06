#!/usr/bin/env python3
"""Fail-closed recovery of the ten principal-authorized missing Q3.4 rows.

This is a separate recovery executor.  It never edits the incident journal or
its inconsistent seal, never scores outputs, and can execute only the exact ten
keys recorded by the read-only incident audit and approved by the principal.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import execute_q3_fresh_qualification as frozen  # noqa: E402

from epistemic_geometry.research.durable_journal import (  # noqa: E402
    SingleWriterJournal,
    decode,
    digest,
    snapshot,
    write_exclusive,
)
from epistemic_geometry.research.reliability import validate_logical_rows  # noqa: E402

INCIDENT = ROOT / "review/q3_journal_persistence_incident/READONLY_JOURNAL_AUDIT.json"
ORIGINAL_EXECUTOR_COMMIT = "dda4f6b40d371eaa93cde575838451d98b953fc6"
ORIGINAL_JOURNAL_SHA256 = "ae65de79f99f6ef12b423c6e3604b0afea952b9d6cf835bb1668786cf15ed811"
ORIGINAL_SEAL_SHA256 = "2b1c848af719d8be923949abc70a9fcdc809121166d4f72cfaf4faea61756e1c"
ORIGINAL_PREOPEN_SHA256 = "68711ceb4a8cfc7e1bcc770aed7ffa7b5f025977a255649115ad01f4ec04329f"
ORIGINAL_ENGINE_SHA256 = "abc904db907d834a4b1d6719f8b39818b88b739fe3034d671a8d05d1591c81e4"
ORIGINAL_EXECUTOR_SHA256 = "31789b3d159303f23c73d256cc17ade1cfdbc7e0a8fc9c9cc106157de89bbf9b"
EXPECTED_BRANCH = "research/q3-fresh-instrument-qualification-recovery"
RECOVERY_STATUS = "AUTHORIZED_TEN_MISSING_KEYS_NO_CORRECTNESS_INSPECTION"
PROVENANCE = "REEXECUTED_MISSING_PERSISTED_KEY"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def scientific_executor_ast(source: bytes) -> dict[str, str]:
    selected = {
        "load_schedule",
        "load_prompts",
        "load_vectors",
        "load_router",
        "select_policy",
        "build_backend",
        "model_item",
        "RoutedHook",
        "generation_context",
        "_retryable",
    }
    result: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected:
            result[node.name] = ast.dump(node, include_attributes=False)
    if set(result) != selected:
        raise RuntimeError("Q3_RECOVERY_EXECUTOR_AST_INCOMPLETE")
    return result


def scientific_executor_constants(source: bytes) -> dict[str, str]:
    selected = {
        "REVIEW",
        "LOCK",
        "SCHEDULE",
        "SYSTEM",
        "MODEL",
        "MODEL_REVISION",
        "EXPECTED_ENVIRONMENT",
        "PRIVATE_ROUTER_SHA",
        "EXPECTED_ROWS",
        "KEY_FIELDS",
        "MAX_INFRASTRUCTURE_ATTEMPTS",
    }
    result: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in selected:
                    result[target.id] = ast.dump(node.value, include_attributes=False)
    if set(result) != selected:
        raise RuntimeError("Q3_RECOVERY_EXECUTOR_CONSTANTS_INCOMPLETE")
    return result


def verify_committed_code() -> dict[str, str]:
    """Authenticate both historical scientific executor and current recovery code."""
    old = git_blob(ORIGINAL_EXECUTOR_COMMIT, "scripts/execute_q3_fresh_qualification.py")
    if sha256_bytes(old) != ORIGINAL_EXECUTOR_SHA256:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_EXECUTOR_MISMATCH")
    head = frozen.git_head()
    if frozen.git_branch() != EXPECTED_BRANCH:
        raise RuntimeError("Q3_RECOVERY_BRANCH_INVALID")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("Q3_RECOVERY_WORKTREE_NOT_CLEAN")
    recovery_path = "scripts/recover_q3_fresh_qualification_missing.py"
    durable_path = "src/epistemic_geometry/research/durable_journal.py"
    normal_executor_path = "scripts/execute_q3_fresh_qualification.py"
    current_recovery = Path(__file__).resolve().read_bytes()
    current_durable = (ROOT / durable_path).read_bytes()
    current_normal_executor = (ROOT / normal_executor_path).read_bytes()
    if current_recovery != git_blob(head, recovery_path):
        raise RuntimeError("Q3_RECOVERY_UNCOMMITTED_EXECUTOR")
    if current_durable != git_blob(head, durable_path):
        raise RuntimeError("Q3_RECOVERY_UNCOMMITTED_JOURNAL_LIBRARY")
    if current_normal_executor != git_blob(head, normal_executor_path):
        raise RuntimeError("Q3_RECOVERY_UNCOMMITTED_IMPORTED_EXECUTOR")
    if scientific_executor_ast(current_normal_executor) != scientific_executor_ast(old):
        raise RuntimeError("Q3_RECOVERY_SCIENTIFIC_EXECUTOR_DRIFT")
    if scientific_executor_constants(current_normal_executor) != scientific_executor_constants(old):
        raise RuntimeError("Q3_RECOVERY_SCIENTIFIC_CONSTANT_DRIFT")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", ORIGINAL_EXECUTOR_COMMIT, head], cwd=ROOT
    ).returncode:
        raise RuntimeError("Q3_RECOVERY_LINEAGE_INVALID")
    return {
        "head": head,
        "original_executor_commit": ORIGINAL_EXECUTOR_COMMIT,
        "original_executor_sha256": ORIGINAL_EXECUTOR_SHA256,
        "recovery_executor_sha256": sha256_bytes(current_recovery),
        "durable_journal_sha256": sha256_bytes(current_durable),
        "imported_executor_sha256": sha256_bytes(current_normal_executor),
        "scientific_executor_ast_matches_original": True,
        "scientific_constants_match_original": True,
    }


def verify_scientific_objects(private_prompts: Path, private_router: Path) -> dict[str, Any]:
    """Recheck frozen objects without pretending the additive recovery is the old runner."""
    lock = frozen.read_json(frozen.LOCK)
    if (
        lock.get("status") != "FROZEN_BEFORE_QWEN_QUALIFICATION"
        or lock.get("implementation", {}).get("runner_sha256") != ORIGINAL_EXECUTOR_SHA256
    ):
        raise RuntimeError("Q3_RECOVERY_FROZEN_LOCK_INVALID")
    if frozen.sha256_file(frozen.SCHEDULE) != lock["schedule_sha256"]:
        raise RuntimeError("Q3_RECOVERY_SCHEDULE_HASH_MISMATCH")
    if frozen.sha256_file(frozen.SYSTEM) != lock["candidate_system"]["sha256"]:
        raise RuntimeError("Q3_RECOVERY_SYSTEM_HASH_MISMATCH")
    if frozen.sha256_file(private_prompts) != lock["private_prompt_only_dataset"]["sha256"]:
        raise RuntimeError("Q3_RECOVERY_PROMPT_HASH_MISMATCH")
    if frozen.sha256_file(private_router) != frozen.PRIVATE_ROUTER_SHA:
        raise RuntimeError("Q3_RECOVERY_ROUTER_HASH_MISMATCH")
    schedule = frozen.load_schedule()
    frozen.load_prompts(private_prompts)
    frozen.load_router(private_router)
    frozen.load_vectors()
    return {
        "lock_sha256": frozen.sha256_file(frozen.LOCK),
        "schedule_sha256": frozen.sha256_file(frozen.SCHEDULE),
        "system_sha256": frozen.sha256_file(frozen.SYSTEM),
        "private_prompt_sha256": frozen.sha256_file(private_prompts),
        "private_router_sha256": frozen.sha256_file(private_router),
        "schedule_rows": len(schedule),
    }


def read_first_identity(raw: bytes) -> dict[str, Any]:
    first = raw.splitlines(keepends=True)[0]
    if not first.endswith(b"\n"):
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_JOURNAL_INVALID")
    try:
        wrapper = json.loads(first)
        identity = wrapper["identity"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_JOURNAL_INVALID") from exc
    if not isinstance(identity, dict):
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_JOURNAL_INVALID")
    return identity


def approved_missing(schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    incident = frozen.read_json(INCIDENT)
    rows = list(incident["missing"])
    if len(rows) != 10 or any(row.get("classification") != "UNRESOLVED" for row in rows):
        raise RuntimeError("Q3_RECOVERY_APPROVED_KEY_MANIFEST_INVALID")
    selected: list[dict[str, Any]] = []
    for missing in rows:
        index = int(missing["schedule_index"])
        planned = schedule[index]
        for field in (*frozen.KEY_FIELDS, "seed"):
            if planned[field] != missing[field]:
                raise RuntimeError("Q3_RECOVERY_APPROVED_KEY_BINDING_MISMATCH")
        selected.append(planned)
    keys = [tuple(row[field] for field in frozen.KEY_FIELDS) for row in selected]
    if len(set(keys)) != 10:
        raise RuntimeError("Q3_RECOVERY_APPROVED_KEY_MANIFEST_INVALID")
    return selected


def audit_source(
    source_dir: Path, schedule: list[dict[str, Any]]
) -> tuple[bytes, dict[str, Any], set[tuple[Any, ...]]]:
    journal_path = source_dir / "journal.jsonl"
    seal_path = source_dir / "COLLECTION_COMPLETE_SEAL.json"
    preopen_path = source_dir / "PREOPEN_SEAL.json"
    raw = snapshot(journal_path)
    if digest(raw) != ORIGINAL_JOURNAL_SHA256:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_JOURNAL_HASH_MISMATCH")
    if frozen.sha256_file(seal_path) != ORIGINAL_SEAL_SHA256:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_SEAL_HASH_MISMATCH")
    if frozen.sha256_file(preopen_path) != ORIGINAL_PREOPEN_SHA256:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_PREOPEN_HASH_MISMATCH")
    if frozen.sha256_file(source_dir / "ENGINE_VALIDATION.json") != ORIGINAL_ENGINE_SHA256:
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_ENGINE_HASH_MISMATCH")
    identity = read_first_identity(raw)
    view = decode(raw, identity, frozen.KEY_FIELDS)
    expected = {tuple(row[field] for field in frozen.KEY_FIELDS) for row in schedule}
    if len(view.rows) != 5990 or not set(view.rows).issubset(expected):
        raise RuntimeError("Q3_RECOVERY_ORIGINAL_COVERAGE_INVALID")
    for key, row in view.rows.items():
        index = int(row["schedule_index"])
        planned = schedule[index]
        if key != tuple(planned[field] for field in frozen.KEY_FIELDS):
            raise RuntimeError("Q3_RECOVERY_ORIGINAL_KEY_MISMATCH")
        if any(row.get(field) != value for field, value in planned.items()):
            raise RuntimeError("Q3_RECOVERY_ORIGINAL_SCHEDULE_MISMATCH")
    return raw, identity, expected - set(view.rows)


def audit_candidate(
    candidate_path: Path,
    identity: dict[str, Any],
    schedule: list[dict[str, Any]],
    approved_keys: set[tuple[Any, ...]],
) -> tuple[Any, set[tuple[Any, ...]]]:
    raw = snapshot(candidate_path)
    view = decode(raw, identity, frozen.KEY_FIELDS)
    expected = {tuple(row[field] for field in frozen.KEY_FIELDS) for row in schedule}
    if not set(view.rows).issubset(expected):
        raise RuntimeError("Q3_RECOVERY_CANDIDATE_UNEXPECTED_KEY")
    source_keys = expected - approved_keys
    if not source_keys.issubset(view.rows):
        raise RuntimeError("Q3_RECOVERY_CANDIDATE_LOST_ORIGINAL_KEY")
    for key, row in view.rows.items():
        index = int(row["schedule_index"])
        planned = schedule[index]
        if key != tuple(planned[field] for field in frozen.KEY_FIELDS):
            raise RuntimeError("Q3_RECOVERY_CANDIDATE_KEY_MISMATCH")
        if any(row.get(field) != value for field, value in planned.items()):
            raise RuntimeError("Q3_RECOVERY_CANDIDATE_SCHEDULE_MISMATCH")
        if key in approved_keys:
            if row.get("persistence_provenance") != PROVENANCE:
                raise RuntimeError("Q3_RECOVERY_PROVENANCE_MISSING")
        elif "persistence_provenance" in row:
            raise RuntimeError("Q3_RECOVERY_ORIGINAL_ROW_MODIFIED")
    return view, expected - set(view.rows)


def immutable_preflight(
    source_dir: Path,
    recovery_dir: Path,
    model_path: str,
    private_prompts: Path,
    private_router: Path,
) -> dict[str, Any]:
    code = verify_committed_code()
    schedule = frozen.load_schedule()
    scientific_lock = verify_scientific_objects(private_prompts, private_router)
    approved = approved_missing(schedule)
    approved_keys = {tuple(row[field] for field in frozen.KEY_FIELDS) for row in approved}
    source_raw, identity, missing = audit_source(source_dir, schedule)
    if missing != approved_keys:
        raise RuntimeError("Q3_RECOVERY_SOURCE_MISSING_SET_MISMATCH")
    recovery_dir.mkdir(parents=True, exist_ok=True)
    candidate = recovery_dir / "journal.recovered-candidate.jsonl"
    if candidate.exists():
        if snapshot(candidate) != source_raw:
            raise RuntimeError("Q3_RECOVERY_PREEXISTING_CANDIDATE")
    else:
        write_exclusive(candidate, source_raw)
    environment = frozen.verify_environment(model_path)
    view, candidate_missing = audit_candidate(candidate, identity, schedule, approved_keys)
    if candidate_missing != approved_keys or view.sha256 != ORIGINAL_JOURNAL_SHA256:
        raise RuntimeError("Q3_RECOVERY_CANDIDATE_INITIAL_STATE_INVALID")
    seal = {
        "schema_version": "q3-fresh-qualification-ten-key-recovery-preopen-v1",
        "status": RECOVERY_STATUS,
        "code": code,
        "environment": environment,
        "scientific_lock": scientific_lock,
        "source": {
            "journal_sha256": ORIGINAL_JOURNAL_SHA256,
            "journal_rows": 5990,
            "inconsistent_seal_sha256": ORIGINAL_SEAL_SHA256,
            "preopen_sha256": ORIGINAL_PREOPEN_SHA256,
            "preserved_unchanged": True,
        },
        "candidate": {
            "path_name": candidate.name,
            "initial_sha256": view.sha256,
            "initial_rows": len(view.rows),
        },
        "authorized_missing": [
            {
                "family_id": row["family_id"],
                "condition": row["condition"],
                "rollout_index": int(row["rollout_index"]),
                "schedule_index": schedule.index(row),
                "seed": int(row["seed"]),
            }
            for row in approved
        ],
        "authorized_reexecutions": 10,
        "correctness_inspected": False,
        "scoring": "NOT_RUN",
        "confirmation_qwen_access": 0,
        "reserve_qwen_access": 0,
        "model_loaded_during_preflight": False,
    }
    seal_path = recovery_dir / "RECOVERY_PREOPEN_SEAL.json"
    if seal_path.exists():
        raise RuntimeError("Q3_RECOVERY_PREOPEN_ALREADY_EXISTS")
    write_exclusive(seal_path, (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode())
    return seal


def validate_preopen_for_collect(
    source_dir: Path,
    recovery_dir: Path,
    model_path: str,
    private_prompts: Path,
    private_router: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    seal_path = recovery_dir / "RECOVERY_PREOPEN_SEAL.json"
    if not seal_path.is_file():
        raise RuntimeError("Q3_RECOVERY_PREOPEN_REQUIRED")
    seal = frozen.read_json(seal_path)
    code = verify_committed_code()
    schedule = frozen.load_schedule()
    scientific_lock = verify_scientific_objects(private_prompts, private_router)
    approved = approved_missing(schedule)
    approved_keys = {tuple(row[field] for field in frozen.KEY_FIELDS) for row in approved}
    source_raw, identity, source_missing = audit_source(source_dir, schedule)
    environment = frozen.verify_environment(model_path)
    candidate = recovery_dir / "journal.recovered-candidate.jsonl"
    view, candidate_missing = audit_candidate(candidate, identity, schedule, approved_keys)
    if source_missing != approved_keys or not candidate_missing.issubset(approved_keys):
        raise RuntimeError("Q3_RECOVERY_STATE_INVALID")
    expected_immutable = {
        "status": RECOVERY_STATUS,
        "code": code,
        "environment": environment,
        "scientific_lock": scientific_lock,
    }
    expected_missing = [
        {
            "family_id": row["family_id"],
            "condition": row["condition"],
            "rollout_index": int(row["rollout_index"]),
            "schedule_index": schedule.index(row),
            "seed": int(row["seed"]),
        }
        for row in approved
    ]
    if any(seal.get(key) != value for key, value in expected_immutable.items()):
        raise RuntimeError("Q3_RECOVERY_PREOPEN_INVALID")
    if (
        seal.get("schema_version") != "q3-fresh-qualification-ten-key-recovery-preopen-v1"
        or seal.get("source", {}).get("journal_sha256") != digest(source_raw)
        or seal.get("source", {}).get("inconsistent_seal_sha256") != ORIGINAL_SEAL_SHA256
        or seal.get("source", {}).get("preopen_sha256") != ORIGINAL_PREOPEN_SHA256
        or seal.get("source", {}).get("preserved_unchanged") is not True
        or seal.get("candidate", {}).get("path_name") != candidate.name
        or seal.get("candidate", {}).get("initial_sha256") != ORIGINAL_JOURNAL_SHA256
        or seal.get("candidate", {}).get("initial_rows") != 5990
        or seal.get("authorized_missing") != expected_missing
        or seal.get("authorized_reexecutions") != 10
        or seal.get("correctness_inspected") is not False
        or seal.get("scoring") != "NOT_RUN"
        or seal.get("confirmation_qwen_access") != 0
        or seal.get("reserve_qwen_access") != 0
        or seal.get("model_loaded_during_preflight") is not False
    ):
        raise RuntimeError("Q3_RECOVERY_PREOPEN_INVALID")
    return seal, schedule, identity, environment


def collect(
    source_dir: Path,
    recovery_dir: Path,
    model_path: str,
    private_prompts: Path,
    private_router: Path,
) -> dict[str, Any]:
    preopen, schedule, identity, environment = validate_preopen_for_collect(
        source_dir, recovery_dir, model_path, private_prompts, private_router
    )
    final_seal = recovery_dir / "COLLECTION_COMPLETE_RECOVERY_SEAL.json"
    if final_seal.exists():
        raise RuntimeError("Q3_RECOVERY_ALREADY_SEALED")
    approved = approved_missing(schedule)
    approved_keys = {tuple(row[field] for field in frozen.KEY_FIELDS) for row in approved}
    prompts = frozen.load_prompts(private_prompts)
    vectors, metadata, order = frozen.load_vectors()
    router = frozen.load_router(private_router)
    candidate = recovery_dir / "journal.recovered-candidate.jsonl"
    with SingleWriterJournal(candidate, identity=identity, key_fields=frozen.KEY_FIELDS) as journal:
        existing = set(journal.rows)
        expected_original = {
            tuple(row[field] for field in frozen.KEY_FIELDS) for row in schedule
        } - approved_keys
        if not expected_original.issubset(existing) or not existing.issubset(
            expected_original | approved_keys
        ):
            raise RuntimeError("Q3_RECOVERY_CANDIDATE_COVERAGE_INVALID")
        backend = frozen.build_backend(model_path)
        started = time.monotonic()
        for row in approved:
            index = schedule.index(row)
            key = tuple(row[field] for field in frozen.KEY_FIELDS)
            if key in journal.rows:
                continue
            item = frozen.model_item(row["family_id"], prompts[row["family_id"]])
            attempts = 0
            retry_reasons: list[str] = []
            while True:
                try:
                    context, condition_meta = frozen.generation_context(
                        backend, item, row["condition"], vectors, metadata, order, router
                    )
                    trajectory_started = time.perf_counter()
                    with context as trace:
                        output = backend.generate_reasoning(
                            item,
                            sampling_seed=int(row["seed"]),
                            max_new_tokens=4096,
                            token_stop_predicate=frozen.extreme_mechanical_repetition_v1,
                            token_stop_name=frozen.EXTREME_REPETITION_NAME,
                        )
                    elapsed = time.perf_counter() - trajectory_started
                    terminal = frozen.frozen_terminal_metadata(output)
                    trace_meta = trace.metadata()
                    if row["condition"] == "ONLINE_ROUTED" and trace_meta["selection_count"] != 1:
                        raise RuntimeError("Q3_FRESH_ROUTER_SELECTION_COUNT_INVALID")
                    journal.append(
                        {
                            **row,
                            "schedule_index": index,
                            "raw_output": output.raw_output,
                            "generated_token_ids": output.metadata["generated_token_ids"],
                            **terminal,
                            "condition_metadata": condition_meta,
                            "hook_trace": trace_meta,
                            "model": frozen.MODEL,
                            "model_revision": frozen.MODEL_REVISION,
                            "seed": int(row["seed"]),
                            "retry_count": attempts,
                            "retry_reasons": retry_reasons,
                            "elapsed_seconds": elapsed,
                            "runtime_error": None,
                            "semantic_scoring": "DEFERRED_UNTIL_COMPLETE_RAW_SEAL",
                            "persistence_provenance": PROVENANCE,
                            "recovery_executor_commit": frozen.git_head(),
                        }
                    )
                    completed_recovery = len(approved_keys & set(journal.rows))
                    print(
                        json.dumps(
                            {
                                "recovery_completed": completed_recovery,
                                "recovery_pending": len(approved_keys) - completed_recovery,
                            }
                        ),
                        flush=True,
                    )
                    break
                except BaseException as exc:
                    if frozen._retryable(exc) and attempts + 1 < frozen.MAX_INFRASTRUCTURE_ATTEMPTS:
                        attempts += 1
                        retry_reasons.append(f"{type(exc).__name__}: {exc}")
                        continue
                    raise
        persisted = journal.persisted_rows(schedule)
        rows = list(persisted.rows.values())
        expected_keys = [tuple(row[field] for field in frozen.KEY_FIELDS) for row in schedule]
        coverage = validate_logical_rows(
            rows, key_fields=frozen.KEY_FIELDS, expected_keys=expected_keys
        )
        recovered = [row for row in rows if row.get("persistence_provenance") == PROVENANCE]
        if (
            not coverage.valid
            or len(rows) != frozen.EXPECTED_ROWS
            or len(recovered) != 10
            or {tuple(row[field] for field in frozen.KEY_FIELDS) for row in recovered}
            != approved_keys
        ):
            raise RuntimeError("Q3_RECOVERY_COMPLETION_INVALID")
        tokens = [int(row["generated_token_count"]) for row in rows]
        metadata_seal = {
            "schema_version": "q3-fresh-qualification-recovery-collection-seal-v1",
            "status": "COLLECTION_COMPLETE_RAW_UNSCORED_AFTER_TEN_KEY_RECOVERY",
            "original_persisted_rows": 5990,
            "reexecuted_missing_rows": 10,
            "reexecution_provenance": PROVENANCE,
            "original_journal_sha256": ORIGINAL_JOURNAL_SHA256,
            "original_inconsistent_seal_sha256": ORIGINAL_SEAL_SHA256,
            "replacements": 0,
            "retry_rows": sum(int(row["retry_count"]) > 0 for row in rows),
            "runtime_errors": sum(row["runtime_error"] is not None for row in rows),
            "repetition_stops": sum(
                row["terminal_reason"] == frozen.EXTREME_REPETITION_NAME for row in rows
            ),
            "hard_caps": sum(row["terminal_reason"] == "max_new_tokens" for row in rows),
            "generated_tokens": sum(tokens),
            "generated_token_mean": float(np.mean(tokens)),
            "generated_token_median": float(np.median(tokens)),
            "recovery_elapsed_seconds": time.monotonic() - started,
            "recovery_preopen_sha256": frozen.sha256_file(
                recovery_dir / "RECOVERY_PREOPEN_SEAL.json"
            ),
            "correctness_inspected": False,
            "semantic_scoring": "NOT_RUN",
            "confirmation_qwen_access": 0,
            "reserve_qwen_access": 0,
            "environment": environment,
            "principal_decision": "REEXECUTE_EXACTLY_TEN_MISSING_FROZEN_LOGICAL_KEYS",
        }
        return journal.seal(final_seal, schedule, lambda _audited: metadata_seal)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "collect"), required=True)
    parser.add_argument("--source-execution-dir", type=Path, required=True)
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--private-prompts", type=Path, required=True)
    parser.add_argument("--private-router", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "preflight":
        result = immutable_preflight(
            args.source_execution_dir,
            args.recovery_dir,
            args.model_path,
            args.private_prompts,
            args.private_router,
        )
    else:
        result = collect(
            args.source_execution_dir,
            args.recovery_dir,
            args.model_path,
            args.private_prompts,
            args.private_router,
        )
    print(
        json.dumps(
            {key: result[key] for key in result if key in {"status", "completed", "expected"}},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
