#!/usr/bin/env python3
"""Preflight and run the frozen 80-row Q1 LiveCodeBench Stage A2 on Spark 2."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate6_2_first_stage_repair import build_backend, model_item  # noqa: E402

from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_a2 as stage_a2  # noqa: E402
from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (  # noqa: E402
    evaluate_livecodebench_output_stage_a2,
)

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
AUDIT = REVIEW / "stage_a_failure_audit"
AUTHORIZATION = AUDIT / "STAGE_A2_PRINCIPAL_AUTHORIZATION.json"
AMENDMENT = AUDIT / "AMENDMENT2_LOCK_DRAFT.json"
MANIFEST = AUDIT / "STAGE_A2_FAMILY_MANIFEST.json"
SCHEDULE = AUDIT / "STAGE_A2_SCHEDULE.json"
PARSER_SOURCE = ROOT / "src/epistemic_geometry/experiments/q1_second_task_stage_a_failure.py"
TEST_SOURCE = ROOT / "tests/test_q1_second_task_stage_a_failure.py"
PARSER_VERSION = "TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(value: Any) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def is_ancestor(ancestor: str, descendant: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=ROOT
    ).returncode == 0


def load_dataset_pages(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("rows_*.json"), key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not paths:
        raise RuntimeError("pinned LiveCodeBench page cache is absent")
    rows = [entry for path in paths for entry in read_json(path)["rows"]]
    rows.sort(key=lambda entry: int(entry["row_idx"]))
    return [dict(entry["row"]) for entry in rows]


def load_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_locks() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    authorization = read_json(AUTHORIZATION)
    expected = authorization["reviewed_hashes"]
    actual = {
        "amendment2_draft": sha256(AMENDMENT),
        "stage_a2_manifest": sha256(MANIFEST),
        "stage_a2_schedule": sha256(SCHEDULE),
    }
    if actual != expected:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A2_PREOPEN_HASH_MISMATCH")
    if authorization["authorized"] != "STAGE_A2_ONLY":
        raise RuntimeError("Stage-A2 principal authorization is absent")
    if authorization["stage_b"] != "CLOSED_NOT_AUTHORIZED":
        raise RuntimeError("Stage B is not sealed")
    source_hashes = authorization["implementation_hashes"]
    actual_source_hashes = {
        "parser_source": sha256(PARSER_SOURCE),
        "evaluator_function": source_sha256(evaluate_livecodebench_output_stage_a2),
        "prompt_builder_function": source_sha256(q1s.build_livecodebench_prompt),
        "adversarial_tests": sha256(TEST_SOURCE),
    }
    if actual_source_hashes != source_hashes:
        raise RuntimeError("Stage-A2 parser/evaluator/prompt/test identity mismatch")
    current = git_commit()
    if not is_ancestor(authorization["implementation_commit"], current):
        raise RuntimeError("Stage-A2 implementation commit is not an ancestor")
    schedule = read_json(SCHEDULE)
    stage_a2.validate_schedule(schedule)
    manifest = read_json(MANIFEST)
    if manifest["n_families"] != 20 or manifest["n_selected_rows"] != 20:
        raise RuntimeError("Stage-A2 manifest dimension mismatch")
    return authorization, schedule, manifest


def environment_record() -> dict[str, Any]:
    import torch
    import transformers

    return {
        "hostname": platform.node(),
        "architecture": platform.machine(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "dtype": "BF16",
        "attention": "SDPA",
        "torch_disable_native_jit": os.environ.get("TORCH_DISABLE_NATIVE_JIT", "(unset)"),
        "model_revision": q1s.MODEL_REVISION,
    }


def environment_fingerprint(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reconstruct_items(
    dataset_pages: Path, parquet: Path, manifest: dict[str, Any]
) -> dict[str, q1s.LiveCodeBenchItem]:
    if sha256(parquet) != q1s.LIVECODEBENCH_PARQUET_SHA256:
        raise RuntimeError("pinned LiveCodeBench parquet mismatch")
    all_items = {
        item.item_id: item
        for index, row in enumerate(load_dataset_pages(dataset_pages))
        for item in [q1s.normalize_livecodebench_row(row, index)]
    }
    selected: dict[str, q1s.LiveCodeBenchItem] = {}
    for family in manifest["ordered_families"]:
        expected = family["selected_item"]
        item = all_items[expected["item_id"]]
        actual = item.public_manifest_record()
        if (
            actual["item_id"] != expected["item_id"]
            or actual["item_sha256"] != expected["item_sha256"]
            or item.question_id != family["family_id"]
        ):
            raise RuntimeError("selected benchmark row differs from frozen Stage-A2 manifest")
        selected[item.item_id] = item
    if len(selected) != 20:
        raise RuntimeError("selected Stage-A2 row count mismatch")
    return selected


def _key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return stage_a2.logical_key(row)


def validate_completed(
    rows: list[dict[str, Any]], schedule_by_key: dict[tuple[str, str, str, int], dict[str, Any]]
) -> set[tuple[str, str, str, int]]:
    completed: set[tuple[str, str, str, int]] = set()
    for row in rows:
        key = _key(row)
        if key not in schedule_by_key or key in completed:
            raise RuntimeError("journal contains unexpected or duplicate logical key")
        locked = schedule_by_key[key]
        for field in ("family_id", "item_id", "item_sha256", "condition", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"journal lock mismatch: {field}")
        if not row.get("terminal_model_output", False):
            raise RuntimeError("journal row is not terminal")
        completed.add(key)
    return completed


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if platform.node() != "spark2":
        raise RuntimeError("Spark-2 host guard failed")
    if git_commit() != args.execution_commit:
        raise RuntimeError("runtime checkout differs from execution commit")
    authorization, schedule, manifest = validate_locks()
    environment = environment_record()
    fingerprint = environment_fingerprint(environment)
    if fingerprint != authorization["spark2_environment_fingerprint"]:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A2_ENVIRONMENT_DRIFT")
    if Path(args.model_path).name != f"Qwen3-8B-{q1s.MODEL_REVISION}":
        raise RuntimeError("model path does not identify frozen revision")
    selected = reconstruct_items(args.dataset_pages, args.parquet, manifest)
    if {row["item_id"] for row in schedule} != set(selected):
        raise RuntimeError("schedule/manifest item mismatch")
    a1 = read_json(REVIEW / "STAGE_A_FAMILY_MANIFEST.json")
    stage_b = read_json(REVIEW / "STAGE_B_FAMILY_MANIFEST.json")
    a2_families = {row["family_id"] for row in schedule}
    a1_families = {row["family_id"] for row in a1["ordered_families"]}
    b_families = {row["family_id"] for row in stage_b["ordered_families"]}
    if a2_families & a1_families or a2_families & b_families:
        raise RuntimeError("Stage-A2 family collision")
    a1_schedule = read_json(REVIEW / "STAGE_A_SCHEDULE.json")
    b_schedule = read_json(REVIEW / "STAGE_B_SCHEDULE.json")
    seeds = {int(row["seed"]) for row in schedule}
    if seeds & {int(row["seed"]) for row in [*a1_schedule, *b_schedule]}:
        raise RuntimeError("Stage-A2 seed collision")
    journal = args.output_dir / "journal.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise RuntimeError("Stage-A2 journal is not empty")
    report = {
        "classification": "Q1_SECOND_TASK_STAGE_A2_PREOPEN_PASS",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "execution_commit": args.execution_commit,
        "authorization_sha256": sha256(AUTHORIZATION),
        "reviewed_hashes_verified": True,
        "implementation_hashes_verified": True,
        "environment": environment,
        "environment_fingerprint": fingerprint,
        "families": 20,
        "selected_rows": 20,
        "logical_keys": 80,
        "unique_seeds": 80,
        "validity_evaluability_minimum_count": 38,
        "family_and_seed_splits_disjoint": True,
        "activation_hook_active": False,
        "meaningful_or_null_vector_loaded": False,
        "stage_b_conditions_present": False,
        "journal_empty_before_opening": True,
        "q2_paths_read": False,
        "q2_process_touched": False,
        "fresh_stage_a2_outcomes_before_opening": 0,
        "fresh_stage_a2_correctness_inspected": False,
    }
    write_json(args.output_dir / "PREOPEN_SEAL.json", report)
    return report


def execute(args: argparse.Namespace) -> None:
    authorization, schedule, manifest = validate_locks()
    preopen = read_json(args.output_dir / "PREOPEN_SEAL.json")
    if preopen["classification"] != "Q1_SECOND_TASK_STAGE_A2_PREOPEN_PASS":
        raise RuntimeError("Stage-A2 pre-opening seal is absent")
    if preopen["execution_commit"] != args.execution_commit:
        raise RuntimeError("pre-opening/execution commit mismatch")
    fingerprint = environment_fingerprint(environment_record())
    if fingerprint != authorization["spark2_environment_fingerprint"]:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A2_ENVIRONMENT_DRIFT")
    selected = reconstruct_items(args.dataset_pages, args.parquet, manifest)
    schedule_by_key = {_key(row): row for row in schedule}
    journal = args.output_dir / "journal.jsonl"
    completed = validate_completed(load_journal(journal), schedule_by_key)
    backend = build_backend(str(args.model_path))
    careful_prompt = read_json(REVIEW.parent / "CONTROLLER_PROVENANCE_LOCK.json")[
        "textual_careful"
    ]
    started = time.perf_counter()
    for locked in schedule:
        key = _key(locked)
        if key in completed:
            continue
        item = selected[locked["item_id"]]
        external = ExternalItem(
            item_id=item.item_id,
            benchmark="LiveCodeBench",
            subtask="test_output_prediction",
            prompt=item.prompt,
            reference_answer=item.reference_json,
            evaluator="livecodebench-exact-literal-stage-a2-v1",
            source_revision=q1s.LIVECODEBENCH_DATASET_REVISION,
            metadata={"response_channel": "external-semantic-v3"},
        )
        system_prompt = careful_prompt if locked["condition"] == "TEXTUAL_CAREFUL" else None
        row_started = time.perf_counter()
        try:
            output = backend.generate_reasoning(
                model_item(external, system_prompt=system_prompt),
                sampling_seed=int(locked["seed"]),
                max_new_tokens=4096,
                intervention_metadata={"intervention": "none", "stage": "STAGE_A2"},
            )
        except Exception as exc:
            append_jsonl(
                args.output_dir / "retry_ledger.jsonl",
                {
                    "logical_key": list(key),
                    "seed": int(locked["seed"]),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "failure_class": type(exc).__name__,
                    "terminal_scientific_output_persisted": False,
                },
            )
            raise
        metadata = dict(output.metadata)
        token_ids = [int(value) for value in metadata.get("generated_token_ids", [])]
        record = {
            **locked,
            "prompt_sha256": item.prompt_sha256,
            "reference_answer": item.reference_json,
            "raw_output": output.raw_output,
            "output_sha256": hashlib.sha256(output.raw_output.encode()).hexdigest(),
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "truncated": len(token_ids) >= 4096,
            "rendered_prompt_hash": metadata.get("rendered_prompt_hash"),
            "model": q1s.MODEL,
            "model_revision": q1s.MODEL_REVISION,
            "tokenizer_revision": q1s.TOKENIZER_REVISION,
            "parser_version": PARSER_VERSION,
            "environment_fingerprint": fingerprint,
            "intervention": "NONE",
            "activation_hook_active": False,
            "vector_hash": None,
            "layer": None,
            "terminal_model_output": True,
            "retry_provenance": "PRIMARY_OR_MISSING_KEY_RESUME",
            "elapsed_seconds": time.perf_counter() - row_started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        append_jsonl(journal, record)
        completed.add(key)
        print(f"completed={len(completed)} pending={80-len(completed)}", flush=True)
    final_completed = validate_completed(load_journal(journal), schedule_by_key)
    if len(final_completed) != 80:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A2_EXECUTION_INCOMPLETE")
    write_json(
        args.output_dir / "EXECUTION_COMPLETE.json",
        {
            "classification": "STAGE_A2_COLLECTION_COMPLETE_UNANALYZED",
            "completed": 80,
            "missing": 0,
            "duplicates": 0,
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "correctness_inspected": False,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-pages", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args)
    else:
        execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
