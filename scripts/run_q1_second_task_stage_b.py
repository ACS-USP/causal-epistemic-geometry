#!/usr/bin/env python3
"""Preflight and serial collection for frozen Q1 LiveCodeBench Stage B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gate7_fresh_l27_replication as qwen_runner  # noqa: E402

from epistemic_geometry.benchmarks.external.base import ExternalItem  # noqa: E402
from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_b as stage_b  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design"
AMENDMENT = REVIEW / "amendment1_hierarchical_unit"
AUTHORIZATION = AMENDMENT / "stage_b_preopen/STAGE_B_PRINCIPAL_AUTHORIZATION.json"
ADDENDUM = AMENDMENT / "stage_b_preopen/STAGE_B_PARSER_AND_EXECUTION_ADDENDUM.json"
MANIFEST = AMENDMENT / "STAGE_B_FAMILY_MANIFEST.json"
SCHEDULE = AMENDMENT / "STAGE_B_SCHEDULE.json"
CONTROLLER = REVIEW / "CONTROLLER_PROVENANCE_LOCK.json"
NULL_BANK = REVIEW / "RANDOM_BANK_LOCK.json"
STAGE_A2_CLOSEOUT = AMENDMENT / "stage_a2_closeout/STAGE_A2_CLOSEOUT.json"
PARSER_SOURCE = ROOT / "src/epistemic_geometry/experiments/q1_second_task_stage_a_failure.py"
PARSER_VERSION = "TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_locks() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    authorization = read_json(AUTHORIZATION)
    addendum = read_json(ADDENDUM)
    actual = {
        "stage_b_manifest": sha256(MANIFEST),
        "stage_b_schedule": sha256(SCHEDULE),
        "controller_lock": sha256(CONTROLLER),
        "null_bank_lock": sha256(NULL_BANK),
        "parser_addendum": sha256(ADDENDUM),
        "stage_a2_closeout": sha256(STAGE_A2_CLOSEOUT),
    }
    if actual != authorization["frozen_hashes"]:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_B_PREOPEN_HASH_MISMATCH")
    if authorization["stage_b"] != "AUTHORIZED_NOT_YET_OPENED":
        raise RuntimeError("Stage-B authorization is absent")
    if addendum["stage_b_outcomes_before_addendum"] != 0:
        raise RuntimeError("Stage-B parser addendum was not pre-outcome")
    if sha256(PARSER_SOURCE) != addendum["parser_source_sha256"]:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_B_PARSER_FREEZE_FAILURE")
    schedule = read_json(SCHEDULE)
    stage_b.validate_schedule(schedule)
    manifest = read_json(MANIFEST)
    if manifest["n_families"] != 130 or manifest["n_selected_rows"] != 130:
        raise RuntimeError("Stage-B manifest dimension mismatch")
    return authorization, schedule, manifest


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
            raise RuntimeError("selected benchmark row differs from frozen Stage-B manifest")
        selected[item.item_id] = item
    if len(selected) != 130:
        raise RuntimeError("selected Stage-B row count mismatch")
    return selected


def load_deltas() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    controller = read_json(CONTROLLER)
    bank = read_json(NULL_BANK)
    records = {
        "MEANINGFUL_FIXED_QWEN_L27_D75": {
            "path": controller["vector_path"],
            "file_sha256": controller["vector_file_sha256"],
            "vector_hash": controller["vector_hash"],
        },
        **{
            name: {
                "path": record["vector_path"],
                "file_sha256": record["file_sha256"],
                "vector_hash": record["canonical_float64_vector_sha256"],
            }
            for name, record in bank["records"].items()
        },
    }
    scale = float(controller["effective_delta_norm"])
    deltas: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for name, record in records.items():
        path = ROOT / record["path"]
        value = np.load(path, allow_pickle=False).astype(np.float64)
        if sha256(path) != record["file_sha256"] or vector_sha256(value) != record["vector_hash"]:
            raise RuntimeError(f"Stage-B vector hash mismatch: {name}")
        if not np.isclose(np.linalg.norm(value), 1.0, atol=1e-12, rtol=0):
            raise RuntimeError(f"Stage-B vector is not unit normalized: {name}")
        deltas[name] = value * scale
        hashes[name] = record["vector_hash"]
    return deltas, hashes


def validate_completed(
    rows: list[dict[str, Any]], schedule_by_key: dict[tuple[str, str, str, int], dict[str, Any]]
) -> set[tuple[str, str, str, int]]:
    completed: set[tuple[str, str, str, int]] = set()
    for row in rows:
        key = stage_b.logical_key(row)
        if key not in schedule_by_key or key in completed:
            raise RuntimeError("Stage-B journal contains unexpected or duplicate logical key")
        locked = schedule_by_key[key]
        for field in ("family_id", "item_id", "item_sha256", "condition", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"Stage-B journal lock mismatch: {field}")
        if not row.get("terminal_model_output", False):
            raise RuntimeError("Stage-B journal row is not terminal")
        completed.add(key)
    return completed


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    if platform.node() != "spark2" or git_commit() != args.execution_commit:
        raise RuntimeError("Spark-2 host or execution-commit guard failed")
    if args.model_path.name != f"Qwen3-8B-{q1s.MODEL_REVISION}":
        raise RuntimeError("model path does not identify the frozen revision")
    authorization, schedule, manifest = validate_locks()
    environment = environment_record()
    fingerprint = environment_fingerprint(environment)
    if fingerprint != authorization["spark2_environment_fingerprint"]:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_B_ENVIRONMENT_DRIFT")
    selected = reconstruct_items(args.dataset_pages, args.parquet, manifest)
    if {row["item_id"] for row in schedule} != set(selected):
        raise RuntimeError("Stage-B schedule/manifest item mismatch")
    a1 = read_json(AMENDMENT / "STAGE_A_FAMILY_MANIFEST.json")
    a2 = read_json(AMENDMENT / "stage_a_failure_audit/STAGE_A2_FAMILY_MANIFEST.json")
    b_families = {row["family_id"] for row in schedule}
    earlier_families = {
        *(row["family_id"] for row in a1["ordered_families"]),
        *(row["family_id"] for row in a2["ordered_families"]),
    }
    if b_families & earlier_families:
        raise RuntimeError("Stage-B family collision with Stage A1/A2")
    earlier_schedules = [
        *read_json(AMENDMENT / "STAGE_A_SCHEDULE.json"),
        *read_json(AMENDMENT / "stage_a_failure_audit/STAGE_A2_SCHEDULE.json"),
    ]
    if {int(row["seed"]) for row in schedule} & {int(row["seed"]) for row in earlier_schedules}:
        raise RuntimeError("Stage-B seed collision with Stage A1/A2")
    deltas, hashes = load_deltas()
    if set(deltas) != {"MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES}:
        raise RuntimeError("Stage-B vector condition set mismatch")
    journal = args.output_dir / "journal.jsonl"
    if journal.exists() and journal.stat().st_size:
        raise RuntimeError("Stage-B journal is not empty")
    report = {
        "classification": "Q1_SECOND_TASK_STAGE_B_PREOPEN_PASS",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "execution_commit": args.execution_commit,
        "authorization_sha256": sha256(AUTHORIZATION),
        "addendum_sha256": sha256(ADDENDUM),
        "frozen_hashes_verified": True,
        "environment": environment,
        "environment_fingerprint": fingerprint,
        "families": 130,
        "selected_rows": 130,
        "logical_keys": 5720,
        "unique_seeds": 5720,
        "conditions": list(stage_b.CONDITIONS),
        "rollouts": 4,
        "family_and_seed_splits_disjoint": True,
        "controller_and_null_hashes": hashes,
        "effective_delta_norms": {
            name: float(np.linalg.norm(value)) for name, value in deltas.items()
        },
        "activation_hook_preopen_active": False,
        "journal_empty_before_opening": True,
        "q2_paths_read": False,
        "q2_process_touched": False,
        "stage_b_outcomes_before_opening": 0,
        "stage_b_correctness_inspected": False,
    }
    write_json(args.output_dir / "PREOPEN_SEAL.json", report)
    return report


def condition_context(
    backend: Any,
    item: ExternalItem,
    condition: str,
    deltas: dict[str, np.ndarray],
    hashes: dict[str, str],
    careful_prompt: str,
) -> tuple[Any, Any, str, dict[str, Any]]:
    system = careful_prompt if condition == "TEXTUAL_CAREFUL" else None
    row = qwen_runner.model_item(item, system)
    prompt_ids, _rendered, rendered_hash = qwen_runner.prompt_tokens(backend, row)
    if condition not in deltas:
        return nullcontext(), row, rendered_hash, {"intervention": "none"}
    tensor = backend.torch.tensor(
        deltas[condition], dtype=backend.torch.float32, device=backend.device
    ).view(1, 1, -1)
    context = Gate6HookTrace(
        layers={27: backend.layer_module(27)},
        deltas={27: tensor},
        target_positions=[len(prompt_ids) - 1],
    )
    return (
        context,
        row,
        rendered_hash,
        {
            "intervention": condition,
            "intervention_layer": 27,
            "intervention_duration": "sustained_current_token",
            "intervention_scope": "final_prompt_token_then_current_decode_token",
            "intervention_vector_hash": hashes[condition],
            "effective_delta_norm": float(np.linalg.norm(deltas[condition])),
        },
    )


def execute(args: argparse.Namespace) -> None:
    authorization, schedule, manifest = validate_locks()
    preopen = read_json(args.output_dir / "PREOPEN_SEAL.json")
    if preopen["classification"] != "Q1_SECOND_TASK_STAGE_B_PREOPEN_PASS":
        raise RuntimeError("Stage-B pre-opening seal is absent")
    if (
        environment_fingerprint(environment_record())
        != authorization["spark2_environment_fingerprint"]
    ):
        raise RuntimeError("Q1_SECOND_TASK_STAGE_B_ENVIRONMENT_DRIFT")
    selected = reconstruct_items(args.dataset_pages, args.parquet, manifest)
    schedule_by_key = {stage_b.logical_key(row): row for row in schedule}
    journal = args.output_dir / "journal.jsonl"
    completed = validate_completed(load_journal(journal), schedule_by_key)
    backend = qwen_runner.build_backend(str(args.model_path))
    deltas, hashes = load_deltas()
    careful_prompt = read_json(CONTROLLER)["textual_careful"]
    started = time.perf_counter()
    for locked in schedule:
        if (args.output_dir / "STOP_AFTER_CURRENT").exists():
            write_json(
                args.output_dir / "RESOURCE_PREEMPTED_CLEANLY.json",
                {
                    "classification": "EXTERNAL_RESOURCE_PREEMPTION_CLEAN_STOP",
                    "completed": len(completed),
                    "pending": 5720 - len(completed),
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "scientific_metrics_inspected": False,
                },
            )
            return
        key = stage_b.logical_key(locked)
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
        context, model_row, rendered_hash, intervention = condition_context(
            backend, external, str(locked["condition"]), deltas, hashes, careful_prompt
        )
        row_started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=int(locked["seed"]),
                    max_new_tokens=4096,
                    intervention_metadata=intervention,
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
        if str(locked["condition"]) in deltas:
            metadata["intervention_forward_trace"] = trace.metadata()
        token_ids = [int(value) for value in metadata.get("generated_token_ids", [])]
        record = {
            **locked,
            "execution_commit": args.execution_commit,
            "prompt_sha256": item.prompt_sha256,
            "reference_answer": item.reference_json,
            "raw_output": output.raw_output,
            "output_sha256": hashlib.sha256(output.raw_output.encode()).hexdigest(),
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "truncated": len(token_ids) >= 4096,
            "rendered_prompt_hash": rendered_hash,
            "model": q1s.MODEL,
            "model_revision": q1s.MODEL_REVISION,
            "tokenizer_revision": q1s.TOKENIZER_REVISION,
            "parser_version": PARSER_VERSION,
            "environment_fingerprint": preopen["environment_fingerprint"],
            "intervention": intervention["intervention"],
            "activation_hook_active": str(locked["condition"]) in deltas,
            "vector_hash": hashes.get(str(locked["condition"])),
            "eta": q1s.ETA if str(locked["condition"]) in deltas else None,
            "reference_scale": (
                q1s.REFERENCE_SCALE if str(locked["condition"]) in deltas else None
            ),
            "layer": q1s.LAYER if str(locked["condition"]) in deltas else None,
            "condition_metadata": intervention,
            "backend_metadata": metadata,
            "terminal_model_output": True,
            "retry_provenance": "PRIMARY_OR_MISSING_KEY_RESUME",
            "elapsed_seconds": time.perf_counter() - row_started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        append_jsonl(journal, record)
        completed.add(key)
        print(f"completed={len(completed)} pending={5720 - len(completed)}", flush=True)
    final_completed = validate_completed(load_journal(journal), schedule_by_key)
    if len(final_completed) != 5720:
        raise RuntimeError("Q1_SECOND_TASK_EXECUTION_INCOMPLETE")
    write_json(
        args.output_dir / "COLLECTION_COMPLETE_SEAL.json",
        {
            "classification": "STAGE_B_COLLECTION_COMPLETE_UNANALYZED",
            "completed": 5720,
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
