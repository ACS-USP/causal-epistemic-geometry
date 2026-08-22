#!/usr/bin/env python3
"""Crash-safe Gate 7 engineering checks and frozen scientific collection."""

from __future__ import annotations

import argparse
import json
import os
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

from run_gate6_2_first_stage_repair import (  # noqa: E402
    build_backend,
    load_external,
    model_item,
    prompt_tokens,
)

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL  # noqa: E402
from epistemic_geometry.experiments.gate7 import (  # noqa: E402
    CONDITIONS,
    ETA,
    LAYER,
    MAX_NEW_TOKENS,
    MEANINGFUL,
    MODEL,
    MODEL_REVISION,
    RANDOM_NAMES,
    REFERENCE_SCALE,
    TEXTUAL,
    file_sha256,
    vector_sha256,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review" / "gate7_fresh_l27_replication"
MEANINGFUL_PATH = (
    ROOT
    / "review"
    / "gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS"
    / "PROMPT_BOUNDARY"
    / "L27.npy"
)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_lock(review: Path, experiment_source_commit: str) -> dict[str, Any]:
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_PRE_OUTCOME" or lock["lifecycle"] != "PROSPECTIVE_LOCK":
        raise RuntimeError("Gate 7 protocol is not prospectively locked")
    if lock["instrument"]["evaluator"]["version"] != PARSER_VERSION:
        raise RuntimeError("runtime semantic parser differs from Gate 7 lock")
    if lock["instrument"]["evaluator"]["module_sha256"] != file_sha256(
        ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    ):
        raise RuntimeError("semantic V3 module hash differs from Gate 7 lock")
    if git_commit() != experiment_source_commit:
        raise RuntimeError(
            f"execution checkout {git_commit()} != experiment source {experiment_source_commit}"
        )
    binding = json.loads((review / "EXPERIMENT_SOURCE_COMMIT.json").read_text(encoding="utf-8"))
    if binding.get("experiment_source_commit") != experiment_source_commit:
        raise RuntimeError("experiment source binding differs from execution checkout")
    if binding.get("protocol_lock_sha256") != file_sha256(review / "PROTOCOL_LOCK.json"):
        raise RuntimeError("experiment source binding targets a different protocol lock")
    if lock["model"]["id"] != MODEL or lock["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("Gate 7 model provenance differs from frozen runner")
    return lock


def load_vectors(review: Path, lock: dict[str, Any]) -> dict[str, np.ndarray]:
    paths = {MEANINGFUL: MEANINGFUL_PATH}
    paths.update(
        {name: ROOT / lock["random_bank"]["records"][name]["vector_path"] for name in RANDOM_NAMES}
    )
    vectors: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        values = np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
        expected = (
            lock["controller"]["canonical_float64_vector_sha256"]
            if name == MEANINGFUL
            else lock["random_bank"]["records"][name]["canonical_float64_vector_sha256"]
        )
        if vector_sha256(values) != expected:
            raise RuntimeError(f"Gate 7 controller hash mismatch for {name}")
        vectors[name] = values
    return vectors


def deltas(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: values * ETA * REFERENCE_SCALE for name, values in vectors.items()}


def condition_context(
    backend: Any,
    item: Any,
    condition: str,
    controller_deltas: dict[str, np.ndarray],
    controller_hashes: dict[str, str],
) -> tuple[Any, Any, dict[str, Any]]:
    system = SYSTEM_CAREFUL if condition == TEXTUAL else None
    row = model_item(item, system)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    if condition not in controller_deltas:
        return (
            nullcontext(),
            row,
            {
                "prompt_length": len(prompt_ids),
                "rendered_prompt_hash_preflight": prompt_hash,
                "system_prompt": system,
                "intervention": "none",
            },
        )
    delta = controller_deltas[condition]
    tensor = backend.torch.tensor(delta, dtype=backend.torch.float32, device=backend.device).view(
        1, 1, -1
    )
    context = Gate6HookTrace(
        layers={LAYER: backend.layer_module(LAYER)},
        deltas={LAYER: tensor},
        target_positions=[len(prompt_ids) - 1],
    )
    return (
        context,
        row,
        {
            "prompt_length": len(prompt_ids),
            "rendered_prompt_hash_preflight": prompt_hash,
            "system_prompt": system,
            "intervention": condition,
            "intervention_layer": LAYER,
            "intervention_duration": "sustained_current_token",
            "intervention_scope": "final_prompt_token_then_current_decode_token",
            "intervention_vector_hash": controller_hashes[condition],
            "eta": ETA,
            "reference_scale": REFERENCE_SCALE,
            "delta_norm": float(np.linalg.norm(delta)),
        },
    )


def score(raw_output: str, reference: str, token_count: int) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        raw_output,
        reference,
        truncated=token_count >= MAX_NEW_TOKENS,
    )
    if result.correct:
        status = "VALID_CORRECT"
    elif result.commitment_valid and result.semantic_evaluable:
        status = "VALID_WRONG"
    elif result.failure_reason == "truncated or unclosed response":
        status = "TRUNCATED"
    else:
        status = "INVALID_FORMAT"
    return {
        "status": status,
        "correct": result.correct,
        "commitment_valid": result.commitment_valid,
        "semantic_evaluable": result.semantic_evaluable,
        "value_type": result.value_type,
        "canonical_value": result.canonical_value,
        "parsed_answer": result.payload,
        "parse_reason": result.failure_reason,
    }


def _completed(path: Path, expected_commit: str) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys: list[tuple[str, str, int]] = []
    for row in rows:
        if row.get("experiment_source_commit") != expected_commit:
            raise RuntimeError("journal mixes experiment source commits")
        if (
            row.get("model_revision") != MODEL_REVISION
            or row.get("parser_version") != PARSER_VERSION
        ):
            raise RuntimeError("journal mixes model/parser provenance")
        keys.append((str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])))
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 7 journal contains duplicate logical keys")
    return set(keys)


def _is_infrastructure_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = (
        "cuda",
        "out of memory",
        "cublas",
        "cudnn",
        "connection",
        "transport",
        "nccl",
        "device-side",
    )
    return any(marker in message for marker in markers)


def engineering_gate(
    backend: Any,
    review: Path,
    lock: dict[str, Any],
    controller_deltas: dict[str, np.ndarray],
    controller_hashes: dict[str, str],
) -> dict[str, Any]:
    old_manifest = ROOT / "review/gate6_2_first_stage_repair_mean_bridge/MANIPULATION_MANIFEST.json"
    items = load_external(old_manifest)[:5]
    checks: dict[str, Any] = {
        "engineering_prompt_count": len(items),
        "gate7_evaluation_items_used": False,
        "controller_checks": {},
    }
    identity_passes: list[bool] = []
    cleanup_passes: list[bool] = []
    all_trace_rows: list[dict[str, Any]] = []
    trace_forward_counts: list[int] = []
    exercised_conditions: set[str] = set()
    for index, item in enumerate(items):
        seed = 700_000 + index
        row = model_item(item)
        clean = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        prompt_ids, _rendered, _hash = prompt_tokens(backend, row)
        zero = backend.torch.zeros(
            (1, 1, len(next(iter(controller_deltas.values())))),
            dtype=backend.torch.float32,
            device=backend.device,
        )
        with Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: zero},
            target_positions=[len(prompt_ids) - 1],
        ) as zero_trace:
            identity = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        identity_passes.append(
            clean.metadata.get("generated_token_ids")
            == identity.metadata.get("generated_token_ids")
        )
        if zero_trace.forward_count < 1:
            raise RuntimeError("alpha-zero engineering trace observed no forward")

        conditions_to_exercise = (MEANINGFUL, *RANDOM_NAMES) if index == 0 else (MEANINGFUL,)
        for offset, condition in enumerate(conditions_to_exercise):
            context, steered_row, _meta = condition_context(
                backend, item, condition, controller_deltas, controller_hashes
            )
            with context as trace:
                backend.generate_reasoning(
                    steered_row,
                    sampling_seed=seed + 100 + offset,
                    max_new_tokens=16,
                )
            trace_meta = trace.metadata()
            all_trace_rows.extend(trace_meta["applications"])
            trace_forward_counts.append(int(trace_meta["forward_count"]))
            exercised_conditions.add(condition)
        clean_after = backend.generate_reasoning(row, sampling_seed=seed, max_new_tokens=16)
        cleanup_passes.append(
            clean.metadata.get("generated_token_ids")
            == clean_after.metadata.get("generated_token_ids")
        )

    for condition in (MEANINGFUL, *RANDOM_NAMES):
        delta = controller_deltas[condition]
        checks["controller_checks"][condition] = {
            "layer": LAYER,
            "delta_norm": float(np.linalg.norm(delta)),
            "duration": "sustained_current_token",
            "scope": "current_token",
            "vector_hash": controller_hashes[condition],
        }
    delta_norms = [record["delta_norm"] for record in checks["controller_checks"].values()]
    max_abs_shift_error = max(float(row["shift_error"]) for row in all_trace_rows)
    max_relative_shift_error = max(float(row["relative_shift_error"]) for row in all_trace_rows)
    checks.update(
        {
            "alpha_zero_identity": all(identity_passes),
            "hook_cleanup": all(cleanup_passes),
            "per_forward_exact_shift": bool(all_trace_rows) and max_relative_shift_error <= 2.0,
            "max_abs_shift_error": max_abs_shift_error,
            "max_relative_shift_error_bf16_eps": max_relative_shift_error,
            "bf16_relative_tolerance": 2.0,
            "current_token_scope": bool(all_trace_rows)
            and max(abs(float(row["non_current_change"])) for row in all_trace_rows) <= 0.125,
            "one_application_per_forward": bool(trace_forward_counts)
            and len(all_trace_rows) == sum(trace_forward_counts),
            "cache_safety": any(int(row["sequence_length"]) == 1 for row in all_trace_rows),
            "hook_trace_application_count": len(all_trace_rows),
            "hook_trace_forward_count": sum(trace_forward_counts),
            "exercised_controller_conditions": sorted(exercised_conditions),
            "all_controllers_exercised": exercised_conditions == {MEANINGFUL, *RANDOM_NAMES},
            "random_matching": float(np.ptp(delta_norms)) <= 1e-9,
            "condition_metadata": all(
                set(record) >= {"layer", "delta_norm", "duration", "scope", "vector_hash"}
                for record in checks["controller_checks"].values()
            ),
            "environment_profile": lock["model"]["environment_profile"],
        }
    )
    required = (
        "alpha_zero_identity",
        "hook_cleanup",
        "per_forward_exact_shift",
        "current_token_scope",
        "one_application_per_forward",
        "cache_safety",
        "random_matching",
        "condition_metadata",
        "all_controllers_exercised",
    )
    checks["pass"] = all(bool(checks[key]) for key in required)
    checks["classification"] = (
        "GATE7_ENGINEERING_PASS" if checks["pass"] else "GATE7_ENGINE_FAILURE"
    )
    write_json(review / "ENGINEERING_CHECKS.json", checks)
    if not checks["pass"]:
        raise RuntimeError("GATE7_ENGINE_FAILURE")
    return checks


def collect(
    backend: Any,
    review: Path,
    lock: dict[str, Any],
    controller_deltas: dict[str, np.ndarray],
    controller_hashes: dict[str, str],
    experiment_source_commit: str,
) -> None:
    manifest = review / "EVALUATION_MANIFEST.json"
    schedule_path = review / "EVALUATION_SCHEDULE.json"
    if file_sha256(manifest) != lock["sample"]["manifest_file_sha256"]:
        raise RuntimeError("Gate 7 manifest hash differs from lock")
    if file_sha256(schedule_path) != lock["schedule"]["file_sha256"]:
        raise RuntimeError("Gate 7 schedule hash differs from lock")
    items = load_external(manifest)
    item_by_id = {item.item_id: item for item in items}
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if len(schedule) != lock["schedule"]["logical_rows"]:
        raise RuntimeError("Gate 7 schedule row count differs from lock")
    if {row["condition"] for row in schedule} != set(CONDITIONS):
        raise RuntimeError("Gate 7 schedule condition set differs from lock")
    journal = review / "journal.jsonl"
    retry_ledger = review / "RETRY_LEDGER.jsonl"
    completed = _completed(journal, experiment_source_commit)
    for schedule_index, row in enumerate(schedule):
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in completed:
            continue
        item = item_by_id[key[0]]
        condition = key[1]
        context, model_row, context_meta = condition_context(
            backend, item, condition, controller_deltas, controller_hashes
        )
        started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=int(row["seed"]),
                    max_new_tokens=MAX_NEW_TOKENS,
                    intervention_metadata={
                        "experiment_id": lock["experiment_id"],
                        "experiment_source_commit": experiment_source_commit,
                        "condition": condition,
                        "intervention": condition if condition in controller_deltas else "none",
                        "intervention_duration": (
                            "sustained_current_token" if condition in controller_deltas else "none"
                        ),
                        "intervention_layer": LAYER if condition in controller_deltas else None,
                        "intervention_vector_hash": context_meta.get("intervention_vector_hash"),
                        "eta": ETA if condition in controller_deltas else None,
                        "parser_version": PARSER_VERSION,
                        "environment_profile": "CORE_QWEN",
                    },
                )
            elapsed = time.perf_counter() - started
            metadata = dict(output.metadata)
            if condition in controller_deltas:
                metadata["intervention_forward_trace"] = trace.metadata()
            token_count = int(metadata.get("generated_token_count", 0))
            scored = score(output.raw_output, item.reference_answer, token_count)
            record = {
                **row,
                **scored,
                "experiment_source_commit": experiment_source_commit,
                "runtime_source_commit": git_commit(),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "raw_output": output.raw_output,
                "generated_token_ids": metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "reference_answer": item.reference_answer,
                "reference_canonical_type": item.metadata.get("reference_canonical_type"),
                "evaluator": item.evaluator,
                "prompt_hash": item.prompt_hash,
                "rendered_prompt_hash": metadata.get("rendered_prompt_hash"),
                "source_revision": item.source_revision,
                "item_metadata": dict(item.metadata),
                "condition_metadata": context_meta,
                "backend_metadata": metadata,
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "retry_count": 0,
                "schedule_index": schedule_index,
            }
        except RuntimeError as exc:
            elapsed = time.perf_counter() - started
            if _is_infrastructure_failure(exc):
                append_jsonl(
                    retry_ledger,
                    {
                        **row,
                        "classification": "INFRASTRUCTURE_ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "elapsed_seconds": elapsed,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "scientific_row_written": False,
                    },
                )
                raise
            record = {
                **row,
                "status": "RUNTIME_ERROR",
                "correct": False,
                "commitment_valid": False,
                "semantic_evaluable": False,
                "raw_output": "",
                "generated_token_ids": [],
                "generated_token_count": 0,
                "parsed_answer": None,
                "parse_reason": str(exc),
                "experiment_source_commit": experiment_source_commit,
                "runtime_source_commit": git_commit(),
                "model": MODEL,
                "model_revision": MODEL_REVISION,
                "tokenizer_revision": MODEL_REVISION,
                "parser_version": PARSER_VERSION,
                "reference_answer": item.reference_answer,
                "reference_canonical_type": item.metadata.get("reference_canonical_type"),
                "evaluator": item.evaluator,
                "prompt_hash": item.prompt_hash,
                "source_revision": item.source_revision,
                "item_metadata": dict(item.metadata),
                "condition_metadata": context_meta,
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "retry_count": 0,
                "schedule_index": schedule_index,
            }
        append_jsonl(journal, record)
        completed.add(key)
        if len(completed) % 100 == 0:
            print(json.dumps({"health": "running", "completed_rows": len(completed)}), flush=True)
    write_json(
        review / "COLLECTION_METADATA.json",
        {
            "experiment_source_commit": experiment_source_commit,
            "runtime_source_commit": git_commit(),
            "completed_rows": len(completed),
            "expected_rows": lock["schedule"]["logical_rows"],
            "model": MODEL,
            "model_revision": MODEL_REVISION,
            "parser_version": PARSER_VERSION,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("engineering", "collect"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    parser.add_argument("--experiment-source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 7 {args.mode}")
    review = args.review_dir.resolve()
    lock = load_lock(review, args.experiment_source_commit)
    vectors = load_vectors(review, lock)
    controller_deltas = deltas(vectors)
    controller_hashes = {name: vector_sha256(vector) for name, vector in vectors.items()}
    backend = build_backend(args.model_path)
    if args.mode == "engineering":
        result = engineering_gate(backend, review, lock, controller_deltas, controller_hashes)
        print(json.dumps({"classification": result["classification"]}, indent=2))
    else:
        engineering = json.loads((review / "ENGINEERING_CHECKS.json").read_text(encoding="utf-8"))
        if engineering.get("classification") != "GATE7_ENGINEERING_PASS":
            raise RuntimeError("Gate 7 collection requires a passed engineering gate")
        collect(
            backend,
            review,
            lock,
            controller_deltas,
            controller_hashes,
            args.experiment_source_commit,
        )
        print(json.dumps({"collection": "complete", "rows": lock["schedule"]["logical_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
