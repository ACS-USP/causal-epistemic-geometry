#!/usr/bin/env python3
"""Crash-safe Gate 8 engineering checks and matched dose calibration."""

from __future__ import annotations

import argparse
import json
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
from run_gate7_fresh_l27_replication import (  # noqa: E402
    _is_infrastructure_failure,
    append_jsonl,
    git_commit,
    score,
    write_json,
)

from epistemic_geometry.benchmarks.external.semantic_v3 import PARSER_VERSION  # noqa: E402
from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL  # noqa: E402
from epistemic_geometry.experiments.gate7 import (  # noqa: E402
    LAYER,
    MAX_NEW_TOKENS,
    MODEL,
    MODEL_REVISION,
    REFERENCE_SCALE,
)
from epistemic_geometry.experiments.gate8 import (  # noqa: E402
    CONDITIONS,
    EXPERIMENT_ID,
    MEANINGFUL_VECTOR,
    RANDOM_VECTOR_NAMES,
    TEXTUAL,
    condition_spec,
    file_sha256,
    vector_sha256,
)
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

REVIEW = ROOT / "review/gate8_l27_dose_calibration"
MEANINGFUL_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def load_lock(review: Path, experiment_source_commit: str) -> dict[str, Any]:
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if lock["status"] != "FROZEN_PRE_OUTCOME" or lock["lifecycle"] != "PROSPECTIVE_LOCK":
        raise RuntimeError("Gate 8 protocol is not prospectively locked")
    evaluator = lock["instrument"]["evaluator"]
    if evaluator["version"] != PARSER_VERSION:
        raise RuntimeError("runtime semantic parser differs from Gate 8 lock")
    if evaluator["module_sha256"] != file_sha256(
        ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    ):
        raise RuntimeError("semantic V3 module hash differs from Gate 8 lock")
    if git_commit() != experiment_source_commit:
        raise RuntimeError("execution checkout differs from Gate 8 source binding")
    binding = json.loads((review / "EXPERIMENT_SOURCE_COMMIT.json").read_text(encoding="utf-8"))
    if binding.get("experiment_source_commit") != experiment_source_commit:
        raise RuntimeError("Gate 8 source binding differs from checkout")
    if binding.get("protocol_lock_sha256") != file_sha256(review / "PROTOCOL_LOCK.json"):
        raise RuntimeError("Gate 8 source binding targets a different lock")
    if lock["model"]["id"] != MODEL or lock["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("Gate 8 model provenance differs from frozen runner")
    return lock


def load_vectors(review: Path, lock: dict[str, Any]) -> dict[str, np.ndarray]:
    paths = {MEANINGFUL_VECTOR: MEANINGFUL_PATH}
    paths.update(
        {
            name: ROOT / lock["random_bank"]["records"][name]["vector_path"]
            for name in RANDOM_VECTOR_NAMES
        }
    )
    vectors: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        vector = np.load(path, allow_pickle=False).astype(np.float64).reshape(-1)
        expected = (
            lock["controller"]["canonical_float64_vector_sha256"]
            if name == MEANINGFUL_VECTOR
            else lock["random_bank"]["records"][name]["canonical_float64_vector_sha256"]
        )
        if vector_sha256(vector) != expected:
            raise RuntimeError(f"Gate 8 vector hash mismatch for {name}")
        vectors[name] = vector
    return vectors


def condition_deltas(vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        spec = condition_spec(condition)
        if spec["kind"] in {"meaningful", "random"}:
            result[condition] = vectors[spec["vector"]] * float(spec["eta"]) * REFERENCE_SCALE
    return result


def condition_hashes(vectors: dict[str, np.ndarray]) -> dict[str, str]:
    return {name: vector_sha256(vector) for name, vector in vectors.items()}


def condition_context(
    backend: Any,
    item: Any,
    condition: str,
    deltas: dict[str, np.ndarray],
    hashes: dict[str, str],
) -> tuple[Any, Any, dict[str, Any]]:
    spec = condition_spec(condition)
    system = SYSTEM_CAREFUL if condition == TEXTUAL else None
    row = model_item(item, system)
    prompt_ids, _rendered, prompt_hash = prompt_tokens(backend, row)
    metadata = {
        "prompt_length": len(prompt_ids),
        "rendered_prompt_hash_preflight": prompt_hash,
        "system_prompt": system,
        "condition": condition,
        "dose": spec["dose"],
        "dose_fraction": spec["fraction"],
        "eta": spec["eta"],
        "seed_regime": "MATCHED_COUPLING_CALIBRATION",
    }
    if condition not in deltas:
        return nullcontext(), row, {**metadata, "intervention": "none"}
    delta = deltas[condition]
    vector_name = str(spec["vector"])
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
            **metadata,
            "intervention": condition,
            "intervention_layer": LAYER,
            "intervention_duration": "sustained_current_token",
            "intervention_scope": "final_prompt_token_then_current_decode_token",
            "intervention_vector_name": vector_name,
            "intervention_vector_hash": hashes[vector_name],
            "reference_scale": REFERENCE_SCALE,
            "delta_norm": float(np.linalg.norm(delta)),
        },
    )


def _completed(path: Path, expected_commit: str) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    keys: list[tuple[str, str, int]] = []
    for row in rows:
        if row.get("experiment_source_commit") != expected_commit:
            raise RuntimeError("Gate 8 journal mixes experiment source commits")
        if (
            row.get("model_revision") != MODEL_REVISION
            or row.get("parser_version") != PARSER_VERSION
        ):
            raise RuntimeError("Gate 8 journal mixes model/parser provenance")
        keys.append((str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])))
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 8 journal contains duplicate logical keys")
    return set(keys)


def engineering_gate(
    backend: Any,
    review: Path,
    lock: dict[str, Any],
    deltas: dict[str, np.ndarray],
    hashes: dict[str, str],
) -> dict[str, Any]:
    old_manifest = ROOT / "review/gate6_2_first_stage_repair_mean_bridge/MANIPULATION_MANIFEST.json"
    items = load_external(old_manifest)[:5]
    checks: dict[str, Any] = {
        "engineering_prompt_count": len(items),
        "gate8_calibration_items_used": False,
        "condition_checks": {},
    }
    identity_passes: list[bool] = []
    cleanup_passes: list[bool] = []
    applications: list[dict[str, Any]] = []
    forward_counts: list[int] = []
    exercised: set[str] = set()
    for index, item in enumerate(items):
        seed = 800_000 + index
        clean_row = model_item(item)
        clean = backend.generate_reasoning(clean_row, sampling_seed=seed, max_new_tokens=16)
        prompt_ids, _rendered, _hash = prompt_tokens(backend, clean_row)
        zero = backend.torch.zeros((1, 1, 4096), dtype=backend.torch.float32, device=backend.device)
        with Gate6HookTrace(
            layers={LAYER: backend.layer_module(LAYER)},
            deltas={LAYER: zero},
            target_positions=[len(prompt_ids) - 1],
        ) as zero_trace:
            identity = backend.generate_reasoning(clean_row, sampling_seed=seed, max_new_tokens=16)
        identity_passes.append(
            clean.metadata.get("generated_token_ids")
            == identity.metadata.get("generated_token_ids")
        )
        if zero_trace.forward_count < 1:
            raise RuntimeError("Gate 8 alpha-zero trace observed no forward")
        exercise = (
            tuple(deltas)
            if index == 0
            else tuple(condition for condition in deltas if condition.startswith("MEAN_"))
        )
        for offset, condition in enumerate(exercise):
            context, row, _metadata = condition_context(backend, item, condition, deltas, hashes)
            with context as trace:
                backend.generate_reasoning(row, sampling_seed=seed + 100 + offset, max_new_tokens=8)
            trace_metadata = trace.metadata()
            applications.extend(trace_metadata["applications"])
            forward_counts.append(int(trace_metadata["forward_count"]))
            exercised.add(condition)
        clean_after = backend.generate_reasoning(clean_row, sampling_seed=seed, max_new_tokens=16)
        cleanup_passes.append(
            clean.metadata.get("generated_token_ids")
            == clean_after.metadata.get("generated_token_ids")
        )

    full_delta_norm = float(np.linalg.norm(deltas["MEAN_D100"]))
    for condition, delta in deltas.items():
        spec = condition_spec(condition)
        checks["condition_checks"][condition] = {
            "layer": LAYER,
            "dose_fraction": spec["fraction"],
            "eta": spec["eta"],
            "delta_norm": float(np.linalg.norm(delta)),
            "expected_delta_norm": full_delta_norm * float(spec["fraction"]),
            "linear_scaling": bool(
                np.isclose(
                    float(np.linalg.norm(delta)),
                    full_delta_norm * float(spec["fraction"]),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
            "vector_hash": hashes[str(spec["vector"])],
            "duration": "sustained_current_token",
            "scope": "current_token",
        }
    max_relative = max(float(row["relative_shift_error"]) for row in applications)
    per_dose_matching = {
        dose: float(
            np.ptp(
                [
                    checks["condition_checks"][condition]["delta_norm"]
                    for condition in deltas
                    if condition.endswith(dose)
                ]
            )
        )
        for dose in ("D25", "D50", "D75", "D100")
    }
    checks.update(
        {
            "alpha_zero_identity": all(identity_passes),
            "linear_dose_scaling": all(
                record["linear_scaling"] for record in checks["condition_checks"].values()
            ),
            "random_matching_by_dose": all(value <= 1e-9 for value in per_dose_matching.values()),
            "random_matching_norm_ptp": per_dose_matching,
            "per_forward_exact_shift": bool(applications) and max_relative <= 2.0,
            "max_relative_shift_error_bf16_eps": max_relative,
            "current_token_scope": bool(applications)
            and max(abs(float(row["non_current_change"])) for row in applications) <= 0.125,
            "one_application_per_forward": len(applications) == sum(forward_counts),
            "cache_safety": any(int(row["sequence_length"]) == 1 for row in applications),
            "hook_cleanup": all(cleanup_passes),
            "all_controller_conditions_exercised": exercised == set(deltas),
            "condition_metadata": all(
                set(record)
                >= {
                    "layer",
                    "dose_fraction",
                    "eta",
                    "delta_norm",
                    "vector_hash",
                    "duration",
                    "scope",
                }
                for record in checks["condition_checks"].values()
            ),
            "environment_profile": lock["model"]["environment_profile"],
        }
    )
    required = (
        "alpha_zero_identity",
        "linear_dose_scaling",
        "random_matching_by_dose",
        "per_forward_exact_shift",
        "current_token_scope",
        "one_application_per_forward",
        "cache_safety",
        "hook_cleanup",
        "all_controller_conditions_exercised",
        "condition_metadata",
    )
    checks["pass"] = all(bool(checks[name]) for name in required)
    checks["classification"] = (
        "GATE8_ENGINEERING_PASS" if checks["pass"] else "GATE8_ENGINE_FAILURE"
    )
    write_json(review / "ENGINEERING_CHECKS.json", checks)
    if not checks["pass"]:
        raise RuntimeError("GATE8_ENGINE_FAILURE")
    return checks


def collect(
    backend: Any,
    review: Path,
    lock: dict[str, Any],
    deltas: dict[str, np.ndarray],
    hashes: dict[str, str],
    experiment_source_commit: str,
) -> None:
    manifest = review / "CALIBRATION_MANIFEST.json"
    schedule_path = review / "CALIBRATION_SCHEDULE.json"
    if file_sha256(manifest) != lock["sample"]["manifest_file_sha256"]:
        raise RuntimeError("Gate 8 manifest hash differs from lock")
    if file_sha256(schedule_path) != lock["schedule"]["file_sha256"]:
        raise RuntimeError("Gate 8 schedule hash differs from lock")
    items = load_external(manifest)
    item_by_id = {item.item_id: item for item in items}
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if len(schedule) != 2200 or {row["condition"] for row in schedule} != set(CONDITIONS):
        raise RuntimeError("Gate 8 schedule is not the frozen 2,200-row condition set")
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
            backend, item, condition, deltas, hashes
        )
        started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=int(row["seed"]),
                    max_new_tokens=MAX_NEW_TOKENS,
                    intervention_metadata={
                        "experiment_id": EXPERIMENT_ID,
                        "experiment_source_commit": experiment_source_commit,
                        "condition": condition,
                        "dose": row["dose"],
                        "dose_fraction": row["dose_fraction"],
                        "eta": row["eta"],
                        "seed_block_id": row["seed_block_id"],
                        "seed_regime": "MATCHED_COUPLING_CALIBRATION",
                        "intervention": condition if condition in deltas else "none",
                        "intervention_duration": (
                            "sustained_current_token" if condition in deltas else "none"
                        ),
                        "intervention_layer": LAYER if condition in deltas else None,
                        "intervention_vector_hash": context_meta.get("intervention_vector_hash"),
                        "parser_version": PARSER_VERSION,
                        "environment_profile": "CORE_QWEN",
                    },
                )
            elapsed = time.perf_counter() - started
            metadata = dict(output.metadata)
            if condition in deltas:
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
    require_remote_hf_execution(f"Gate 8 {args.mode}")
    review = args.review_dir.resolve()
    lock = load_lock(review, args.experiment_source_commit)
    vectors = load_vectors(review, lock)
    deltas = condition_deltas(vectors)
    hashes = condition_hashes(vectors)
    backend = build_backend(args.model_path)
    if args.mode == "engineering":
        result = engineering_gate(backend, review, lock, deltas, hashes)
        print(json.dumps({"classification": result["classification"]}, indent=2))
    else:
        engineering = json.loads((review / "ENGINEERING_CHECKS.json").read_text(encoding="utf-8"))
        if engineering.get("classification") != "GATE8_ENGINEERING_PASS":
            raise RuntimeError("Gate 8 collection requires engineering PASS")
        collect(backend, review, lock, deltas, hashes, args.experiment_source_commit)
        print(json.dumps({"collection": "complete", "rows": 2200}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
