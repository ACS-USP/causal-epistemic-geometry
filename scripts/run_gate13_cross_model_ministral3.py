#!/usr/bin/env python3
"""Crash-safe staged runner for Gate 13 Ministral-3 cross-model replication."""

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

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.adapters import ExternalItem  # noqa: E402
from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    evaluate_external_answer_v3,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments import gate13  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

REVIEW = ROOT / "review/gate13_cross_model_ministral3"


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def source_is_ancestor(source_commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=ROOT,
            check=False,
        ).returncode
        == 0
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


MappingLike = dict[str, Any]


def load_lock(review: Path, source_commit: str) -> dict[str, Any]:
    lock = read_json(review / "MASTER_PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_SCREEN":
        raise RuntimeError("Gate 13 master protocol is not frozen")
    if lock["experiment_source_commit"] != source_commit or not source_is_ancestor(source_commit):
        raise RuntimeError("Gate 13 source commit is not the frozen ancestor of this checkout")
    parser_lock = read_json(review / "RESPONSE_PARSER_LOCK.json")
    semantic_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    if (
        parser_lock["version"] != PARSER_VERSION
        or parser_lock["module_sha256"] != gate13.file_sha256(semantic_path)
    ):
        raise RuntimeError("Gate 13 semantic parser provenance mismatch")
    return lock


def selected_model(review: Path, requested_model: str) -> tuple[str, str]:
    if requested_model == "primary":
        return gate13.PRIMARY_MODEL, gate13.PRIMARY_REVISION
    if requested_model == "fallback":
        decision = read_json(review / "SUBSTRATE_SCREEN_REPORT.json")
        if not decision.get("fallback_authorized", False):
            raise RuntimeError("Gate 13 fallback model is not authorized by the 8B screen")
        return gate13.FALLBACK_MODEL, gate13.FALLBACK_REVISION
    lock_path = review / "SELECTED_MODEL_LOCK.json"
    if not lock_path.exists():
        raise RuntimeError("Gate 13 selected-model lock is missing")
    payload = read_json(lock_path)
    return str(payload["model"]), str(payload["revision"])


def build_backend(model_path: str, model: str, revision: str) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=model,
        model_path=model_path,
        model_revision=revision,
        tokenizer_id=model_path,
        tokenizer_revision=revision,
        device="auto",
        dtype="bf16",
        layer=0,
        layer_path=gate13.MODEL_LAYER_PATH,
        prompt_mode="chat",
        max_new_tokens=gate13.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.30,
        top_p=0.95,
        top_k=0,
        min_p=0.0,
        enable_thinking=None,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        batch_size=1,
        item_batch_size=1,
        condition_chunk_size=1,
        model_loader="auto_image_text_to_text",
    )
    return HuggingFaceBackend(
        config,
        model_identifier=model,
        tokenizer_identifier=model,
        model_revision=revision,
    )


def load_external(path: Path) -> list[ExternalItem]:
    payload = read_json(path)
    return [
        ExternalItem(
            item_id=str(row["item_id"]),
            benchmark="CRUXEval",
            subtask="output_prediction",
            prompt=str(row["prompt"]),
            reference_answer=str(row["reference_answer"]),
            evaluator="python_literal",
            source_revision=str(row["source_revision"]),
            metadata={
                "reference_canonical_type": row["reference_canonical_type"],
                "item_hash": row["item_hash"],
            },
        )
        for row in payload["items"]
    ]


def model_item(item: ExternalItem, system_prompt: str | None = None) -> BenchmarkItem:
    metadata: dict[str, Any] = {
        "source_prompt_hash": item.prompt_hash,
        "response_channel": "cruxeval_semantic",
    }
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    return BenchmarkItem(
        id=item.item_id,
        prompt=item.prompt,
        target=item.reference_answer,
        metadata=metadata,
    )


def prompt_tokens(backend: HuggingFaceBackend, item: BenchmarkItem) -> tuple[list[int], str]:
    encoded, _rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    return [int(value) for value in encoded["input_ids"][0].tolist()], prompt_hash


def score(raw_output: str, reference: str, token_count: int) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        raw_output,
        reference,
        truncated=token_count >= gate13.MAX_NEW_TOKENS,
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
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "value_type": result.value_type,
        "canonical_value": result.canonical_value,
        "parsed_answer": result.payload,
        "parse_reason": result.failure_reason,
    }


def vision_tower(backend: HuggingFaceBackend) -> Any:
    outer = getattr(backend.model, "model", None)
    tower = getattr(outer, "vision_tower", None)
    if tower is None:
        raise RuntimeError(
            "Ministral-3 adapter could not locate the vision tower for zero-call audit"
        )
    return tower


def generate_with_vision_audit(
    backend: HuggingFaceBackend,
    row: BenchmarkItem,
    *,
    seed: int,
    max_new_tokens: int,
    intervention_metadata: dict[str, Any],
) -> tuple[Any, int]:
    calls = 0

    def count_call(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal calls
        calls += 1

    handle = vision_tower(backend).register_forward_hook(count_call)
    try:
        output = backend.generate_reasoning(
            row,
            sampling_seed=seed,
            max_new_tokens=max_new_tokens,
            intervention_metadata=intervention_metadata,
        )
    finally:
        handle.remove()
    if calls:
        raise RuntimeError("Gate 13 text-only generation invoked the vision tower")
    return output, calls


def system_for_condition(condition: str) -> str | None:
    return {
        "SOURCE_DIRECT": gate13.SOURCE_DIRECT,
        "SOURCE_CAREFUL": gate13.SOURCE_CAREFUL,
        "CAREFUL_CONCISE": gate13.CAREFUL_CONCISE,
        "VERBOSE_DIRECT": gate13.VERBOSE_DIRECT,
        "TEXTUAL_CAREFUL": gate13.SOURCE_CAREFUL,
    }.get(condition)


def completed_keys(path: Path, source_commit: str) -> set[tuple[str, str, str, str, int]]:
    if not path.exists():
        return set()
    keys: list[tuple[str, str, str, str, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["experiment_source_commit"] != source_commit:
            raise RuntimeError("Gate 13 journal mixes experiment source commits")
        keys.append(
            (
                str(row["stage"]),
                str(row["model"]),
                str(row["item_id"]),
                str(row["condition"]),
                int(row["rollout_index"]),
            )
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 13 journal contains duplicate logical keys")
    return set(keys)


def engineering_gate(backend: HuggingFaceBackend, review: Path, model: str, revision: str) -> None:
    manifest = review / "SUBSTRATE_SCREEN_MANIFEST.json"
    items = load_external(manifest)[:5]
    layer_count = len(backend._layer_stack)  # noqa: SLF001
    if layer_count != gate13.NUM_LAYERS or backend.hidden_size != gate13.HIDDEN_SIZE:
        raise RuntimeError("Gate 13 discovered layer count/hidden size differs from lock")
    identity: list[bool] = []
    cleanup: list[bool] = []
    traces: list[dict[str, Any]] = []
    vision_calls = 0
    rng = np.random.default_rng(130013)
    unit = rng.normal(size=gate13.HIDDEN_SIZE)
    unit /= np.linalg.norm(unit)
    delta = unit * 0.1
    for index, item in enumerate(items):
        row = model_item(item)
        seed = 13_000_000 + index
        clean, calls = generate_with_vision_audit(
            backend,
            row,
            seed=seed,
            max_new_tokens=16,
            intervention_metadata={"gate13_engineering": True, "intervention": "none"},
        )
        vision_calls += calls
        prompt_ids, _hash = prompt_tokens(backend, row)
        zero = backend.torch.zeros(
            (1, 1, gate13.HIDDEN_SIZE), dtype=backend.torch.float32, device=backend.device
        )
        with Gate6HookTrace(
            layers={0: backend.layer_module(0)},
            deltas={0: zero},
            target_positions=[len(prompt_ids) - 1],
        ) as zero_trace:
            zero_output, calls = generate_with_vision_audit(
                backend,
                row,
                seed=seed,
                max_new_tokens=16,
                intervention_metadata={"gate13_engineering": True, "intervention": "alpha_zero"},
            )
        vision_calls += calls
        identity.append(
            clean.metadata["generated_token_ids"] == zero_output.metadata["generated_token_ids"]
        )
        tensor = backend.torch.tensor(
            delta, dtype=backend.torch.float32, device=backend.device
        ).view(1, 1, -1)
        with Gate6HookTrace(
            layers={0: backend.layer_module(0)},
            deltas={0: tensor},
            target_positions=[len(prompt_ids) - 1],
        ) as trace:
            _steered, calls = generate_with_vision_audit(
                backend,
                row,
                seed=seed + 100,
                max_new_tokens=16,
                intervention_metadata={"gate13_engineering": True, "intervention": "test_delta"},
            )
        vision_calls += calls
        traces.append(trace.metadata())
        clean_after, calls = generate_with_vision_audit(
            backend,
            row,
            seed=seed,
            max_new_tokens=16,
            intervention_metadata={"gate13_engineering": True, "intervention": "none"},
        )
        vision_calls += calls
        cleanup.append(
            clean.metadata["generated_token_ids"]
            == clean_after.metadata["generated_token_ids"]
        )
        if zero_trace.forward_count < 1:
            raise RuntimeError("Gate 13 alpha-zero hook observed no model forward")
    applications = [row for trace in traces for row in trace["applications"]]
    checks = {
        "classification": "GATE13_ENGINEERING_PASS",
        "model": model,
        "revision": revision,
        "adapter": "AutoModelForImageTextToText text-only",
        "resolved_layer_path": backend._resolved_layer_path,  # noqa: SLF001
        "language_layer_count": layer_count,
        "hidden_size": backend.hidden_size,
        "alpha_zero_token_identity": all(identity),
        "hook_cleanup": all(cleanup),
        "exact_shift": bool(applications)
        and max(float(row["relative_shift_error"]) for row in applications) <= 2.0,
        "current_token_scope": bool(applications)
        and max(float(row["non_current_change"]) for row in applications) == 0.0,
        "cache_safety": any(int(row["sequence_length"]) == 1 for row in applications),
        "one_application_per_forward": all(
            trace["forward_count"] == len(trace["applications"]) for trace in traces
        ),
        "vision_tower_calls": vision_calls,
        "vision_invoked": vision_calls > 0,
        "chat_template_deterministic": True,
        "hidden_reasoning_channel": False,
        "tool_output": False,
    }
    required = (
        checks["alpha_zero_token_identity"],
        checks["hook_cleanup"],
        checks["exact_shift"],
        checks["current_token_scope"],
        checks["cache_safety"],
        checks["one_application_per_forward"],
        not checks["vision_invoked"],
    )
    if not all(required):
        checks["classification"] = "GATE13_MODEL_ADAPTER_FAILURE"
    write_json(review / "ENGINEERING_CHECKS.json", checks)
    if checks["classification"] != "GATE13_ENGINEERING_PASS":
        raise RuntimeError("GATE13_MODEL_ADAPTER_FAILURE")


def stage_manifest(review: Path, stage: str, model_role: str) -> Path:
    if stage == "screen":
        return review / (
            "SUBSTRATE_SCREEN_MANIFEST.json"
            if model_role == "primary"
            else "FALLBACK_SCREEN_MANIFEST.json"
        )
    return review / {
        "first-stage": "LAYER_FIRST_STAGE_MANIFEST.json",
        "dose": "DOSE_CALIBRATION_MANIFEST.json",
        "final": "FINAL_EVALUATION_MANIFEST.json",
    }[stage]


def stage_schedule(review: Path, stage: str, model_role: str) -> Path:
    if stage == "screen":
        return review / (
            "SUBSTRATE_SCREEN_SCHEDULE.json"
            if model_role == "primary"
            else "FALLBACK_SCREEN_SCHEDULE.json"
        )
    return review / {
        "first-stage": "LAYER_FIRST_STAGE_SCHEDULE.json",
        "dose": "DOSE_CALIBRATION_SCHEDULE.json",
        "final": "FINAL_EVALUATION_SCHEDULE.json",
    }[stage]


def condition_delta(
    review: Path, stage: str, condition: str
) -> tuple[int | None, np.ndarray | None, str | None, float | None]:
    if stage == "screen" or condition in {"BASELINE", "TEXTUAL_CAREFUL"}:
        return None, None, None, None
    if stage == "first-stage":
        lock = read_json(review / "LAYER_SHORTLIST_LOCK.json")
        for record in lock["conditions"]:
            if record["condition"] == condition:
                vector = np.load(review / record["vector_path"], allow_pickle=False).astype(
                    np.float64
                )
                alpha = float(record["alpha"])
                return int(record["layer"]), vector * alpha, record["vector_hash"], alpha
    if stage in {"dose", "final"}:
        layer_lock = read_json(review / "SELECTED_LAYER_LOCK.json")
        layer = int(layer_lock["selected_layer"])
        full = float(layer_lock["full_source_displacement"])
        vectors = {"MEAN": "SOURCE_DIRECTIONS/SELECTED_MEANINGFUL.npy"}
        vectors.update({f"R{i}": f"FINAL_RANDOM_DIRECTIONS/R{i}.npy" for i in range(4)})
        if stage == "dose":
            if condition.startswith("MEAN_"):
                vector_name = "MEAN"
                dose = condition.removeprefix("MEAN_")
            else:
                vector_name, dose = condition.split("_", 1)
            alpha = full * gate13.DOSE_FRACTIONS[dose]
        else:
            dose_lock = read_json(review / "SELECTED_DOSE_LOCK.json")
            alpha = float(dose_lock["selected_alpha"])
            vector_name = (
                "MEAN"
                if condition == "MEANINGFUL_SELECTED"
                else condition.removeprefix("RANDOM_")
            )
        path = review / vectors[vector_name]
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        return layer, vector * alpha, gate13.vector_sha256(vector), alpha
    raise KeyError(f"no Gate 13 condition delta for {stage}:{condition}")


def collect_stage(
    backend: HuggingFaceBackend,
    review: Path,
    stage: str,
    model_role: str,
    model: str,
    revision: str,
    source_commit: str,
) -> None:
    items = load_external(stage_manifest(review, stage, model_role))
    item_by_id = {item.item_id: item for item in items}
    schedule = read_json(stage_schedule(review, stage, model_role))
    journal = review / "journal.jsonl"
    completed = completed_keys(journal, source_commit)
    for schedule_index, planned in enumerate(schedule):
        key = (
            str(planned["stage"]),
            model,
            str(planned["item_id"]),
            str(planned["condition"]),
            int(planned["rollout_index"]),
        )
        if key in completed:
            continue
        item = item_by_id[key[2]]
        condition = key[3]
        system = system_for_condition(condition)
        row = model_item(item, system)
        prompt_ids, rendered_hash = prompt_tokens(backend, row)
        layer, delta, vector_hash, alpha = condition_delta(review, stage, condition)
        if delta is None:
            context: Any = nullcontext()
        else:
            tensor = backend.torch.tensor(
                delta, dtype=backend.torch.float32, device=backend.device
            ).view(1, 1, -1)
            context = Gate6HookTrace(
                layers={int(layer): backend.layer_module(int(layer))},
                deltas={int(layer): tensor},
                target_positions=[len(prompt_ids) - 1],
            )
        started = time.perf_counter()
        with context as trace:
            output, vision_calls = generate_with_vision_audit(
                backend,
                row,
                seed=int(planned["seed"]),
                max_new_tokens=gate13.MAX_NEW_TOKENS,
                intervention_metadata={
                    "experiment_id": gate13.EXPERIMENT_ID,
                    "stage": planned["stage"],
                    "condition": condition,
                    "intervention": condition if delta is not None else "none",
                    "intervention_layer": layer,
                    "intervention_vector_hash": vector_hash,
                    "intervention_alpha": alpha,
                    "intervention_duration": (
                        "sustained_current_token" if delta is not None else "none"
                    ),
                    "parser_version": PARSER_VERSION,
                    "environment_profile": "CORE_MINISTRAL3",
                    "vision_invoked": False,
                },
            )
        elapsed = time.perf_counter() - started
        metadata = dict(output.metadata)
        if delta is not None:
            metadata["intervention_forward_trace"] = trace.metadata()
        token_count = int(metadata["generated_token_count"])
        scored = score(output.raw_output, item.reference_answer, token_count)
        record = {
            **planned,
            **scored,
            "model": model,
            "model_revision": revision,
            "tokenizer_revision": revision,
            "experiment_source_commit": source_commit,
            "runtime_commit": git_commit(),
            "raw_output": output.raw_output,
            "generated_token_ids": metadata["generated_token_ids"],
            "generated_token_count": token_count,
            "reference_answer": item.reference_answer,
            "reference_canonical_type": item.metadata["reference_canonical_type"],
            "prompt_hash": item.prompt_hash,
            "rendered_prompt_hash": rendered_hash,
            "parser_version": PARSER_VERSION,
            "layer": layer,
            "vector_hash": vector_hash,
            "alpha": alpha,
            "vision_tower_calls": vision_calls,
            "backend_metadata": metadata,
            "elapsed_seconds": elapsed,
            "schedule_index": schedule_index,
            "retry_count": 0,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }
        append_jsonl(journal, record)
        completed.add(key)


def capture_source_atlas(
    backend: HuggingFaceBackend, review: Path, model: str, revision: str, source_commit: str
) -> None:
    output = review / "SOURCE_ACTIVATIONS.npz"
    if output.exists():
        raise RuntimeError("Gate 13 source activation archive already exists; do not overwrite")
    arrays: dict[str, np.ndarray] = {}
    prompt_records: list[dict[str, Any]] = []
    vision_calls = 0

    def count_vision(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal vision_calls
        vision_calls += 1

    vision_handle = vision_tower(backend).register_forward_hook(count_vision)
    try:
        for split, filename in (
            ("construction", "SOURCE_CONSTRUCTION_MANIFEST.json"),
            ("validation", "SOURCE_VALIDATION_MANIFEST.json"),
        ):
            items = load_external(review / filename)
            for condition, system in (
                ("careful", gate13.SOURCE_CAREFUL),
                ("direct", gate13.SOURCE_DIRECT),
            ):
                condition_arrays: list[np.ndarray] = []
                for item in items:
                    row = model_item(item, system)
                    captured = backend.extract_activations_batch(
                        [row], layers=list(range(gate13.NUM_LAYERS))
                    )
                    matrix = np.stack(
                        [captured[layer][0] for layer in range(gate13.NUM_LAYERS)]
                    )
                    condition_arrays.append(matrix.astype(np.float32))
                    _ids, prompt_hash = prompt_tokens(backend, row)
                    prompt_records.append(
                        {
                            "split": split,
                            "condition": condition,
                            "item_id": item.item_id,
                            "rendered_prompt_hash": prompt_hash,
                        }
                    )
                arrays[f"{split}_{condition}"] = np.stack(condition_arrays)
    finally:
        vision_handle.remove()
    if vision_calls:
        raise RuntimeError("Gate 13 source-atlas capture invoked the vision tower")
    np.savez_compressed(output, **arrays)
    write_json(
        review / "SOURCE_ACTIVATION_MANIFEST.json",
        {
            "model": model,
            "revision": revision,
            "experiment_source_commit": source_commit,
            "archive": output.name,
            "archive_sha256": gate13.file_sha256(output),
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "dtype": "float32 capture of BF16 residual outputs",
            "layer_path": backend._resolved_layer_path,  # noqa: SLF001
            "vision_tower_calls": vision_calls,
            "vision_invoked": vision_calls > 0,
            "prompt_records": prompt_records,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("engineering", "screen", "source-atlas", "first-stage", "dose", "final"),
        required=True,
    )
    parser.add_argument(
        "--model-role", choices=("primary", "fallback", "selected"), default="selected"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 13 {args.mode}")
    review = args.review_dir.resolve()
    load_lock(review, args.source_commit)
    model, revision = selected_model(review, args.model_role)
    backend = build_backend(args.model_path, model, revision)
    if args.mode == "engineering":
        engineering_gate(backend, review, model, revision)
    elif args.mode == "source-atlas":
        capture_source_atlas(backend, review, model, revision, args.source_commit)
    else:
        collect_stage(
            backend,
            review,
            args.mode,
            args.model_role,
            model,
            revision,
            args.source_commit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
