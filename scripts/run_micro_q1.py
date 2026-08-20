#!/usr/bin/env python3
"""Execute the frozen Gate-4 direction gate and CRUXEval trajectories.

This script is intentionally remote-only for real Qwen execution.  It stops
before evaluation if direction or hook checks fail and journals every completed
trajectory with an fsync.  It does not compute or inspect scientific summary
metrics while collecting rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalItem,
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.micro_q1 import (  # noqa: E402
    construct_paired_direction,
    heldout_signed_gaps,
    random_orthogonal_direction,
    vector_sha256,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    require_remote_hf_execution,
    stable_seed,
)
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER = 17
MAX_NEW_TOKENS = 4096
ALPHA_SEED = 20260819
RANDOM_DIRECTION_SEED = 20260819
PARSER_VERSION = "external-semantic-v1"
SYSTEM_CAREFUL = (
    "You are a meticulous program tracer. Carefully track every operation, mutation, "
    "intermediate value, branch, and loop. Verify the result before answering. End with "
    "exactly one line in the form FINAL: <answer>."
)
SYSTEM_DIRECT = (
    "Answer the program-output question immediately. Do not trace, deliberate, explain, "
    "or verify. End with exactly one line in the form FINAL: <answer>."
)
CONDITIONS = ("BASELINE", "CPLUS", "CMINUS", "CRANDOM")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_items(path: Path) -> list[ExternalItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"]
    items = [
        ExternalItem(
            item_id=str(row["item_id"]),
            benchmark="CRUXEval",
            subtask="output_prediction",
            prompt=str(row["prompt"]),
            reference_answer=str(row["reference_answer"]),
            evaluator="python_literal",
            source_revision=str(row["source_revision"]),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]
    if len({item.item_id for item in items}) != len(items):
        raise ValueError(f"duplicate IDs in {path}")
    return items


def _bench_item(item: ExternalItem, *, system_prompt: str | None = None) -> BenchmarkItem:
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


def _build_backend(model_path: str) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_path=model_path,
        model_revision=MODEL_REVISION,
        tokenizer_id=model_path,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        layer=LAYER,
        layer_path="model.model.layers",
        prompt_mode="chat",
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=False,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        batch_size=1,
        item_batch_size=1,
        condition_chunk_size=1,
    )
    return HuggingFaceBackend(
        config,
        model_identifier=MODEL_ID,
        tokenizer_identifier=MODEL_ID,
        model_revision=MODEL_REVISION,
    )


def _capture_forward(
    backend: HuggingFaceBackend,
    item: BenchmarkItem,
    intervention: Intervention | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Capture the full layer output and the rendered prompt hash once."""

    encoded, _rendered, prompt_hash = backend._encode_item(item)  # noqa: SLF001
    captured: list[np.ndarray] = []
    layer = backend.layer_module(LAYER)

    def capture(_module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(hidden, backend.torch.Tensor):
            raise TypeError("layer output did not contain a tensor")
        captured.append(hidden.detach().float().cpu().numpy().copy())
        return output

    context = (
        backend.steer_prefill_once(intervention) if intervention is not None else nullcontext()
    )
    with context:
        handle = layer.register_forward_hook(capture)
        try:
            with backend.torch.inference_mode():
                backend.model(**encoded, use_cache=False)
        finally:
            handle.remove()
    if not captured:
        raise RuntimeError("activation hook did not capture layer output")
    mask = encoded["attention_mask"][0].detach().cpu().numpy().astype(bool)
    return captured[0], mask, prompt_hash


def _activation(
    backend: HuggingFaceBackend, item: ExternalItem, system_prompt: str
) -> tuple[np.ndarray, str]:
    hidden, mask, prompt_hash = _capture_forward(
        backend, _bench_item(item, system_prompt=system_prompt)
    )
    return hidden[0, int(mask.sum()) - 1, :], prompt_hash


def _make_vector(values: np.ndarray, constructor: str, metadata: dict[str, Any]) -> SteeringVector:
    return SteeringVector(
        values=np.asarray(values, dtype=np.float64),
        layer=LAYER,
        constructor=constructor,
        normalization="unit",
        metadata=metadata,
        hash=vector_sha256(values),
    )


def _intervention(vector: SteeringVector, alpha: float, vector_id: str) -> Intervention:
    return Intervention(
        layer=LAYER,
        alpha=float(alpha),
        vector_id=vector_id,
        token_scope="last_token",
        vector=vector,
        token_index=None,
    )


def _generate(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    seed: int,
    intervention: Intervention | None,
) -> Any:
    context = (
        backend.steer_prefill_once(intervention) if intervention is not None else nullcontext()
    )
    with context:
        return backend.generate_reasoning(
            _bench_item(item), sampling_seed=seed, max_new_tokens=MAX_NEW_TOKENS
        )


def _direction_gate(
    backend: HuggingFaceBackend,
    construction: list[ExternalItem],
    validation: list[ExternalItem],
    output_dir: Path,
) -> tuple[SteeringVector, SteeringVector, float, dict[str, Any]]:
    careful_rows = []
    direct_rows = []
    construction_prompt_hashes: dict[str, dict[str, str]] = {}
    for item in construction:
        careful, careful_hash = _activation(backend, item, SYSTEM_CAREFUL)
        direct, direct_hash = _activation(backend, item, SYSTEM_DIRECT)
        careful_rows.append(careful)
        direct_rows.append(direct)
        construction_prompt_hashes[item.item_id] = {"careful": careful_hash, "direct": direct_hash}
    careful_array = np.stack(careful_rows)
    direct_array = np.stack(direct_rows)
    direction, delta, raw = construct_paired_direction(careful_array, direct_array)
    vector_meta = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "layer": LAYER,
        "layer_path": "model.model.layers",
        "activation": "block output residual stream",
        "token_position": "final non-padding prompt token",
        "constructor": "paired_careful_minus_direct_mean_difference",
        "source_item_ids": [item.item_id for item in construction],
        "prompt_hashes": construction_prompt_hashes,
        "vector_norm": float(np.linalg.norm(direction)),
        "raw_mean_difference_norm": float(np.linalg.norm(raw)),
        "vector_hash": vector_sha256(direction),
    }
    _write_json(output_dir / "DIRECTION_METADATA.json", vector_meta)
    np.save(output_dir / "DIRECTION.npy", direction)

    validation_careful = []
    validation_direct = []
    validation_hashes: dict[str, dict[str, str]] = {}
    for item in validation:
        careful, careful_hash = _activation(backend, item, SYSTEM_CAREFUL)
        direct, direct_hash = _activation(backend, item, SYSTEM_DIRECT)
        validation_careful.append(careful)
        validation_direct.append(direct)
        validation_hashes[item.item_id] = {"careful": careful_hash, "direct": direct_hash}
    gaps = heldout_signed_gaps(direction, np.stack(validation_careful), np.stack(validation_direct))
    validation_rows = []
    for item, gap in zip(validation, gaps, strict=True):
        validation_rows.append(
            {"item_id": item.item_id, "signed_gap": float(gap), "positive": bool(gap > 0)}
        )
    with (output_dir / "DIRECTION_VALIDATION.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["item_id", "signed_gap", "positive"])
        writer.writeheader()
        writer.writerows(validation_rows)
    validation_pass = bool(
        np.isfinite(direction).all()
        and np.linalg.norm(direction) > 0
        and float(gaps.mean()) > 0
        and int((gaps > 0).sum()) >= 12
    )
    if not validation_pass:
        raise RuntimeError("MICRO_Q1_DIRECTION_NOT_VALIDATED")
    random_values, random_cosine = random_orthogonal_direction(direction, RANDOM_DIRECTION_SEED)
    random_vector = _make_vector(
        random_values,
        "random_orthogonal",
        {"random_seed": RANDOM_DIRECTION_SEED, "orthogonality_cosine": random_cosine},
    )
    _write_json(
        output_dir / "RANDOM_DIRECTION_METADATA.json",
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "constructor": "gaussian_remove_parallel_component",
            "random_seed": RANDOM_DIRECTION_SEED,
            "vector_hash": random_vector.hash,
            "norm": float(np.linalg.norm(random_values)),
            "orthogonality_cosine": random_cosine,
        },
    )
    np.save(output_dir / "RANDOM_DIRECTION.npy", random_values)
    return (
        _make_vector(direction, "paired_careful_minus_direct", vector_meta),
        random_vector,
        float(delta),
        {
            "validation_rows": validation_rows,
            "validation_positive_count": int((gaps > 0).sum()),
            "validation_mean_gap": float(gaps.mean()),
            "validation_pass": validation_pass,
            "construction_delta": float(delta),
            "alpha": 0.5 * float(delta),
            "random_cosine": random_cosine,
            "validation_prompt_hashes": validation_hashes,
        },
    )


def _engineering_gate(
    backend: HuggingFaceBackend,
    items: list[ExternalItem],
    direction: SteeringVector,
    random: SteeringVector,
    alpha: float,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    item = items[0]
    seed = stable_seed("MICRO-Q1-ENGINEERING", item.item_id)
    clean = _generate(backend, item, seed, None)
    zero = _generate(backend, item, seed, _intervention(direction, 0.0, "alpha_zero"))
    checks["alpha_zero_identity"] = {
        "pass": clean.metadata.get("generated_token_ids")
        == zero.metadata.get("generated_token_ids"),
        "token_ids_equal": clean.metadata.get("generated_token_ids")
        == zero.metadata.get("generated_token_ids"),
        "raw_equal": clean.raw_output == zero.raw_output,
    }
    if not checks["alpha_zero_identity"]["pass"]:
        raise RuntimeError("MICRO_Q1_INTERVENTION_ENGINE_FAILURE: alpha-zero identity")
    probe = _bench_item(item)
    base_hidden, mask, _ = _capture_forward(backend, probe, None)
    target_position = int(mask.sum()) - 1
    for name, vector, signed_alpha in (
        ("plus_delib", direction, alpha),
        ("minus_delib", direction, -alpha),
        ("plus_random", random, alpha),
    ):
        after_hidden, _, _ = _capture_forward(
            backend, probe, _intervention(vector, signed_alpha, name)
        )
        observed = after_hidden - base_hidden
        expected = np.zeros_like(observed)
        expected[:, target_position, :] = np.asarray(signed_alpha * vector.values, dtype=np.float32)
        max_error = float(np.max(np.abs(observed - expected)))
        non_target_error = float(
            np.max(np.abs(np.delete(observed, target_position, axis=1)))
            if observed.shape[1] > 1
            else 0.0
        )
        passed = max_error <= 0.08 and non_target_error <= 0.08
        checks[name] = {
            "pass": passed,
            "max_abs_shift_error_bf16": max_error,
            "max_abs_non_target_change": non_target_error,
            "target_position": target_position,
        }
        if not passed:
            raise RuntimeError(f"MICRO_Q1_INTERVENTION_ENGINE_FAILURE: {name}")
    clean_after = _generate(backend, item, seed, None)
    checks["hook_cleanup"] = {
        "pass": clean.metadata.get("generated_token_ids")
        == clean_after.metadata.get("generated_token_ids"),
        "token_ids_equal": clean.metadata.get("generated_token_ids")
        == clean_after.metadata.get("generated_token_ids"),
    }
    if not checks["hook_cleanup"]["pass"]:
        raise RuntimeError("MICRO_Q1_INTERVENTION_ENGINE_FAILURE: hook cleanup")
    checks["last_prompt_token_scope"] = all(
        checks[name]["max_abs_non_target_change"] <= 0.08
        for name in ("plus_delib", "minus_delib", "plus_random")
    )
    if not checks["last_prompt_token_scope"]:
        raise RuntimeError("MICRO_Q1_INTERVENTION_ENGINE_FAILURE: token scope")
    return checks


def _journal_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return str(record["item_id"]), str(record["condition"]), int(record["rollout_index"])


def _read_completed(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "trajectory":
            continue
        key = _journal_key(record)
        if key in rows and rows[key] != record:
            raise RuntimeError(f"duplicate/conflicting trajectory key: {key}")
        rows[key] = record
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construction", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    require_remote_hf_execution("Gate 4 Qwen inference")
    args.output.mkdir(parents=True, exist_ok=True)
    construction = _load_items(args.construction)
    validation = _load_items(args.validation)
    evaluation = _load_items(args.evaluation)
    if (len(construction), len(validation), len(evaluation)) != (64, 16, 50):
        raise ValueError("Gate 4 allocation must be exactly 64/16/50")
    if len({item.item_id for item in construction + validation + evaluation}) != 130:
        raise ValueError("Gate 4 allocation contains duplicate IDs")
    backend = _build_backend(args.model_path)
    direction, random, delta, gate = _direction_gate(backend, construction, validation, args.output)
    alpha = 0.5 * delta
    if not np.isfinite(alpha) or alpha <= 0:
        raise RuntimeError("MICRO_Q1_DIRECTION_NOT_VALIDATED: alpha is not positive")
    engineering = _engineering_gate(backend, validation[:5], direction, random, alpha)
    _write_json(args.output / "ENGINEERING_CHECKS.json", engineering)
    _write_json(
        args.output / "PRE_EVALUATION_GATE.json",
        {
            "source_commit": args.source_commit,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "layer": LAYER,
            "delta": delta,
            "alpha": alpha,
            "direction_hash": direction.hash,
            "random_hash": random.hash,
            "direction_gate": gate,
            "engineering": engineering,
            "scientific_evaluation_authorized": True,
            "created_utc": datetime.now(UTC).isoformat(),
        },
    )
    journal = args.output / "journal.jsonl"
    completed = _read_completed(journal)
    vectors = {
        "CPLUS": _intervention(direction, alpha, "v_delib_plus"),
        "CMINUS": _intervention(direction, -alpha, "v_delib_minus"),
        "CRANDOM": _intervention(random, alpha, "v_random_orthogonal"),
    }
    for item in evaluation:
        for condition in CONDITIONS:
            for rollout_index in (0, 1):
                key = (item.item_id, condition, rollout_index)
                if key in completed:
                    continue
                seed = stable_seed(
                    "MICRO-Q1", "INDEPENDENT_PRIMARY", item.item_id, condition, rollout_index
                )
                intervention = vectors.get(condition)
                started = time.perf_counter()
                try:
                    output = _generate(backend, item, seed, intervention)
                except RuntimeError as exc:
                    raise RuntimeError(f"infrastructure/model failure for {key}: {exc}") from exc
                elapsed = time.perf_counter() - started
                token_count = int(output.metadata.get("generated_token_count", 0))
                scored = score_external_response(
                    item,
                    output.raw_output,
                    rollout_seed=seed,
                    truncated=token_count >= MAX_NEW_TOKENS,
                    token_count=token_count,
                    metadata={
                        "condition": condition,
                        "rollout_index": rollout_index,
                        "seed_regime": "INDEPENDENT_PRIMARY",
                        "generation_seconds": elapsed,
                        "vector_hash": vectors[condition].vector.hash if intervention else None,
                        "alpha": intervention.alpha if intervention else 0.0,
                        "layer": LAYER,
                        "engine": "hf_generate_serial_prefill_one_shot_hook",
                        "parser_version": PARSER_VERSION,
                        "stop_metadata": output.metadata,
                    },
                )
                record = {
                    "event": "trajectory",
                    "item_id": item.item_id,
                    "condition": condition,
                    "rollout_index": rollout_index,
                    "seed": seed,
                    "status": (
                        "TRUNCATED"
                        if scored.status == ExternalStatus.TRUNCATED_THINKING
                        else scored.status.value
                    ),
                    "correct": scored.correct,
                    "parsed_answer": scored.parsed_answer,
                    "reference_answer": item.reference_answer,
                    "raw_output": scored.raw_output,
                    "generated_token_ids": output.metadata.get("generated_token_ids", []),
                    "generated_token_count": token_count,
                    "prompt_hash": item.prompt_hash,
                    "source_revision": item.source_revision,
                    "evaluator": item.evaluator,
                    "elapsed_seconds": elapsed,
                    "metadata": scored.metadata,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
                _append_jsonl(journal, record)
                completed[key] = record
    _write_json(
        args.output / "RUN_PROVENANCE.json",
        {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "source_commit": args.source_commit,
            "layer": LAYER,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
            "conditions": CONDITIONS,
            "rollouts_per_item_condition": 2,
            "scientific_rows": len(completed),
            "engine": "hf_generate_serial_prefill_one_shot_hook",
            "attention": "sdpa",
        },
    )
    print(json.dumps({"scientific_rows": len(completed), "status": "COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
