#!/usr/bin/env python3
"""Remote-only Gate-5 collection runner.

The runner journals raw outputs immediately and never computes or displays
phase outcomes while a phase is being collected.  It supports resume by the
full logical key and keeps source, matched manipulation, and independent
primary schedules separate.
"""

from __future__ import annotations

import argparse
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
    score_external_response,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.experiments.gate5 import (  # noqa: E402
    ALPHA,
    CONDITIONS,
    DATASET_REVISION,
    LAYER,
    MODEL,
    MODEL_REVISION,
    SYSTEM_CAREFUL,
    SYSTEM_DIRECT,
    evaluation_seed,
    manipulation_seed,
    source_seed,
)
from epistemic_geometry.experiments.micro_q1 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector  # noqa: E402

MAX_NEW_TOKENS = 4096
PARSER_VERSION = "external-semantic-v1"
SOURCE_CONDITIONS = ("ORDINARY", "CAREFUL", "DIRECT")
SOURCE_PHASE = "SOURCE_CHECK"
MANIP_PHASE = "SUSTAINED_MANIPULATION"
EVAL_PHASE = "SUSTAINED_EVALUATION"


def _write_json(path: Path, payload: Any) -> None:
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
    return [
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
        for row in payload["items"]
    ]


def _benchmark_item(item: ExternalItem, system_prompt: str | None = None) -> BenchmarkItem:
    metadata: dict[str, Any] = {
        "source_prompt_hash": item.prompt_hash,
        "response_channel": "cruxeval_semantic",
    }
    if system_prompt is not None:
        metadata["system_prompt"] = system_prompt
    return BenchmarkItem(
        id=item.item_id, prompt=item.prompt, target=item.reference_answer, metadata=metadata
    )


def _build_backend(model_path: str) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL,
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
        model_identifier=MODEL,
        tokenizer_identifier=MODEL,
        model_revision=MODEL_REVISION,
    )


def _load_vectors(output_dir: Path, gate4_dir: Path) -> dict[str, SteeringVector]:
    paths = {"v_delib": gate4_dir / "DIRECTION.npy"}
    paths.update({f"R{i}": output_dir / f"R{i}.npy" for i in range(4)})
    vectors: dict[str, SteeringVector] = {}
    for name, path in paths.items():
        values = np.load(path, allow_pickle=False).astype(np.float64)
        vectors[name] = SteeringVector(
            values=values,
            layer=LAYER,
            constructor="gate4_exact" if name == "v_delib" else "gate5_orthogonal_random_bank",
            normalization="unit",
            metadata={"path": str(path), "gate5_controller": name},
            hash=vector_sha256(values),
        )
    return vectors


def _intervention(vector: SteeringVector, alpha: float, vector_id: str) -> Intervention:
    return Intervention(
        layer=LAYER,
        alpha=float(alpha),
        vector_id=vector_id,
        token_scope="last_token",
        vector=vector,
    )


def _condition_intervention(
    condition: str, vectors: dict[str, SteeringVector]
) -> tuple[str, Intervention | None, str]:
    if condition == "BASELINE":
        return "none", None, "none"
    if condition == "ONE_SHOT_PLUS":
        return "one_shot", _intervention(vectors["v_delib"], ALPHA, "v_delib_plus"), "v_delib"
    if condition == "ONE_SHOT_MINUS":
        return "one_shot", _intervention(vectors["v_delib"], -ALPHA, "v_delib_minus"), "v_delib"
    if condition == "SUSTAINED_PLUS":
        return (
            "sustained",
            _intervention(vectors["v_delib"], ALPHA, "v_delib_plus_sustained"),
            "v_delib",
        )
    if condition == "SUSTAINED_MINUS":
        return (
            "sustained",
            _intervention(vectors["v_delib"], -ALPHA, "v_delib_minus_sustained"),
            "v_delib",
        )
    if condition.startswith("SUSTAINED_RANDOM_R"):
        name = condition.removeprefix("SUSTAINED_RANDOM_")
        return "sustained", _intervention(vectors[name], ALPHA, f"{name.lower()}_sustained"), name
    raise ValueError(f"unknown Gate-5 condition: {condition}")


def _metadata(
    condition: str, duration: str, intervention: Intervention | None, vector_name: str
) -> dict[str, Any]:
    if intervention is None:
        return {
            "intervention": "none",
            "intervention_duration": "none",
            "intervention_layer": None,
            "intervention_alpha": 0.0,
            "intervention_vector_hash": None,
            "intervention_vector_name": "none",
        }
    return {
        "intervention": intervention.vector_id,
        "intervention_duration": duration,
        "intervention_layer": intervention.layer,
        "intervention_alpha": intervention.alpha,
        "intervention_vector_hash": intervention.vector.hash,
        "intervention_vector_name": vector_name,
    }


def _scheduled_rows(phase: str, items: list[ExternalItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if phase == SOURCE_PHASE:
        for item in items:
            for condition in SOURCE_CONDITIONS:
                for rollout in (0, 1):
                    rows.append(
                        {
                            "phase": phase,
                            "item_id": item.item_id,
                            "condition": condition,
                            "rollout_index": rollout,
                            "seed": source_seed(item.item_id, condition, rollout),
                            "seed_regime": "INDEPENDENT_PRIMARY",
                        }
                    )
    elif phase == MANIP_PHASE:
        for item in items:
            for condition in CONDITIONS:
                rows.append(
                    {
                        "phase": phase,
                        "item_id": item.item_id,
                        "condition": condition,
                        "rollout_index": 0,
                        "seed": manipulation_seed(item.item_id),
                        "seed_regime": "MATCHED_COUPLING_SECONDARY",
                    }
                )
    elif phase == EVAL_PHASE:
        for item in items:
            for condition in CONDITIONS:
                for rollout in (0, 1):
                    rows.append(
                        {
                            "phase": phase,
                            "item_id": item.item_id,
                            "condition": condition,
                            "rollout_index": rollout,
                            "seed": evaluation_seed(item.item_id, condition, rollout),
                            "seed_regime": "INDEPENDENT_PRIMARY",
                        }
                    )
    else:
        raise ValueError(f"unknown phase: {phase}")
    return rows


def _read_completed(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                completed.add(
                    (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
                )
    return completed


def _execute_row(
    backend: HuggingFaceBackend,
    item: ExternalItem,
    row: dict[str, Any],
    vectors: dict[str, SteeringVector],
) -> dict[str, Any]:
    phase = row["phase"]
    condition = row["condition"]
    system_prompt = {
        "ORDINARY": None,
        "CAREFUL": SYSTEM_CAREFUL,
        "DIRECT": SYSTEM_DIRECT,
    }.get(condition)
    duration, intervention, vector_name = (
        _condition_intervention(condition, vectors)
        if phase != SOURCE_PHASE
        else ("none", None, "none")
    )
    if phase == SOURCE_PHASE:
        item_for_model = _benchmark_item(item, system_prompt=system_prompt)
    else:
        item_for_model = _benchmark_item(item)
    started = time.perf_counter()
    trace: dict[str, Any] | None = None
    try:
        if duration == "one_shot":
            context = backend.steer_prefill_once(intervention)
        elif duration == "sustained":
            context = backend.steer_sustained_current_token(intervention)
        else:
            context = nullcontext()
        with context as active_trace:
            trace = active_trace
            output = backend.generate_reasoning(
                item_for_model,
                sampling_seed=int(row["seed"]),
                max_new_tokens=MAX_NEW_TOKENS,
                intervention_metadata=_metadata(condition, duration, intervention, vector_name),
            )
    except RuntimeError as exc:
        return {
            **row,
            "status": "RUNTIME_ERROR",
            "correct": False,
            "parsed_answer": None,
            "raw_output": "",
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started,
        }
    elapsed = time.perf_counter() - started
    output_metadata = dict(output.metadata)
    output_metadata.update(_metadata(condition, duration, intervention, vector_name))
    if trace is not None:
        output_metadata["intervention_forward_trace"] = trace
        output_metadata["intervention_forward_count"] = trace["forward_count"]
        output_metadata["intervention_prefill_applications"] = trace["prefill_applications"]
        output_metadata["intervention_decode_applications"] = trace["decode_applications"]
    token_count = int(output_metadata.get("generated_token_count", 0))
    scored = score_external_response(
        item,
        output.raw_output,
        rollout_seed=int(row["seed"]),
        truncated=token_count >= MAX_NEW_TOKENS,
        token_count=token_count,
        metadata={
            "phase": phase,
            "condition": condition,
            "rollout_index": row["rollout_index"],
            "seed_regime": row["seed_regime"],
            "generation_seconds": elapsed,
            "stop_metadata": output_metadata,
        },
    )
    return {
        **row,
        "status": scored.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
        "correct": bool(scored.correct),
        "parsed_answer": scored.parsed_answer,
        "reference_answer": item.reference_answer,
        "raw_output": scored.raw_output,
        "generated_token_ids": output_metadata.get("generated_token_ids", []),
        "generated_token_count": token_count,
        "prompt_hash": item.prompt_hash,
        "rendered_prompt_hash": output_metadata.get("rendered_prompt_hash"),
        "source_revision": DATASET_REVISION,
        "evaluator": item.evaluator,
        "metadata": scored.metadata,
        "elapsed_seconds": elapsed,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def run_phase(args: argparse.Namespace) -> int:
    require_remote_hf_execution(f"Gate 5 {args.phase} inference")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == SOURCE_PHASE:
        items = _load_items(args.source_manifest)
    elif args.phase == MANIP_PHASE:
        items = _load_items(args.manipulation_manifest)
    else:
        items = _load_items(args.evaluation_manifest)
    schedule = _scheduled_rows(args.phase, items)
    schedule_path = output_dir / f"{args.phase}_SCHEDULE.json"
    if schedule_path.exists():
        existing = json.loads(schedule_path.read_text(encoding="utf-8"))
        if existing != schedule:
            raise RuntimeError("frozen Gate-5 schedule differs from existing schedule")
    else:
        _write_json(schedule_path, schedule)
    journal = output_dir / "journal.jsonl"
    completed = _read_completed(journal)
    vectors = _load_vectors(output_dir, args.gate4_dir)
    backend = _build_backend(args.model_path)
    by_id = {item.item_id: item for item in items}
    for row in schedule:
        key = (row["item_id"], row["condition"], row["rollout_index"])
        if key in completed:
            continue
        record = _execute_row(backend, by_id[row["item_id"]], row, vectors)
        _append_jsonl(journal, record)
        if record["status"] == "RUNTIME_ERROR":
            raise RuntimeError(f"model/runtime failure for {key}: {record.get('error')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=(SOURCE_PHASE, MANIP_PHASE, EVAL_PHASE), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "review" / "gate5_source_duration"
    )
    parser.add_argument("--gate4-dir", type=Path, default=ROOT / "review" / "micro_q1")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "review" / "gate5_source_duration" / "SOURCE_CHECK.json",
    )
    parser.add_argument(
        "--manipulation-manifest",
        type=Path,
        default=ROOT / "review" / "gate5_source_duration" / "SUSTAINED_MANIPULATION.json",
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=ROOT / "review" / "gate5_source_duration" / "SUSTAINED_EVALUATION.json",
    )
    return run_phase(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
