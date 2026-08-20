#!/usr/bin/env python3
"""Run only the conditional Gate 6.3 phases on the canonical remote A40.

The parser reanalysis and random-bank construction are CPU-only.  This runner
is deliberately limited to the frozen 80-row matched random supplement and,
only when called explicitly after that gate passes, the 840-row evaluation.
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
sys.path.insert(0, str(ROOT / "scripts"))

from run_gate6_2_first_stage_repair import (  # noqa: E402
    MAX_NEW_TOKENS,
    _completed,
    build_backend,
    load_external,
    model_item,
    prompt_tokens,
)

from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalStatus,
    evaluate_external_answer,
)
from epistemic_geometry.benchmarks.external.semantic_v2 import (  # noqa: E402
    PARSER_VERSION,
    parse_external_answer_v2,
)
from epistemic_geometry.experiments.gate6 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    git_metadata,
    require_remote_hf_execution,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace  # noqa: E402

RANDOM_CONDITIONS = tuple(f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
EVALUATION_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "BEST_SINGLE_MEAN_PLUS",
    *RANDOM_CONDITIONS,
)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _v2_score(
    raw_output: str,
    *,
    reference_answer: str,
    evaluator: str,
    token_count: int,
) -> dict[str, Any]:
    parsed = parse_external_answer_v2(
        raw_output,
        truncated=token_count >= MAX_NEW_TOKENS,
    )
    if parsed.status is not None:
        status = parsed.status.value.replace("TRUNCATED_THINKING", "TRUNCATED")
        return {
            "status": status,
            "correct": False,
            "parsed_answer": parsed.answer_text,
            "parse_reason": parsed.parse_reason,
        }
    try:
        correct = evaluate_external_answer(parsed.answer_text or "", reference_answer, evaluator)
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": ExternalStatus.INVALID_FORMAT.value,
            "correct": False,
            "parsed_answer": parsed.answer_text,
            "parse_reason": f"typed evaluator rejected answer: {type(exc).__name__}",
        }
    return {
        "status": ExternalStatus.VALID_CORRECT.value
        if correct
        else ExternalStatus.VALID_WRONG.value,
        "correct": bool(correct),
        "parsed_answer": parsed.answer_text,
        "parse_reason": None,
    }


def _load_lock(review: Path) -> dict[str, Any]:
    lock = json.loads((review / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if lock["parser"]["version"] != PARSER_VERSION:
        raise RuntimeError("runtime parser does not match Gate 6.3 lock")
    return lock


def _load_deltas(review: Path, condition: str) -> dict[int, np.ndarray] | None:
    if condition == "BASELINE" or condition == "TEXTUAL_CAREFUL_REFERENCE":
        return None
    lock = _load_lock(review)
    controller = lock["controller"]
    if condition == "BEST_SINGLE_MEAN_PLUS":
        vector_path = (
            ROOT
            / "review/gate6_2_first_stage_repair_mean_bridge"
            / "PAIRED_MEAN_DIRECTIONS"
            / "PROMPT_BOUNDARY"
            / "L27.npy"
        )
        expected_hash = controller["vector_sha256"]
    elif condition in RANDOM_CONDITIONS:
        record = lock["random_bank"]["random_conditions"][condition]
        vector_path = ROOT / record["vector_path"]
        expected_hash = record["vector_sha256"]
    else:
        raise KeyError(condition)
    vector = np.load(vector_path, allow_pickle=False).astype(np.float64).reshape(-1)
    actual_hash = vector_sha256(vector)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"controller hash mismatch for {condition}: {actual_hash} != {expected_hash}"
        )
    delta = vector * (
        float(lock["controller"]["eta0"]) * float(lock["controller"]["reference_scale"])
    )
    if not np.isfinite(delta).all():
        raise RuntimeError(f"non-finite controller delta for {condition}")
    return {27: delta}


def _condition_context(
    backend: Any,
    item: Any,
    condition: str,
    deltas: dict[int, np.ndarray] | None,
) -> tuple[Any, Any, dict[str, Any]]:
    system = None
    if condition == "TEXTUAL_CAREFUL_REFERENCE":
        from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL

        system = SYSTEM_CAREFUL
    model_row = model_item(item, system)
    prompt_ids, _rendered, _hash = prompt_tokens(backend, model_row)
    if deltas is None:
        return nullcontext(), model_row, {"prompt_length": len(prompt_ids), "system_prompt": system}
    torch = backend.torch
    delta_tensors = {
        layer: torch.tensor(value, dtype=torch.float32, device=backend.device).view(1, 1, -1)
        for layer, value in deltas.items()
    }
    context = Gate6HookTrace(
        layers={layer: backend.layer_module(layer) for layer in delta_tensors},
        deltas=delta_tensors,
        target_positions=[len(prompt_ids) - 1],
    )
    return (
        context,
        model_row,
        {
            "prompt_length": len(prompt_ids),
            "system_prompt": system,
            "intervention_duration": "sustained_current_token",
            "intervention_layer": 27,
        },
    )


def execute_phase(backend: Any, review: Path, phase: str, manifest: Path) -> None:
    require_remote_hf_execution(f"Gate 6.3 {phase} inference")
    lock = _load_lock(review)
    items = load_external(manifest)
    if phase == "MATCHED_RANDOM":
        schedule = json.loads((review / "MATCHED_RANDOM_SCHEDULE.json").read_text())
        expected_conditions = set(RANDOM_CONDITIONS)
    elif phase == "EVALUATION":
        schedule = json.loads((review / "EVALUATION_SCHEDULE.json").read_text())
        expected_conditions = set(EVALUATION_CONDITIONS)
    else:
        raise ValueError(phase)
    item_by_id = {item.item_id: item for item in items}
    if set(item_by_id) != {str(row["item_id"]) for row in schedule}:
        raise RuntimeError(f"{phase} manifest and frozen schedule item sets differ")
    if {str(row["condition"]) for row in schedule} != expected_conditions:
        raise RuntimeError(f"{phase} condition set differs from lock")
    journal = review / "journal.jsonl"
    completed = _completed(journal)
    for row in schedule:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in completed:
            continue
        item = item_by_id[str(row["item_id"])]
        condition = str(row["condition"])
        deltas = _load_deltas(review, condition)
        context, model_row, context_meta = _condition_context(backend, item, condition, deltas)
        started = time.perf_counter()
        try:
            with context as trace:
                output = backend.generate_reasoning(
                    model_row,
                    sampling_seed=int(row["seed"]),
                    max_new_tokens=MAX_NEW_TOKENS,
                    intervention_metadata={
                        "gate6_3_phase": phase,
                        "condition": condition,
                        "intervention": condition if deltas else "none",
                        "intervention_duration": "sustained_current_token" if deltas else "none",
                        "intervention_layers": sorted(deltas) if deltas else [],
                        "intervention_vector_hashes": (
                            {str(layer): vector_sha256(value) for layer, value in deltas.items()}
                            if deltas
                            else {}
                        ),
                        "eta0": lock["controller"]["eta0"],
                        "standardized_reference_scale": lock["controller"]["reference_scale"],
                        "parser_version": PARSER_VERSION,
                    },
                )
            elapsed = time.perf_counter() - started
            output_metadata = dict(output.metadata)
            if deltas:
                output_metadata["intervention_forward_trace"] = trace.metadata()
            token_count = int(output_metadata.get("generated_token_count", 0))
            scored = _v2_score(
                output.raw_output,
                reference_answer=item.reference_answer,
                evaluator=item.evaluator,
                token_count=token_count,
            )
            record = {
                **row,
                "parser_version": PARSER_VERSION,
                "status": scored["status"],
                "correct": scored["correct"],
                "parsed_answer": scored["parsed_answer"],
                "parse_reason": scored["parse_reason"],
                "reference_answer": item.reference_answer,
                "raw_output": output.raw_output,
                "generated_token_ids": output_metadata.get("generated_token_ids", []),
                "generated_token_count": token_count,
                "prompt_hash": item.prompt_hash,
                "rendered_prompt_hash": output_metadata.get("rendered_prompt_hash"),
                "source_revision": item.source_revision,
                "evaluator": item.evaluator,
                "metadata": {"stop_metadata": output_metadata, "context_metadata": context_meta},
                "elapsed_seconds": elapsed,
                "timestamp_utc": datetime.now(UTC).isoformat(),
                "runtime_source_commit": git_metadata(ROOT).get("git_commit"),
            }
        except RuntimeError as exc:
            record = {
                **row,
                "parser_version": PARSER_VERSION,
                "status": "RUNTIME_ERROR",
                "correct": False,
                "parsed_answer": None,
                "raw_output": "",
                "error": str(exc),
                "elapsed_seconds": time.perf_counter() - started,
                "runtime_source_commit": git_metadata(ROOT).get("git_commit"),
            }
        append_jsonl(journal, record)
        completed.add(key)
        if record["status"] == "RUNTIME_ERROR":
            raise RuntimeError(f"Gate 6.3 runtime failure for {key}: {record['error']}")
    write_json(
        review / f"RUN_METADATA_{phase}.json",
        {
            "phase": phase,
            "model": lock["model"],
            "parser_version": PARSER_VERSION,
            "protocol_lock_source_commit": lock["source_commit"],
            "runtime_source_commit": git_metadata(ROOT).get("git_commit"),
            "rows_in_journal_after_phase": len(journal.read_text().splitlines()),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("MATCHED_RANDOM", "EVALUATION"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "review" / "gate6_3_single_mean_semantic_evaluation",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    require_remote_hf_execution(f"Gate 6.3 {args.phase} setup")
    backend = build_backend(args.model_path)
    execute_phase(backend, args.review_dir.resolve(), args.phase, args.manifest.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
