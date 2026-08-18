#!/usr/bin/env python3
"""Run the bounded Q1/Q2 external-benchmark baseline qualification.

The only remote model operation in this script is frozen Qwen3 generation. It
never accepts steering parameters, never loads a holdout, and journals one
item/seed trajectory before moving on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.external.adapters import (  # noqa: E402
    adapter_for,
    candidate_specs,
)
from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalItem,
    ExternalResult,
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.benchmarks.external.metrics import summarize_qualification  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    git_metadata,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
GENERATION_CONFIG = {
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dtype": "bf16",
    "prompt_mode": "chat",
    "enable_thinking": True,
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _spec(name: str):
    for spec in candidate_specs():
        if spec.name == name:
            return spec
    raise ValueError(f"unknown candidate: {name}")


def _selection(
    items: list[ExternalItem], *, candidate: str, offset: int, limit: int
) -> list[ExternalItem]:
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")
    ordered = sorted(
        items,
        key=lambda item: (stable_digest("EXTERNAL-SAMPLE", candidate, item.item_id), item.item_id),
    )
    selected = ordered[offset : offset + limit]
    if len(selected) < limit:
        raise ValueError(f"source contains only {len(selected)} items after offset {offset}")
    return selected


def _seed(candidate: str, item_id: str, rollout_index: int) -> int:
    return stable_seed("EXTERNAL-QUALIFICATION", MODEL_REVISION, candidate, item_id, rollout_index)


def _result_from_runtime_error(item: ExternalItem, seed: int, exc: Exception) -> ExternalResult:
    return ExternalResult(
        item_id=item.item_id,
        benchmark=item.benchmark,
        subtask=item.subtask,
        rollout_seed=seed,
        raw_output="",
        parsed_answer=None,
        status=ExternalStatus.RUNTIME_ERROR,
        correct=False,
        reference_answer=item.reference_answer,
        evaluator=item.evaluator,
        prompt_hash=item.prompt_hash,
        metadata={"exception_type": type(exc).__name__, "exception": str(exc)},
    )


def _load_journal(path: Path) -> tuple[dict[tuple[str, int], ExternalResult], list[dict[str, Any]]]:
    results: dict[tuple[str, int], ExternalResult] = {}
    events: list[dict[str, Any]] = []
    if not path.exists():
        return results, events
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"journal line {line_number} is malformed") from exc
            if record.get("event") == "runtime_error":
                events.append(record)
                continue
            result = ExternalResult.from_record(record)
            key = (result.item_id, result.rollout_seed)
            if key in results and results[key].to_record() != result.to_record():
                raise ValueError(f"conflicting duplicate journal key: {key}")
            results[key] = result
    return results, events


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", required=True, choices=[spec.name for spec in candidate_specs()]
    )
    parser.add_argument("--data", type=Path, required=True, help="Remote normalized JSONL")
    parser.add_argument("--stage", choices=["q1_smoke", "q2_qualification"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=True,
        help="Prospectively frozen generous cap selected from completion diagnostics",
    )
    args = parser.parse_args()
    spec = _spec(args.candidate)
    expected_limit = 20 if args.stage == "q1_smoke" else 50
    expected_seeds = 1 if args.stage == "q1_smoke" else 2
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_new_tokens < 8192:
        parser.error("Q1/Q2 caps below 8192 are not allowed after the low-cap correction")

    adapter = adapter_for(args.candidate)
    items = adapter.load_items(args.data)
    selected = _selection(items, candidate=args.candidate, offset=args.offset, limit=expected_limit)
    rollout_seeds = list(range(expected_seeds))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    journal_path = output / "journal.jsonl"
    identity = {
        "campaign": "external-benchmark-qualification",
        "stage": args.stage,
        "candidate": args.candidate,
        "subtask": spec.subtask,
        "item_ids": [item.item_id for item in selected],
        "item_hashes": [item.item_hash for item in selected],
        "data_path": str(args.data),
        "data_digest": stable_digest(
            "EXTERNAL-DATA", canonical_json([item.to_record() for item in items])
        ),
        "offset": args.offset,
        "limit": expected_limit,
        "rollout_indices": rollout_seeds,
        "generation_config": {**GENERATION_CONFIG, "max_new_tokens": args.max_new_tokens},
        "cap_policy": {
            "kind": "prospective_completion_diagnostic",
            "diagnostic_caps": list(spec.completion_diagnostic_caps),
            "low_cap_2048_runs": "LOW_CAP_DIAGNOSTIC_ONLY",
        },
        "steering": False,
        "holdout": False,
        "source_commit": git_metadata(ROOT).get("git_commit"),
    }
    identity_hash = stable_digest("EXTERNAL-RUN-IDENTITY", canonical_json(identity))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") != identity_hash:
            raise RuntimeError(
                "refusing resume: candidate, item selection, or generation config changed"
            )
    _atomic_json(
        manifest_path,
        {
            "status": "RUNNING",
            "identity": identity,
            "identity_hash": identity_hash,
            "started_utc": datetime.now(UTC).isoformat(),
            "model_outcomes": True,
            "steering": False,
            "geometry": False,
            "confirmatory_accessed": False,
        },
    )
    completed, runtime_events = _load_journal(journal_path)
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=True,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        item_batch_size=1,
        batch_size=1,
    )
    backend = HuggingFaceBackend(config)
    started = time.perf_counter()
    try:
        for item in selected:
            for rollout_index in rollout_seeds:
                seed = _seed(args.candidate, item.item_id, rollout_index)
                key = (item.item_id, seed)
                if key in completed:
                    continue
                benchmark_item = BenchmarkItem(
                    id=item.item_id,
                    prompt=item.prompt,
                    target=item.reference_answer,
                    metadata={
                        "external_benchmark": item.benchmark,
                        "external_subtask": item.subtask,
                        "source_revision": item.source_revision,
                        "prompt_hash": item.prompt_hash,
                    },
                )
                generation_started = time.perf_counter()
                try:
                    backend_output = backend.generate_reasoning(
                        benchmark_item,
                        sampling_seed=seed,
                        max_new_tokens=args.max_new_tokens,
                    )
                    token_count = int(backend_output.metadata.get("generated_token_count", 0))
                    result = score_external_response(
                        item,
                        backend_output.raw_output,
                        rollout_seed=seed,
                        truncated=token_count >= args.max_new_tokens,
                        token_count=token_count,
                        metadata={
                            "rollout_index": rollout_index,
                            "generation_seconds": time.perf_counter() - generation_started,
                            "stop_metadata": backend_output.metadata,
                        },
                    )
                except RuntimeError as exc:
                    runtime_event = {
                        "event": "runtime_error",
                        "item_id": item.item_id,
                        "rollout_seed": seed,
                        "error": _result_from_runtime_error(item, seed, exc).to_record(),
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                    }
                    _append_jsonl(journal_path, runtime_event)
                    runtime_events.append(runtime_event)
                    raise
                _append_jsonl(journal_path, result.to_record())
                completed[key] = result
                try:
                    print(
                        f"completed {args.candidate} item={item.item_id} seed={seed} "
                        f"status={result.status.value} tokens={result.token_count}",
                        flush=True,
                    )
                except BrokenPipeError:
                    # The item row was fsynced before reporting progress.
                    pass
        rows = [completed[(item.item_id, _seed(args.candidate, item.item_id, rollout))]
                for item in selected for rollout in rollout_seeds]
        summary = summarize_qualification(rows)
        _atomic_json(output / "results.json", {"rows": [row.to_record() for row in rows]})
        _atomic_json(output / "summary.json", summary.to_record())
        _atomic_json(
            manifest_path,
            {
                "status": "COMPLETE",
                "identity": identity,
                "identity_hash": identity_hash,
                "completed_utc": datetime.now(UTC).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "model_provenance": backend.provenance(),
                "summary": summary.to_record(),
                "runtime_error_events": len(runtime_events),
            },
        )
        return 0
    except BaseException as exc:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior.update(
            {
                "status": "INTERRUPTED",
                "interrupted_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "completed_rows": len(completed),
            }
        )
        _atomic_json(manifest_path, prior)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
