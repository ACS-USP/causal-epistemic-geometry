#!/usr/bin/env python3
"""Estimate natural completion lengths before the external Q1 smoke.

This is a remote-only, development-only diagnostic.  It uses 3--5 deterministic
items and the fixed 8192 -> 16384 -> 32768 ladder.  Diagnostic outcomes never
enter Q1/Q2 qualification tables or scientific error matrices.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
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
    ExternalStatus,
    score_external_response,
)
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
CAP_LADDER = (8192, 16384, 32768)


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_journal(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"journal line {line_number} is malformed") from exc
            key = (row["item_id"], int(row["metadata"]["diagnostic_cap"]))
            if key in rows and rows[key] != row:
                raise ValueError(f"conflicting duplicate diagnostic key: {key}")
            rows[key] = row
    return rows


def _selection(items: list[ExternalItem], candidate: str, limit: int) -> list[ExternalItem]:
    ordered = sorted(
        items,
        key=lambda item: (
            stable_digest("EXTERNAL-COMPLETION-DIAGNOSTIC", candidate, item.item_id),
            item.item_id,
        ),
    )
    if not 3 <= limit <= 5:
        raise ValueError("completion diagnostics require 3 to 5 items")
    if len(ordered) < limit:
        raise ValueError(f"source contains only {len(ordered)} items")
    return ordered[:limit]


def _seed(candidate: str, item_id: str) -> int:
    return stable_seed("EXTERNAL-COMPLETION-DIAGNOSTIC", MODEL_REVISION, candidate, item_id, 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", required=True, choices=[spec.name for spec in candidate_specs()]
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, default=5)
    args = parser.parse_args()
    spec = next(spec for spec in candidate_specs() if spec.name == args.candidate)
    adapter = adapter_for(args.candidate)
    selected = _selection(adapter.load_items(args.data), args.candidate, args.items)
    identity = {
        "campaign": "external-benchmark-qualification",
        "stage": "completion_diagnostic",
        "candidate": args.candidate,
        "subtask": spec.subtask,
        "item_ids": [item.item_id for item in selected],
        "item_hashes": [item.item_hash for item in selected],
        "data_path": str(args.data),
        "data_digest": stable_digest(
            "EXTERNAL-DATA",
            canonical_json([item.to_record() for item in adapter.load_items(args.data)]),
        ),
        "caps": list(CAP_LADDER),
        "rollout_index": 0,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "steering": False,
        "geometry": False,
        "confirmatory_accessed": False,
        "source_commit": git_metadata(ROOT).get("git_commit"),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    identity_hash = stable_digest("EXTERNAL-COMPLETION-DIAGNOSTIC", canonical_json(identity))
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") != identity_hash:
            raise RuntimeError("refusing resume: completion diagnostic identity changed")
    _atomic_json(
        manifest_path,
        {
            "status": "RUNNING",
            "classification": "DEVELOPMENT_ONLY_NOT_SCIENTIFIC_OUTCOMES",
            "identity": identity,
            "identity_hash": identity_hash,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    journal_path = output / "journal.jsonl"
    completed_rows = _load_journal(journal_path)
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=max(CAP_LADDER),
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
    completed = 0
    started = time.perf_counter()
    try:
        for item in selected:
            seed = _seed(args.candidate, item.item_id)
            benchmark_item = BenchmarkItem(
                id=item.item_id,
                prompt=item.prompt,
                target=item.reference_answer,
                metadata={"external_benchmark": item.benchmark, "external_subtask": item.subtask},
            )
            for attempt_index, cap in enumerate(CAP_LADDER):
                prior = completed_rows.get((item.item_id, cap))
                if prior is not None:
                    if prior["status"] != ExternalStatus.TRUNCATED_THINKING.value:
                        break
                    continue
                generation_started = time.perf_counter()
                backend_output = backend.generate_reasoning(
                    benchmark_item, sampling_seed=seed, max_new_tokens=cap
                )
                token_count = int(backend_output.metadata.get("generated_token_count", 0))
                result = score_external_response(
                    item,
                    backend_output.raw_output,
                    rollout_seed=seed,
                    truncated=token_count >= cap,
                    token_count=token_count,
                    metadata={
                        "diagnostic_cap": cap,
                        "attempt_index": attempt_index,
                        "generation_seconds": time.perf_counter() - generation_started,
                        "stop_metadata": backend_output.metadata,
                    },
                )
                result = replace(
                    result,
                    metadata={
                        **result.metadata,
                        "natural_completion": result.status != ExternalStatus.TRUNCATED_THINKING,
                    },
                )
                _append_jsonl(journal_path, result.to_record())
                completed_rows[(item.item_id, cap)] = result.to_record()
                completed += 1
                try:
                    print(
                        f"completed diagnostic candidate={args.candidate} item={item.item_id} "
                        f"cap={cap} status={result.status.value} tokens={token_count}",
                        flush=True,
                    )
                except BrokenPipeError:
                    # The journal is already durable; a disconnected SSH stdout
                    # stream must not invalidate a completed diagnostic row.
                    pass
                if result.status != ExternalStatus.TRUNCATED_THINKING:
                    break
        _atomic_json(
            output / "summary.json",
            {
                "candidate": args.candidate,
                "classification": "LOW_CAP_DIAGNOSTICS_DO_NOT_USE_FOR_QUALIFICATION",
                "rows": len(completed_rows),
                "caps": list(CAP_LADDER),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        _atomic_json(
            manifest_path,
            {
                "status": "COMPLETE",
                "classification": "DEVELOPMENT_ONLY_NOT_SCIENTIFIC_OUTCOMES",
                "identity": identity,
                "identity_hash": identity_hash,
                "completed_utc": datetime.now(UTC).isoformat(),
                "rows": len(completed_rows),
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
                "completed_rows": completed,
            }
        )
        _atomic_json(manifest_path, prior)
        raise
if __name__ == "__main__":
    raise SystemExit(main())
