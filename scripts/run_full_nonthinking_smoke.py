#!/usr/bin/env python3
"""Execute the authorized Gate 1 full non-thinking smoke on RunPod.

The runner is intentionally serial and quiet while outcomes are being
generated.  It journals every completed trajectory, applies Stage 1 only as a
technical continuation gate, and does not run any steering or activation code.
"""

from __future__ import annotations

import argparse
import csv
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
from epistemic_geometry.benchmarks.external.adapters import adapter_for  # noqa: E402
from epistemic_geometry.benchmarks.external.base import (  # noqa: E402
    ExternalItem,
    ExternalResult,
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.benchmarks.external.gate1 import (  # noqa: E402
    classify_full_n20,
    stage1_technical_pass,
    summarize_gate1,
)
from epistemic_geometry.benchmarks.v4.character_parser import (  # noqa: E402
    parse_final_integer,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_NEW_TOKENS = 4096
GPU_RATE_USD_PER_HOUR = 0.44
INSTRUMENTS = ("FRESH_PSEUDOWORD_LONG", "CRUXEVAL_SEMANTIC")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed JSONL") from exc
    return rows


def _load_char_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items")
    expected = stable_digest("FULL-NONTHINKING-CHARCOUNT-MANIFEST", canonical_json(items))
    if payload.get("manifest_hash") != expected or payload.get("n_items") != 20:
        raise ValueError("Gate 1 character-count manifest hash or size mismatch")
    if len(items) != 20 or len({row["item_id"] for row in items}) != 20:
        raise ValueError("Gate 1 character-count manifest must contain 20 unique rows")
    if not all(
        str(row["item_id"]).startswith("gate1_full_nonthinking_charcount_")
        for row in items
    ):
        raise ValueError("Gate 1 character-count manifest contains non-fresh IDs")
    return items


def _load_crux_items(path: Path) -> list[ExternalItem]:
    items = adapter_for("CRUXEval").load_items(path)
    if len(items) != 20:
        raise ValueError("Gate 1 CRUXEval manifest must contain exactly 20 items")
    return items


def _seed(instrument: str, item_id: str, *, thinking: bool = False) -> int:
    namespace = (
        "FULL-NONTHINKING-SMOKE-NATIVE-THINKING-REFERENCE"
        if thinking
        else "FULL-NONTHINKING-SMOKE"
    )
    return stable_seed(namespace, MODEL_REVISION, instrument, item_id, 0)


def _char_result(
    row: dict[str, Any], raw_output: str, *, seed: int, token_count: int, metadata: dict[str, Any]
) -> ExternalResult:
    status, parsed, reason = parse_final_integer(
        raw_output, truncated=token_count >= MAX_NEW_TOKENS
    )
    if status == "PARSED":
        outcome = (
            ExternalStatus.VALID_CORRECT
            if parsed == int(row["answer"])
            else ExternalStatus.VALID_WRONG
        )
    elif status == "TRUNCATED_THINKING":
        outcome = ExternalStatus.TRUNCATED_THINKING
    else:
        outcome = ExternalStatus.INVALID_FORMAT
    return ExternalResult(
        item_id=str(row["item_id"]),
        benchmark="FRESH_PSEUDOWORD_LONG",
        subtask="procedural_character_count",
        rollout_seed=seed,
        raw_output=raw_output,
        parsed_answer=str(parsed) if parsed is not None else None,
        status=outcome,
        correct=outcome == ExternalStatus.VALID_CORRECT,
        reference_answer=str(row["answer"]),
        evaluator="exact_integer",
        token_count=token_count,
        prompt_hash=str(row["prompt_hash"]),
        metadata={**metadata, "parse_reason": reason},
    )


def _result_from_record(record: dict[str, Any]) -> ExternalResult:
    return ExternalResult.from_record(record)


def _load_completed(path: Path) -> dict[tuple[str, str, int], ExternalResult]:
    completed: dict[tuple[str, str, int], ExternalResult] = {}
    if not path.exists():
        return completed
    for record in _load_jsonl(path):
        if record.get("event") != "trajectory":
            continue
        result = _result_from_record(record["result"])
        key = (str(record["instrument"]), result.item_id, result.rollout_seed)
        if key in completed and completed[key].to_record() != result.to_record():
            raise RuntimeError(f"conflicting completed trajectory key: {key}")
        completed[key] = result
    return completed


def _build_backend(*, enable_thinking: bool) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=enable_thinking,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        item_batch_size=1,
        batch_size=1,
    )
    return HuggingFaceBackend(config)


def _run_one(
    backend: HuggingFaceBackend,
    instrument: str,
    item: dict[str, Any] | ExternalItem,
    *,
    thinking: bool,
) -> ExternalResult:
    if isinstance(item, ExternalItem):
        item_id = item.item_id
        prompt = item.prompt
        target = item.reference_answer
        metadata = {
            "external_benchmark": item.benchmark,
            "external_subtask": item.subtask,
            "source_revision": item.source_revision,
            "prompt_hash": item.prompt_hash,
        }
    else:
        item_id = str(item["item_id"])
        prompt = str(item["prompt"])
        target = str(item["answer"])
        metadata = {
            "stratum": item["stratum"],
            "text": item["text"],
            "target_character": item["target_character"],
            "prompt_hash": item["prompt_hash"],
        }
    seed = _seed(instrument, item_id, thinking=thinking)
    benchmark_item = BenchmarkItem(id=item_id, prompt=prompt, target=target, metadata=metadata)
    started = time.perf_counter()
    output = backend.generate_reasoning(
        benchmark_item, sampling_seed=seed, max_new_tokens=MAX_NEW_TOKENS
    )
    elapsed = time.perf_counter() - started
    token_count = int(output.metadata.get("generated_token_count", 0))
    result_metadata = {
        "instrument": instrument,
        "response_mode": "native_thinking_reference" if thinking else "full_nonthinking",
        "rollout_index": 0,
        "generation_seconds": elapsed,
        "stop_metadata": output.metadata,
    }
    if isinstance(item, ExternalItem):
        return score_external_response(
            item,
            output.raw_output,
            rollout_seed=seed,
            truncated=token_count >= MAX_NEW_TOKENS,
            token_count=token_count,
            metadata=result_metadata,
        )
    return _char_result(
        item,
        output.raw_output,
        seed=seed,
        token_count=token_count,
        metadata=result_metadata,
    )


def _record(result: ExternalResult, *, instrument: str, item: Any) -> dict[str, Any]:
    return {
        "event": "trajectory",
        "instrument": instrument,
        "item_id": result.item_id,
        "result": result.to_record(),
        "reference_prompt": item.prompt if isinstance(item, ExternalItem) else item["prompt"],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _report(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    *,
    total_trajectories: int,
    elapsed_seconds: float,
    thinking_summaries: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Gate 1 — Full Non-Thinking Generation Smoke",
        "",
        "**EXPLORATION ONLY. No steering, activation collection, PCA, geometry, "
        "Q2, or holdout access occurred.**",
        "",
        f"- Total trajectories: {total_trajectories}",
        f"- Wall time seconds: {elapsed_seconds:.3f}",
        f"- Frozen cap: `{MAX_NEW_TOKENS}` new tokens",
        "- Model: `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`",
        "- Generation: BF16, SDPA, sampled, temperature 0.6, top-p 0.95, top-k 20, min-p 0",
        "",
        "## Instrument outcomes",
        "",
        "| Instrument | n | Valid | Correct | Wrong | Mechanical failures | "
        "Stage 1 | Classification |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for instrument, summary in summaries.items():
        lines.append(
            f"| `{instrument}` | {summary['n']} | {summary['valid_count']} "
            f"({summary['valid_completion']:.1%}) | {summary['correct_count']} | "
            f"{summary['wrong_count']} | {summary['mechanical_failure_count']} | "
            f"{summary['stage1_technical_pass']} | `{summary['classification']}` |"
        )
    lines += [
        "",
        "## Native-thinking reference",
        "",
        "This arm is descriptive only and is run only for a full n=20 non-thinking "
        "instrument classified `PROMISING`.",
    ]
    if thinking_summaries:
        lines += ["", "| Instrument | n | Valid | Correct | Wrong |", "|---|---:|---:|---:|---:|"]
        for instrument, summary in thinking_summaries.items():
            lines.append(
                f"| `{instrument}` | {summary['n']} | {summary['valid_count']} "
                f"({summary['valid_completion']:.1%}) | {summary['correct_count']} | "
                f"{summary['wrong_count']} |"
            )
    else:
        lines.append("\nNot run: no non-thinking instrument was classified `PROMISING`.")
    lines += [
        "",
        "## Scientific boundary",
        "",
        "These results can qualify or reject a cheap measurement instrument only. "
        "They are not evidence for or against the causal-geometry question.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charcount-manifest", type=Path, required=True)
    parser.add_argument("--cruxeval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if MAX_NEW_TOKENS != 4096:
        raise RuntimeError("Gate 1 cap changed unexpectedly")
    require_remote_hf_execution("Gate 1 Qwen generation")
    char_items = _load_char_items(args.charcount_manifest)
    crux_items = _load_crux_items(args.cruxeval_data)
    items_by_instrument: dict[str, list[Any]] = {
        "FRESH_PSEUDOWORD_LONG": char_items,
        "CRUXEVAL_SEMANTIC": crux_items,
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "journal.jsonl"
    identity = {
        "experiment": "FULL_NONTHINKING_SMOKE",
        "stage": "EXPLORATION",
        "source_commit": args.source_commit,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bf16",
        "attention_implementation": "sdpa",
        "enable_thinking": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        },
        "instruments": {
            key: {
                "item_ids": [
                    item.item_id if isinstance(item, ExternalItem) else item["item_id"]
                    for item in value
                ],
                "manifest_digest": stable_digest(
                    "GATE1-ITEMS",
                    canonical_json(
                        [
                            item.to_record()
                            if isinstance(item, ExternalItem)
                            else item
                            for item in value
                        ]
                    ),
                ),
            }
            for key, value in items_by_instrument.items()
        },
        "steering": False,
        "activation_collection": False,
        "geometry": False,
        "holdout": False,
    }
    identity_hash = stable_digest("FULL-NONTHINKING-RUN", canonical_json(identity))
    manifest_path = output / "MANIFEST.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("identity_hash") != identity_hash:
            raise RuntimeError("refusing resume: Gate 1 identity changed")
    _atomic_json(
        manifest_path,
        {
            "status": "RUNNING",
            "identity": identity,
            "identity_hash": identity_hash,
            "started_utc": datetime.now(UTC).isoformat(),
            "model_outcomes": True,
        },
    )
    completed = _load_completed(journal)
    all_results: dict[str, list[ExternalResult]] = {instrument: [] for instrument in INSTRUMENTS}
    summaries: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    backend = _build_backend(enable_thinking=False)
    try:
        stage1_results_by_instrument: dict[str, list[ExternalResult]] = {}
        technical_pass_by_instrument: dict[str, bool] = {}
        # Complete the five-item technical screen for every candidate before
        # allowing any candidate to spend the additional fifteen rows.
        for instrument in INSTRUMENTS:
            items = items_by_instrument[instrument]
            stage1_items = items[:5]
            stage1_results: list[ExternalResult] = []
            for item in stage1_items:
                item_id = item.item_id if isinstance(item, ExternalItem) else str(item["item_id"])
                seed = _seed(instrument, item_id)
                key = (instrument, item_id, seed)
                if key not in completed:
                    result = _run_one(backend, instrument, item, thinking=False)
                    _append_jsonl(journal, _record(result, instrument=instrument, item=item))
                    completed[key] = result
                stage1_results.append(completed[key])
            stage1_results_by_instrument[instrument] = stage1_results
            technical_pass_by_instrument[instrument] = stage1_technical_pass(stage1_results)

        for instrument in INSTRUMENTS:
            items = items_by_instrument[instrument]
            technical_pass = technical_pass_by_instrument[instrument]
            stage1_results = stage1_results_by_instrument[instrument]
            results = list(stage1_results)
            if technical_pass:
                for item in items[5:20]:
                    item_id = (
                        item.item_id
                        if isinstance(item, ExternalItem)
                        else str(item["item_id"])
                    )
                    seed = _seed(instrument, item_id)
                    key = (instrument, item_id, seed)
                    if key not in completed:
                        result = _run_one(backend, instrument, item, thinking=False)
                        _append_jsonl(journal, _record(result, instrument=instrument, item=item))
                        completed[key] = result
                    results.append(completed[key])
            classification = (
                classify_full_n20(results)
                if len(results) == 20
                else "STAGE1_TECHNICAL_FAIL"
            )
            summary = summarize_gate1(
                instrument,
                results,
                stage1=technical_pass,
                classification=classification,
            ).to_record()
            all_results[instrument] = results
            summaries[instrument] = summary
        thinking_summaries: dict[str, dict[str, Any]] = {}
        promising = [
            key
            for key, summary in summaries.items()
            if summary["classification"] == "PROMISING"
        ]
        if promising:
            thinking_backend = _build_backend(enable_thinking=True)
            for instrument in promising:
                items = items_by_instrument[instrument][:5]
                results = []
                for item in items:
                    item_id = (
                        item.item_id
                        if isinstance(item, ExternalItem)
                        else str(item["item_id"])
                    )
                    seed = _seed(instrument, item_id, thinking=True)
                    key = (f"{instrument}:native_thinking_reference", item_id, seed)
                    if key not in completed:
                        result = _run_one(thinking_backend, instrument, item, thinking=True)
                        _append_jsonl(
                            journal,
                            _record(
                                result,
                                instrument=f"{instrument}:native_thinking_reference",
                                item=item,
                            ),
                        )
                        completed[key] = result
                    results.append(completed[key])
                thinking_summaries[instrument] = summarize_gate1(
                    instrument, results, stage1=None, classification="DESCRIPTIVE_ONLY"
                ).to_record()
        elapsed = time.perf_counter() - started
        total = len(completed)
        _report(
            output,
            summaries,
            total_trajectories=total,
            elapsed_seconds=elapsed,
            thinking_summaries=thinking_summaries,
        )
        rows = [
            {
                "instrument": instrument,
                "item_id": result.item_id,
                "seed": result.rollout_seed,
                "status": result.status.value,
                "correct": result.correct,
                "reference_answer": result.reference_answer,
                "parsed_answer": result.parsed_answer,
                "token_count": result.token_count,
            }
            for instrument, results in all_results.items()
            for result in results
        ]
        with (output / "RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])) if rows else None
            if writer:
                writer.writeheader()
                writer.writerows(rows)
        cost = {
            "gpu_rate_usd_per_a40_hour": GPU_RATE_USD_PER_HOUR,
            "wall_seconds": elapsed,
            "estimated_gpu_hours": elapsed / 3600,
            "estimated_cost_usd": elapsed / 3600 * GPU_RATE_USD_PER_HOUR,
            "trajectory_count": total,
            "cost_cap_usd": 0.30,
        }
        _atomic_json(output / "COST.json", cost)
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior.update(
            {
                "status": "COMPLETE",
                "completed_utc": datetime.now(UTC).isoformat(),
                "trajectory_count": total,
                "summaries": summaries,
                "thinking_summaries": thinking_summaries,
                "cost": cost,
                "journal_sha256": stable_digest("FULL-NONTHINKING-JOURNAL", journal.read_text()),
            }
        )
        _atomic_json(manifest_path, prior)
        print(
            json.dumps(
                {
                    "summaries": summaries,
                    "thinking_summaries": thinking_summaries,
                    "cost": cost,
                },
                indent=2,
                sort_keys=True,
            )
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
                "completed_trajectories": len(completed),
            }
        )
        _atomic_json(manifest_path, prior)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
