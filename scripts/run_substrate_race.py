#!/usr/bin/env python3
"""Run the bounded, baseline-only Gate 3 model-policy substrate race.

The runner is deliberately serial across model arms and item rows. It keeps a
single append-only journal, performs the frozen technical gate before spending
Stage 2 rows, and selects at most two eligible cells for a second independent
seed using the pre-registered ranking rule.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
from epistemic_geometry.benchmarks.external.gate1 import stage1_technical_pass  # noqa: E402
from epistemic_geometry.benchmarks.v4.character_parser import parse_final_integer  # noqa: E402
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.metrics.errors import error_jaccard, phi_correlation  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MAX_NEW_TOKENS = 4096
GPU_RATE_USD_PER_HOUR = 0.44
SEED_REGIME = "INDEPENDENT_PRIMARY"
INSTRUMENTS = ("FRESH_PSEUDOWORD_LONG", "CRUXEVAL_SEMANTIC")
ARMS = ("QWEN_NONTHINKING", "LLAMA_INSTRUCT")
VALID_STATUSES = {ExternalStatus.VALID_CORRECT, ExternalStatus.VALID_WRONG}


def _atomic_json(path: Path, payload: object) -> None:
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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "journal has malformed line "
                    f"{line_number}; preserve it and repair before resume"
                ) from exc
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping in {path}")
    return value


def _load_char_items(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    items = payload.get("items")
    expected = stable_digest("GATE3-SUBSTRATE-RACE-CHARCOUNT-MANIFEST", canonical_json(items))
    if payload.get("manifest_hash") != expected or payload.get("n_items") != 20:
        raise ValueError("Gate 3 character-count manifest hash or size mismatch")
    if not isinstance(items, list) or len(items) != 20:
        raise ValueError("Gate 3 character-count manifest must contain 20 rows")
    ids = {str(row["item_id"]) for row in items}
    if len(ids) != 20 or not all(
        item_id.startswith("gate3_substrate_charcount_") for item_id in ids
    ):
        raise ValueError("Gate 3 character-count manifest contains non-fresh or duplicate IDs")
    return items


def _load_crux_items(path: Path) -> tuple[list[ExternalItem], dict[str, Any]]:
    manifest = _load_json(path)
    items_path = path.with_suffix(".jsonl")
    items = adapter_for("CRUXEval").load_items(items_path)
    item_records = [item.to_record() for item in items]
    expected = stable_digest("GATE3-SUBSTRATE-RACE-CRUX-MANIFEST", canonical_json(item_records))
    if manifest.get("manifest_hash") != expected or len(items) != 20:
        raise ValueError("Gate 3 CRUXEval manifest hash or size mismatch")
    if len({item.item_id for item in items}) != 20:
        raise ValueError("Gate 3 CRUXEval manifest contains duplicate IDs")
    return items, manifest


def _seed(model_policy: str, instrument: str, item_id: str, rollout_index: int) -> int:
    return stable_seed(
        "GATE3-SUBSTRATE-RACE",
        model_policy,
        instrument,
        item_id,
        rollout_index,
    )


def _model_spec(
    arm: str,
    *,
    qwen_path: str,
    llama_path: str,
    qwen_revision: str,
    llama_revision: str,
) -> dict[str, Any]:
    if arm == "QWEN_NONTHINKING":
        return {
            "arm": arm,
            "model_id": "Qwen/Qwen3-8B",
            "revision": qwen_revision,
            "path": qwen_path,
            "enable_thinking": False,
        }
    if arm == "LLAMA_INSTRUCT":
        return {
            "arm": arm,
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "revision": llama_revision,
            "path": llama_path,
            "enable_thinking": None,
        }
    raise ValueError(f"unknown model arm {arm}")


def _build_backend(spec: dict[str, Any]) -> HuggingFaceBackend:
    config = BackendConfig(
        type="huggingface",
        model_id=spec["model_id"],
        model_path=spec["path"],
        model_revision=spec["revision"],
        tokenizer_id=spec["path"],
        tokenizer_revision=spec["revision"],
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=spec["enable_thinking"],
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        item_batch_size=1,
        batch_size=1,
    )
    return HuggingFaceBackend(
        config,
        model_identifier=spec["model_id"],
        tokenizer_identifier=spec["model_id"],
        model_revision=spec["revision"],
    )


def _make_char_result(
    item: dict[str, Any],
    output: Any,
    *,
    seed: int,
    model_policy: str,
    rollout_index: int,
    elapsed: float,
) -> ExternalResult:
    token_count = int(output.metadata.get("generated_token_count", 0))
    status, parsed, reason = parse_final_integer(
        output.raw_output, truncated=token_count >= MAX_NEW_TOKENS
    )
    if status == "PARSED":
        result_status = (
            ExternalStatus.VALID_CORRECT
            if parsed == int(item["answer"])
            else ExternalStatus.VALID_WRONG
        )
    elif status == "TRUNCATED_THINKING":
        result_status = ExternalStatus.TRUNCATED_THINKING
    else:
        result_status = ExternalStatus.INVALID_FORMAT
    return ExternalResult(
        item_id=str(item["item_id"]),
        benchmark="FRESH_PSEUDOWORD_LONG",
        subtask="procedural_character_count",
        rollout_seed=seed,
        raw_output=output.raw_output,
        parsed_answer=str(parsed) if parsed is not None else None,
        status=result_status,
        correct=result_status == ExternalStatus.VALID_CORRECT,
        reference_answer=str(item["answer"]),
        evaluator="exact_integer",
        token_count=token_count,
        prompt_hash=str(item["prompt_hash"]),
        metadata={
            "model_policy": model_policy,
            "rollout_index": rollout_index,
            "generation_seconds": elapsed,
            "parse_reason": reason,
            "stop_metadata": output.metadata,
        },
    )


def _run_one(
    backend: HuggingFaceBackend,
    model_policy: str,
    instrument: str,
    item: dict[str, Any] | ExternalItem,
    *,
    rollout_index: int,
) -> ExternalResult:
    item_id = item.item_id if isinstance(item, ExternalItem) else str(item["item_id"])
    prompt = item.prompt if isinstance(item, ExternalItem) else str(item["prompt"])
    target = item.reference_answer if isinstance(item, ExternalItem) else str(item["answer"])
    metadata = {
        "instrument": instrument,
        "model_policy": model_policy,
        "seed_regime": SEED_REGIME,
        "rollout_index": rollout_index,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    benchmark_item = BenchmarkItem(
        id=item_id,
        prompt=prompt,
        target=target,
        metadata=metadata,
    )
    seed = _seed(model_policy, instrument, item_id, rollout_index)
    started = time.perf_counter()
    output = backend.generate_reasoning(
        benchmark_item,
        sampling_seed=seed,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    elapsed = time.perf_counter() - started
    if isinstance(item, ExternalItem):
        return score_external_response(
            item,
            output.raw_output,
            rollout_seed=seed,
            truncated=int(output.metadata.get("generated_token_count", 0)) >= MAX_NEW_TOKENS,
            token_count=int(output.metadata.get("generated_token_count", 0)),
            metadata={**metadata, "generation_seconds": elapsed, "stop_metadata": output.metadata},
        )
    return _make_char_result(
        item,
        output,
        seed=seed,
        model_policy=model_policy,
        rollout_index=rollout_index,
        elapsed=elapsed,
    )


def _journal_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    result = record["result"]
    return (
        str(record["model_policy"]),
        str(record["instrument"]),
        str(result["item_id"]),
        int(record["rollout_index"]),
    )


def _load_completed(path: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    completed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("event") != "trajectory":
            continue
        key = _journal_key(record)
        if key in completed and completed[key] != record:
            raise RuntimeError(f"conflicting completed trajectory key: {key}")
        completed[key] = record
    return completed


def _record(
    result: ExternalResult,
    *,
    model_policy: str,
    instrument: str,
    rollout_index: int,
    item: dict[str, Any] | ExternalItem,
) -> dict[str, Any]:
    return {
        "event": "trajectory",
        "model_policy": model_policy,
        "instrument": instrument,
        "rollout_index": rollout_index,
        "item_id": result.item_id,
        "reference_prompt": item.prompt if isinstance(item, ExternalItem) else item["prompt"],
        "result": result.to_record(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def _cell_rows(
    completed: dict[tuple[str, str, str, int], dict[str, Any]],
    model_policy: str,
    instrument: str,
    items: list[dict[str, Any] | ExternalItem],
    rollout_index: int,
) -> list[ExternalResult]:
    rows: list[ExternalResult] = []
    for item in items:
        item_id = item.item_id if isinstance(item, ExternalItem) else str(item["item_id"])
        key = (model_policy, instrument, item_id, rollout_index)
        if key not in completed:
            raise RuntimeError(f"missing completed row for {key}")
        rows.append(ExternalResult.from_record(completed[key]["result"]))
    return rows


def _run_items(
    backend: HuggingFaceBackend,
    completed: dict[tuple[str, str, str, int], dict[str, Any]],
    journal: Path,
    *,
    model_policy: str,
    instrument: str,
    items: list[dict[str, Any] | ExternalItem],
    rollout_index: int,
) -> None:
    for item in items:
        item_id = item.item_id if isinstance(item, ExternalItem) else str(item["item_id"])
        key = (model_policy, instrument, item_id, rollout_index)
        if key in completed:
            continue
        result = _run_one(
            backend,
            model_policy,
            instrument,
            item,
            rollout_index=rollout_index,
        )
        record = _record(
            result,
            model_policy=model_policy,
            instrument=instrument,
            rollout_index=rollout_index,
            item=item,
        )
        _append_jsonl(journal, record)
        completed[key] = record


def _summary(rows: list[ExternalResult], *, technical_pass: bool | None) -> dict[str, Any]:
    counts = Counter(row.status.value for row in rows)
    valid = counts[ExternalStatus.VALID_CORRECT.value] + counts[ExternalStatus.VALID_WRONG.value]
    tokens = [row.token_count for row in rows if row.token_count is not None]
    timings = [float(row.metadata.get("generation_seconds", 0.0)) for row in rows]
    correct = counts[ExternalStatus.VALID_CORRECT.value]
    wrong = counts[ExternalStatus.VALID_WRONG.value]
    mechanical = len(rows) - valid
    return {
        "n": len(rows),
        "valid": valid,
        "valid_completion": valid / len(rows) if rows else 0.0,
        "correct": correct,
        "wrong": wrong,
        "mechanical_failures": mechanical,
        "status_counts": dict(sorted(counts.items())),
        "accuracy_conditional_valid": correct / valid if valid else None,
        "mean_tokens": statistics.mean(tokens) if tokens else None,
        "median_tokens": statistics.median(tokens) if tokens else None,
        "max_tokens": max(tokens) if tokens else None,
        "generation_seconds": sum(timings),
        "estimated_gpu_cost_usd": sum(timings) / 3600.0 * GPU_RATE_USD_PER_HOUR,
        "technical_pass": technical_pass,
        "stage2_complete": len(rows) == 20,
        "eligible_for_resampling": (
            len(rows) == 20
            and valid / len(rows) >= 0.90
            and correct >= 2
            and wrong >= 2
            and mechanical <= wrong
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _select_cells(cell_summaries: dict[str, dict[str, Any]]) -> list[str]:
    eligible = [
        (cell_id, summary)
        for cell_id, summary in cell_summaries.items()
        if summary["eligible_for_resampling"]
    ]
    eligible.sort(
        key=lambda pair: (
            -pair[1]["valid_completion"],
            -min(pair[1]["correct"], pair[1]["wrong"]),
            pair[1]["mechanical_failures"],
            pair[1]["mean_tokens"] if pair[1]["mean_tokens"] is not None else math.inf,
            pair[0],
        )
    )
    return [cell_id for cell_id, _summary in eligible[:2]]


def _resampling_metrics(
    seed_a: list[ExternalResult], seed_b: list[ExternalResult]
) -> dict[str, Any]:
    left = {row.item_id: row for row in seed_a}
    right = {row.item_id: row for row in seed_b}
    ids = sorted(set(left) & set(right))
    valid_a = [row for row in seed_a if row.status in VALID_STATUSES]
    valid_b = [row for row in seed_b if row.status in VALID_STATUSES]
    paired_ids = [
        item_id
        for item_id in ids
        if left[item_id].status in VALID_STATUSES and right[item_id].status in VALID_STATUSES
    ]
    errors_a = np.asarray([not left[item_id].correct for item_id in paired_ids], dtype=bool)
    errors_b = np.asarray([not right[item_id].correct for item_id in paired_ids], dtype=bool)
    if len(paired_ids):
        disagreement = float(np.logical_xor(errors_a, errors_b).mean())
        jaccard = float(error_jaccard(errors_a.tolist(), errors_b.tolist()))
        double_fault = float(np.logical_and(errors_a, errors_b).mean())
        pair_oracle = 1.0 - double_fault
        covariance = float(
            np.mean(
                (errors_a.astype(float) - errors_a.mean())
                * (errors_b.astype(float) - errors_b.mean())
            )
        )
        phi = phi_correlation(errors_a.tolist(), errors_b.tolist())
        phi_value = None if math.isnan(phi) else float(phi)
        cc = int(np.logical_and(~errors_a, ~errors_b).sum())
        cw = int(np.logical_and(~errors_a, errors_b).sum())
        wc = int(np.logical_and(errors_a, ~errors_b).sum())
        ww = int(np.logical_and(errors_a, errors_b).sum())
    else:
        disagreement = jaccard = double_fault = pair_oracle = covariance = None
        phi_value = None
        cc = cw = wc = ww = 0
    acc_a = sum(row.correct for row in valid_a) / len(valid_a) if valid_a else None
    acc_b = sum(row.correct for row in valid_b) / len(valid_b) if valid_b else None
    mean_acc = (
        (sum(row.correct for row in valid_a) + sum(row.correct for row in valid_b))
        / (len(valid_a) + len(valid_b))
        if valid_a or valid_b
        else None
    )
    return {
        "n_items": len(ids),
        "paired_valid_n": len(paired_ids),
        "valid_a": len(valid_a),
        "valid_b": len(valid_b),
        "accuracy_a": acc_a,
        "accuracy_b": acc_b,
        "accuracy_difference": abs(acc_a - acc_b)
        if acc_a is not None and acc_b is not None
        else None,
        "mean_single_rollout_accuracy": mean_acc,
        "n_cc": cc,
        "n_cw": cw,
        "n_wc": wc,
        "n_ww": ww,
        "disagreement": disagreement,
        "error_jaccard": jaccard,
        "double_fault": double_fault,
        "pair_oracle_accuracy": pair_oracle,
        "resampling_gain_vs_max_accuracy": pair_oracle - max(acc_a, acc_b)
        if pair_oracle is not None and acc_a is not None and acc_b is not None
        else None,
        "resampling_gain_vs_mean_accuracy": pair_oracle - mean_acc
        if pair_oracle is not None and mean_acc is not None
        else None,
        "error_covariance": covariance,
        "error_phi": phi_value,
        "error_phi_status": "undefined_zero_variance" if phi_value is None else "defined",
        "low_resolution_status": "LOW_RESOLUTION_TWO_ROLLOUT_PLUGIN_ATTENUATED",
    }


def _unload_backend(backend: HuggingFaceBackend | None) -> None:
    if backend is None:
        return
    torch = backend.torch
    del backend
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_reports(
    output: Path,
    *,
    manifest: dict[str, Any],
    completed: dict[tuple[str, str, str, int], dict[str, Any]],
    cell_summaries: dict[str, dict[str, Any]],
    selected: list[str],
) -> dict[str, Any]:
    cell_rows: list[dict[str, Any]] = []
    for cell_id, summary in sorted(cell_summaries.items()):
        arm, instrument = cell_id.split("/", 1)
        cell_rows.append(
            {"cell": cell_id, "model_policy": arm, "instrument": instrument, **summary}
        )
    _write_csv(output / "CELL_RESULTS.csv", cell_rows)
    resampling_rows: list[dict[str, Any]] = []
    resampling_payload: dict[str, Any] = {}
    for cell_id in selected:
        arm, instrument = cell_id.split("/", 1)
        items = manifest["identity"]["items"][instrument]["items"]
        rows_a = [
            ExternalResult.from_record(completed[(arm, instrument, item_id, 0)]["result"])
            for item_id in items
        ]
        rows_b = [
            ExternalResult.from_record(completed[(arm, instrument, item_id, 1)]["result"])
            for item_id in items
        ]
        metrics = _resampling_metrics(rows_a, rows_b)
        resampling_payload[cell_id] = metrics
        resampling_rows.append({"cell": cell_id, **metrics})
    _write_csv(output / "RESAMPLING_RESULTS.csv", resampling_rows)
    report_lines = [
        "# Gate 3 — Incremental model × policy × benchmark substrate race",
        "",
        "**EXPLORATION ONLY. Baseline generation only; no activation collection, "
        "steering, PCA, geometry, Q2, or holdout access occurred.**",
        "",
        f"- Source commit: `{manifest['identity']['source_commit']}`",
        f"- Maximum cap: `{MAX_NEW_TOKENS}` new tokens",
        f"- Seed regime: `{SEED_REGIME}`",
        "",
        "## Stage 2 cells",
        "",
        "| Cell | Valid/N | Correct | Wrong | Accuracy | Mechanical | Mean tokens | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in cell_rows:
        accuracy = row["accuracy_conditional_valid"]
        report_lines.append(
            f"| `{row['cell']}` | {row['valid']}/{row['n']} ({row['valid_completion']:.1%}) | "
            f"{row['correct']} | {row['wrong']} | {accuracy:.1%} | "
            f"{row['mechanical_failures']} | {row['mean_tokens']:.1f} | "
            f"{row['eligible_for_resampling']} |"
        )
    report_lines += [
        "",
        "## Stage 3 resampling",
        "",
        "| Cell | Acc A | Acc B | Paired valid N | Disagreement | Error Jaccard | "
        "Double fault | Pair oracle | Gain vs mean | Error covariance | Phi | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell_id, metrics in resampling_payload.items():
        report_lines.append(
            f"| `{cell_id}` | {metrics['accuracy_a']:.1%} | {metrics['accuracy_b']:.1%} | "
            f"{metrics['paired_valid_n']} | {metrics['disagreement']:.1%} | "
            f"{metrics['error_jaccard']:.3f} | {metrics['double_fault']:.3f} | "
            f"{metrics['pair_oracle_accuracy']:.1%} | "
            f"{metrics['resampling_gain_vs_mean_accuracy']:.1%} | "
            f"{metrics['error_covariance']:.4f} | "
            f"{metrics['error_phi'] if metrics['error_phi'] is not None else 'null'} | "
            f"`{metrics['low_resolution_status']}` |"
        )
    report_lines += [
        "",
        "## Frozen interpretation boundary",
        "",
        "The race selects a development substrate only. Two-rollout statistics are "
        "low-resolution diagnostics, not item-propensity reliability claims. No "
        "original Q1 intervention was executed.",
        "",
        f"Selected cells for second seed: `{', '.join(selected) if selected else 'NONE'}`.",
    ]
    (output / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return resampling_payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charcount-manifest", type=Path, required=True)
    parser.add_argument("--cruxeval-manifest", type=Path, required=True)
    parser.add_argument("--qwen-path", required=True)
    parser.add_argument("--llama-path", required=True)
    parser.add_argument("--qwen-revision", required=True)
    parser.add_argument("--llama-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_remote_hf_execution("Gate 3 substrate race generation")
    char_items = _load_char_items(args.charcount_manifest)
    crux_items, crux_manifest = _load_crux_items(args.cruxeval_manifest)
    items_by_instrument: dict[str, list[dict[str, Any] | ExternalItem]] = {
        "FRESH_PSEUDOWORD_LONG": char_items,
        "CRUXEVAL_SEMANTIC": crux_items,
    }
    item_identity = {
        "FRESH_PSEUDOWORD_LONG": {
            "manifest_hash": _load_json(args.charcount_manifest)["manifest_hash"],
            "items": [str(item["item_id"]) for item in char_items],
        },
        "CRUXEVAL_SEMANTIC": {
            "manifest_hash": crux_manifest["manifest_hash"],
            "items": [item.item_id for item in crux_items],
            "dataset_revision": crux_manifest["dataset_revision"],
        },
    }
    if len(
        set(item_identity["FRESH_PSEUDOWORD_LONG"]["items"])
        & set(item_identity["CRUXEVAL_SEMANTIC"]["items"])
    ):
        raise ValueError("cross-instrument item IDs must be disjoint")
    model_specs = {
        arm: _model_spec(
            arm,
            qwen_path=args.qwen_path,
            llama_path=args.llama_path,
            qwen_revision=args.qwen_revision,
            llama_revision=args.llama_revision,
        )
        for arm in ARMS
    }
    identity: dict[str, Any] = {
        "experiment": "GATE3_SUBSTRATE_RACE",
        "stage": "EXPLORATION",
        "source_commit": args.source_commit,
        "models": {
            arm: {
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "enable_thinking": spec["enable_thinking"],
                "prompt_mode": "chat",
            }
            for arm, spec in model_specs.items()
        },
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "dtype": "bf16",
            "attention_implementation": "sdpa",
        },
        "seed_regime": SEED_REGIME,
        "seed_namespace": "GATE3-SUBSTRATE-RACE",
        "items": item_identity,
        "stage_plan": {
            "stage1_items": 5,
            "stage2_total_items": 20,
            "max_stage3_cells": 2,
            "max_trajectories": 120,
        },
        "steering": False,
        "activation_collection": False,
        "pca": False,
        "geometry": False,
        "holdout": False,
    }
    identity_hash = stable_digest("GATE3-SUBSTRATE-RACE-IDENTITY", canonical_json(identity))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "MANIFEST.json"
    journal_path = args.output / "journal.jsonl"
    if manifest_path.exists():
        previous = _load_json(manifest_path)
        if previous.get("identity_hash") != identity_hash:
            raise RuntimeError("refusing resume: Gate 3 identity changed")
    manifest = {
        "status": "RUNNING",
        "identity": identity,
        "identity_hash": identity_hash,
        "started_utc": datetime.now(UTC).isoformat(),
        "model_outcomes": True,
    }
    _atomic_json(manifest_path, manifest)
    completed = _load_completed(journal_path)
    cell_summaries: dict[str, dict[str, Any]] = {}
    selected: list[str] = []
    backend: HuggingFaceBackend | None = None
    try:
        for arm in ARMS:
            backend = _build_backend(model_specs[arm])
            for instrument in INSTRUMENTS:
                items = items_by_instrument[instrument]
                cell_id = f"{arm}/{instrument}"
                _run_items(
                    backend,
                    completed,
                    journal_path,
                    model_policy=arm,
                    instrument=instrument,
                    items=items[:5],
                    rollout_index=0,
                )
                stage1_rows = _cell_rows(completed, arm, instrument, items[:5], 0)
                technical_pass = stage1_technical_pass(stage1_rows)
                if technical_pass:
                    _run_items(
                        backend,
                        completed,
                        journal_path,
                        model_policy=arm,
                        instrument=instrument,
                        items=items[5:],
                        rollout_index=0,
                    )
                rows = _cell_rows(
                    completed,
                    arm,
                    instrument,
                    items if technical_pass else items[:5],
                    0,
                )
                cell_summaries[cell_id] = _summary(rows, technical_pass=technical_pass)
            _unload_backend(backend)
            backend = None
        selected = _select_cells(cell_summaries)
        selection_payload = {
            "identity_hash": identity_hash,
            "ranking_rule": [
                "higher valid-completion rate",
                "larger min(valid_correct, valid_wrong)",
                "lower mechanical-failure rate",
                "lower mean generated-token count",
                "lexical arm-ID tie-break",
            ],
            "eligible_cells": [
                cell_id
                for cell_id, summary in cell_summaries.items()
                if summary["eligible_for_resampling"]
            ],
            "selected_cells": selected,
        }
        _atomic_json(args.output / "SELECTION.json", selection_payload)
        for arm in ARMS:
            needed = [cell_id for cell_id in selected if cell_id.startswith(f"{arm}/")]
            if not needed:
                continue
            backend = _build_backend(model_specs[arm])
            for cell_id in needed:
                _arm, instrument = cell_id.split("/", 1)
                _run_items(
                    backend,
                    completed,
                    journal_path,
                    model_policy=arm,
                    instrument=instrument,
                    items=items_by_instrument[instrument],
                    rollout_index=1,
                )
            _unload_backend(backend)
            backend = None
        resampling = _write_reports(
            args.output,
            manifest=manifest,
            completed=completed,
            cell_summaries=cell_summaries,
            selected=selected,
        )
        total_rows = len(completed)
        total_generation_seconds = sum(
            float(record["result"].get("metadata", {}).get("generation_seconds", 0.0))
            for record in completed.values()
        )
        _atomic_json(
            args.output / "COST.json",
            {
                "gpu_rate_usd_per_hour": GPU_RATE_USD_PER_HOUR,
                "trajectory_rows": total_rows,
                "generation_seconds_sum": total_generation_seconds,
                "generation_time_cost_estimate_usd": total_generation_seconds
                / 3600.0
                * GPU_RATE_USD_PER_HOUR,
                "maximum_trajectories": 120,
                "pod_runtime_cost_recorded_after_stop": True,
            },
        )
        manifest.update(
            {
                "status": "COMPLETE",
                "completed_trajectory_rows": total_rows,
                "selected_cells": selected,
                "resampling": resampling,
                "finished_utc": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
    except Exception as exc:
        _unload_backend(backend)
        manifest.update(
            {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "failed_utc": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        raise
    print(
        json.dumps(
            {"status": "COMPLETE", "selected_cells": selected, "trajectory_rows": len(completed)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
