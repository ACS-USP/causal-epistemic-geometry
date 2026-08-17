#!/usr/bin/env python3
"""Bounded real-Qwen equivalence/performance gate for Q1 V3 engines.

This script is intentionally limited to already-consumed Stage-A engineering
items.  It never evaluates the holdout, never constructs steering directions,
and never launches the full Stage-A calibration.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from epistemic_geometry.backends.base import build_backend
from epistemic_geometry.benchmarks.reasoning.engines import (
    BATCHED_REASONING,
    MAX_BUDGET_PREFIX_REUSE,
    derive_budget_outputs,
    physical_generation_id,
)
from epistemic_geometry.benchmarks.reasoning.rendering import render_reasoning
from epistemic_geometry.benchmarks.reasoning.rollouts import (
    rollout_record_from_output,
    rollout_seed,
)
from epistemic_geometry.benchmarks.reasoning.splits import ReasoningSplit
from epistemic_geometry.config import load_config
from epistemic_geometry.reproducibility import stable_digest


def _load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("manifests"), dict):
        raise ValueError("manifest must contain a manifests mapping")
    return payload


def _record_summary(record: Any) -> dict[str, Any]:
    return {
        "token_ids": list(record.token_ids),
        "token_hash": stable_digest("q1-v3-token-row", list(record.token_ids)),
        "parse_status": record.parse_status,
        "parsed_answer": record.parsed_answer,
        "correct": record.correct,
    }


def _compare_rows(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_ids = left["token_ids"]
    right_ids = right["token_ids"]
    first_mismatch = next(
        (
            index
            for index, (left_token, right_token) in enumerate(
                zip(left_ids, right_ids, strict=False)
            )
            if left_token != right_token
        ),
        None,
    )
    if first_mismatch is None and len(left_ids) != len(right_ids):
        first_mismatch = min(len(left_ids), len(right_ids))
    return {
        "token_ids_equal": left_ids == right_ids,
        "parse_equal": (
            left["parse_status"] == right["parse_status"]
            and left["parsed_answer"] == right["parsed_answer"]
        ),
        "correct_equal": left["correct"] == right["correct"],
        "left_length": len(left_ids),
        "right_length": len(right_ids),
        "first_token_mismatch": first_mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest-key",
        action="append",
        dest="manifest_keys",
        help="Repeat for one or more family/cell groups; defaults to the first group.",
    )
    parser.add_argument("--max-items", type=int, default=2)
    parser.add_argument("--max-rollouts", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--padding-side", choices=("left", "right"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    if args.max_items <= 0 or args.max_rollouts <= 0 or args.batch_size <= 0:
        raise SystemExit("max-items, max-rollouts, and batch-size must be positive")

    config = load_config(args.config)
    backend_config = config.backend
    if args.padding_side is not None:
        backend_config = replace(backend_config, padding_side=args.padding_side)
    if args.model_path is not None:
        backend_config = replace(backend_config, model_path=str(args.model_path))
    if args.tokenizer_path is not None:
        backend_config = replace(backend_config, tokenizer_id=str(args.tokenizer_path))
    if backend_config is not config.backend:
        config = replace(config, backend=backend_config)
    payload = _load_payload(args.manifest)
    available = sorted(payload["manifests"])
    requested_groups = args.manifest_keys or sorted(
        {key.rsplit("/", 1)[0] for key in available}
    )[:1]
    selected: dict[tuple[str, str], dict[int, ReasoningSplit]] = {}
    for group in requested_groups:
        rows = {
            int(key.rsplit("/", 1)[1]): ReasoningSplit.from_record(
                payload["manifests"][key], development=True
            )
            for key in available
            if key.rsplit("/", 1)[0] == group
        }
        if not rows:
            raise KeyError(f"manifest group not found: {group}")
        selected[tuple(group.split("/", 1))] = rows

    backend = build_backend(config)
    budgets = tuple(sorted(next(iter(selected.values()))))
    serial_rows: dict[str, dict[int, dict[str, Any]]] = {}
    prefix_rows: dict[str, dict[int, dict[str, Any]]] = {}
    batched_rows: dict[str, dict[int, dict[str, Any]]] = {}
    pairwise: list[dict[str, Any]] = []
    serial_seconds = 0.0
    prefix_seconds = 0.0
    batched_seconds = 0.0
    physical_serial = 0
    physical_prefix = 0
    physical_batched = 0

    for (_family, _cell), budget_rows in sorted(selected.items()):
        source_split = budget_rows[max(budgets)]
        items = source_split.items[: args.max_items]
        tasks: list[tuple[Any, int]] = []
        for item in items:
            view = render_reasoning(item, surface="canonical")
            for rollout_index in range(args.max_rollouts):
                seed = rollout_seed(
                    config.experiment.seed,
                    view.latent_id,
                    "baseline",
                    rollout_index,
                    regime="independent",
                )
                tasks.append((view, seed))

        for view, seed in tasks:
            independent: dict[int, dict[str, Any]] = {}
            started = time.perf_counter()
            for budget in budgets:
                output = backend.generate_reasoning_view(
                    view, sampling_seed=seed, max_new_tokens=budget
                )
                record = rollout_record_from_output(
                    view,
                    output,
                    intervention_id="baseline",
                    rollout_index=0,
                    sampling_seed=seed,
                    generation_config={"max_new_tokens": budget},
                )
                independent[budget] = _record_summary(record)
                physical_serial += 1
            serial_seconds += time.perf_counter() - started

            started = time.perf_counter()
            source_output = backend.generate_reasoning_view(
                view, sampling_seed=seed, max_new_tokens=max(budgets)
            )
            physical_prefix += 1
            derived = derive_budget_outputs(
                source_output,
                view_id=view.view_id,
                sampling_seed=seed,
                source_max_budget=max(budgets),
                budgets=budgets,
                decode_tokens=lambda ids: backend.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
            )
            prefix_for_task: dict[int, dict[str, Any]] = {}
            for budget in budgets:
                record = rollout_record_from_output(
                    view,
                    derived[budget],
                    intervention_id="baseline",
                    rollout_index=0,
                    sampling_seed=seed,
                    generation_config={"max_new_tokens": budget},
                )
                prefix_for_task[budget] = _record_summary(record)
                pairwise.append(
                    {
                        "view_id": view.view_id,
                        "budget": budget,
                        "independent_vs_2048_prefix": _compare_rows(
                            independent[budget], prefix_for_task[budget]
                        ),
                        "physical_generation_id": physical_generation_id(
                            view_id=view.view_id,
                            sampling_seed=seed,
                            source_max_budget=max(budgets),
                        ),
                    }
                )
            prefix_seconds += time.perf_counter() - started
            serial_rows[view.view_id] = independent
            prefix_rows[view.view_id] = prefix_for_task

        started = time.perf_counter()
        batch_outputs = backend.generate_reasoning_batch(
            tasks,
            max_new_tokens=max(budgets),
            batch_size=args.batch_size,
            max_prefill_tokens=config.backend.max_prefill_tokens,
        )
        batched_seconds += time.perf_counter() - started
        physical_batched += len(tasks)
        for (view, seed), output in zip(tasks, batch_outputs, strict=True):
            derived = derive_budget_outputs(
                output,
                view_id=view.view_id,
                sampling_seed=seed,
                source_max_budget=max(budgets),
                budgets=budgets,
                decode_tokens=lambda ids: backend.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
            )
            batched_rows[view.view_id] = {}
            for budget in budgets:
                record = rollout_record_from_output(
                    view,
                    derived[budget],
                    intervention_id="baseline",
                    rollout_index=0,
                    sampling_seed=seed,
                    generation_config={"max_new_tokens": budget},
                )
                batched_rows[view.view_id][budget] = _record_summary(record)

    prefix_comparisons = [row["independent_vs_2048_prefix"] for row in pairwise]
    serial_prefix_mismatches = sum(not row["token_ids_equal"] for row in prefix_comparisons)
    serial_prefix_parse_mismatches = sum(not row["parse_equal"] for row in prefix_comparisons)
    batch_comparisons = []
    batch_comparison_details = []
    for view_id, rows in serial_rows.items():
        for budget in budgets:
            comparison = _compare_rows(rows[budget], batched_rows[view_id][budget])
            batch_comparisons.append(comparison)
            batch_comparison_details.append(
                {"view_id": view_id, "budget": budget, **comparison}
            )
    batch_token_mismatches = sum(not row["token_ids_equal"] for row in batch_comparisons)
    batch_parse_mismatches = sum(not row["parse_equal"] for row in batch_comparisons)
    result = {
        "status": "SOFTWARE_VALIDATION_ONLY_REAL_QWEN_BOUNDED",
        "model_provenance": backend.provenance(),
        "manifest": str(args.manifest),
        "manifest_groups": requested_groups,
        "items_per_group": args.max_items,
        "rollouts_per_item": args.max_rollouts,
        "budgets": list(budgets),
        "padding_side": config.backend.padding_side,
        "engines": {
            "serial_reasoning_reference": {
                "seconds": serial_seconds,
                "physical_generations": physical_serial,
            },
            MAX_BUDGET_PREFIX_REUSE: {
                "seconds": prefix_seconds,
                "physical_generations": physical_prefix,
                "independent_budget_vs_2048_prefix_token_mismatches": serial_prefix_mismatches,
                "independent_budget_vs_2048_prefix_parse_mismatches": (
                    serial_prefix_parse_mismatches
                ),
            },
            BATCHED_REASONING: {
                "seconds": batched_seconds,
                "physical_generations": physical_batched,
                "batch_size": args.batch_size,
                "serial_vs_batched_token_mismatches": batch_token_mismatches,
                "serial_vs_batched_parse_mismatches": batch_parse_mismatches,
                "serial_vs_batched_comparisons": batch_comparison_details,
            },
        },
        "pairwise": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if serial_prefix_mismatches or serial_prefix_parse_mismatches or batch_token_mismatches:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
