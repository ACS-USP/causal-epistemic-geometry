#!/usr/bin/env python3
"""Benchmark Q1 V3 reasoning engines on a network-free tiny transformer.

This is an engineering benchmark only.  It deliberately uses a randomly
initialized model and toy prompts; its outputs are not scientific evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend
from epistemic_geometry.benchmarks.reasoning.base import ReasoningView
from epistemic_geometry.benchmarks.reasoning.engines import (
    derive_budget_outputs,
    physical_generation_id,
)
from epistemic_geometry.config import BackendConfig


def _views() -> list[ReasoningView]:
    prompts = (
        "Compute the final value. alpha beta",
        "Compute the final value. alpha beta gamma delta",
        "Compute the final value. alpha beta gamma delta epsilon zeta",
        "Compute the final value. alpha beta gamma delta epsilon zeta eta theta",
    )
    return [
        ReasoningView(
            latent_id=f"TINY-ENGINE-{index}",
            view_id=f"TINY-ENGINE-{index}:canonical",
            family="TINY_ENGINE",
            cell="software_fixture",
            surface="canonical",
            answer=0,
            prompt=prompt,
            prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            template_hash="tiny-engine-template-v1",
        )
        for index, prompt in enumerate(prompts)
    ]


def _backend() -> TinyRandomTransformerBackend:
    return TinyRandomTransformerBackend(
        BackendConfig(
            type="tiny_transformer",
            model_id="TINY_RANDOM_GPT2_CONFIG_ONLY",
            hidden_size=32,
            device="cpu",
            dtype="float32",
            layer=0,
            layer_path="transformer.h",
            prompt_mode="plain",
            max_new_tokens=12,
            enable_thinking=True,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            min_p=0.0,
            padding_side="right",
        ),
        seed=19,
    )


def _serial(views: list[ReasoningView], seeds: list[int], budgets: tuple[int, ...]):
    backend = _backend()
    started = time.perf_counter()
    rows: dict[tuple[str, int, int], tuple[int, ...]] = {}
    for budget in budgets:
        for view, seed in zip(views, seeds, strict=True):
            output = backend.generate_reasoning_view(
                view, sampling_seed=seed, max_new_tokens=budget
            )
            rows[(view.view_id, seed, budget)] = tuple(output.metadata["generated_token_ids"])
    return rows, time.perf_counter() - started, len(views) * len(budgets)


def _prefix(
    views: list[ReasoningView],
    seeds: list[int],
    budgets: tuple[int, ...],
    *,
    batched: bool,
    batch_size: int,
):
    backend = _backend()
    tasks = list(zip(views, seeds, strict=True))
    started = time.perf_counter()
    if batched:
        outputs = backend.generate_reasoning_batch(
            tasks,
            max_new_tokens=max(budgets),
            batch_size=batch_size,
            max_prefill_tokens=256,
        )
    else:
        outputs = [
            backend.generate_reasoning_view(view, sampling_seed=seed, max_new_tokens=max(budgets))
            for view, seed in tasks
        ]
    rows: dict[tuple[str, int, int], tuple[int, ...]] = {}
    for (view, seed), output in zip(tasks, outputs, strict=True):
        physical_id = physical_generation_id(
            view_id=view.view_id, sampling_seed=seed, source_max_budget=max(budgets)
        )
        derived = derive_budget_outputs(
            output,
            view_id=view.view_id,
            sampling_seed=seed,
            source_max_budget=max(budgets),
            budgets=budgets,
            decode_tokens=lambda token_ids: backend.tokenizer.decode(
                token_ids, skip_special_tokens=True
            ),
        )
        assert all(
            derived[budget].metadata["physical_generation_id"] == physical_id
            for budget in budgets
        )
        for budget in budgets:
            rows[(view.view_id, seed, budget)] = tuple(
                derived[budget].metadata["generated_token_ids"]
            )
    return rows, time.perf_counter() - started, len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/q1_v3_reasoning_engine_benchmark.json"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    views = _views()
    seeds = [101, 202, 303, 404]
    budgets = (4, 8, 12)
    serial_rows, serial_seconds, serial_physical = _serial(views, seeds, budgets)
    prefix_rows, prefix_seconds, prefix_physical = _prefix(
        views, seeds, budgets, batched=False, batch_size=args.batch_size
    )
    batch_rows, batch_seconds, batch_physical = _prefix(
        views, seeds, budgets, batched=True, batch_size=args.batch_size
    )

    prefix_mismatches = sum(serial_rows[key] != prefix_rows[key] for key in serial_rows)
    batch_mismatches = sum(serial_rows[key] != batch_rows[key] for key in serial_rows)
    scientific_rows = len(serial_rows)
    result = {
        "status": "SOFTWARE_VALIDATION_ONLY",
        "model": "TINY_RANDOM_GPT2_CONFIG_ONLY",
        "budgets": list(budgets),
        "views": len(views),
        "scientific_rows": scientific_rows,
        "modes": [
            {
                "mode": "serial_reasoning_reference",
                "seconds": serial_seconds,
                "physical_generations": serial_physical,
                "row_token_mismatches": 0,
            },
            {
                "mode": "max_budget_prefix_reuse",
                "seconds": prefix_seconds,
                "physical_generations": prefix_physical,
                "row_token_mismatches": prefix_mismatches,
                "speedup_vs_serial": serial_seconds / prefix_seconds,
            },
            {
                "mode": "batched_reasoning",
                "seconds": batch_seconds,
                "physical_generations": batch_physical,
                "batch_size": args.batch_size,
                "row_token_mismatches": batch_mismatches,
                "speedup_vs_serial": serial_seconds / batch_seconds,
            },
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
