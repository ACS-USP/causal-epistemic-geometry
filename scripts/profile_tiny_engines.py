#!/usr/bin/env python3
"""Profile local execution engines on a random tiny transformer only.

This creates engineering metadata, not scientific data or a model result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from epistemic_geometry.backends.tiny import TinyRandomTransformerBackend
from epistemic_geometry.config import BackendConfig
from epistemic_geometry.inference.performance import ModeMeasurement, compare_modes
from epistemic_geometry.types import BenchmarkItem, Intervention, SteeringVector


def _items() -> list[BenchmarkItem]:
    return [
        BenchmarkItem(
            id=f"tiny-profile-{index}",
            prompt="Choose one answer " + "alpha " * (index + 1),
            target="A",
            metadata={"candidate_labels": ["A", "B", "C", "D"]},
        )
        for index in range(4)
    ]


def _backend(mode: str) -> TinyRandomTransformerBackend:
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
            inference_mode="choice_loglikelihood",
            execution_mode=mode,
            candidate_head_mode="full_vocab_reference",
            item_batch_size=4,
            condition_chunk_size=2,
            max_prefill_tokens=512,
            padding_side="left",
            attention_implementation="eager",
        ),
        seed=19,
    )


def main() -> None:
    items = _items()
    vector = SteeringVector(
        values=np.linspace(-0.2, 0.2, 32),
        layer=0,
        constructor="profile_fixture",
        normalization="none",
        hash="profile-fixture",
    )
    conditions = [
        ({"condition": "baseline", "alpha": 0.0, "layer": 0}, None),
        ({"condition": "steered", "alpha": 0.4, "layer": 0}, vector),
    ]
    rows = []
    prompt_lengths = []
    for mode in ("serial_reference", "full_prompt_batched", "cached_decode"):
        backend = _backend(mode)
        backend.reset_execution_stats()
        started = time.perf_counter()
        if mode == "serial_reference":
            for item in items:
                backend.predict(item)
                with backend.steer(Intervention(0, 0.4, vector.hash, "last_token", vector)):
                    backend.predict(item)
        else:
            prepared = backend.prepare_choice_items(items)
            prompt_lengths = [item.prompt_length for item in prepared]
            backend.predict_choice_batch(prepared, conditions)
        elapsed = time.perf_counter() - started
        rows.append(
            ModeMeasurement(
                mode=mode,
                seconds=elapsed,
                item_conditions=len(items) * len(conditions),
                prompt_tokens=sum(item.prompt.count("alpha") + 3 for item in items),
            )
        )
        rows[-1] = {
            **rows[-1].__dict__,
            "execution_stats": backend.execution_stats(),
            "attention_implementation": backend.provenance().get("attention_implementation"),
        }
    measurements = [
        ModeMeasurement(row["mode"], row["seconds"], row["item_conditions"]) for row in rows
    ]
    output = {
        "status": "TINY_RANDOM_TRANSFORMER_ENGINEERING_ONLY",
        "model": "random GPT2 config, no downloaded weights",
        "items": [item.id for item in items],
        "candidate_labels": ["A", "B", "C", "D"],
        "prompt_length_summary": {
            "mean": float(np.mean(prompt_lengths)),
            "p50": float(np.percentile(prompt_lengths, 50)),
            "p90": float(np.percentile(prompt_lengths, 90)),
            "p99": float(np.percentile(prompt_lengths, 99)),
        },
        "measurements": rows,
        "cost_summaries": compare_modes(measurements),
    }
    path = Path("benchmarks/serial_reference_profile.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
