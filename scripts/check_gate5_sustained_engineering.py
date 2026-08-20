#!/usr/bin/env python3
"""Real-Qwen, non-scientific Gate-5 sustained-hook engineering checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate5 import ALPHA, LAYER  # noqa: E402
from epistemic_geometry.types import Intervention  # noqa: E402
from scripts.run_gate5_source_duration import (  # noqa: E402
    _benchmark_item,
    _build_backend,
    _intervention,
    _load_items,
    _load_vectors,
)

MAX_NEW_TOKENS = 32
TOLERANCE = 1e-4


def _token_digest(output: Any) -> str:
    tokens = output.metadata.get("generated_token_ids", [])
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _run(
    backend: Any,
    item: Any,
    seed: int,
    *,
    duration: str,
    intervention: Intervention | None,
) -> tuple[Any, dict[str, Any] | None]:
    if duration == "sustained":
        context = backend.steer_sustained_current_token(intervention)
    elif duration == "one_shot":
        context = backend.steer_prefill_once(intervention)
    else:
        from contextlib import nullcontext

        context = nullcontext()
    with context as trace:
        output = backend.generate_reasoning(
            item,
            sampling_seed=seed,
            max_new_tokens=MAX_NEW_TOKENS,
            intervention_metadata={
                "intervention": intervention.vector_id if intervention else "none",
                "intervention_duration": duration,
                "intervention_layer": intervention.layer if intervention else None,
                "intervention_alpha": intervention.alpha if intervention else 0.0,
                "intervention_vector_hash": intervention.vector.hash if intervention else None,
            },
        )
    return output, trace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--gate5-dir", type=Path, required=True)
    parser.add_argument("--gate4-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = _load_items(args.validation_manifest)[:5]
    if len(items) != 5:
        raise ValueError("Gate-5 engineering check requires five validation items")
    vectors = _load_vectors(args.gate5_dir, args.gate4_dir)
    backend = _build_backend(args.model_path)

    records: list[dict[str, Any]] = []
    identity_pass = True
    cleanup_pass = True
    exact_shift_pass = True
    scope_pass = True
    forward_count_pass = True
    cache_safety_pass = True
    for index, external_item in enumerate(items):
        item = _benchmark_item(external_item)
        seed = 7_000_000 + index
        baseline_before, _ = _run(backend, item, seed, duration="none", intervention=None)
        zero = _intervention(vectors["v_delib"], 0.0, "v_delib_alpha_zero")
        zero_output, zero_trace = _run(backend, item, seed, duration="one_shot", intervention=zero)
        identity_equal = baseline_before.metadata.get(
            "generated_token_ids"
        ) == zero_output.metadata.get("generated_token_ids")
        identity_pass &= identity_equal

        traces: dict[str, dict[str, Any] | None] = {}
        for direction_name, sign in (("v_delib_plus", 1.0), ("v_delib_minus", -1.0), ("R0", 1.0)):
            vector_name = "v_delib" if direction_name.startswith("v_delib") else "R0"
            intervention = _intervention(vectors[vector_name], sign * ALPHA, direction_name)
            output, trace = _run(
                backend, item, seed + 100, duration="sustained", intervention=intervention
            )
            traces[direction_name] = trace
            if trace is None:
                exact_shift_pass = False
                scope_pass = False
                forward_count_pass = False
                cache_safety_pass = False
                continue
            exact_shift_pass &= trace["max_abs_shift_error"] <= TOLERANCE
            scope_pass &= trace["max_abs_non_current_change"] <= TOLERANCE
            forward_count_pass &= (
                trace["forward_count"] == len(trace["applications"])
                and trace["forward_count"]
                == trace["prefill_applications"] + trace["decode_applications"]
            )
            cache_safety_pass &= all(
                application["sequence_length"] == 1
                or application["token_position"] == application["sequence_length"] - 1
                for application in trace["applications"]
            )

        baseline_after, _ = _run(backend, item, seed, duration="none", intervention=None)
        cleanup_equal = baseline_before.metadata.get(
            "generated_token_ids"
        ) == baseline_after.metadata.get("generated_token_ids")
        cleanup_pass &= cleanup_equal
        records.append(
            {
                "item_id": external_item.item_id,
                "seed": seed,
                "alpha_zero_token_identity": identity_equal,
                "baseline_cleanup_token_identity": cleanup_equal,
                "baseline_token_digest": _token_digest(baseline_before),
                "alpha_zero_token_digest": _token_digest(zero_output),
                "traces": traces,
                "zero_trace": zero_trace,
            }
        )

    checks = {
        "alpha_zero_identity": identity_pass,
        "exact_additive_shift_at_every_forward": exact_shift_pass,
        "last_prompt_token_scope": scope_pass,
        "one_application_per_forward": forward_count_pass,
        "cache_safety": cache_safety_pass,
        "hook_cleanup": cleanup_pass,
        "condition_metadata": all(
            record["traces"]["v_delib_plus"] is not None for record in records
        ),
        "tolerance": TOLERANCE,
        "items": [item.item_id for item in items],
        "max_new_tokens_engineering_only": MAX_NEW_TOKENS,
        "direction_layer": LAYER,
        "alpha": ALPHA,
        "scientific_outcomes_collected": False,
    }
    checks["pass"] = all(
        checks[name]
        for name in (
            "alpha_zero_identity",
            "exact_additive_shift_at_every_forward",
            "last_prompt_token_scope",
            "one_application_per_forward",
            "cache_safety",
            "hook_cleanup",
            "condition_metadata",
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not checks["pass"]:
        raise SystemExit("GATE5_SUSTAINED_ENGINE_FAILURE")
    print(json.dumps({"pass": True, "items": len(items)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
