#!/usr/bin/env python3
"""Prepare and prospectively lock Gate 12.1 engineering qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate12_1  # noqa: E402

REVIEW = ROOT / "review/gate12_1_continuous_geometry_engine"
SPEC = ROOT / "experiments/specs/gate12_1_continuous_geometry_engine.yaml"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def design() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    fixtures = gate12_1.engineering_fixtures()
    fixture_payload = {
        "namespace": gate12_1.FIXTURE_NAMESPACE,
        "classification": "SYNTHETIC_NON_SCIENTIFIC_ENGINEERING_ONLY",
        "scientific_item_count": 0,
        "historical_outcomes_available_to_runner": False,
        "fixtures": fixtures,
    }
    write_json(REVIEW / "ENGINEERING_FIXTURES.json", fixture_payload)
    directions = gate12_1.engineering_directions()
    np.savez_compressed(
        REVIEW / "ENGINEERING_DIRECTIONS.npz",
        directions=directions,
        seeds=np.asarray(gate12_1.ENGINE_DIRECTION_SEEDS, dtype=np.int64),
    )
    direction_records = []
    for index, vector in enumerate(directions):
        direction_records.append(
            {
                "index": index,
                "seed": gate12_1.ENGINE_DIRECTION_SEEDS[index],
                "norm": float(np.linalg.norm(vector)),
                "float64_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
            }
        )
    write_json(
        REVIEW / "ENGINE_MATRIX.json",
        {
            "cells": {
                "E0": ["BF16", "historical/default SDPA", "sequential KV", "none"],
                "E1": ["BF16", "SDPA math", "sequential KV", "none"],
                "E2": ["BF16", "SDPA math", "full sequence", "none"],
                "E3": [
                    "FP32 lift",
                    "eager preferred; SDPA-math frozen fallback",
                    "sequential KV",
                    "none",
                ],
                "E4": ["FP32 lift", "same as E3", "full sequence", "exact AD"],
            },
            "alphas": ["NO_HOOK", *gate12_1.ALL_ALPHAS],
            "epsilon_ladder": list(gate12_1.EPSILONS),
            "directions": direction_records,
            "eager_fallback_rule": (
                "Before qualification metrics, use FP32 SDPA math iff an eager shape/forward "
                "availability probe fails; persist the reason."
            ),
        },
    )
    premortem = {
        "classification": "PREMORTEM_PASS",
        "checks": {
            "semantic_vs_numerical_mismatch": (
                "PASS: indexing/masks/cache fields captured separately"
            ),
            "bf16_differentiability": (
                "PASS: formal AD is distinct from rounded-BF16 finite differences"
            ),
            "fp32_lift": (
                "PASS: load BF16 once, hash, cast those values in place, hash and round-trip"
            ),
            "kernel_comparability": (
                "PASS: dtype, kernel, and cache cells vary one factor at a time"
            ),
            "finite_difference_scales": "PASS: fixed absolute FP32 ladder 1e-4 through 1e-1",
            "independent_identities": "PASS: forward JVP, independent JVP, VJP, Hessian, and KL",
            "claim_boundary": "PASS: no predictive Fisher/pullback claim is authorized",
        },
        "scientific_items": 0,
        "historical_outcome_access": False,
        "free_generation": False,
    }
    write_json(REVIEW / "PREMORTEM.json", premortem)
    (REVIEW / "PREMORTEM.md").write_text(
        "# Gate 12.1 adversarial premortem\n\n"
        "Classification: `PREMORTEM_PASS`.\n\n"
        "The qualification isolates sequence indexing, hook registration, cache mode, "
        "attention kernel, and dtype. It treats automatic differentiation of the FP32 "
        "computational lift as distinct from finite differences of rounded BF16 execution. "
        "The epsilon ladder is absolute and local, and JVP/VJP, Fisher/Hessian, utility, "
        "and KL identities are independently checked.\n\n"
        "Only synthetic token fixtures and frozen engineering random directions are visible "
        "to the runner. No benchmark manifest, semantic outcome, scientific item, free "
        "generation, Q2 analysis, or holdout access is permitted. A passing engine would "
        "qualify only the local directional geometry of the FP32 computational lift of the "
        "frozen BF16-valued parameters.\n",
        encoding="utf-8",
    )
    SPEC.parent.mkdir(parents=True, exist_ok=True)
    SPEC.write_text(
        "id: GATE12_1_CONTINUOUS_GEOMETRY_ENGINE\n"
        "status: PROSPECTIVE_ENGINEERING_LOCK\n"
        "stage: ENGINEERING_NUMERICAL_QUALIFICATION\n"
        "model: Qwen/Qwen3-8B\n"
        f"revision: {gate12_1.MODEL_REVISION}\n"
        "layer: 27\n"
        "scientific_items: 0\n"
        "historical_outcomes: FORBIDDEN\n"
        "free_generation: FORBIDDEN\n"
        "q2: NOT_RUN\n"
        "holdout: UNTOUCHED\n"
        "target_cost_usd: 0.50\n"
        "hard_ceiling_usd: 1.50\n",
        encoding="utf-8",
    )


def lock(source_commit: str) -> None:
    if git_commit() != source_commit:
        raise RuntimeError("lock must be created at the exact implementation source commit")
    fixtures = REVIEW / "ENGINEERING_FIXTURES.json"
    directions = REVIEW / "ENGINEERING_DIRECTIONS.npz"
    matrix = REVIEW / "ENGINE_MATRIX.json"
    runner = ROOT / "scripts/run_gate12_1_continuous_geometry_engine.py"
    runner_text = runner.read_text(encoding="utf-8").lower()
    forbidden = (
        "gate9_selected",
        "gate10_cross",
        "gate11_domain",
        "control_validation_items",
        "utility_prediction_items",
        "journal.jsonl",
    )
    present = [value for value in forbidden if value in runner_text]
    if present:
        raise RuntimeError(f"scientific firewall violation in runner: {present}")
    payload = {
        "status": "FROZEN_PRE_QUALIFICATION",
        "experiment_source_commit": source_commit,
        "model": gate12_1.MODEL,
        "model_revision": gate12_1.MODEL_REVISION,
        "parameter_rule": "load exact BF16 checkpoint values once; cast those values to FP32",
        "layer": gate12_1.LAYER,
        "fixtures_sha256": sha256(fixtures),
        "directions_sha256": sha256(directions),
        "engine_matrix_sha256": sha256(matrix),
        "runner_sha256": sha256(runner),
        "fixture_count": 12,
        "scientific_item_count": 0,
        "historical_outcomes_revealed": False,
        "alphas": [0.0, *gate12_1.SMALL_ALPHAS, gate12_1.D75_ALPHA],
        "epsilons": list(gate12_1.EPSILONS),
        "fp32_sequence_thresholds": {
            "top1": 1.0,
            "median_js": 1e-8,
            "p99_js": 1e-6,
            "median_target_logp_abs": 1e-5,
            "max_target_logp_abs": 1e-3,
            "median_logit_cosine": 0.999999,
        },
        "bf16_bridge_thresholds": {
            "top1": 0.99,
            "median_js": 1e-4,
            "p95_js": 5e-3,
            "median_target_logp_abs": 0.02,
        },
        "derivative_thresholds": {
            "jvp_cosine": 0.99999,
            "jvp_relative_norm": 0.005,
            "vjp_duality_relative": 1e-4,
            "fisher_hessian_relative": 0.01,
            "utility_autograd_relative": 0.01,
            "finite_window_length": 3,
            "finite_jvp_cosine": 0.999,
            "finite_relative": 0.05,
        },
        "qualified_object": (
            "local directional geometry of the FP32 computational lift of the frozen "
            "BF16-valued Qwen3-8B parameters"
        ),
        "cost": {"target_usd": 0.5, "hard_ceiling_usd": 1.5},
        "classifications": list(gate12_1.CLASSIFICATIONS),
        "scientific_firewall": {
            "free_generation": False,
            "semantic_evaluation": False,
            "historical_outcomes": False,
            "scientific_gate12_collection": False,
            "q2": "NOT_RUN",
            "holdout": "UNTOUCHED",
        },
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", payload)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Gate 12.1 prospective lock\n\n"
        f"Source commit: `{source_commit}`. Status: `FROZEN_PRE_QUALIFICATION`.\n\n"
        "This lock authorizes numerical engineering on twelve synthetic token fixtures only. "
        "It authorizes no scientific geometry collection or historical-outcome reveal.\n",
        encoding="utf-8",
    )
    pre = {}
    for path in sorted(REVIEW.iterdir()):
        if path.is_file() and path.name != "artifact_hashes_prequalification.json":
            pre[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        REVIEW / "artifact_hashes_prequalification.json",
        {"experiment_source_commit": source_commit, "artifacts": pre},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("design", "lock"), required=True)
    parser.add_argument("--experiment-source-commit")
    args = parser.parse_args()
    if args.phase == "design":
        design()
    else:
        if not args.experiment_source_commit:
            parser.error("--experiment-source-commit is required for lock")
        lock(args.experiment_source_commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
