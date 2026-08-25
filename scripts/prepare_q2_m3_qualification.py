#!/usr/bin/env python3
"""Prepare and freeze the outcome-free Q2 M3 numerical qualification."""

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

from epistemic_geometry.analysis import m3_qualification as m3  # noqa: E402

REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def historical_audit() -> None:
    gate12 = json.loads(
        (ROOT / "review/gate12_utility_aligned_pullback/NUMERICAL_VALIDATION.json").read_text()
    )
    gate12_1 = json.loads(
        (ROOT / "review/gate12_1_continuous_geometry_engine/ENGINE_QUALIFICATION.json").read_text()
    )
    finite = json.loads(
        (
            ROOT / "review/gate12_1_continuous_geometry_engine/FINITE_DIFFERENCE_WINDOW.json"
        ).read_text()
    )
    bridge = json.loads(
        (ROOT / "review/gate12_1_continuous_geometry_engine/BF16_FP32_BRIDGE.json").read_text()
    )
    payload = {
        "gate12_classification": gate12.get("classification", "GATE12_JVP_ENGINE_FAILURE"),
        "gate12_scientific_geometry_shards": 0,
        "gate12_historical_outcomes_revealed": False,
        "gate12_1_classification": gate12_1["classification"],
        "fp32_sequence_pass": gate12_1["fp32_sequence_pass"],
        "exact_jvp_pass": gate12_1["exact_jvp_pass"],
        "jvp_vjp_pass": gate12_1["jvp_vjp_pass"],
        "fisher_hessian_pass": gate12_1["fisher_hessian_pass"],
        "utility_derivative_pass": gate12_1["utility_derivative_pass"],
        "bf16_bridge_pass": gate12_1["bf16_bridge_pass"],
        "bf16_bridge_top1": bridge["top1_agreement"],
        "finite_window_pass": gate12_1["finite_difference_window_pass"],
        "historical_passing_epsilons": [0.03, 0.1],
        "historical_required_window_length": 3,
        "historical_window": finite["three_consecutive_window"],
        "historical_interpretation": (
            "Exact FP32 automatic-differentiation identities and sequence semantics passed. "
            "The BF16 behavioral bridge and the prespecified three-scale finite-difference "
            "window failed. This is engineering evidence, not evidence about predictive geometry."
        ),
    }
    write_json(REVIEW / "GATE12_HISTORY_AUDIT.json", payload)
    (REVIEW / "GATE12_HISTORY_AUDIT.md").write_text(
        "# Gate 12 / 12.1 derivative-engine audit\n\n"
        "Gate 12 stopped before scientific geometry and before historical outcomes. Gate 12.1 "
        "used the FP32 computational lift of the exact BF16-valued checkpoint. FP32 full-sequence "
        "versus KV semantics, forward/independent JVP, JVP/VJP duality, Fisher/Hessian, "
        "and utility derivative identities passed. The historical BF16 bridge missed top-1 "
        "(0.977444 versus 0.99), and only epsilon 0.03 and 0.1 passed consecutively where "
        "three scales were required. "
        "The mismatch was attributed to mixed BF16 kernel/cache/reduction-order and dtype effects, "
        "not an off-by-one sequence bug.\n\n"
        "M3 therefore receives a new prospective qualification tailored to a teacher-forced "
        "controller-span Gram; no historical failure is reinterpreted as scientific evidence.\n",
        encoding="utf-8",
    )


def design() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    historical_audit()
    fixtures = m3.engineering_fixtures()
    directions = m3.engineering_directions()
    fixture_payload = {
        "schema_version": 1,
        "namespace": m3.FIXTURE_NAMESPACE,
        "fixture_count": len(fixtures),
        "scientific_items": 0,
        "semantic_outcomes": 0,
        "fixtures": fixtures,
    }
    write_json(REVIEW / "M3_ENGINEERING_FIXTURES.json", fixture_payload)
    np.savez_compressed(REVIEW / "M3_ENGINEERING_DIRECTIONS.npz", directions=directions)
    direction_payload = {
        "direction_count": len(directions),
        "hidden_size": directions.shape[1],
        "engineering_only": True,
        "norms": np.linalg.norm(directions, axis=1).tolist(),
        "maximum_off_diagonal_absolute_cosine": float(
            np.max(np.abs(directions @ directions.T - np.eye(len(directions))))
        ),
        "npz_sha256": sha256(REVIEW / "M3_ENGINEERING_DIRECTIONS.npz"),
    }
    write_json(REVIEW / "M3_ENGINEERING_DIRECTIONS.json", direction_payload)

    protocol = {
        "schema_version": 1,
        "status": "PROSPECTIVE_DRAFT_NOT_YET_COMMITTED",
        "model": m3.MODEL,
        "model_revision": m3.MODEL_REVISION,
        "layer": m3.LAYER,
        "object_name": "teacher-forced multi-checkpoint categorical-Fisher controller-span Gram",
        "literal_estimand": (
            "Gamma_ij is the uniform mean over frozen synthetic fixture/checkpoint rows of "
            "(J_z v_i)^T (diag(p)-p p^T) (J_z v_j), conditioned on one fixed teacher-forced "
            "token sequence per fixture in the FP32 computational lift of BF16-valued parameters."
        ),
        "sequence_context": {
            "execution": "FP32 full-sequence use_cache=False",
            "attention": (
                "eager, with prospectively declared SDPA-math fallback only on unavailability"
            ),
            "trajectory": "fixed arbitrary synthetic continuation; no generation",
            "checkpoints": (
                "final prompt token and frozen continuation offsets available per fixture"
            ),
            "checkpoint_offsets": list(m3.CHECKPOINT_OFFSETS),
            "averaging": "uniform over all persisted fixture/checkpoint rows",
            "claim_boundary": (
                "local output-information geometry along a fixed teacher-forced technical "
                "trajectory; "
                "not Fisher geometry of the full free-running sequence distribution"
            ),
        },
        "parameter_realization": (
            "load exact BF16 checkpoint values once and cast those already-rounded values to FP32; "
            "do not load a different FP32 checkpoint"
        ),
        "fixture_count": m3.FIXTURE_COUNT,
        "engineering_direction_count": m3.DIRECTION_COUNT,
        "qualification_cases": {
            "exact_gram_fixture_indices": list(range(m3.FIXTURE_COUNT)),
            "sequence_equivalence_fixture_indices": list(range(m3.FIXTURE_COUNT)),
            "bf16_geometry_bridge_fixture_indices": list(m3.BRIDGE_FIXTURE_INDICES),
            "finite_difference_fixture_indices": list(m3.DIFFERENTIAL_FIXTURE_INDICES),
            "polarization_fixture_indices": list(m3.POLARIZATION_FIXTURE_INDICES),
            "independent_exact_crosscheck_cases": [
                list(value) for value in m3.EXACT_CROSSCHECK_CASES
            ],
            "batch_order_sensitivity": (
                "repeat direction passes, reverse direction order, and compare single-pass "
                "versus two-chunk sufficient-statistic aggregation"
            ),
        },
        "epsilons": list(m3.EPSILONS),
        "local_epsilon_max": m3.LOCAL_EPSILON_MAX,
        "bf16_bridge_epsilon": m3.BF16_BRIDGE_EPSILON,
        "thresholds": m3.THRESHOLDS,
        "threshold_rationale": {
            "exact_derivatives": "retain Gate-12.1 tolerances that passed independently",
            "finite_window": (
                "requires three consecutive scales bounded by epsilon<=1; expanded ladder is fixed "
                "from historical cancellation evidence, not from new qualification values"
            ),
            "bf16_bridge": (
                "retains the prior top-1/JS bridge and adds rank preservation of controller-span "
                "radii/distances; failure excludes M3 from Q2 V3"
            ),
            "psd": "permits only roundoff-scale negative eigenvalues; no PSD clipping is allowed",
        },
        "classifications": list(m3.CLASSIFICATIONS),
        "cost": {"target_usd": 0.75, "hard_incremental_ceiling_usd": 2.0},
        "firewall": {
            "scientific_items": 0,
            "semantic_correctness": False,
            "free_generation": False,
            "q2_v3_behavioral_outcomes": False,
            "q3": "NOT_RUN",
        },
    }
    write_json(REVIEW / "M3_QUALIFICATION_PROTOCOL.json", protocol)
    (REVIEW / "M3_QUALIFICATION_PROTOCOL.md").write_text(
        "# Prospective M3 numerical qualification\n\n"
        "Status: `DRAFT — MUST BE COMMITTED BEFORE REAL-MODEL MEASUREMENT`.\n\n"
        "M3 is the teacher-forced, multi-checkpoint categorical-Fisher Gram restricted to six "
        "engineering-only directions at Qwen block 27. Each row conditions on a frozen arbitrary "
        "token prefix and averages uniformly over final-prompt and prescribed continuation "
        "checkpoints. It is the local output-information geometry of the FP32 computational lift "
        "of the frozen BF16-valued parameters, not a semantic-error metric and not the Fisher "
        "geometry of the full free-running trajectory distribution.\n\n"
        "The exact thresholds, epsilon ladder, BF16 bridge, sequence checks, PSD rule, direct/"
        "polarization comparison, cost ceiling, and classification vocabulary are machine-frozen "
        "in `M3_QUALIFICATION_PROTOCOL.json`. No clipping of an indefinite Gram is allowed. "
        "Failure "
        "of the BF16 bridge excludes M3 from Q2 V3 even when exact FP32 identities pass.\n",
        encoding="utf-8",
    )


def lock(experiment_source_commit: str) -> None:
    protocol_path = REVIEW / "M3_QUALIFICATION_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["status"] = "FROZEN_PRE_REAL_MODEL_QUALIFICATION"
    protocol["experiment_source_commit"] = experiment_source_commit
    protocol["protocol_lock_commit_parent"] = git_head()
    protocol["fixtures_sha256"] = sha256(REVIEW / "M3_ENGINEERING_FIXTURES.json")
    protocol["directions_npz_sha256"] = sha256(REVIEW / "M3_ENGINEERING_DIRECTIONS.npz")
    write_json(protocol_path, protocol)
    lock_payload = {
        "status": "FROZEN_PRE_REAL_MODEL_QUALIFICATION",
        "experiment_source_commit": experiment_source_commit,
        "protocol_sha256": sha256(protocol_path),
        "fixtures_sha256": protocol["fixtures_sha256"],
        "directions_npz_sha256": protocol["directions_npz_sha256"],
        "real_model_measurements_at_lock": 0,
        "q2_v3_semantic_trajectories": 0,
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock_payload)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Q2 M3 qualification prospective lock\n\n"
        f"Experiment source commit: `{experiment_source_commit}`.\n\n"
        "The exact M3 object, fixtures, directions, sequence checkpoints, numerical thresholds, "
        "classification rule, BF16 bridge, and US$2 hard ceiling were frozen before any new real-"
        "model qualification measurement. Q2 V3 behavioral execution remains forbidden.\n",
        encoding="utf-8",
    )
    hashes = {}
    for path in sorted(REVIEW.iterdir()):
        if path.is_file() and path.name != "artifact_hashes_prequalification.json":
            hashes[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    write_json(
        REVIEW / "artifact_hashes_prequalification.json",
        {"experiment_source_commit": experiment_source_commit, "artifacts": hashes},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("design", "lock"))
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
