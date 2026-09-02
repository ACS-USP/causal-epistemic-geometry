#!/usr/bin/env python3
"""Freeze the Q2 OOS V2 semantic schedule and prediction-lock package.

This script is CPU-only and may run only after the selected-bank and label-free
qualifications pass.  It creates no model output and deliberately leaves future
semantic execution unauthorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
V2_STREAM = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
V41 = ROOT / "review/q2_v4_1_prediction_lock"
SELECTED = REVIEW / "V2_SELECTED_CONTROLLER_BANK.json"
LABEL_FREE = REVIEW / "LABEL_FREE_QUALIFICATION.json"
MATRIX_METADATA = REVIEW / "PREDICTION_MATRIX_METADATA.json"
MATRICES = REVIEW / "PREDICTION_MATRICES.npz"
EFFICIENCY = V2_STREAM / "Q2_OOS_V2_AMENDED_SEMANTIC_EXECUTION_LOCK.json"
PROTOCOL = V2_STREAM / "V2_FINAL_PROTOCOL_LOCK.json"
STREAM_MANIFEST = V2_STREAM / "V2_CANDIDATE_BANK_MANIFEST.json"
V41_SCHEDULE = V41 / "FUTURE_SEMANTIC_SCHEDULE.json"
V41_NORMATIVE = V41 / "Q2_V4_1_NORMATIVE_EXECUTION_AND_ANALYSIS_LOCK.json"
SHELLS = ("MEDIUM", "STRONG")
EXPECTED_ROWS = 16 * 2 * 300 * 2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_seed64(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def semantic_item_rollouts() -> list[dict[str, Any]]:
    schedule = read_json(V41_SCHEDULE)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in schedule["rows"]:
        key = (str(row["item_id"]), int(row["rollout_index"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "item_id": key[0],
                "rollout_index": key[1],
                "prompt_sha256": str(row["prompt_sha256"]),
                "reference_type": str(row["reference_type"]),
            }
        )
    if len(rows) != 600 or len({row["item_id"] for row in rows}) != 300:
        raise RuntimeError("Q2_OOS_V2_SEMANTIC_PANEL_IDENTITY_FAILURE")
    return rows


def build_schedule() -> dict[str, Any]:
    selected = read_json(SELECTED)
    names = [str(value) for value in selected["selected_ids"]]
    conditions = [f"{name}_{shell}" for name in names for shell in SHELLS]
    prelock_commit = read_json(STREAM_MANIFEST)["prelock_commit"]
    namespace = "Q2-OOS-V2-FRESH-CONTROLLER-SEMANTIC-V1"
    rows: list[dict[str, Any]] = []
    for pair in semantic_item_rollouts():
        order_seed = stable_seed64(
            namespace,
            "CONDITION_ORDER",
            prelock_commit,
            pair["item_id"],
            pair["rollout_index"],
        )
        generator = np.random.Generator(np.random.PCG64DXSM(order_seed))
        for condition_order, condition in enumerate(generator.permutation(conditions).tolist()):
            candidate, shell = str(condition).rsplit("_", 1)
            deployment = selected["controllers"][condition]
            rows.append(
                {
                    **pair,
                    "condition": condition,
                    "condition_order": condition_order,
                    "candidate_id": candidate,
                    "shell": shell,
                    "alpha": float(deployment["alpha"]),
                    "controller_vector_hash": str(deployment["vector_hash"]),
                    "layer": 27,
                    "duration": "sustained_current_token",
                    "seed": stable_seed64(
                        namespace,
                        "GENERATION_SEED",
                        prelock_commit,
                        candidate,
                        shell,
                        pair["item_id"],
                        pair["rollout_index"],
                    ),
                }
            )
    keys = {
        (row["item_id"], row["condition"], row["rollout_index"])
        for row in rows
    }
    seeds = {row["seed"] for row in rows}
    if len(rows) != EXPECTED_ROWS or len(keys) != EXPECTED_ROWS or len(seeds) != EXPECTED_ROWS:
        raise RuntimeError("Q2_OOS_V2_SEMANTIC_SCHEDULE_INTEGRITY_FAILURE")
    return {
        "schema_version": "q2-oos-v2-future-semantic-schedule-v1",
        "status": "FROZEN_NOT_AUTHORIZED_NOT_RUN",
        "namespace": namespace,
        "selected_controller_order": names,
        "shells": list(SHELLS),
        "item_count": 300,
        "rollouts": 2,
        "condition_count": 32,
        "row_count": len(rows),
        "unique_logical_keys": len(keys),
        "unique_seeds": len(seeds),
        "balance": "item-rollout blocks with deterministic PCG64DXSM condition permutation",
        "semantic_outcomes": 0,
        "rows": rows,
    }


def freeze(output_dir: Path) -> None:
    label_free = read_json(LABEL_FREE)
    if label_free["classification"] != "Q2_OOS_V2_LABEL_FREE_INSTRUMENT_QUALIFIED":
        raise RuntimeError("Q2_OOS_V2_LABEL_FREE_INSTRUMENT_NOT_QUALIFIED")
    selected = read_json(SELECTED)
    if selected["classification"] != "Q2_OOS_V2_SELECTED_BANK_GATE_PASS":
        raise RuntimeError("Q2_OOS_V2_SELECTED_BANK_NOT_QUALIFIED")
    efficiency = read_json(EFFICIENCY)
    schedule = build_schedule()
    schedule_path = output_dir / "FUTURE_SEMANTIC_SCHEDULE.json"
    atomic_json(schedule_path, schedule)
    inference = {
        "schema_version": "q2-oos-v2-inference-lock-v1",
        "status": "FROZEN_NOT_RUN",
        "scientific_unit": "one prospectively sampled safety-conditioned fresh controller",
        "reference_atlas": "fixed historical 31-controller V4.1 bank in frozen order",
        "fresh_old_primary": {
            "row_statistic": (
                "r_i=0.5*(Spearman_j(A0_MEDIUM(i,j),Dshape_MEDIUM(i,j))"
                "+Spearman_j(A0_STRONG(i,j),Dshape_STRONG(i,j)))"
            ),
            "bernoulli": "X_i=1 iff finite r_i>0; X_i=0 iff finite r_i<=0",
            "all_16_finite_required": True,
            "positive_count_required": 12,
            "test": "exact one-sided Binomial upper-tail",
            "null": "P(r_i>0)<=0.5",
            "alpha": 0.05,
            "global_fresh_old_rho": "DESCRIPTIVE_ONLY",
            "original_row_QAP": "DIAGNOSTIC_ONLY",
            "studentized_controller_mean": "SENSITIVITY_ONLY",
        },
        "fresh_fresh_secondary": {
            "method": "NODE_JACKKNIFE_PSEUDOVALUE_T",
            "role": "SECONDARY_ONLY_CANNOT_RESCUE_PRIMARY",
        },
        "item_bootstrap": {
            "unit": "semantic panel item",
            "resamples": 50000,
            "conditions_and_rollouts_coupled_within_item": True,
            "seed": stable_seed64(
                "Q2-OOS-V2",
                "ITEM-BOOTSTRAP",
                read_json(STREAM_MANIFEST)["prelock_commit"],
            ),
            "role": "UNCERTAINTY_AND_SENSITIVITY_NOT_PRIMARY_SIGN_TEST_REPLACEMENT",
        },
        "LOFO": {
            "unit": "fresh controller",
            "all_16_omissions": True,
            "role": "SENSITIVITY_ONLY",
        },
        "terminal_precedence": [
            "EXECUTION_INCOMPLETE",
            "INFERENCE_DEGENERATE",
            "PRIMARY_FRESH_OLD_SIGN_TEST",
            "SECONDARY_FRESH_FRESH_DESCRIPTIVE_OR_SUPPORT",
        ],
        "semantic_outcomes": 0,
    }
    inference_path = output_dir / "INFERENCE_LOCK.json"
    atomic_json(inference_path, inference)
    runtime = {
        "schema_version": "q2-oos-v2-runtime-monitor-lock-v1",
        "status": "FROZEN_NOT_RUN",
        "monitoring_only": True,
        "cannot_change_science": True,
        "checkpoints": ["P50", "P80", "P95"],
        "normal_seconds": efficiency["future_semantic_execution"].get(
            "runtime_forecast_seconds",
            read_json(V2_STREAM / "Q2_OOS_V2_SEMANTIC_EFFICIENCY_AMENDMENT.json")[
                "runtime_counterfactual"
            ]["selected_future_19200_seconds"],
        ),
        "stress_1_5x_seconds": read_json(
            V2_STREAM / "Q2_OOS_V2_SEMANTIC_EFFICIENCY_AMENDMENT.json"
        )["runtime_counterfactual"]["stress_1_5x_future_19200_seconds"],
        "stress_2x_seconds": read_json(
            V2_STREAM / "Q2_OOS_V2_SEMANTIC_EFFICIENCY_AMENDMENT.json"
        )["runtime_counterfactual"]["stress_2x_future_19200_seconds"],
    }
    runtime_path = output_dir / "RUNTIME_MONITOR_LOCK.json"
    atomic_json(runtime_path, runtime)
    matrix_metadata = read_json(MATRIX_METADATA)
    normative = read_json(V41_NORMATIVE)
    prediction = {
        "schema_version": "q2-oos-v2-prediction-lock-v1",
        "status": "Q2_OOS_V2_READY_FOR_PREDICTION_LOCK",
        "semantic_execution_authorized": False,
        "semantic_trajectories": 0,
        "correctness_inspected": False,
        "selected_controller_bank_sha256": sha256_file(SELECTED),
        "selected_controller_order": selected["selected_ids"],
        "prediction_matrices_sha256": sha256_file(MATRICES),
        "prediction_matrix_hashes": matrix_metadata["matrix_hashes"],
        "label_free_qualification_sha256": sha256_file(LABEL_FREE),
        "semantic_schedule_sha256": sha256_file(schedule_path),
        "inference_lock_sha256": sha256_file(inference_path),
        "runtime_monitor_lock_sha256": sha256_file(runtime_path),
        "efficient_termination_lock_sha256": sha256_file(EFFICIENCY),
        "semantic_generation": {
            "max_new_tokens": 4096,
            "repetition_stop": "EXTREME_MECHANICAL_REPETITION_V1",
            "all_other_parameters": normative["generation_specification"],
        },
        "retry_resume": normative["retry_resume_specification"],
        "semantic_estimands": normative["semantic_estimands"],
        "fresh_old_primary": inference["fresh_old_primary"],
        "fresh_fresh_secondary": inference["fresh_fresh_secondary"],
        "historical_safety_policy": "UNCHANGED_MAX_NEW_TOKENS_4096_NO_REPETITION_STOP",
        "new_controller_streams": 0,
        "redraws": 0,
        "replacements": 0,
        "Spark1": "FUTURE_SEMANTIC_BACKEND_ONLY_NOT_EXECUTED",
        "Spark2": "FORBIDDEN",
        "RunPod": "FORBIDDEN",
        "Q3": "NOT_RUN",
    }
    prediction_path = output_dir / "PREDICTION_LOCK.json"
    atomic_json(prediction_path, prediction)
    artifacts = [
        SELECTED,
        MATRICES,
        MATRIX_METADATA,
        LABEL_FREE,
        schedule_path,
        inference_path,
        runtime_path,
        prediction_path,
    ]
    atomic_json(
        output_dir / "PREDICTION_LOCK_HASHES.json",
        {
            "schema_version": "q2-oos-v2-prediction-lock-hashes-v1",
            "files": {
                str(path.relative_to(ROOT)): sha256_file(path) for path in artifacts
            },
            "semantic_trajectories": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": prediction["status"],
                "schedule_rows": schedule["row_count"],
                "schedule_sha256": sha256_file(schedule_path),
                "prediction_lock_sha256": sha256_file(prediction_path),
                "semantic_trajectories": 0,
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    freeze(args.output_dir)


if __name__ == "__main__":
    main()
