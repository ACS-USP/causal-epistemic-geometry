#!/usr/bin/env python3
"""Materialize the V4 prediction lock without reading semantic outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.q2_v4_presemantic import (  # noqa: E402
    PRIMARY_N,
    QAP_MAPS,
    SELECTED_COUNT,
    semantic_schedule,
    unique_controller_permutations,
    unique_shell_swaps,
)

REVIEW = ROOT / "review/q2_v4_spark1_presemantic"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(str(values.dtype).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def throughput_projection() -> dict[str, Any]:
    engine = read_json(REVIEW / "SPARK1_ENGINE_QUALIFICATION.json")
    source = read_json(REVIEW / "SOURCE_THROUGHPUT.json")
    safety = read_json(REVIEW / "CANDIDATE_SAFETY_REPORT.json")
    phases = []
    engine_seconds = sum(float(row["seconds"]) for row in engine["throughput_fixtures"])
    engine_tokens = sum(int(row["generated_tokens"]) for row in engine["throughput_fixtures"])
    phases.append(
        {
            "phase": "technical_fixtures",
            "rows": len(engine["throughput_fixtures"]),
            "tokens": engine_tokens,
            "seconds": engine_seconds,
        }
    )
    phases.append(
        {
            "phase": "source_qualification",
            "rows": int(source["rows"]),
            "tokens": int(source["new_generated_tokens"]),
            "seconds": float(source["elapsed_seconds"]),
        }
    )
    phases.append(
        {
            "phase": "candidate_safety",
            "rows": 12 * 2 * 81,
            "tokens": int(safety["new_generated_tokens"]),
            "seconds": float(safety["elapsed_seconds"]),
        }
    )
    for row in phases:
        row["tokens_per_second"] = row["tokens"] / max(row["seconds"], 1e-12)
        row["trajectories_per_hour"] = row["rows"] * 3600.0 / max(row["seconds"], 1e-12)
    representative = phases[-1]["trajectories_per_hour"]
    projected_hours = 39_000 / representative
    return {
        "schema_version": "q2-v4-throughput-projection-v1",
        "phases": phases,
        "representative_basis": "candidate safety label-free free-generation phase",
        "projected_semantic_rows": 39_000,
        "projected_wall_hours": projected_hours,
        "tail_margin_fraction": 0.50,
        "projected_wall_hours_with_tail": 1.5 * projected_hours,
        "scientific_settings_optimized_for_speed": False,
        "spark2_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelock-commit", required=True)
    args = parser.parse_args()
    if len(args.prelock_commit) != 40:
        raise RuntimeError("PRELOCK must be a full commit hash")
    prelock = read_json(REVIEW / "PRELOCK.json")
    if prelock["status"] != "Q2_V4_PRELOCK_READY_FOR_COMMIT":
        raise RuntimeError("PRELOCK artifact is not valid")
    required = {
        "CANDIDATE_BANK_MANIFEST.json": "Q2_V4_CANDIDATE_BANK_ALGEBRAIC_PASS",
        "CANDIDATE_SAFETY_REPORT.json": "Q2_V4_SAFE_BANK_QUALIFIED",
        "SELECTED_CONTROLLER_BANK.json": "Q2_V4_BANK_IDENTIFIABILITY_PASS",
        "A1_INSTRUMENT_QUALIFICATION.json": "Q2_V4_A1_INSTRUMENT_QUALIFIED",
        "A2_INSTRUMENT_QUALIFICATION.json": "Q2_V4_A2_INSTRUMENT_QUALIFIED",
    }
    for name, classification in required.items():
        if read_json(REVIEW / name)["classification"] != classification:
            raise RuntimeError(f"prediction lock gate failed: {name}")
    candidate = read_json(REVIEW / "CANDIDATE_BANK_MANIFEST.json")
    selected = read_json(REVIEW / "SELECTED_CONTROLLER_BANK.json")
    if candidate["prelock_commit"] != args.prelock_commit:
        raise RuntimeError("candidate stream was not derived from the supplied PRELOCK")
    if selected["prelock_commit"] != args.prelock_commit:
        raise RuntimeError("selected bank was not derived from the supplied PRELOCK")
    selected_ids = list(selected["selected_ids"])
    if len(selected_ids) != SELECTED_COUNT:
        raise RuntimeError("selected bank is not exactly 32 directions")

    permutations, qap_seed = unique_controller_permutations(args.prelock_commit)
    swaps, swap_seed = unique_shell_swaps(args.prelock_commit)
    np.save(REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy", permutations)
    np.save(REVIEW / "RADIAL_SHELL_SWAPS.npy", swaps)
    write_json(
        REVIEW / "QAP_SCHEDULE.json",
        {
            "prelock_commit": args.prelock_commit,
            "maps": QAP_MAPS,
            "identity_first": bool(np.array_equal(permutations[0], np.arange(SELECTED_COUNT))),
            "unique_maps": len({row.tobytes() for row in permutations}),
            "seed": str(qap_seed),
            "seed_hex_128": f"{qap_seed:032x}",
            "byte_order": "big",
            "rng": "NumPy PCG64DXSM",
            "archive_sha256": sha256(REVIEW / "QAP_CONTROLLER_PERMUTATIONS.npy"),
            "same_permutation_both_shells_all_metrics": True,
            "p_value": "count(T_perm>=T_observed)/50000",
            "multiplicity": "single-step maxT across A0/A1/A2",
        },
    )
    write_json(
        REVIEW / "RADIAL_SWAP_SCHEDULE.json",
        {
            "prelock_commit": args.prelock_commit,
            "maps": QAP_MAPS,
            "identity_first": not bool(swaps[0].any()),
            "unique_maps": len({row.tobytes() for row in swaps}),
            "seed": str(swap_seed),
            "seed_hex_128": f"{swap_seed:032x}",
            "byte_order": "big",
            "rng": "NumPy PCG64DXSM",
            "archive_sha256": sha256(REVIEW / "RADIAL_SHELL_SWAPS.npy"),
            "R_total_and_R_shape_separate": True,
        },
    )

    panel = read_json(REVIEW / "PRIMARY_PANEL_MANIFEST.json")
    if panel["item_count"] != PRIMARY_N or panel["semantic_outcomes"] != 0:
        raise RuntimeError("semantic panel firewall failed")
    schedule = semantic_schedule(panel["item_ids"], selected_ids, args.prelock_commit)
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in schedule}
    conditions = {row["condition"] for row in schedule}
    if len(schedule) != 39_000 or len(keys) != 39_000 or len(conditions) != 65:
        raise RuntimeError("future schedule cardinality failure")
    write_json(
        REVIEW / "FUTURE_SEMANTIC_SCHEDULE.json",
        {
            "schema_version": "q2-v4-future-semantic-schedule-v1",
            "status": "FROZEN_NOT_AUTHORIZED_NOT_RUN",
            "prelock_commit": args.prelock_commit,
            "item_count": PRIMARY_N,
            "condition_count": 65,
            "rollouts": 2,
            "row_count": len(schedule),
            "unique_logical_keys": len(keys),
            "unique_seeds": len({int(row["seed"]) for row in schedule}),
            "selected_direction_order": selected_ids,
            "semantic_outcomes": 0,
            "rows": schedule,
        },
    )

    matrices = np.load(REVIEW / "PREDICTION_MATRICES.npz", allow_pickle=False)
    expected_matrices = {
        "A0_MEDIUM",
        "A0_STRONG",
        "A1_MEDIUM",
        "A1_STRONG",
        "A2_MEDIUM",
        "A2_STRONG",
        "D2_MEDIUM",
        "D2_STRONG",
    }
    if set(matrices.files) != expected_matrices:
        raise RuntimeError("prediction matrix archive has wrong members")
    matrix_hashes = {}
    for name in sorted(matrices.files):
        values = np.asarray(matrices[name], dtype=np.float64)
        if values.shape != (SELECTED_COUNT, SELECTED_COUNT):
            raise RuntimeError(f"wrong matrix shape: {name}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite matrix: {name}")
        if np.max(np.abs(values - values.T)) > 1e-10:
            raise RuntimeError(f"asymmetric matrix: {name}")
        if np.max(np.abs(np.diag(values))) > 1e-10:
            raise RuntimeError(f"nonzero matrix diagonal: {name}")
        matrix_hashes[name] = array_hash(values)
    projection = throughput_projection()
    write_json(REVIEW / "SPARK1_THROUGHPUT_PROJECTION.json", projection)
    (REVIEW / "EXECUTION_FREEZE_DRAFT.md").write_text(
        "# Q2 V4 future execution-freeze draft\n\n"
        "Status: `DRAFTED_ONLY / NOT AUTHORIZED / NOT RUN`.\n\n"
        "A future principal lock may execute exactly the frozen 39,000-row schedule "
        "on Spark 1 only, from the prediction-lock commit, using one GB10 and the "
        "recorded environment/model manifests. Direct SSH is the current safe contract "
        "because the inspected dstack configuration does not guarantee Spark-1-only "
        "placement. Spark 2, distributed execution, semantic peeking, Q3, and predictor "
        "changes remain forbidden.\n",
        encoding="utf-8",
    )
    artifact_names = (
        "PRELOCK.json",
        "CANDIDATE_BANK_MANIFEST.json",
        "CANDIDATE_SAFETY_REPORT.json",
        "SELECTED_CONTROLLER_BANK.json",
        "SHELL_CALIBRATION_MANIFEST_RESULT.json",
        "A1_COVARIANCE_ACTIVATIONS.npz",
        "A1_COVARIANCE_FIT.npz",
        "A1_INSTRUMENT_QUALIFICATION.json",
        "A2_INSTRUMENT_QUALIFICATION.json",
        "PREDICTION_MATRICES.npz",
        "PREDICTION_MATRIX_METADATA.json",
        "PRIMARY_PANEL_MANIFEST.json",
        "QAP_CONTROLLER_PERMUTATIONS.npy",
        "QAP_SCHEDULE.json",
        "RADIAL_SHELL_SWAPS.npy",
        "RADIAL_SWAP_SCHEDULE.json",
        "FUTURE_SEMANTIC_SCHEDULE.json",
        "V4_STATISTICAL_ANALYSIS_PLAN.json",
        "SPARK1_ENVIRONMENT_LOCK.json",
        "EXACT_MODEL_MANIFEST.json",
    )
    hashes = {name: sha256(REVIEW / name) for name in artifact_names}
    write_json(
        REVIEW / "PREDICTION_LOCK.json",
        {
            "schema_version": "q2-v4-prediction-lock-v1",
            "status": "Q2_V4_PREDICTION_LOCK_READY_FOR_COMMIT",
            "sealing_commit": "THE_COMMIT_CONTAINING_THIS_ARTIFACT",
            "prelock_commit": args.prelock_commit,
            "controller_order": selected_ids,
            "matrix_hashes": matrix_hashes,
            "artifact_hashes": hashes,
            "future_rows": 39_000,
            "semantic_outcomes": 0,
            "Q3": "NOT_RUN",
        },
    )
    (REVIEW / "PREDICTION_LOCK.md").write_text(
        "# Q2 V4 prediction lock\n\n"
        "Status: `Q2_V4_PREDICTION_LOCK_READY_FOR_COMMIT`.\n\n"
        "The commit containing this artifact seals the source subspace, one candidate "
        "stream, first-32-safe controller order, shell doses, A0/A1/A2/D2 matrices, "
        "300-item panel, 39,000-row future schedule, QAP, radial swaps, bootstrap, and "
        "classification. Semantic outcomes remain zero.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "Q2_V4_PREDICTION_LOCK_READY_FOR_COMMIT"}))


if __name__ == "__main__":
    main()
