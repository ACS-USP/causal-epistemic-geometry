#!/usr/bin/env python3
"""Materialize the one frozen Q2 OOS candidate stream after PRELOCK."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    fresh_candidate_bank,
    protocol_seed,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design"
HISTORICAL = ROOT / "review/q2_v4_spark1_presemantic"
NAMESPACE = "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def stream_checks(coefficients: np.ndarray, vectors: np.ndarray) -> dict[str, Any]:
    singular = np.linalg.svd(coefficients, compute_uv=False)
    energy = np.square(singular)
    probability = energy / np.sum(energy)
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability))))
    cosine = np.abs(coefficients @ coefficients.T)
    upper = cosine[np.triu_indices(len(coefficients), 1)]
    checks = {
        "finite": bool(np.isfinite(coefficients).all() and np.isfinite(vectors).all()),
        "coefficient_unit_norm_error_at_most_1e_12": float(
            np.max(np.abs(np.linalg.norm(coefficients, axis=1) - 1.0))
        )
        <= 1e-12,
        "vector_unit_norm_error_at_most_1e_10": float(
            np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0))
        )
        <= 1e-10,
        "rank_8": int(np.linalg.matrix_rank(coefficients, tol=1e-10)) == 8,
        "effective_rank_at_least_6": effective_rank >= 6.0,
        "condition_number_at_most_3": float(singular[0] / singular[-1]) <= 3.0,
        "maximum_absolute_pair_cosine_below_0_98": float(np.max(upper)) < 0.98,
    }
    return {
        "rank": int(np.linalg.matrix_rank(coefficients, tol=1e-10)),
        "effective_rank": effective_rank,
        "condition_number": float(singular[0] / singular[-1]),
        "maximum_absolute_pair_cosine": float(np.max(upper)),
        "checks": checks,
        "pass": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelock-commit", required=True)
    args = parser.parse_args()
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    prelock = read_json(REVIEW / "PRELOCK.json")
    if prelock["candidate_bank_exists"] is not False:
        raise RuntimeError("PRELOCK does not certify an unopened candidate stream")
    expected_lock_hash = prelock["protocol_lock_sha256"]
    if sha256(REVIEW / "PROTOCOL_LOCK.json") != expected_lock_hash:
        raise RuntimeError("protocol lock hash mismatch")
    q_path = HISTORICAL / "SPARK1_SUBSPACE_Q.npy"
    if sha256(q_path) != lock["source_hashes"]["source_subspace_q"]:
        raise RuntimeError("frozen Q-basis hash mismatch")
    count = int(lock["fresh_controller_policy"]["candidate_count"])
    seed = protocol_seed(NAMESPACE, args.prelock_commit)
    basis = np.load(q_path, allow_pickle=False).astype(np.float64)
    coefficients, vectors = fresh_candidate_bank(basis, count=count, seed=seed)
    checks = stream_checks(coefficients, vectors)

    historical_manifest = read_json(HISTORICAL / "CANDIDATE_BANK_MANIFEST.json")
    historical_coefficients = np.asarray(
        [row["coefficients"] for row in historical_manifest["candidates"]], dtype=np.float64
    )
    historical_hashes = {
        array_hash(np.load(ROOT / row["path"], allow_pickle=False).astype(np.float64))
        for row in historical_manifest["candidates"]
    }
    cross_cosine = np.abs(coefficients @ historical_coefficients.T)
    exact_coefficient_overlap = any(
        np.array_equal(fresh, old) for fresh in coefficients for old in historical_coefficients
    )
    output_dir = REVIEW / "CANDIDATE_DIRECTIONS"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    fresh_array_hashes = set()
    for index, (coefficient, vector) in enumerate(zip(coefficients, vectors, strict=True)):
        candidate_id = f"Q2_OOS_DIRECTION_{index:02d}"
        path = output_dir / f"{candidate_id}.npy"
        np.save(path, vector.astype(np.float64), allow_pickle=False)
        value_hash = array_hash(vector)
        fresh_array_hashes.add(value_hash)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "coefficients": coefficient.tolist(),
                "coefficient_array_sha256": array_hash(coefficient),
                "path": str(path.relative_to(ROOT)),
                "file_sha256": sha256(path),
                "vector_array_sha256": value_hash,
                "vector_norm": float(np.linalg.norm(vector)),
            }
        )
    overlap = {
        "historical_candidate_count": len(historical_coefficients),
        "fresh_candidate_count": len(coefficients),
        "candidate_id_overlap": False,
        "exact_coefficient_overlap": exact_coefficient_overlap,
        "vector_array_hash_overlap": bool(fresh_array_hashes & historical_hashes),
        "maximum_absolute_cross_stream_cosine": float(np.max(cross_cosine)),
        "pass": (
            not exact_coefficient_overlap
            and not bool(fresh_array_hashes & historical_hashes)
            and float(np.max(cross_cosine)) < 1.0 - 1e-12
        ),
    }
    if not overlap["pass"]:
        raise RuntimeError("Q2_OOS_FRESH_CONTROLLER_HISTORICAL_OVERLAP")
    payload = {
        "schema_version": "q2-oos-fresh-controller-candidate-stream-v1",
        "prelock_commit": args.prelock_commit,
        "namespace": NAMESPACE,
        "seed": seed,
        "rng": "NumPy PCG64DXSM",
        "draw": "normalized Gaussian in R8; v=Qc",
        "generated_once": True,
        "redraw": "FORBIDDEN",
        "candidate_count": count,
        "Q_basis_sha256": sha256(q_path),
        "candidates": rows,
        "algebraic_checks": checks,
        "historical_overlap_audit": overlap,
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "classification": (
            "Q2_OOS_FRESH_CONTROLLER_CANDIDATE_STREAM_FROZEN"
            if checks["pass"]
            else "Q2_OOS_FRESH_CONTROLLER_CANDIDATE_STREAM_ALGEBRAIC_FAIL"
        ),
        "deterministic_materialization_note": (
            "The first invocation derived this same seed/stream and stopped before persistence "
            "when the frozen algebraic gate failed. This artifact is an exact deterministic "
            "rematerialization of that one stream, not a redraw."
        ),
    }
    write_json(REVIEW / "CANDIDATE_BANK_MANIFEST.json", payload)
    write_json(REVIEW / "HISTORICAL_OVERLAP_AUDIT.json", overlap)
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "seed": seed,
                "candidate_count": count,
                "max_historical_cosine": overlap["maximum_absolute_cross_stream_cosine"],
            },
            indent=2,
        )
    )
    if not checks["pass"]:
        raise RuntimeError("Q2_OOS_FRESH_CONTROLLER_CANDIDATE_STREAM_ALGEBRAIC_FAIL")


if __name__ == "__main__":
    main()
