#!/usr/bin/env python3
"""Materialize exactly one prospective Q2 OOS V2 candidate stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    coefficient_bank_diagnostics,
    fresh_candidate_bank,
    protocol_seed,
)

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
V1 = ROOT / "review/q2_oos_fresh_controller_design"
V4 = ROOT / "review/q2_v4_spark1_presemantic"
NAMESPACE = "Q2-OOS-FRESH-CONTROLLER-DIRECTIONS-V2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_manifest_coefficients(path: Path) -> np.ndarray:
    manifest = read_json(path)
    return np.asarray(
        [row["coefficients"] for row in manifest["candidates"]], dtype=np.float64
    )


def load_manifest_vector_hashes(path: Path) -> set[str]:
    manifest = read_json(path)
    return {
        str(row.get("vector_array_sha256", row.get("canonical_vector_hash", "")))
        for row in manifest["candidates"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prelock-commit", required=True)
    args = parser.parse_args()
    lock_path = REVIEW / "V2_FINAL_PROTOCOL_LOCK.json"
    prelock_path = REVIEW / "V2_PRELOCK.json"
    lock = read_json(lock_path)
    prelock = read_json(prelock_path)
    if lock["status"] != "Q2_OOS_V2_FINAL_PROTOCOL_FROZEN":
        raise RuntimeError("V2 final protocol lock is not frozen")
    if prelock["candidate_stream_exists"] is not False:
        raise RuntimeError("PRELOCK does not certify an unopened stream")
    if sha256(lock_path) != prelock["protocol_lock_sha256"]:
        raise RuntimeError("protocol lock hash mismatch")
    if lock["candidate_generation"]["namespace"] != NAMESPACE:
        raise RuntimeError("V2 namespace mismatch")
    if int(lock["design"]["candidate_count"]) != 34:
        raise RuntimeError("V2 candidate count mismatch")

    output_dir = REVIEW / "CANDIDATE_DIRECTIONS"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("V2 candidate output directory is not empty")

    q_path = V4 / "SPARK1_SUBSPACE_Q.npy"
    if sha256(q_path) != lock["source_hashes"]["rank8_Q_basis"]:
        raise RuntimeError("frozen rank-8 Q hash mismatch")
    basis = np.load(q_path, allow_pickle=False).astype(np.float64)
    seed = protocol_seed(NAMESPACE, args.prelock_commit)
    coefficients, vectors = fresh_candidate_bank(basis, count=34, seed=seed)
    diagnostics = coefficient_bank_diagnostics(coefficients)
    coefficient_norm_error = float(
        np.max(np.abs(np.linalg.norm(coefficients, axis=1) - 1.0))
    )
    vector_norm_error = float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0)))
    checks = {
        "finite": bool(np.isfinite(coefficients).all() and np.isfinite(vectors).all()),
        "rank_8": diagnostics["rank"] == 8,
        "coefficient_unit_norm_error_at_most_1e_12": coefficient_norm_error <= 1e-12,
        "vector_unit_norm_error_at_most_1e_10": vector_norm_error <= 1e-10,
        "maximum_absolute_pair_cosine_below_0_98": (
            diagnostics["maximum_absolute_pair_cosine"] < 0.98
        ),
    }

    v1_manifest_path = V1 / "CANDIDATE_BANK_MANIFEST.json"
    v4_manifest_path = V4 / "CANDIDATE_BANK_MANIFEST.json"
    v1_coefficients = load_manifest_coefficients(v1_manifest_path)
    v4_coefficients = load_manifest_coefficients(v4_manifest_path)
    v1_hashes = load_manifest_vector_hashes(v1_manifest_path)
    v4_hashes = load_manifest_vector_hashes(v4_manifest_path)
    candidate_hashes = {array_hash(vector) for vector in vectors}
    exact_v1_overlap = any(
        np.array_equal(candidate, historical)
        for candidate in coefficients
        for historical in v1_coefficients
    )
    exact_v4_overlap = any(
        np.array_equal(candidate, historical)
        for candidate in coefficients
        for historical in v4_coefficients
    )
    overlap_checks = {
        "no_exact_V1_coefficient_overlap": not exact_v1_overlap,
        "no_V1_vector_hash_overlap": not bool(candidate_hashes & v1_hashes),
        "no_exact_V4_coefficient_overlap": not exact_v4_overlap,
        "no_V4_vector_hash_overlap": not bool(candidate_hashes & v4_hashes),
    }
    all_checks = {**checks, **overlap_checks}

    output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, (coefficient, vector) in enumerate(zip(coefficients, vectors, strict=True)):
        candidate_id = f"Q2_OOS_V2_DIRECTION_{index:02d}"
        path = output_dir / f"{candidate_id}.npy"
        np.save(path, vector.astype(np.float64), allow_pickle=False)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_index": index,
                "coefficients": coefficient.tolist(),
                "coefficient_array_sha256": array_hash(coefficient),
                "path": str(path.relative_to(ROOT)),
                "file_sha256": sha256(path),
                "vector_array_sha256": array_hash(vector),
                "vector_norm": float(np.linalg.norm(vector)),
            }
        )
    payload = {
        "schema_version": "q2-oos-v2-candidate-stream-v1",
        "prelock_commit": args.prelock_commit,
        "namespace": NAMESPACE,
        "seed": str(seed),
        "seed_hex_128": f"{seed:032x}",
        "byte_order": "big",
        "rng": "NumPy PCG64DXSM",
        "draw": "g~N(0,I_8); c=g/||g||; v=Qc",
        "generated_once": True,
        "redraws": 0,
        "replacement": "FORBIDDEN",
        "additional_candidates": "FORBIDDEN",
        "candidate_count": 34,
        "Q_basis_sha256": sha256(q_path),
        "candidates": rows,
        "stream_integrity": {
            "metrics": {
                **diagnostics,
                "coefficient_unit_norm_error": coefficient_norm_error,
                "vector_unit_norm_error": vector_norm_error,
                "maximum_absolute_cosine_with_V1": float(
                    np.max(np.abs(coefficients @ v1_coefficients.T))
                ),
                "maximum_absolute_cosine_with_V4": float(
                    np.max(np.abs(coefficients @ v4_coefficients.T))
                ),
            },
            "checks": all_checks,
            "effective_rank_role": "DESCRIPTIVE_ONLY",
            "condition_number_role": "DESCRIPTIVE_ONLY",
            "pass": bool(all(all_checks.values())),
        },
        "classification": (
            "Q2_OOS_V2_CANDIDATE_STREAM_INTEGRITY_PASS"
            if all(all_checks.values())
            else "Q2_OOS_V2_STREAM_INTEGRITY_BLOCKED"
        ),
        "semantic_outcomes": 0,
        "correctness_inspected": False,
    }
    write_json(REVIEW / "V2_CANDIDATE_BANK_MANIFEST.json", payload)
    if payload["classification"] != "Q2_OOS_V2_CANDIDATE_STREAM_INTEGRITY_PASS":
        raise RuntimeError("Q2_OOS_V2_STREAM_INTEGRITY_BLOCKED")
    print(
        json.dumps(
            {
                "classification": payload["classification"],
                "candidate_count": 34,
                "seed_hex_128": payload["seed_hex_128"],
                "effective_rank_descriptive": diagnostics["effective_rank"],
                "condition_number_descriptive": diagnostics["condition_number"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
