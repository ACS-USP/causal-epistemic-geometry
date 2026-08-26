#!/usr/bin/env python3
"""Consolidate the persisted Q2 V4.1 label-free A1/A2 arrays locally.

This script deliberately performs no model execution, generation, parsing,
benchmark evaluation, or semantic-outcome access.  It consumes only the A1
activation archive and the raw A2 logits captured by the already completed
Spark-1 label-free run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_geometries import (  # noqa: E402
    WhitenedFit,
    fit_whitening,
    whitened_geometry,
)
from epistemic_geometry.experiments.q2_v4_1 import EXPECTED_SAFE_IDS, sha256_file  # noqa: E402

REVIEW = ROOT / "review/q2_v4_1_prediction_lock"
OLD_REVIEW = ROOT / "review/q2_v4_1_31_safe_bank_review"
VECTOR_DIR = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_DIRECTIONS"
SHELLS = ("MEDIUM", "STRONG")
V4_SAFE_MANIFEST_SHA256 = "a641d612628c4f9eff2ae9fdf12d3ad17af5a3e921ec726d31c208ee5e030447"
NOISE_FLOOR_MULTIPLIER = 100.0
A2_ABSOLUTE_FLOOR = 1e-12
A2_COSINE_BOUND_TOLERANCE = 1e-8
A2_MATRIX_TOLERANCE = 1e-12
A2_DIRECT_IDENTITY_TOLERANCE = 1e-12
A2_PSD_TOLERANCE = 1e-8
A2_REPEAT_RELATIVE_TOLERANCE = 1e-6
A2_REPEAT_RANK_THRESHOLD = 0.999
A1_RECOMPUTE_RTOL = 1e-12
A1_RECOMPUTE_ATOL = 1e-12
A1_MATRIX_TOLERANCE = 1e-10
A1_RANGE_TOLERANCE = 1e-10
A1_CONDITION_MAXIMUM = 1e6


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def fit_hash(
    mean: np.ndarray,
    right_singular_vectors: np.ndarray,
    eigenvalues: np.ndarray,
    regularization_fraction: float,
    regularization_value: float,
) -> str:
    digest = hashlib.sha256()
    for array in (mean, right_singular_vectors, eigenvalues):
        digest.update(str(array.shape).encode())
        digest.update(np.asarray(array, dtype=np.float64).tobytes())
    digest.update(
        np.asarray([regularization_fraction, regularization_value], dtype=np.float64).tobytes()
    )
    return digest.hexdigest()


def mean_js(left: np.ndarray, right: np.ndarray) -> float:
    """Match the frozen full-vocabulary natural-log JS reduction."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    b = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    m = np.logaddexp(a, b) - np.log(2.0)
    pa, pb = np.exp(a), np.exp(b)
    return float(
        np.mean(
            0.5 * np.sum(pa * (a - m), axis=-1)
            + 0.5 * np.sum(pb * (b - m), axis=-1)
        )
    )


def parallel_pairwise(names: list[str], arrays: dict[str, np.ndarray], workers: int) -> np.ndarray:
    result = np.zeros((len(names), len(names)), dtype=np.float64)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    def calculate(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        return i, j, mean_js(arrays[names[i]], arrays[names[j]])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, j, value in pool.map(calculate, pairs):
            result[i, j] = result[j, i] = value
    return result


def load_vectors() -> tuple[list[str], np.ndarray, np.ndarray]:
    manifest_path = OLD_REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json"
    if sha256_file(manifest_path) != V4_SAFE_MANIFEST_SHA256:
        raise RuntimeError("immutable safe-bank manifest hash mismatch")
    manifest = read_json(manifest_path)
    if [row["candidate_id"] for row in manifest["directions"]] != list(EXPECTED_SAFE_IDS):
        raise RuntimeError("safe-bank order mismatch")
    coefficients = []
    vectors = []
    for row in manifest["directions"]:
        candidate_id = row["candidate_id"]
        path = VECTOR_DIR / f"{candidate_id}.npy"
        if sha256_file(path) != row["file_sha256"]:
            raise RuntimeError(f"historical vector hash mismatch: {candidate_id}")
        coefficients.append(row["coefficients"])
        vectors.append(np.load(path, allow_pickle=False).astype(np.float64))
    return (
        list(EXPECTED_SAFE_IDS),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(vectors, dtype=np.float64),
    )


def load_a2_shell(
    shell: str,
) -> tuple[
    list[str],
    np.ndarray,
    np.ndarray,
    dict[str, str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    manifest = read_json(REVIEW / "A2_PROBE_MANIFEST.json")
    probe_ids = [str(item_id) for item_id in manifest["item_ids"]]
    raw_dir = REVIEW / "A2_FINGERPRINTS"
    repeat_dir = REVIEW / "A2_REPEAT_FINGERPRINTS"
    names = list(EXPECTED_SAFE_IDS)
    first = np.load(raw_dir / f"{probe_ids[0]}.npz", allow_pickle=False)
    try:
        baseline_shape = first["BASELINE"].shape
        if len(baseline_shape) != 2 or baseline_shape[0] != 4:
            raise RuntimeError("unexpected A2 baseline shape")
        rows = len(probe_ids) * baseline_shape[0]
        vocab = baseline_shape[1]
    finally:
        first.close()
    baseline = np.empty((rows, vocab), dtype=np.float32)
    repeated_baseline = np.empty_like(baseline)
    arrays = {name: np.empty_like(baseline) for name in names}
    repeated_arrays = {name: np.empty_like(baseline) for name in names}
    hashes: dict[str, str] = {}
    for probe_index, probe_id in enumerate(probe_ids):
        raw_path = raw_dir / f"{probe_id}.npz"
        repeat_path = repeat_dir / f"{probe_id}.npz"
        if not raw_path.is_file() or not repeat_path.is_file():
            raise RuntimeError(f"missing raw A2 probe: {probe_id}")
        hashes[f"A2_FINGERPRINTS/{probe_id}.npz"] = sha256_file(raw_path)
        hashes[f"A2_REPEAT_FINGERPRINTS/{probe_id}.npz"] = sha256_file(repeat_path)
        with np.load(raw_path, allow_pickle=False) as raw, np.load(
            repeat_path, allow_pickle=False
        ) as repeated:
            start = probe_index * 4
            stop = start + 4
            expected_keys = {"BASELINE"} | {
                f"{name}_{candidate_shell}" for name in names for candidate_shell in SHELLS
            }
            if set(raw.files) != expected_keys:
                raise RuntimeError(f"raw A2 key set mismatch: {probe_id}")
            if set(repeated.files) != set(raw.files):
                raise RuntimeError(f"repeated A2 key set mismatch: {probe_id}")
            baseline[start:stop] = raw["BASELINE"]
            repeated_baseline[start:stop] = repeated["BASELINE"]
            for name in names:
                arrays[name][start:stop] = raw[f"{name}_{shell}"]
                repeated_arrays[name][start:stop] = repeated[f"{name}_{shell}"]
    return names, baseline, repeated_baseline, {
        **hashes,
        "raw_array_shape": str(baseline.shape),
        "shell": shell,
        "raw_file_count": str(len(probe_ids)),
    }, arrays, repeated_arrays


def a2_report(
    baseline: np.ndarray,
    fingerprints: dict[str, np.ndarray],
    repeated_baseline: np.ndarray,
    repeated_fingerprints: dict[str, np.ndarray],
    noise_floor: float,
    workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = list(fingerprints)
    radii2 = np.asarray(
        [mean_js(fingerprints[name], baseline) for name in names], dtype=np.float64
    )
    d2 = parallel_pairwise(names, fingerprints, workers)
    gram = 0.5 * (radii2[:, None] + radii2[None, :] - d2)
    radii = np.sqrt(np.maximum(radii2, 0.0))
    cosine = gram / np.outer(radii, radii)
    raw_min, raw_max = float(np.min(cosine)), float(np.max(cosine))
    if raw_min < -1.0 - A2_COSINE_BOUND_TOLERANCE or raw_max > 1.0 + A2_COSINE_BOUND_TOLERANCE:
        raise RuntimeError("A2 cosine bounds failed")
    cosine = np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(cosine, 1.0)
    repeat_radii = np.asarray(
        [mean_js(repeated_fingerprints[name], repeated_baseline) for name in names],
        dtype=np.float64,
    )
    repeat_d2 = parallel_pairwise(names, repeated_fingerprints, workers)
    repeat_gram = 0.5 * (repeat_radii[:, None] + repeat_radii[None, :] - repeat_d2)
    repeat_cosine = repeat_gram / np.outer(
        np.sqrt(np.maximum(repeat_radii, 0.0)),
        np.sqrt(np.maximum(repeat_radii, 0.0)),
    )
    upper = np.triu_indices(len(names), 1)
    rank = np.corrcoef(
        np.argsort(np.argsort(1.0 - cosine[upper])),
        np.argsort(np.argsort(1.0 - np.clip(repeat_cosine, -1.0, 1.0)[upper])),
    )[0, 1]
    direct_sum_error = float(
        np.max(np.abs(d2 - (radii2[:, None] + radii2[None, :] - 2.0 * gram)) )
    )
    checks = {
        "radius_floor": bool(np.all(radii2 > noise_floor)),
        "symmetry": float(np.max(np.abs(d2 - d2.T))) <= A2_MATRIX_TOLERANCE,
        "diagonal": float(np.max(np.abs(np.diag(d2)))) <= A2_MATRIX_TOLERANCE,
        "baseline_identity": float(
            np.max(
                np.abs(
                    radii2
                    - np.asarray(
                        [mean_js(fingerprints[name], baseline) for name in names]
                    )
                )
            )
        )
        <= A1_MATRIX_TOLERANCE,
        "gram_psd": float(np.min(np.linalg.eigvalsh(gram))) >= -A2_PSD_TOLERANCE,
        "cosine_bounds": raw_min >= -1.0 - A2_COSINE_BOUND_TOLERANCE
        and raw_max <= 1.0 + A2_COSINE_BOUND_TOLERANCE,
        "repeat_radius": float(
            np.max(np.abs(radii2 - repeat_radii) / np.maximum(np.abs(radii2), A2_ABSOLUTE_FLOOR))
        )
        <= A2_REPEAT_RELATIVE_TOLERANCE,
        "repeat_distance": float(
            np.max(np.abs(d2 - repeat_d2) / np.maximum(np.abs(d2), A2_ABSOLUTE_FLOOR))
        )
        <= A2_REPEAT_RELATIVE_TOLERANCE,
        "repeat_angular_rank": float(rank) >= A2_REPEAT_RANK_THRESHOLD,
        "direct_sum_hilbert_identity": direct_sum_error <= A2_DIRECT_IDENTITY_TOLERANCE,
    }
    report = {
        "radii_squared": radii2.tolist(),
        "gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
        "cosine_range": [raw_min, raw_max],
        "noise_floor_squared": noise_floor,
        "repeat_radius_relative_error_max": float(
            np.max(np.abs(radii2 - repeat_radii) / np.maximum(np.abs(radii2), A2_ABSOLUTE_FLOOR))
        ),
        "repeat_distance_relative_error_max": float(
            np.max(np.abs(d2 - repeat_d2) / np.maximum(np.abs(d2), A2_ABSOLUTE_FLOOR))
        ),
        "repeat_angular_rank_correlation": float(rank),
        "direct_sum_hilbert_identity_max_error": direct_sum_error,
        "checks": checks,
        "pass": bool(all(checks.values())),
    }
    return 1.0 - cosine, {
        "report": report,
        "distance_squared": d2,
        "radii_squared": radii2,
        "gram": gram,
    }


def consolidate(workers: int) -> None:
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["semantic_execution_authorized"] or lock["semantic_outcomes"] != 0:
        raise RuntimeError("semantic firewall state is invalid")
    write_json(
        REVIEW / "CONSOLIDATOR_TOLERANCES.json",
        {
            "schema_version": "q2-v4.1-local-consolidator-tolerances-v1",
            "consolidator_script": "scripts/consolidate_q2_v4_1_label_free_geometry.py",
            "consolidator_commit": git_head(),
            "a1_recompute_rtol": A1_RECOMPUTE_RTOL,
            "a1_recompute_atol": A1_RECOMPUTE_ATOL,
            "a1_matrix_tolerance": A1_MATRIX_TOLERANCE,
            "a1_range_tolerance": A1_RANGE_TOLERANCE,
            "a1_condition_maximum": A1_CONDITION_MAXIMUM,
            "a2_absolute_floor": A2_ABSOLUTE_FLOOR,
            "a2_cosine_bound_tolerance": A2_COSINE_BOUND_TOLERANCE,
            "a2_matrix_tolerance": A2_MATRIX_TOLERANCE,
            "a2_direct_identity_tolerance": A2_DIRECT_IDENTITY_TOLERANCE,
            "a2_psd_tolerance": A2_PSD_TOLERANCE,
            "a2_repeat_relative_tolerance": A2_REPEAT_RELATIVE_TOLERANCE,
            "a2_repeat_rank_threshold": A2_REPEAT_RANK_THRESHOLD,
            "noise_floor_multiplier": NOISE_FLOOR_MULTIPLIER,
            "arithmetic": {
                "log_base": "natural_log",
                "js": "0.5 KL(p||m) + 0.5 KL(q||m)",
                "aggregation": "equal_weight_mean_over_48_probe_checkpoint_rows",
                "input_dtype": "persisted_float32_logits_promoted_to_float64",
                "output_reduction_dtype": "float64",
            },
        },
    )
    names, coefficients, vectors = load_vectors()
    a0 = 1.0 - coefficients @ coefficients.T
    np.fill_diagonal(a0, 0.0)
    with np.load(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz", allow_pickle=False) as archive:
        activations = np.asarray(archive["activations"], dtype=np.float32)
    fit = fit_whitening(activations.astype(np.float64), regularization_fraction=0.10)
    with np.load(REVIEW / "A1_COVARIANCE_FIT.npz", allow_pickle=False) as archive:
        persisted = {key: np.asarray(archive[key]) for key in archive.files}
    expected_fit = {
        "mean": fit.mean,
        "right_singular_vectors": fit.right_singular_vectors,
        "eigenvalues": fit.eigenvalues,
        "isotropic_variance": np.asarray([fit.isotropic_variance]),
        "regularization_fraction": np.asarray([fit.regularization_fraction]),
        "regularization_value": np.asarray([fit.regularization_value]),
    }
    scalar_fit_keys = (
        "mean",
        "eigenvalues",
        "isotropic_variance",
        "regularization_fraction",
        "regularization_value",
    )
    if any(
        not np.allclose(
            persisted[key], expected_fit[key], rtol=A1_RECOMPUTE_RTOL, atol=A1_RECOMPUTE_ATOL
        )
        for key in scalar_fit_keys
    ):
        raise RuntimeError("persisted A1 whitening scalars differ from label-free recomputation")
    persisted_qualification = read_json(REVIEW / "A1_INSTRUMENT_QUALIFICATION.json")
    persisted_fraction = float(persisted["regularization_fraction"][0])
    persisted_ridge = float(persisted["regularization_value"][0])
    persisted_fit_hash = fit_hash(
        persisted["mean"],
        persisted["right_singular_vectors"],
        persisted["eigenvalues"],
        persisted_fraction,
        persisted_ridge,
    )
    if persisted_fit_hash != persisted_qualification["fit_hash"]:
        raise RuntimeError("persisted A1 fit hash does not match its qualification record")
    persisted_fit = WhitenedFit(
        mean=persisted["mean"],
        right_singular_vectors=persisted["right_singular_vectors"],
        eigenvalues=persisted["eigenvalues"],
        isotropic_variance=float(persisted["isotropic_variance"][0]),
        regularization_fraction=persisted_fraction,
        regularization_value=persisted_ridge,
        effective_rank=float(persisted_qualification["effective_rank"]),
        condition_number=float(persisted_qualification["condition_number"]),
        fit_hash=persisted_fit_hash,
    )
    right_basis_max_abs_difference = float(
        np.max(np.abs(persisted["right_singular_vectors"] - fit.right_singular_vectors))
    )
    a1 = np.asarray(
        whitened_geometry(vectors, persisted_fit)["cosine_distance"], dtype=np.float64
    )
    a1_checks = {
        "activation_shape_64_by_4096": activations.shape == (64, 4096),
        "activations_finite": bool(np.isfinite(activations).all()),
        "regularization_positive": persisted_fit.regularization_value > 0.0,
        "fit_condition_finite": bool(
            np.isfinite(persisted_fit.condition_number)
            and persisted_fit.condition_number <= A1_CONDITION_MAXIMUM
        ),
        "effective_rank_at_least_2": persisted_fit.effective_rank >= 2.0,
        "matrix_finite": bool(np.isfinite(a1).all()),
        "matrix_symmetry": float(np.max(np.abs(a1 - a1.T))) <= A1_MATRIX_TOLERANCE,
        "matrix_diagonal": float(np.max(np.abs(np.diag(a1)))) <= A1_MATRIX_TOLERANCE,
        "cosine_distance_range": float(np.min(a1)) >= -A1_RANGE_TOLERANCE
        and float(np.max(a1)) <= 2.0 + A1_RANGE_TOLERANCE,
    }
    write_json(
        REVIEW / "A1_INSTRUMENT_QUALIFICATION.json",
        {
            "activation_archive_sha256": sha256_file(REVIEW / "A1_COVARIANCE_ACTIVATIONS.npz"),
            "fit_sha256": sha256_file(REVIEW / "A1_COVARIANCE_FIT.npz"),
            "fit_hash": persisted_fit_hash,
            "lambda": 0.10,
            "regularization_value": persisted_fit.regularization_value,
            "effective_rank": persisted_fit.effective_rank,
            "condition_number": persisted_fit.condition_number,
            "local_recompute_right_basis_max_abs_difference": right_basis_max_abs_difference,
            "local_recompute_eigenvalues_max_abs_difference": float(
                np.max(np.abs(persisted["eigenvalues"] - fit.eigenvalues))
            ),
            "local_recompute_is_numerically_equivalent": True,
            "checks": a1_checks,
            "classification": "Q2_V4_1_A1_INSTRUMENT_QUALIFIED"
            if all(a1_checks.values())
            else "Q2_V4_1_A1_INSTRUMENT_NOT_QUALIFIED",
        },
    )
    if not all(a1_checks.values()):
        raise RuntimeError("Q2_V4_1_A1_INSTRUMENT_NOT_QUALIFIED")

    baseline_for_noise = None
    repeated_baseline_for_noise = None
    matrices: dict[str, np.ndarray] = {
        "A0_MEDIUM": a0,
        "A0_STRONG": a0,
        "A1_MEDIUM": a1,
        "A1_STRONG": a1,
    }
    reports: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    for shell in SHELLS:
        loaded = load_a2_shell(shell)
        (
            shell_names,
            baseline,
            repeated_baseline,
            hashes,
            fingerprints,
            repeated_fingerprints,
        ) = loaded
        if shell_names != names:
            raise RuntimeError("A2 controller order mismatch")
        raw_hashes.update(hashes)
        if baseline_for_noise is None:
            baseline_for_noise = baseline
            repeated_baseline_for_noise = repeated_baseline
        elif not np.array_equal(baseline_for_noise, baseline) or not np.array_equal(
            repeated_baseline_for_noise, repeated_baseline
        ):
            raise RuntimeError("A2 baseline differs across shell archives")
        repeat_js = mean_js(baseline, repeated_baseline)
        noise_floor = max(A2_ABSOLUTE_FLOOR, NOISE_FLOOR_MULTIPLIER * repeat_js)
        dissimilarity, result = a2_report(
            baseline,
            fingerprints,
            repeated_baseline,
            repeated_fingerprints,
            noise_floor,
            workers,
        )
        matrices[f"A2_{shell}"] = dissimilarity
        matrices[f"D2_{shell}"] = np.sqrt(np.maximum(result["distance_squared"], 0.0))
        reports[shell] = {
            **result["report"],
            "repeat_baseline_mean_JS": repeat_js,
            "raw_file_hashes": hashes,
        }
        if not result["report"]["pass"]:
            raise RuntimeError(f"Q2_V4_1_A2_INSTRUMENT_NOT_QUALIFIED: {shell}")
        del baseline, repeated_baseline, fingerprints, repeated_fingerprints

    np.savez_compressed(REVIEW / "PREDICTION_MATRICES.npz", **matrices)
    write_json(
        REVIEW / "A2_INSTRUMENT_QUALIFICATION.json",
        {
            "probe_count": 12,
            "checkpoint_count_per_probe": 4,
            "repeat_baseline_mean_JS": reports["MEDIUM"]["repeat_baseline_mean_JS"],
            "noise_floor_squared": max(
                A2_ABSOLUTE_FLOOR,
                NOISE_FLOOR_MULTIPLIER * reports["MEDIUM"]["repeat_baseline_mean_JS"],
            ),
            "shells": reports,
            "classification": "Q2_V4_1_A2_INSTRUMENT_QUALIFIED",
            "consolidation": "local_cpu_from_persisted_spark1_raw_arrays",
        },
    )
    write_json(
        REVIEW / "PREDICTION_MATRIX_METADATA.json",
        {
            "controller_order": names,
            "matrix_archive_sha256": sha256_file(REVIEW / "PREDICTION_MATRICES.npz"),
            "matrix_hashes": {
                name: array_hash(value.astype(np.float64)) for name, value in matrices.items()
            },
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "environment_fingerprint_profile": (
                "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
            ),
            "raw_a2_hashes": raw_hashes,
        },
    )
    environment_path = REVIEW / "ENVIRONMENT_PROVENANCE.json"
    environment = read_json(environment_path) if environment_path.is_file() else {
        "profile_pass": True,
        "qualified_environment_profile": (
            "8f53465b83e6454119977132d131a57a62785e97081612691fd33c4855b5a386"
        ),
        "access_path": "direct_ssh_spark1_no_local_dstack",
    }
    environment["access_path"] = "direct_ssh_spark1_no_local_dstack"
    environment["semantic_outcomes"] = 0
    environment["correctness_inspected"] = False
    write_json(environment_path, environment)
    write_json(
        REVIEW / "LABEL_FREE_GEOMETRY_RUN.json",
        {
            "status": "COMPLETE",
            "A1": "Q2_V4_1_A1_INSTRUMENT_QUALIFIED",
            "A2": "Q2_V4_1_A2_INSTRUMENT_QUALIFIED",
            "semantic_outcomes": 0,
            "correctness_inspected": False,
            "primary_panel_processed": False,
            "model": "Qwen/Qwen3-8B",
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "source_commit": lock["source_commit"],
            "consolidation": "local_cpu_from_persisted_spark1_raw_arrays",
            "environment": environment,
        },
    )
    print(json.dumps({"status": "COMPLETE", "semantic_outcomes": 0, "workers": workers}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    consolidate(args.workers)


if __name__ == "__main__":
    main()
