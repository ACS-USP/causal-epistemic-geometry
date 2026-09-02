#!/usr/bin/env python3
"""Consolidate Q2 OOS V2 A0/A1/A2/D2 without semantic outcomes.

The implementation is frozen before fresh A2 capture.  It combines the exact
historical V4.1 raw A2 archive with the fresh-controller capture, uses the
historical whitening fit unchanged, and writes only aggregate matrices and
qualification metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_geometries import WhitenedFit, whitened_geometry  # noqa: E402
from epistemic_geometry.experiments.q2_v4_1 import EXPECTED_SAFE_IDS  # noqa: E402

REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
V2_STREAM = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
V41 = ROOT / "review/q2_v4_1_prediction_lock"
V41_SAFE = ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
V41_VECTOR_DIR = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_DIRECTIONS"
SELECTED_BANK = REVIEW / "V2_SELECTED_CONTROLLER_BANK.json"
STREAM_MANIFEST = V2_STREAM / "V2_CANDIDATE_BANK_MANIFEST.json"
SHELLS = ("MEDIUM", "STRONG")
PROBE_ROWS = 48
NOISE_FLOOR_MULTIPLIER = 100.0
A2_ABSOLUTE_FLOOR = 1e-12
A2_COSINE_BOUND_TOLERANCE = 1e-8
A2_MATRIX_TOLERANCE = 1e-12
A2_PSD_TOLERANCE = 1e-8
A2_REPEAT_RELATIVE_TOLERANCE = 1e-6
A2_REPEAT_RANK_THRESHOLD = 0.999
BASELINE_MATCH_TOLERANCE = 1e-6
A1_MATRIX_TOLERANCE = 1e-10
HISTORICAL_MATRIX_TOLERANCE = 1e-10


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def mean_js(left: np.ndarray, right: np.ndarray) -> float:
    """Natural-log JS with 0.5/0.5 weights and a uniform row mean."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - np.logaddexp.reduce(a, axis=-1, keepdims=True)
    b = b - np.logaddexp.reduce(b, axis=-1, keepdims=True)
    mixture = np.logaddexp(a, b) - np.log(2.0)
    pa, pb = np.exp(a), np.exp(b)
    return float(
        np.mean(
            0.5 * np.sum(pa * (a - mixture), axis=-1)
            + 0.5 * np.sum(pb * (b - mixture), axis=-1)
        )
    )


def pairwise_js(names: list[str], arrays: dict[str, np.ndarray], workers: int) -> np.ndarray:
    result = np.zeros((len(names), len(names)), dtype=np.float64)
    pairs = [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))]

    def calculate(pair: tuple[int, int]) -> tuple[int, int, float]:
        i, j = pair
        return i, j, mean_js(arrays[names[i]], arrays[names[j]])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, j, value in pool.map(calculate, pairs):
            result[i, j] = result[j, i] = value
    return result


def load_fit() -> WhitenedFit:
    qualification = read_json(V41 / "A1_INSTRUMENT_QUALIFICATION.json")
    fit_path = V41 / "A1_COVARIANCE_FIT.npz"
    if sha256_file(fit_path) != qualification["fit_sha256"]:
        raise RuntimeError("Q2_OOS_V2_A1_FROZEN_FIT_HASH_MISMATCH")
    with np.load(fit_path, allow_pickle=False) as archive:
        return WhitenedFit(
            mean=np.asarray(archive["mean"], dtype=np.float64),
            right_singular_vectors=np.asarray(
                archive["right_singular_vectors"], dtype=np.float64
            ),
            eigenvalues=np.asarray(archive["eigenvalues"], dtype=np.float64),
            isotropic_variance=float(archive["isotropic_variance"][0]),
            regularization_fraction=float(archive["regularization_fraction"][0]),
            regularization_value=float(archive["regularization_value"][0]),
            effective_rank=float(qualification["effective_rank"]),
            condition_number=float(qualification["condition_number"]),
            fit_hash=str(qualification["fit_hash"]),
        )


def load_vectors() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    selected = read_json(SELECTED_BANK)
    fresh_names = [str(value) for value in selected["selected_ids"]]
    stream = read_json(STREAM_MANIFEST)
    stream_rows = {row["candidate_id"]: row for row in stream["candidates"]}
    fresh_coefficients = np.asarray(
        [stream_rows[name]["coefficients"] for name in fresh_names], dtype=np.float64
    )
    fresh_vectors = np.stack(
        [
            np.load(ROOT / stream_rows[name]["path"], allow_pickle=False).astype(np.float64)
            for name in fresh_names
        ]
    )
    historical = read_json(V41_SAFE)
    reference_names = [str(row["candidate_id"]) for row in historical["directions"]]
    if reference_names != list(EXPECTED_SAFE_IDS):
        raise RuntimeError("Q2_OOS_V2_REFERENCE_ORDER_MISMATCH")
    reference_coefficients = np.asarray(
        [row["coefficients"] for row in historical["directions"]], dtype=np.float64
    )
    reference_vectors = np.stack(
        [
            np.load(V41_VECTOR_DIR / f"{name}.npy", allow_pickle=False).astype(np.float64)
            for name in reference_names
        ]
    )
    return (
        fresh_names,
        reference_names,
        np.concatenate([fresh_coefficients, reference_coefficients], axis=0),
        np.concatenate([fresh_vectors, reference_vectors], axis=0),
    )


def load_probe_archives(
    historical_dir: Path,
    fresh_dir: Path,
    shell: str,
    fresh_names: list[str],
    reference_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], float]:
    probes = read_json(V41 / "A2_PROBE_MANIFEST.json")["item_ids"]
    old_hashes = read_json(V41 / "A2_RAW_ARCHIVE_HASHES.json")
    fresh_hashes = read_json(fresh_dir / "A2_FRESH_RAW_ARCHIVE_HASHES.json")
    historical_baseline: list[np.ndarray] = []
    historical_repeat_baseline: list[np.ndarray] = []
    fresh_baseline: list[np.ndarray] = []
    fresh_repeat_baseline: list[np.ndarray] = []
    arrays = {name: [] for name in [*fresh_names, *reference_names]}
    repeated = {name: [] for name in [*fresh_names, *reference_names]}
    for probe in probes:
        old_raw = historical_dir / "A2_FINGERPRINTS" / f"{probe}.npz"
        old_repeat = historical_dir / "A2_REPEAT_FINGERPRINTS" / f"{probe}.npz"
        new_raw = fresh_dir / "A2_FRESH_FINGERPRINTS" / f"{probe}.npz"
        new_repeat = fresh_dir / "A2_FRESH_REPEAT_FINGERPRINTS" / f"{probe}.npz"
        expected_old_raw = old_hashes["files"][f"A2_FINGERPRINTS/{probe}.npz"]
        expected_old_repeat = old_hashes["files"][f"A2_REPEAT_FINGERPRINTS/{probe}.npz"]
        expected_new_raw = fresh_hashes["files"][f"A2_FRESH_FINGERPRINTS/{probe}.npz"]
        expected_new_repeat = fresh_hashes["files"][
            f"A2_FRESH_REPEAT_FINGERPRINTS/{probe}.npz"
        ]
        observed = [
            sha256_file(old_raw),
            sha256_file(old_repeat),
            sha256_file(new_raw),
            sha256_file(new_repeat),
        ]
        expected = [
            expected_old_raw,
            expected_old_repeat,
            expected_new_raw,
            expected_new_repeat,
        ]
        if observed != expected:
            raise RuntimeError(f"Q2_OOS_V2_A2_RAW_HASH_MISMATCH:{probe}")
        with np.load(old_raw, allow_pickle=False) as old, np.load(
            old_repeat, allow_pickle=False
        ) as old_rep, np.load(new_raw, allow_pickle=False) as new, np.load(
            new_repeat, allow_pickle=False
        ) as new_rep:
            historical_baseline.append(old["BASELINE"])
            historical_repeat_baseline.append(old_rep["BASELINE"])
            fresh_baseline.append(new["BASELINE"])
            fresh_repeat_baseline.append(new_rep["BASELINE"])
            for name in reference_names:
                arrays[name].append(old[f"{name}_{shell}"])
                repeated[name].append(old_rep[f"{name}_{shell}"])
            for name in fresh_names:
                arrays[name].append(new[f"{name}_{shell}"])
                repeated[name].append(new_rep[f"{name}_{shell}"])
    old_base = np.concatenate(historical_baseline, axis=0).astype(np.float64)
    old_repeat_base = np.concatenate(historical_repeat_baseline, axis=0).astype(np.float64)
    new_base = np.concatenate(fresh_baseline, axis=0).astype(np.float64)
    new_repeat_base = np.concatenate(fresh_repeat_baseline, axis=0).astype(np.float64)
    if old_base.shape[0] != PROBE_ROWS or new_base.shape != old_base.shape:
        raise RuntimeError("Q2_OOS_V2_A2_ROW_SHAPE_MISMATCH")
    baseline_error = max(
        float(np.max(np.abs(old_base - new_base))),
        float(np.max(np.abs(old_repeat_base - new_repeat_base))),
    )
    if baseline_error > BASELINE_MATCH_TOLERANCE:
        raise RuntimeError("Q2_OOS_V2_A2_BASELINE_RECAPTURE_MISMATCH")
    return (
        old_base,
        old_repeat_base,
        {
            name: np.concatenate(values, axis=0).astype(np.float64)
            for name, values in arrays.items()
        },
        {
            name: np.concatenate(values, axis=0).astype(np.float64)
            for name, values in repeated.items()
        },
        baseline_error,
    )


def a2_geometry(
    names: list[str],
    baseline: np.ndarray,
    repeated_baseline: np.ndarray,
    arrays: dict[str, np.ndarray],
    repeated: dict[str, np.ndarray],
    workers: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    radii2 = np.asarray([mean_js(arrays[name], baseline) for name in names])
    repeat_radii2 = np.asarray([mean_js(repeated[name], repeated_baseline) for name in names])
    distances2 = pairwise_js(names, arrays, workers)
    repeat_distances2 = pairwise_js(names, repeated, workers)
    gram = 0.5 * (radii2[:, None] + radii2[None, :] - distances2)
    repeat_gram = 0.5 * (
        repeat_radii2[:, None] + repeat_radii2[None, :] - repeat_distances2
    )
    cosine = gram / np.outer(np.sqrt(radii2), np.sqrt(radii2))
    repeat_cosine = repeat_gram / np.outer(np.sqrt(repeat_radii2), np.sqrt(repeat_radii2))
    raw_min, raw_max = float(np.min(cosine)), float(np.max(cosine))
    upper = np.triu_indices(len(names), 1)
    rank = float(
        np.corrcoef(
            np.argsort(np.argsort(1.0 - np.clip(cosine[upper], -1.0, 1.0))),
            np.argsort(np.argsort(1.0 - np.clip(repeat_cosine[upper], -1.0, 1.0))),
        )[0, 1]
    )
    radius_relative = float(
        np.max(np.abs(radii2 - repeat_radii2) / np.maximum(np.abs(radii2), 1e-12))
    )
    distance_relative = float(
        np.max(
            np.abs(distances2 - repeat_distances2)
            / np.maximum(np.abs(distances2), 1e-12)
        )
    )
    noise_floor = max(
        A2_ABSOLUTE_FLOOR,
        NOISE_FLOOR_MULTIPLIER * mean_js(baseline, repeated_baseline),
    )
    checks = {
        "radius_floor": bool(np.all(radii2 > noise_floor)),
        "symmetry": float(np.max(np.abs(distances2 - distances2.T))) <= A2_MATRIX_TOLERANCE,
        "diagonal": float(np.max(np.abs(np.diag(distances2)))) <= A2_MATRIX_TOLERANCE,
        "gram_psd": float(np.min(np.linalg.eigvalsh(gram))) >= -A2_PSD_TOLERANCE,
        "cosine_bounds": raw_min >= -1.0 - A2_COSINE_BOUND_TOLERANCE
        and raw_max <= 1.0 + A2_COSINE_BOUND_TOLERANCE,
        "repeat_radius": radius_relative <= A2_REPEAT_RELATIVE_TOLERANCE,
        "repeat_distance": distance_relative <= A2_REPEAT_RELATIVE_TOLERANCE,
        "repeat_angular_rank": rank >= A2_REPEAT_RANK_THRESHOLD,
    }
    a2 = 1.0 - np.clip(cosine, -1.0, 1.0)
    np.fill_diagonal(a2, 0.0)
    d2 = np.sqrt(np.maximum(distances2, 0.0))
    return a2, d2, {
        "checks": checks,
        "pass": bool(all(checks.values())),
        "noise_floor_squared": noise_floor,
        "gram_min_eigenvalue": float(np.min(np.linalg.eigvalsh(gram))),
        "cosine_range": [raw_min, raw_max],
        "repeat_radius_relative_error_max": radius_relative,
        "repeat_distance_relative_error_max": distance_relative,
        "repeat_angular_rank_correlation": rank,
    }


def block_matrices(matrix: np.ndarray, fresh_count: int) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:fresh_count, :fresh_count], matrix[:fresh_count, fresh_count:]


def consolidate(historical_dir: Path, fresh_dir: Path, output_dir: Path, workers: int) -> None:
    fresh_names, reference_names, coefficients, vectors = load_vectors()
    names = [*fresh_names, *reference_names]
    fresh_count = len(fresh_names)
    a0 = 1.0 - np.clip(coefficients @ coefficients.T, -1.0, 1.0)
    np.fill_diagonal(a0, 0.0)
    fit = load_fit()
    a1 = np.asarray(whitened_geometry(vectors, fit)["cosine_distance"], dtype=np.float64)
    a1_checks = {
        "fit_hash_matches": fit.fit_hash
        == read_json(V41 / "A1_INSTRUMENT_QUALIFICATION.json")["fit_hash"],
        "finite": bool(np.isfinite(a1).all()),
        "symmetry": float(np.max(np.abs(a1 - a1.T))) <= A1_MATRIX_TOLERANCE,
        "diagonal": float(np.max(np.abs(np.diag(a1)))) <= A1_MATRIX_TOLERANCE,
        "range": float(np.min(a1)) >= -A1_MATRIX_TOLERANCE
        and float(np.max(a1)) <= 2.0 + A1_MATRIX_TOLERANCE,
    }
    matrices: dict[str, np.ndarray] = {}
    for metric, matrix in (("A0", a0), ("A1", a1)):
        fresh_fresh, fresh_reference = block_matrices(matrix, fresh_count)
        for shell in SHELLS:
            matrices[f"{metric}_{shell}_FRESH_FRESH"] = fresh_fresh
            matrices[f"{metric}_{shell}_FRESH_REFERENCE"] = fresh_reference
    a2_reports: dict[str, Any] = {}
    historical_sealed = np.load(V41 / "PREDICTION_MATRICES.npz", allow_pickle=False)
    for shell in SHELLS:
        baseline, repeated_baseline, arrays, repeated, baseline_error = load_probe_archives(
            historical_dir, fresh_dir, shell, fresh_names, reference_names
        )
        a2, d2, report = a2_geometry(
            names, baseline, repeated_baseline, arrays, repeated, workers
        )
        fresh_fresh_a2, fresh_reference_a2 = block_matrices(a2, fresh_count)
        fresh_fresh_d2, fresh_reference_d2 = block_matrices(d2, fresh_count)
        matrices[f"A2_{shell}_FRESH_FRESH"] = fresh_fresh_a2
        matrices[f"A2_{shell}_FRESH_REFERENCE"] = fresh_reference_a2
        matrices[f"D2_{shell}_FRESH_FRESH"] = fresh_fresh_d2
        matrices[f"D2_{shell}_FRESH_REFERENCE"] = fresh_reference_d2
        old_slice = slice(fresh_count, len(names))
        historical_a2_error = float(
            np.max(np.abs(a2[old_slice, old_slice] - historical_sealed[f"A2_{shell}"]))
        )
        historical_d2_error = float(
            np.max(np.abs(d2[old_slice, old_slice] - historical_sealed[f"D2_{shell}"]))
        )
        report["fresh_historical_baseline_max_abs_difference"] = baseline_error
        report["historical_A2_reproduction_max_abs_difference"] = historical_a2_error
        report["historical_D2_reproduction_max_abs_difference"] = historical_d2_error
        report["checks"]["historical_A2_reproduction"] = (
            historical_a2_error <= HISTORICAL_MATRIX_TOLERANCE
        )
        report["checks"]["historical_D2_reproduction"] = (
            historical_d2_error <= HISTORICAL_MATRIX_TOLERANCE
        )
        report["pass"] = bool(all(report["checks"].values()))
        a2_reports[shell] = report
    historical_sealed.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "PREDICTION_MATRICES.npz"
    temporary = archive_path.with_name(archive_path.name + ".tmp.npz")
    np.savez_compressed(temporary, **matrices)
    os.replace(temporary, archive_path)
    a1_pass = bool(all(a1_checks.values()))
    a2_pass = bool(all(value["pass"] for value in a2_reports.values()))
    atomic_json(
        output_dir / "A1_INSTRUMENT_QUALIFICATION.json",
        {
            "classification": "Q2_OOS_V2_A1_INSTRUMENT_QUALIFIED"
            if a1_pass
            else "Q2_OOS_V2_A1_INSTRUMENT_NOT_QUALIFIED",
            "frozen_fit_sha256": sha256_file(V41 / "A1_COVARIANCE_FIT.npz"),
            "frozen_fit_hash": fit.fit_hash,
            "checks": a1_checks,
        },
    )
    atomic_json(
        output_dir / "A2_INSTRUMENT_QUALIFICATION.json",
        {
            "classification": "Q2_OOS_V2_A2_INSTRUMENT_QUALIFIED"
            if a2_pass
            else "Q2_OOS_V2_A2_INSTRUMENT_NOT_QUALIFIED",
            "arithmetic": {
                "log_base": "natural_log",
                "mixture_weights": [0.5, 0.5],
                "aggregation": "uniform_mean_over_48_probe_checkpoint_rows",
                "controller_order": "fresh_16_then_reference_31",
            },
            "shells": a2_reports,
        },
    )
    atomic_json(
        output_dir / "PREDICTION_MATRIX_METADATA.json",
        {
            "schema_version": "q2-oos-v2-prediction-matrix-metadata-v1",
            "fresh_controller_order": fresh_names,
            "reference_controller_order": reference_names,
            "matrix_archive_sha256": sha256_file(archive_path),
            "matrix_hashes": {name: array_hash(value) for name, value in matrices.items()},
            "semantic_outcomes": 0,
            "correctness_inspected": False,
        },
    )
    classification = (
        "Q2_OOS_V2_LABEL_FREE_INSTRUMENT_QUALIFIED"
        if a1_pass and a2_pass
        else "Q2_OOS_V2_LABEL_FREE_INSTRUMENT_NOT_QUALIFIED"
    )
    atomic_json(
        output_dir / "LABEL_FREE_QUALIFICATION.json",
        {
            "classification": classification,
            "A0": "QUALIFIED_BY_SELECTED_BANK_GATE",
            "A1": "Q2_OOS_V2_A1_INSTRUMENT_QUALIFIED" if a1_pass else "NOT_QUALIFIED",
            "A2": "Q2_OOS_V2_A2_INSTRUMENT_QUALIFIED" if a2_pass else "NOT_QUALIFIED",
            "D2": "SECONDARY_SEALED" if a2_pass else "NOT_QUALIFIED",
            "prediction_matrices_sha256": sha256_file(archive_path),
            "semantic_outcomes": 0,
            "correctness_inspected": False,
        },
    )
    print(json.dumps({"classification": classification, "semantic_outcomes": 0}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-raw-dir", required=True, type=Path)
    parser.add_argument("--fresh-raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    consolidate(args.historical_raw_dir, args.fresh_raw_dir, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
