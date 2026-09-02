#!/usr/bin/env python3
"""Independent read-only forensic audit of Q2 OOS V2 label-free geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_geometries import WhitenedFit  # noqa: E402

REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_presemantic_closeout"
V2_STREAM = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"
V41 = ROOT / "review/q2_v4_1_prediction_lock"
V41_SAFE = ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
V41_VECTOR_DIR = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_DIRECTIONS"
SHELLS = ("MEDIUM", "STRONG")
TOLERANCE = 1e-10


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scalar_mean_js(left: np.ndarray, right: np.ndarray) -> float:
    """Row-loop natural-log JS reference, separate from the primary reducer."""

    values: list[float] = []
    for left_row, right_row in zip(left, right, strict=True):
        a = np.asarray(left_row, dtype=np.float64)
        b = np.asarray(right_row, dtype=np.float64)
        a -= np.logaddexp.reduce(a)
        b -= np.logaddexp.reduce(b)
        mixture = np.logaddexp(a, b) - np.log(2.0)
        values.append(
            float(
                0.5 * np.sum(np.exp(a) * (a - mixture))
                + 0.5 * np.sum(np.exp(b) * (b - mixture))
            )
        )
    return float(sum(values) / len(values))


def load_fit() -> WhitenedFit:
    qualification = read_json(V41 / "A1_INSTRUMENT_QUALIFICATION.json")
    with np.load(V41 / "A1_COVARIANCE_FIT.npz", allow_pickle=False) as archive:
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


def identities_and_vectors() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    selected = read_json(REVIEW / "V2_SELECTED_CONTROLLER_BANK.json")
    stream = read_json(V2_STREAM / "V2_CANDIDATE_BANK_MANIFEST.json")
    rows = {row["candidate_id"]: row for row in stream["candidates"]}
    fresh = [str(value) for value in selected["selected_ids"]]
    historical = read_json(V41_SAFE)
    reference = [str(row["candidate_id"]) for row in historical["directions"]]
    coefficients = np.concatenate(
        [
            np.asarray([rows[name]["coefficients"] for name in fresh], dtype=np.float64),
            np.asarray([row["coefficients"] for row in historical["directions"]], dtype=np.float64),
        ]
    )
    vectors = np.concatenate(
        [
            np.stack(
                [
                    np.load(ROOT / rows[name]["path"], allow_pickle=False).astype(np.float64)
                    for name in fresh
                ]
            ),
            np.stack(
                [
                    np.load(V41_VECTOR_DIR / f"{name}.npy", allow_pickle=False).astype(np.float64)
                    for name in reference
                ]
            ),
        ]
    )
    return fresh, reference, coefficients, vectors


def independent_a1(vectors: np.ndarray, fit: WhitenedFit) -> np.ndarray:
    ridge = fit.regularization_value
    projected = vectors @ fit.right_singular_vectors.T
    adjusted = (1.0 - fit.regularization_fraction) * fit.eigenvalues + ridge
    gram = (vectors @ vectors.T) / ridge
    gram += (projected * ((1.0 / adjusted) - (1.0 / ridge))[None, :]) @ projected.T
    gram = 0.5 * (gram + gram.T)
    norms = np.sqrt(np.diag(gram))
    result = 1.0 - np.clip(gram / np.outer(norms, norms), -1.0, 1.0)
    np.fill_diagonal(result, 0.0)
    return result


def load_condition(
    directory: Path,
    subdir: str,
    name: str,
    shell: str | None,
) -> np.ndarray:
    probes = read_json(V41 / "A2_PROBE_MANIFEST.json")["item_ids"]
    values = []
    key = "BASELINE" if shell is None else f"{name}_{shell}"
    for probe in probes:
        with np.load(directory / subdir / f"{probe}.npz", allow_pickle=False) as archive:
            values.append(np.asarray(archive[key], dtype=np.float64))
    return np.concatenate(values, axis=0)


def audit_a2_subset(
    matrices: Any,
    historical_dir: Path,
    fresh_dir: Path,
    fresh: list[str],
    reference: list[str],
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    maximum = 0.0
    pairs = [
        (fresh[0], fresh[1], "FRESH_FRESH", 0, 1),
        *[
            (fresh[0], reference[index], "FRESH_REFERENCE", 0, index)
            for index in range(3)
        ],
    ]
    for shell in SHELLS:
        baseline = load_condition(historical_dir, "A2_FINGERPRINTS", "", None)
        for left, right, block, row, column in pairs:
            left_values = load_condition(
                fresh_dir, "A2_FRESH_FINGERPRINTS", left, shell
            )
            if block == "FRESH_FRESH":
                right_values = load_condition(
                    fresh_dir, "A2_FRESH_FINGERPRINTS", right, shell
                )
            else:
                right_values = load_condition(
                    historical_dir, "A2_FINGERPRINTS", right, shell
                )
            left_radius = scalar_mean_js(left_values, baseline)
            right_radius = scalar_mean_js(right_values, baseline)
            squared_distance = scalar_mean_js(left_values, right_values)
            cosine = (left_radius + right_radius - squared_distance) / (
                2.0 * np.sqrt(left_radius * right_radius)
            )
            observed_a2 = float(matrices[f"A2_{shell}_{block}"][row, column])
            observed_d2 = float(matrices[f"D2_{shell}_{block}"][row, column])
            a2_error = abs((1.0 - cosine) - observed_a2)
            d2_error = abs(np.sqrt(max(squared_distance, 0.0)) - observed_d2)
            maximum = max(maximum, a2_error, d2_error)
            records.append(
                {
                    "shell": shell,
                    "block": block,
                    "pair_index": [row, column],
                    "A2_absolute_difference": a2_error,
                    "D2_absolute_difference": d2_error,
                }
            )
    return records, maximum


def audit(historical_dir: Path, fresh_dir: Path, output_dir: Path) -> None:
    metadata = read_json(output_dir / "PREDICTION_MATRIX_METADATA.json")
    fresh, reference, coefficients, vectors = identities_and_vectors()
    with np.load(output_dir / "PREDICTION_MATRICES.npz", allow_pickle=False) as matrices:
        hash_checks = {
            name: array_hash(np.asarray(matrices[name])) == expected
            for name, expected in metadata["matrix_hashes"].items()
        }
        unit = coefficients / np.linalg.norm(coefficients, axis=1)[:, None]
        a0 = 1.0 - np.clip(unit @ unit.T, -1.0, 1.0)
        np.fill_diagonal(a0, 0.0)
        a1 = independent_a1(vectors, load_fit())
        fresh_count = len(fresh)
        a0_difference = 0.0
        a1_difference = 0.0
        for shell in SHELLS:
            a0_difference = max(
                a0_difference,
                float(
                    np.max(
                        np.abs(
                            matrices[f"A0_{shell}_FRESH_FRESH"]
                            - a0[:fresh_count, :fresh_count]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            matrices[f"A0_{shell}_FRESH_REFERENCE"]
                            - a0[:fresh_count, fresh_count:]
                        )
                    )
                ),
            )
            a1_difference = max(
                a1_difference,
                float(
                    np.max(
                        np.abs(
                            matrices[f"A1_{shell}_FRESH_FRESH"]
                            - a1[:fresh_count, :fresh_count]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            matrices[f"A1_{shell}_FRESH_REFERENCE"]
                            - a1[:fresh_count, fresh_count:]
                        )
                    )
                ),
            )
        subset, a2_difference = audit_a2_subset(
            matrices, historical_dir, fresh_dir, fresh, reference
        )
    qualification = read_json(output_dir / "LABEL_FREE_QUALIFICATION.json")
    a2_qualification = read_json(output_dir / "A2_INSTRUMENT_QUALIFICATION.json")
    checks = {
        "all_matrix_hashes": bool(all(hash_checks.values())),
        "A0_independent_full_recompute": a0_difference <= TOLERANCE,
        "A1_independent_full_recompute": a1_difference <= TOLERANCE,
        "A2_D2_scalar_subset_reference": a2_difference <= TOLERANCE,
        "primary_label_free_qualified": qualification["classification"]
        == "Q2_OOS_V2_LABEL_FREE_INSTRUMENT_QUALIFIED",
        "all_repeat_and_historical_checks": all(
            all(shell["checks"].values()) for shell in a2_qualification["shells"].values()
        ),
        "semantic_outcomes_zero": qualification["semantic_outcomes"] == 0,
        "correctness_not_inspected": qualification["correctness_inspected"] is False,
    }
    # Comparisons involving NumPy scalars can yield ``np.bool_``.  Normalize only
    # the report boundary so the frozen audit mathematics remains unchanged and
    # the result is portable through the standard-library JSON encoder.
    checks = {name: bool(value) for name, value in checks.items()}
    classification = (
        "Q2_OOS_V2_LABEL_FREE_FORENSIC_CLEAN"
        if all(checks.values())
        else "Q2_OOS_V2_LABEL_FREE_FORENSIC_DISAGREEMENT"
    )
    atomic_json(
        output_dir / "LABEL_FREE_FORENSIC_AUDIT.json",
        {
            "schema_version": "q2-oos-v2-label-free-forensic-v1",
            "classification": classification,
            "checks": checks,
            "matrix_archive_sha256": sha256_file(output_dir / "PREDICTION_MATRICES.npz"),
            "A0_maximum_absolute_difference": a0_difference,
            "A1_maximum_absolute_difference": a1_difference,
            "A2_D2_scalar_subset_maximum_absolute_difference": a2_difference,
            "A2_D2_scalar_subset": subset,
            "maximum_primary_audit_difference": max(
                a0_difference, a1_difference, a2_difference
            ),
            "raw_text_inspected": False,
            "correctness_inspected": False,
            "semantic_outcomes": 0,
        },
    )
    print(json.dumps({"classification": classification}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-raw-dir", required=True, type=Path)
    parser.add_argument("--fresh-raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    audit(args.historical_raw_dir, args.fresh_raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
