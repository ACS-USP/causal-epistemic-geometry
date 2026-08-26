"""Outcome-free geometry and planning primitives for Q2 V4.1.

This module intentionally has no model, benchmark, parser, semantic-outcome,
or GPU imports. It audits the already frozen V4 candidate coefficient bank
and runs only synthetic planning calculations.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

V4_PRELOCK = "99782d6f4f3ce1ca52d2cf6caeacafd4d0de9081"
V4_CANDIDATE_COMMIT = "c82c1cb79392f9a5d9bd9e8d258a1d1b54e8fd41"
V4_FINAL_COMMIT = "186a6b6d5b81ba26aea9fe607dabe8eaae6ff0e1"
V4_CLASSIFICATION = "Q2_V4_SAFE_BANK_INSUFFICIENT"
EXPECTED_SAFE_IDS = (
    "V4_DIRECTION_00",
    "V4_DIRECTION_01",
    "V4_DIRECTION_02",
    "V4_DIRECTION_03",
    "V4_DIRECTION_04",
    "V4_DIRECTION_06",
    "V4_DIRECTION_07",
    "V4_DIRECTION_08",
    "V4_DIRECTION_09",
    "V4_DIRECTION_10",
    "V4_DIRECTION_11",
    "V4_DIRECTION_13",
    "V4_DIRECTION_15",
    "V4_DIRECTION_17",
    "V4_DIRECTION_18",
    "V4_DIRECTION_19",
    "V4_DIRECTION_20",
    "V4_DIRECTION_22",
    "V4_DIRECTION_23",
    "V4_DIRECTION_24",
    "V4_DIRECTION_26",
    "V4_DIRECTION_28",
    "V4_DIRECTION_29",
    "V4_DIRECTION_30",
    "V4_DIRECTION_31",
    "V4_DIRECTION_32",
    "V4_DIRECTION_33",
    "V4_DIRECTION_34",
    "V4_DIRECTION_35",
    "V4_DIRECTION_37",
    "V4_DIRECTION_39",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(singular_values)
    probabilities = energy / np.sum(energy)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _upper(values: np.ndarray) -> np.ndarray:
    return values[np.triu_indices(len(values), 1)]


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "q10": None,
            "q50": None,
            "q90": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "q10": float(np.quantile(values, 0.10)),
        "q50": float(np.quantile(values, 0.50)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def bank_geometry(coefficients: np.ndarray) -> dict[str, Any]:
    """Return the V4 bank diagnostics, generalized only over bank cardinality."""

    values = _finite_matrix(coefficients, name="coefficients")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0.0):
        raise ValueError("coefficient rows must have positive norms")
    unit = values / norms[:, None]
    singular = np.linalg.svd(unit, compute_uv=False)
    rank = int(np.linalg.matrix_rank(unit, tol=1e-10))
    gram = unit @ unit.T
    pair_cos = _upper(gram)
    pair_abs_cos = np.abs(pair_cos)
    a0_distance = 1.0 - gram
    a0_upper = _upper(a0_distance)
    angular = np.arccos(np.clip(gram, -1.0, 1.0))
    nearest = np.asarray([np.min(np.delete(angular[index], index)) for index in range(len(unit))])
    # H is the row-space leverage for the coefficient observations.
    leverage = np.diag(unit @ np.linalg.pinv(unit.T @ unit, hermitian=True) @ unit.T)
    coordinate_signs = {
        str(index): {
            "negative": int(np.sum(unit[:, index] < 0.0)),
            "zero": int(np.sum(unit[:, index] == 0.0)),
            "positive": int(np.sum(unit[:, index] > 0.0)),
        }
        for index in range(unit.shape[1])
    }
    axis = {}
    for index in range(unit.shape[1]):
        coordinate = unit[:, index]
        axis[str(index)] = {
            "mean": float(np.mean(coordinate)),
            "variance": float(np.var(coordinate, ddof=1)),
            "min": float(np.min(coordinate)),
            "max": float(np.max(coordinate)),
            "mean_absolute": float(np.mean(np.abs(coordinate))),
            "q10_absolute": float(np.quantile(np.abs(coordinate), 0.10)),
            "signs": coordinate_signs[str(index)],
        }
    return {
        "count": int(len(unit)),
        "dimension": int(unit.shape[1]),
        "row_norm_max_error": float(np.max(np.abs(norms - 1.0))),
        "singular_values": singular.tolist(),
        "relative_singular_values": (singular / singular[0]).tolist(),
        "rank": rank,
        "effective_rank": _effective_rank(singular),
        "stable_rank": float(np.sum(np.square(singular)) / singular[0] ** 2),
        "condition_number": float(singular[0] / singular[-1]),
        "pairwise_cosine": _summary(pair_cos),
        "pairwise_absolute_cosine": _summary(pair_abs_cos),
        "a0_angular_chord_squared": _summary(a0_upper),
        "nearest_neighbor_angle_radians": _summary(nearest),
        "nearest_neighbor_angle_degrees": _summary(np.degrees(nearest)),
        "max_nearest_neighbor_angle_degrees": float(np.max(np.degrees(nearest))),
        "leverage": _summary(leverage),
        "leverage_values": leverage.tolist(),
        "mean_direction": unit.mean(axis=0).tolist(),
        "mean_direction_norm": float(np.linalg.norm(unit.mean(axis=0))),
        "axis": axis,
        "axis_mean_abs_max": float(np.max(np.abs(np.mean(unit, axis=0)))),
        "axis_mean_abs_min": float(np.min(np.abs(np.mean(unit, axis=0)))),
        "gram_symmetry_error": float(np.max(np.abs(gram - gram.T))),
        "coordinate_space": "row-normalized frozen V4 coefficients",
        "coverage_proxy": (
            "maximum nearest-neighbor angular distance; no canonical spherical "
            "covering radius is asserted in dimension 8"
        ),
    }


def selected_bank_coverage_checks(
    coefficients: np.ndarray,
    amplitudes: np.ndarray,
    *,
    expected_count: int = 31,
) -> dict[str, Any]:
    """Apply V4 selected-bank checks with only K generalized to 31."""

    values = _finite_matrix(coefficients, name="coefficients")
    amplitudes_array = _finite_matrix(amplitudes, name="amplitudes")
    if amplitudes_array.shape[0] != len(values):
        raise ValueError("amplitudes must have one row per controller")
    diagnostics = bank_geometry(values)
    shell_cvs = np.std(amplitudes_array, axis=0) / np.mean(amplitudes_array, axis=0)
    checks = {
        "selected_count_31": len(values) == expected_count,
        "full_subspace_rank": diagnostics["rank"] == values.shape[1],
        "entropy_effective_rank_at_least_0_75r": diagnostics["effective_rank"]
        >= 0.75 * values.shape[1],
        "condition_number_at_most_3": diagnostics["condition_number"] <= 3.0,
        "max_absolute_pair_cosine_below_0_98": diagnostics["pairwise_absolute_cosine"]["max"]
        < 0.98,
        "a0_q90_q10_at_least_0_20": diagnostics["a0_angular_chord_squared"]["q90"]
        - diagnostics["a0_angular_chord_squared"]["q10"]
        >= 0.20,
        "shell_amplitude_cv_at_most_0_03": float(np.max(shell_cvs)) <= 0.03,
    }
    return {
        "diagnostics": diagnostics,
        "shell_amplitude_cv": shell_cvs.tolist(),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "lineage": "V4 selected_bank_checks; only cardinality generalized from 32 to 31",
    }


def load_frozen_candidates(
    candidate_manifest: Any,
    safety_report: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate and return all candidates and the immutable 31-safe subset."""

    import json

    with open(candidate_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(safety_report, encoding="utf-8") as handle:
        safety = json.load(handle)
    candidates = manifest["candidates"]
    if len(candidates) != 40 or safety["safe_count"] != 31:
        raise ValueError("historical V4 candidate/safe counts do not match")
    if safety["classification"] != V4_CLASSIFICATION:
        raise ValueError("historical V4 classification changed")
    if safety["correctness_used"]:
        raise ValueError("safety artifact unexpectedly used correctness")
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if tuple(safety["selected_first_32_safe"]) != EXPECTED_SAFE_IDS:
        raise ValueError("frozen safe order does not match the historical V4 artifact")
    rows: list[dict[str, Any]] = []
    safe_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        candidate_safety = safety["candidates"][candidate_id]
        row = {
            "candidate_id": candidate_id,
            "generation_index": candidate["generation_index"],
            "coefficients": candidate["coefficients"],
            "canonical_vector_hash": candidate["canonical_vector_hash"],
            "file_sha256": candidate["file_sha256"],
            "joint_safe": bool(candidate_safety["both_shells_pass"]),
            "medium": candidate_safety["shells"]["MEDIUM"],
            "strong": candidate_safety["shells"]["STRONG"],
            "failure_reasons": {
                shell: {
                    key: value
                    for key, value in shell_data.items()
                    if key
                    in {
                        "pass",
                        "validity",
                        "evaluability",
                        "truncation",
                        "raw_sequence_movement",
                    }
                }
                for shell, shell_data in candidate_safety["shells"].items()
            },
        }
        rows.append(row)
        if row["joint_safe"]:
            safe_rows.append(row)
    if len(safe_rows) != 31 or [row["candidate_id"] for row in safe_rows] != list(
        EXPECTED_SAFE_IDS
    ):
        raise ValueError("safe subset reconstruction failed")
    if set(by_id) != {row["candidate_id"] for row in rows}:
        raise ValueError("candidate IDs do not reconcile")
    return rows, safe_rows


def safety_structure(
    coefficients: np.ndarray,
    safe_mask: np.ndarray,
    *,
    permutations: int = 10_000,
    bootstrap: int = 5_000,
    seed: int = 2026082601,
) -> dict[str, Any]:
    """Descriptive uncertainty for whether safety labels carve coefficient space."""

    values = _finite_matrix(coefficients, name="coefficients")
    mask = np.asarray(safe_mask, dtype=bool)
    if mask.shape != (len(values),) or np.sum(mask) < 2 or np.sum(~mask) < 2:
        raise ValueError("safety mask needs both groups")
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    safe_mean = np.mean(unit[mask], axis=0)
    unsafe_mean = np.mean(unit[~mask], axis=0)
    observed = float(np.linalg.norm(safe_mean - unsafe_mean))
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    permutation_values = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = rng.permutation(mask)
        permutation_values[index] = np.linalg.norm(
            np.mean(unit[shuffled], axis=0) - np.mean(unit[~shuffled], axis=0)
        )
    p_value = float((1 + np.sum(permutation_values >= observed)) / (permutations + 1))
    bootstrap_values = np.empty(bootstrap, dtype=np.float64)
    safe_indices = np.flatnonzero(mask)
    unsafe_indices = np.flatnonzero(~mask)
    for index in range(bootstrap):
        sampled_safe = rng.choice(safe_indices, size=len(safe_indices), replace=True)
        sampled_unsafe = rng.choice(unsafe_indices, size=len(unsafe_indices), replace=True)
        bootstrap_values[index] = np.linalg.norm(
            np.mean(unit[sampled_safe], axis=0) - np.mean(unit[sampled_unsafe], axis=0)
        )
    coordinate_associations = []
    y = mask.astype(np.float64)
    for coordinate in unit.T:
        coordinate_associations.append(
            {
                "point_biserial": float(np.corrcoef(coordinate, y)[0, 1]),
                "safe_mean": float(np.mean(coordinate[mask])),
                "unsafe_mean": float(np.mean(coordinate[~mask])),
            }
        )
    return {
        "safe_count": int(np.sum(mask)),
        "unsafe_count": int(np.sum(~mask)),
        "safe_mean": safe_mean.tolist(),
        "unsafe_mean": unsafe_mean.tolist(),
        "centroid_difference_norm": observed,
        "centroid_difference_bootstrap_95": [
            float(np.quantile(bootstrap_values, 0.025)),
            float(np.quantile(bootstrap_values, 0.975)),
        ],
        "permutation": {
            "statistic": "safe-versus-unsafe centroid Euclidean distance",
            "maps": permutations,
            "p_value_plus_one": p_value,
            "null_q95": float(np.quantile(permutation_values, 0.95)),
        },
        "coordinate_associations": coordinate_associations,
        "model": "descriptive linear logistic separation only; no prediction or selection",
        "seed": seed,
    }


def binomial_safe_probability(candidate_count: int, p_safe: float, minimum: int = 32) -> float:
    if candidate_count < 0 or not 0.0 <= p_safe <= 1.0:
        raise ValueError("invalid binomial inputs")
    return float(
        sum(
            math.comb(candidate_count, count)
            * p_safe**count
            * (1.0 - p_safe) ** (candidate_count - count)
            for count in range(minimum, candidate_count + 1)
        )
    )


def reserve_fragility(
    probabilities: Sequence[float] = (0.70, 0.75, 0.775, 0.80, 0.85, 0.90),
    candidate_count: int = 40,
    minimum: int = 32,
) -> dict[str, Any]:
    rows = []
    for probability in probabilities:
        needed = None
        for count in range(minimum, 201):
            if binomial_safe_probability(count, probability, minimum) >= 0.95:
                needed = count
                break
        rows.append(
            {
                "p_safe": probability,
                "candidate_count": candidate_count,
                "probability_at_least_minimum": binomial_safe_probability(
                    candidate_count, probability, minimum
                ),
                "minimum_candidates_for_95pct": needed,
            }
        )
    return {
        "minimum_safe": minimum,
        "historical_candidate_count": candidate_count,
        "rows": rows,
        "interpretation": "planning only; no candidates generated",
    }


def synthetic_adequacy_criteria() -> dict[str, Any]:
    """Frozen before observed 31-bank PASS/FAIL application."""

    return {
        "status": "FROZEN_BEFORE_OBSERVED_BANK_APPLICATION",
        "coverage": {
            "lineage": (
                "V4 selected_bank_checks with only selected cardinality "
                "generalized 32 -> 31"
            ),
            "rank": "full retained coefficient-space rank",
            "effective_rank_fraction_min": 0.75,
            "condition_number_max": 3.0,
            "max_absolute_pair_cosine_max_exclusive": 0.98,
            "a0_q90_minus_q10_min": 0.20,
            "shell_amplitude_cv_max": 0.03,
            "cardinality": 31,
            "descriptive_only": [
                "axis loading/sign balance",
                "mean-direction anisotropy",
                "leverage",
                "nearest-neighbor angular coverage",
                "safety-label coefficient-space structure",
            ],
        },
        "power": {
            "primary_regime": "K=31, N=300",
            "reference_regime": "K=32, N=300",
            "scientifically_meaningful_rho": 0.25,
            "omnibus_power_min": 0.80,
            "omnibus_fpr_range": [0.025, 0.075],
            "relative_omnibus_power_loss_max": 0.10,
            "ci_width_ratio_max": 1.10,
            "a2_attribution": (
                "reported and compared to K=32; no separate binary gate because "
                "V4 explicitly documented marginal superiority limitations"
            ),
            "g3_superiority": "reported honestly, not used as a K=31 adequacy gate",
        },
        "decision": {
            "adequate_if": [
                "all coverage checks pass",
                "K31/N300 omnibus power >= 0.80 at rho=0.25",
                "K31/N300 omnibus FPR lies in [0.025, 0.075]",
                "K31/N300 power loss versus K32/N300 <= 0.10 absolute",
                "K31/N300 planning CI width ratio versus K32/N300 <= 1.10",
                "no gross safety-conditioned collapse is present; diagnostics are "
                "interpreted jointly, not by aesthetic spherical uniformity",
            ],
            "inadequate_if": "any primary adequacy requirement fails",
            "no_semantic_outcomes": True,
        },
    }


__all__ = [
    "EXPECTED_SAFE_IDS",
    "V4_CANDIDATE_COMMIT",
    "V4_CLASSIFICATION",
    "V4_FINAL_COMMIT",
    "V4_PRELOCK",
    "bank_geometry",
    "binomial_safe_probability",
    "load_frozen_candidates",
    "reserve_fragility",
    "safety_structure",
    "selected_bank_coverage_checks",
    "sha256_file",
    "synthetic_adequacy_criteria",
]
