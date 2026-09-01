#!/usr/bin/env python3
"""CPU-only adequacy audit for Q2 OOS V1 gates and prospective V2 design.

This script reads only public controller coefficients, presemantic safety
labels, and frozen aggregate planning inputs. It never loads benchmark text,
item-level semantic outcomes, correctness, or model artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_oos_fresh_controller import (
    angular_cross_block,
    fresh_row_permutations,
    spearman_flat,
)
from epistemic_geometry.experiments.q2_v4 import average_ranks

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design"
OUT = REVIEW / "v2_gate_audit"
PRECHECK = OUT / "AUDIT_PRECHECK.json"
V1_COMMIT = "249543e044f3d07713ac90dc6b68988e237f5119"
COUNTS = (12, 16, 19, 22, 24, 28, 32, 36, 40, 48, 56, 64)
P_SAFE = (0.60, 0.65, 0.70, 0.75, 0.775, 0.80, 0.85, 0.90)
K = 10
DIMENSION = 8
REFERENCE_A0_RHO = 0.5638183484033006


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_sphere(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    values = rng.standard_normal(shape)
    return values / np.linalg.norm(values, axis=-1, keepdims=True)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return float("nan"), float("nan")
    p = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (p + z**2 / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z**2 / (4.0 * trials**2)) / denominator
    return center - half, center + half


def binomial_tail(n: int, p: float, threshold: int = K) -> float:
    return float(
        sum(math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(threshold, n + 1))
    )


def batched_metrics(values: np.ndarray) -> dict[str, np.ndarray]:
    scatter = np.einsum("bni,bnj->bij", values, values)
    eigen = np.linalg.eigvalsh(scatter)
    eigen = np.maximum(eigen, 0.0)
    probabilities = eigen / np.sum(eigen, axis=1, keepdims=True)
    effective = np.exp(
        -np.sum(np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0), axis=1)
    )
    condition = np.sqrt(eigen[:, -1] / eigen[:, 0])
    stable = np.sum(eigen, axis=1) / eigen[:, -1]
    gram = np.einsum("bni,bmi->bnm", values, values)
    diagonal = np.arange(values.shape[1])
    gram[:, diagonal, diagonal] = 0.0
    maximum_cosine = np.max(np.abs(gram), axis=(1, 2))
    rank = np.sum(eigen > 1e-20, axis=1)
    return {
        "rank": rank,
        "effective_rank": effective,
        "stable_rank": stable,
        "condition_number": condition,
        "maximum_absolute_pair_cosine": maximum_cosine,
    }


def candidate_pass(metrics: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (metrics["rank"] == 8)
        & (metrics["effective_rank"] >= 6.0)
        & (metrics["condition_number"] <= 3.0)
        & (metrics["maximum_absolute_pair_cosine"] < 0.98)
    )


def summarize_distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.quantile(values, (0.05, 0.25, 0.50, 0.75, 0.95))
    return {
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
    }


def algebraic_calibration(
    replicates: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows: list[dict[str, Any]] = []
    v1_percentiles: dict[str, Any] = {}
    realized = read_json(REVIEW / "CANDIDATE_STREAM_CLOSEOUT.json")["candidate_stream"]
    for n in COUNTS:
        collected: dict[str, list[np.ndarray]] = {
            name: []
            for name in (
                "rank",
                "effective_rank",
                "stable_rank",
                "condition_number",
                "maximum_absolute_pair_cosine",
                "pass",
            )
        }
        batch_size = 1000
        for start in range(0, replicates, batch_size):
            size = min(batch_size, replicates - start)
            values = unit_sphere(rng, (size, n, DIMENSION))
            metrics = batched_metrics(values)
            for name, metric in metrics.items():
                collected[name].append(metric)
            collected["pass"].append(candidate_pass(metrics))
        merged = {name: np.concatenate(parts) for name, parts in collected.items()}
        passed = merged["pass"].astype(bool)
        lower, upper = wilson(int(np.sum(passed)), replicates)
        row = {
            "candidate_count": n,
            "replicates": replicates,
            "rank_8_probability": float(np.mean(merged["rank"] == 8)),
            "effective_rank_ge_6_probability": float(np.mean(merged["effective_rank"] >= 6.0)),
            "condition_number_le_3_probability": float(np.mean(merged["condition_number"] <= 3.0)),
            "max_pair_cosine_lt_0_98_probability": float(
                np.mean(merged["maximum_absolute_pair_cosine"] < 0.98)
            ),
            "joint_candidate_gate_probability": float(np.mean(passed)),
            "joint_candidate_gate_ci95_low": lower,
            "joint_candidate_gate_ci95_high": upper,
            "joint_candidate_gate_mc_se": float(
                math.sqrt(np.mean(passed) * (1.0 - np.mean(passed)) / replicates)
            ),
            **summarize_distribution(merged["effective_rank"], "effective_rank"),
            **summarize_distribution(merged["condition_number"], "condition_number"),
            **summarize_distribution(
                merged["maximum_absolute_pair_cosine"], "maximum_absolute_pair_cosine"
            ),
        }
        rows.append(row)
        if n == 19:
            v1_percentiles = {
                "replicates": replicates,
                "realized": realized,
                "effective_rank_cdf_percentile": float(
                    np.mean(merged["effective_rank"] <= realized["effective_rank"])
                ),
                "condition_number_cdf_percentile": float(
                    np.mean(merged["condition_number"] <= realized["condition_number"])
                ),
                "condition_number_upper_tail": float(
                    np.mean(merged["condition_number"] >= realized["condition_number"])
                ),
                "maximum_pair_cosine_cdf_percentile": float(
                    np.mean(
                        merged["maximum_absolute_pair_cosine"]
                        <= realized["maximum_absolute_pair_cosine"]
                    )
                ),
                "probability_fail_effective_and_condition": float(
                    np.mean((merged["effective_rank"] < 6.0) & (merged["condition_number"] > 3.0))
                ),
                "probability_any_candidate_gate_failure": float(np.mean(~passed)),
                "interpretation": (
                    "model-free position of the immutable V1 stream; "
                    "not a semantic outcome"
                ),
            }
    return rows, v1_percentiles


def fit_historical_safety(seed: int, bootstrap_replicates: int = 10000) -> dict[str, Any]:
    manifest = read_json(ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json")
    safety = read_json(ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json")
    coefficients = np.asarray([row["coefficients"] for row in manifest["candidates"]])
    labels = np.asarray(
        [
            safety["candidates"][row["candidate_id"]]["both_shells_pass"]
            for row in manifest["candidates"]
        ],
        dtype=np.float64,
    )
    coordinate = coefficients[:, 4]
    mean = float(np.mean(coordinate))
    standard_deviation = float(np.std(coordinate))
    standardized = (coordinate - mean) / standard_deviation

    def fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        beta = np.asarray([math.log(np.mean(y) / (1.0 - np.mean(y))), 0.0])
        for _ in range(100):
            probability = 1.0 / (1.0 + np.exp(-(design @ beta)))
            weight = probability * (1.0 - probability)
            information = design.T @ (weight[:, None] * design) + np.eye(2) * 1e-8
            step = np.linalg.solve(information, design.T @ (y - probability))
            beta += step
            if np.max(np.abs(step)) < 1e-10:
                break
        return beta

    estimate = fit(standardized, labels)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    slopes = []
    for _ in range(bootstrap_replicates):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled = labels[indices]
        if np.all(sampled == sampled[0]):
            continue
        try:
            value = fit(standardized[indices], sampled)[1]
        except (np.linalg.LinAlgError, FloatingPointError):
            continue
        if np.isfinite(value) and abs(value) < 20.0:
            slopes.append(float(value))
    structure = read_json(ROOT / "review/q2_v4_1_31_safe_bank_review/SAFETY_STRUCTURE.json")
    return {
        "historical_n": 40,
        "safe_count": int(np.sum(labels)),
        "safe_rate": float(np.mean(labels)),
        "selected_coordinate_zero_based": 4,
        "selection_reason": "largest absolute historical coordinate point-biserial association",
        "coordinate_mean": mean,
        "coordinate_standard_deviation": standard_deviation,
        "logistic_intercept": float(estimate[0]),
        "logistic_slope": float(estimate[1]),
        "bootstrap_slope_q025": float(np.quantile(slopes, 0.025)),
        "bootstrap_slope_q975": float(np.quantile(slopes, 0.975)),
        "bootstrap_valid_replicates": len(slopes),
        "centroid_permutation_p": structure["permutation"]["p_value_plus_one"],
        "interpretation": (
            "small presemantic sample; weak/moderate sensitivity scenarios only; "
            "not a controller-selection model"
        ),
    }


def solve_intercept(coordinate: np.ndarray, slope: float, target: float) -> float:
    low, high = -20.0, 20.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        probability = 1.0 / (1.0 + np.exp(-(middle + slope * coordinate)))
        if float(np.mean(probability)) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def selected_metrics(values: np.ndarray, reference: np.ndarray) -> dict[str, np.ndarray]:
    metrics = batched_metrics(values)
    a0 = 1.0 - np.einsum("bki,ri->bkr", values, reference)
    flattened = a0.reshape(len(values), -1)
    q10, q90 = np.quantile(flattened, (0.10, 0.90), axis=1)
    centered = a0 - np.mean(a0, axis=2, keepdims=True)
    row_gram = np.einsum("bki,bji->bkj", centered, centered)
    squared_norm = np.diagonal(row_gram, axis1=1, axis2=2)
    squared_distance = squared_norm[:, :, None] + squared_norm[:, None, :] - 2.0 * row_gram
    upper = np.triu_indices(K, 1)
    row_diversity = np.mean(
        np.sqrt(np.maximum(squared_distance[:, upper[0], upper[1]], 0.0)), axis=1
    )
    return {
        **metrics,
        "a0_q90_minus_q10": q90 - q10,
        "row_diversity_mean": row_diversity,
    }


def selected_pass(metrics: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (metrics["rank"] == 8)
        & (metrics["effective_rank"] >= 4.8)
        & (metrics["condition_number"] <= 10.0)
        & (metrics["maximum_absolute_pair_cosine"] < 0.98)
        & (metrics["a0_q90_minus_q10"] >= 0.20)
    )


def first_safe(values: np.ndarray, safe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe_count = np.sum(safe, axis=1)
    eligible = safe_count >= K
    positions = np.broadcast_to(np.arange(values.shape[1]), safe.shape)
    indices = np.sort(np.where(safe, positions, values.shape[1]), axis=1)[:, :K]
    selected = values[eligible][np.arange(np.sum(eligible))[:, None], indices[eligible]]
    return selected, eligible


def selected_bank_calibration(
    replicates: int,
    seed: int,
    reference: np.ndarray,
    safety_model: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    rows: list[dict[str, Any]] = []
    population = unit_sphere(rng, (250000, DIMENSION))
    coordinate_population = (population[:, 4] - safety_model["coordinate_mean"]) / safety_model[
        "coordinate_standard_deviation"
    ]
    slopes = {
        "INDEPENDENT": 0.0,
        "WEAK_AXIS4": 0.5 * safety_model["logistic_slope"],
        "MODERATE_AXIS4": safety_model["logistic_slope"],
    }
    for n in COUNTS:
        values = unit_sphere(rng, (replicates, n, DIMENSION))
        full_metrics = batched_metrics(values)
        full_pass = candidate_pass(full_metrics)
        uniforms = {name: rng.random((replicates, n)) for name in slopes}
        standardized = (values[:, :, 4] - safety_model["coordinate_mean"]) / safety_model[
            "coordinate_standard_deviation"
        ]
        for scenario, slope in slopes.items():
            for p_safe in P_SAFE:
                if slope == 0.0:
                    probabilities = np.full_like(standardized, p_safe)
                else:
                    intercept = solve_intercept(coordinate_population, slope, p_safe)
                    probabilities = 1.0 / (1.0 + np.exp(-(intercept + slope * standardized)))
                safe = uniforms[scenario] < probabilities
                selected, eligible = first_safe(values, safe)
                reserve_successes = int(np.sum(eligible))
                if reserve_successes:
                    metrics = selected_metrics(selected, reference)
                    bank_pass = selected_pass(metrics)
                    full_for_eligible = full_pass[eligible]
                else:
                    metrics = {
                        name: np.asarray([])
                        for name in (
                            "rank",
                            "effective_rank",
                            "condition_number",
                            "maximum_absolute_pair_cosine",
                            "a0_q90_minus_q10",
                        )
                    }
                    bank_pass = np.asarray([], dtype=bool)
                    full_for_eligible = np.asarray([], dtype=bool)
                joint_pass = int(np.sum(bank_pass & full_for_eligible))
                bank_successes = int(np.sum(bank_pass))
                reserve_low, reserve_high = wilson(reserve_successes, replicates)
                unconditional_low, unconditional_high = wilson(bank_successes, replicates)
                rows.append(
                    {
                        "candidate_count": n,
                        "scenario": scenario,
                        "p_safe": p_safe,
                        "replicates": replicates,
                        "reserve_probability": reserve_successes / replicates,
                        "reserve_ci95_low": reserve_low,
                        "reserve_ci95_high": reserve_high,
                        "selected_gate_probability_given_reserve": (
                            bank_successes / reserve_successes
                            if reserve_successes
                            else float("nan")
                        ),
                        "unconditional_selected_qualification_probability": bank_successes
                        / replicates,
                        "unconditional_selected_ci95_low": unconditional_low,
                        "unconditional_selected_ci95_high": unconditional_high,
                        "candidate_and_selected_joint_probability": joint_pass / replicates,
                        "selected_pass_given_full_candidate_failure": float(
                            np.mean(bank_pass[~full_for_eligible])
                        )
                        if np.any(~full_for_eligible)
                        else float("nan"),
                        "selected_pass_given_candidate_condition_failure": float(
                            np.mean(bank_pass[full_metrics["condition_number"][eligible] > 3.0])
                        )
                        if np.any(full_metrics["condition_number"][eligible] > 3.0)
                        else float("nan"),
                        "selected_rank_pass_given_reserve": float(np.mean(metrics["rank"] == 8))
                        if reserve_successes
                        else float("nan"),
                        "selected_effective_rank_pass_given_reserve": float(
                            np.mean(metrics["effective_rank"] >= 4.8)
                        )
                        if reserve_successes
                        else float("nan"),
                        "selected_condition_pass_given_reserve": float(
                            np.mean(metrics["condition_number"] <= 10.0)
                        )
                        if reserve_successes
                        else float("nan"),
                        "selected_pair_cosine_pass_given_reserve": float(
                            np.mean(metrics["maximum_absolute_pair_cosine"] < 0.98)
                        )
                        if reserve_successes
                        else float("nan"),
                        "selected_cross_block_spread_pass_given_reserve": float(
                            np.mean(metrics["a0_q90_minus_q10"] >= 0.20)
                        )
                        if reserve_successes
                        else float("nan"),
                    }
                )
        adversarial_indices = np.argsort(values[:, :, 4], axis=1)[:, -K:]
        adversarial = values[np.arange(replicates)[:, None], adversarial_indices]
        adversarial_metrics = selected_metrics(adversarial, reference)
        adversarial_pass = selected_pass(adversarial_metrics)
        rows.append(
            {
                "candidate_count": n,
                "scenario": "ADVERSARIAL_AXIS4_TOP10",
                "p_safe": "NA",
                "replicates": replicates,
                "reserve_probability": 1.0,
                "reserve_ci95_low": 1.0,
                "reserve_ci95_high": 1.0,
                "selected_gate_probability_given_reserve": float(np.mean(adversarial_pass)),
                "unconditional_selected_qualification_probability": float(
                    np.mean(adversarial_pass)
                ),
                "unconditional_selected_ci95_low": wilson(
                    int(np.sum(adversarial_pass)), replicates
                )[0],
                "unconditional_selected_ci95_high": wilson(
                    int(np.sum(adversarial_pass)), replicates
                )[1],
                "candidate_and_selected_joint_probability": float(
                    np.mean(adversarial_pass & full_pass)
                ),
                "selected_pass_given_full_candidate_failure": float(
                    np.mean(adversarial_pass[~full_pass])
                )
                if np.any(~full_pass)
                else float("nan"),
                "selected_pass_given_candidate_condition_failure": float(
                    np.mean(adversarial_pass[full_metrics["condition_number"] > 3.0])
                )
                if np.any(full_metrics["condition_number"] > 3.0)
                else float("nan"),
                "selected_rank_pass_given_reserve": float(
                    np.mean(adversarial_metrics["rank"] == 8)
                ),
                "selected_effective_rank_pass_given_reserve": float(
                    np.mean(adversarial_metrics["effective_rank"] >= 4.8)
                ),
                "selected_condition_pass_given_reserve": float(
                    np.mean(adversarial_metrics["condition_number"] <= 10.0)
                ),
                "selected_pair_cosine_pass_given_reserve": float(
                    np.mean(adversarial_metrics["maximum_absolute_pair_cosine"] < 0.98)
                ),
                "selected_cross_block_spread_pass_given_reserve": float(
                    np.mean(adversarial_metrics["a0_q90_minus_q10"] >= 0.20)
                ),
            }
        )
    return rows


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    ranks = average_ranks(np.asarray(values).reshape(-1))
    ranks -= np.mean(ranks)
    return ranks / np.linalg.norm(ranks)


def power_calibration(
    replicates: int,
    permutations_count: int,
    seed: int,
    reference: np.ndarray,
    safety_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    permutations = fresh_row_permutations(K, permutations_count, seed=seed ^ 0x5A5A5A)
    rows: list[dict[str, Any]] = []
    per_n = max(1, replicates // len(COUNTS))
    slope = safety_model["logistic_slope"]
    population = unit_sphere(rng, (250000, DIMENSION))
    population_z = (population[:, 4] - safety_model["coordinate_mean"]) / safety_model[
        "coordinate_standard_deviation"
    ]
    intercept = solve_intercept(population_z, slope, 0.775)
    for n in COUNTS:
        completed = 0
        while completed < per_n:
            candidates = unit_sphere(rng, (n, DIMENSION))
            standardized = (candidates[:, 4] - safety_model["coordinate_mean"]) / safety_model[
                "coordinate_standard_deviation"
            ]
            probability = 1.0 / (1.0 + np.exp(-(intercept + slope * standardized)))
            safe = rng.random(n) < probability
            if np.sum(safe) < K:
                continue
            selected = candidates[np.flatnonzero(safe)[:K]]
            diagnostic = selected_metrics(selected[None, :, :], reference)
            geometry = angular_cross_block(selected, reference)
            x = normalized_ranks(geometry).reshape(K, len(reference))
            permutation_cache = x[permutations, :].reshape(len(permutations), -1)
            shell_y = []
            for _shell in range(2):
                row_noise = rng.standard_normal((K, 1))
                column_noise = rng.standard_normal((1, len(reference)))
                iid_noise = rng.standard_normal(geometry.shape)
                nuisance = iid_noise + 0.75 * row_noise + 0.25 * column_noise
                nuisance_flat = nuisance.reshape(-1)
                x_flat = x.reshape(-1)
                nuisance_flat -= np.dot(nuisance_flat, x_flat) * x_flat
                nuisance_flat /= np.linalg.norm(nuisance_flat)
                continuous = (
                    0.50 * REFERENCE_A0_RHO * x_flat
                    + math.sqrt(1.0 - (0.50 * REFERENCE_A0_RHO) ** 2) * nuisance_flat
                )
                shell_y.append(normalized_ranks(continuous).reshape(-1))
            qap = np.mean(
                np.stack([permutation_cache @ outcome for outcome in shell_y], axis=0), axis=0
            )
            observed = float(qap[0])
            p_value = float(np.sum(qap >= observed) / len(qap))
            lofo = []
            for omitted in range(K):
                keep = np.arange(K) != omitted
                lofo.append(
                    np.mean(
                        [
                            spearman_flat(geometry[keep], outcome.reshape(K, -1)[keep])
                            for outcome in shell_y
                        ]
                    )
                )
            rows.append(
                {
                    "candidate_count": n,
                    "effect_rho_input": 0.50 * REFERENCE_A0_RHO,
                    "planning_permutations": len(permutations),
                    "observed_aggregate_rho": observed,
                    "permutation_p": p_value,
                    "permutation_pass": bool(observed > 0.0 and p_value <= 0.05),
                    "all_lofo_positive": bool(np.all(np.asarray(lofo) > 0.0)),
                    "selected_effective_rank": float(diagnostic["effective_rank"][0]),
                    "selected_condition_number": float(diagnostic["condition_number"][0]),
                    "cross_block_A0_q90_minus_q10": float(diagnostic["a0_q90_minus_q10"][0]),
                    "cross_block_row_diversity": float(diagnostic["row_diversity_mean"][0]),
                }
            )
            completed += 1
    response = np.asarray([row["permutation_pass"] for row in rows], dtype=np.float64)
    predictors = np.column_stack(
        [
            np.asarray([row["cross_block_A0_q90_minus_q10"] for row in rows]),
            np.asarray([row["cross_block_row_diversity"] for row in rows]),
            np.asarray([row["selected_effective_rank"] for row in rows]),
            np.log(np.asarray([row["selected_condition_number"] for row in rows])),
        ]
    )
    predictor_names = [
        "cross_block_A0_q90_minus_q10",
        "cross_block_row_diversity",
        "selected_effective_rank",
        "log_selected_condition_number",
    ]
    standardized = (predictors - np.mean(predictors, axis=0)) / np.std(predictors, axis=0)
    design = np.column_stack([np.ones(len(standardized)), standardized])
    beta = np.zeros(design.shape[1])
    for _ in range(100):
        probability = 1.0 / (1.0 + np.exp(-(design @ beta)))
        weight = probability * (1.0 - probability)
        information = design.T @ (weight[:, None] * design) + np.eye(len(beta)) * 1e-6
        step = np.linalg.solve(information, design.T @ (response - probability))
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    quartiles = []
    for index, name in enumerate(predictor_names):
        bins = np.quantile(predictors[:, index], (0.25, 0.50, 0.75))
        group = np.digitize(predictors[:, index], bins)
        quartiles.append(
            {
                "metric": name,
                "quartile_edges": bins.tolist(),
                "power_by_quartile": [float(np.mean(response[group == q])) for q in range(4)],
            }
        )
    summary = {
        "replicates": len(rows),
        "effect_rho_input": 0.50 * REFERENCE_A0_RHO,
        "planning_permutations": len(permutations),
        "overall_permutation_power": float(np.mean(response)),
        "overall_all_lofo_positive": float(np.mean([row["all_lofo_positive"] for row in rows])),
        "standardized_logistic_coefficients_conditioning_on_all_metrics": {
            name: float(value) for name, value in zip(predictor_names, beta[1:], strict=True)
        },
        "quartile_power": quartiles,
        "interpretation": (
            "synthetic relational planning model only; coefficients diagnose whether "
            "selected-bank condition/effective rank add power information after cross-block spread"
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algebraic-replicates", type=int)
    parser.add_argument("--selected-replicates", type=int)
    parser.add_argument("--power-replicates", type=int)
    parser.add_argument("--planning-permutations", type=int)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if precheck["historical_v1_commit"] != V1_COMMIT:
        raise RuntimeError("audit precheck lineage mismatch")
    algebraic_replicates = (
        args.algebraic_replicates or precheck["algebraic_replicates_per_candidate_count"]
    )
    selected_replicates = (
        args.selected_replicates or precheck["selected_bank_replicates_per_candidate_count"]
    )
    power_replicates = args.power_replicates or precheck["power_replicates"]
    planning_permutations = args.planning_permutations or precheck["power_planning_permutations"]
    reference_manifest = read_json(
        ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
    )
    reference = np.asarray(
        [row["coefficients"] for row in reference_manifest["directions"]], dtype=np.float64
    )
    if reference.shape != (31, 8):
        raise RuntimeError("frozen reference bank is not 31x8")
    algebraic_rows, v1_percentiles = algebraic_calibration(
        algebraic_replicates, precheck["seeds"]["ALGEBRAIC"]
    )
    safety_model = fit_historical_safety(precheck["seeds"]["SAFETY_BOOTSTRAP"])
    reserve_rows = [
        {
            "candidate_count": n,
            **{
                f"p_at_least_10_safe_p_{str(p).replace('.', '_')}": binomial_tail(n, p)
                for p in P_SAFE
            },
        }
        for n in COUNTS
    ]
    selected_rows = selected_bank_calibration(
        selected_replicates,
        precheck["seeds"]["SELECTED_BANK"],
        reference,
        safety_model,
    )
    power_rows, power_summary = power_calibration(
        power_replicates,
        planning_permutations,
        precheck["seeds"]["POWER"],
        reference,
        safety_model,
    )
    write_csv(OUT / "ALGEBRAIC_GATE_CALIBRATION.csv", algebraic_rows)
    write_json(OUT / "V1_REALIZED_PERCENTILES.json", v1_percentiles)
    write_json(OUT / "HISTORICAL_SAFETY_GEOMETRY.json", safety_model)
    write_csv(OUT / "SAFETY_RESERVE.csv", reserve_rows)
    write_csv(OUT / "SELECTED_BANK_CALIBRATION.csv", selected_rows)
    write_csv(OUT / "CROSS_BLOCK_POWER.csv", power_rows)
    write_json(OUT / "CROSS_BLOCK_POWER_SUMMARY.json", power_summary)
    metadata = {
        "schema_version": "q2-oos-v2-candidate-gate-audit-results-v1",
        "precheck_sha256": sha256(PRECHECK),
        "historical_v1_commit": V1_COMMIT,
        "algebraic_replicates_per_count": algebraic_replicates,
        "selected_replicates_per_cell": selected_replicates,
        "power_replicates": len(power_rows),
        "planning_permutations": planning_permutations,
        "candidate_counts": COUNTS,
        "safety_probability_grid": P_SAFE,
        "model_inference": 0,
        "semantic_outcomes": 0,
        "correctness_inspected": False,
        "new_controller_stream_generated": False,
        "source_files": {
            "v1_closeout": sha256(REVIEW / "CANDIDATE_STREAM_CLOSEOUT.json"),
            "v1_manifest": sha256(REVIEW / "CANDIDATE_BANK_MANIFEST.json"),
            "reference_safe_bank": sha256(
                ROOT / "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
            ),
            "historical_safety_report": sha256(
                ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json"
            ),
        },
    }
    write_json(OUT / "SIMULATION_METADATA.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
