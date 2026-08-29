"""Frozen Stage-B contracts for Q1 LiveCodeBench fixed-controller transfer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from epistemic_geometry.experiments import q1_second_task as base

EXPERIMENT_ID = "Q1_SECOND_TASK_LIVECODEBENCH_SPARK2_AMENDMENT1"
STAGE = "STAGE_B"
FAMILIES = 130
ROLLOUTS = 4
CONDITIONS = base.STAGE_B_CONDITIONS
RANDOM_NAMES = base.RANDOM_NAMES
LOGICAL_ROWS = FAMILIES * ROLLOUTS * len(CONDITIONS)
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 2026082902


def logical_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["stage"]),
        str(row["family_id"]),
        str(row["condition"]),
        int(row["rollout_index"]),
    )


def validate_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [logical_key(row) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != LOGICAL_ROWS or len(keys) != len(set(keys)):
        raise ValueError("Stage-B schedule must contain 5,720 unique logical rows")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Stage-B schedule contains duplicate seeds")
    if {str(row["stage"]) for row in rows} != {STAGE}:
        raise ValueError("Stage-B schedule stage mismatch")
    if {str(row["condition"]) for row in rows} != set(CONDITIONS):
        raise ValueError("Stage-B schedule condition mismatch")
    if len({str(row["family_id"]) for row in rows}) != FAMILIES:
        raise ValueError("Stage-B schedule family mismatch")
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["family_id"]), str(row["condition"]))
        counts[key] = counts.get(key, 0) + 1
    if set(counts.values()) != {ROLLOUTS}:
        raise ValueError("Stage-B requires four rollouts per family-condition")


def classify(
    *,
    summaries: Mapping[str, Mapping[str, float]],
    estimands: Mapping[str, Mapping[str, float]],
    intervals: Mapping[str, Mapping[str, float]],
    split_halves: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    meaningful = estimands["MEANINGFUL_FIXED_QWEN_L27_D75"]
    null_values = [estimands[name]["C"] for name in RANDOM_NAMES]
    scientific = {
        "P1_C_lower_gt_zero": intervals["C_meaningful"]["q025"] > 0.0,
        "P2a_delta_C_nullmean_lower_gt_zero": intervals["delta_C_nullmean"]["q025"] > 0.0,
        "P2b_C_meaningful_gt_every_null": meaningful["C"] > max(null_values),
        "split_half_A": bool(split_halves["A"]["passes"]),
        "split_half_B": bool(split_halves["B"]["passes"]),
    }
    baseline = summaries["BASELINE"]
    steered = summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]
    safety = {
        "commitment_validity": steered["commitment_validity"]
        >= baseline["commitment_validity"] - 0.05,
        "semantic_evaluability": steered["semantic_evaluability"]
        >= baseline["semantic_evaluability"] - 0.05,
        "accuracy": steered["accuracy"] >= baseline["accuracy"] - 0.10,
    }
    scientific_pass = all(scientific.values())
    safety_pass = all(safety.values())
    if scientific_pass and safety_pass:
        terminal = "Q1_SECOND_TASK_FIXED_CONTROLLER_PASS"
    elif scientific_pass:
        terminal = "Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL"
    else:
        terminal = "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY"
    return {
        "classification": terminal,
        "scientific_checks": scientific,
        "safety_checks": safety,
        "scientific_conjunction_pass": scientific_pass,
        "safety_conjunction_pass": safety_pass,
    }


def primary_bootstrap(
    baseline_errors: np.ndarray,
    condition_errors: Mapping[str, np.ndarray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    chunk_size: int = 1_000,
) -> dict[str, dict[str, float]]:
    """Frozen percentile family bootstrap for the two primary quantities."""

    baseline = np.asarray(baseline_errors, dtype=np.float64)
    if baseline.shape != (FAMILIES, ROLLOUTS):
        raise ValueError("baseline errors must have frozen shape (130, 4)")
    expected = {"MEANINGFUL_FIXED_QWEN_L27_D75", *RANDOM_NAMES}
    if set(condition_errors) != expected:
        raise ValueError("bootstrap condition set differs from frozen primary family")
    conditions = {name: np.asarray(condition_errors[name], dtype=np.float64) for name in expected}
    if any(values.shape != baseline.shape for values in conditions.values()):
        raise ValueError("all condition error matrices must have frozen shape (130, 4)")
    if resamples <= 0 or chunk_size <= 0:
        raise ValueError("resamples and chunk_size must be positive")

    generator = np.random.default_rng(seed)
    meaningful_values = np.empty(resamples, dtype=np.float64)
    contrast_values = np.empty(resamples, dtype=np.float64)
    ordered = ("MEANINGFUL_FIXED_QWEN_L27_D75", *RANDOM_NAMES)
    baseline_mean = baseline.mean(axis=1)
    baseline_total = baseline.sum(axis=1)
    baseline_within = (np.square(baseline_total) - np.square(baseline).sum(axis=1)) / (
        ROLLOUTS * (ROLLOUTS - 1)
    )
    condition_means = {name: values.mean(axis=1) for name, values in conditions.items()}
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(0, FAMILIES, size=(stop - start, FAMILIES))
        estimates = np.empty((stop - start, len(ordered)), dtype=np.float64)
        sampled_baseline_mean = baseline_mean[indices]
        baseline_sum = sampled_baseline_mean.sum(axis=1)
        b00 = baseline_within[indices].mean(axis=1)
        u00 = (np.square(baseline_sum) - np.square(sampled_baseline_mean).sum(axis=1)) / (
            FAMILIES * (FAMILIES - 1)
        )
        for column, name in enumerate(ordered):
            sampled_condition_mean = condition_means[name][indices]
            condition_sum = sampled_condition_mean.sum(axis=1)
            b0j = np.mean(sampled_baseline_mean * sampled_condition_mean, axis=1)
            u0j = (
                baseline_sum * condition_sum
                - np.sum(sampled_baseline_mean * sampled_condition_mean, axis=1)
            ) / (FAMILIES * (FAMILIES - 1))
            estimates[:, column] = b00 - b0j - u00 + u0j
        meaningful_values[start:stop] = estimates[:, 0]
        contrast_values[start:stop] = estimates[:, 0] - estimates[:, 1:].mean(axis=1)

    def interval(values: np.ndarray) -> dict[str, float]:
        return {
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }

    return {
        "C_meaningful": interval(meaningful_values),
        "delta_C_nullmean": interval(contrast_values),
    }


def split_half_checks(
    baseline_errors: np.ndarray, condition_errors: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """Apply the two prospectively fixed R=2 consistency checks."""

    per_condition = {
        name: base.split_half_estimands(baseline_errors, values)
        for name, values in condition_errors.items()
    }
    result: dict[str, dict[str, Any]] = {}
    meaningful_name = "MEANINGFUL_FIXED_QWEN_L27_D75"
    for half in ("A", "B"):
        meaningful_c = float(per_condition[meaningful_name][half]["C"])
        null_cs = [float(per_condition[name][half]["C"]) for name in RANDOM_NAMES]
        null_mean = float(np.mean(null_cs))
        checks = {
            "C_meaningful_gt_zero": meaningful_c > 0.0,
            "delta_C_nullmean_gt_zero": meaningful_c - null_mean > 0.0,
            "C_meaningful_gt_mean_nulls": meaningful_c > null_mean,
        }
        result[half] = {
            "C_meaningful": meaningful_c,
            "null_C_values": dict(zip(RANDOM_NAMES, null_cs, strict=True)),
            "null_C_mean": null_mean,
            "delta_C_nullmean": meaningful_c - null_mean,
            "checks": checks,
            "passes": all(checks.values()),
        }
    return result


__all__ = [
    "BOOTSTRAP_RESAMPLES",
    "BOOTSTRAP_SEED",
    "CONDITIONS",
    "EXPERIMENT_ID",
    "FAMILIES",
    "LOGICAL_ROWS",
    "RANDOM_NAMES",
    "ROLLOUTS",
    "STAGE",
    "classify",
    "logical_key",
    "primary_bootstrap",
    "split_half_checks",
    "validate_schedule",
]
