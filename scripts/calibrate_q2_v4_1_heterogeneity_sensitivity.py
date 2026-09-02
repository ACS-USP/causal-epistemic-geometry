#!/usr/bin/env python3
# ruff: noqa: E501
"""Model-free calibration of historical Q2 V4.1 heterogeneity sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.heterogeneity_robust import node_jackknife_test
from scripts.calibrate_q2_oos_v2_row_qap import normalized_ranks, wilson

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference"
MATRICES = ROOT / "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz"
PERMUTATIONS = ROOT / "review/q2_v4_1_prediction_lock/QAP_CONTROLLER_PERMUTATIONS.npy"
PRECHECK_COMMIT = "d1166eaa202fddc68af8da5a98c8f18f747939e6"
SHELLS = ("MEDIUM", "STRONG")
METRICS = ("A0", "A1", "A2")
K = 31


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def seed(base: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{label}".encode()).digest()[:16], "big")


def load_geometry() -> dict[str, dict[str, np.ndarray]]:
    with np.load(MATRICES) as data:
        return {
            metric: {shell: np.asarray(data[f"{metric}_{shell}"], dtype=np.float64) for shell in SHELLS}
            for metric in METRICS
        }


def qap_cache(geometry: dict[str, np.ndarray], permutations: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(K, 1)
    cache = np.empty((len(permutations), len(upper[0]), len(SHELLS)), dtype=np.float32)
    for index, permutation in enumerate(permutations):
        for shell_index, shell in enumerate(SHELLS):
            matrix = geometry[shell][np.ix_(permutation, permutation)]
            cache[index, :, shell_index] = normalized_ranks(matrix[upper]).astype(np.float32)
    return cache


def synthetic_outcome(
    geometry: dict[str, np.ndarray], scenario: str, target: float, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    if scenario == "NODE_HETEROGENEITY_NULL":
        beta = np.asarray([0.75] * 15 + [-0.75] * 15 + [0.0])
        rng.shuffle(beta)
    elif scenario == "HEAVY_NODE_HETEROGENEITY_NULL":
        magnitudes = np.clip(np.abs(rng.standard_cauchy(15)), 0.25, 2.5)
        beta = np.concatenate([magnitudes, -magnitudes, [0.0]])
        rng.shuffle(beta)
    elif scenario == "SAFETY_AXIS_HETEROGENEITY_NULL":
        beta = np.linspace(-1.0, 1.0, K)
        rng.shuffle(beta)
    else:
        beta = np.full(K, target)
    result = {}
    upper = np.triu_indices(K, 1)
    for shell in SHELLS:
        values = geometry[shell]
        standardized = (values - np.mean(values[upper])) / np.std(values[upper])
        noise = rng.standard_normal((K, K))
        noise = 0.5 * (noise + noise.T)
        coefficient = 0.5 * (beta[:, None] + beta[None, :])
        outcome = coefficient * standardized + noise
        np.fill_diagonal(outcome, 0.0)
        result[shell] = outcome
    return result


def qap_test(cache: np.ndarray, outcome: dict[str, np.ndarray]) -> tuple[float, float]:
    upper = np.triu_indices(K, 1)
    ranks = np.column_stack([normalized_ranks(outcome[shell][upper]) for shell in SHELLS])
    statistics = np.mean(np.einsum("pds,ds->ps", cache, ranks), axis=1)
    return float(statistics[0]), float(np.mean(statistics >= statistics[0]))


def summarize(values: list[bool]) -> dict[str, float | int]:
    count = int(np.sum(values))
    low, high = wilson(count, len(values))
    return {"replicates": len(values), "rejections": count, "rate": count / len(values), "Wilson_95_low": low, "Wilson_95_high": high}


def run(scale: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = read_json(OUT / "Q2_V4_1_HETEROGENEITY_ROBUST_SENSITIVITY_PROTOCOL.json")
    base_seed = int(protocol["calibration"]["seed"])
    geometries = load_geometry()
    permutations = np.load(PERMUTATIONS)[:1000]
    scenarios = [(name, 0.0, max(20, int(5000 * scale))) for name in protocol["calibration"]["nulls"]]
    scenarios.append(("POSITIVE_RHO_LIKE_0_15", 0.15, max(20, int(3000 * scale))))
    rows = []
    for metric in METRICS:
        cache = qap_cache(geometries[metric], permutations)
        for scenario, target, replicates in scenarios:
            qap_rejections = []
            jackknife_rejections = []
            qap_estimates = []
            jackknife_estimates = []
            jackknife_coverage = []
            for replicate in range(replicates):
                rng = np.random.Generator(np.random.PCG64DXSM(seed(base_seed, f"{metric}|{scenario}|{replicate}")))
                outcome = synthetic_outcome(geometries[metric], scenario, target, rng)
                observed, pvalue = qap_test(cache, outcome)
                jackknife = node_jackknife_test(geometries[metric], outcome)
                qap_rejections.append(bool(observed > 0.0 and pvalue <= 0.05))
                jackknife_rejections.append(bool(jackknife["reject_0_05"]))
                qap_estimates.append(observed)
                jackknife_estimates.append(float(jackknife["full_association"]))
                jackknife_coverage.append(not bool(jackknife["reject_0_05"]))
            for method, rejected, estimates, coverage in (
                ("HISTORICAL_CONTROLLER_LABEL_QAP", qap_rejections, qap_estimates, None),
                ("NODE_JACKKNIFE_PSEUDOVALUE_T", jackknife_rejections, jackknife_estimates, jackknife_coverage),
            ):
                rows.append({"metric": metric, "scenario": scenario, "kind": "alternative" if target else "null", "method": method, **summarize(rejected), "mean_estimate": float(np.mean(estimates)), "coverage": float(np.mean(coverage)) if coverage is not None else "NA"})
    jackknife_nulls = [row for row in rows if row["kind"] == "null" and row["method"] == "NODE_JACKKNIFE_PSEUDOVALUE_T"]
    calibrated = bool(all(row["rate"] <= 0.065 and row["Wilson_95_low"] <= 0.055 and row["coverage"] >= 0.925 for row in jackknife_nulls))
    result = {
        "schema_version": "q2-v4-1-heterogeneity-calibration-result-v1",
        "precheck_commit": PRECHECK_COMMIT,
        "selected_robust_estimator": "NODE_JACKKNIFE_PSEUDOVALUE_T" if calibrated else None,
        "calibrated": calibrated,
        "historical_semantic_matrix_accessed": False,
        "model_inference": 0,
        "historical_classification_modified": False,
    }
    return rows, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.benchmark == args.full:
        raise SystemExit("choose one mode")
    scale = 0.01 if args.benchmark else 1.0
    started = time.monotonic()
    rows, result = run(scale)
    elapsed = time.monotonic() - started
    if args.benchmark:
        print(json.dumps({"benchmark_seconds": elapsed, "projection_minutes": elapsed / scale / 60.0, "local_full_run_eligible": elapsed / scale <= 1800}, indent=2))
        return
    write_csv(OUT / "Q2_V4_1_QAP_HETEROGENEITY_CALIBRATION.csv", rows)
    result["elapsed_seconds"] = elapsed
    write_json(OUT / "Q2_V4_1_QAP_HETEROGENEITY_CALIBRATION.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
