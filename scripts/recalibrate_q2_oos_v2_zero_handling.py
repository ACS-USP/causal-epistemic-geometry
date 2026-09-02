#!/usr/bin/env python3
"""Narrow model-free recalibration of the zero-inclusive V2 sign test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.heterogeneity_robust import (
    exact_positive_sign_test,
    row_spearman,
)
from scripts.calibrate_q2_oos_v2_row_qap import reference_coefficients, stress_setup, wilson
from scripts.review_q2_heterogeneity_robust_inference import tournament_panel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference"
PRECHECK = OUT / "ZERO_HANDLING_SIGN_TEST_ERRATUM_PRECHECK.json"
PRECHECK_SHA256 = "990725399099b89a1f09ae592d238fbe517d4b54526a2ecc18a4d9d8475b6147"
PRECHECK_COMMIT = "9e3151820e25d7ff871952fb79c709d24f8f6561"
SHELLS = ("MEDIUM", "STRONG")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def quantiles(values: list[float], points: tuple[float, ...]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"q{int(point * 100):02d}": float(np.quantile(array, point)) for point in points}


def summarize_scenario(spec: dict[str, Any], scale: float) -> dict[str, Any]:
    reference = reference_coefficients()
    banks, latents, _ = stress_setup(reference)
    scenario = str(spec["id"])
    seed = int(PRECHECK_PAYLOAD["seeds"][spec["seed_key"]])
    target = float(spec.get("target", 0.0))
    replicates = max(20, int(int(spec["replicates"]) * scale))
    rejected: list[bool] = []
    zero_counts: list[int] = []
    nonpositive_counts: list[int] = []
    positive_counts: list[int] = []
    p_values: list[float] = []
    degenerate_panels = 0
    for index in range(replicates):
        geometry, outcomes = tournament_panel(
            scenario, index, reference, banks, latents, seed, target
        )
        rows = row_spearman({shell: geometry for shell in SHELLS}, outcomes)
        test = exact_positive_sign_test(rows)
        degenerate_panels += int(bool(test["degenerate"]))
        rejected.append(bool(test["reject_0_05"]))
        zero_counts.append(int(test["zeros"]))
        positive_counts.append(int(test["positives"]))
        nonpositive_counts.append(int(test["zeros"]) + int(test["negatives"]))
        p_values.append(float(test["p_value"]))
    rejections = int(np.sum(rejected))
    low, high = wilson(rejections, replicates)
    total_rows = replicates * 16
    exact_zero_rows = int(np.sum(zero_counts))
    return {
        "scenario": scenario,
        "kind": spec["kind"],
        "replicates": replicates,
        "rejections": rejections,
        "rate": rejections / replicates,
        "Wilson_95_low": low,
        "Wilson_95_high": high,
        "exact_zero_rows": exact_zero_rows,
        "exact_zero_row_fraction": exact_zero_rows / total_rows,
        "panels_with_exact_zero": int(np.sum(np.asarray(zero_counts) > 0)),
        "degenerate_panels": degenerate_panels,
        "nonpositive_count_distribution": quantiles(nonpositive_counts, (0, 0.25, 0.5, 0.75, 1)),
        "positive_count_distribution": quantiles(positive_counts, (0, 0.25, 0.5, 0.75, 1)),
        "exact_p_value_distribution": quantiles(p_values, (0, 0.05, 0.5, 0.95, 1)),
    }


def flatten(summary: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in summary.items() if not isinstance(value, dict)}
    distributions = (
        "nonpositive_count_distribution",
        "positive_count_distribution",
        "exact_p_value_distribution",
    )
    for key in distributions:
        for quantile, value in summary[key].items():
            row[f"{key}_{quantile}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.benchmark == args.full:
        raise SystemExit("choose exactly one of --benchmark or --full")
    observed_hash = hashlib.sha256(PRECHECK.read_bytes()).hexdigest()
    if observed_hash != PRECHECK_SHA256:
        raise SystemExit(f"precheck hash mismatch: {observed_hash}")
    started = time.monotonic()
    summaries = [
        summarize_scenario(spec, 0.01 if args.benchmark else 1.0)
        for spec in PRECHECK_PAYLOAD["scenarios"]
    ]
    nulls = [summary for summary in summaries if summary["kind"] == "null"]
    alternatives = {
        summary["scenario"]: summary
        for summary in summaries
        if summary["kind"] == "alternative"
    }
    calibrated = bool(
        all(
            summary["rate"] <= 0.065
            and summary["Wilson_95_low"] <= 0.055
            and summary["degenerate_panels"] == 0
            for summary in nulls
        )
    )
    power_adequate = bool(
        alternatives["POSITIVE_50_PERCENT_CLOSED_A0"]["rate"] >= 0.60
        and alternatives["POSITIVE_RHO_LIKE_0_15"]["rate"] >= 0.60
    )
    result = {
        "schema_version": "q2-oos-v2-zero-inclusive-sign-recalibration-v1",
        "precheck_commit": PRECHECK_COMMIT,
        "precheck_sha256": PRECHECK_SHA256,
        "method": "B_ROW_SPEARMAN_SIGN_ZERO_INCLUSIVE",
        "zero_rule": "finite r_i == 0 is a non-success in the K=16 denominator",
        "summaries": summaries,
        "calibrated_all_nulls": calibrated,
        "preferred_power_adequate": power_adequate,
        "classification": (
            "Q2_OOS_V2_SIGN_TEST_CALIBRATED"
            if calibrated and power_adequate
            else "Q2_OOS_V2_SIGN_TEST_CALIBRATION_BLOCKED"
        ),
        "elapsed_seconds": time.monotonic() - started,
        "methods_rerun": ["B_ROW_SPEARMAN_SIGN_ZERO_INCLUSIVE"],
        "historical_outcomes_accessed": False,
        "new_controller_streams": 0,
        "Qwen_inference": 0,
        "semantic_trajectories": 0
    }
    if args.benchmark:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    write_json(OUT / "ZERO_HANDLING_SIGN_TEST_RECALIBRATION.json", result)
    rows = [flatten(summary) for summary in summaries]
    with (OUT / "ZERO_HANDLING_SIGN_TEST_RECALIBRATION.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True))


PRECHECK_PAYLOAD = read_json(PRECHECK)


if __name__ == "__main__":
    main()
