#!/usr/bin/env python3
"""Exactly-once post-hoc heterogeneity-robust sensitivity for closed Q2 V4.1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.heterogeneity_robust import node_jackknife_test

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference"
LOCK = OUT / "Q2_V4_1_ROBUST_SENSITIVITY_ESTIMATOR_LOCK.json"
LOCK_SHA256 = "76381274ae271c63b10aa0bd2b783f008ae274e7b13ed73a86c3aa3f89720a90"
ESTIMANDS = ROOT / "review/q2_v4_1_semantic_execution/ESTIMANDS.json"
ESTIMANDS_SHA256 = "17ae335447c441801ba9d5ab838a721ee21d1d4906700a2030d332d5054bd1f6"
MATRICES = ROOT / "review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz"
RESULT = OUT / "POST_HOC_Q2_HETEROGENEITY_ROBUST_SENSITIVITY.json"
REPORT = OUT / "POST_HOC_Q2_HETEROGENEITY_ROBUST_SENSITIVITY.md"
SHELLS = ("MEDIUM", "STRONG")
METRICS = ("A0", "A1", "A2")
T975_DF30 = 2.042272456


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, (count - index) * pvalues[key])
        adjusted[key] = min(running, 1.0)
    return adjusted


def main() -> None:
    if RESULT.exists() or REPORT.exists():
        raise RuntimeError("exactly-once sensitivity output already exists")
    if sha256(LOCK) != LOCK_SHA256:
        raise RuntimeError("robust estimator lock changed")
    lock = json.loads(LOCK.read_text())
    if lock["status"] != "FROZEN_AFTER_MODEL_FREE_CALIBRATION_BEFORE_HISTORICAL_DSHAPE_ACCESS":
        raise RuntimeError("robust estimator is not frozen")
    if sha256(ESTIMANDS) != ESTIMANDS_SHA256:
        raise RuntimeError("sealed ESTIMANDS artifact changed")
    payload = json.loads(ESTIMANDS.read_text())
    if payload["classification"] != "Q2_V4_1_G2":
        raise RuntimeError("historical classification mismatch")
    dshape = {
        shell: np.asarray(
            payload["semantic_distance"]["D_shape_superpopulation"][shell],
            dtype=np.float64,
        )
        for shell in SHELLS
    }
    with np.load(MATRICES) as data:
        geometries = {
            metric: {
                shell: np.asarray(data[f"{metric}_{shell}"], dtype=np.float64)
                for shell in SHELLS
            }
            for metric in METRICS
        }
    results = {}
    pvalues = {}
    for metric in METRICS:
        audit = node_jackknife_test(geometries[metric], dshape)
        pseudovalues = np.asarray(audit["pseudovalues"], dtype=np.float64)
        estimate = float(np.mean(pseudovalues))
        standard_error = float(audit["jackknife_standard_error"])
        lower = estimate - T975_DF30 * standard_error
        upper = estimate + T975_DF30 * standard_error
        pvalue = float(audit["p_value"])
        pvalues[metric] = pvalue
        results[metric] = {
            "historical_full_association": float(audit["full_association"]),
            "jackknife_pseudovalue_mean": estimate,
            "jackknife_standard_error": standard_error,
            "CI95": [lower, upper],
            "t": float(audit["t"]),
            "one_sided_p": pvalue,
            "leave_one_node_min": float(np.min(audit["leave_one_out"])),
            "leave_one_node_max": float(np.max(audit["leave_one_out"])),
            "leave_one_node_all_positive": bool(np.all(audit["leave_one_out"] > 0.0)),
        }
    adjusted = holm(pvalues)
    supported = []
    for metric in METRICS:
        results[metric]["Holm_p"] = adjusted[metric]
        results[metric]["robust_support"] = bool(
            adjusted[metric] <= 0.05 and results[metric]["CI95"][0] > 0.0
        )
        supported.append(results[metric]["robust_support"])
    if all(supported):
        classification = "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT"
    elif any(supported):
        classification = "Q2_V4_1_HETEROGENEITY_SENSITIVITY_MIXED"
    else:
        classification = "Q2_V4_1_HETEROGENEITY_ROBUST_SUPPORT_NOT_ESTABLISHED"
    output = {
        "schema_version": "q2-v4-1-posthoc-heterogeneity-robust-sensitivity-v1",
        "label": "POST_HOC_HETEROGENEITY_ROBUST_Q2_SENSITIVITY",
        "estimator_lock_sha256": LOCK_SHA256,
        "estimands_sha256": ESTIMANDS_SHA256,
        "historical_classification": "Q2_V4_1_G2",
        "historical_classification_modified": False,
        "results": results,
        "classification": classification,
        "model_inference": 0,
        "raw_journal_read": False,
        "correctness_read": False,
        "q3": "NOT_RUN",
    }
    write_json(RESULT, output)
    lines = [
        "# Post-hoc Q2 V4.1 heterogeneity-robust sensitivity",
        "",
        "**POST_HOC_HETEROGENEITY_ROBUST_Q2_SENSITIVITY.** This analysis cannot "
        "change the historical `Q2_V4_1_G2`, `RS+`, or `RT+` classifications.",
        "",
        "| Metric | Full rho | Jackknife estimate | 95% CI | Holm p | Support |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in METRICS:
        row = results[metric]
        lines.append(
            f"| {metric} | {row['historical_full_association']:.6f} | "
            f"{row['jackknife_pseudovalue_mean']:.6f} | "
            f"[{row['CI95'][0]:.6f}, {row['CI95'][1]:.6f}] | "
            f"{row['Holm_p']:.6g} | {'PASS' if row['robust_support'] else 'FAIL'} |"
        )
    lines.extend(["", f"Sensitivity classification: `{classification}`.", ""])
    REPORT.write_text("\n".join(lines))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
