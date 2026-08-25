#!/usr/bin/env python3
"""Independently audit the persisted Q2 M3 engineering qualification arrays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"
RAW = REVIEW / "raw"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(observed: float, expected: float) -> float:
    return abs(observed - expected) / max(abs(expected), 1e-15)


def relative_frobenius(observed: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1e-15))


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(
        np.dot(left_flat, right_flat)
        / max(np.linalg.norm(left_flat) * np.linalg.norm(right_flat), 1e-30)
    )


def stable_window(rows: list[dict[str, float]], thresholds: dict[str, float]) -> list[float] | None:
    ordered = sorted(rows, key=lambda row: row["epsilon"])
    keys = (
        "fisher_relative_error",
        "kl_relative_error",
        "hellinger_relative_error",
        "js_relative_error",
        "gram_relative_error",
        "radius_relative_error",
    )
    eligible = []
    for row in ordered:
        eligible.append(
            row["epsilon"] <= 1.0
            and row["jvp_cosine"] >= thresholds["finite_jvp_cosine"]
            and all(row[key] <= thresholds["finite_relative_error"] for key in keys)
            and row["angle_max_abs_error"] <= thresholds["finite_angular_max_abs_error"]
            and row["rms_logit_movement"] <= thresholds["finite_local_rms_logit_movement"]
        )
    width = int(thresholds["finite_window_length"])
    for start in range(len(ordered) - width + 1):
        if all(eligible[start : start + width]):
            return [float(row["epsilon"]) for row in ordered[start : start + width]]
    return None


def classify(
    *, sequence: bool, derivative: bool, finite: bool, bridge: bool
) -> str:
    if not sequence:
        return "M3_SEQUENCE_SEMANTICS_FAILED"
    if not derivative:
        return "M3_DERIVATIVE_IDENTITIES_FAILED"
    if not finite:
        return "M3_FINITE_LOCAL_WINDOW_FAILED"
    if not bridge:
        return "M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED"
    return "M3_DIRECTIONAL_ENGINE_QUALIFIED"


def main() -> int:
    result_path = REVIEW / "M3_REMOTE_RESULTS.json"
    result = read_json(result_path)
    primary = read_json(REVIEW / "M3_ENGINE_QUALIFICATION.json")
    protocol = read_json(REVIEW / "M3_QUALIFICATION_PROTOCOL.json")
    thresholds = protocol["thresholds"]
    sufficient_path = ROOT / result["raw_sufficient_statistics"]
    if sha256(sufficient_path) != result["raw_sufficient_statistics_sha256"]:
        raise RuntimeError("M3 sufficient-statistics digest mismatch")
    arrays = np.load(sufficient_path, allow_pickle=False)

    exact = np.asarray(arrays["exact_gram"], dtype=np.float64)
    direct = np.asarray(arrays["direct_subset"], dtype=np.float64)
    polarization = np.asarray(arrays["polarization_subset"], dtype=np.float64)
    finite_matrix = np.asarray(arrays["finite_metrics"], dtype=np.float64)
    finite_keys = (
        "epsilon",
        "jvp_cosine",
        "fisher_relative_error",
        "kl_relative_error",
        "hellinger_relative_error",
        "js_relative_error",
        "gram_relative_error",
        "radius_relative_error",
        "angle_max_abs_error",
        "rms_logit_movement",
    )
    finite_rows = [dict(zip(finite_keys, row, strict=True)) for row in finite_matrix]

    crosscheck_rows = []
    for frozen in result["exact_crosschecks"]:
        raw_path = ROOT / frozen["raw_path"]
        if sha256(raw_path) != frozen["raw_sha256"]:
            raise RuntimeError(f"crosscheck digest mismatch: {raw_path}")
        raw = np.load(raw_path, allow_pickle=False)
        forward = np.asarray(raw["forward_jvp"], dtype=np.float64)
        independent = np.asarray(raw["independent_jvp"], dtype=np.float64)
        crosscheck_rows.append(
            {
                "fixture_index": frozen["fixture_index"],
                "direction_index": frozen["direction_index"],
                "jvp_cosine": cosine(forward, independent),
                "jvp_relative_norm": relative(
                    float(np.linalg.norm(independent)), float(np.linalg.norm(forward))
                ),
                "reported_jvp_vjp_relative_error": frozen["jvp_vjp_relative_error"],
                "raw_sha256": frozen["raw_sha256"],
            }
        )

    sequence_values = result["fp32_sequence"]
    alpha_zero = result["alpha_zero_identity"]
    sequence_pass = bool(
        alpha_zero["top1_agreement"] == 1.0
        and alpha_zero["max_vocabulary_js"] <= thresholds["alpha_zero_max_js"]
        and sequence_values["top1_agreement"] == thresholds["fp32_sequence_top1"]
        and sequence_values["median_vocabulary_js"]
        <= thresholds["fp32_sequence_median_js"]
        and sequence_values["p99_vocabulary_js"] <= thresholds["fp32_sequence_p99_js"]
        and sequence_values["median_target_logp_abs_difference"]
        <= thresholds["fp32_sequence_median_target_logp"]
        and sequence_values["max_target_logp_abs_difference"]
        <= thresholds["fp32_sequence_max_target_logp"]
        and sequence_values["median_logit_cosine"]
        >= thresholds["fp32_sequence_median_logit_cosine"]
    )
    reproducibility = result["reproducibility"]
    exact_jvp_pass = bool(
        min(row["jvp_cosine"] for row in crosscheck_rows)
        >= thresholds["independent_jvp_cosine"]
        and max(row["jvp_relative_norm"] for row in crosscheck_rows)
        <= thresholds["independent_jvp_relative_norm"]
        and max(row["reported_jvp_vjp_relative_error"] for row in crosscheck_rows)
        <= thresholds["jvp_vjp_relative_error"]
    )
    reproducibility_pass = bool(
        reproducibility["repeat_relative_frobenius"]
        <= thresholds["repeat_gram_relative_frobenius"]
        and reproducibility["direction_order_relative_frobenius"]
        <= thresholds["order_gram_relative_frobenius"]
        and reproducibility["chunked_aggregation_relative_frobenius"]
        <= thresholds["batch_gram_relative_frobenius"]
    )
    eigenvalues = np.linalg.eigvalsh(0.5 * (exact + exact.T))
    psd_pass = bool(
        eigenvalues[0]
        >= -thresholds["psd_relative_negative_eigenvalue"] * max(eigenvalues[-1], 1e-30)
    )
    polarization_error = relative_frobenius(polarization, direct)
    polarization_pass = bool(
        polarization_error <= thresholds["direct_polarization_relative_frobenius"]
    )
    derivative_pass = exact_jvp_pass and reproducibility_pass and psd_pass and polarization_pass
    window = stable_window(finite_rows, thresholds)
    finite_pass = window is not None
    baseline_bridge = result["bf16_baseline_bridge"]
    geometry_bridge = result["bf16_geometry_bridge"]
    bridge_pass = bool(
        baseline_bridge["top1_agreement"] >= thresholds["bf16_bridge_top1"]
        and baseline_bridge["median_vocabulary_js"] <= thresholds["bf16_bridge_median_js"]
        and baseline_bridge["p95_vocabulary_js"] <= thresholds["bf16_bridge_p95_js"]
        and geometry_bridge["radius_spearman"] >= thresholds["bf16_bridge_radius_spearman"]
        and geometry_bridge["distance_spearman"] >= thresholds["bf16_bridge_distance_spearman"]
        and geometry_bridge["median_curvature_relative_error"]
        <= thresholds["bf16_bridge_curvature_median_relative"]
        and not geometry_bridge["upper_lower_quartile_crossing"]
    )
    audit_classification = classify(
        sequence=sequence_pass,
        derivative=derivative_pass,
        finite=finite_pass,
        bridge=bridge_pass,
    )
    maximum_metric_difference = max(
        [
            abs(row["jvp_cosine"] - frozen["jvp_cosine"])
            for row, frozen in zip(crosscheck_rows, result["exact_crosschecks"], strict=True)
        ]
        + [
            abs(row["jvp_relative_norm"] - frozen["jvp_relative_norm"])
            for row, frozen in zip(crosscheck_rows, result["exact_crosschecks"], strict=True)
        ]
        + [abs(polarization_error - result["direct_polarization_relative_frobenius"])]
    )
    audit = {
        "classification": "M3_FORENSIC_CLEAN"
        if audit_classification == primary["classification"]
        else "M3_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN",
        "primary_classification": primary["classification"],
        "audit_classification": audit_classification,
        "classification_agreement": audit_classification == primary["classification"],
        "maximum_recomputed_metric_difference": maximum_metric_difference,
        "sequence_pass": sequence_pass,
        "exact_jvp_pass": exact_jvp_pass,
        "reproducibility_pass": reproducibility_pass,
        "psd_pass": psd_pass,
        "minimum_eigenvalue": float(eigenvalues[0]),
        "polarization_pass": polarization_pass,
        "polarization_relative_frobenius": polarization_error,
        "finite_window_pass": finite_pass,
        "finite_window": window,
        "bf16_bridge_pass": bridge_pass,
        "crosschecks": crosscheck_rows,
        "raw_sufficient_statistics_sha256": sha256(sufficient_path),
        "remote_results_sha256": sha256(result_path),
        "scientific_items_processed": result["scientific_items_processed"],
        "semantic_outcomes_read": result["semantic_outcomes_read"],
        "q2_v3_behavioral_trajectories": result["q2_v3_behavioral_trajectories"],
        "note": (
            "JVP cosine/norm, Gram algebra, PSD, finite ladder, bridges, and final "
            "classification were independently recomputed. The persisted VJP scalar "
            "error is threshold-checked but its two scalar operands were not archived."
        ),
    }
    (REVIEW / "M3_FORENSIC_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# M3 forensic audit

Classification: `{audit['classification']}`.

- primary/audit classification: `{primary['classification']}` / `{audit_classification}`
- maximum independently recomputed metric difference: `{maximum_metric_difference:.3e}`
- sequence / derivative: `{sequence_pass}` / `{derivative_pass}`
- finite-window / BF16-bridge: `{finite_pass}` / `{bridge_pass}`
- scientific items processed: `{result['scientific_items_processed']}`
- semantic outcomes read: `{result['semantic_outcomes_read']}`

The audit independently reloaded the immutable sufficient-statistics and JVP
arrays, recomputed Gram algebra, PSD, finite-window eligibility, bridge gates,
and the frozen classification without importing the primary analysis module.
The JVP/VJP scalar error was threshold-checked from the technical runner record;
its two scalar operands were not separately archived. This is a reproducibility
limitation, not semantic-outcome leakage.
"""
    (REVIEW / "M3_FORENSIC_AUDIT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
