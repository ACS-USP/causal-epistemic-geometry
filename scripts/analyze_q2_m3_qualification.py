#!/usr/bin/env python3
"""Apply the frozen M3 numerical qualification rule to remote engineering data."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis import m3_qualification as m3  # noqa: E402

REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    result = read_json(REVIEW / "M3_REMOTE_RESULTS.json")
    thresholds = m3.THRESHOLDS
    sequence = result["fp32_sequence"]
    alpha_zero = result["alpha_zero_identity"]
    sequence_pass = bool(
        alpha_zero["top1_agreement"] == 1.0
        and alpha_zero["max_vocabulary_js"] <= thresholds["alpha_zero_max_js"]
        and sequence["top1_agreement"] == thresholds["fp32_sequence_top1"]
        and sequence["median_vocabulary_js"] <= thresholds["fp32_sequence_median_js"]
        and sequence["p99_vocabulary_js"] <= thresholds["fp32_sequence_p99_js"]
        and sequence["median_target_logp_abs_difference"]
        <= thresholds["fp32_sequence_median_target_logp"]
        and sequence["max_target_logp_abs_difference"]
        <= thresholds["fp32_sequence_max_target_logp"]
        and sequence["median_logit_cosine"] >= thresholds["fp32_sequence_median_logit_cosine"]
    )

    crosschecks = result["exact_crosschecks"]
    exact_jvp_pass = bool(
        min(row["jvp_cosine"] for row in crosschecks) >= thresholds["independent_jvp_cosine"]
        and max(row["jvp_relative_norm"] for row in crosschecks)
        <= thresholds["independent_jvp_relative_norm"]
        and max(row["jvp_vjp_relative_error"] for row in crosschecks)
        <= thresholds["jvp_vjp_relative_error"]
    )
    reproducibility = result["reproducibility"]
    reproducibility_pass = bool(
        reproducibility["repeat_relative_frobenius"] <= thresholds["repeat_gram_relative_frobenius"]
        and reproducibility["direction_order_relative_frobenius"]
        <= thresholds["order_gram_relative_frobenius"]
        and reproducibility["chunked_aggregation_relative_frobenius"]
        <= thresholds["batch_gram_relative_frobenius"]
    )
    eigenvalues = np.asarray(result["exact_gram_geometry"]["eigenvalues"])
    psd_pass = bool(
        eigenvalues[0]
        >= -thresholds["psd_relative_negative_eigenvalue"] * max(float(eigenvalues[-1]), 1e-30)
    )
    polarization_pass = bool(
        result["direct_polarization_relative_frobenius"]
        <= thresholds["direct_polarization_relative_frobenius"]
    )
    derivative_pass = bool(
        exact_jvp_pass and reproducibility_pass and psd_pass and polarization_pass
    )

    stable_window = m3.stable_local_window(result["finite_ladder"])
    finite_pass = stable_window is not None
    bridge_baseline = result["bf16_baseline_bridge"]
    bridge_geometry = result["bf16_geometry_bridge"]
    bridge_pass = bool(
        bridge_baseline["top1_agreement"] >= thresholds["bf16_bridge_top1"]
        and bridge_baseline["median_vocabulary_js"] <= thresholds["bf16_bridge_median_js"]
        and bridge_baseline["p95_vocabulary_js"] <= thresholds["bf16_bridge_p95_js"]
        and bridge_geometry["radius_spearman"] >= thresholds["bf16_bridge_radius_spearman"]
        and bridge_geometry["distance_spearman"] >= thresholds["bf16_bridge_distance_spearman"]
        and bridge_geometry["median_curvature_relative_error"]
        <= thresholds["bf16_bridge_curvature_median_relative"]
        and not bridge_geometry["upper_lower_quartile_crossing"]
    )
    classification = m3.classify_m3(
        sequence_pass=sequence_pass,
        derivative_pass=derivative_pass,
        finite_window_pass=finite_pass,
        bf16_bridge_pass=bridge_pass,
    )
    qualification = {
        "classification": classification,
        "m3_status": "QUALIFIED"
        if classification == "M3_DIRECTIONAL_ENGINE_QUALIFIED"
        else "NOT_QUALIFIED",
        "object": (
            "teacher-forced multi-checkpoint categorical-Fisher controller-span Gram "
            "of the FP32 computational lift of frozen BF16-valued Qwen3-8B parameters"
        ),
        "sequence_pass": sequence_pass,
        "alpha_zero_identity_pass": bool(
            alpha_zero["top1_agreement"] == 1.0
            and alpha_zero["max_vocabulary_js"] <= thresholds["alpha_zero_max_js"]
        ),
        "reproducibility_pass": reproducibility_pass,
        "exact_jvp_vjp_pass": exact_jvp_pass,
        "psd_pass": psd_pass,
        "polarization_pass": polarization_pass,
        "finite_local_window_pass": finite_pass,
        "finite_local_window": stable_window,
        "bf16_bridge_pass": bridge_pass,
        "semantic_outcomes_used": False,
        "scientific_items_processed": 0,
        "q2_v3_behavioral_trajectories": 0,
        "thresholds": thresholds,
    }
    write_json(REVIEW / "M3_ENGINE_QUALIFICATION.json", qualification)
    write_json(
        REVIEW / "M3_SEQUENCE_EQUIVALENCE.json",
        {
            "pass": sequence_pass,
            "alpha_zero": alpha_zero,
            "fp32_full_vs_sequential": sequence,
        },
    )
    write_json(
        REVIEW / "M3_BF16_BRIDGE.json",
        {"pass": bridge_pass, "baseline": bridge_baseline, "geometry": bridge_geometry},
    )
    write_json(
        REVIEW / "M3_EXACT_DERIVATIVE_CROSSCHECK.json",
        {
            "pass": derivative_pass,
            "exact_jvp_vjp_pass": exact_jvp_pass,
            "reproducibility_pass": reproducibility_pass,
            "psd_pass": psd_pass,
            "polarization_pass": polarization_pass,
            "crosschecks": crosschecks,
            "reproducibility": reproducibility,
            "direct_polarization_relative_frobenius": result[
                "direct_polarization_relative_frobenius"
            ],
            "eigenvalues": eigenvalues.tolist(),
        },
    )
    with (REVIEW / "M3_FINITE_DIFFERENCE_LADDER.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["finite_ladder"][0]))
        writer.writeheader()
        writer.writerows(result["finite_ladder"])

    report = f"""# M3 real-Qwen engineering qualification

Classification: `{classification}`. M3 status: `{qualification["m3_status"]}`.

## Exact object

`Gamma_ij` is the uniform mean, over 16 frozen synthetic teacher-forced
fixtures and their prescribed checkpoints, of
`(J_z v_i)^T (diag(p)-pp^T) (J_z v_j)`. The derivative is evaluated in the
FP32 computational lift of the exact BF16-valued Qwen3-8B checkpoint at block
27. This is not semantic-error geometry and not the Fisher geometry of the
full free-running trajectory distribution.

## Frozen gates

- alpha-zero / FP32 sequence semantics: `{sequence_pass}`
- repeated/order/chunked reproducibility: `{reproducibility_pass}`
- independent JVP and JVP/VJP: `{exact_jvp_pass}`
- PSD without clipping: `{psd_pass}`
- direct versus polarization: `{polarization_pass}`
- three-scale finite local window: `{finite_pass}` — `{stable_window}`
- historical BF16 bridge: `{bridge_pass}`

The BF16 bridge is constitutive for inclusion in Q2 V3. Exact FP32 coherence
alone is insufficient. No semantic correctness, CRUXEval scientific item, free
generation, or Q2 V3 behavioral trajectory was used.
"""
    (REVIEW / "M3_QUALIFICATION_REPORT.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
