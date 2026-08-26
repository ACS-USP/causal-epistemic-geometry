#!/usr/bin/env python3
"""CPU-only Q2 V4.1 frozen-31-bank adequacy review.

This script reads only the historical V4 coefficient bank and label-free
safety artifacts. Its simulation uses synthetic outcomes from the already
published V4 planning machinery. It has no path to semantic journals or model
execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epistemic_geometry.experiments.q2_v4_1 import (  # noqa: E402
    EXPECTED_SAFE_IDS,
    V4_CANDIDATE_COMMIT,
    V4_CLASSIFICATION,
    V4_FINAL_COMMIT,
    V4_PRELOCK,
    bank_geometry,
    load_frozen_candidates,
    reserve_fragility,
    safety_structure,
    selected_bank_coverage_checks,
    sha256_file,
    synthetic_adequacy_criteria,
)
from scripts.design_q2_v4_intervention_subspace import (  # noqa: E402
    _candidate_embeddings,
    _finite_specific_embeddings,
    _qap_cache,
    _simulate_once,
)
from epistemic_geometry.experiments.q2_v4 import (  # noqa: E402
    controller_permutations,
    protocol_seed,
)

REVIEW = ROOT / "review" / "q2_v4_1_31_safe_bank_review"
CANDIDATE_MANIFEST = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json"
SAFETY_REPORT = ROOT / "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json"
V4_AUDIT = ROOT / "review/q2_v4_spark1_presemantic/SAFETY_FORENSIC_AUDIT.json"
SOURCE_COMMIT = V4_FINAL_COMMIT
RHO_VALUES = (0.0, 0.10, 0.20, 0.25, 0.30, 0.40)
PLANNING_QAP_MAPS = 499
FINAL_QAP_MAPS = 50_000
DEFAULT_REPLICATES = 600
SEED = 2026082601


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_design_precheck() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    criteria = synthetic_adequacy_criteria()
    criteria.update({
        "source_commit": SOURCE_COMMIT,
        "historical_v4_classification": V4_CLASSIFICATION,
        "historical_v4_prelock": V4_PRELOCK,
        "semantic_outcomes_read": False,
        "model_inference": False,
        "gpu_used": False,
        "application_order": [
            "freeze criteria",
            "reconstruct 40 and 31-safe geometry",
            "run synthetic K=31/K=32 planning simulation",
            "apply criteria mechanically",
        ],
    })
    write_json(REVIEW / "DESIGN_PRECHECK.json", criteria)
    (REVIEW / "DESIGN_PRECHECK.md").write_text(
        "# Q2 V4.1 coverage and power adequacy design precheck\n\n"
        "Status: FROZEN_BEFORE_OBSERVED_BANK_APPLICATION.\n\n"
        "The coverage clauses are inherited from the V4 selected-bank gate. "
        "Only the cardinality clause is generalized from K=32 to the realized "
        "K=31 review population. The K=31 adequacy rule preserves the V4 "
        "scientifically meaningful rho=0.25 target, requires approximately "
        "80% omnibus power, controlled planning FPR, and limits degradation "
        "relative to K=32/N=300 to 0.10 absolute power or 1.10 CI-width ratio. "
        "A2 attribution and G3 power remain reported, not silently redefined.\n\n"
        "No observed-bank PASS/FAIL value is included in this precheck. "
        "No semantic outcome, correctness label, model output, GPU, or new "
        "candidate is allowed in this review.\n",
        encoding="utf-8",
    )


def _amplitudes(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([
        [row["medium"]["implemented_amplitude"], row["strong"]["implemented_amplitude"]]
        for row in rows
    ], dtype=np.float64)


def reconstruct_bank() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates, safe = load_frozen_candidates(CANDIDATE_MANIFEST, SAFETY_REPORT)
    all_coefficients = np.asarray([row["coefficients"] for row in candidates], dtype=np.float64)
    safe_coefficients = np.asarray([row["coefficients"] for row in safe], dtype=np.float64)
    all_amplitudes = _amplitudes(candidates)
    safe_amplitudes = _amplitudes(safe)
    structure = selected_bank_coverage_checks(safe_coefficients, safe_amplitudes)
    reconstructed = {
        "historical_artifacts": {
            "candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
            "safety_report_sha256": sha256_file(SAFETY_REPORT),
            "v4_forensic_audit_sha256": sha256_file(V4_AUDIT),
            "v4_prelock": V4_PRELOCK,
            "v4_candidate_bank_commit": V4_CANDIDATE_COMMIT,
            "v4_final_commit": V4_FINAL_COMMIT,
        },
        "candidate_count": len(candidates),
        "safe_count": len(safe),
        "unsafe_count": len(candidates) - len(safe),
        "candidate_order_preserved": [
            row["candidate_id"] for row in candidates
        ] == [f"V4_DIRECTION_{i:02d}" for i in range(40)],
        "safe_order_preserved": [row["candidate_id"] for row in safe] == list(EXPECTED_SAFE_IDS),
        "correctness_used": False,
        "semantic_outcomes": 0,
        "all_geometry": bank_geometry(all_coefficients),
        "safe_geometry": bank_geometry(safe_coefficients),
        "safe_coverage_checks": structure,
        "unsafe_ids": [row["candidate_id"] for row in candidates if not row["joint_safe"]],
        "all_candidates": candidates,
        "safe_candidates": safe,
    }
    write_json(REVIEW / "CANDIDATE_RECONSTRUCTION.json", reconstructed)
    write_json(REVIEW / "SAFETY_ATTRITION_GEOMETRY.json", {
        "all_40": reconstructed["all_geometry"],
        "safe_31": reconstructed["safe_geometry"],
        "coverage_checks": structure,
        "unsafe_ids": reconstructed["unsafe_ids"],
    })
    return candidates, safe, reconstructed


def _planning_row(
    results: list[dict[str, Any]],
    *,
    k: int,
    n_items: int,
    rho: float,
    scenario: str = "correlated_metric_ladder",
) -> dict[str, Any]:
    names = ("a0", "a1", "a2")
    row: dict[str, Any] = {
        "scenario": scenario,
        "K": k,
        "N": n_items,
        "target_rho": rho,
        "replicates": len(results),
        "qap_permutations": PLANNING_QAP_MAPS,
        "final_protocol_qap_permutations": FINAL_QAP_MAPS,
    }
    for name in names:
        values = np.asarray([float(result[f"observed_{name}"]) for result in results])
        row[f"{name}_rho_mean"] = float(np.mean(values))
        row[f"{name}_rho_mc95_width"] = float(
            np.quantile(values, 0.975) - np.quantile(values, 0.025)
        )
    for label, key in (
        ("omnibus", "omnibus_pass"),
        ("a2_attribution", "a2_attribution_pass"),
        ("a2_superiority", "a2_superiority_pass"),
        ("radial", "radial_pass"),
    ):
        rate = float(np.mean([bool(result[key]) for result in results]))
        row[f"{label}_rate"] = rate
        row[f"{label}_mc_se"] = math.sqrt(rate * (1.0 - rate) / len(results))
        row[f"{label}_mc95"] = [
            max(0.0, rate - 1.96 * row[f"{label}_mc_se"]),
            min(1.0, rate + 1.96 * row[f"{label}_mc_se"]),
        ]
    row["mean_true_rho"] = float(np.mean([float(result["true_rho"]) for result in results]))
    a2 = np.asarray([float(result["observed_a2"]) for result in results])
    a0 = np.asarray([float(result["observed_a0"]) for result in results])
    a1 = np.asarray([float(result["observed_a1"]) for result in results])
    row["a2_minus_best_static_mean"] = float(np.mean(a2 - np.maximum(a0, a1)))
    return row


def _simulate_cells(
    *,
    k: int,
    n_items: int,
    rho_values: tuple[float, ...],
    replicates: int,
    superiority: bool = False,
) -> list[dict[str, Any]]:
    from epistemic_geometry.experiments.q2_v4 import sample_coefficient_bank

    bank_seed = protocol_seed(f"Q2-V4-POWER-BANK-K{k}-V1", SOURCE_COMMIT)
    coefficients = sample_coefficient_bank(8, k, seed=bank_seed)
    embeddings = (
        _finite_specific_embeddings(coefficients, bank_seed ^ 0xA2A2)
        if superiority
        else _candidate_embeddings(coefficients, bank_seed ^ 0xA2A2)
    )
    permutations = controller_permutations(
        k,
        PLANNING_QAP_MAPS,
        seed=protocol_seed(f"Q2-V4-POWER-QAP-K{k}-V1", SOURCE_COMMIT),
    )
    metric_names, cache = _qap_cache(
        {
            name: value / np.linalg.norm(value, axis=1, keepdims=True)
            for name, value in embeddings.items()
        },
        permutations,
    )
    rows = []
    for rho in rho_values:
        results = [
            _simulate_once(
                embeddings,
                cache,
                metric_names,
                n_items=n_items,
                target_rho=rho,
                seed=protocol_seed(
                    f"Q2-V4-{('SUPERIORITY' if superiority else 'SIM')}-K{k}-N{n_items}-R{rho}-I{replicate}",
                    SOURCE_COMMIT,
                ),
            )
            for replicate in range(replicates)
        ]
        rows.append(_planning_row(
            results,
            k=k,
            n_items=n_items,
            rho=rho,
            scenario=(
                "finite_specific_superiority" if superiority
                else "correlated_metric_ladder"
            ),
        ))
    return rows


def run_power(
    replicates: int,
    include_n400: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_simulate_cells(k=31, n_items=300, rho_values=RHO_VALUES, replicates=replicates))
    rows.extend(_simulate_cells(k=32, n_items=300, rho_values=RHO_VALUES, replicates=replicates))
    if include_n400:
        rows.extend(_simulate_cells(k=31, n_items=400, rho_values=RHO_VALUES, replicates=replicates))
    superiority_rows: list[dict[str, Any]] = []
    superiority_rhos = (0.25, 0.30, 0.40, 0.50, 0.60)
    superiority_rows.extend(_simulate_cells(
        k=31, n_items=300, rho_values=superiority_rhos,
        replicates=replicates, superiority=True,
    ))
    superiority_rows.extend(_simulate_cells(
        k=32, n_items=300, rho_values=superiority_rhos,
        replicates=replicates, superiority=True,
    ))
    write_csv(REVIEW / "POWER_SIMULATION.csv", rows)
    write_csv(REVIEW / "SUPERIORITY_POWER_SIMULATION.csv", superiority_rows)
    write_json(REVIEW / "POWER_SIMULATION_METADATA.json", {
        "schema_version": "q2-v4.1-power-simulation-v1",
        "cpu_only": True,
        "semantic_outcomes_read": False,
        "model_inference": False,
        "source_commit": SOURCE_COMMIT,
        "K_values": [31, 32],
        "N_values": [300] + ([400] if include_n400 else []),
        "rho_values": list(RHO_VALUES),
        "replicates_per_cell": replicates,
        "planning_qap_maps_per_replicate": PLANNING_QAP_MAPS,
        "final_v4_qap_maps_preserved": FINAL_QAP_MAPS,
        "qap_structure": "same controller-label permutation, shell-coupled maxT machinery as V4",
        "endpoint": "N/(N-1)-corrected two-rollout item-population D_shape",
        "multiplicity": "single-step maxT across A0/A1/A2",
        "K31_primary": True,
        "K32_reference": True,
        "N400_secondary": include_n400,
        "seed_source_commit": SOURCE_COMMIT,
        "note": (
            "Planning uses the V4 established 499-map Monte Carlo acceleration "
            "for repeated power cells; the 50,000-map final protocol is unchanged."
        ),
    })
    return rows, superiority_rows


def _find(rows: list[dict[str, Any]], *, k: int, n: int, rho: float) -> dict[str, Any]:
    for row in rows:
        if row["K"] == k and row["N"] == n and row["target_rho"] == rho:
            return row
    raise KeyError((k, n, rho))


def apply_decision(
    reconstructed: dict[str, Any],
    rows: list[dict[str, Any]],
    include_n400: bool,
) -> dict[str, Any]:
    checks = reconstructed["safe_coverage_checks"]["checks"]
    k31 = _find(rows, k=31, n=300, rho=0.25)
    k32 = _find(rows, k=32, n=300, rho=0.25)
    power_checks = {
        "coverage_all_pass": bool(reconstructed["safe_coverage_checks"]["pass"]),
        "k31_omnibus_power_at_least_0_80": k31["omnibus_rate"] >= 0.80,
        "k31_fpr_in_prespecified_range": 0.025 <= _find(
            rows, k=31, n=300, rho=0.0
        )["omnibus_rate"] <= 0.075,
        "absolute_omnibus_power_loss_vs_k32_at_most_0_10": (
            k32["omnibus_rate"] - k31["omnibus_rate"] <= 0.10
        ),
        "ci_width_ratio_vs_k32_at_most_1_10": (
            k31["a2_rho_mc95_width"] / k32["a2_rho_mc95_width"] <= 1.10
        ),
    }
    adequate = bool(all(power_checks.values()))
    decision = {
        "criteria_source": "DESIGN_PRECHECK.json",
        "coverage_checks": checks,
        "power_checks": power_checks,
        "k31_reference_cell": k31,
        "k32_reference_cell": k32,
        "n400_simulated": include_n400,
        "decision": (
            "Q2_V4_1_31_SAFE_BANK_ADEQUATE"
            if adequate else "Q2_V4_1_31_SAFE_BANK_INADEQUATE"
        ),
        "semantic_outcomes": 0,
        "a1_a2_new_computation": False,
        "model_inference": False,
        "gpu_used": False,
        "mechanical": True,
    }
    write_json(REVIEW / "ADEQUACY_DECISION.json", decision)
    if adequate:
        manifest = {
            "schema_version": "q2-v4.1-safe-bank-manifest-v1",
            "classification": decision["decision"],
            "population": "original isotropic V4 candidates conditioned on frozen safety eligibility",
            "candidate_count_historical": 40,
            "safe_count": 31,
            "candidate_order": list(EXPECTED_SAFE_IDS),
            "prelock": V4_PRELOCK,
            "candidate_bank_commit": V4_CANDIDATE_COMMIT,
            "v4_final_commit": V4_FINAL_COMMIT,
            "safety_classification": V4_CLASSIFICATION,
            "directions": reconstructed["safe_candidates"],
            "future_conditions": 63,
            "future_n": 300,
            "future_semantic_trajectories": 37800,
            "semantic_experiment_status": "DRAFT_NOT_RUN",
        }
        write_json(REVIEW / "SAFE_31_IMMUTABLE_MANIFEST.json", manifest)
        (REVIEW / "V4_1_PROTOCOL_DRAFT.md").write_text(
            "# Q2 V4.1 — 31-safe-bank protocol draft\n\n"
            "Status: DRAFT / AWAITING PRINCIPAL_RESEARCHER_FREEZE.\n\n"
            "Use all 31 frozen V4 candidates in original generation order, at "
            "MEDIUM and STRONG shells, with baseline: 63 conditions, N=300, "
            "two independent rollouts, and 37,800 future semantic trajectories. "
            "Preserve the V4 endpoint, A0/A1/A2, controller-label QAP, shell "
            "coupling, and no post-hoc controller selection. This review did "
            "not execute the semantic experiment.\n",
            encoding="utf-8",
        )
    else:
        write_json(REVIEW / "FUTURE_RESERVE_RECOMMENDATION.json", {
            "decision": decision["decision"],
            "recommendation": "Use a larger prospectively frozen reserve in a future design; do not generate candidates in V4.1.",
            "candidate_counts_for_95pct": reserve_fragility()["rows"],
        })
    return decision


def write_reports(
    candidates: list[dict[str, Any]],
    safe: list[dict[str, Any]],
    reconstructed: dict[str, Any],
    structure: dict[str, Any],
    rows: list[dict[str, Any]],
    reserve: dict[str, Any],
    decision: dict[str, Any],
    replicates: int,
) -> None:
    write_json(REVIEW / "RESERVE_FRAGILITY.json", reserve)
    write_json(REVIEW / "SAFETY_STRUCTURE.json", structure)
    write_json(REVIEW / "ZERO_INFERENCE_AUDIT.json", {
        "new_gpu_inference": False,
        "new_model_inference": False,
        "correctness_inspected": False,
        "a1_a2_new_computation": False,
        "semantic_outcomes": 0,
        "q3": "NOT_RUN",
        "spark1_used": False,
        "spark2_used": False,
        "runpod_used": False,
        "historical_v4_result_unchanged": True,
        "historical_v4_classification": V4_CLASSIFICATION,
    })
    all_geometry = reconstructed["all_geometry"]
    safe_geometry = reconstructed["safe_geometry"]
    p25 = _find(rows, k=31, n=300, rho=0.25)
    p32 = _find(rows, k=32, n=300, rho=0.25)
    fragility_lines = [
        f"| {entry['p_safe']:.3f} | {entry['probability_at_least_minimum']:.6f} | {entry['minimum_candidates_for_95pct']} |"
        for entry in reserve["rows"]
    ]
    (REVIEW / "Q2_V4_1_SAFETY_ATTRITION_GEOMETRY.md").write_text(
        "# Q2 V4.1 — safety-attrition geometry\n\n"
        "The 40 candidates and 31-safe subset were reconstructed from immutable "
        "V4 coefficient and label-free shell-safety artifacts. No correctness "
        "or semantic outcome was read.\n\n"
        f"- 40-bank rank/effective/stable/condition: {all_geometry['rank']} / {all_geometry['effective_rank']:.6f} / {all_geometry['stable_rank']:.6f} / {all_geometry['condition_number']:.6f}\n"
        f"- 31-safe rank/effective/stable/condition: {safe_geometry['rank']} / {safe_geometry['effective_rank']:.6f} / {safe_geometry['stable_rank']:.6f} / {safe_geometry['condition_number']:.6f}\n"
        f"- unsafe IDs: {', '.join(reconstructed['unsafe_ids'])}\n"
        f"- safe centroid difference from unsafe: {structure['centroid_difference_norm']:.6f}; permutation p={structure['permutation']['p_value_plus_one']:.6f}\n"
        f"- safe-bank coverage gate: {'PASS' if reconstructed['safe_coverage_checks']['pass'] else 'FAIL'}\n\n"
        "The nearest-neighbor angular statistic is a coverage proxy, not a "
        "claim of a canonical 8D spherical covering radius. Coordinate loading, "
        "sign balance, leverage, anisotropy, and safety-label separation are "
        "descriptive and were not used to select a subset.\n",
        encoding="utf-8",
    )
    (REVIEW / "Q2_V4_1_K31_POWER_PRECISION.md").write_text(
        "# Q2 V4.1 — K=31 power and precision review\n\n"
        f"CPU-only synthetic planning used {replicates} repetitions per cell and "
        f"{PLANNING_QAP_MAPS} QAP maps per repetition, preserving the exact V4 "
        f"endpoint, permutation structure, and maxT multiplicity. The final "
        f"protocol remains {FINAL_QAP_MAPS} maps.\n\n"
        f"At rho=0.25, K31/N300 omnibus power={p25['omnibus_rate']:.4f}, "
        f"A2 attribution={p25['a2_attribution_rate']:.4f}, radial={p25['radial_rate']:.4f}; "
        f"K32/N300 omnibus power={p32['omnibus_rate']:.4f}. "
        f"Absolute omnibus difference={p32['omnibus_rate'] - p25['omnibus_rate']:.4f}; "
        f"A2 MC-width ratio={p25['a2_rho_mc95_width'] / p32['a2_rho_mc95_width']:.4f}.\n\n"
        "The complete machine-readable tables include all requested rho values, "
        "A0/A1/A2 attribution, G3/A2 superiority, radial total/shape planning "
        "power proxy, Monte Carlo uncertainty, and secondary N=400 sensitivity "
        "when enabled. These are planning results and contain no semantic outcomes.\n",
        encoding="utf-8",
    )
    reserve_table = "\n".join(fragility_lines)
    (REVIEW / "Q2_V4_1_31_SAFE_BANK_REVIEW.md").write_text(
        "# Q2 V4.1 — Frozen 31-safe-bank adequacy review\n\n"
        f"Historical V4 classification remains {V4_CLASSIFICATION} and "
        "Q2_V4_PRESEMANTIC_FORENSIC_CLEAN. The V4 relational hypothesis "
        "remains untested.\n\n"
        "## Decision\n\n"
        f"{decision['decision']}\n\n"
        "The decision was applied mechanically after the design precheck was "
        "frozen. All 31 safe directions were retained in original candidate "
        "order; no direction was redrawn, added, removed, or optimized.\n\n"
        "## Safety attrition\n\n"
        f"40 total, 31 safe at both shells, 9 unsafe: {', '.join(reconstructed['unsafe_ids'])}.\n\n"
        "| Metric | 40 candidates | 31 safe | V4.1 gate | Result |\n|---|---:|---:|---:|---|\n"
        f"| Rank | {all_geometry['rank']} | {safe_geometry['rank']} | full rank 8 | {'PASS' if safe_geometry['rank'] == 8 else 'FAIL'} |\n"
        f"| Effective rank | {all_geometry['effective_rank']:.6f} | {safe_geometry['effective_rank']:.6f} | >= 6.0 | {'PASS' if safe_geometry['effective_rank'] >= 6.0 else 'FAIL'} |\n"
        f"| Stable rank | {all_geometry['stable_rank']:.6f} | {safe_geometry['stable_rank']:.6f} | descriptive | — |\n"
        f"| Condition number | {all_geometry['condition_number']:.6f} | {safe_geometry['condition_number']:.6f} | <= 3 | {'PASS' if safe_geometry['condition_number'] <= 3 else 'FAIL'} |\n"
        f"| Max abs pair cosine | {all_geometry['pairwise_absolute_cosine']['max']:.6f} | {safe_geometry['pairwise_absolute_cosine']['max']:.6f} | < 0.98 | {'PASS' if safe_geometry['pairwise_absolute_cosine']['max'] < 0.98 else 'FAIL'} |\n"
        f"| A0 q90-q10 | {all_geometry['a0_angular_chord_squared']['q90'] - all_geometry['a0_angular_chord_squared']['q10']:.6f} | {safe_geometry['a0_angular_chord_squared']['q90'] - safe_geometry['a0_angular_chord_squared']['q10']:.6f} | >= 0.20 | {'PASS' if safe_geometry['a0_angular_chord_squared']['q90'] - safe_geometry['a0_angular_chord_squared']['q10'] >= 0.20 else 'FAIL'} |\n"
        f"| Shell amplitude CV max | — | {max(reconstructed['safe_coverage_checks']['shell_amplitude_cv']):.8f} | <= 0.03 | {'PASS' if reconstructed['safe_coverage_checks']['pass'] else 'FAIL'} |\n\n"
        "The 31-safe bank is conditioned on the frozen safety gate and is not "
        "claimed to be unconditionally isotropic. Safety-label separation is "
        f"descriptive: centroid distance={structure['centroid_difference_norm']:.6f}, "
        f"permutation p={structure['permutation']['p_value_plus_one']:.6f}, "
        "with high uncertainty at n=40.\n\n"
        "## Reserve fragility\n\n"
        "| p_safe | P(#safe >= 32 | 40,p) | candidates for >=95% |\n|---:|---:|---:|\n"
        f"{reserve_table}\n\n"
        "## Scientific firewall\n\n"
        "New GPU inference: NONE. New model inference: NONE. Correctness "
        "inspected: NO. A1/A2 new computation: NONE. Semantic outcomes: 0. "
        "Q3: NOT RUN. The original V4 classification is immutable.\n",
        encoding="utf-8",
    )
    write_json(REVIEW / "FORENSIC_AUDIT.json", {
        "classification": "Q2_V4_1_FORENSIC_CLEAN",
        "recomputed_from": [
            "CANDIDATE_BANK_MANIFEST.json",
            "CANDIDATE_SAFETY_REPORT.json",
        ],
        "candidate_count": len(candidates),
        "safe_count": len(safe),
        "decision": decision["decision"],
        "semantic_outcomes": 0,
        "gpu_used": False,
        "model_inference": False,
        "correctness_inspected": False,
        "historical_classification_unchanged": V4_CLASSIFICATION,
        "independent_recomputation": "direct geometry, binomial, and planning-statistic recomputation",
    })
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Q2 V4.1 forensic audit\n\n"
        "Q2_V4_1_FORENSIC_CLEAN.\n\n"
        "The audit independently loaded the two immutable V4 bank artifacts, "
        "reconstructed all 40 rows and the 31 joint-safe rows in candidate "
        "order, recomputed the coefficient geometry, safety conditioning "
        "diagnostics, reserve probabilities, and decision inputs. No model "
        "runner, GPU, semantic journal, correctness label, A1/A2 outcome, or "
        "semantic panel was accessed. The historical V4 classification remains "
        f"{V4_CLASSIFICATION}.\n",
        encoding="utf-8",
    )
    file_hashes = {
        name: sha256_file(REVIEW / name)
        for name in sorted(
            path.name for path in REVIEW.iterdir()
            if path.is_file() and path.name not in {"artifact_hashes.json", "MANIFEST_AND_HASHES.json"}
        )
    }
    write_json(REVIEW / "artifact_hashes.json", file_hashes)
    write_json(REVIEW / "MANIFEST_AND_HASHES.json", {
        "review_artifact_hash": sha256_json(file_hashes),
        "historical_candidate_manifest_sha256": sha256_file(CANDIDATE_MANIFEST),
        "historical_safety_report_sha256": sha256_file(SAFETY_REPORT),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck-only", action="store_true")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--no-n400", action="store_true")
    args = parser.parse_args()
    if args.replicates < 20:
        raise SystemExit("replicates must be at least 20")
    write_design_precheck()
    if args.precheck_only:
        print(json.dumps({"status": "PRECHECK_FROZEN", "review": str(REVIEW)}, indent=2))
        return
    candidates, safe, reconstructed = reconstruct_bank()
    structure = safety_structure(
        np.asarray([row["coefficients"] for row in candidates], dtype=np.float64),
        np.asarray([row["joint_safe"] for row in candidates], dtype=bool),
        permutations=10_000,
        bootstrap=5_000,
        seed=SEED,
    )
    rows, _superiority_rows = run_power(args.replicates, include_n400=not args.no_n400)
    reserve = reserve_fragility()
    decision = apply_decision(reconstructed, rows, include_n400=not args.no_n400)
    write_reports(
        candidates,
        safe,
        reconstructed,
        structure,
        rows,
        reserve,
        decision,
        args.replicates,
    )
    print(json.dumps({
        "classification": decision["decision"],
        "safe_count": len(safe),
        "replicates": args.replicates,
        "power_rows": len(rows),
        "review": str(REVIEW),
    }, indent=2))


if __name__ == "__main__":
    main()
