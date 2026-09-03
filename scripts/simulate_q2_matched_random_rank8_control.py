#!/usr/bin/env python3
"""CPU-only planning analysis for the matched random rank-8 control.

The script never constructs an ambient random basis. It audits the already
frozen coefficient identities and simulates only subspace-level scalar
statistics for prospective power/compute planning.
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

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q2_matched_random_rank8_control_design"
PRECHECK = REVIEW / "PLANNING_PRECHECK.json"
HIST_PROTOCOL = ROOT / "review" / "q2_v4_1_prediction_lock" / "PROTOCOL_LOCK.json"
HIST_BANK = ROOT / "review" / "q2_v4_spark1_presemantic" / "CANDIDATE_BANK_MANIFEST.json"
FRESH_BANK = (
    ROOT
    / "review"
    / "q2_oos_fresh_controller_design"
    / "v2_final_presemantic"
    / "V2_CANDIDATE_BANK_MANIFEST.json"
)
FRESH_SELECTED = (
    ROOT
    / "review"
    / "q2_oos_fresh_controller_design"
    / "v2_presemantic_closeout"
    / "V2_SELECTED_CONTROLLER_BANK.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array.astype("<f8", copy=False))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2))
    return center - half / denominator, center + half / denominator


def coefficient_audit() -> dict[str, Any]:
    protocol = read_json(HIST_PROTOCOL)
    historical_manifest = read_json(HIST_BANK)
    fresh_manifest = read_json(FRESH_BANK)
    selected = read_json(FRESH_SELECTED)

    historical_order = protocol["controller_order"]
    historical_by_id = {row["candidate_id"]: row for row in historical_manifest["candidates"]}
    fresh_order = selected["selected_ids"]
    fresh_by_id = {row["candidate_id"]: row for row in fresh_manifest["candidates"]}

    coefficients: list[list[float]] = []
    identities: list[dict[str, Any]] = []
    for population, order, source in (
        ("HISTORICAL_REFERENCE", historical_order, historical_by_id),
        ("FRESH", fresh_order, fresh_by_id),
    ):
        for controller_id in order:
            row = source[controller_id]
            coefficient = np.asarray(row["coefficients"], dtype=np.float64)
            coefficients.append(coefficient.tolist())
            identities.append(
                {
                    "population": population,
                    "controller_id": controller_id,
                    "coefficient_norm": float(np.linalg.norm(coefficient)),
                    "source_vector_sha256": row["file_sha256"],
                }
            )

    matrix = np.asarray(coefficients, dtype=np.float64)
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    gram = normalized @ normalized.T
    a0_chord = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * gram))
    identity_blob = "\n".join(row["controller_id"] for row in identities).encode()
    return {
        "schema_version": "q2-matched-random-rank8-coefficient-audit-v1",
        "status": "EXACT_47_COEFFICIENT_IDENTITIES_RECONSTRUCTED",
        "historical_count": len(historical_order),
        "fresh_count": len(fresh_order),
        "total_count": len(identities),
        "coefficient_dimension": int(matrix.shape[1]),
        "coefficient_matrix_float64_sha256": canonical_sha(matrix),
        "normalized_gram_float64_sha256": canonical_sha(gram),
        "a0_chord_float64_sha256": canonical_sha(a0_chord),
        "ordered_identity_sha256": hashlib.sha256(identity_blob).hexdigest(),
        "minimum_coefficient_norm": float(np.min(np.linalg.norm(matrix, axis=1))),
        "maximum_coefficient_norm": float(np.max(np.linalg.norm(matrix, axis=1))),
        "maximum_gram_asymmetry": float(np.max(np.abs(gram - gram.T))),
        "maximum_gram_diagonal_error": float(np.max(np.abs(np.diag(gram) - 1.0))),
        "mapping_identity": "v_k_random = Q_random c_k",
        "orthonormal_mapping_implication": (
            "Any 4096x8 basis with Q_random.T @ Q_random = I preserves this normalized Gram "
            "matrix and coefficient-space A0 exactly up to numerical tolerance."
        ),
        "identities": identities,
        "source_hashes": {
            str(HIST_PROTOCOL.relative_to(ROOT)): text_sha(HIST_PROTOCOL),
            str(HIST_BANK.relative_to(ROOT)): text_sha(HIST_BANK),
            str(FRESH_SELECTED.relative_to(ROOT)): text_sha(FRESH_SELECTED),
            str(FRESH_BANK.relative_to(ROOT)): text_sha(FRESH_BANK),
        },
        "final_random_basis_generated": 0,
        "semantic_outcomes_used": 0,
    }


def simulate_cell(
    rng: np.random.Generator,
    *,
    replicates: int,
    subspaces: int,
    measurement_sd: float,
    subspace_sd: float,
    random_mean: float,
    advantage: float,
    common_correlation: float,
    alpha: float,
) -> dict[str, float]:
    common = rng.standard_normal((replicates, 1))
    independent_random = rng.standard_normal((replicates, subspaces))
    independent_learned = rng.standard_normal(replicates)
    random_latent = random_mean + subspace_sd * rng.standard_normal((replicates, subspaces))
    learned_latent = random_mean + advantage + subspace_sd * rng.standard_normal(replicates)
    shared_scale = math.sqrt(common_correlation)
    independent_scale = math.sqrt(1.0 - common_correlation)
    random_noise = measurement_sd * (
        shared_scale * common + independent_scale * independent_random
    )
    learned_noise = measurement_sd * (
        shared_scale * common[:, 0] + independent_scale * independent_learned
    )
    random_stats = np.clip(random_latent + random_noise, -0.999, 0.999)
    learned_stats = np.clip(learned_latent + learned_noise, -0.999, 0.999)
    exceedances = np.sum(random_stats >= learned_stats[:, None], axis=1)
    p_values = (1.0 + exceedances) / (subspaces + 1.0)
    rejects = p_values <= alpha
    successes = int(np.sum(rejects))
    low, high = wilson(successes, replicates)

    # Resolution after deleting one random subspace; the worst deletion removes
    # a non-exceeding random statistic. This is a sensitivity, not a new gate.
    worst_exceedances = exceedances
    p_after_one_deletion = (1.0 + worst_exceedances) / subspaces
    leave_one_stable = np.mean(rejects & (p_after_one_deletion <= alpha))
    random_mean_estimate = np.mean(random_stats, axis=1)
    contrast = learned_stats - random_mean_estimate
    return {
        "rejection_rate": float(np.mean(rejects)),
        "wilson_low": low,
        "wilson_high": high,
        "mean_contrast": float(np.mean(contrast)),
        "contrast_bias": float(np.mean(contrast) - advantage),
        "contrast_sd": float(np.std(contrast, ddof=1)),
        "random_mean_interval_width_approx": float(
            2.0 * 1.959963984540054 * np.std(random_stats, ddof=1) / math.sqrt(subspaces)
        ),
        "leave_one_subspace_rejection_stability": float(leave_one_stable),
        "minimum_attainable_p": 1.0 / (subspaces + 1.0),
        "minimum_attainable_p_after_one_deletion": 1.0 / subspaces,
    }


def planning_rows(precheck: dict[str, Any]) -> list[dict[str, Any]]:
    grid = precheck["planning_grid"]
    simulation = precheck["simulation"]
    rng = np.random.Generator(np.random.PCG64DXSM(simulation["planning_seed"]))
    observed = precheck["fixed_inputs"]["closed_learned_subspace_planning_statistic"]["value"]
    advantage_map = dict(grid["learned_advantage"])
    advantage_map["TWENTY_FIVE_PERCENT_OF_OBSERVED"] = 0.25 * observed
    advantage_map["FIFTY_PERCENT_OF_OBSERVED"] = 0.50 * observed
    rows: list[dict[str, Any]] = []
    for subspaces in grid["random_subspaces_S"]:
        for items in grid["items_N"]:
            for rollouts in grid["rollouts_R"]:
                for split in grid["controller_splits"]:
                    k_ref = split["K_reference"]
                    k_fresh = split["K_fresh"]
                    measurement_sd = simulation["baseline_measurement_sd_at_N300_R2_K31x16"]
                    measurement_sd *= math.sqrt(300.0 / items)
                    measurement_sd *= math.sqrt(2.0 / rollouts)
                    measurement_sd *= math.sqrt((31.0 * 16.0) / (k_ref * k_fresh))
                    for route, correlation in (
                        ("ROUTE_1", 0.0),
                        ("ROUTE_2", simulation["common_panel_measurement_correlation_route2"]),
                    ):
                        for subspace_sd in grid["subspace_sd"]:
                            for random_mean in grid["generic_random_alignment_mean"]:
                                for scenario, advantage in advantage_map.items():
                                    result = simulate_cell(
                                        rng,
                                        replicates=simulation["replicates_per_cell"],
                                        subspaces=subspaces,
                                        measurement_sd=measurement_sd,
                                        subspace_sd=subspace_sd,
                                        random_mean=random_mean,
                                        advantage=advantage,
                                        common_correlation=correlation,
                                        alpha=simulation["alpha"],
                                    )
                                    rows.append(
                                        {
                                            "route": route,
                                            "S": subspaces,
                                            "N": items,
                                            "R": rollouts,
                                            "K_reference": k_ref,
                                            "K_fresh": k_fresh,
                                            "controller_split_role": split["role"],
                                            "subspace_sd": subspace_sd,
                                            "generic_random_mean": random_mean,
                                            "scenario": scenario,
                                            "learned_advantage": advantage,
                                            "measurement_sd": measurement_sd,
                                            **result,
                                        }
                                    )
    return rows


def compute_rows(precheck: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = precheck["fixed_inputs"]["validated_semantic_runtime"]
    hours_per_row = runtime["wall_hours"] / runtime["rows"]
    rows: list[dict[str, Any]] = []
    for subspaces in precheck["planning_grid"]["random_subspaces_S"]:
        for items in precheck["planning_grid"]["items_N"]:
            for rollouts in precheck["planning_grid"]["rollouts_R"]:
                for split in precheck["planning_grid"]["controller_splits"]:
                    per_orientation = (
                        (split["K_reference"] + split["K_fresh"]) * 2 * items * rollouts
                    )
                    for route, arms in (
                        ("ROUTE_1", subspaces),
                        ("ROUTE_2", subspaces + 1),
                    ):
                        semantic_rows = per_orientation * arms
                        rows.append(
                            {
                                "route": route,
                                "S": subspaces,
                                "N": items,
                                "R": rollouts,
                                "K_reference": split["K_reference"],
                                "K_fresh": split["K_fresh"],
                                "controller_split_role": split["role"],
                                "rows_per_subspace_orientation": per_orientation,
                                "semantic_rows": semantic_rows,
                                "projected_spark1_hours": semantic_rows * hours_per_row,
                            }
                        )
    return rows


def safety_rows(precheck: dict[str, Any]) -> list[dict[str, Any]]:
    safety = precheck["safety_feasibility_scenarios"]
    fixed = safety["required_fixed_identities_per_subspace"]
    target = min(precheck["planning_grid"]["random_subspaces_S"])
    rows: list[dict[str, Any]] = []
    for probability in safety["per_controller_qualification_probability"]:
        all_pass = probability**fixed
        rows.append(
            {
                "per_controller_pass_probability": probability,
                "fixed_identities": fixed,
                "all_47_pass_probability_under_independence": all_pass,
                "expected_orientation_draws_for_20_qualified": target / all_pass,
                "safety_rows_per_evaluated_orientation": (
                    fixed * safety["safety_items"] * safety["shells"]
                ),
                "interpretation": "SENSITIVITY_ONLY_CONTROLLER_PASS_DEPENDENCE_UNKNOWN",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    precheck = read_json(PRECHECK)
    if precheck["status"] != "FROZEN_MODEL_FREE_PLANNING_PRECHECK":
        raise RuntimeError("MATCHED_RANDOM_RANK8_PRECHECK_NOT_FROZEN")
    if not precheck["prohibitions"]["generate_final_random_basis"]:
        raise RuntimeError("MATCHED_RANDOM_RANK8_RANDOM_BASIS_FIREWALL_FAILURE")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficient = coefficient_audit()
    power = planning_rows(precheck)
    compute = compute_rows(precheck)
    safety = safety_rows(precheck)
    (args.output_dir / "COEFFICIENT_GEOMETRY_AUDIT.json").write_text(
        json.dumps(coefficient, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "POWER_GRID.csv", power)
    write_csv(args.output_dir / "COMPUTE_GRID.csv", compute)
    write_csv(args.output_dir / "SAFETY_FEASIBILITY.csv", safety)
    summary = {
        "schema_version": "q2-matched-random-rank8-planning-results-v1",
        "status": "MODEL_FREE_PLANNING_SIMULATION_COMPLETE",
        "precheck_sha256": text_sha(PRECHECK),
        "power_grid_rows": len(power),
        "compute_grid_rows": len(compute),
        "safety_grid_rows": len(safety),
        "coefficient_audit_status": coefficient["status"],
        "simulation_replicates_per_cell": precheck["simulation"]["replicates_per_cell"],
        "final_random_bases_generated": 0,
        "experimental_seeds_generated": 0,
        "semantic_trajectories": 0,
        "qwen_loaded": False,
        "gpu_used": False,
    }
    (args.output_dir / "PLANNING_RESULTS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
