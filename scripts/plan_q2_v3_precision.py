#!/usr/bin/env python3
"""Empirical, dependence-preserving precision planning for draft Q2 V3.

The simulation uses Q2 V2 only as a development prior.  It resamples items and
omits one entire controller family per replicate; it is not a confirmatory power
calculation and cannot authorize Q2 V3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review/q2_controller_bank_v2"
OUTPUT = ROOT / "review/q2_v2_principal_researcher_review"
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_exploratory import (  # noqa: E402
    fit_linear,
    linear_predict,
    pair_feature_matrix,
    spearman,
    unbiased_error_distance,
)

SIMULATIONS = 2_000
SEED = 2026082501
SAMPLE_SIZES = (120, 160, 200, 240)
DOSE_FRACTIONS = {"D_LOW": 0.25, "D_MEDIUM": 0.50, "D_HIGH": 0.75, "D_VERY_HIGH": 1.0}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line)["row"])
    return rows


def edge_matrix(matrices: list[np.ndarray], edges: list[tuple[int, int]]) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray([matrix[left, right] for left, right in edges], dtype=np.float64)
            for matrix in matrices
        ]
    )


def percentile(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> int:
    lock = read_json(SOURCE / "V2_FINAL_PROTOCOL_LOCK.json")
    manifest = read_json(SOURCE / "V2_COMMON_PANEL_MANIFEST.json")
    names = list(lock["meaningful_controllers"])
    metadata = lock["meaningful_controllers"]
    families = sorted({metadata[name]["source_axis"] for name in names})
    family_by_index = [metadata[name]["source_axis"] for name in names]
    rows = read_rows(SOURCE / "V2_COMMON_PANEL_JOURNAL.jsonl")
    lookup = {
        (row["item_id"], row["condition"], int(row["rollout_index"])): row for row in rows
    }
    errors = np.asarray(
        [
            [
                [int(not lookup[(item, name, rollout)]["correct"]) for rollout in (0, 1)]
                for item in manifest["item_ids"]
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    target = unbiased_error_distance(errors)
    m2 = np.asarray(read_json(SOURCE / "V2_GEOMETRY_METRICS.json")["M2_FINITE_SECANT"])
    delta = [metadata[name]["delta_norm"] for name in names]
    dose = [DOSE_FRACTIONS[metadata[name]["selected_dose"]] for name in names]
    nuisance_matrices = [
        pair_feature_matrix(delta, "absolute_difference"),
        pair_feature_matrix(delta, "mean"),
        pair_feature_matrix(dose, "absolute_difference"),
        pair_feature_matrix(dose, "mean"),
    ]
    all_edges = [
        (left, right)
        for left in range(len(names))
        for right in range(left + 1, len(names))
    ]
    all_nuisance = edge_matrix(nuisance_matrices, all_edges)
    all_m2 = edge_matrix([m2], all_edges)
    all_target = np.asarray([target[left, right] for left, right in all_edges])
    nuisance_coefficients = fit_linear(all_nuisance, all_target)
    augmented_coefficients = fit_linear(
        np.column_stack([all_nuisance, all_m2]), all_target
    )

    rng = np.random.default_rng(SEED)
    output: dict[str, Any] = {}
    for sample_size in SAMPLE_SIZES:
        residual_rho = np.empty(SIMULATIONS)
        rmse_ratio = np.empty(SIMULATIONS)
        positive_families = np.empty(SIMULATIONS)
        nuisance_rmse = np.empty(SIMULATIONS)
        augmented_rmse = np.empty(SIMULATIONS)
        for simulation in range(SIMULATIONS):
            item_indices = rng.integers(0, errors.shape[1], size=sample_size)
            sampled_target = unbiased_error_distance(errors[:, item_indices, :])
            selected_families = set(rng.choice(families, size=5, replace=False))
            selected = [
                index for index, family in enumerate(family_by_index) if family in selected_families
            ]
            edges = [
                (left, right)
                for offset, left in enumerate(selected)
                for right in selected[offset + 1 :]
                if family_by_index[left] != family_by_index[right]
            ]
            nuisance_x = edge_matrix(nuisance_matrices, edges)
            augmented_x = np.column_stack([nuisance_x, edge_matrix([m2], edges)])
            observed = np.asarray([sampled_target[left, right] for left, right in edges])
            nuisance_prediction = linear_predict(nuisance_x, nuisance_coefficients)
            augmented_prediction = linear_predict(augmented_x, augmented_coefficients)
            nuisance_rmse[simulation] = np.sqrt(
                np.mean(np.square(observed - nuisance_prediction))
            )
            augmented_rmse[simulation] = np.sqrt(
                np.mean(np.square(observed - augmented_prediction))
            )
            rmse_ratio[simulation] = augmented_rmse[simulation] / nuisance_rmse[simulation]
            residual_rho[simulation] = spearman(
                augmented_prediction - nuisance_prediction,
                observed - nuisance_prediction,
            )
            family_passes = 0
            for family in selected_families:
                family_edges = [
                    edge
                    for edge in edges
                    if family_by_index[edge[0]] == family or family_by_index[edge[1]] == family
                ]
                family_nuisance = edge_matrix(nuisance_matrices, family_edges)
                family_augmented = np.column_stack(
                    [family_nuisance, edge_matrix([m2], family_edges)]
                )
                family_observed = np.asarray(
                    [sampled_target[left, right] for left, right in family_edges]
                )
                family_nuisance_prediction = linear_predict(
                    family_nuisance, nuisance_coefficients
                )
                family_augmented_prediction = linear_predict(
                    family_augmented, augmented_coefficients
                )
                family_ratio = np.sqrt(
                    np.mean(np.square(family_observed - family_augmented_prediction))
                ) / np.sqrt(
                    np.mean(np.square(family_observed - family_nuisance_prediction))
                )
                family_rho = spearman(
                    family_augmented_prediction - family_nuisance_prediction,
                    family_observed - family_nuisance_prediction,
                )
                family_passes += int(family_ratio < 1.0 and family_rho > 0.0)
            positive_families[simulation] = family_passes
        pass_proxy = (
            (rmse_ratio <= 0.90) & (residual_rho >= 0.25) & (positive_families >= 4)
        )
        output[str(sample_size)] = {
            "residual_rho": {
                "median": float(np.median(residual_rho)),
                "interval_95": percentile(residual_rho),
                "interval_width": float(np.diff(percentile(residual_rho))[0]),
            },
            "augmented_to_nuisance_rmse_ratio": {
                "median": float(np.median(rmse_ratio)),
                "interval_95": percentile(rmse_ratio),
                "interval_width": float(np.diff(percentile(rmse_ratio))[0]),
            },
            "paired_rmse_improvement": {
                "median": float(np.median(nuisance_rmse - augmented_rmse)),
                "interval_95": percentile(nuisance_rmse - augmented_rmse),
            },
            "families_positive": {
                "median": float(np.median(positive_families)),
                "fraction_at_least_4_of_5": float(np.mean(positive_families >= 4)),
            },
            "empirical_V2_prior_pass_proxy_fraction": float(np.mean(pass_proxy)),
        }
    write_json(
        OUTPUT / "Q2_V3_PRECISION_SIMULATION.json",
        {
            "schema_version": "q2-v3-development-planning-simulation-v1",
            "status": "DESIGN_ONLY_NOT_A_POWER_GUARANTEE",
            "simulations_per_sample_size": SIMULATIONS,
            "seed": SEED,
            "dependence_preserved": {
                "items": "resampled as clusters with both rollouts and all controllers",
                "controllers": "all four controllers travel with each sampled family",
                "families": "five of six complete V2 families sampled without replacement",
                "dyads": "recomputed from controller error vectors; never treated as independent",
            },
            "important_limitation": (
                "The fixed mappings and empirical effects come from V2 controllers, so this is "
                "optimistic planning input, not evidence about genuinely new V3 families."
            ),
            "candidate_sample_sizes": output,
            "selected_draft_n": 200,
            "selection_rationale": (
                "N=200 materially narrows item noise relative to V2 while family novelty, not "
                "item count, remains the dominant uncertainty."
            ),
        },
    )
    print(json.dumps({"phase": "q2_v3_precision_plan", "simulations": SIMULATIONS}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
