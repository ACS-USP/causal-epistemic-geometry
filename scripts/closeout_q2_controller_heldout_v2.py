#!/usr/bin/env python3
"""Materialize the authorized descriptive Q2-V2 closeout evidence vector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"


def read_json(name: str) -> Any:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> None:
    path = REVIEW / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def ranks(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    output = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        output[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return output


def spearman(left: list[float] | np.ndarray, right: list[float] | np.ndarray) -> float | None:
    left_rank, right_rank = ranks(left), ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def upper_edges(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), k=1)]


def aggregate_metric(prediction: dict[str, Any], name: str) -> dict[str, float]:
    return prediction[name]["aggregate"][name]


def main() -> int:
    lock = read_json("V2_FINAL_PROTOCOL_LOCK.json")
    calibration = read_json("V2_DOSE_CALIBRATION.json")["controllers"]
    conditions = read_json("V2_CONDITION_SUMMARY.json")
    estimands = read_json("V2_ESTIMANDS.json")
    prediction = read_json("V2_PREDICTION_RESULTS.json")
    classification = read_json("V2_CLASSIFICATION.json")
    bootstrap = read_json("V2_BOOTSTRAP_INTERVALS.json")
    geometry = read_json("V2_GEOMETRY_METRICS.json")
    distance_payload = read_json("V2_D_MATRIX.json")

    names = list(lock["meaningful_controllers"])
    nulls = list(lock["random_controllers"])
    baseline = conditions["BASELINE"]
    distance = np.asarray(distance_payload["values"], dtype=np.float64)
    if names != distance_payload["controllers"]:
        raise RuntimeError("Q2 V2 descriptive closeout controller order mismatch")

    selected = {}
    for name in names:
        dose_name = lock["meaningful_controllers"][name]["selected_dose"]
        dose = calibration[name]["doses"][dose_name]
        selected[name] = {
            "dose": dose_name,
            "delta_norm": float(dose["delta_norm"]),
            "raw_sequence_movement": float(dose["raw_sequence_movement"]),
            "semantic_movement": float(dose["semantic_movement"]),
            "mean_token_delta": float(dose["mean_token_delta"]),
            "causal_pass": bool(dose["causal_pass"]),
            "safe_pass": bool(dose["safe_pass"]),
        }

    movement = [selected[name]["semantic_movement"] for name in names]
    raw_movement = [selected[name]["raw_sequence_movement"] for name in names]
    delta_norms = [selected[name]["delta_norm"] for name in names]
    mean_error_distance = [
        float(np.mean(np.delete(distance[index], index))) for index in range(len(names))
    ]
    accuracy_change = [conditions[name]["accuracy"] - baseline["accuracy"] for name in names]
    g_values = [float(estimands[name]["G"]) for name in names]
    c_values = [float(estimands[name]["C"]) for name in names]
    d_values = [float(estimands[name]["D"]) for name in names]

    dose_bins: dict[str, dict[str, Any]] = {}
    for dose_name in sorted({selected[name]["dose"] for name in names}):
        members = [name for name in names if selected[name]["dose"] == dose_name]
        dose_bins[dose_name] = {
            "controllers": members,
            "count": len(members),
            "delta_norm": summarize([selected[name]["delta_norm"] for name in members]),
            "calibration_semantic_movement": summarize(
                [selected[name]["semantic_movement"] for name in members]
            ),
            "common_panel_accuracy_change": summarize(
                [conditions[name]["accuracy"] - baseline["accuracy"] for name in members]
            ),
            "common_panel_G": summarize([estimands[name]["G"] for name in members]),
            "common_panel_C": summarize([estimands[name]["C"] for name in members]),
            "common_panel_D": summarize([estimands[name]["D"] for name in members]),
        }

    families: dict[str, dict[str, Any]] = {}
    for family in lock["source_families"]:
        members = [
            name
            for name in names
            if lock["meaningful_controllers"][name]["source_axis"] == family
        ]
        families[family] = {
            "controllers": members,
            "accuracy_change": summarize(
                [conditions[name]["accuracy"] - baseline["accuracy"] for name in members]
            ),
            "calibration_semantic_movement": summarize(
                [selected[name]["semantic_movement"] for name in members]
            ),
            "folds": {
                metric: prediction[metric]["folds"][family]["metrics"][metric]
                for metric in prediction
            },
        }

    null_summary = {
        "accuracy_change": summarize(
            [conditions[name]["accuracy"] - baseline["accuracy"] for name in nulls]
        ),
        "commitment_validity_change": summarize(
            [
                conditions[name]["commitment_validity"] - baseline["commitment_validity"]
                for name in nulls
            ]
        ),
        "semantic_evaluability_change": summarize(
            [
                conditions[name]["semantic_evaluability"] - baseline["semantic_evaluability"]
                for name in nulls
            ]
        ),
        "G": summarize([estimands[name]["G"] for name in nulls]),
        "C": summarize([estimands[name]["C"] for name in nulls]),
        "D": summarize([estimands[name]["D"] for name in nulls]),
        "rescue": summarize([estimands[name]["rescue"] for name in nulls]),
        "damage": summarize([estimands[name]["damage"] for name in nulls]),
        "token_change": summarize(
            [conditions[name]["mean_tokens"] - baseline["mean_tokens"] for name in nulls]
        ),
    }

    global_pair_association = {
        metric: spearman(upper_edges(np.asarray(values)), upper_edges(distance))
        for metric, values in geometry.items()
    }
    result = {
        "schema_version": "q2-v2-closeout-evidence-v1",
        "scientific_role": "DEVELOPMENT",
        "classification": classification["classification"],
        "classification_thresholds_unchanged": True,
        "primary": {
            metric: {
                **aggregate_metric(prediction, metric),
                "bootstrap_mean_spearman_interval_95": bootstrap["metrics"][metric][
                    "mean_spearman"
                ]["interval_95"],
                "bootstrap_standardized_rmse_interval_95": bootstrap["metrics"][metric][
                    "standardized_rmse"
                ]["interval_95"],
            }
            for metric in prediction
        },
        "controller_causal_movement": {
            "selected_dose_semantic_movement": summarize(movement),
            "selected_dose_raw_sequence_movement": summarize(raw_movement),
            "causal_pass_count": sum(int(selected[name]["causal_pass"]) for name in names),
            "safe_pass_count": sum(int(selected[name]["safe_pass"]) for name in names),
            "per_controller": selected,
        },
        "radial_dose_behavior": {
            "selected_delta_norm": summarize(delta_norms),
            "dose_bins": dose_bins,
            "spearman_delta_norm_vs_mean_error_distance": spearman(
                delta_norms, mean_error_distance
            ),
            "spearman_delta_norm_vs_absolute_accuracy_change": spearman(
                delta_norms, np.abs(accuracy_change)
            ),
        },
        "angular_directional_behavior": {
            "global_pairwise_spearman_geometry_vs_error_distance": global_pair_association,
            "note": "Descriptive all-edge association; family-held-out estimates remain primary.",
        },
        "movement_vs_error_relation": {
            "spearman_calibration_movement_vs_mean_error_distance": spearman(
                movement, mean_error_distance
            ),
            "spearman_calibration_movement_vs_absolute_G": spearman(movement, np.abs(g_values)),
            "spearman_calibration_movement_vs_absolute_C": spearman(movement, np.abs(c_values)),
            "spearman_calibration_movement_vs_absolute_D": spearman(movement, np.abs(d_values)),
            "interpretation_role": "DESCRIPTIVE_NOT_CONTROLLER_SELECTION",
        },
        "family_behavior": families,
        "null_controls": null_summary,
        "safety_competence": {
            "baseline": baseline,
            "meaningful_accuracy_change": summarize(accuracy_change),
            "meaningful_commitment_validity": summarize(
                [conditions[name]["commitment_validity"] for name in names]
            ),
            "meaningful_semantic_evaluability": summarize(
                [conditions[name]["semantic_evaluability"] for name in names]
            ),
            "meaningful_G": summarize(g_values),
            "meaningful_C": summarize(c_values),
            "meaningful_D": summarize(d_values),
        },
        "metric_dissociation": {
            "M2_stronger_than_M0_M1_on_rho": aggregate_metric(
                prediction, "M2_FINITE_SECANT"
            )["mean_spearman"]
            > max(
                aggregate_metric(prediction, "M0_FLAT")["mean_spearman"],
                aggregate_metric(prediction, "M1_WHITENED")["mean_spearman"],
            ),
            "M2_qap_passes_0_05": aggregate_metric(prediction, "M2_FINITE_SECANT")[
                "qap_p_one_sided"
            ]
            <= 0.05,
            "M2_rho_passes_0_30": aggregate_metric(prediction, "M2_FINITE_SECANT")[
                "mean_spearman"
            ]
            >= 0.30,
            "M2_rmse_ratio_passes_0_90": aggregate_metric(
                prediction, "M2_FINITE_SECANT"
            )["rmse_ratio_to_constant"]
            <= 0.90,
            "composite_classification_preserved": classification["classification"],
        },
        "claim_boundary": {
            "Q1": "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL_UNCHANGED",
            "Q3": "NOT_RUN",
            "confirmatory": False,
            "next_experiment_executed": False,
        },
    }
    write_json("V2_CLOSEOUT_EVIDENCE.json", result)
    write_json("V2_EVIDENCE_VECTOR.json", result)
    print(json.dumps({"classification": result["classification"], "families": len(families)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
