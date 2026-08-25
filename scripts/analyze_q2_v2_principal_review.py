#!/usr/bin/env python3
"""Post-hoc principal-researcher diagnostics for the frozen Q2 V2 result.

This script reads only completed Q2 V2 artifacts.  It cannot change the frozen
classification and does not expose any model-inference path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "review/q2_controller_bank_v2"
OUTPUT = ROOT / "review/q2_v2_principal_researcher_review"
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis.q2_exploratory import (  # noqa: E402
    family_heldout_incremental,
    family_heldout_predictions,
    fit_linear,
    linear_predict,
    pair_feature_matrix,
    pearson,
    spearman,
    unbiased_error_distance,
    upper_edges,
)
from epistemic_geometry.analysis.q2_geometries import (  # noqa: E402
    _js_from_logits,
    fit_whitening,
    flat_geometry,
    whitened_geometry,
)

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026082401
QAP_PERMUTATIONS = 10_000
QAP_SEED = 2026082402
DOSE_FRACTIONS = {"D_LOW": 0.25, "D_MEDIUM": 0.50, "D_HIGH": 0.75, "D_VERY_HIGH": 1.0}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_journal(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        wrapper = json.loads(line)
        row = wrapper.get("row")
        fields = wrapper.get("key_fields")
        if (
            wrapper.get("version") != "research-os-jsonl-v1"
            or not isinstance(row, dict)
            or not isinstance(fields, list)
            or wrapper.get("key") != [row[field] for field in fields]
        ):
            raise RuntimeError(f"invalid frozen journal envelope at line {line_number}")
        rows.append(row)
    return rows


def fast_fold_scores(
    target: np.ndarray,
    metric: np.ndarray,
    names: list[str],
    family_by_name: dict[str, str],
) -> tuple[float, float, float, dict[str, dict[str, float]]]:
    folds: dict[str, dict[str, float]] = {}
    for family in sorted(set(family_by_name.values())):
        train = [index for index, name in enumerate(names) if family_by_name[name] != family]
        test = [index for index, name in enumerate(names) if family_by_name[name] == family]
        train_y = upper_edges(target, train)
        train_x = upper_edges(metric, train)
        test_edges = [(left, right) for left in test for right in train]
        test_y = np.asarray([target[left, right] for left, right in test_edges])
        test_x = np.asarray([metric[left, right] for left, right in test_edges])
        prediction = linear_predict(test_x, fit_linear(train_x, train_y))
        constant = np.full(len(test_y), np.mean(train_y))
        folds[family] = {
            "spearman": spearman(test_x, test_y),
            "pearson": pearson(test_x, test_y),
            "rmse": float(np.sqrt(np.mean(np.square(prediction - test_y)))),
            "constant_rmse": float(np.sqrt(np.mean(np.square(constant - test_y)))),
        }
    mean_rho = float(np.nanmean([fold["spearman"] for fold in folds.values()]))
    mean_rmse = float(np.mean([fold["rmse"] for fold in folds.values()]))
    mean_constant = float(np.mean([fold["constant_rmse"] for fold in folds.values()]))
    return mean_rho, mean_rmse, mean_rmse / mean_constant, folds


def qap_p_values(
    target: np.ndarray,
    metrics: dict[str, np.ndarray],
    names: list[str],
    family_by_name: dict[str, str],
) -> dict[str, dict[str, float]]:
    observed = {
        name: fast_fold_scores(target, metric, names, family_by_name)[0]
        for name, metric in metrics.items()
    }
    null = {name: np.empty(QAP_PERMUTATIONS, dtype=np.float64) for name in metrics}
    families = sorted(set(family_by_name.values()))
    rng = np.random.default_rng(QAP_SEED)
    for permutation_index in range(QAP_PERMUTATIONS):
        order = np.arange(len(names))
        for family in families:
            indices = [index for index, name in enumerate(names) if family_by_name[name] == family]
            order[indices] = rng.permutation(indices)
        permuted = target[np.ix_(order, order)]
        for metric_name, metric in metrics.items():
            null[metric_name][permutation_index] = fast_fold_scores(
                permuted, metric, names, family_by_name
            )[0]
    return {
        metric_name: {
            "observed": observed[metric_name],
            "p_one_sided": float(
                (1 + np.sum(values >= observed[metric_name])) / (1 + len(values))
            ),
            "null_mean": float(np.mean(values)),
            "null_p95": float(np.quantile(values, 0.95)),
        }
        for metric_name, values in null.items()
    }


def controller_errors(
    rows: list[dict[str, Any]], item_ids: list[str], conditions: list[str]
) -> np.ndarray:
    lookup = {
        (row["item_id"], row["condition"], int(row["rollout_index"])): row for row in rows
    }
    return np.asarray(
        [
            [
                [int(not lookup[(item_id, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item_id in item_ids
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )


def controller_vectors(lock: dict[str, Any], names: list[str]) -> np.ndarray:
    metadata = {**lock["meaningful_controllers"], **lock["random_controllers"]}
    return np.stack(
        [
            np.load(ROOT / metadata[name]["path"], allow_pickle=False)
            .astype(np.float64)
            .reshape(-1)
            for name in names
        ]
    )


def all_controller_geometries(
    lock: dict[str, Any], names: list[str], meaningful_metrics: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    vectors = controller_vectors(lock, names)
    m0 = flat_geometry(vectors)["normalized_euclidean"]
    activations = np.load(SOURCE / "V2_COVARIANCE_ACTIVATIONS.npz", allow_pickle=False)[
        "activations"
    ]
    whitening = fit_whitening(
        activations.astype(np.float64),
        regularization_fraction=float(lock["geometry"]["M1"]["lambda"]),
    )
    m1 = whitened_geometry(vectors, whitening)["normalized_euclidean"]
    m2 = np.zeros((len(names), len(names)), dtype=np.float64)
    meaningful_count = len(lock["meaningful_controllers"])
    m2[:meaningful_count, :meaningful_count] = meaningful_metrics["M2_FINITE_SECANT"]
    required_pairs = [
        (left, right)
        for left in range(len(names))
        for right in range(left + 1, len(names))
        if left >= meaningful_count or right >= meaningful_count
    ]
    archive = read_json(SOURCE / "V2_FINITE_SECANT_ARCHIVE.json")
    sums = {(left, right): 0.0 for left, right in required_pairs}
    for record in archive["records"]:
        arrays = np.load(SOURCE / record["path"], allow_pickle=False)
        for left, right in required_pairs:
            sums[(left, right)] += _js_from_logits(arrays[names[left]], arrays[names[right]])
    for (left, right), value in sums.items():
        m2[left, right] = m2[right, left] = float(
            np.sqrt(value / len(archive["records"]))
        )
    return {"M0_FLAT": np.asarray(m0), "M1_WHITENED": np.asarray(m1), "M2_FINITE_SECANT": m2}


def bootstrap(
    errors: np.ndarray,
    metrics: dict[str, np.ndarray],
    names: list[str],
    family_by_name: dict[str, str],
) -> dict[str, Any]:
    samples = {
        metric: {
            "rho": np.empty(BOOTSTRAP_RESAMPLES),
            "rmse": np.empty(BOOTSTRAP_RESAMPLES),
            "rmse_ratio": np.empty(BOOTSTRAP_RESAMPLES),
        }
        for metric in metrics
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for sample_index in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, errors.shape[1], size=errors.shape[1])
        target = unbiased_error_distance(errors[:, indices, :])
        for metric_name, metric in metrics.items():
            rho, rmse, ratio, _folds = fast_fold_scores(
                target, metric, names, family_by_name
            )
            samples[metric_name]["rho"][sample_index] = rho
            samples[metric_name]["rmse"][sample_index] = rmse
            samples[metric_name]["rmse_ratio"][sample_index] = ratio

    arrays = {
        f"{metric}_{quantity}": values
        for metric, quantities in samples.items()
        for quantity, values in quantities.items()
    }
    np.savez_compressed(OUTPUT / "ITEM_BOOTSTRAP_SAMPLES.npz", **arrays)

    def interval(values: np.ndarray) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    contrasts = {
        "delta_rho_M2_minus_M0": (
            samples["M2_FINITE_SECANT"]["rho"] - samples["M0_FLAT"]["rho"]
        ),
        "delta_rho_M2_minus_M1": (
            samples["M2_FINITE_SECANT"]["rho"] - samples["M1_WHITENED"]["rho"]
        ),
        "delta_rmse_M0_minus_M2": (
            samples["M0_FLAT"]["rmse"] - samples["M2_FINITE_SECANT"]["rmse"]
        ),
        "delta_rmse_M1_minus_M2": (
            samples["M1_WHITENED"]["rmse"] - samples["M2_FINITE_SECANT"]["rmse"]
        ),
    }
    return {
        "schema_version": "q2-v2-posthoc-paired-item-bootstrap-v1",
        "post_hoc_exploratory": True,
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "item; all controllers and both rollouts move together",
        "metrics": {
            metric: {
                quantity: {
                    "median": float(np.median(values)),
                    "interval_95": interval(values),
                }
                for quantity, values in quantities.items()
            }
            for metric, quantities in samples.items()
        },
        "paired_contrasts": {
            name: {
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "interval_95": interval(values),
                "fraction_positive": float(np.mean(values > 0)),
                "confirmatory_p_value": None,
            }
            for name, values in contrasts.items()
        },
    }


def calibration_diagnostics(records: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric, frame in records.groupby("metric", sort=True):
        observed = frame["observed"].to_numpy(float)
        predicted = frame["predicted"].to_numpy(float)
        calibration = fit_linear(predicted, observed)
        calibrated = linear_predict(predicted, calibration)
        quadratic = np.column_stack([predicted, np.square(predicted)])
        quadratic_fit = linear_predict(quadratic, fit_linear(quadratic, observed))
        residual = observed - predicted
        output[metric] = {
            "directed_fold_edge_count": len(frame),
            "calibration_intercept": float(calibration[0]),
            "calibration_slope": float(calibration[1]),
            "observed_predicted_spearman": spearman(predicted, observed),
            "observed_predicted_pearson": pearson(predicted, observed),
            "residual_predicted_spearman": spearman(predicted, residual),
            "residual_absolute_predicted_spearman": spearman(
                predicted, np.abs(residual)
            ),
            "uncalibrated_rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "pooled_recalibrated_rmse_descriptive": float(
                np.sqrt(np.mean(np.square(observed - calibrated)))
            ),
            "pooled_quadratic_rmse_descriptive": float(
                np.sqrt(np.mean(np.square(observed - quadratic_fit)))
            ),
            "residual_quantiles": {
                key: float(np.quantile(residual, quantile))
                for key, quantile in (("p05", 0.05), ("median", 0.5), ("p95", 0.95))
            },
        }
    return output


def pair_table(
    names: list[str],
    target: np.ndarray,
    metrics: dict[str, np.ndarray],
    metadata: dict[str, Any],
    calibration: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            left_name, right_name = names[left], names[right]
            left_meta, right_meta = metadata[left_name], metadata[right_name]
            left_cal = calibration["controllers"][left_name]["doses"][
                left_meta["selected_dose"]
            ]
            right_cal = calibration["controllers"][right_name]["doses"][
                right_meta["selected_dose"]
            ]
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "same_family": left_meta["source_axis"] == right_meta["source_axis"],
                    "same_source_location": (
                        left_meta["source_location"] == right_meta["source_location"]
                    ),
                    "same_direction_base": (
                        left_meta["source_axis"] == right_meta["source_axis"]
                        and left_meta["source_location"] == right_meta["source_location"]
                    ),
                    "same_sign": left_meta["sign"] == right_meta["sign"],
                    "dose_left": left_meta["selected_dose"],
                    "dose_right": right_meta["selected_dose"],
                    "same_dose": left_meta["selected_dose"] == right_meta["selected_dose"],
                    "dose_difference": abs(
                        DOSE_FRACTIONS[left_meta["selected_dose"]]
                        - DOSE_FRACTIONS[right_meta["selected_dose"]]
                    ),
                    "dose_mean": 0.5
                    * (
                        DOSE_FRACTIONS[left_meta["selected_dose"]]
                        + DOSE_FRACTIONS[right_meta["selected_dose"]]
                    ),
                    "delta_norm_difference": abs(
                        left_meta["delta_norm"] - right_meta["delta_norm"]
                    ),
                    "delta_norm_mean": 0.5
                    * (left_meta["delta_norm"] + right_meta["delta_norm"]),
                    "semantic_movement_difference": abs(
                        left_cal["semantic_movement"] - right_cal["semantic_movement"]
                    ),
                    "semantic_movement_mean": 0.5
                    * (left_cal["semantic_movement"] + right_cal["semantic_movement"]),
                    "behavioral_D": float(target[left, right]),
                    **{metric: float(matrix[left, right]) for metric, matrix in metrics.items()},
                }
            )
    return pd.DataFrame(rows)


def nuisance_analysis(
    target: np.ndarray,
    metrics: dict[str, np.ndarray],
    names: list[str],
    metadata: dict[str, Any],
    family_by_name: dict[str, str],
) -> dict[str, Any]:
    delta = [metadata[name]["delta_norm"] for name in names]
    dose = [DOSE_FRACTIONS[metadata[name]["selected_dose"]] for name in names]
    nuisance = {
        "DELTA_NORM_DIFFERENCE": pair_feature_matrix(delta, "absolute_difference"),
        "DELTA_NORM_MEAN": pair_feature_matrix(delta, "mean"),
        "DOSE_FRACTION_DIFFERENCE": pair_feature_matrix(dose, "absolute_difference"),
        "DOSE_FRACTION_MEAN": pair_feature_matrix(dose, "mean"),
    }
    individual = {
        name: family_heldout_predictions(target, [matrix], names, family_by_name)[
            "aggregate"
        ]
        for name, matrix in nuisance.items()
    }
    combined = family_heldout_predictions(
        target, list(nuisance.values()), names, family_by_name
    )
    incremental = family_heldout_incremental(
        target,
        list(nuisance.values()),
        metrics["M2_FINITE_SECANT"],
        names,
        family_by_name,
    )
    observed_residual_rho = incremental["aggregate"]["mean_residual_spearman"]
    rng = np.random.default_rng(QAP_SEED + 1)
    qap_values = np.empty(QAP_PERMUTATIONS, dtype=np.float64)
    families = sorted(set(family_by_name.values()))
    for permutation_index in range(QAP_PERMUTATIONS):
        order = np.arange(len(names))
        for family in families:
            indices = [
                index for index, name in enumerate(names) if family_by_name[name] == family
            ]
            order[indices] = rng.permutation(indices)
        permuted_target = target[np.ix_(order, order)]
        qap_values[permutation_index] = family_heldout_incremental(
            permuted_target,
            list(nuisance.values()),
            metrics["M2_FINITE_SECANT"],
            names,
            family_by_name,
        )["aggregate"]["mean_residual_spearman"]
    return {
        "post_hoc_exploratory": True,
        "feature_set_frozen_for_this_review": list(nuisance),
        "individual_nuisance_baselines": individual,
        "combined_nuisance_baseline": combined["aggregate"],
        "M2_incremental_over_combined_nuisance": incremental,
        "M2_residual_family_QAP": {
            "permutations": QAP_PERMUTATIONS,
            "seed": QAP_SEED + 1,
            "observed_mean_residual_spearman": observed_residual_rho,
            "p_one_sided": float(
                (1 + np.sum(qap_values >= observed_residual_rho))
                / (1 + QAP_PERMUTATIONS)
            ),
            "null_mean": float(np.mean(qap_values)),
            "null_p95": float(np.quantile(qap_values, 0.95)),
            "confirmatory": False,
        },
        "claim_boundary": (
            "Diagnostics condition on simple scalar strength features; they do not create a "
            "new Q2 V2 pass/fail rule."
        ),
    }


def group_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in frame.groupby(column, sort=True):
        observed = group["behavioral_D"].to_numpy(float)
        predicted = group["M2_FINITE_SECANT"].to_numpy(float)
        result[str(value)] = {
            "pairs": len(group),
            "mean_behavioral_D": float(np.mean(observed)),
            "mean_M2_distance": float(np.mean(predicted)),
            "M2_spearman": spearman(predicted, observed),
            "M2_pearson": pearson(predicted, observed),
        }
    return result


def null_pair_analysis(
    names: list[str],
    meaningful_count: int,
    target: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            if left < meaningful_count and right < meaningful_count:
                pair_class = "MEANINGFUL_MEANINGFUL"
            elif left >= meaningful_count and right >= meaningful_count:
                pair_class = "NULL_NULL"
            else:
                pair_class = "MEANINGFUL_NULL"
            rows.append(
                {
                    "left": names[left],
                    "right": names[right],
                    "pair_class": pair_class,
                    "behavioral_D": float(target[left, right]),
                    **{metric: float(matrix[left, right]) for metric, matrix in metrics.items()},
                }
            )
    frame = pd.DataFrame(rows)
    result: dict[str, Any] = {
        "post_hoc_exploratory": True,
        "frozen_primary_population": "24 meaningful controllers only",
        "aggregate_primary_depended_on_nulls": False,
        "pair_classes": {},
    }
    for pair_class, group in frame.groupby("pair_class", sort=True):
        target_values = group["behavioral_D"].to_numpy(float)
        record = {
            "pairs": len(group),
            "behavioral_D_mean": float(np.mean(target_values)),
            "behavioral_D_median": float(np.median(target_values)),
            "metrics": {},
        }
        for metric in metrics:
            values = group[metric].to_numpy(float)
            coefficients = fit_linear(values, target_values) if len(group) >= 4 else None
            predictions = (
                linear_predict(values, coefficients)
                if coefficients is not None
                else np.full(len(group), np.mean(target_values))
            )
            record["metrics"][metric] = {
                "spearman": spearman(values, target_values),
                "pearson": pearson(values, target_values),
                "descriptive_in_sample_rmse": float(
                    np.sqrt(np.mean(np.square(predictions - target_values)))
                ),
            }
        result["pair_classes"][pair_class] = record
    return result, frame


def robustness(
    pair_frame: pd.DataFrame,
    target: np.ndarray,
    metrics: dict[str, np.ndarray],
    names: list[str],
    family_by_name: dict[str, str],
) -> dict[str, Any]:
    all_edges = {
        metric: {
            "spearman": spearman(pair_frame[metric], pair_frame["behavioral_D"]),
            "pearson": pearson(pair_frame[metric], pair_frame["behavioral_D"]),
        }
        for metric in metrics
    }
    leave_one_controller: list[dict[str, Any]] = []
    for removed in names:
        keep = [index for index, name in enumerate(names) if name != removed]
        reduced_names = [names[index] for index in keep]
        reduced_target = target[np.ix_(keep, keep)]
        reduced_metric = metrics["M2_FINITE_SECANT"][np.ix_(keep, keep)]
        rho, rmse, ratio, _folds = fast_fold_scores(
            reduced_target, reduced_metric, reduced_names, family_by_name
        )
        leave_one_controller.append(
            {"removed": removed, "mean_spearman": rho, "mean_rmse": rmse, "rmse_ratio": ratio}
        )
    m0_edges = pair_frame["M0_FLAT"].to_numpy(float)
    m1_edges = pair_frame["M1_WHITENED"].to_numpy(float)
    m2_edges = pair_frame["M2_FINITE_SECANT"].to_numpy(float)
    target_edges = pair_frame["behavioral_D"].to_numpy(float)
    pair_slopes = []
    for left in range(len(m2_edges)):
        denominator = m2_edges[left + 1 :] - m2_edges[left]
        numerator = target_edges[left + 1 :] - target_edges[left]
        valid = np.abs(denominator) > 1e-12
        pair_slopes.extend((numerator[valid] / denominator[valid]).tolist())
    robust_slope = float(np.median(pair_slopes))
    robust_intercept = float(np.median(target_edges - robust_slope * m2_edges))
    return {
        "post_hoc_exploratory": True,
        "all_unique_meaningful_edges": all_edges,
        "M0_M1_distance_matrix_spearman": spearman(m0_edges, m1_edges),
        "M0_M1_distance_matrix_pearson": pearson(m0_edges, m1_edges),
        "M2_Theil_Sen_all_edge_slope": robust_slope,
        "M2_Theil_Sen_all_edge_intercept": robust_intercept,
        "M2_Theil_Sen_all_edge_mae": float(
            np.mean(np.abs(target_edges - (robust_intercept + robust_slope * m2_edges)))
        ),
        "leave_one_controller": leave_one_controller,
        "leave_one_controller_ranges": {
            key: [
                float(min(row[key] for row in leave_one_controller)),
                float(max(row[key] for row in leave_one_controller)),
            ]
            for key in ("mean_spearman", "mean_rmse", "rmse_ratio")
        },
    }


def make_figures(
    predictions: pd.DataFrame,
    bootstrap_result: dict[str, Any],
    family_table: pd.DataFrame,
    pair_frame: pd.DataFrame,
    null_frame: pd.DataFrame,
) -> None:
    del bootstrap_result
    samples = np.load(OUTPUT / "ITEM_BOOTSTRAP_SAMPLES.npz", allow_pickle=False)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, color in zip(
        ("M0_FLAT", "M1_WHITENED", "M2_FINITE_SECANT"),
        ("#7a7a7a", "#4c78a8", "#e45756"),
        strict=True,
    ):
        axes[0, 0].hist(samples[f"{metric}_rho"], bins=50, alpha=0.45, label=metric, color=color)
        axes[0, 1].hist(
            samples[f"{metric}_rmse_ratio"], bins=50, alpha=0.45, label=metric, color=color
        )
    axes[0, 0].set_title("Item-bootstrap mean family-held-out rho")
    axes[0, 1].set_title("Item-bootstrap RMSE ratio")
    delta_20 = samples["M2_FINITE_SECANT_rho"] - samples["M0_FLAT_rho"]
    delta_21 = samples["M2_FINITE_SECANT_rho"] - samples["M1_WHITENED_rho"]
    axes[1, 0].hist(delta_20, bins=50, alpha=0.55, label="M2-M0")
    axes[1, 0].hist(delta_21, bins=50, alpha=0.55, label="M2-M1")
    axes[1, 0].axvline(0, color="black", linewidth=1)
    axes[1, 0].set_title("Paired rho contrasts")
    error_20 = samples["M0_FLAT_rmse"] - samples["M2_FINITE_SECANT_rmse"]
    error_21 = samples["M1_WHITENED_rmse"] - samples["M2_FINITE_SECANT_rmse"]
    axes[1, 1].hist(error_20, bins=50, alpha=0.55, label="M0-M2")
    axes[1, 1].hist(error_21, bins=50, alpha=0.55, label="M1-M2")
    axes[1, 1].axvline(0, color="black", linewidth=1)
    axes[1, 1].set_title("Paired RMSE improvement")
    for axis in axes.ravel():
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "BOOTSTRAP_DISTRIBUTIONS.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    for axis, metric in zip(axes, sorted(predictions["metric"].unique()), strict=True):
        frame = predictions[predictions["metric"] == metric]
        axis.scatter(frame["predicted"], frame["observed"], s=10, alpha=0.45)
        low = min(frame["predicted"].min(), frame["observed"].min())
        high = max(frame["predicted"].max(), frame["observed"].max())
        axis.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1)
        axis.set_title(metric)
        axis.set_xlabel("Held-out predicted D")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed D")
    figure.tight_layout()
    figure.savefig(OUTPUT / "PREDICTION_VS_OBSERVATION.png", dpi=180)
    plt.close(figure)

    pivot = family_table.pivot(index="family", columns="metric", values="spearman")
    pivot.plot(kind="bar", figsize=(12, 5), color=["#7a7a7a", "#4c78a8", "#e45756"])
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Held-out Spearman rho")
    plt.title("Exploratory family decomposition (frozen folds)")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT / "FAMILY_DECOMPOSITION.png", dpi=180)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(
        pair_frame["delta_norm_difference"], pair_frame["behavioral_D"], alpha=0.45, s=16
    )
    axes[0].set_xlabel("Absolute delta-norm difference")
    axes[0].set_ylabel("Behavioral D")
    axes[0].set_title("Scalar magnitude baseline")
    axes[1].scatter(
        pair_frame["M2_FINITE_SECANT"], pair_frame["behavioral_D"], alpha=0.45, s=16
    )
    axes[1].set_xlabel("M2 finite-secant distance")
    axes[1].set_ylabel("Behavioral D")
    axes[1].set_title("Relational finite displacement")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "DOSE_MAGNITUDE_DIAGNOSTICS.png", dpi=180)
    plt.close(figure)

    order = ["MEANINGFUL_MEANINGFUL", "MEANINGFUL_NULL", "NULL_NULL"]
    null_frame.boxplot(column="behavioral_D", by="pair_class", figsize=(10, 5))
    plt.suptitle("")
    plt.title("Behavioral distance by exploratory pair class")
    plt.ylabel("Behavioral D")
    plt.xticks(range(1, len(order) + 1), order, rotation=15)
    plt.tight_layout()
    plt.savefig(OUTPUT / "NULL_PAIR_CLASSES.png", dpi=180)
    plt.close()


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = read_json(SOURCE / "V2_FINAL_PROTOCOL_LOCK.json")
    frozen_prediction = read_json(SOURCE / "V2_PREDICTION_RESULTS.json")
    audit = read_json(SOURCE / "V2_FORENSIC_AUDIT.json")
    calibration = read_json(SOURCE / "V2_DOSE_CALIBRATION.json")
    manifest = read_json(SOURCE / "V2_COMMON_PANEL_MANIFEST.json")
    rows = read_journal(SOURCE / "V2_COMMON_PANEL_JOURNAL.jsonl")
    expected = int(lock["common_panel"]["expected_rows"])
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} immutable common-panel rows, found {len(rows)}")

    meaningful_names = list(lock["meaningful_controllers"])
    all_names = list(lock["controller_ids"])
    family_by_name = {
        name: lock["meaningful_controllers"][name]["source_axis"]
        for name in meaningful_names
    }
    all_errors = controller_errors(rows, list(manifest["item_ids"]), all_names)
    meaningful_errors = all_errors[: len(meaningful_names)]
    target = unbiased_error_distance(meaningful_errors)
    metrics = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in read_json(SOURCE / "V2_GEOMETRY_METRICS.json").items()
    }

    qap = qap_p_values(target, metrics, meaningful_names, family_by_name)
    reproduction: dict[str, Any] = {
        "frozen_classification": "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL",
        "forensic_classification": audit["classification"],
        "post_hoc_exploratory": False,
        "metrics": {},
    }
    prediction_frames = []
    family_rows = []
    for metric_name, matrix in metrics.items():
        result = family_heldout_predictions(
            target, [matrix], meaningful_names, family_by_name
        )
        frozen = frozen_prediction[metric_name]["aggregate"][metric_name]
        observed = result["aggregate"]
        values = {
            "mean_spearman": observed["mean_spearman"],
            "mean_rmse": observed["mean_rmse"],
            "mean_constant_rmse": observed["mean_constant_rmse"],
            "rmse_ratio_to_constant": observed["rmse_ratio"],
            "qap_p_one_sided": qap[metric_name]["p_one_sided"],
        }
        differences = {
            key: abs(float(values[key]) - float(frozen[key])) for key in values
        }
        reproduction["metrics"][metric_name] = {
            "recomputed": values,
            "frozen": {key: frozen[key] for key in values},
            "absolute_differences": differences,
            "agreement": max(differences.values()) <= 1e-12,
        }
        for record in result["edge_records"]:
            prediction_frames.append({"metric": metric_name, **record})
        frozen_bootstrap = read_json(SOURCE / "V2_BOOTSTRAP_INTERVALS.json")
        for family, fold in result["folds"].items():
            intervals = frozen_bootstrap["metrics"][metric_name]["fold_intervals_95"][family]
            family_rows.append(
                {
                    "family": family,
                    "metric": metric_name,
                    **{key: value for key, value in fold.items() if key != "coefficients"},
                    "spearman_ci_low": intervals["spearman"][0],
                    "spearman_ci_high": intervals["spearman"][1],
                    "rmse_ci_low": intervals["rmse"][0],
                    "rmse_ci_high": intervals["rmse"][1],
                }
            )
    reproduction["maximum_absolute_difference"] = max(
        difference
        for metric in reproduction["metrics"].values()
        for difference in metric["absolute_differences"].values()
    )
    reproduction["all_exact_within_1e_12"] = all(
        metric["agreement"] for metric in reproduction["metrics"].values()
    )
    if not reproduction["all_exact_within_1e_12"]:
        raise RuntimeError("independent Q2 V2 reconstruction did not match frozen metrics")
    write_json(OUTPUT / "PRIMARY_REPRODUCTION.json", reproduction)

    prediction_frame = pd.DataFrame(prediction_frames)
    prediction_frame.to_csv(OUTPUT / "HELDOUT_PREDICTIONS.csv", index=False)
    family_frame = pd.DataFrame(family_rows)
    family_frame.to_csv(OUTPUT / "FAMILY_DECOMPOSITION.csv", index=False)
    write_json(OUTPUT / "CALIBRATION_DIAGNOSTICS.json", calibration_diagnostics(prediction_frame))

    bootstrap_result = bootstrap(
        meaningful_errors, metrics, meaningful_names, family_by_name
    )
    write_json(OUTPUT / "BOOTSTRAP_CONTRASTS.json", bootstrap_result)

    metadata = lock["meaningful_controllers"]
    pairs = pair_table(meaningful_names, target, metrics, metadata, calibration)
    pairs.to_csv(OUTPUT / "MEANINGFUL_PAIR_DIAGNOSTICS.csv", index=False)
    nuisance = nuisance_analysis(target, metrics, meaningful_names, metadata, family_by_name)
    write_json(OUTPUT / "NUISANCE_BASELINES.json", nuisance)

    same_dose = pairs[pairs["same_dose"]]
    close_dose = pairs[pairs["dose_difference"] <= 0.25]
    local = {
        "post_hoc_exploratory": True,
        "by_same_dose": group_summary(pairs, "same_dose"),
        "by_same_family": group_summary(pairs, "same_family"),
        "by_same_direction_base": group_summary(pairs, "same_direction_base"),
        "by_dose_mean": group_summary(pairs, "dose_mean"),
        "same_dose_M2_spearman": spearman(
            same_dose["M2_FINITE_SECANT"], same_dose["behavioral_D"]
        ),
        "close_dose_pairs": len(close_dose),
        "close_dose_M2_spearman": spearman(
            close_dose["M2_FINITE_SECANT"], close_dose["behavioral_D"]
        ),
        "M2_distance_quartiles": group_summary(
            pairs.assign(
                M2_quartile=pd.qcut(
                    pairs["M2_FINITE_SECANT"], 4, labels=["Q1", "Q2", "Q3", "Q4"]
                )
            ),
            "M2_quartile",
        ),
    }
    write_json(OUTPUT / "DOSE_LOCAL_VALIDITY_ANALYSIS.json", local)

    all_target = unbiased_error_distance(all_errors)
    all_metrics = all_controller_geometries(lock, all_names, metrics)
    null_result, null_frame = null_pair_analysis(
        all_names, len(meaningful_names), all_target, all_metrics
    )
    null_frame.to_csv(OUTPUT / "NULL_PAIR_DIAGNOSTICS.csv", index=False)
    write_json(OUTPUT / "NULL_PAIR_ANALYSIS.json", null_result)

    robustness_result = robustness(
        pairs, target, metrics, meaningful_names, family_by_name
    )
    write_json(OUTPUT / "ROBUSTNESS_SENSITIVITY.json", robustness_result)

    m2_predictions = prediction_frame[prediction_frame["metric"] == "M2_FINITE_SECANT"].copy()
    m2_predictions["absolute_residual"] = m2_predictions["residual"].abs()
    influential = m2_predictions.nlargest(20, "absolute_residual")
    influential.to_csv(OUTPUT / "HIGH_LEVERAGE_PAIRS.csv", index=False)
    counts = Counter(influential["left"].tolist() + influential["right"].tolist())
    write_json(
        OUTPUT / "INFLUENCE_SUMMARY.json",
        {
            "post_hoc_exploratory": True,
            "top_20_directed_fold_edges_by_absolute_M2_residual": len(influential),
            "controller_frequency": dict(counts.most_common()),
            "note": (
                "Cross-family edges appear once for each held-out-family role by frozen design."
            ),
        },
    )

    make_figures(prediction_frame, bootstrap_result, family_frame, pairs, null_frame)

    enumerated = {
        "all_post_hoc_exploratory_unless_marked_reproduction": True,
        "analyses": [
            "A1 independent primary reconstruction and frozen QAP reproduction",
            "A2 six-family M0/M1/M2 fold decomposition with frozen item-bootstrap intervals",
            "A3 paired 10,000-item-bootstrap metric contrasts",
            "A4 held-out prediction/calibration/residual/nonlinearity diagnostics",
            "A5 four simple dose/delta-norm nuisance baselines",
            "A6 same/close-dose and nuisance-residualized M2 diagnostics",
            "A7 meaningful-meaningful, meaningful-null, and null-null pair classes",
            "A8 family/direction/dose/local-distance decomposition",
            "A9 M0/M1 matrix similarity and finite-secant interpretation",
            "robust Pearson, Theil-Sen, leave-one-controller, and leverage diagnostics",
        ],
        "unreported_metric_searches": 0,
        "new_pass_fail_rule_for_q2_v2": None,
    }
    write_json(OUTPUT / "EXPLORATORY_ANALYSIS_ENUMERATION.json", enumerated)
    provenance_inputs = [
        "REPORT.md",
        "V2_FINAL_PROTOCOL_LOCK.json",
        "V2_COMMON_PANEL_MANIFEST.json",
        "V2_COMMON_PANEL_SCHEDULE.json",
        "V2_COMMON_PANEL_JOURNAL.jsonl",
        "V2_GEOMETRY_METRICS.json",
        "V2_D_MATRIX.json",
        "V2_PREDICTION_RESULTS.json",
        "V2_BOOTSTRAP_INTERVALS.json",
        "V2_DOSE_CALIBRATION.json",
        "V2_FINITE_SECANT_ARCHIVE.json",
        "V2_COVARIANCE_ACTIVATIONS.npz",
    ]
    write_json(
        OUTPUT / "ANALYSIS_PROVENANCE.json",
        {
            "schema_version": "q2-v2-principal-review-provenance-v1",
            "frozen_q2_v2_head": "64a551ef95557d303639fb17d240b3f4f7f96a65",
            "frozen_classification": "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL",
            "forensic_classification": "Q2_V2_FORENSIC_CLEAN",
            "input_sha256": {
                name: sha256(SOURCE / name) for name in provenance_inputs
            },
            "new_inference": False,
            "scientific_trajectories_created": 0,
            "analysis_status": "POST_HOC_EXPLORATORY",
        },
    )
    print(
        json.dumps(
            {
                "phase": "q2_v2_principal_review_analysis",
                "reproduced": True,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "output": str(OUTPUT.relative_to(ROOT)),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
