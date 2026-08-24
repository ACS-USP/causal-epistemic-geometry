#!/usr/bin/env python3
"""Primary, outcome-after-collection analysis for Q2-V2."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6 import two_rollout_estimands  # noqa: E402
from epistemic_geometry.experiments.q2_controller_heldout_v2 import (  # noqa: E402
    BASELINE,
    pairwise_unbiased_distance_matrix,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = 0.5 * (index + end - 1) + 1.0
        index = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = rank_average(x)
    ry = rank_average(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pair_edges(matrix: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    values: list[float] = []
    edges: list[tuple[str, str]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            values.append(float(matrix[left, right]))
            edges.append((names[left], names[right]))
    return np.asarray(values, dtype=np.float64), edges


def linear_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train)), x_train])
    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return np.column_stack([np.ones(len(x_test)), x_test]) @ coefficients


def load_journal() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lock = read_json(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json")
    if lock["status"] != "FROZEN_PRE_COMMON_PANEL":
        raise RuntimeError("Q2 V2 final lock is not frozen")
    rows = [
        json.loads(line)
        for line in (REVIEW / "V2_COMMON_PANEL_JOURNAL.jsonl").read_text().splitlines()
        if line.strip()
    ]
    expected = int(lock["common_panel"]["expected_rows"])
    keys = [(row["item_id"], row["condition"], int(row["rollout_index"])) for row in rows]
    if len(rows) != expected or len(keys) != len(set(keys)):
        raise RuntimeError("Q2 V2 common journal is incomplete or duplicated")
    return lock, rows


def condition_arrays(
    lock: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    item_ids = list(lock["common_panel"].get("item_ids", []))
    if not item_ids:
        item_ids = list(read_json(REVIEW / "V2_COMMON_PANEL_MANIFEST.json")["item_ids"])
    conditions = [BASELINE, *lock["controller_ids"]]
    lookup = {(row["item_id"], row["condition"], int(row["rollout_index"])): row for row in rows}
    errors: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    for condition in conditions:
        values = []
        valid = []
        evaluable = []
        tokens = []
        for item_id in item_ids:
            item_rows = [lookup[(item_id, condition, rollout)] for rollout in (0, 1)]
            values.extend(int(not bool(row["correct"])) for row in item_rows)
            valid.extend(bool(row["commitment_valid"]) for row in item_rows)
            evaluable.extend(bool(row["semantic_evaluable"]) for row in item_rows)
            tokens.extend(int(row["generated_token_count"]) for row in item_rows)
        errors[condition] = np.asarray(values, dtype=np.int8).reshape(len(item_ids), 2)
        summaries[condition] = {
            "n_items": len(item_ids),
            "rollouts": 2,
            "commitment_validity": float(np.mean(valid)),
            "semantic_evaluability": float(np.mean(evaluable)),
            "accuracy": float(1.0 - np.mean(values)),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "max_tokens": int(max(tokens)),
        }
    return item_ids, conditions, errors, summaries


def load_controller_vectors(lock: dict[str, Any]) -> dict[str, np.ndarray]:
    metadata = {**lock["meaningful_controllers"], **lock["random_controllers"]}
    vectors: dict[str, np.ndarray] = {}
    for name in lock["controller_ids"]:
        vector = np.load(ROOT / metadata[name]["path"], allow_pickle=False).astype(np.float64)
        vectors[name] = vector.reshape(-1)
    return vectors


def m0_geometry(lock: dict[str, Any], vectors: dict[str, np.ndarray]) -> np.ndarray:
    names = lock["meaningful_controllers"]
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1, len(names)):
            right_name = list(names)[right]
            cosine = float(np.dot(vectors[left_name], vectors[right_name]))
            matrix[left, right] = matrix[right, left] = float(np.sqrt(max(0.0, 2.0 - 2.0 * cosine)))
    return matrix


def m1_geometry(lock: dict[str, Any], vectors: dict[str, np.ndarray]) -> np.ndarray:
    names = list(lock["meaningful_controllers"])
    activations = np.load(REVIEW / "V2_COVARIANCE_ACTIVATIONS.npz", allow_pickle=False)[
        "activations"
    ]
    centered = activations.astype(np.float64) - np.mean(activations, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    regularization = float(lock["geometry"]["M1"]["lambda"] * np.mean(eigenvalues))
    inverse = (
        eigenvectors
        @ np.diag(1.0 / np.maximum(eigenvalues + regularization, 1e-10))
        @ eigenvectors.T
    )
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1, len(names)):
            right_name = names[right]
            difference = vectors[left_name] - vectors[right_name]
            matrix[left, right] = matrix[right, left] = float(
                np.sqrt(max(0.0, difference @ inverse @ difference))
            )
    return matrix


def _softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _js(left: np.ndarray, right: np.ndarray) -> float:
    p = _softmax(left)
    q = _softmax(right)
    midpoint = 0.5 * (p + q)
    per_checkpoint = 0.5 * np.sum(
        p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(midpoint, 1e-300))),
        axis=-1,
    ) + 0.5 * np.sum(
        q * (np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(midpoint, 1e-300))),
        axis=-1,
    )
    return float(np.mean(per_checkpoint))


def m2_geometry(lock: dict[str, Any]) -> np.ndarray:
    names = list(lock["meaningful_controllers"])
    archive = read_json(REVIEW / "V2_FINITE_SECANT_ARCHIVE.json")
    records = {record["item_id"]: record for record in archive["records"]}
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1, len(names)):
            right_name = names[right]
            values: list[float] = []
            for _item_id, record in records.items():
                arrays = np.load(REVIEW / record["path"], allow_pickle=False)
                values.extend(_js(arrays[left_name], arrays[right_name]) for _ in range(1))
            matrix[left, right] = matrix[right, left] = float(np.sqrt(np.mean(values)))
    return matrix


def family_prediction(
    lock: dict[str, Any], target: np.ndarray, metrics: dict[str, np.ndarray]
) -> dict[str, Any]:
    names = list(lock["meaningful_controllers"])
    families = sorted({lock["meaningful_controllers"][name]["source_axis"] for name in names})
    family_by_name = {name: lock["meaningful_controllers"][name]["source_axis"] for name in names}
    folds: dict[str, Any] = {}
    for heldout in families:
        train = [name for name in names if family_by_name[name] != heldout]
        test = [name for name in names if family_by_name[name] == heldout]
        train_indices = [names.index(name) for name in train]
        test_indices = [names.index(name) for name in test]
        train_values, _ = pair_edges(target[np.ix_(train_indices, train_indices)], train)
        fold: dict[str, Any] = {"train_count": len(train), "test_count": len(test), "metrics": {}}
        test_edges: list[tuple[int, int]] = []
        test_values: list[float] = []
        for left in test_indices:
            for right in range(len(names)):
                if right == left or right in test_indices:
                    continue
                test_edges.append((left, right))
                test_values.append(float(target[left, right]))
        for metric_name, metric in metrics.items():
            train_metric, _ = pair_edges(metric[np.ix_(train_indices, train_indices)], train)
            test_metric = np.asarray([metric[left, right] for left, right in test_edges])
            prediction = linear_fit_predict(train_metric, train_values, test_metric)
            baseline_prediction = np.full(len(test_values), np.mean(train_values))
            fold["metrics"][metric_name] = {
                "spearman": spearman(test_metric, np.asarray(test_values)),
                "rmse": float(np.sqrt(np.mean((prediction - test_values) ** 2))),
                "constant_rmse": float(np.sqrt(np.mean((baseline_prediction - test_values) ** 2))),
                "test_edges": len(test_values),
            }
        folds[heldout] = fold

    observed: dict[str, dict[str, float]] = {}
    for metric_name in metrics:
        spearmans = [fold["metrics"][metric_name]["spearman"] for fold in folds.values()]
        rmses = [fold["metrics"][metric_name]["rmse"] for fold in folds.values()]
        constants = [fold["metrics"][metric_name]["constant_rmse"] for fold in folds.values()]
        observed[metric_name] = {
            "mean_spearman": float(np.nanmean(spearmans)),
            "median_spearman": float(np.nanmedian(spearmans)),
            "mean_rmse": float(np.mean(rmses)),
            "mean_constant_rmse": float(np.mean(constants)),
            "rmse_ratio_to_constant": float(np.mean(rmses) / max(np.mean(constants), 1e-12)),
        }

    qap_seed = int(lock["prediction"].get("qap_seed", 2026082402))
    rng = np.random.default_rng(qap_seed)
    permutations = int(lock["prediction"].get("qap_permutations", 10000))
    null_scores = {name: [] for name in metrics}
    for _ in range(permutations):
        permutation = np.arange(len(names))
        for family in families:
            indices = [index for index, name in enumerate(names) if family_by_name[name] == family]
            permutation[indices] = rng.permutation(indices)
        permuted_target = target[np.ix_(permutation, permutation)]
        for metric_name, metric in metrics.items():
            fold_scores: list[float] = []
            for heldout in families:
                train = [name for name in names if family_by_name[name] != heldout]
                test = [name for name in names if family_by_name[name] == heldout]
                train_indices = [names.index(name) for name in train]
                test_indices = [names.index(name) for name in test]
                train_values, _ = pair_edges(
                    permuted_target[np.ix_(train_indices, train_indices)], train
                )
                test_edges = [
                    (left, right)
                    for left in test_indices
                    for right in range(len(names))
                    if right != left and right not in test_indices
                ]
                test_values = np.asarray(
                    [permuted_target[left, right] for left, right in test_edges]
                )
                test_metric = np.asarray([metric[left, right] for left, right in test_edges])
                fold_scores.append(spearman(test_metric, test_values))
            null_scores[metric_name].append(float(np.nanmean(fold_scores)))
    for metric_name in metrics:
        observed[metric_name]["qap_p_one_sided"] = float(
            (
                1
                + np.sum(
                    np.asarray(null_scores[metric_name]) >= observed[metric_name]["mean_spearman"]
                )
            )
            / (1 + len(null_scores[metric_name]))
        )
        observed[metric_name]["qap_null_mean"] = float(np.nanmean(null_scores[metric_name]))
        observed[metric_name]["qap_null_p95"] = float(
            np.nanquantile(null_scores[metric_name], 0.95)
        )
    return {"families": families, "folds": folds, "aggregate": observed}


def main() -> int:
    lock, rows = load_journal()
    item_ids, conditions, errors, summaries = condition_arrays(lock, rows)
    meaningful = list(lock["meaningful_controllers"])
    meaningful_arrays = {name: errors[name] for name in meaningful}
    baseline_estimands = {
        name: two_rollout_estimands(errors[BASELINE], errors[name])
        for name in conditions
        if name != BASELINE
    }
    d_matrix = pairwise_unbiased_distance_matrix(meaningful_arrays, meaningful)
    vectors = load_controller_vectors(lock)
    geometry = {
        "M0_FLAT": m0_geometry(lock, vectors),
        "M1_WHITENED": m1_geometry(lock, vectors),
        "M2_FINITE_SECANT": m2_geometry(lock),
    }
    prediction = {
        name: family_prediction(lock, d_matrix, {name: value}) for name, value in geometry.items()
    }
    for name in prediction:
        prediction[name]["geometry_name"] = name
    thresholds = lock["prediction"]["classification_thresholds"]
    qualifying_metrics = [
        name
        for name, result in prediction.items()
        if (
            result["aggregate"][name]["mean_spearman"]
            >= thresholds["spearman_min"]
            and result["aggregate"][name]["qap_p_one_sided"]
            <= thresholds["qap_one_sided_p_max"]
            and result["aggregate"][name]["rmse_ratio_to_constant"]
            <= thresholds["rmse_ratio_to_constant_max"]
        )
    ]
    classification = (
        "Q2_V2_FAMILY_HELDOUT_GEOMETRY_SIGNAL"
        if qualifying_metrics
        else "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL"
    )

    write_json(REVIEW / "V2_CONDITION_SUMMARY.json", summaries)
    write_json(REVIEW / "V2_ESTIMANDS.json", baseline_estimands)
    write_json(
        REVIEW / "V2_D_MATRIX.json", {"controllers": meaningful, "values": d_matrix.tolist()}
    )
    write_json(
        REVIEW / "V2_GEOMETRY_METRICS.json",
        {name: value.tolist() for name, value in geometry.items()},
    )
    write_json(REVIEW / "V2_PREDICTION_RESULTS.json", prediction)
    write_json(
        REVIEW / "V2_CLASSIFICATION.json",
        {
            "classification": classification,
            "qualifying_metrics": qualifying_metrics,
            "thresholds": thresholds,
            "all_metrics_reported": True,
            "best_metric_cherry_picking": False,
        },
    )
    evidence = {
        "controller_count": len(meaningful),
        "family_count": len(
            {lock["meaningful_controllers"][name]["source_axis"] for name in meaningful}
        ),
        "common_panel_items": len(item_ids),
        "random_controls": len(lock["random_controllers"]),
        "all_metrics_reported": True,
        "accuracy_used_for_controller_selection": False,
        "best_metric_cherry_picking": False,
        "classification": classification,
    }
    write_json(REVIEW / "V2_EVIDENCE_VECTOR.json", evidence)
    report_lines = [
        "# Q2 V2 — calibrated controller-family-held-out geometry",
        "",
        f"Common panel: {len(item_ids)} items; conditions: {len(conditions)}; rows: {len(rows)}.",
        "The primary population is meaningful controllers only. Nulls are secondary controls.",
        "All M0/M1/M2 results are reported; no geometry was selected after outcomes.",
        "",
    ]
    for name in prediction:
        report_lines.append(f"## {name}")
        report_lines.append("")
        report_lines.append("```json")
        report_lines.append(json.dumps(prediction[name]["aggregate"], indent=2, sort_keys=True))
        report_lines.append("```")
        report_lines.append("")
    (REVIEW / "V2_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"phase": "analysis", "rows": len(rows), "controllers": len(meaningful)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
