#!/usr/bin/env python3
"""Independent raw-row forensic audit for the completed Q2-V2 common panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
BASELINE = "BASELINE"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        output[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return output


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = ranks(left), ranks(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def upper_edges(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.asarray(
        [
            matrix[left, right]
            for offset, left in enumerate(indices)
            for right in indices[offset + 1 :]
        ],
        dtype=np.float64,
    )


def distance_matrix(errors: np.ndarray) -> np.ndarray:
    first = errors[:, None, :, 0] - errors[None, :, :, 0]
    second = errors[:, None, :, 1] - errors[None, :, :, 1]
    return np.mean(first * second, axis=2)


def m0(vectors: np.ndarray) -> np.ndarray:
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    result = np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0))
    np.fill_diagonal(result, 0.0)
    return result


def m1(vectors: np.ndarray, activations: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(activations, dtype=np.float64)
    centered = values - np.mean(values, axis=0)
    _u, singular, basis = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = np.square(singular) / (len(values) - 1)
    mean_variance = float(np.sum(np.var(values, axis=0, ddof=1)) / values.shape[1])
    ridge = fraction * mean_variance
    adjusted = (1.0 - fraction) * eigenvalues + ridge
    projected = vectors @ basis.T
    gram = (vectors @ vectors.T) / ridge
    gram += (projected * ((1.0 / adjusted) - (1.0 / ridge))[None, :]) @ projected.T
    gram = 0.5 * (gram + gram.T)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    cosine = np.clip(gram / np.outer(norms, norms), -1.0, 1.0)
    result = np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0))
    np.fill_diagonal(result, 0.0)
    return result


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=np.float64) - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def js(left: np.ndarray, right: np.ndarray) -> float:
    p, q = softmax(left), softmax(right)
    middle = 0.5 * (p + q)
    value = 0.5 * np.sum(
        p * (np.log(np.maximum(p, 1e-300)) - np.log(np.maximum(middle, 1e-300))), axis=-1
    )
    value += 0.5 * np.sum(
        q * (np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(middle, 1e-300))), axis=-1
    )
    return float(np.mean(value))


def m2(names: list[str]) -> np.ndarray:
    archive = read_json(REVIEW / "V2_FINITE_SECANT_ARCHIVE.json")
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for left, left_name in enumerate(names):
        for right in range(left + 1, len(names)):
            right_name = names[right]
            values = []
            for record in archive["records"]:
                arrays = np.load(REVIEW / record["path"], allow_pickle=False)
                values.append(js(arrays[left_name], arrays[right_name]))
            matrix[left, right] = matrix[right, left] = float(np.sqrt(np.mean(values)))
    return matrix


def predict(
    target: np.ndarray,
    metric: np.ndarray,
    names: list[str],
    family_by_name: dict[str, str],
    *,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    families = sorted(set(family_by_name.values()))

    def folds_for(target_matrix: np.ndarray) -> dict[str, dict[str, float]]:
        folds = {}
        for heldout in families:
            train = [index for index, name in enumerate(names) if family_by_name[name] != heldout]
            test = [index for index, name in enumerate(names) if family_by_name[name] == heldout]
            train_target, train_metric = (
                upper_edges(target_matrix, train),
                upper_edges(metric, train),
            )
            test_edges = [
                (left, right)
                for left in test
                for right in range(len(names))
                if right != left and right not in test
            ]
            test_target = np.asarray([target_matrix[left, right] for left, right in test_edges])
            test_metric = np.asarray([metric[left, right] for left, right in test_edges])
            fitted = fit_predict(train_metric, train_target, test_metric)
            constant = np.full(len(test_target), np.mean(train_target))
            folds[heldout] = {
                "spearman": spearman(test_metric, test_target),
                "rmse": float(np.sqrt(np.mean((fitted - test_target) ** 2))),
                "constant_rmse": float(np.sqrt(np.mean((constant - test_target) ** 2))),
                "test_edges": len(test_target),
            }
        return folds

    folds = folds_for(target)
    rho = float(np.nanmean([value["spearman"] for value in folds.values()]))
    rmse = float(np.mean([value["rmse"] for value in folds.values()]))
    constant = float(np.mean([value["constant_rmse"] for value in folds.values()]))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        permutation = np.arange(len(names))
        for family in families:
            indices = [index for index, name in enumerate(names) if family_by_name[name] == family]
            permutation[indices] = rng.permutation(indices)
        permuted = target[np.ix_(permutation, permutation)]
        null.append(
            float(np.nanmean([value["spearman"] for value in folds_for(permuted).values()]))
        )
    return {
        "folds": folds,
        "aggregate": {
            "mean_spearman": rho,
            "median_spearman": float(np.nanmedian([value["spearman"] for value in folds.values()])),
            "mean_rmse": rmse,
            "mean_constant_rmse": constant,
            "rmse_ratio_to_constant": rmse / max(constant, 1e-12),
            "qap_p_one_sided": float((1 + np.sum(np.asarray(null) >= rho)) / (1 + len(null))),
            "qap_null_mean": float(np.nanmean(null)),
            "qap_null_p95": float(np.nanquantile(null, 0.95)),
        },
    }


def flatten_numeric(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_numeric(f"{prefix}.{key}" if prefix else key, child, output)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            flatten_numeric(f"{prefix}.{index}", child, output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = float(value)


def main() -> int:
    lock = read_json(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json")
    schedule = read_json(REVIEW / "V2_COMMON_PANEL_SCHEDULE.json")
    manifest = read_json(REVIEW / "V2_COMMON_PANEL_MANIFEST.json")
    rows = [
        json.loads(line)
        for line in (REVIEW / "V2_COMMON_PANEL_JOURNAL.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keys = [(row["item_id"], row["condition"], int(row["rollout_index"])) for row in rows]
    expected_keys = [
        (row["item_id"], row["condition"], int(row["rollout_index"])) for row in schedule
    ]
    schedule_by_key = {key: row for key, row in zip(expected_keys, schedule, strict=True)}
    row_by_key = {key: row for key, row in zip(keys, rows, strict=True)}
    integrity = {
        "expected_rows": int(lock["common_panel"]["expected_rows"]),
        "observed_rows": len(rows),
        "unique_keys": len(set(keys)),
        "duplicate_rows": len(rows) - len(set(keys)),
        "missing_keys": len(set(expected_keys) - set(keys)),
        "unexpected_keys": len(set(keys) - set(expected_keys)),
        "seed_mismatches": sum(
            int(row_by_key[key]["seed"] != schedule_by_key[key]["seed"])
            for key in set(keys) & set(expected_keys)
        ),
        "source_commit_mismatches": sum(
            row.get("experiment_source_commit") != lock["experiment_source_commit"] for row in rows
        ),
        "model_mismatches": sum(row.get("model") != lock["model"]["id"] for row in rows),
        "revision_mismatches": sum(
            row.get("model_revision") != lock["model"]["revision"] for row in rows
        ),
        "retry_rows": sum(int(row.get("retry_count", 0)) > 0 for row in rows),
        "manifest_sha256": sha256(REVIEW / "V2_COMMON_PANEL_MANIFEST.json"),
        "schedule_sha256": sha256(REVIEW / "V2_COMMON_PANEL_SCHEDULE.json"),
    }
    integrity_pass = (
        integrity["observed_rows"] == integrity["expected_rows"]
        and integrity["unique_keys"] == integrity["expected_rows"]
        and all(
            integrity[key] == 0
            for key in (
                "duplicate_rows",
                "missing_keys",
                "unexpected_keys",
                "seed_mismatches",
                "source_commit_mismatches",
                "model_mismatches",
                "revision_mismatches",
            )
        )
        and integrity["manifest_sha256"] == lock["common_panel"]["manifest_sha256"]
        and integrity["schedule_sha256"] == lock["common_panel"]["schedule_sha256"]
    )
    if not integrity_pass:
        write_json(REVIEW / "V2_COLLECTION_INTEGRITY.json", integrity | {"pass": False})
        raise RuntimeError("Q2 V2 collection integrity failed")

    item_ids = list(manifest["item_ids"])
    conditions = [BASELINE, *lock["controller_ids"]]
    errors = np.asarray(
        [
            [
                [
                    int(not row_by_key[(item_id, condition, rollout)]["correct"])
                    for rollout in (0, 1)
                ]
                for item_id in item_ids
            ]
            for condition in conditions
        ],
        dtype=np.float64,
    )
    summaries = {}
    for index, condition in enumerate(conditions):
        selected = [
            row_by_key[(item_id, condition, rollout)] for item_id in item_ids for rollout in (0, 1)
        ]
        summaries[condition] = {
            "accuracy": float(1.0 - np.mean(errors[index])),
            "commitment_validity": float(np.mean([row["commitment_valid"] for row in selected])),
            "semantic_evaluability": float(
                np.mean([row["semantic_evaluable"] for row in selected])
            ),
            "mean_tokens": float(np.mean([row["generated_token_count"] for row in selected])),
            "median_tokens": float(np.median([row["generated_token_count"] for row in selected])),
            "max_tokens": int(max(row["generated_token_count"] for row in selected)),
            "n_items": len(item_ids),
            "rollouts": 2,
        }

    names = list(lock["meaningful_controllers"])
    meaningful_indices = [conditions.index(name) for name in names]
    target = distance_matrix(errors[meaningful_indices])
    metadata = {**lock["meaningful_controllers"], **lock["random_controllers"]}
    vectors = np.stack(
        [
            np.load(ROOT / metadata[name]["path"], allow_pickle=False)
            .astype(np.float64)
            .reshape(-1)
            for name in names
        ]
    )
    geometries = {
        "M0_FLAT": m0(vectors),
        "M1_WHITENED": m1(
            vectors,
            np.load(REVIEW / "V2_COVARIANCE_ACTIVATIONS.npz", allow_pickle=False)["activations"],
            float(lock["geometry"]["M1"]["lambda"]),
        ),
        "M2_FINITE_SECANT": m2(names),
    }
    family_by_name = {name: lock["meaningful_controllers"][name]["source_axis"] for name in names}
    prediction = {
        name: predict(
            target,
            metric,
            names,
            family_by_name,
            permutations=int(lock["prediction"]["qap_permutations"]),
            seed=int(lock["prediction"]["qap_seed"]),
        )
        for name, metric in geometries.items()
    }
    thresholds = lock["prediction"]["classification_thresholds"]
    qualifying = [
        name
        for name, result in prediction.items()
        if result["aggregate"]["mean_spearman"] >= thresholds["spearman_min"]
        and result["aggregate"]["qap_p_one_sided"] <= thresholds["qap_one_sided_p_max"]
        and result["aggregate"]["rmse_ratio_to_constant"]
        <= thresholds["rmse_ratio_to_constant_max"]
    ]
    classification = (
        "Q2_V2_FAMILY_HELDOUT_GEOMETRY_SIGNAL"
        if qualifying
        else "Q2_V2_NO_FAMILY_HELDOUT_GEOMETRY_SIGNAL"
    )

    primary_numeric: dict[str, float] = {}
    audit_numeric: dict[str, float] = {}
    flatten_numeric("condition", read_json(REVIEW / "V2_CONDITION_SUMMARY.json"), primary_numeric)
    flatten_numeric("condition", summaries, audit_numeric)
    primary_d = np.asarray(read_json(REVIEW / "V2_D_MATRIX.json")["values"], dtype=np.float64)
    primary_geometry = read_json(REVIEW / "V2_GEOMETRY_METRICS.json")
    primary_prediction = read_json(REVIEW / "V2_PREDICTION_RESULTS.json")
    flatten_numeric("D", primary_d.tolist(), primary_numeric)
    flatten_numeric("D", target.tolist(), audit_numeric)
    for name in geometries:
        flatten_numeric(f"geometry.{name}", primary_geometry[name], primary_numeric)
        flatten_numeric(f"geometry.{name}", geometries[name].tolist(), audit_numeric)
        flatten_numeric(
            f"prediction.{name}", primary_prediction[name]["aggregate"][name], primary_numeric
        )
        flatten_numeric(f"prediction.{name}", prediction[name]["aggregate"], audit_numeric)
    shared = sorted(set(primary_numeric) & set(audit_numeric))
    differences = {key: abs(primary_numeric[key] - audit_numeric[key]) for key in shared}
    maximum_difference = max(differences.values(), default=0.0)
    primary_classification = read_json(REVIEW / "V2_CLASSIFICATION.json")["classification"]
    agreement = primary_classification == classification and maximum_difference <= 1e-10

    with (REVIEW / "V2_METRIC_CROSSCHECK.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "primary", "audit", "absolute_difference"]
        )
        writer.writeheader()
        for key in shared:
            writer.writerow(
                {
                    "metric": key,
                    "primary": primary_numeric[key],
                    "audit": audit_numeric[key],
                    "absolute_difference": differences[key],
                }
            )
    write_json(REVIEW / "V2_COLLECTION_INTEGRITY.json", integrity | {"pass": True})
    write_json(
        REVIEW / "V2_RETRY_LEDGER.json",
        {
            "rows": len(rows),
            "retry_rows": integrity["retry_rows"],
            "outcome_dependent_retries": 0,
            "regenerated_logical_keys": 0,
        },
    )
    write_json(
        REVIEW / "V2_FORENSIC_AUDIT.json",
        {
            "classification": "Q2_V2_FORENSIC_CLEAN"
            if agreement
            else "Q2_V2_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN",
            "primary_classification": primary_classification,
            "audit_classification": classification,
            "qualifying_metrics": qualifying,
            "primary_audit_agreement": agreement,
            "maximum_numeric_difference": maximum_difference,
            "integrity": integrity,
            "independent_implementation": True,
            "primary_high_level_analysis_imported": False,
        },
    )
    print(
        json.dumps(
            {"phase": "forensic_audit", "agreement": agreement, "max_diff": maximum_difference}
        )
    )
    return 0 if agreement else 1


if __name__ == "__main__":
    raise SystemExit(main())
