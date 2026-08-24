#!/usr/bin/env python3
"""Independent low-level forensic recomputation of the Q2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_heldout_geometry"
METRICS = ("M0_FLAT", "M1_WHITENED", "M2_FINITE_SECANT")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_journal(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        wrapper = json.loads(line)
        if wrapper.get("version") != "research-os-jsonl-v1":
            raise RuntimeError("unexpected Q2 journal wrapper")
        rows.append(dict(wrapper["row"]))
    return rows


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        output[order[cursor:end]] = (cursor + end - 1) / 2
        cursor = end
    return output


def rho(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(ranks(left), ranks(right))[0, 1])


def edges(names: list[str], train: set[str]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    calibration: list[tuple[int, int]] = []
    heldout: list[tuple[int, int]] = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            (calibration if names[left] in train and names[right] in train else heldout).append(
                (left, right)
            )
    return calibration, heldout


def values(matrix: np.ndarray, selected: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([matrix[left, right] for left, right in selected], dtype=np.float64)


def score(
    geometry: np.ndarray,
    target: np.ndarray,
    train_edges: list[tuple[int, int]],
    test_edges: list[tuple[int, int]],
) -> dict[str, float | int | None]:
    train_x, train_y = values(geometry, train_edges), values(target, train_edges)
    test_x, test_y = values(geometry, test_edges), values(target, test_edges)
    design = np.column_stack((np.ones(len(train_x)), train_x))
    intercept, slope = np.linalg.lstsq(design, train_y, rcond=None)[0]
    prediction = intercept + slope * test_x
    rmse = float(np.sqrt(np.mean(np.square(prediction - test_y))))
    scale = float(np.std(train_y, ddof=1))
    constant_rmse = float(np.sqrt(np.mean(np.square(np.mean(train_y) - test_y))))
    return {
        "heldout_spearman_rho": rho(test_x, test_y),
        "heldout_edge_count": len(test_edges),
        "train_edge_count": len(train_edges),
        "affine_intercept": float(intercept),
        "affine_slope": float(slope),
        "heldout_rmse": rmse,
        "heldout_standardized_rmse": rmse / scale,
        "constant_heldout_rmse": constant_rmse,
        "rmse_ratio_to_constant": rmse / constant_rmse,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    review = args.review_dir.resolve()
    lock = read_json(review / "PROTOCOL_LOCK.json")
    rows = raw_journal(review / "journal.jsonl")
    schedule = read_json(review / "COMMON_PANEL_SCHEDULE.json")
    expected_keys = {
        (row["item_id"], row["condition"], row["rollout_index"]) for row in schedule
    }
    observed_keys = [
        (row["item_id"], row["condition"], row["rollout_index"]) for row in rows
    ]
    schedule_clean = (
        len(rows) == 4080
        and len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
    )
    seeds = [int(row["seed"]) for row in rows]
    seed_clean = len(seeds) == len(set(seeds))
    provenance_clean = all(
        row["experiment_source_commit"] == lock["experiment_source_commit"]
        and row["model"] == lock["model"]["id"]
        and row["model_revision"] == lock["model"]["revision"]
        for row in rows
    )
    names = list(lock["controller_bank"]["controller_order"])
    item_ids = [row["item_id"] for row in read_json(review / "DEVELOPMENT_PANEL_MANIFEST.json")]
    lookup = {
        (row["item_id"], row["condition"], row["rollout_index"]): int(not row["correct"])
        for row in rows
    }
    arrays = {
        name: np.asarray(
            [[lookup[(item_id, name, rollout)] for rollout in (0, 1)] for item_id in item_ids],
            dtype=np.float64,
        )
        for name in names
    }
    distance = np.zeros((len(names), len(names)), dtype=np.float64)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            estimate = float(
                np.mean(
                    (arrays[names[left]][:, 0] - arrays[names[right]][:, 0])
                    * (arrays[names[left]][:, 1] - arrays[names[right]][:, 1])
                )
            )
            distance[left, right] = estimate
            distance[right, left] = estimate
    primary_distance = np.load(review / "ERROR_DISTANCE_MATRIX.npy", allow_pickle=False)
    distance_difference = float(np.max(np.abs(distance - primary_distance)))
    split = read_json(review / "CONTROLLER_SPLIT_LOCK.json")
    train_edges, test_edges = edges(names, set(split["train_controllers"]))
    with np.load(review / "GEOMETRY_MATRICES.npz", allow_pickle=False) as archive:
        audit_scores = {
            name: score(archive[name].astype(np.float64), distance, train_edges, test_edges)
            for name in METRICS
        }
    primary = read_json(review / "PREDICTION_RESULTS.json")
    differences: dict[str, float] = {}
    for name in METRICS:
        for key, audit_value in audit_scores[name].items():
            primary_value = primary[name]["score"][key]
            if audit_value is None or primary_value is None:
                continue
            differences[f"{name}:{key}"] = abs(float(audit_value) - float(primary_value))
    maximum_metric_difference = max(differences.values(), default=0.0)
    classification = read_json(review / "CLASSIFICATION.json")["classification"]
    integrity = bool(
        schedule_clean
        and seed_clean
        and provenance_clean
        and distance_difference <= 1e-15
        and maximum_metric_difference <= 1e-12
        and lock["instrument"]["confirmatory_holdout_outcomes_read"] is False
    )
    forensic_classification = (
        "Q2_FORENSIC_CLEAN"
        if integrity
        else "Q2_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    result = {
        "classification": forensic_classification,
        "primary_classification_preserved": classification,
        "schedule_complete_unique": schedule_clean,
        "independent_seed_provenance": seed_clean,
        "model_source_provenance": provenance_clean,
        "D_max_absolute_difference": distance_difference,
        "prediction_metric_max_absolute_difference": maximum_metric_difference,
        "train_edge_count": len(train_edges),
        "heldout_edge_count": len(test_edges),
        "bootstrap_unit": "item",
        "confirmatory_outcomes_used": False,
        "Q1_changed": False,
        "Q3_run": False,
    }
    write_json(review / "FORENSIC_AUDIT.json", result)
    (review / "FORENSIC_AUDIT.md").write_text(
        "# Independent Q2 forensic audit\n\n"
        f"Classification: `{forensic_classification}`.\n\n"
        f"Maximum D difference: `{distance_difference}`. Maximum prediction-score "
        f"difference: `{maximum_metric_difference}`. The audit reconstructed the "
        "two-rollout matrix and train-only affine held-out scores without importing "
        "the primary Q2 analysis modules.\n",
        encoding="utf-8",
    )
    write_json(
        review / "METRIC_CROSSCHECK.json",
        {"audit_scores": audit_scores, "absolute_differences": differences},
    )
    write_json(
        review / "RETRY_LEDGER.json",
        {
            "scientific_rows": len(rows),
            "rows_with_retry_count_nonzero": sum(
                int(row.get("retry_count", 0) != 0) for row in rows
            ),
            "duplicate_logical_keys": len(observed_keys) - len(set(observed_keys)),
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if integrity else 1


if __name__ == "__main__":
    raise SystemExit(main())
