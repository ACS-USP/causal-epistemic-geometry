#!/usr/bin/env python3
"""Frozen item-cluster uncertainty for the completed Q2-V2 common panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_controller_bank_v2"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 2026082401


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def average_ranks(values: np.ndarray) -> np.ndarray:
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


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = average_ranks(np.asarray(left, dtype=np.float64))
    right_ranks = average_ranks(np.asarray(right, dtype=np.float64))
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


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


def fold_scores(
    target: np.ndarray,
    metric: np.ndarray,
    names: list[str],
    family_by_name: dict[str, str],
) -> tuple[float, float, float, dict[str, dict[str, float]]]:
    folds: dict[str, dict[str, float]] = {}
    for heldout in sorted(set(family_by_name.values())):
        train_indices = [
            index for index, name in enumerate(names) if family_by_name[name] != heldout
        ]
        test_indices = [
            index for index, name in enumerate(names) if family_by_name[name] == heldout
        ]
        train_target = upper_edges(target, train_indices)
        train_metric = upper_edges(metric, train_indices)
        test_edges = [
            (left, right)
            for left in test_indices
            for right in range(len(names))
            if right != left and right not in test_indices
        ]
        test_target = np.asarray([target[left, right] for left, right in test_edges])
        test_metric = np.asarray([metric[left, right] for left, right in test_edges])
        prediction = fit_predict(train_metric, train_target, test_metric)
        constant = np.full(len(test_target), np.mean(train_target))
        folds[heldout] = {
            "spearman": spearman(test_metric, test_target),
            "rmse": float(np.sqrt(np.mean((prediction - test_target) ** 2))),
            "constant_rmse": float(np.sqrt(np.mean((constant - test_target) ** 2))),
        }
    mean_spearman = float(np.nanmean([value["spearman"] for value in folds.values()]))
    mean_rmse = float(np.mean([value["rmse"] for value in folds.values()]))
    mean_constant = float(np.mean([value["constant_rmse"] for value in folds.values()]))
    return mean_spearman, mean_rmse, mean_rmse / max(mean_constant, 1e-12), folds


def percentile(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    return [float(np.nanquantile(array, 0.025)), float(np.nanquantile(array, 0.975))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.resamples != DEFAULT_RESAMPLES or args.seed != DEFAULT_SEED:
        raise RuntimeError("Q2 V2 bootstrap count and seed are frozen")

    lock = read_json(REVIEW / "V2_FINAL_PROTOCOL_LOCK.json")
    protocol = read_json(REVIEW / "PROTOCOL_LOCK.json")
    uncertainty = protocol["uncertainty"]
    if uncertainty != {
        "bootstrap_resamples": DEFAULT_RESAMPLES,
        "bootstrap_seed": DEFAULT_SEED,
        "bootstrap_unit": "item",
    }:
        raise RuntimeError("Q2 V2 frozen uncertainty lock mismatch")

    journal_path = REVIEW / "V2_COMMON_PANEL_JOURNAL.jsonl"
    rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    expected = int(lock["common_panel"]["expected_rows"])
    if len(rows) != expected:
        raise RuntimeError("Q2 V2 bootstrap requires the complete common panel")

    manifest = read_json(REVIEW / "V2_COMMON_PANEL_MANIFEST.json")
    item_ids = list(manifest["item_ids"])
    names = list(lock["meaningful_controllers"])
    lookup = {(row["item_id"], row["condition"], int(row["rollout_index"])): row for row in rows}
    errors = np.asarray(
        [
            [
                [int(not lookup[(item_id, name, rollout)]["correct"]) for rollout in (0, 1)]
                for item_id in item_ids
            ]
            for name in names
        ],
        dtype=np.float64,
    )
    geometries = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in read_json(REVIEW / "V2_GEOMETRY_METRICS.json").items()
    }
    family_by_name = {name: lock["meaningful_controllers"][name]["source_axis"] for name in names}
    samples: dict[str, dict[str, list[float]]] = {
        name: {"mean_spearman": [], "mean_rmse": [], "standardized_rmse": []} for name in geometries
    }
    fold_samples: dict[str, dict[str, dict[str, list[float]]]] = {
        metric: {
            family: {"spearman": [], "rmse": []} for family in sorted(set(family_by_name.values()))
        }
        for metric in geometries
    }
    rng = np.random.default_rng(args.seed)
    for _ in range(args.resamples):
        indices = rng.integers(0, len(item_ids), size=len(item_ids))
        selected = errors[:, indices, :]
        difference_0 = selected[:, None, :, 0] - selected[None, :, :, 0]
        difference_1 = selected[:, None, :, 1] - selected[None, :, :, 1]
        target = np.mean(difference_0 * difference_1, axis=2)
        for metric_name, metric in geometries.items():
            rho, rmse, standardized, folds = fold_scores(target, metric, names, family_by_name)
            samples[metric_name]["mean_spearman"].append(rho)
            samples[metric_name]["mean_rmse"].append(rmse)
            samples[metric_name]["standardized_rmse"].append(standardized)
            for family, values in folds.items():
                fold_samples[metric_name][family]["spearman"].append(values["spearman"])
                fold_samples[metric_name][family]["rmse"].append(values["rmse"])

    output = {
        "schema_version": "q2-v2-item-cluster-bootstrap-v1",
        "resamples": args.resamples,
        "seed": args.seed,
        "unit": "item",
        "classification_uses_bootstrap": False,
        "metrics": {
            metric: {
                name: {"interval_95": percentile(values)} for name, values in metric_samples.items()
            }
            | {
                "fold_intervals_95": {
                    family: {name: percentile(values) for name, values in family_samples.items()}
                    for family, family_samples in fold_samples[metric].items()
                }
            }
            for metric, metric_samples in samples.items()
        },
    }
    write_json(REVIEW / "V2_BOOTSTRAP_INTERVALS.json", output)
    print(json.dumps({"phase": "bootstrap", "resamples": args.resamples}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
