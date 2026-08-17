#!/usr/bin/env python3
"""Run the model-free E3-10 structural validity and shortcut gate.

The script never imports Torch/Transformers and never renders a model prompt.
All acceptance decisions use generator validity and exact procedural oracle
targets only.  Its output directory is versioned separately from the earlier
model-free design artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from epistemic_geometry.benchmarks.e3.base import GENERATOR_VERSION
from epistemic_geometry.benchmarks.e3.rendering import render_latent
from epistemic_geometry.benchmarks.e3.splits import (
    CALIBRATION_SPLIT,
    FAMILY_CELLS,
    generate_balanced_items_with_stats,
    generate_latent,
)
from epistemic_geometry.benchmarks.e3.structural import (
    fsm_sensitivity,
    shallow_heuristic_prediction,
    structural_features,
    validate_structural_item,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURE_EXCLUSIONS = {
    "reachable_subgraph_size",
    "frontier_expansions",
    "raw_satisfying_count",
    "satisfying_fraction",
    "shortest_depth_1",
    "shortest_depth_2",
    "shortest_depth_3",
}
SHORTCUT_WARNING = 0.25
SHORTCUT_FAILURE = 0.40


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _feature_matrix(items: list[Any]) -> tuple[np.ndarray, list[str]]:
    keys = sorted(
        key
        for key in {feature for item in items for feature in structural_features(item)}
        if key not in FEATURE_EXCLUSIONS
    )
    matrix = np.asarray(
        [[structural_features(item).get(key, 0.0) for key in keys] for item in items],
        dtype=np.float64,
    )
    return matrix, keys


def _accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(predictions == targets))


def _logistic_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    iterations: int = 180,
) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale == 0] = 1.0
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = np.column_stack([np.ones(len(train)), train])
    test = np.column_stack([np.ones(len(test)), test])
    weights = np.zeros((train.shape[1], 10), dtype=np.float64)
    one_hot = np.eye(10)[train_y]
    for _ in range(iterations):
        logits = train @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = (train.T @ (probabilities - one_hot)) / len(train)
        gradient[1:] += 1e-3 * weights[1:]
        weights -= 0.25 * gradient
    return np.argmax(test @ weights, axis=1)


def _gini(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=10) / len(labels)
    return float(1.0 - np.square(counts).sum())


def _tree_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    max_depth: int = 4,
) -> np.ndarray:
    def majority(labels: np.ndarray) -> int:
        counts = np.bincount(labels, minlength=10)
        return int(np.flatnonzero(counts == counts.max())[0])

    def best_split(x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        base = _gini(y)
        best: tuple[float, int, float] | None = None
        for feature in range(x.shape[1]):
            values = np.unique(x[:, feature])
            if len(values) < 2:
                continue
            thresholds = np.quantile(values, np.linspace(0.05, 0.95, min(19, len(values))))
            for threshold in np.unique(thresholds):
                left = x[:, feature] <= threshold
                if not left.any() or left.all():
                    continue
                impurity = (left.mean() * _gini(y[left])) + ((~left).mean() * _gini(y[~left]))
                gain = base - impurity
                candidate = (float(gain), feature, float(threshold))
                if best is None or candidate > best:
                    best = candidate
        if best is None or best[0] <= 1e-12:
            return None
        return best[1], best[2]

    class Node:
        def __init__(self, x: np.ndarray, labels: np.ndarray, depth: int) -> None:
            self.prediction = majority(labels)
            self.feature: int | None = None
            self.threshold: float | None = None
            self.left: Node | None = None
            self.right: Node | None = None
            if depth >= max_depth or len(np.unique(labels)) == 1:
                return
            split = best_split(x, labels)
            if split is None:
                return
            feature, threshold = split
            mask = x[:, feature] <= threshold
            if not mask.any() or mask.all():
                return
            self.feature, self.threshold = feature, threshold
            self.left = Node(x[mask], labels[mask], depth + 1)
            self.right = Node(x[~mask], labels[~mask], depth + 1)

    root = Node(train_x, train_y, 0)

    def predict_row(row: np.ndarray, node: Node) -> int:
        if node.feature is None or node.left is None or node.right is None:
            return node.prediction
        branch = node.left if row[node.feature] <= node.threshold else node.right
        return predict_row(row, branch)

    return np.asarray([predict_row(row, root) for row in test_x], dtype=int)


def _shortcut_results(train: list[Any], test: list[Any]) -> dict[str, Any]:
    train_x, feature_names = _feature_matrix(train)
    test_x, _ = _feature_matrix(test)
    train_y = np.asarray([item.target for item in train], dtype=int)
    test_y = np.asarray([item.target for item in test], dtype=int)
    frequency = int(np.bincount(train_y, minlength=10).argmax())
    predictions = {
        "target_frequency": np.full(len(test_y), frequency, dtype=int),
        "multinomial_logistic": _logistic_predict(train_x, train_y, test_x),
        "decision_tree_depth_4": _tree_fit_predict(train_x, train_y, test_x, max_depth=4),
        "semantic_heuristic": np.asarray(
            [shallow_heuristic_prediction(item) for item in test], dtype=int
        ),
    }
    accuracies = {name: _accuracy(prediction, test_y) for name, prediction in predictions.items()}
    classifier_values = {
        name: value for name, value in accuracies.items() if name != "target_frequency"
    }
    max_classifier = max(classifier_values.values())
    if max_classifier >= SHORTCUT_FAILURE:
        status = "STRUCTURAL_SHORTCUT_FAILURE"
    elif max_classifier >= SHORTCUT_WARNING:
        status = "STRUCTURAL_SHORTCUT_WARNING"
    else:
        status = "PASS"
    return {
        "feature_names": feature_names,
        "train_items": len(train),
        "test_items": len(test),
        "accuracies": accuracies,
        "status": status,
        "warning_threshold": SHORTCUT_WARNING,
        "failure_threshold": SHORTCUT_FAILURE,
    }


def _target_feature_audit(items: list[Any]) -> dict[str, Any]:
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in items:
        grouped[item.target].append(item)
    feature_names = sorted({name for item in items for name in structural_features(item)})
    distributions: dict[str, Any] = {}
    largest_ranges: list[tuple[float, str]] = []
    for name in feature_names:
        means = {
            str(target): float(
                np.mean([structural_features(item).get(name, 0.0) for item in grouped[target]])
            )
            for target in range(10)
        }
        values = list(means.values())
        range_value = max(values) - min(values)
        largest_ranges.append((range_value, name))
        distributions[name] = {"target_means": means, "range": range_value}
    largest_ranges.sort(reverse=True)
    return {
        "features": distributions,
        "largest_target_mean_ranges": [
            {"feature": name, "range": value} for value, name in largest_ranges[:10]
        ],
    }


def _surface_audit(items: list[Any]) -> dict[str, Any]:
    token_pattern = re.compile(r"\w+|[^\w\s]")
    stats: list[dict[str, Any]] = []
    semantic_equal = True
    nontrivial = True
    for item in items:
        canonical = render_latent(item)
        twin = render_latent(item, surface="surface_twin")
        semantic_equal &= canonical.target == twin.target
        canonical_tokens = token_pattern.findall(canonical.prompt)
        twin_tokens = token_pattern.findall(twin.prompt)
        overlap = len(set(canonical_tokens) & set(twin_tokens)) / max(
            1, len(set(canonical_tokens) | set(twin_tokens))
        )
        char_similarity = SequenceMatcher(None, canonical.prompt, twin.prompt).ratio()
        nontrivial &= canonical.prompt_hash != twin.prompt_hash
        stats.append(
            {
                "latent_id": item.latent_id,
                "character_similarity": char_similarity,
                "token_jaccard": overlap,
                "changed_token_fraction": 1.0 - overlap,
                "prompt_hash_distinct": canonical.prompt_hash != twin.prompt_hash,
            }
        )
    return {
        "n_items": len(stats),
        "semantic_equality": semantic_equal,
        "nontrivial_surface_change": nontrivial,
        "mean_character_similarity": float(np.mean([row["character_similarity"] for row in stats])),
        "mean_token_jaccard": float(np.mean([row["token_jaccard"] for row in stats])),
        "mean_changed_token_fraction": float(
            np.mean([row["changed_token_fraction"] for row in stats])
        ),
        "sample_statistics": stats[:20],
    }


def _effective_depth_audit(items_by_cell: dict[str, list[Any]]) -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {}
    medians: list[float] = []
    nominal_depths: list[int] = []
    for cell, items in items_by_cell.items():
        values = np.asarray(
            [int(validate_structural_item(item)["effective_operation_count"]) for item in items]
        )
        nominal = int(items[0].difficulty["nominal_depth"])
        nominal_depths.append(nominal)
        medians.append(float(np.median(values)))
        report[cell] = {
            "nominal_depth": nominal,
            "n_items": len(items),
            "mean_effective_operations": float(np.mean(values)),
            "median_effective_operations": float(np.median(values)),
            "min": int(values.min()),
            "q10": float(np.quantile(values, 0.10)),
            "q90": float(np.quantile(values, 0.90)),
            "max": int(values.max()),
        }
    order = np.argsort(nominal_depths)
    monotonic = all(
        medians[order[index]] < medians[order[index + 1]] for index in range(len(order) - 1)
    )
    return report, monotonic


def _make_effective_depth_figure(report: dict[str, Any], path: Path) -> None:
    ordered = sorted(report.values(), key=lambda row: row["nominal_depth"])
    x = [row["nominal_depth"] for row in ordered]
    median = [row["median_effective_operations"] for row in ordered]
    q10 = [row["q10"] for row in ordered]
    q90 = [row["q90"] for row in ordered]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(x, median, marker="o", label="median effective operations")
    axis.fill_between(x, q10, q90, alpha=0.2, label="q10–q90")
    axis.set_xlabel("nominal MODREG10 depth")
    axis.set_ylabel("effective operations in query dependency cone")
    axis.set_title("E3-10 MODREG10 effective-depth audit")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_gate(
    *, n_items: int, shortcut_train_items: int, shortcut_test_items: int, output: Path
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    structural_audit: dict[str, Any] = {}
    support_audit: dict[str, Any] = {}
    rejection_audit: dict[str, Any] = {}
    shortcut_audit: dict[str, Any] = {}
    target_feature_audit: dict[str, Any] = {}
    structural_validity: dict[str, Any] = {}
    balance_audit: dict[str, Any] = {}
    leakage_audit: dict[str, Any] = {}
    examples: list[dict[str, Any]] = []
    pool_records: list[dict[str, Any]] = []
    depth_items: dict[str, list[Any]] = {}
    all_namespace_ids: dict[str, set[str]] = {}
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            key = f"{family}/{cell}"
            items_tuple, generation_stats = generate_balanced_items_with_stats(
                family,
                cell,
                n_items,
                20260817,
                split_name="STRUCTURAL_AUDIT",
            )
            items = list(items_tuple)
            pool_records.extend({"family_cell": key, "latent": item.to_record()} for item in items)
            balance_audit[key] = {
                "n_items": len(items),
                "target_counts": {
                    str(digit): sum(item.target == digit for item in items) for digit in range(10)
                },
                "model_outcomes_used": False,
            }
            example = items[0]
            examples.append(
                {
                    "latent": example.to_record(),
                    "canonical_decimal": render_latent(example).to_record(),
                    "surface_twin_decimal": render_latent(
                        example, surface="surface_twin"
                    ).to_record(),
                    "canonical_number_word": render_latent(
                        example, response_channel="number_word"
                    ).to_record(),
                }
            )
            all_namespace_ids[key] = {item.latent_id for item in items}
            validity_rows = [validate_structural_item(item) for item in items]
            structural_validity[key] = {
                "valid": all(row["valid"] for row in validity_rows),
                "invalid_count": sum(not row["valid"] for row in validity_rows),
            }
            if family == "MODREG10":
                depth_items[cell] = items
            raw_targets = [generate_latent(family, cell, seed).target for seed in range(5_000)]
            support_counts = Counter(raw_targets)
            support_audit[key] = {
                "raw_sample_size": len(raw_targets),
                "raw_target_counts": {str(digit): support_counts[digit] for digit in range(10)},
                "all_ten_targets_observed": set(support_counts) == set(range(10)),
                "balanced_target_counts": {
                    str(digit): sum(item.target == digit for item in items) for digit in range(10)
                },
            }
            rejection_audit[key] = generation_stats.to_record()
            target_feature_audit[key] = _target_feature_audit(items)
            surface_sample = items[: min(500, len(items))]
            structural_audit[key] = {
                "family": family,
                "cell": cell,
                "validity": structural_validity[key],
                "surface": _surface_audit(surface_sample),
                "feature_audit": target_feature_audit[key],
            }
            if family == "FSM10":
                sensitivities = [fsm_sensitivity(item) for item in surface_sample]
                structural_audit[key]["fsm_sensitivity"] = {
                    "replacement_sensitivity_fraction": float(
                        np.mean([row["replacement_sensitivity_fraction"] for row in sensitivities])
                    ),
                    "removal_sensitivity_fraction": float(
                        np.mean([row["removal_sensitivity_fraction"] for row in sensitivities])
                    ),
                }
            train_tuple, _ = generate_balanced_items_with_stats(
                family,
                cell,
                shortcut_train_items,
                20260817,
                split_name="SHORTCUT_TRAIN",
            )
            test_tuple, _ = generate_balanced_items_with_stats(
                family,
                cell,
                shortcut_test_items,
                20260817,
                split_name="SHORTCUT_TEST",
            )
            train, test = list(train_tuple), list(test_tuple)
            train_ids = {item.latent_id for item in train}
            test_ids = {item.latent_id for item in test}
            calibration_tuple, _ = generate_balanced_items_with_stats(
                family,
                cell,
                100,
                20260817,
                split_name=CALIBRATION_SPLIT,
            )
            calibration_ids = {item.latent_id for item in calibration_tuple}
            namespace_items = {
                "train": train,
                "test": test,
                "calibration": list(calibration_tuple),
            }
            all_namespace_ids[key] |= train_ids | test_ids | calibration_ids
            leakage_audit[key] = {
                "train_test_overlap": sorted(train_ids & test_ids),
                "train_calibration_overlap": sorted(train_ids & calibration_ids),
                "test_calibration_overlap": sorted(test_ids & calibration_ids),
                "seed_overlap": {
                    "train_test": sorted(
                        {item.latent_seed for item in train} & {item.latent_seed for item in test}
                    ),
                    "train_calibration": sorted(
                        {item.latent_seed for item in train}
                        & {item.latent_seed for item in calibration_tuple}
                    ),
                    "test_calibration": sorted(
                        {item.latent_seed for item in test}
                        & {item.latent_seed for item in calibration_tuple}
                    ),
                },
                "prompt_hash_overlap": {
                    "train_test": sorted(
                        {render_latent(item).prompt_hash for item in namespace_items["train"]}
                        & {render_latent(item).prompt_hash for item in namespace_items["test"]}
                    ),
                    "train_calibration": sorted(
                        {render_latent(item).prompt_hash for item in namespace_items["train"]}
                        & {
                            render_latent(item).prompt_hash
                            for item in namespace_items["calibration"]
                        }
                    ),
                    "test_calibration": sorted(
                        {render_latent(item).prompt_hash for item in namespace_items["test"]}
                        & {
                            render_latent(item).prompt_hash
                            for item in namespace_items["calibration"]
                        }
                    ),
                },
            }
            if (
                leakage_audit[key]["train_test_overlap"]
                or leakage_audit[key]["train_calibration_overlap"]
                or leakage_audit[key]["test_calibration_overlap"]
                or any(leakage_audit[key]["seed_overlap"].values())
                or any(leakage_audit[key]["prompt_hash_overlap"].values())
            ):
                raise RuntimeError(f"latent namespace collision detected for {key}")
            shortcut_audit[key] = _shortcut_results(train, test)
    depth_report, monotonic = _effective_depth_audit(depth_items)
    _make_effective_depth_figure(depth_report, output / "modreg_effective_depth_audit.png")
    eligible: list[str] = []
    warnings: list[str] = []
    failures: list[str] = []
    for key in structural_audit:
        structural_ok = (
            structural_validity[key]["valid"] and support_audit[key]["all_ten_targets_observed"]
        )
        status = shortcut_audit[key]["status"]
        if status == "STRUCTURAL_SHORTCUT_WARNING":
            warnings.append(key)
        if status == "STRUCTURAL_SHORTCUT_FAILURE":
            failures.append(key)
        if structural_ok and status != "STRUCTURAL_SHORTCUT_FAILURE":
            eligible.append(key)
    eligible_families = sorted({key.split("/", 1)[0] for key in eligible})
    worst_efficiency_key, worst_efficiency = max(
        rejection_audit.items(), key=lambda pair: pair[1]["attempts_per_accepted"]
    )
    worst_target_key, worst_target = min(
        (
            (f"{key}/target_{target}", rate)
            for key, value in rejection_audit.items()
            for target, rate in value["acceptance_rate_by_target"].items()
        ),
        key=lambda pair: pair[1],
    )
    result = {
        "gate": "E3-10 PRE-MODEL STRUCTURAL VALIDITY GATE",
        "model_inference": {"qwen_used": False, "runpod_used": False, "steering_used": False},
        "generator_version": GENERATOR_VERSION,
        "n_items_per_family_cell": n_items,
        "structural_pool_records": len(pool_records),
        "modreg_effective_depth": depth_report,
        "modreg_monotonic_difficulty_realization": monotonic,
        "target_support": support_audit,
        "structural_validity": structural_validity,
        "shortcut_audit": shortcut_audit,
        "target_conditional_features": target_feature_audit,
        "major_generator_artifacts_found": {
            key: value["largest_target_mean_ranges"][:3]
            for key, value in target_feature_audit.items()
        },
        "rejection_efficiency": rejection_audit,
        "target_balance_audit": balance_audit,
        "latent_leakage_audit": leakage_audit,
        "surface_twin_audit": {key: value["surface"] for key, value in structural_audit.items()},
        "latent_namespace_audit": {key: len(ids) for key, ids in all_namespace_ids.items()},
        "structurally_eligible_cells": sorted(eligible),
        "shortcut_warnings": sorted(warnings),
        "shortcut_failures": sorted(failures),
        "structurally_eligible_families": eligible_families,
        "excluded_cells": sorted(failures),
        "rejection_operational_assessment": {
            "worst_cell": worst_efficiency_key,
            "worst_attempts_per_accepted": worst_efficiency["attempts_per_accepted"],
            "worst_target": worst_target_key,
            "worst_target_acceptance_rate": worst_target,
            "projected_attempts_for_1000_items_at_worst_cell": (
                worst_efficiency["attempts_per_accepted"] * 1000
            ),
            "operationally_viable": worst_efficiency["attempts_per_accepted"] < 1000,
        },
        "structural_approval": bool(monotonic and len(eligible_families) >= 2),
        "qualification_thresholds_unchanged": {
            "accuracy": [0.30, 0.75],
            "decimal_word_agreement": 0.85,
            "surface_twin_agreement": 0.80,
            "normalized_entropy": 0.80,
        },
    }
    _json_write(output / "generator_structural_audit.json", structural_audit)
    _json_write(output / "target_support_audit.json", support_audit)
    _json_write(output / "rejection_efficiency_audit.json", rejection_audit)
    _json_write(output / "shortcut_baseline_audit.json", shortcut_audit)
    _json_write(output / "target_conditional_feature_audit.json", target_feature_audit)
    _json_write(output / "target_balance_audit.json", balance_audit)
    _json_write(output / "split_leakage_audit.json", leakage_audit)
    pool_manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in pool_records)
    (output / "structural_pool_manifest.jsonl").write_text(pool_manifest, encoding="utf-8")
    result["structural_pool_manifest_sha256"] = hashlib.sha256(pool_manifest.encode()).hexdigest()
    _json_write(output / "structural_gate_summary.json", result)
    (output / "example_items.jsonl").write_text(
        "".join(json.dumps(example, sort_keys=True) + "\n" for example in examples),
        encoding="utf-8",
    )
    _json_write(
        output / "frozen_qualification_rules.json",
        {
            "baseline_only": True,
            "generator_version": GENERATOR_VERSION,
            "structural_prerequisites": {
                "target_support": "all ten digits observed",
                "generator_structural_validity": "all family-specific checks pass",
                "no_structural_shortcut_failure": True,
            },
            "accuracy_range": [0.30, 0.75],
            "decimal_word_agreement": 0.85,
            "surface_twin_agreement": 0.80,
            "normalized_entropy": 0.80,
            "structural_shortcut_warning": SHORTCUT_WARNING,
            "structural_shortcut_failure": SHORTCUT_FAILURE,
            "selection": "qualifying cell closest to 0.50, lower nominal difficulty tie-break",
        },
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-items", type=int, default=5_000)
    parser.add_argument("--shortcut-train-items", type=int, default=2_000)
    parser.add_argument("--shortcut-test-items", type=int, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "review" / "q1_v2_instrument_design_v2",
    )
    args = parser.parse_args()
    result = run_gate(
        n_items=args.n_items,
        shortcut_train_items=args.shortcut_train_items,
        shortcut_test_items=args.shortcut_test_items,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["structural_approval"]:
        print("E3_10_PRE_MODEL_STRUCTURAL_GATE_FAILED")
        raise SystemExit(2)
    print("E3_10_PRE_MODEL_STRUCTURAL_GATE_PASS")
    print("RUNPOD_REQUIRED_FOR_Q1_V2_INSTRUMENT_CALIBRATION")


if __name__ == "__main__":
    main()
