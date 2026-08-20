#!/usr/bin/env python3
"""Offline Gate-6 atlas, manipulation, and evaluation analysis.

This script never loads Qwen and never changes a frozen phase decision.  It
reconstructs every reported quantity from the source artifacts and append-only
journal, with CRUXEval item IDs as the resampling unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    LAYERS,
    SOURCE_LOCATIONS,
    two_rollout_estimands,
)

MANIPULATION_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "TEXTUAL_DIRECT_REFERENCE",
    "BEST_SINGLE_RFM_PLUS",
    "MULTILAYER_MEAN_PLUS",
    "MULTILAYER_RFM_PLUS",
    "MULTILAYER_RFM_MINUS",
    "MULTILAYER_RANDOM_R0",
    "MULTILAYER_RANDOM_R1",
    "MULTILAYER_RANDOM_R2",
    "MULTILAYER_RANDOM_R3",
)
EVALUATION_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "BEST_SINGLE_RFM_PLUS",
    "MULTILAYER_MEAN_PLUS",
    "MULTILAYER_RFM_PLUS",
    "MULTILAYER_RFM_MINUS",
    "MULTILAYER_RANDOM_R0",
    "MULTILAYER_RANDOM_R1",
    "MULTILAYER_RANDOM_R2",
    "MULTILAYER_RANDOM_R3",
)
VALID_STATUSES = {"VALID_CORRECT", "VALID_WRONG"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid journal JSON at line {line_number}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def status(row: dict[str, Any]) -> str:
    return str(row.get("status", "RUNTIME_ERROR"))


def valid(row: dict[str, Any]) -> bool:
    return status(row) in VALID_STATUSES


def error(row: dict[str, Any]) -> bool:
    return status(row) != "VALID_CORRECT"


def tokens(row: dict[str, Any]) -> int:
    return int(row.get("generated_token_count", 0) or 0)


def semantic_outcome(row: dict[str, Any]) -> str:
    if valid(row) and row.get("parsed_answer") is not None:
        return f"ANSWER:{row['parsed_answer']}"
    return f"STATUS:{status(row)}"


def raw_tokens(row: dict[str, Any]) -> tuple[int, ...]:
    values = row.get("generated_token_ids", ())
    return tuple(int(value) for value in values) if isinstance(values, list) else tuple()


def key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))


def require_phase(
    rows: list[dict[str, Any]],
    conditions: tuple[str, ...],
    n_items: int,
    rollouts: tuple[int, ...],
) -> tuple[list[str], dict[tuple[str, str, int], dict[str, Any]]]:
    keyed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in keyed:
            raise ValueError(f"duplicate Gate-6 logical row: {row_key}")
        keyed[row_key] = row
    item_ids = sorted({row_key[0] for row_key in keyed})
    if len(item_ids) != n_items:
        raise ValueError(f"expected {n_items} items, found {len(item_ids)}")
    expected = {
        (item_id, condition, rollout)
        for item_id in item_ids
        for condition in conditions
        for rollout in rollouts
    }
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"phase schedule mismatch: missing={missing[:3]} extra={extra[:3]}")
    return item_ids, keyed


def summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    counts = Counter(status(row) for row in selected)
    token_values = [tokens(row) for row in selected]
    valid_count = sum(counts[name] for name in VALID_STATUSES)
    return {
        "condition": condition,
        "n": len(selected),
        "valid": valid_count,
        "correct": counts["VALID_CORRECT"],
        "wrong": counts["VALID_WRONG"],
        "invalid": counts["INVALID_FORMAT"],
        "truncated": counts["TRUNCATED"] + counts["TRUNCATED_THINKING"],
        "runtime_error": counts["RUNTIME_ERROR"],
        "validity": valid_count / len(selected),
        "accuracy": counts["VALID_CORRECT"] / len(selected),
        "mean_tokens": float(np.mean(token_values)) if token_values else math.nan,
        "median_tokens": float(np.median(token_values)) if token_values else math.nan,
        "max_tokens": max(token_values, default=0),
        "status_counts": dict(counts),
    }


def divergence(left: tuple[int, ...], right: tuple[int, ...]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def manipulation_analysis(
    rows: list[dict[str, Any]], selection: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    item_ids, keyed = require_phase(rows, MANIPULATION_CONDITIONS, 20, (0,))
    results = {condition: summary(rows, condition) for condition in MANIPULATION_CONDITIONS}
    for condition in MANIPULATION_CONDITIONS[1:]:
        semantic_changes: list[bool] = []
        raw_changes: list[bool] = []
        first: list[int] = []
        for item_id in item_ids:
            baseline = keyed[(item_id, "BASELINE", 0)]
            current = keyed[(item_id, condition, 0)]
            semantic_changes.append(semantic_outcome(baseline) != semantic_outcome(current))
            left, right = raw_tokens(baseline), raw_tokens(current)
            raw_changes.append(left != right)
            index = divergence(left, right)
            if index is not None:
                first.append(index)
        results[condition].update(
            {
                "semantic_change_rate": float(np.mean(semantic_changes)),
                "raw_sequence_change_rate": float(np.mean(raw_changes)),
                "first_divergence_mean": float(np.mean(first)) if first else None,
                "first_divergence_count": len(first),
            }
        )

    random_names = [f"MULTILAYER_RANDOM_R{i}" for i in range(4)]
    random_semantic = [results[name]["semantic_change_rate"] for name in random_names]
    random_raw = [results[name]["raw_sequence_change_rate"] for name in random_names]
    results["RANDOM_SUMMARY"] = {
        "semantic_change_rate_mean": float(np.mean(random_semantic)),
        "semantic_change_rate_max": float(np.max(random_semantic)),
        "raw_sequence_change_rate_mean": float(np.mean(random_raw)),
        "raw_sequence_change_rate_max": float(np.max(random_raw)),
    }
    # The frozen manipulation gate requires the sustained controller to move
    # toward the corresponding textual policy in generated-token count.
    careful_mean = results["TEXTUAL_CAREFUL_REFERENCE"]["mean_tokens"]
    direct_mean = results["TEXTUAL_DIRECT_REFERENCE"]["mean_tokens"]
    baseline_mean = results["BASELINE"]["mean_tokens"]
    if careful_mean >= direct_mean:
        plus_coherent = abs(results["MULTILAYER_RFM_PLUS"]["mean_tokens"] - careful_mean) < abs(
            baseline_mean - careful_mean
        )
        minus_coherent = abs(results["MULTILAYER_RFM_MINUS"]["mean_tokens"] - direct_mean) < abs(
            baseline_mean - direct_mean
        )
    else:
        plus_coherent = False
        minus_coherent = False
    rfm_first_stage_pass = bool(selection.get("source_only_passes"))
    pass_by_sign = {}
    random_mean = results["RANDOM_SUMMARY"]["semantic_change_rate_mean"]
    for sign, coherent in (("PLUS", plus_coherent), ("MINUS", minus_coherent)):
        sustained = results[f"MULTILAYER_RFM_{sign}"]
        pass_by_sign[sign] = bool(
            rfm_first_stage_pass
            and sustained["validity"] >= 0.90
            and sustained["semantic_change_rate"] >= 0.15
            and sustained["semantic_change_rate"] - random_mean >= 0.05
            and sustained["semantic_change_rate"]
            > results["MULTILAYER_MEAN_PLUS"]["semantic_change_rate"]
            and coherent
        )
    gate = {
        "random_summary": results["RANDOM_SUMMARY"],
        "token_direction_coherent": {"PLUS": plus_coherent, "MINUS": minus_coherent},
        "source_first_stage_pass": rfm_first_stage_pass,
        "pass_by_sign": pass_by_sign,
    }
    return [
        {"condition": condition, **values}
        for condition, values in results.items()
        if condition != "RANDOM_SUMMARY"
    ], gate, any(pass_by_sign.values())


def error_matrix(
    item_ids: list[str],
    keyed: dict[tuple[str, str, int], dict[str, Any]],
    condition: str,
) -> np.ndarray:
    return np.asarray(
        [
            [error(keyed[(item_id, condition, rollout)]) for rollout in (0, 1)]
            for item_id in item_ids
        ],
        dtype=np.int8,
    )


def evaluation_analysis(
    rows: list[dict[str, Any]], review: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    item_ids, keyed = require_phase(rows, EVALUATION_CONDITIONS, 60, (0, 1))
    matrices = {
        condition: error_matrix(item_ids, keyed, condition) for condition in EVALUATION_CONDITIONS
    }
    summaries = {condition: summary(rows, condition) for condition in EVALUATION_CONDITIONS}
    estimates: dict[str, dict[str, Any]] = {
        "BASELINE": {
            "validity": summaries["BASELINE"]["validity"],
            "accuracy": summaries["BASELINE"]["accuracy"],
        }
    }
    for condition in EVALUATION_CONDITIONS[1:]:
        estimates[condition] = {
            **two_rollout_estimands(matrices["BASELINE"], matrices[condition]),
            "validity": summaries[condition]["validity"],
            "accuracy": summaries[condition]["accuracy"],
        }
    random_names = [f"MULTILAYER_RANDOM_R{i}" for i in range(4)]
    random_summary = {
        metric: {
            "mean": float(np.mean([estimates[name][metric] for name in random_names])),
            "median": float(np.median([estimates[name][metric] for name in random_names])),
            "min": float(np.min([estimates[name][metric] for name in random_names])),
            "max": float(np.max([estimates[name][metric] for name in random_names])),
        }
        for metric in ("G", "C", "D", "accuracy", "validity")
    }
    bootstrap = evaluation_bootstrap(matrices, summaries)
    return [
        {"condition": condition, **values, **summaries[condition]}
        for condition, values in estimates.items()
    ], {"random": random_summary, "matrices": matrices}, bootstrap


def evaluation_bootstrap(
    matrices: dict[str, np.ndarray], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    n_items = matrices["BASELINE"].shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(EVALUATION_CONDITIONS[1:])
    metric_names = ("accuracy_change", "validity_change", "G", "C", "D", "rescue", "damage")
    values: dict[str, list[float]] = {
        f"{name}:{metric}": [] for name in names for metric in metric_names
    }
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, n_items, size=n_items)
        baseline = matrices["BASELINE"][indices]
        baseline_validity = summaries["BASELINE"]["validity"]
        for name in names:
            result = two_rollout_estimands(baseline, matrices[name][indices])
            values[f"{name}:accuracy_change"].append(
                result["accuracy_condition"] - result["accuracy_baseline"]
            )
            values[f"{name}:validity_change"].append(
                summaries[name]["validity"] - baseline_validity
            )
            for metric in ("G", "C", "D", "rescue", "damage"):
                values[f"{name}:{metric}"].append(result[metric])
    return {
        "method": "item_cluster_percentile_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "n_resamples": BOOTSTRAP_RESAMPLES,
        "confidence": 0.95,
        "intervals": {
            name: {
                "estimate": float(np.mean(sample)),
                "lower": float(np.quantile(sample, 0.025)),
                "upper": float(np.quantile(sample, 0.975)),
            }
            for name, sample in values.items()
        },
    }


def classify_gate6(
    source_readout_pass: bool,
    source_first_stage_pass: bool,
    manipulation_pass: bool,
    estimates: dict[str, dict[str, Any]] | None,
) -> str:
    if not source_readout_pass:
        return "GATE6_RFM_SOURCE_NOT_VALIDATED"
    if not source_first_stage_pass:
        return "GATE6_READOUT_WITHOUT_CAUSAL_FIRST_STAGE"
    if not manipulation_pass or estimates is None:
        return "GATE6_CAUSAL_FIRST_STAGE_NOT_EXPRESSED_IN_GENERATION"
    baseline = estimates["BASELINE"]
    random_names = [f"MULTILAYER_RANDOM_R{i}" for i in range(4)]
    movements: dict[str, bool] = {}
    useful: dict[str, bool] = {}
    for sign in ("PLUS", "MINUS"):
        name = f"MULTILAYER_RFM_{sign}"
        current = estimates[name]
        random_d = [estimates[random]["D"] for random in random_names]
        random_c = [estimates[random]["C"] for random in random_names]
        valid_guard = (
            current["validity"] >= 0.90
            and current["validity"] >= baseline["validity"] - 0.05
        )
        competence_guard = current["accuracy"] >= baseline["accuracy"] - 0.10
        movements[sign] = bool(
            valid_guard
            and competence_guard
            and current["D"] >= 0.05
            and current["D"] - float(np.mean(random_d)) >= 0.05
            and current["D"] > max(random_d)
            and current["D"] - estimates["BEST_SINGLE_RFM_PLUS"]["D"] >= 0.02
            and current["D"] - estimates["MULTILAYER_MEAN_PLUS"]["D"] >= 0.02
        )
        useful[sign] = bool(
            movements[sign]
            and current["G"] >= 0.03
            and current["C"] >= 0.03
            and current["C"] - float(np.mean(random_c)) >= 0.05
            and current["C"] > max(random_c)
        )
    if any(useful.values()):
        return "GATE6_DISTRIBUTED_USEFUL_COMPLEMENTARITY_SIGNAL"
    if any(movements.values()):
        distributed = any(
            estimates[f"MULTILAYER_RFM_{sign}"]["D"]
            - estimates["BEST_SINGLE_RFM_PLUS"]["D"] >= 0.02
            and estimates[f"MULTILAYER_RFM_{sign}"]["D"]
            - estimates["MULTILAYER_MEAN_PLUS"]["D"] >= 0.02
            for sign in ("PLUS", "MINUS")
        )
        return (
            "GATE6_DISTRIBUTED_ERROR_PROFILE_MOVEMENT_ONLY"
            if distributed
            else "GATE6_DISTRIBUTED_CONTROL_NOT_BETTER_THAN_SINGLE_OR_MEAN"
        )
    if all(
        estimates[f"MULTILAYER_RFM_{sign}"]["validity"] < 0.90
        or estimates[f"MULTILAYER_RFM_{sign}"]["accuracy"] < baseline["accuracy"] - 0.10
        for sign in ("PLUS", "MINUS")
    ):
        return "GATE6_DISTRIBUTED_DESTRUCTIVE"
    return "GATE6_CAUSAL_STYLE_CONTROL_WITHOUT_ERROR_CONTROL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--pod-runtime-seconds", type=float, default=None)
    parser.add_argument("--pod-id", default=None)
    args = parser.parse_args()
    review = args.review_dir
    rows = read_jsonl(args.journal or review / "journal.jsonl")
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_phase.setdefault(str(row.get("phase", "UNKNOWN")), []).append(row)
    source_metrics = (
        read_json(review / "SOURCE_METRICS.json")
        if (review / "SOURCE_METRICS.json").exists()
        else {}
    )
    first_stage = (
        read_json(review / "FIRST_STAGE_RESULTS.json")
        if (review / "FIRST_STAGE_RESULTS.json").exists()
        else {}
    )
    selection = (
        read_json(review / "CONTROLLER_SELECTION.json")
        if (review / "CONTROLLER_SELECTION.json").exists()
        else {}
    )
    source_readout_pass = any(
        values.get("readout", {}).get("auroc", 0.0) >= 0.80
        and values.get("readout", {}).get("positive_gap_fraction", 0.0) * 32 >= 24
        for values in source_metrics.values()
    )
    individual_first_stage_pass = bool(selection.get("source_only_passes"))
    source_first_stage_pass = bool(
        individual_first_stage_pass and len(selection.get("multilayer_keys", [])) >= 2
    )
    manipulation_rows = by_phase.get("CONTROLLER_MANIPULATION", [])
    manipulation_table: list[dict[str, Any]] = []
    manipulation_gate: dict[str, Any] = {"pass": False}
    manipulation_pass = False
    if manipulation_rows:
        manipulation_table, manipulation_gate, manipulation_pass = manipulation_analysis(
            manipulation_rows, selection
        )
    evaluation_table: list[dict[str, Any]] = []
    evaluation_details: dict[str, Any] = {}
    bootstrap: dict[str, Any] | None = None
    estimates: dict[str, dict[str, Any]] | None = None
    evaluation_rows = by_phase.get("CONTROLLER_EVALUATION", [])
    if evaluation_rows:
        evaluation_table, evaluation_details, bootstrap = evaluation_analysis(
            evaluation_rows, review
        )
        estimates = {row["condition"]: row for row in evaluation_table}
    classification = classify_gate6(
        source_readout_pass, source_first_stage_pass, manipulation_pass, estimates
    )
    if (
        manipulation_rows
        and not manipulation_pass
        and not evaluation_rows
        and source_readout_pass
        and source_first_stage_pass
    ):
        classification = "GATE6_CAUSAL_FIRST_STAGE_NOT_EXPRESSED_IN_GENERATION"
    write_csv(review / "MANIPULATION_RESULTS.csv", manipulation_table)
    write_csv(review / "CONDITION_SUMMARY.csv", evaluation_table)
    write_json(
        review / "ESTIMANDS.json",
        {
            "source_readout_pass": source_readout_pass,
            "individual_first_stage_pass": individual_first_stage_pass,
            "source_first_stage_pass": source_first_stage_pass,
            "manipulation": manipulation_gate,
            "evaluation": estimates,
            "random_summary": evaluation_details.get("random"),
            "classification": classification,
        },
    )
    write_json(review / "BOOTSTRAP_INTERVALS.json", bootstrap or {"status": "not_executed"})
    source_atlas = {
        "layers": list(LAYERS),
        "source_locations": list(SOURCE_LOCATIONS),
        "source_metrics": source_metrics,
        "first_stage": first_stage,
        "individual_first_stage_pass": individual_first_stage_pass,
        "selection": selection,
    }
    write_json(review / "SOURCE_ATLAS_SUMMARY.json", source_atlas)
    trajectory_seconds = float(sum(float(row.get("elapsed_seconds", 0.0)) for row in rows))
    generated_tokens = int(sum(tokens(row) for row in rows))
    cost = {
        "rate_usd_per_a40_hour": 0.44,
        "trajectories": len(rows),
        "generated_tokens": generated_tokens,
        "trajectory_seconds": trajectory_seconds,
        "trajectory_cost_usd": trajectory_seconds / 3600.0 * 0.44,
        "pod_id": args.pod_id,
        "pod_runtime_seconds": args.pod_runtime_seconds,
        "pod_wallclock_cost_usd": None
        if args.pod_runtime_seconds is None
        else args.pod_runtime_seconds / 3600.0 * 0.44,
        "analysis_gpu_cost_usd": 0.0,
    }
    write_json(review / "COST.json", cost)
    hash_names = [
        "PROTOCOL_LOCK.json",
        "SOURCE_TRAIN_MANIFEST.json",
        "SOURCE_VALIDATION.json",
        "CONTROLLER_MANIPULATION.json",
        "CONTROLLER_EVALUATION.json",
        "HISTORICAL_EXCLUSION_DIGEST.json",
        "SOURCE_ACTIVATIONS.npz",
        "SOURCE_GENERATIONS.jsonl",
        "journal.jsonl",
    ]
    hashes = {
        name: hashlib.sha256((review / name).read_bytes()).hexdigest()
        for name in hash_names
        if (review / name).exists()
    }
    write_json(review / "manifest_hashes.json", hashes)
    report_lines = [
        "GATE 6 — LAYER–SOURCE–AGOP CONTROL ATLAS",
        "======================================================================",
        "",
        "This is development evidence only; Gate 5 artifacts remain immutable.",
        "",
        "## Source/layer atlas",
        "",
        f"source readout pass: {source_readout_pass}",
        f"RFM first-stage pass: {source_first_stage_pass}",
        f"selected source: {selection.get('selected_source')}",
        "passing layers: "
        + str([int(read_key.split(":L")[-1]) for read_key in selection.get("multilayer_keys", [])]),
        "",
        "## Manipulation gate",
        "",
        f"executed: {bool(manipulation_rows)}",
        f"pass: {manipulation_pass}",
        "random summary: "
        + json.dumps(manipulation_gate.get("random_summary", {}), sort_keys=True),
        "",
        "## Primary evaluation",
        "",
        f"executed: {bool(evaluation_rows)}",
        f"classification: {classification}",
        "",
        "| condition | validity | accuracy | G | C | D | rescue | damage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation_table:
        report_lines.append(
            f"| {row['condition']} | {row['validity']:.3f} | {row['accuracy']:.3f} | "
            f"{row.get('G', float('nan')):.4f} | {row.get('C', float('nan')):.4f} | "
            f"{row.get('D', float('nan')):.4f} | {row.get('rescue', float('nan')):.4f} | "
            f"{row.get('damage', float('nan')):.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Firewall",
            "",
            "Q2: NOT RUN",
            "character count: NOT RUN",
            "confirmatory holdout: UNTOUCHED",
            "",
            f"trajectories analyzed: {len(rows)}",
            f"estimated trajectory cost USD: {cost['trajectory_cost_usd']:.6f}",
        ]
    )
    (review / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    write_json(
        review / "FINAL_CLASSIFICATION.json",
        {
            "classification": classification,
            "source_readout_pass": source_readout_pass,
            "source_first_stage_pass": source_first_stage_pass,
            "manipulation_pass": manipulation_pass,
            "primary_evaluation_executed": bool(evaluation_rows),
            "trajectories": len(rows),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
