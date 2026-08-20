#!/usr/bin/env python3
"""Offline Gate-5 source, duration, and primary-evaluation analysis.

This script is deliberately separate from the remote collector.  It never
loads a model and it never changes a frozen phase decision.  All estimates are
computed from the append-only journal, with item IDs as the bootstrap unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate5 import (  # noqa: E402
    CONDITIONS,
    SOURCE_CONDITIONS,
    classify_gate5,
    classify_manipulation,
    classify_source,
    independent_estimands,
    source_disagreement,
)

BOOTSTRAP_RESAMPLES = 5_000
BOOTSTRAP_SEED = 2_026_0820
VALID_STATUSES = {"VALID_CORRECT", "VALID_WRONG"}
ERROR_STATUSES = VALID_STATUSES | {
    "INVALID_FORMAT",
    "TRUNCATED",
    "TRUNCATED_THINKING",
    "RUNTIME_ERROR",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _status(row: dict[str, Any]) -> str:
    return str(row.get("status", "RUNTIME_ERROR"))


def _valid(row: dict[str, Any]) -> bool:
    return _status(row) in VALID_STATUSES


def _error(row: dict[str, Any]) -> bool:
    return _status(row) != "VALID_CORRECT"


def _tokens(row: dict[str, Any]) -> int:
    return int(row.get("generated_token_count", 0) or 0)


def _semantic_outcome(row: dict[str, Any]) -> str:
    """Return a deterministic answer/status category for source comparisons."""

    if _valid(row) and row.get("parsed_answer") is not None:
        return f"ANSWER:{row['parsed_answer']}"
    return f"STATUS:{_status(row)}"


def _raw_tokens(row: dict[str, Any]) -> tuple[int, ...]:
    value = row.get("generated_token_ids", ())
    if isinstance(value, str):
        return tuple()
    return tuple(int(token) for token in value)


def _key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))


def _partition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    phases: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, int]] = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate Gate-5 logical row: {key}")
        seen.add(key)
        phases.setdefault(str(row.get("phase", "UNKNOWN")), []).append(row)
    return phases


def _keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {_key(row): row for row in rows}


def _require_schedule(
    rows: list[dict[str, Any]],
    *,
    conditions: tuple[str, ...],
    rollouts: tuple[int, ...],
    n_items: int,
) -> tuple[list[str], dict[tuple[str, str, int], dict[str, Any]]]:
    keyed = _keyed(rows)
    item_ids = sorted({key[0] for key in keyed})
    if len(item_ids) != n_items:
        raise ValueError(f"expected {n_items} items, observed {len(item_ids)}")
    expected = {
        (item_id, condition, rollout)
        for item_id in item_ids
        for condition in conditions
        for rollout in rollouts
    }
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"incomplete phase: missing={missing[:3]}, extra={extra[:3]}")
    return item_ids, keyed


def _summary(
    rows: list[dict[str, Any]], *, condition: str, n_items: int, rollouts: int
) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    expected = n_items * rollouts
    if len(selected) != expected:
        raise ValueError(f"{condition}: expected {expected} rows, observed {len(selected)}")
    counts = Counter(_status(row) for row in selected)
    valid_count = sum(counts[status] for status in VALID_STATUSES)
    token_values = [_tokens(row) for row in selected]
    return {
        "condition": condition,
        "n": expected,
        "valid": valid_count,
        "correct": counts["VALID_CORRECT"],
        "wrong": counts["VALID_WRONG"],
        "invalid": counts["INVALID_FORMAT"],
        "truncated": counts["TRUNCATED"] + counts["TRUNCATED_THINKING"],
        "runtime_error": counts["RUNTIME_ERROR"],
        "validity": valid_count / expected,
        "accuracy": counts["VALID_CORRECT"] / expected,
        "mean_tokens": float(np.mean(token_values)),
        "median_tokens": float(np.median(token_values)),
        "max_tokens": max(token_values),
        "status_counts": dict(counts),
    }


def _source_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    counts = Counter(_status(row) for row in selected)
    tokens = [_tokens(row) for row in selected]
    valid_count = sum(counts[status] for status in VALID_STATUSES)
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
        "mean_tokens": float(np.mean(tokens)),
        "median_tokens": float(np.median(tokens)),
        "max_tokens": max(tokens),
        "semantic_outcome_distribution": dict(Counter(_semantic_outcome(row) for row in selected)),
        "status_counts": dict(counts),
    }


def _source_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    item_ids, keyed = _require_schedule(
        rows, conditions=SOURCE_CONDITIONS, rollouts=(0, 1), n_items=40
    )
    outcomes: dict[str, dict[str, list[str]]] = {condition: {} for condition in SOURCE_CONDITIONS}
    for item_id in item_ids:
        for condition in SOURCE_CONDITIONS:
            outcomes[condition][item_id] = [
                _semantic_outcome(keyed[(item_id, condition, rollout)]) for rollout in (0, 1)
            ]
    disagreement = source_disagreement(outcomes)
    summaries = {condition: _source_summary(rows, condition) for condition in SOURCE_CONDITIONS}
    metrics = {
        "careful_validity": summaries["CAREFUL"]["validity"],
        "direct_validity": summaries["DIRECT"]["validity"],
        **disagreement,
    }
    return summaries, {**metrics, "outcomes": outcomes, "item_ids": item_ids}


def _source_bootstrap(rows: list[dict[str, Any]], n_resamples: int) -> dict[str, Any]:
    item_ids, keyed = _require_schedule(
        rows, conditions=SOURCE_CONDITIONS, rollouts=(0, 1), n_items=40
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = {"X": [], "W": [], "S": [], "accuracy_difference": []}
    for _ in range(n_resamples):
        sample = rng.integers(0, len(item_ids), size=len(item_ids))
        cross: list[bool] = []
        within_careful: list[bool] = []
        within_direct: list[bool] = []
        accuracy: dict[str, list[int]] = {condition: [] for condition in SOURCE_CONDITIONS}
        for position in sample:
            item_id = item_ids[int(position)]
            careful = [
                _semantic_outcome(keyed[(item_id, "CAREFUL", rollout)]) for rollout in (0, 1)
            ]
            direct = [_semantic_outcome(keyed[(item_id, "DIRECT", rollout)]) for rollout in (0, 1)]
            cross.extend(left != right for left in careful for right in direct)
            within_careful.append(careful[0] != careful[1])
            within_direct.append(direct[0] != direct[1])
            for condition in SOURCE_CONDITIONS:
                accuracy[condition].extend(
                    int(keyed[(item_id, condition, rollout)]["status"] == "VALID_CORRECT")
                    for rollout in (0, 1)
                )
        x = float(np.mean(cross))
        w = float(0.5 * (np.mean(within_careful) + np.mean(within_direct)))
        values["X"].append(x)
        values["W"].append(w)
        values["S"].append(x - w)
        values["accuracy_difference"].append(
            float(np.mean(accuracy["CAREFUL"]) - np.mean(accuracy["DIRECT"]))
        )
    return _percentile_intervals(values, n_resamples=n_resamples)


def _percentile_intervals(values: dict[str, list[float]], *, n_resamples: int) -> dict[str, Any]:
    return {
        "method": "item_cluster_percentile_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "n_resamples": n_resamples,
        "confidence": 0.95,
        "intervals": {
            name: {
                "estimate": float(np.mean(samples)),
                "lower": float(np.percentile(samples, 2.5)),
                "upper": float(np.percentile(samples, 97.5)),
            }
            for name, samples in values.items()
        },
    }


def _matrix(
    item_ids: list[str], keyed: dict[tuple[str, str, int], dict[str, Any]], condition: str
) -> np.ndarray:
    return np.asarray(
        [
            [_error(keyed[(item_id, condition, rollout)]) for rollout in (0, 1)]
            for item_id in item_ids
        ],
        dtype=bool,
    )


def _summary_matrices(
    item_ids: list[str], keyed: dict[tuple[str, str, int], dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    matrices = {condition: _matrix(item_ids, keyed, condition) for condition in CONDITIONS}
    summaries: dict[str, dict[str, Any]] = {}
    all_rows = list(keyed.values())
    for condition in CONDITIONS:
        summaries[condition] = _summary(
            all_rows, condition=condition, n_items=len(item_ids), rollouts=2
        )
    return matrices, summaries


def _attach_validity(values: dict[str, float], summary: dict[str, Any]) -> dict[str, float]:
    return {
        **values,
        "validity": float(summary["validity"]),
        "accuracy": float(summary["accuracy"]),
    }


def _evaluation_metrics(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[str],
]:
    item_ids, keyed = _require_schedule(rows, conditions=CONDITIONS, rollouts=(0, 1), n_items=60)
    matrices, summaries = _summary_matrices(item_ids, keyed)
    valid_matrices = {
        condition: np.asarray(
            [
                [_valid(keyed[(item_id, condition, rollout)]) for rollout in (0, 1)]
                for item_id in item_ids
            ],
            dtype=bool,
        )
        for condition in CONDITIONS
    }
    estimates: dict[str, dict[str, Any]] = {"BASELINE": _attach_validity({}, summaries["BASELINE"])}
    for condition in CONDITIONS:
        if condition == "BASELINE":
            continue
        estimates[condition] = _attach_validity(
            independent_estimands(matrices["BASELINE"], matrices[condition]),
            summaries[condition],
        )
    return estimates, matrices, valid_matrices, item_ids


def _evaluation_bootstrap(
    matrices: dict[str, np.ndarray],
    valid_matrices: dict[str, np.ndarray],
    n_resamples: int,
) -> dict[str, Any]:
    n_items = matrices["BASELINE"].shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names = list(CONDITIONS[1:])
    metric_names = (
        "accuracy_change",
        "validity_change",
        "G",
        "C",
        "D",
        "rescue",
        "damage",
    )
    values: dict[str, list[float]] = {
        f"{condition}:{metric}": [] for condition in names for metric in metric_names
    }
    for _ in range(n_resamples):
        indices = rng.integers(0, n_items, size=n_items)
        baseline = matrices["BASELINE"][indices]
        baseline_valid = valid_matrices["BASELINE"][indices]
        point: dict[str, dict[str, float]] = {}
        for condition in names:
            result = independent_estimands(baseline, matrices[condition][indices])
            point[condition] = result
            values[f"{condition}:accuracy_change"].append(
                result["accuracy_condition"] - result["accuracy_baseline"]
            )
            values[f"{condition}:validity_change"].append(
                float(np.mean(valid_matrices[condition][indices]) - np.mean(baseline_valid))
            )
            for metric in ("G", "C", "D", "rescue", "damage"):
                values[f"{condition}:{metric}"].append(result[metric])
        random_conditions = [f"SUSTAINED_RANDOM_R{i}" for i in range(4)]
        for metric in ("G", "C", "D"):
            random_mean = float(np.mean([point[name][metric] for name in random_conditions]))
            for sign in ("PLUS", "MINUS"):
                key = f"SUSTAINED_{sign}:{metric}_minus_random_mean"
                values.setdefault(key, []).append(point[f"SUSTAINED_{sign}"][metric] - random_mean)
        for sign in ("PLUS", "MINUS"):
            key = f"SUSTAINED_{sign}:D_minus_one_shot"
            values.setdefault(key, []).append(
                point[f"SUSTAINED_{sign}"]["D"] - point[f"ONE_SHOT_{sign}"]["D"]
            )
    return _percentile_intervals(values, n_resamples=n_resamples)


def _manipulation_metrics(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], list[str]]:
    item_ids, keyed = _require_schedule(rows, conditions=CONDITIONS, rollouts=(0,), n_items=20)
    by_condition: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        selected = [keyed[(item_id, condition, 0)] for item_id in item_ids]
        by_condition[condition] = {
            "validity": float(np.mean([_valid(row) for row in selected])),
            "accuracy": float(np.mean([row["status"] == "VALID_CORRECT" for row in selected])),
            "semantic_change_rate": 0.0,
            "raw_sequence_change_rate": 0.0,
        }
    for condition in CONDITIONS:
        if condition == "BASELINE":
            continue
        semantic_changes = []
        raw_changes = []
        first_divergence: list[int | None] = []
        for item_id in item_ids:
            baseline = keyed[(item_id, "BASELINE", 0)]
            current = keyed[(item_id, condition, 0)]
            semantic_changes.append(_semantic_outcome(baseline) != _semantic_outcome(current))
            left = _raw_tokens(baseline)
            right = _raw_tokens(current)
            raw_changes.append(left != right)
            first = None
            for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
                if left_token != right_token:
                    first = index
                    break
            if first is None and len(left) != len(right):
                first = min(len(left), len(right))
            first_divergence.append(first)
        by_condition[condition]["semantic_change_rate"] = float(np.mean(semantic_changes))
        by_condition[condition]["raw_sequence_change_rate"] = float(np.mean(raw_changes))
        by_condition[condition]["first_divergence_mean"] = (
            float(np.mean([value for value in first_divergence if value is not None]))
            if any(value is not None for value in first_divergence)
            else None
        )
        by_condition[condition]["first_divergence_count"] = float(
            sum(value is not None for value in first_divergence)
        )
    random_names = [f"SUSTAINED_RANDOM_R{i}" for i in range(4)]
    random_changes = [by_condition[name]["semantic_change_rate"] for name in random_names]
    by_condition["RANDOM_SUMMARY"] = {
        "semantic_change_rate_mean": float(np.mean(random_changes)),
        "semantic_change_rate_max": float(np.max(random_changes)),
        "raw_sequence_change_rate_mean": float(
            np.mean([by_condition[name]["raw_sequence_change_rate"] for name in random_names])
        ),
        "raw_sequence_change_rate_max": float(
            np.max([by_condition[name]["raw_sequence_change_rate"] for name in random_names])
        ),
    }
    for sign in ("PLUS", "MINUS"):
        for prefix in ("semantic_change_rate", "raw_sequence_change_rate"):
            by_condition[f"CONTRAST_{sign}"] = by_condition.get(f"CONTRAST_{sign}", {})
            by_condition[f"CONTRAST_{sign}"][f"sustained_minus_one_shot_{prefix}"] = (
                by_condition[f"SUSTAINED_{sign}"][prefix] - by_condition[f"ONE_SHOT_{sign}"][prefix]
            )
            by_condition[f"CONTRAST_{sign}"][f"sustained_minus_random_mean_{prefix}"] = (
                by_condition[f"SUSTAINED_{sign}"][prefix]
                - by_condition["RANDOM_SUMMARY"][f"{prefix}_mean"]
            )
            by_condition[f"CONTRAST_{sign}"][f"sustained_minus_random_max_{prefix}"] = (
                by_condition[f"SUSTAINED_{sign}"][prefix]
                - by_condition["RANDOM_SUMMARY"][f"{prefix}_max"]
            )
    pass_gate = classify_manipulation(by_condition)
    return by_condition, {"manipulation_pass": pass_gate}, item_ids


def _gate6_protocol(source_classification: str, gate5_classification: str) -> tuple[str, str]:
    if gate5_classification in {
        "GATE5_SUSTAINED_USEFUL_COMPLEMENTARITY_SIGNAL",
        "GATE5_SUSTAINED_ERROR_PROFILE_MOVEMENT_ONLY",
    }:
        name = "GATE6_DISTRIBUTED_REPLICATION_PROTOCOL"
        focus = (
            "Replicate the sustained controller with a larger sample, four random controls, "
            "and character-count transfer."
        )
    else:
        name = "GATE6_LAYER_SOURCE_ATLAS_PROTOCOL"
        focus = (
            "Test layer-specific, execution-boundary or verified-trajectory sources without "
            "selecting layers from semantic outcomes."
        )
    if source_classification != "SOURCE_SEMANTIC_BEHAVIOR_PASS":
        focus += (
            " Because source labels did not pass the behavioral separation gate, require "
            "execution-boundary or verified-trajectory sources."
        )
    return name, focus


def _report(
    output: Path,
    source_summaries: dict[str, Any],
    source_values: dict[str, Any],
    source_classification: str,
    manipulation: dict[str, Any],
    manipulation_pass: bool,
    evaluation: dict[str, Any] | None,
    gate5_classification: str,
    gate6_name: str,
    gate6_focus: str,
    cost: dict[str, Any],
) -> None:
    lines = [
        "GATE 5 — SOURCE VALIDITY AND TEMPORAL PERSISTENCE",
        "======================================================================",
        "",
        "This is a development bridge. Gate-4 historical artifacts remain immutable.",
        "",
        "## Gate-4 audit",
        "",
        "classification: GATE4_AUDIT_MINOR_NONSCIENTIFIC_ISSUES",
        "scientific result changed: NO",
        "provenance corrections: documented in review/micro_q1/forensic_audit/",
        "",
        "## Source check",
        "",
        "| condition | valid | accuracy | mean tokens | median tokens | max tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in SOURCE_CONDITIONS:
        summary = source_summaries[condition]
        lines.append(
            f"| {condition} | {summary['validity']:.3f} | {summary['accuracy']:.3f} | "
            f"{summary['mean_tokens']:.1f} | {summary['median_tokens']:.1f} | "
            f"{summary['max_tokens']} |"
        )
    lines.extend(
        [
            "",
            f"cross disagreement X: {source_values['X_cross_disagreement']:.6f}",
            f"within disagreement W: {source_values['W_within_disagreement']:.6f}",
            f"excess source disagreement S: {source_values['S_excess']:.6f}",
            f"source classification: {source_classification}",
            "",
            "## Sustained engineering",
            "",
            "Engineering checks are recorded in SUSTAINED_ENGINEERING_CHECKS.json.",
            "",
            "## Manipulation gate",
            "",
            "| condition | validity | semantic change | raw change | first divergence mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, values in manipulation.items():
        if condition.startswith("CONTRAST_") or condition == "RANDOM_SUMMARY":
            continue
        lines.append(
            f"| {condition} | {values.get('validity', float('nan')):.3f} | "
            f"{values.get('semantic_change_rate', float('nan')):.3f} | "
            f"{values.get('raw_sequence_change_rate', float('nan')):.3f} | "
            f"{values.get('first_divergence_mean', float('nan'))} |"
        )
    lines.extend(
        [
            "",
            f"manipulation pass: {'YES' if manipulation_pass else 'NO'}",
            "",
            "## Primary evaluation",
            "",
        ]
    )
    if evaluation is None:
        lines.append("not executed because the manipulation gate failed.")
    else:
        lines.extend(
            [
                "| condition | validity | accuracy | G | C | D | rescue | damage |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, values in evaluation.items():
            lines.append(
                f"| {condition} | {values['validity']:.3f} | {values['accuracy']:.3f} | "
                f"{values.get('G', 0.0):.4f} | {values.get('C', 0.0):.4f} | "
                f"{values.get('D', 0.0):.4f} | {values.get('rescue', 0.0):.4f} | "
                f"{values.get('damage', 0.0):.4f} |"
            )
        random_conditions = [f"SUSTAINED_RANDOM_R{i}" for i in range(4)]
        lines.extend(
            [
                "",
                "### Random-control null",
                "",
                "| metric | mean | median | min | max |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for metric in ("G", "C", "D"):
            values = [float(evaluation[name][metric]) for name in random_conditions]
            lines.append(
                f"| {metric} | {np.mean(values):.4f} | {np.median(values):.4f} | "
                f"{np.min(values):.4f} | {np.max(values):.4f} |"
            )
        lines.extend(["", "### Duration contrasts", ""])
        for sign in ("PLUS", "MINUS"):
            sustained = evaluation[f"SUSTAINED_{sign}"]
            one_shot = evaluation[f"ONE_SHOT_{sign}"]
            lines.append(
                f"- {sign}: D(sustained) - D(one-shot) = "
                f"{sustained['D'] - one_shot['D']:.4f}; "
                f"G contrast = {sustained['G'] - one_shot['G']:.4f}; "
                f"C contrast = {sustained['C'] - one_shot['C']:.4f}."
            )
        baseline = evaluation["BASELINE"]
        lines.extend(
            [
                "",
                "### Frozen guards",
                "",
                f"- baseline accuracy: {baseline['accuracy']:.4f}",
                f"- baseline validity: {baseline['validity']:.4f}",
                "- meaningful signs satisfy validity and competence guards: "
                + str(
                    all(
                        evaluation[f"SUSTAINED_{sign}"]["validity"] >= 0.90
                        and evaluation[f"SUSTAINED_{sign}"]["validity"]
                        >= baseline["validity"] - 0.05
                        and evaluation[f"SUSTAINED_{sign}"]["accuracy"]
                        >= baseline["accuracy"] - 0.10
                        for sign in ("PLUS", "MINUS")
                    )
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Classification",
            "",
            gate5_classification,
            "",
            "## Cost",
            "",
            f"trajectory elapsed seconds: {cost['trajectory_elapsed_seconds']:.3f}",
            f"generated tokens: {cost['generated_token_count']}",
            f"Pod wall-clock seconds: {cost.get('pod_wallclock_runtime_seconds')}",
            f"estimated A40 cost USD: {cost.get('pod_wallclock_cost_usd')}",
            "",
            "## Gate-6 draft",
            "",
            f"protocol: {gate6_name}",
            f"focus: {gate6_focus}",
            "",
            "## Firewall",
            "",
            "original steering beyond this frozen bridge: NOT RUN",
            "Q2: NOT RUN",
            "character count: NOT RUN",
            "confirmatory holdout: UNTOUCHED",
            "RunPod: STOPPED after collection",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pod-runtime-seconds", type=float, default=None)
    parser.add_argument("--pod-id", default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.journal)
    phases = _partition(rows)
    source_rows = phases.get("SOURCE_CHECK", [])
    manipulation_rows = phases.get("SUSTAINED_MANIPULATION", [])
    evaluation_rows = phases.get("SUSTAINED_EVALUATION", [])
    if not source_rows or not manipulation_rows:
        raise ValueError("Gate-5 source and manipulation phases are required for analysis")

    source_summaries, source_values = _source_metrics(source_rows)
    source_bootstrap = _source_bootstrap(source_rows, BOOTSTRAP_RESAMPLES)
    source_classification = classify_source(
        {
            "careful_validity": source_values["careful_validity"],
            "direct_validity": source_values["direct_validity"],
            "X_cross_disagreement": source_values["X_cross_disagreement"],
            "S_excess": source_values["S_excess"],
            "careful_mean_tokens": source_summaries["CAREFUL"]["mean_tokens"],
            "direct_mean_tokens": source_summaries["DIRECT"]["mean_tokens"],
            "careful_median_tokens": source_summaries["CAREFUL"]["median_tokens"],
            "direct_median_tokens": source_summaries["DIRECT"]["median_tokens"],
        }
    )

    manipulation, manipulation_gate, _ = _manipulation_metrics(manipulation_rows)
    manipulation_pass = bool(manipulation_gate["manipulation_pass"])
    evaluation_estimates: dict[str, Any] | None = None
    evaluation_bootstrap: dict[str, Any] | None = None
    if manipulation_pass and evaluation_rows:
        evaluation_estimates, matrices, valid_matrices, _ = _evaluation_metrics(evaluation_rows)
        evaluation_bootstrap = _evaluation_bootstrap(matrices, valid_matrices, BOOTSTRAP_RESAMPLES)
        gate5_classification = classify_gate5(
            evaluation_estimates, engineering_pass=True, manipulation_pass=True
        )
    else:
        gate5_classification = "GATE5_NO_BEHAVIORAL_FIRST_STAGE"

    gate6_name, gate6_focus = _gate6_protocol(source_classification, gate5_classification)
    _write_json(
        args.output / "SOURCE_ESTIMANDS.json",
        {
            "summaries": source_summaries,
            "point": {
                key: value
                for key, value in source_values.items()
                if key not in {"outcomes", "item_ids"}
            },
            "bootstrap": source_bootstrap,
            "classification": source_classification,
        },
    )
    _write_json(
        args.output / "MANIPULATION_ESTIMANDS.json",
        {"conditions": manipulation, "pass": manipulation_pass},
    )
    _write_json(
        args.output / "ESTIMANDS.json",
        {
            "source_classification": source_classification,
            "manipulation_pass": manipulation_pass,
            "gate5_classification": gate5_classification,
            "evaluation": evaluation_estimates,
        },
    )
    _write_json(
        args.output / "BOOTSTRAP_INTERVALS.json",
        {"source": source_bootstrap, "evaluation": evaluation_bootstrap},
    )
    source_fields = [
        "condition",
        "n",
        "valid",
        "correct",
        "wrong",
        "invalid",
        "truncated",
        "runtime_error",
        "validity",
        "accuracy",
        "mean_tokens",
        "median_tokens",
        "max_tokens",
    ]
    _write_csv(
        args.output / "SOURCE_RESULTS.csv",
        [source_summaries[condition] for condition in SOURCE_CONDITIONS],
        source_fields,
    )
    manipulation_fields = [
        "condition",
        "validity",
        "accuracy",
        "semantic_change_rate",
        "raw_sequence_change_rate",
        "first_divergence_mean",
        "first_divergence_count",
    ]
    _write_csv(
        args.output / "MANIPULATION_RESULTS.csv",
        [
            {"condition": condition, **values}
            for condition, values in manipulation.items()
            if not condition.startswith("CONTRAST_") and condition != "RANDOM_SUMMARY"
        ],
        manipulation_fields,
    )
    if evaluation_estimates is not None:
        evaluation_fields = [
            "condition",
            "validity",
            "accuracy",
            "B00",
            "B0j",
            "O00",
            "O0j",
            "G",
            "U00",
            "U0j",
            "C",
            "D",
            "rescue",
            "damage",
        ]
        _write_csv(
            args.output / "CONDITION_SUMMARY.csv",
            [
                {"condition": condition, **values}
                for condition, values in evaluation_estimates.items()
            ],
            evaluation_fields,
        )
    else:
        _write_csv(args.output / "CONDITION_SUMMARY.csv", [], ["condition", "status"])
    trajectory_elapsed_seconds = float(sum(float(row.get("elapsed_seconds", 0.0)) for row in rows))
    generated_token_count = int(sum(int(row.get("generated_token_count", 0)) for row in rows))
    cost = {
        "rate_usd_per_a40_hour": 0.44,
        "scientific_trajectories_collected": len(rows),
        "trajectory_elapsed_seconds": trajectory_elapsed_seconds,
        "trajectory_elapsed_cost_usd": trajectory_elapsed_seconds / 3600.0 * 0.44,
        "generated_token_count": generated_token_count,
        "analysis_gpu_runtime_seconds": 0.0,
        "analysis_gpu_cost_usd": 0.0,
        "pod_id": args.pod_id,
        "pod_wallclock_runtime_seconds": args.pod_runtime_seconds,
        "pod_wallclock_cost_usd": (
            None
            if args.pod_runtime_seconds is None
            else args.pod_runtime_seconds / 3600.0 * 0.44
        ),
        "note": (
            "Analysis is CPU-only and performed after Pod shutdown; Pod wall-clock "
            "cost is an external runtime estimate."
        ),
    }
    _write_json(
        args.output / "COST.json",
        cost,
    )
    hash_names = (
        "PROTOCOL_LOCK.json",
        "SOURCE_CHECK.json",
        "SUSTAINED_MANIPULATION.json",
        "SUSTAINED_EVALUATION.json",
        "HISTORICAL_EXCLUSION_DIGEST.json",
        "RANDOM_BANK_METADATA.json",
        "DIRECTION.npy",
        "R0.npy",
        "R1.npy",
        "R2.npy",
        "R3.npy",
        "journal.jsonl",
    )
    hashes = {}
    for name in hash_names:
        path = args.output / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(args.output / "manifest_hashes.json", hashes)
    _write_json(
        args.output / "GATE5_CLASSIFICATION.json",
        {
            "source_classification": source_classification,
            "manipulation_pass": manipulation_pass,
            "primary_evaluation_executed": evaluation_estimates is not None,
            "gate5_classification": gate5_classification,
            "gate6_protocol": gate6_name,
        },
    )
    _report(
        args.output,
        source_summaries,
        source_values,
        source_classification,
        manipulation,
        manipulation_pass,
        evaluation_estimates,
        gate5_classification,
        gate6_name,
        gate6_focus,
        cost,
    )
    (args.output / f"{gate6_name}.md").write_text(
        f"# {gate6_name}\n\n{gate6_focus}\n\nStatus: DRAFT ONLY; NOT EXECUTED.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
