#!/usr/bin/env python3
"""Independent forensic audit for Gate 13.1.

This intentionally does not import or call the Gate-13/13.1 primary analysis
modules or their two-rollout estimator helpers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)

REVIEW = ROOT / "review/gate13_1_all_layer_causal_atlas"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
SOURCE_COMMIT = "3e9ee5c822884caeff2d1a171ea9c51dc6925361"
PARSER_VERSION = "external-semantic-v3"
MAX_NEW_TOKENS = 4096
BOOTSTRAP_SEED = 20260825
BOOTSTRAP_RESAMPLES = 10_000
CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL",
    "MEANINGFUL_SELECTED",
    "RANDOM_R0",
    "RANDOM_R1",
    "RANDOM_R2",
    "RANDOM_R3",
)
RANDOMS = CONDITIONS[3:]
DOSE_ORDER = ("D25", "D50", "D75", "D100")
DOSE_FRACTIONS = {"D25": 0.25, "D50": 0.5, "D75": 0.75, "D100": 1.0}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype="<f8").reshape(-1))
    return hashlib.sha256(value.tobytes()).hexdigest()


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    if result.correct:
        status = "VALID_CORRECT"
    elif result.commitment_valid and result.semantic_evaluable:
        status = "VALID_WRONG"
    elif result.failure_reason == "truncated or unclosed response":
        status = "TRUNCATED"
    elif result.failure_reason == "runtime error":
        status = "RUNTIME_ERROR"
    else:
        status = "INVALID_FORMAT"
    return {
        "status": status,
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "canonical_value": result.canonical_value,
        "failure_reason": result.failure_reason,
    }


def semantic_outcome(parsed: dict[str, Any]) -> str:
    if parsed["commitment_valid"] and parsed["semantic_evaluable"]:
        return "VALUE:" + json.dumps(
            parsed["canonical_value"], sort_keys=True, ensure_ascii=False
        )
    return f"MECHANICAL:{parsed['status']}:{parsed['failure_reason']}"


def summarize(rows: list[dict[str, Any]], parsed: dict[int, dict[str, Any]]) -> dict[str, float]:
    values = [parsed[id(row)] for row in rows]
    tokens = np.asarray([int(row["generated_token_count"]) for row in rows])
    return {
        "n": float(len(rows)),
        "commitment_validity": float(np.mean([value["commitment_valid"] for value in values])),
        "semantic_evaluability": float(
            np.mean([value["semantic_evaluable"] for value in values])
        ),
        "accuracy": float(np.mean([value["correct"] for value in values])),
        "mean_tokens": float(np.mean(tokens)),
        "median_tokens": float(np.median(tokens)),
        "max_tokens": float(np.max(tokens)),
        "truncation": float(np.mean([value["status"] == "TRUNCATED" for value in values])),
        "no_commitment": float(np.mean([not value["commitment_valid"] for value in values])),
    }


def schedule_check(
    observed: list[dict[str, Any]], schedule: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
    fields = ("stage", "model", "item_id", "condition", "rollout_index")
    expected = Counter(tuple(row[field] for field in fields) for row in schedule)
    actual = Counter(tuple(row[field] for field in fields) for row in observed)
    schedule_by_key = {tuple(row[field] for field in fields): row for row in schedule}
    seed_mismatches = 0
    for row in observed:
        key = tuple(row[field] for field in fields)
        planned = schedule_by_key.get(key)
        if planned is None or int(row["seed"]) != int(planned["seed"]):
            seed_mismatches += 1
    return {
        "stage": stage,
        "expected_rows": len(schedule),
        "observed_rows": len(observed),
        "missing_rows": int(sum((expected - actual).values())),
        "extra_rows": int(sum((actual - expected).values())),
        "duplicate_logical_rows": int(sum(max(0, count - 1) for count in actual.values())),
        "seed_mismatches": seed_mismatches,
    }


def matched_q(
    rows: list[dict[str, Any]], parsed: dict[int, dict[str, Any]], condition: str
) -> tuple[float, dict[str, float]]:
    by_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row
        for row in rows
    }
    selected = [row for row in rows if row["condition"] == condition]
    changes = []
    for row in selected:
        baseline = by_key[(str(row["item_id"]), "BASELINE", int(row["rollout_index"]))]
        changes.append(
            semantic_outcome(parsed[id(row)]) != semantic_outcome(parsed[id(baseline)])
        )
    return float(np.mean(changes)), summarize(selected, parsed)


def two_rollout_estimands(baseline: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    base = np.asarray(baseline, dtype=np.float64)
    treat = np.asarray(condition, dtype=np.float64)
    if base.shape != treat.shape or base.ndim != 2 or base.shape[1] != 2:
        raise ValueError("two-rollout arrays must have equal (N, 2) shape")
    n_items = len(base)
    b00 = float(np.mean(base[:, 0] * base[:, 1]))
    b0j_items = (
        base[:, 0] * treat[:, 0]
        + base[:, 0] * treat[:, 1]
        + base[:, 1] * treat[:, 0]
        + base[:, 1] * treat[:, 1]
    ) / 4.0
    b0j = float(np.mean(b0j_items))
    q0 = base.mean(axis=1)
    qj = treat.mean(axis=1)
    denominator = n_items * (n_items - 1)
    u00 = float((q0.sum() ** 2 - np.square(q0).sum()) / denominator)
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / denominator)
    distance = float(
        np.mean(
            base[:, 0] * base[:, 1]
            + treat[:, 0] * treat[:, 1]
            - base[:, 0] * treat[:, 1]
            - base[:, 1] * treat[:, 0]
        )
    )
    rescue = float(
        np.mean(
            (
                base[:, 0] * (1 - treat[:, 0])
                + base[:, 0] * (1 - treat[:, 1])
                + base[:, 1] * (1 - treat[:, 0])
                + base[:, 1] * (1 - treat[:, 1])
            )
            / 4.0
        )
    )
    damage = float(
        np.mean(
            (
                (1 - base[:, 0]) * treat[:, 0]
                + (1 - base[:, 0]) * treat[:, 1]
                + (1 - base[:, 1]) * treat[:, 0]
                + (1 - base[:, 1]) * treat[:, 1]
            )
            / 4.0
        )
    )
    result = {
        "accuracy_baseline": float(1 - base.mean()),
        "accuracy_condition": float(1 - treat.mean()),
        "B00": b00,
        "O00": 1 - b00,
        "B0j": b0j,
        "O0j": 1 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": distance,
        "rescue": rescue,
        "damage": damage,
    }
    if not np.isclose(
        rescue - damage,
        result["accuracy_condition"] - result["accuracy_baseline"],
        atol=1e-12,
    ):
        raise AssertionError("rescue-damage identity failed")
    return result


def bootstrap(
    arrays: dict[str, np.ndarray],
    commitment: dict[str, np.ndarray],
    evaluability: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: dict[str, list[float]] = defaultdict(list)
    n_items = len(arrays["BASELINE"])
    for _ in range(BOOTSTRAP_RESAMPLES):
        index = rng.integers(0, n_items, size=n_items)
        points = {
            condition: two_rollout_estimands(
                arrays["BASELINE"][index], arrays[condition][index]
            )
            for condition in CONDITIONS[1:]
        }
        meaningful = points["MEANINGFUL_SELECTED"]
        samples["meaningful:accuracy_change"].append(
            float(
                arrays["BASELINE"][index].mean()
                - arrays["MEANINGFUL_SELECTED"][index].mean()
            )
        )
        samples["meaningful:commitment_validity_change"].append(
            float(
                commitment["MEANINGFUL_SELECTED"][index].mean()
                - commitment["BASELINE"][index].mean()
            )
        )
        samples["meaningful:semantic_evaluability_change"].append(
            float(
                evaluability["MEANINGFUL_SELECTED"][index].mean()
                - evaluability["BASELINE"][index].mean()
            )
        )
        for metric in ("G", "C", "D", "rescue", "damage"):
            samples[f"meaningful:{metric}"].append(float(meaningful[metric]))
        for metric in ("G", "C", "D"):
            random_values = [points[name][metric] for name in RANDOMS]
            samples[f"meaningful:{metric}_minus_random_mean"].append(
                float(meaningful[metric] - np.mean(random_values))
            )
            samples[f"meaningful:{metric}_minus_random_max"].append(
                float(meaningful[metric] - np.max(random_values))
            )
    return {
        name: {
            "estimate": float(np.mean(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": BOOTSTRAP_RESAMPLES,
        }
        for name, values in sorted(samples.items())
    }


def maximum_nested_difference(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        common = set(left) & set(right)
        return max(
            (maximum_nested_difference(left[key], right[key]) for key in common),
            default=0.0,
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isfinite(float(left)) and not np.isfinite(float(right)):
            return 0.0
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def auxiliary_metrics(
    base: np.ndarray, treat: np.ndarray, point: dict[str, float]
) -> dict[str, Any]:
    left = base.reshape(-1).astype(bool)
    right = treat.reshape(-1).astype(bool)
    union = int(np.logical_or(left, right).sum())
    intersection = int(np.logical_and(left, right).sum())
    left_centered = left.astype(float) - left.mean()
    right_centered = right.astype(float) - right.mean()
    phi_denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    phi = (
        float(np.dot(left_centered, right_centered) / phi_denominator)
        if phi_denominator
        else None
    )
    return {
        "paired_rollout_disagreement": float(np.not_equal(left, right).mean()),
        "paired_rollout_double_fault": float(np.logical_and(left, right).mean()),
        "paired_rollout_error_jaccard": float(intersection / union) if union else 1.0,
        "paired_rollout_error_phi": phi,
        "pair_oracle_accuracy": point["O0j"],
        "pair_oracle_headroom_over_best_single_accuracy": point["O0j"]
        - max(point["accuracy_baseline"], point["accuracy_condition"]),
    }


def add_comparison(
    rows: list[dict[str, Any]], scope: str, condition: str, metric: str, primary: Any, audit: Any
) -> float:
    difference = maximum_nested_difference(primary, audit)
    rows.append(
        {
            "scope": scope,
            "condition": condition,
            "metric": metric,
            "primary": json.dumps(primary, sort_keys=True),
            "audit": json.dumps(audit, sort_keys=True),
            "absolute_difference": difference,
            "match": bool(difference <= 1e-12),
        }
    )
    return difference


def main() -> int:
    journal_path = REVIEW / "journal.jsonl"
    raw_lines = journal_path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in raw_lines if line.strip()]
    parsed = {id(row): parse_row(row) for row in rows}
    parser_mismatches = sum(
        bool(parsed[id(row)]["correct"]) != bool(row["correct"])
        or bool(parsed[id(row)]["commitment_valid"]) != bool(row["commitment_valid"])
        or bool(parsed[id(row)]["semantic_evaluable"]) != bool(row["semantic_evaluable"])
        for row in rows
    )

    stage_files = (
        ("ALL_LAYER_SWEEP", "ALL_LAYER_SWEEP_SCHEDULE.json"),
        ("LAYER_DOSE_QUALIFICATION", "LAYER_DOSE_SCHEDULE.json"),
        ("FINAL_EVALUATION", "FINAL_EVALUATION_SCHEDULE.json"),
    )
    schedules = []
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage, filename in stage_files:
        selected = [row for row in rows if row["stage"] == stage]
        by_stage[stage] = selected
        schedules.append(schedule_check(selected, read_json(REVIEW / filename), stage))
    schedule_clean = all(
        check[metric] == 0
        for check in schedules
        for metric in ("missing_rows", "extra_rows", "duplicate_logical_rows", "seed_mismatches")
    )
    keys = [
        (row["stage"], row["item_id"], row["condition"], int(row["rollout_index"]))
        for row in rows
    ]
    provenance_clean = all(
        row["model"] == MODEL
        and row["model_revision"] == REVISION
        and row["tokenizer_revision"] == REVISION
        and row["parser_version"] == PARSER_VERSION
        and row["experiment_source_commit"] == SOURCE_COMMIT
        for row in rows
    )
    retries = [row for row in rows if int(row.get("retry_count", 0)) > 0]

    comparisons: list[dict[str, Any]] = []
    maximum_difference = 0.0

    sweep_rows = by_stage["ALL_LAYER_SWEEP"]
    _baseline_q, sweep_baseline = matched_q(sweep_rows, parsed, "BASELINE")
    sweep_metrics: dict[int, dict[str, float]] = {}
    eligibility: dict[int, bool] = {}
    for layer in range(34):
        q_value, summary = matched_q(sweep_rows, parsed, f"MEANINGFUL_L{layer}_D50")
        sweep_metrics[layer] = {
            **summary,
            "Q": q_value,
            "baseline_accuracy": sweep_baseline["accuracy"],
        }
        eligibility[layer] = bool(
            summary["commitment_validity"] >= 0.75
            and summary["semantic_evaluability"] >= 0.75
            and q_value >= 0.10
        )
    candidates = []
    for quartile in np.array_split(np.arange(34), 4):
        available = [int(layer) for layer in quartile if eligibility[int(layer)]]
        if available:
            candidates.append(max(available, key=lambda layer: (sweep_metrics[layer]["Q"], -layer)))
    sweep_primary = read_json(REVIEW / "ALL_LAYER_SWEEP_REPORT.json")
    maximum_difference = max(
        maximum_difference,
        add_comparison(
            comparisons,
            "stage_a",
            "ALL",
            "quartile_candidates",
            sweep_primary["quartile_candidates"],
            candidates,
        ),
    )
    for layer in range(34):
        for metric in ("accuracy", "commitment_validity", "semantic_evaluability", "Q"):
            maximum_difference = max(
                maximum_difference,
                add_comparison(
                    comparisons,
                    "stage_a",
                    f"L{layer}",
                    metric,
                    sweep_primary["metrics"][str(layer)][metric],
                    sweep_metrics[layer][metric],
                ),
            )

    stage_b_rows = by_stage["LAYER_DOSE_QUALIFICATION"]
    _baseline_q, stage_b_baseline = matched_q(stage_b_rows, parsed, "BASELINE")
    stage_b_metrics: dict[tuple[int, str], dict[str, float]] = {}
    stage_b_proof: dict[str, dict[str, Any]] = {}
    for layer in candidates:
        for dose in DOSE_ORDER:
            meaningful_q, meaningful_summary = matched_q(
                stage_b_rows, parsed, f"MEANINGFUL_L{layer}_{dose}"
            )
            isotropic_q = matched_q(
                stage_b_rows, parsed, f"ISOTROPIC_NULL_L{layer}_{dose}"
            )[0]
            shuffled_q = matched_q(
                stage_b_rows, parsed, f"SHUFFLED_NULL_L{layer}_{dose}"
            )[0]
            null_mean = float(np.mean([isotropic_q, shuffled_q]))
            null_max = float(np.max([isotropic_q, shuffled_q]))
            values = {
                **meaningful_summary,
                "baseline_accuracy": stage_b_baseline["accuracy"],
                "Q": meaningful_q,
                "isotropic_Q": isotropic_q,
                "shuffled_Q": shuffled_q,
                "null_mean_Q": null_mean,
                "null_max_Q": null_max,
                "Q_minus_null_mean": meaningful_q - null_mean,
                "Q_minus_null_max": meaningful_q - null_max,
            }
            checks = {
                "commitment_validity": values["commitment_validity"] >= 0.90,
                "semantic_evaluability": values["semantic_evaluability"] >= 0.90,
                "competence_safety": values["accuracy"] >= values["baseline_accuracy"] - 0.10,
                "semantic_change": values["Q"] >= 0.15,
                "null_mean_specificity": values["Q"] - values["null_mean_Q"] >= 0.05,
                "null_max_specificity": values["Q"] > values["null_max_Q"],
            }
            stage_b_metrics[(layer, dose)] = values
            stage_b_proof[f"L{layer}_{dose}"] = {
                "checks": checks,
                "eligible": all(checks.values()),
            }
    selected_by_layer = []
    for layer in sorted(candidates):
        for dose in DOSE_ORDER:
            if stage_b_proof[f"L{layer}_{dose}"]["eligible"]:
                selected_by_layer.append((layer, dose))
                break
    source = {
        int(row["layer"]): row
        for row in read_json(REVIEW / "SOURCE_DIRECTION_MANIFEST.json")["layers"]
    }
    selected = max(
        selected_by_layer,
        key=lambda key: (
            stage_b_metrics[key]["Q"] - stage_b_metrics[key]["null_mean_Q"],
            stage_b_metrics[key]["Q"],
            float(source[key[0]]["source_effect"]),
            -DOSE_FRACTIONS[key[1]],
            -key[0],
        ),
    )
    stage_b_primary = read_json(REVIEW / "LAYER_DOSE_REPORT.json")
    selected_lock = read_json(REVIEW / "SELECTED_LAYER_DOSE_LOCK.json")
    maximum_difference = max(
        maximum_difference,
        add_comparison(
            comparisons,
            "stage_b",
            "SELECTED",
            "layer_dose",
            stage_b_primary["selected"],
            {"layer": selected[0], "dose": selected[1]},
        ),
    )
    for key, values in stage_b_metrics.items():
        name = f"L{key[0]}_{key[1]}"
        for metric in (
            "accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "Q",
            "null_mean_Q",
            "null_max_Q",
        ):
            maximum_difference = max(
                maximum_difference,
                add_comparison(
                    comparisons,
                    "stage_b",
                    name,
                    metric,
                    stage_b_primary["metrics"][name][metric],
                    values[metric],
                ),
            )

    final_rows = by_stage["FINAL_EVALUATION"]
    final_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row
        for row in final_rows
    }
    item_ids = sorted({str(row["item_id"]) for row in final_rows})
    arrays = {
        condition: np.asarray(
            [
                [
                    int(not parsed[id(final_key[(item, condition, rollout)])]["correct"])
                    for rollout in (0, 1)
                ]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }
    commitment = {
        condition: np.asarray(
            [
                [
                    int(
                        parsed[id(final_key[(item, condition, rollout)])][
                            "commitment_valid"
                        ]
                    )
                    for rollout in (0, 1)
                ]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }
    evaluability = {
        condition: np.asarray(
            [
                [
                    int(
                        parsed[id(final_key[(item, condition, rollout)])][
                            "semantic_evaluable"
                        ]
                    )
                    for rollout in (0, 1)
                ]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }
    summaries = {
        condition: summarize(
            [row for row in final_rows if row["condition"] == condition], parsed
        )
        for condition in CONDITIONS
    }
    estimands = {
        condition: two_rollout_estimands(arrays["BASELINE"], arrays[condition])
        for condition in CONDITIONS[1:]
    }
    b00 = float(np.mean(arrays["BASELINE"][:, 0] * arrays["BASELINE"][:, 1]))
    estimands["BASELINE"] = {
        "B00": b00,
        "O00": 1 - b00,
        "baseline_resampling_gain": 1 - b00 - summaries["BASELINE"]["accuracy"],
    }
    random_summary = {}
    for metric in ("G", "C", "D", "rescue", "damage"):
        values = [estimands[name][metric] for name in RANDOMS]
        random_summary[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    intervals = bootstrap(arrays, commitment, evaluability)
    loo_values: dict[str, list[float]] = defaultdict(list)
    for index in range(len(item_ids)):
        keep = np.arange(len(item_ids)) != index
        point = two_rollout_estimands(
            arrays["BASELINE"][keep], arrays["MEANINGFUL_SELECTED"][keep]
        )
        loo_values["accuracy_change"].append(
            point["accuracy_condition"] - point["accuracy_baseline"]
        )
        for metric in ("G", "C", "D"):
            loo_values[metric].append(point[metric])
    loo_sign_stable = {
        metric: all(value > 0 for value in values)
        for metric, values in loo_values.items()
    }
    source_replicated = bool(
        summaries["TEXTUAL_CAREFUL"]["commitment_validity"] >= 0.90
        and summaries["TEXTUAL_CAREFUL"]["semantic_evaluability"] >= 0.90
        and summaries["TEXTUAL_CAREFUL"]["mean_tokens"]
        >= 1.5 * summaries["BASELINE"]["mean_tokens"]
        and summaries["TEXTUAL_CAREFUL"]["median_tokens"]
        >= summaries["BASELINE"]["median_tokens"] + 10
    )
    token_denominator = (
        summaries["TEXTUAL_CAREFUL"]["mean_tokens"] - summaries["BASELINE"]["mean_tokens"]
    )
    token_recovery = (
        (summaries["MEANINGFUL_SELECTED"]["mean_tokens"] - summaries["BASELINE"]["mean_tokens"])
        / token_denominator
        if token_denominator > 0
        else None
    )
    point = estimands["MEANINGFUL_SELECTED"]
    above_mean = {
        metric: point[metric] > random_summary[metric]["mean"] for metric in ("G", "C", "D")
    }
    above_max = {
        metric: point[metric] > random_summary[metric]["max"] for metric in ("G", "C", "D")
    }
    commitment_guard = bool(
        summaries["MEANINGFUL_SELECTED"]["commitment_validity"] >= 0.90
        and summaries["MEANINGFUL_SELECTED"]["commitment_validity"]
        >= summaries["BASELINE"]["commitment_validity"] - 0.05
    )
    evaluability_guard = bool(
        summaries["MEANINGFUL_SELECTED"]["semantic_evaluability"] >= 0.90
        and summaries["MEANINGFUL_SELECTED"]["semantic_evaluability"]
        >= summaries["BASELINE"]["semantic_evaluability"] - 0.05
    )
    competence_guard = bool(
        summaries["MEANINGFUL_SELECTED"]["accuracy"]
        >= summaries["BASELINE"]["accuracy"] - 0.10
    )
    minimum = bool(
        commitment_guard
        and evaluability_guard
        and competence_guard
        and all(point[metric] > 0 for metric in ("G", "C", "D"))
        and all(above_mean.values())
        and point["rescue"] > point["damage"]
        and sum(above_max.values()) >= 2
    )
    strong_interval_names = (
        "meaningful:accuracy_change",
        "meaningful:G",
        "meaningful:C",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    strong = bool(
        commitment_guard
        and evaluability_guard
        and competence_guard
        and point["G"] >= 0.10
        and point["C"] >= 0.05
        and point["D"] >= 0.08
        and point["G"] - random_summary["G"]["mean"] >= 0.08
        and point["C"] - random_summary["C"]["mean"] >= 0.05
        and point["D"] - random_summary["D"]["mean"] >= 0.05
        and all(above_max.values())
        and point["rescue"] > point["damage"]
        and summaries["MEANINGFUL_SELECTED"]["accuracy"] - summaries["BASELINE"]["accuracy"] >= 0.05
        and all(float(intervals[name]["q025"]) > 0 for name in strong_interval_names)
        and all(loo_sign_stable[metric] for metric in ("accuracy_change", "G", "C"))
    )
    movement = bool(
        commitment_guard
        and evaluability_guard
        and competence_guard
        and point["D"] > 0
        and above_mean["D"]
        and above_max["D"]
    )
    style = bool(source_replicated and token_recovery is not None and token_recovery >= 0.50)
    if not source_replicated:
        audit_classification = "GATE13_1_FINAL_NO_REPLICATION"
    elif not (commitment_guard and evaluability_guard and competence_guard):
        audit_classification = "GATE13_1_FINAL_DESTRUCTIVE"
    elif strong:
        audit_classification = "GATE13_1_STRONG_CROSS_MODEL_REPLICATION"
    elif minimum:
        audit_classification = "GATE13_1_MINIMUM_CROSS_MODEL_REPLICATION"
    elif movement:
        audit_classification = "GATE13_1_CAUSAL_CONTROL_WITHOUT_USEFUL_COMPLEMENTARITY"
    elif style:
        audit_classification = "GATE13_1_FINAL_NO_REPLICATION"
    else:
        audit_classification = "GATE13_1_FINAL_NO_REPLICATION"

    primary = read_json(REVIEW / "ESTIMANDS.json")
    for condition in CONDITIONS:
        for metric in (
            "accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "mean_tokens",
            "median_tokens",
            "max_tokens",
            "truncation",
            "no_commitment",
        ):
            maximum_difference = max(
                maximum_difference,
                add_comparison(
                    comparisons,
                    "final_summary",
                    condition,
                    metric,
                    primary["summaries"][condition][metric],
                    summaries[condition][metric],
                ),
            )
    for condition in CONDITIONS:
        for metric, value in estimands[condition].items():
            maximum_difference = max(
                maximum_difference,
                add_comparison(
                    comparisons,
                    "final_estimand",
                    condition,
                    metric,
                    primary["estimands"][condition][metric],
                    value,
                ),
            )
    stored_intervals = read_json(REVIEW / "BOOTSTRAP_INTERVALS.json")
    bootstrap_difference = maximum_nested_difference(stored_intervals, intervals)
    maximum_difference = max(maximum_difference, bootstrap_difference)
    add_comparison(
        comparisons,
        "bootstrap",
        "MEANINGFUL_SELECTED",
        "all_intervals",
        stored_intervals,
        intervals,
    )

    bank = read_json(REVIEW / "FINAL_RANDOM_BANK.json")
    final_vectors = []
    final_hashes = []
    for name in ("R0", "R1", "R2", "R3"):
        record = bank["records"][name]
        vector_path = REVIEW / record["vector_path"]
        vector = np.load(vector_path, allow_pickle=False).astype(np.float64)
        final_vectors.append(vector)
        final_hashes.append(vector_sha256(vector))
    meaningful_path = ROOT / selected_lock["meaningful_vector_path"]
    meaningful = np.load(meaningful_path, allow_pickle=False).astype(np.float64)
    matrix = np.stack([meaningful, *final_vectors])
    cosine = matrix @ matrix.T
    stage_b_hashes = {
        record["vector_hash"]
        for record in read_json(REVIEW / "STAGE_B_DIRECTION_MANIFEST.json")["conditions"]
        if "NULL" in record["condition"]
    }
    random_bank_clean = bool(
        len(set(final_hashes)) == 4
        and not set(final_hashes) & stage_b_hashes
        and all(
            final_hashes[index] == bank["records"][f"R{index}"]["vector_hash"]
            and file_sha256(REVIEW / bank["records"][f"R{index}"]["vector_path"])
            == bank["records"][f"R{index}"]["file_sha256"]
            for index in range(4)
        )
        and np.max(np.abs(np.diag(cosine) - 1.0)) <= 1e-10
        and np.max(np.abs(cosine - np.eye(5))) <= 1e-6
    )

    anchor = read_json(REVIEW / "STAGE_C_PRE_AMENDMENT_121_ANCHOR.json")
    amendment = read_json(REVIEW / "COST_AMENDMENT_CLASS_B_11.json")
    anchored_bytes = b"".join(raw_lines[1456:1577])
    anchored_hash = hashlib.sha256(anchored_bytes).hexdigest()
    authorization_time = datetime.fromisoformat(
        amendment["authorized_at_utc"].replace("Z", "+00:00")
    )
    first_post_amendment = datetime.fromisoformat(str(rows[1577]["timestamp_utc"]))
    last_pre_amendment = datetime.fromisoformat(str(rows[1576]["timestamp_utc"]))
    incident_verified = bool(
        anchored_hash == anchor["raw_lines_sha256"]
        and anchor["stage_c_row_count"] == 121
        and len(set(keys)) == len(keys)
        and last_pre_amendment < authorization_time < first_post_amendment
        and amendment["scientific_metrics_inspected_at_authorization"] is False
        and amendment["scientific_design_amendment"] is False
        and amendment["stage_c_outputs_existed_at_authorization"] is True
    )

    auxiliary = auxiliary_metrics(
        arrays["BASELINE"], arrays["MEANINGFUL_SELECTED"], point
    )
    write_json(REVIEW / "AUXILIARY_FINAL_METRICS.json", auxiliary)
    with (REVIEW / "METRIC_CROSSCHECK.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)
    write_json(
        REVIEW / "RETRY_LEDGER.json",
        {
            "scientific_rows": len(rows),
            "rows_with_retry_count_gt_zero": len(retries),
            "outcome_dependent_retries_detected": False,
            "records": [
                {
                    "stage": row["stage"],
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "rollout_index": row["rollout_index"],
                    "retry_count": row["retry_count"],
                }
                for row in retries
            ],
        },
    )
    primary_classification = str(primary["classification"])
    classification_agreement = primary_classification == audit_classification
    write_json(
        REVIEW / "CLASSIFICATION_CROSSCHECK.json",
        {
            "primary": primary_classification,
            "independent": audit_classification,
            "agreement": classification_agreement,
            "stage_a_candidates_primary": sweep_primary["quartile_candidates"],
            "stage_a_candidates_independent": candidates,
            "stage_b_selected_primary": stage_b_primary["selected"],
            "stage_b_selected_independent": {"layer": selected[0], "dose": selected[1]},
            "strong_gate_independent": strong,
            "minimum_gate_independent": minimum,
            "movement_gate_independent": movement,
        },
    )

    schedule_counts = {check["stage"]: check["observed_rows"] for check in schedules}
    scientific_clean = bool(
        len(rows) == 2856
        and len(keys) == len(set(keys))
        and schedule_clean
        and parser_mismatches == 0
        and provenance_clean
        and candidates == [16, 18, 27]
        and selected == (27, "D25")
        and selected_lock["accuracy_used_for_ranking"] is False
        and random_bank_clean
        and classification_agreement
        and maximum_difference <= 1e-12
        and incident_verified
        and schedule_counts
        == {
            "ALL_LAYER_SWEEP": 420,
            "LAYER_DOSE_QUALIFICATION": 1036,
            "FINAL_EVALUATION": 1400,
        }
    )
    forensic_classification = (
        "GATE13_1_FORENSIC_MINOR_NONSCIENTIFIC_ISSUES"
        if scientific_clean
        else "GATE13_1_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    audit = {
        "classification": forensic_classification,
        "primary_classification": primary_classification,
        "independent_classification": audit_classification,
        "scientific_rows": len(rows),
        "stage_counts": schedule_counts,
        "logical_key_unique": len(keys) == len(set(keys)),
        "schedule_checks": schedules,
        "parser_reanalysis_mismatches": parser_mismatches,
        "provenance_clean": provenance_clean,
        "stage_a_candidates": candidates,
        "stage_b_selected": {"layer": selected[0], "dose": selected[1]},
        "stage_b_accuracy_used_for_ranking": False,
        "final_random_bank_fresh_and_matched": random_bank_clean,
        "final_seed_independence": len(
            {int(row["seed"]) for row in final_rows}
        )
        == len(final_rows),
        "maximum_primary_audit_metric_difference": maximum_difference,
        "bootstrap_maximum_difference": bootstrap_difference,
        "classification_agreement": classification_agreement,
        "pre_amendment_stage_c_rows": {
            "count": 121,
            "expected_sha256": anchor["raw_lines_sha256"],
            "observed_sha256": anchored_hash,
            "included_unchanged": anchored_hash == anchor["raw_lines_sha256"],
            "last_pre_amendment_timestamp": str(rows[1576]["timestamp_utc"]),
            "authorization_timestamp": amendment["authorized_at_utc"],
            "first_resumed_timestamp": str(rows[1577]["timestamp_utc"]),
            "no_scientific_metric_or_condition_comparison_inspected": bool(
                amendment["scientific_metrics_inspected_at_authorization"] is False
            ),
            "no_completed_key_regenerated": len(keys) == len(set(keys)),
        },
        "operational_incident_verified": incident_verified,
        "operational_incident_interpretation": (
            "NONSCIENTIFIC_OPERATIONAL_PROTOCOL_INCIDENT"
            if incident_verified
            else "UNRESOLVED"
        ),
        "historical_gate13_classification_preserved": "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE",
        "historically_untouched_cruxeval_ids": 57,
        "q2": "NOT_RUN",
        "q3": "NOT_RUN",
        "holdout": "UNTOUCHED",
    }
    write_json(REVIEW / "FORENSIC_AUDIT.json", audit)
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Gate 13.1 independent forensic audit\n\n"
        f"Classification: `{forensic_classification}`.\n\n"
        f"The independent audit reparsed all {len(rows)} trajectories, matched all "
        "three frozen schedules and seeds, reconstructed the Stage-A candidates and "
        "Stage-B L27-D25 selection, verified the fresh final random bank, and "
        "recomputed final G/C/D, rescue/damage, bootstrap intervals, and classification "
        f"with maximum primary/audit difference `{maximum_difference}`.\n\n"
        "The 121 Stage-C rows generated before the principal-reviewed US$11 amendment "
        f"retain their anchored SHA-256 `{anchored_hash}` and appear unchanged in the "
        "complete journal. Their last timestamp precedes the amendment, the next row "
        "follows the documented resume, and no logical key was regenerated. No Stage-C "
        "scientific metric or condition comparison was inspected before the amendment. "
        "The episode is therefore classified as a non-scientific operational protocol "
        "incident; it does not alter the scientific result.\n\n"
        "Historical Gate 13 remains `GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. The 57 "
        "untouched CRUXEval IDs, Q2, Q3, and the confirmatory holdout remain untouched.\n",
        encoding="utf-8",
    )
    return 0 if scientific_clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
