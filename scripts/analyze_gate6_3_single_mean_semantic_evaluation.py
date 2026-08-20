#!/usr/bin/env python3
"""Offline analysis for the conditional Gate 6.3 random null and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.benchmarks.external.base import ExternalStatus, evaluate_external_answer
from epistemic_geometry.benchmarks.external.semantic_v2 import (
    PARSER_VERSION,
    parse_external_answer_v2,
)
from epistemic_geometry.experiments.gate6 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    two_rollout_estimands,
)

ROOT = Path(__file__).resolve().parents[1]
RANDOM_CONDITIONS = tuple(f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
EVALUATION_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "BEST_SINGLE_MEAN_PLUS",
    *RANDOM_CONDITIONS,
)
VALID = {ExternalStatus.VALID_CORRECT.value, ExternalStatus.VALID_WRONG.value}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _v2_row(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("status")) == "RUNTIME_ERROR":
        return {
            "status": "RUNTIME_ERROR",
            "parsed_answer": None,
            "correct": False,
            "parse_reason": row.get("error"),
        }
    token_count = int(row.get("generated_token_count", 0))
    parsed = parse_external_answer_v2(
        str(row.get("raw_output", "")),
        truncated=str(row.get("status")) == "TRUNCATED" or token_count >= 4096,
    )
    if parsed.status is not None:
        return {
            "status": parsed.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
            "parsed_answer": parsed.answer_text,
            "correct": False,
            "parse_reason": parsed.parse_reason,
        }
    try:
        correct = evaluate_external_answer(
            parsed.answer_text or "", str(row["reference_answer"]), str(row["evaluator"])
        )
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
        return {
            "status": "INVALID_FORMAT",
            "parsed_answer": parsed.answer_text,
            "correct": False,
            "parse_reason": f"typed evaluator rejected answer: {type(exc).__name__}",
        }
    return {
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "parsed_answer": parsed.answer_text,
        "correct": bool(correct),
        "parse_reason": None,
    }


def _outcome(row: dict[str, Any], v2: dict[str, Any]) -> str:
    if v2["status"] in VALID:
        return f"VALID::{v2['parsed_answer']}"
    return f"MECHANICAL::{v2['status']}"


def _sequence_change(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return list(left.get("generated_token_ids", [])) != list(right.get("generated_token_ids", []))


def _first_divergence(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_ids = list(left.get("generated_token_ids", []))
    right_ids = list(right.get("generated_token_ids", []))
    for index, (a, b) in enumerate(zip(left_ids, right_ids, strict=False)):
        if a != b:
            return index
    return min(len(left_ids), len(right_ids)) if len(left_ids) != len(right_ids) else None


def _summary(
    condition: str,
    rows: list[dict[str, Any]],
    v2: dict[tuple[str, str, int], dict[str, Any]],
    baseline: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    statuses = [
        v2[(str(row["item_id"]), condition, int(row["rollout_index"]))]["status"] for row in rows
    ]
    token_counts = [int(row.get("generated_token_count", 0)) for row in rows]
    return {
        "condition": condition,
        "n": len(rows),
        "valid": sum(status in VALID for status in statuses),
        "validity": sum(status in VALID for status in statuses) / len(rows),
        "correct": sum(status == "VALID_CORRECT" for status in statuses),
        "wrong": sum(status == "VALID_WRONG" for status in statuses),
        "invalid_format": sum(status == "INVALID_FORMAT" for status in statuses),
        "truncated": sum(status == "TRUNCATED" for status in statuses),
        "runtime_error": sum(status == "RUNTIME_ERROR" for status in statuses),
        "accuracy": sum(status == "VALID_CORRECT" for status in statuses) / len(rows),
        "mean_tokens": statistics.mean(token_counts),
        "median_tokens": statistics.median(token_counts),
        "max_tokens": max(token_counts),
        "status_counts": dict(sorted(Counter(statuses).items())),
    }


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze_matched(review: Path) -> dict[str, Any]:
    old_rows = _read_jsonl(Path("review/gate6_2_first_stage_repair_mean_bridge/journal.jsonl"))
    new_rows = _read_jsonl(review / "journal.jsonl")
    random_rows = [
        row for row in new_rows if str(row.get("phase")) == "GATE6_3_MATCHED_RANDOM_SUPPLEMENT"
    ]
    if len(random_rows) != 80:
        raise RuntimeError(f"expected 80 matched-random rows, found {len(random_rows)}")
    keys = [
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        for row in random_rows
    ]
    if len(set(keys)) != 80:
        raise RuntimeError("duplicate matched-random logical row")
    old_v2 = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): _v2_row(row)
        for row in old_rows
    }
    baseline = {
        str(row["item_id"]): (row, old_v2[(str(row["item_id"]), "BASELINE", 0)])
        for row in old_rows
        if str(row["condition"]) == "BASELINE"
    }
    parsed_new = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): _v2_row(row)
        for row in random_rows
    }
    summary_rows = []
    for condition in RANDOM_CONDITIONS:
        selected = [row for row in random_rows if str(row["condition"]) == condition]
        statuses = [parsed_new[(str(row["item_id"]), condition, 0)]["status"] for row in selected]
        semantic_change = [
            _outcome(baseline[str(row["item_id"])][0], baseline[str(row["item_id"])][1])
            != _outcome(row, parsed_new[(str(row["item_id"]), condition, 0)])
            for row in selected
        ]
        raw_change = [_sequence_change(baseline[str(row["item_id"])][0], row) for row in selected]
        divergences = [_first_divergence(baseline[str(row["item_id"])][0], row) for row in selected]
        defined = [value for value in divergences if value is not None]
        summary_rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "valid": sum(status in VALID for status in statuses),
                "validity": sum(status in VALID for status in statuses) / len(selected),
                "correct": sum(status == "VALID_CORRECT" for status in statuses),
                "wrong": sum(status == "VALID_WRONG" for status in statuses),
                "invalid_format": sum(status == "INVALID_FORMAT" for status in statuses),
                "truncated": sum(status == "TRUNCATED" for status in statuses),
                "runtime_error": sum(status == "RUNTIME_ERROR" for status in statuses),
                "accuracy": sum(status == "VALID_CORRECT" for status in statuses) / len(selected),
                "semantic_change_rate": sum(semantic_change) / len(selected),
                "raw_token_sequence_change_rate": sum(raw_change) / len(selected),
                "mean_first_divergence_token": statistics.mean(defined) if defined else None,
                "mean_tokens": statistics.mean(
                    int(row.get("generated_token_count", 0)) for row in selected
                ),
                "median_tokens": statistics.median(
                    int(row.get("generated_token_count", 0)) for row in selected
                ),
                "max_tokens": max(int(row.get("generated_token_count", 0)) for row in selected),
                "status_counts": dict(sorted(Counter(statuses).items())),
            }
        )
    _csv(
        review / "MATCHED_RANDOM_RESULTS.csv",
        summary_rows,
        [
            "condition",
            "n",
            "valid",
            "validity",
            "correct",
            "wrong",
            "invalid_format",
            "truncated",
            "runtime_error",
            "accuracy",
            "semantic_change_rate",
            "raw_token_sequence_change_rate",
            "mean_first_divergence_token",
            "mean_tokens",
            "median_tokens",
            "max_tokens",
            "status_counts",
        ],
    )
    old_summary_rows = list(
        csv.DictReader((review / "SEMANTIC_V2_CONDITION_SUMMARY.csv").open(encoding="utf-8"))
    )
    old_summary = {row["condition"]: row for row in old_summary_rows}
    single_q = float(old_summary["BEST_SINGLE_MEAN_PLUS"]["semantic_change_rate"])
    random_q = [float(row["semantic_change_rate"]) for row in summary_rows]
    careful_mean_tokens = float(old_summary["TEXTUAL_CAREFUL_REFERENCE"]["mean_tokens"])
    direct_mean_tokens = float(old_summary["TEXTUAL_DIRECT_REFERENCE"]["mean_tokens"])
    single_mean_tokens = float(old_summary["BEST_SINGLE_MEAN_PLUS"]["mean_tokens"])
    token_toward_careful = abs(single_mean_tokens - careful_mean_tokens) < abs(
        single_mean_tokens - direct_mean_tokens
    )
    random_mean = float(np.mean(random_q))
    random_max = float(np.max(random_q))
    gate = {
        "validity_pass": float(old_summary["BEST_SINGLE_MEAN_PLUS"]["validity"]) >= 0.85,
        "semantic_change_pass": single_q >= 0.15,
        "single_minus_random_mean_pass": single_q - random_mean >= 0.05,
        "single_gt_random_max_pass": single_q > random_max,
        "token_count_toward_textual_careful_pass": token_toward_careful,
        "single_q": single_q,
        "random_mean_q": random_mean,
        "random_max_q": random_max,
    }
    gate["pass"] = all(
        gate[key]
        for key in (
            "validity_pass",
            "semantic_change_pass",
            "single_minus_random_mean_pass",
            "single_gt_random_max_pass",
            "token_count_toward_textual_careful_pass",
        )
    )
    result = {
        "parser_version": PARSER_VERSION,
        "single_controller": {
            "condition": "BEST_SINGLE_MEAN_PLUS",
            "q": single_q,
            "mean_tokens": single_mean_tokens,
        },
        "random_conditions": summary_rows,
        "random_q_mean": random_mean,
        "random_q_median": float(np.median(random_q)),
        "random_q_min": float(np.min(random_q)),
        "random_q_max": random_max,
        "token_count_reference": {
            "careful_mean": careful_mean_tokens,
            "direct_mean": direct_mean_tokens,
            "single_mean": single_mean_tokens,
            "single_toward_careful": token_toward_careful,
        },
        "gate": gate,
        "classification": "GATE6_3_MATCHED_NULL_PASS"
        if gate["pass"]
        else "GATE6_3_SINGLE_MEAN_NOT_SPECIFIC_TO_MATCHED_RANDOM",
        "evaluation_authorized": gate["pass"],
    }
    (review / "MATCHED_RANDOM_ESTIMANDS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _bootstrap(
    baseline: np.ndarray,
    conditions: dict[str, np.ndarray],
    valid_baseline: np.ndarray,
    valid_conditions: dict[str, np.ndarray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, float | int | None]]:
    rng = np.random.default_rng(seed)
    names = tuple(conditions)
    samples: dict[str, list[float]] = {
        f"{name}:{metric}": []
        for name in names
        for metric in ("accuracy_change", "validity_change", "G", "C", "D", "rescue", "damage")
    }
    samples.update(
        {
            f"{name}:{metric}_minus_random_mean": []
            for name in names
            if not name.startswith("SINGLE_L27_RANDOM_")
            for metric in ("G", "C", "D")
        }
    )
    random_names = tuple(name for name in names if name.startswith("SINGLE_L27_RANDOM_"))
    for _ in range(resamples):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        sampled_base = baseline[indices]
        base_accuracy = 1.0 - float(sampled_base.mean())
        base_validity = float(valid_baseline[indices].mean())
        point: dict[str, dict[str, float]] = {}
        for name in names:
            result = two_rollout_estimands(sampled_base, conditions[name][indices])
            result["validity"] = float(valid_conditions[name][indices].mean())
            point[name] = result
            samples[f"{name}:accuracy_change"].append(result["accuracy_condition"] - base_accuracy)
            samples[f"{name}:validity_change"].append(result["validity"] - base_validity)
            for metric in ("G", "C", "D", "rescue", "damage"):
                samples[f"{name}:{metric}"].append(result[metric])
        for name in tuple(n for n in names if not n.startswith("SINGLE_L27_RANDOM_")):
            for metric in ("G", "C", "D"):
                samples[f"{name}:{metric}_minus_random_mean"].append(
                    point[name][metric]
                    - float(np.mean([point[random][metric] for random in random_names]))
                )
    return {
        key: {
            "estimate": float(np.quantile(values, 0.5)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": resamples,
        }
        for key, values in samples.items()
    }


def analyze_evaluation(review: Path) -> dict[str, Any]:
    rows = _read_jsonl(review / "journal.jsonl")
    eval_rows = [row for row in rows if str(row.get("phase")) == "GATE6_3_PRIMARY_EVALUATION"]
    if len(eval_rows) != 840:
        raise RuntimeError(f"expected 840 evaluation rows, found {len(eval_rows)}")
    keys = [
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in eval_rows
    ]
    if len(set(keys)) != 840:
        raise RuntimeError("duplicate evaluation logical row")
    v2 = {key: _v2_row(row) for key, row in zip(keys, eval_rows, strict=True)}
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_rows:
        by_condition[str(row["condition"])].append(row)
    if {len(value) for value in by_condition.values()} != {120}:
        raise RuntimeError("each evaluation condition must have 120 rows")
    item_ids = sorted({str(row["item_id"]) for row in eval_rows})
    if len(item_ids) != 60:
        raise RuntimeError("evaluation must contain 60 item IDs")
    baseline = np.asarray(
        [
            [
                int(v2[(item, "BASELINE", rollout)]["status"] != "VALID_CORRECT")
                for rollout in (0, 1)
            ]
            for item in item_ids
        ],
        dtype=np.int8,
    )
    valid_base = np.asarray(
        [
            [int(v2[(item, "BASELINE", rollout)]["status"] in VALID) for rollout in (0, 1)]
            for item in item_ids
        ],
        dtype=np.int8,
    )
    conditions: dict[str, np.ndarray] = {}
    valid_conditions: dict[str, np.ndarray] = {}
    for condition in EVALUATION_CONDITIONS:
        conditions[condition] = np.asarray(
            [
                [
                    int(v2[(item, condition, rollout)]["status"] != "VALID_CORRECT")
                    for rollout in (0, 1)
                ]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        valid_conditions[condition] = np.asarray(
            [
                [int(v2[(item, condition, rollout)]["status"] in VALID) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
    summaries = {}
    for condition in EVALUATION_CONDITIONS:
        summaries[condition] = _summary(condition, by_condition[condition], v2, {})
    estimands = {
        condition: two_rollout_estimands(baseline, conditions[condition])
        for condition in EVALUATION_CONDITIONS
        if condition != "BASELINE"
    }
    for condition in estimands:
        estimands[condition]["validity"] = float(valid_conditions[condition].mean())
        estimands[condition]["validity_change"] = estimands[condition]["validity"] - float(
            valid_base.mean()
        )
    estimands["BASELINE"] = {
        "B00": float(np.mean(baseline[:, 0] * baseline[:, 1])),
        "O00": float(1 - np.mean(baseline[:, 0] * baseline[:, 1])),
        "validity": float(valid_base.mean()),
        "accuracy": float(1 - baseline.mean()),
    }
    random_metrics = {
        metric: {
            "mean": float(np.mean([estimands[name][metric] for name in RANDOM_CONDITIONS])),
            "median": float(np.median([estimands[name][metric] for name in RANDOM_CONDITIONS])),
            "min": float(np.min([estimands[name][metric] for name in RANDOM_CONDITIONS])),
            "max": float(np.max([estimands[name][metric] for name in RANDOM_CONDITIONS])),
        }
        for metric in ("G", "C", "D", "accuracy_condition", "validity", "validity_change")
    }
    single = estimands["BEST_SINGLE_MEAN_PLUS"]
    single["G_minus_random_mean"] = single["G"] - random_metrics["G"]["mean"]
    single["C_minus_random_mean"] = single["C"] - random_metrics["C"]["mean"]
    single["D_minus_random_mean"] = single["D"] - random_metrics["D"]["mean"]
    single["G_gt_random_max"] = single["G"] > random_metrics["G"]["max"]
    single["C_gt_random_max"] = single["C"] > random_metrics["C"]["max"]
    single["D_gt_random_max"] = single["D"] > random_metrics["D"]["max"]
    baseline_summary = summaries["BASELINE"]
    validity_guard = (
        summaries["BEST_SINGLE_MEAN_PLUS"]["validity"] >= 0.90
        and summaries["BEST_SINGLE_MEAN_PLUS"]["validity"] >= baseline_summary["validity"] - 0.05
    )
    competence_guard = (
        summaries["BEST_SINGLE_MEAN_PLUS"]["accuracy"] >= baseline_summary["accuracy"] - 0.10
    )
    movement = bool(
        validity_guard
        and competence_guard
        and single["D"] >= 0.05
        and single["D_minus_random_mean"] >= 0.05
        and single["D_gt_random_max"]
    )
    useful = bool(
        movement
        and single["G"] >= 0.03
        and single["C"] >= 0.03
        and single["G_minus_random_mean"] >= 0.05
        and single["C_minus_random_mean"] >= 0.05
        and single["G_gt_random_max"]
        and single["C_gt_random_max"]
    )
    if useful:
        classification = "GATE6_3_SINGLE_MEAN_USEFUL_COMPLEMENTARITY_SIGNAL"
    elif movement:
        classification = "GATE6_3_SINGLE_MEAN_ERROR_PROFILE_MOVEMENT_ONLY"
    elif not validity_guard or not competence_guard:
        classification = "GATE6_3_SINGLE_MEAN_DESTRUCTIVE"
    else:
        classification = "GATE6_3_STYLE_CONTROL_WITHOUT_ERROR_CONTROL"
    bootstrap = _bootstrap(
        baseline,
        {name: conditions[name] for name in EVALUATION_CONDITIONS if name != "BASELINE"},
        valid_base,
        {name: valid_conditions[name] for name in EVALUATION_CONDITIONS if name != "BASELINE"},
    )
    result = {
        "parser_version": PARSER_VERSION,
        "summaries": summaries,
        "estimands": estimands,
        "random_metrics": random_metrics,
        "validity_guard": validity_guard,
        "competence_guard": competence_guard,
        "movement_signal": movement,
        "useful_complementarity_signal": useful,
        "classification": classification,
        "bootstrap": bootstrap,
    }
    rows_out = []
    for row in eval_rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        scored = v2[key]
        rows_out.append(
            {
                **row,
                "v2_status": scored["status"],
                "v2_correct": scored["correct"],
                "v2_parsed_answer": scored["parsed_answer"],
                "v2_parse_reason": scored["parse_reason"],
            }
        )
    fields = list(rows_out[0])
    _csv(review / "EVALUATION_RESULTS.csv", rows_out, fields)
    _csv(
        review / "CONDITION_SUMMARY.csv",
        [
            {**value, "status_counts": json.dumps(value["status_counts"], sort_keys=True)}
            for value in summaries.values()
        ],
        [
            "condition",
            "n",
            "valid",
            "validity",
            "correct",
            "wrong",
            "invalid_format",
            "truncated",
            "runtime_error",
            "accuracy",
            "mean_tokens",
            "median_tokens",
            "max_tokens",
            "status_counts",
        ],
    )
    (review / "ESTIMANDS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (review / "BOOTSTRAP_INTERVALS.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("MATCHED_RANDOM", "EVALUATION"), required=True)
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=ROOT / "review" / "gate6_3_single_mean_semantic_evaluation",
    )
    args = parser.parse_args()
    review = args.review_dir.resolve()
    if args.phase == "MATCHED_RANDOM":
        result = analyze_matched(review)
    else:
        result = analyze_evaluation(review)
    print(
        json.dumps(
            {
                "classification": result["classification"],
                "evaluation_authorized": result.get("evaluation_authorized"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
