#!/usr/bin/env python3
"""Primary matched-coupling calibration analysis for Gate 8."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate8 import (  # noqa: E402
    BASELINE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DOSE_FRACTIONS,
    TEXTUAL,
    classify_source,
    dose_eligibility,
    select_dose,
)

REVIEW = ROOT / "review/gate8_l27_dose_calibration"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def semantic_outcome(row: dict[str, Any]) -> str:
    if bool(row.get("commitment_valid")) and bool(row.get("semantic_evaluable")):
        return "VALUE:" + json.dumps(row.get("canonical_value"), sort_keys=True, ensure_ascii=False)
    status = str(row.get("status", "INVALID_FORMAT"))
    reason = str(row.get("parse_reason", "")).lower()
    if status == "TRUNCATED":
        return "TRUNCATED"
    if status == "RUNTIME_ERROR":
        return "MODEL_RUNTIME_ERROR"
    if "ambiguous" in reason or "multiple" in reason or "conflict" in reason:
        return "AMBIGUOUS_COMMITMENT"
    if bool(row.get("commitment_valid")):
        return "UNEVALUABLE"
    return "NO_COMMITMENT"


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return index
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    tokens = np.asarray([int(row["generated_token_count"]) for row in selected], dtype=np.float64)
    statuses = Counter(str(row["status"]) for row in selected)
    return {
        "condition": condition,
        "n": len(selected),
        "commitment_validity": float(np.mean([bool(row["commitment_valid"]) for row in selected])),
        "semantic_evaluability": float(
            np.mean([bool(row["semantic_evaluable"]) for row in selected])
        ),
        "accuracy": float(np.mean([bool(row["correct"]) for row in selected])),
        "truncation_rate": float(np.mean([row["status"] == "TRUNCATED" for row in selected])),
        "no_commitment_rate": float(
            np.mean([semantic_outcome(row) == "NO_COMMITMENT" for row in selected])
        ),
        "mean_tokens": float(tokens.mean()),
        "median_tokens": float(np.median(tokens)),
        "max_tokens": int(tokens.max()),
        "status_counts": dict(sorted(statuses.items())),
    }


def paired_metrics(
    by_key: dict[tuple[str, str, int], dict[str, Any]],
    item_ids: list[str],
    condition: str,
) -> dict[str, float]:
    semantic_changes: list[float] = []
    raw_changes: list[float] = []
    divergences: list[int] = []
    rescues: list[float] = []
    damages: list[float] = []
    for item_id in item_ids:
        for rollout in (0, 1):
            baseline = by_key[(item_id, BASELINE, rollout)]
            current = by_key[(item_id, condition, rollout)]
            semantic_changes.append(float(semantic_outcome(baseline) != semantic_outcome(current)))
            left = list(baseline.get("generated_token_ids", []))
            right = list(current.get("generated_token_ids", []))
            raw_changes.append(float(left != right))
            divergence = first_divergence(left, right)
            if divergence is not None:
                divergences.append(divergence)
            e0 = int(not bool(baseline["correct"]))
            ej = int(not bool(current["correct"]))
            rescues.append(float(e0 == 1 and ej == 0))
            damages.append(float(e0 == 0 and ej == 1))
    return {
        "Q": float(np.mean(semantic_changes)),
        "raw_token_sequence_change": float(np.mean(raw_changes)),
        "first_divergence_mean": float(np.mean(divergences)) if divergences else float("nan"),
        "first_divergence_median": float(np.median(divergences)) if divergences else float("nan"),
        "matched_rescue": float(np.mean(rescues)),
        "matched_damage": float(np.mean(damages)),
    }


def point_estimates(rows: list[dict[str, Any]], item_ids: list[str]) -> dict[str, Any]:
    by_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row for row in rows
    }
    summaries = {
        condition: condition_summary(rows, condition)
        for condition in sorted({str(row["condition"]) for row in rows})
    }
    baseline = summaries[BASELINE]
    textual = summaries[TEXTUAL]
    source_classification = classify_source(baseline, textual)
    mean_denominator = textual["mean_tokens"] - baseline["mean_tokens"]
    median_denominator = textual["median_tokens"] - baseline["median_tokens"]
    doses: dict[str, Any] = {}
    random_details: dict[str, Any] = {}
    for dose in DOSE_FRACTIONS:
        meaningful_condition = f"MEAN_{dose}"
        meaningful = {
            **summaries[meaningful_condition],
            **paired_metrics(by_key, item_ids, meaningful_condition),
        }
        meaningful["rho_tokens"] = (
            (meaningful["mean_tokens"] - baseline["mean_tokens"]) / mean_denominator
            if mean_denominator > 0
            else float("nan")
        )
        meaningful["rho_tokens_median"] = (
            (meaningful["median_tokens"] - baseline["median_tokens"]) / median_denominator
            if median_denominator > 0
            else float("nan")
        )
        random_rows: dict[str, Any] = {}
        for index in range(4):
            condition = f"RANDOM_R{index}_{dose}"
            random_rows[condition] = {
                **summaries[condition],
                **paired_metrics(by_key, item_ids, condition),
            }
        q_values = np.asarray([value["Q"] for value in random_rows.values()], dtype=np.float64)
        random_q = {
            "mean": float(q_values.mean()),
            "median": float(np.median(q_values)),
            "min": float(q_values.min()),
            "max": float(q_values.max()),
        }
        meaningful["specificity_mean"] = meaningful["Q"] - random_q["mean"]
        meaningful["specificity_max"] = meaningful["Q"] - random_q["max"]
        gates = dose_eligibility(
            baseline=baseline,
            dose=meaningful,
            random_q=random_q,
            source_replicated=source_classification == "CAREFUL_SOURCE_REPLICATED",
        )
        doses[dose] = {**meaningful, "random_Q": random_q, "gates": gates}
        random_details[dose] = random_rows
    eligibility = {dose: value["gates"] for dose, value in doses.items()}
    if source_classification != "CAREFUL_SOURCE_REPLICATED":
        selected = None
        classification = "GATE8_SOURCE_POLICY_NOT_REPLICATED"
    else:
        selected, classification = select_dose(eligibility)
    return {
        "summaries": summaries,
        "source_classification": source_classification,
        "doses": doses,
        "random_details": random_details,
        "selected_dose": selected,
        "classification": classification,
    }


def bootstrap(rows: list[dict[str, Any]], item_ids: list[str]) -> dict[str, Any]:
    by_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row for row in rows
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for dose in DOSE_FRACTIONS:
        condition = f"MEAN_{dose}"
        dose_arrays: dict[str, list[float]] = {
            "commitment_difference": [],
            "evaluability_difference": [],
            "accuracy_difference": [],
            "Q": [],
            "baseline_tokens": [],
            "textual_tokens": [],
            "dose_tokens": [],
            "rescue": [],
            "damage": [],
        }
        random_q: list[list[float]] = [[] for _ in range(4)]
        for item_id in item_ids:
            base_rows = [by_key[(item_id, BASELINE, rollout)] for rollout in (0, 1)]
            text_rows = [by_key[(item_id, TEXTUAL, rollout)] for rollout in (0, 1)]
            dose_rows = [by_key[(item_id, condition, rollout)] for rollout in (0, 1)]
            dose_arrays["commitment_difference"].append(
                float(np.mean([row["commitment_valid"] for row in dose_rows]))
                - float(np.mean([row["commitment_valid"] for row in base_rows]))
            )
            dose_arrays["evaluability_difference"].append(
                float(np.mean([row["semantic_evaluable"] for row in dose_rows]))
                - float(np.mean([row["semantic_evaluable"] for row in base_rows]))
            )
            dose_arrays["accuracy_difference"].append(
                float(np.mean([row["correct"] for row in dose_rows]))
                - float(np.mean([row["correct"] for row in base_rows]))
            )
            dose_arrays["Q"].append(
                float(
                    np.mean(
                        [
                            semantic_outcome(base) != semantic_outcome(current)
                            for base, current in zip(base_rows, dose_rows, strict=True)
                        ]
                    )
                )
            )
            dose_arrays["baseline_tokens"].append(
                float(np.mean([row["generated_token_count"] for row in base_rows]))
            )
            dose_arrays["textual_tokens"].append(
                float(np.mean([row["generated_token_count"] for row in text_rows]))
            )
            dose_arrays["dose_tokens"].append(
                float(np.mean([row["generated_token_count"] for row in dose_rows]))
            )
            base_errors = [int(not row["correct"]) for row in base_rows]
            dose_errors = [int(not row["correct"]) for row in dose_rows]
            dose_arrays["rescue"].append(
                float(
                    np.mean(
                        [
                            base_error == 1 and dose_error == 0
                            for base_error, dose_error in zip(base_errors, dose_errors, strict=True)
                        ]
                    )
                )
            )
            dose_arrays["damage"].append(
                float(
                    np.mean(
                        [
                            base_error == 0 and dose_error == 1
                            for base_error, dose_error in zip(base_errors, dose_errors, strict=True)
                        ]
                    )
                )
            )
            for index in range(4):
                random_rows = [
                    by_key[(item_id, f"RANDOM_R{index}_{dose}", rollout)] for rollout in (0, 1)
                ]
                random_q[index].append(
                    float(
                        np.mean(
                            [
                                semantic_outcome(base) != semantic_outcome(random)
                                for base, random in zip(base_rows, random_rows, strict=True)
                            ]
                        )
                    )
                )
        arrays[dose] = {
            **{name: np.asarray(values, dtype=np.float64) for name, values in dose_arrays.items()},
            **{
                f"random_Q_{index}": np.asarray(values, dtype=np.float64)
                for index, values in enumerate(random_q)
            },
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics: dict[str, list[float]] = {}
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(item_ids), size=len(item_ids))
        for dose, value in arrays.items():
            random_mean = float(
                np.mean([value[f"random_Q_{index}"][indices].mean() for index in range(4)])
            )
            baseline_tokens = float(value["baseline_tokens"][indices].mean())
            denominator = float(value["textual_tokens"][indices].mean()) - baseline_tokens
            rho_tokens = (
                (float(value["dose_tokens"][indices].mean()) - baseline_tokens) / denominator
                if denominator > 0
                else float("nan")
            )
            for name, observed in (
                (
                    "commitment_validity_difference",
                    value["commitment_difference"][indices].mean(),
                ),
                (
                    "semantic_evaluability_difference",
                    value["evaluability_difference"][indices].mean(),
                ),
                ("accuracy_difference", value["accuracy_difference"][indices].mean()),
                ("Q", value["Q"][indices].mean()),
                ("Q_minus_random_mean", value["Q"][indices].mean() - random_mean),
                ("rho_tokens", rho_tokens),
                ("matched_rescue", value["rescue"][indices].mean()),
                ("matched_damage", value["damage"][indices].mean()),
            ):
                metrics.setdefault(f"{dose}:{name}", []).append(float(observed))
    return {
        name: {
            "estimate": float(np.mean(values)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "unit": "item_cluster_all_22_conditions_both_rollout_blocks",
        }
        for name, values in sorted(metrics.items())
    }


def monotonicity(point: dict[str, Any]) -> dict[str, Any]:
    order = list(DOSE_FRACTIONS)
    fields = {
        "commitment_validity": [point["doses"][dose]["commitment_validity"] for dose in order],
        "semantic_evaluability": [point["doses"][dose]["semantic_evaluability"] for dose in order],
        "accuracy": [point["doses"][dose]["accuracy"] for dose in order],
        "Q": [point["doses"][dose]["Q"] for dose in order],
        "rho_tokens": [point["doses"][dose]["rho_tokens"] for dose in order],
        "truncation_rate": [point["doses"][dose]["truncation_rate"] for dose in order],
        "no_commitment_rate": [point["doses"][dose]["no_commitment_rate"] for dose in order],
    }
    directions = {
        "commitment_validity": "nonincreasing",
        "semantic_evaluability": "nonincreasing",
        "accuracy": "reported_not_assumed",
        "Q": "nondecreasing",
        "rho_tokens": "nondecreasing",
        "truncation_rate": "nondecreasing",
        "no_commitment_rate": "nondecreasing",
    }
    violations: list[dict[str, Any]] = []
    for field, values in fields.items():
        direction = directions[field]
        if direction == "reported_not_assumed":
            continue
        for index in range(len(values) - 1):
            violates = (direction == "nondecreasing" and values[index + 1] < values[index]) or (
                direction == "nonincreasing" and values[index + 1] > values[index]
            )
            if violates:
                violations.append(
                    {
                        "metric": field,
                        "from": order[index],
                        "to": order[index + 1],
                        "from_value": values[index],
                        "to_value": values[index + 1],
                    }
                )
    return {
        "dose_order": order,
        "values": fields,
        "expected_directions": directions,
        "violations": violations,
    }


def analyze(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    rows = read_jsonl(review / "journal.jsonl")
    if len(rows) != 2200:
        raise RuntimeError(f"Gate 8 requires 2,200 rows; found {len(rows)}")
    keys = [(row["item_id"], row["condition"], row["rollout_index"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 8 primary analysis found duplicate logical keys")
    item_ids = sorted({str(row["item_id"]) for row in rows})
    point = point_estimates(rows, item_ids)
    intervals = bootstrap(rows, item_ids)
    monotonic = monotonicity(point)

    summary_rows = list(point["summaries"].values())
    write_csv(review / "CONDITION_SUMMARY.csv", summary_rows)
    dose_rows = []
    for dose, value in point["doses"].items():
        dose_rows.append(
            {
                "dose": dose,
                "eta": lock["dose_grid"][dose]["eta"],
                "commitment_validity": value["commitment_validity"],
                "semantic_evaluability": value["semantic_evaluability"],
                "accuracy": value["accuracy"],
                "Q": value["Q"],
                "random_Q_mean": value["random_Q"]["mean"],
                "random_Q_max": value["random_Q"]["max"],
                "specificity_mean": value["specificity_mean"],
                "specificity_max": value["specificity_max"],
                "rho_tokens": value["rho_tokens"],
                "rho_tokens_median": value["rho_tokens_median"],
                "matched_rescue": value["matched_rescue"],
                "matched_damage": value["matched_damage"],
                "eligible": value["gates"]["eligible"],
            }
        )
    write_csv(review / "DOSE_SUMMARY.csv", dose_rows)
    selection = {
        "classification": point["classification"],
        "source_classification": point["source_classification"],
        "selected_dose": point["selected_dose"],
        "selection_rule": "lowest eligible lower dose: D25, then D50, then D75",
        "D100_selectable": False,
        "eligibility": {dose: value["gates"] for dose, value in point["doses"].items()},
        "accuracy_G_C_D_used_for_selection": False,
        "future_evaluation_executed": False,
    }
    write_json(review / "DOSE_SELECTION.json", selection)
    write_json(review / "BOOTSTRAP_INTERVALS.json", intervals)
    write_json(review / "MONOTONICITY_REPORT.json", monotonic)

    baseline = point["summaries"][BASELINE]
    textual = point["summaries"][TEXTUAL]
    dose_lines = "\n".join(
        f"| {row['dose']} | {row['commitment_validity']:.4f} | "
        f"{row['semantic_evaluability']:.4f} | {row['accuracy']:.4f} | "
        f"{row['Q']:.4f} | {row['random_Q_mean']:.4f}/{row['random_Q_max']:.4f} | "
        f"{row['rho_tokens']:.4f} | {row['eligible']} |"
        for row in dose_rows
    )
    report = f"""# Gate 8 — L27 dose calibration

Primary classification: `{point["classification"]}`.

Selected lower dose: `{point["selected_dose"] or "NONE"}`.

This is calibration only. G/C/D were not computed or used as primary evidence,
and no later dose evaluation was executed.

## Source reference

- Classification: `{point["source_classification"]}`
- Baseline validity/evaluability/accuracy: {baseline["commitment_validity"]:.4f} /
  {baseline["semantic_evaluability"]:.4f} / {baseline["accuracy"]:.4f}
- Baseline mean/median tokens: {baseline["mean_tokens"]:.2f} / {baseline["median_tokens"]:.1f}
- CAREFUL validity/evaluability/accuracy: {textual["commitment_validity"]:.4f} /
  {textual["semantic_evaluability"]:.4f} / {textual["accuracy"]:.4f}
- CAREFUL mean/median tokens: {textual["mean_tokens"]:.2f} / {textual["median_tokens"]:.1f}

## Dose curve

| dose | commitment | evaluability | accuracy | Q | random mean/max Q | rho tokens | eligible |
|---|---:|---:|---:|---:|---:|---:|---:|
{dose_lines}

## Selection

The frozen rule selects the lowest eligible lower dose in the order D25, D50,
D75. D100 is diagnostic only. Accuracy was a safety guard only; G/C/D were not
computed. Result: `{point["classification"]}` with selected dose
`{point["selected_dose"] or "NONE"}`.

## Monotonicity

Observed violations of the descriptive expected directions: {len(monotonic["violations"])}.
They are preserved in `MONOTONICITY_REPORT.json` and do not alter selection.

## Interpretation boundary

Gate 8 identifies whether the frozen L27 actuator has a prospectively safe,
specific lower operating point. Calibration items are permanently consumed and
cannot serve as future evaluation evidence. Q2, character count, Gate 9, and
the confirmatory holdout were not run.
"""
    (review / "REPORT.md").write_text(report, encoding="utf-8")
    return {"classification": point["classification"], "selected_dose": point["selected_dose"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = analyze(args.review_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
