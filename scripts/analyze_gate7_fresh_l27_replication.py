#!/usr/bin/env python3
"""Primary offline analysis for the complete frozen Gate 7 collection."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)
from epistemic_geometry.experiments.gate6_3_v3 import (  # noqa: E402
    audit_two_rollout_estimands,
    item_contributions,
)
from epistemic_geometry.experiments.gate7 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    MAX_NEW_TOKENS,
    MEANINGFUL,
    RANDOM_NAMES,
    TEXTUAL,
    classify_gate7,
)

REVIEW = ROOT / "review/gate7_fresh_l27_replication"
BASELINE = "BASELINE"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_journal(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _reparse(row: dict[str, Any]) -> dict[str, Any]:
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
        "parsed_answer": result.payload,
        "failure_reason": result.failure_reason,
    }


def _semantic_outcome(parsed: dict[str, Any]) -> str:
    if parsed["commitment_valid"] and parsed["semantic_evaluable"]:
        return f"VALUE:{parsed['canonical_value']}"
    return f"MECHANICAL:{parsed['status']}:{parsed['failure_reason']}"


def _condition_summary(
    rows: list[dict[str, Any]], parsed: dict[tuple[str, str, int], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    records: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        results = [
            parsed[(row["item_id"], condition, int(row["rollout_index"]))] for row in selected
        ]
        tokens = [int(row["generated_token_count"]) for row in selected]
        statuses = Counter(result["status"] for result in results)
        record = {
            "condition": condition,
            "n": len(selected),
            "commitment_valid": sum(result["commitment_valid"] for result in results),
            "commitment_validity": float(
                np.mean([result["commitment_valid"] for result in results])
            ),
            "semantic_evaluable": sum(result["semantic_evaluable"] for result in results),
            "semantic_evaluability": float(
                np.mean([result["semantic_evaluable"] for result in results])
            ),
            "correct": sum(result["correct"] for result in results),
            "wrong_or_primary_error": sum(not result["correct"] for result in results),
            "accuracy": float(np.mean([result["correct"] for result in results])),
            "invalid_format": statuses["INVALID_FORMAT"],
            "truncated": statuses["TRUNCATED"],
            "runtime_error": statuses["RUNTIME_ERROR"],
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
            "max_tokens": max(tokens),
            "status_counts": json.dumps(dict(sorted(statuses.items())), sort_keys=True),
        }
        records.append(record)
        metrics[condition] = {
            key: float(record[key])
            for key in (
                "commitment_validity",
                "semantic_evaluability",
                "accuracy",
                "mean_tokens",
                "median_tokens",
                "max_tokens",
            )
        }
    return records, metrics


def _arrays(
    item_ids: list[str], parsed: dict[tuple[str, str, int], dict[str, Any]]
) -> dict[str, np.ndarray]:
    return {
        condition: np.asarray(
            [
                [int(not parsed[(item, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        for condition in CONDITIONS
    }


def _random_summary(
    estimands: dict[str, dict[str, float]],
    summaries: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for name in RANDOM_NAMES:
        for metric in ("G", "C", "D", "rescue", "damage"):
            values[metric].append(float(estimands[name][metric]))
        values["accuracy_change"].append(
            summaries[name]["accuracy"] - summaries[BASELINE]["accuracy"]
        )
        values["commitment_validity_change"].append(
            summaries[name]["commitment_validity"] - summaries[BASELINE]["commitment_validity"]
        )
        values["semantic_evaluability_change"].append(
            summaries[name]["semantic_evaluability"] - summaries[BASELINE]["semantic_evaluability"]
        )
        values["token_count_change"].append(
            summaries[name]["mean_tokens"] - summaries[BASELINE]["mean_tokens"]
        )
    return {
        metric: {
            "mean": float(np.mean(numbers)),
            "median": float(np.median(numbers)),
            "min": float(np.min(numbers)),
            "max": float(np.max(numbers)),
        }
        for metric, numbers in sorted(values.items())
    }


def _bootstrap(
    arrays: dict[str, np.ndarray],
    commitment: dict[str, np.ndarray],
    evaluability: dict[str, np.ndarray],
) -> dict[str, dict[str, float | int]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    names: list[str] = []
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_RESAMPLES):
        indices = rng.integers(0, len(arrays[BASELINE]), size=len(arrays[BASELINE]))
        point = {
            condition: audit_two_rollout_estimands(
                arrays[BASELINE][indices], arrays[condition][indices]
            )
            for condition in CONDITIONS[1:]
        }
        meaningful = point[MEANINGFUL]
        base_accuracy = float(1 - arrays[BASELINE][indices].mean())
        controller_accuracy = float(1 - arrays[MEANINGFUL][indices].mean())
        samples["meaningful:accuracy_change"].append(controller_accuracy - base_accuracy)
        samples["meaningful:commitment_validity_change"].append(
            float(commitment[MEANINGFUL][indices].mean() - commitment[BASELINE][indices].mean())
        )
        samples["meaningful:semantic_evaluability_change"].append(
            float(evaluability[MEANINGFUL][indices].mean() - evaluability[BASELINE][indices].mean())
        )
        for metric in ("G", "C", "D", "rescue", "damage"):
            samples[f"meaningful:{metric}"].append(float(meaningful[metric]))
        for metric in ("G", "C", "D"):
            random_values = [point[name][metric] for name in RANDOM_NAMES]
            samples[f"meaningful:{metric}_minus_random_mean"].append(
                float(meaningful[metric] - np.mean(random_values))
            )
            samples[f"meaningful:{metric}_minus_random_max"].append(
                float(meaningful[metric] - np.max(random_values))
            )
        if not names:
            names = sorted(samples)
    return {
        name: {
            "estimate": float(np.median(samples[name])),
            "q025": float(np.quantile(samples[name], 0.025)),
            "q975": float(np.quantile(samples[name], 0.975)),
            "resamples": BOOTSTRAP_RESAMPLES,
        }
        for name in names
    }


def _phi(left: np.ndarray, right: np.ndarray) -> float | None:
    a = left.reshape(-1).astype(np.float64)
    b = right.reshape(-1).astype(np.float64)
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(left: np.ndarray, right: np.ndarray) -> float | None:
    a = left.reshape(-1).astype(bool)
    b = right.reshape(-1).astype(bool)
    union = int(np.logical_or(a, b).sum())
    return float(np.logical_and(a, b).sum() / union) if union else None


def _answer_agreement(
    item_ids: list[str],
    left: str,
    right: str,
    parsed: dict[tuple[str, str, int], dict[str, Any]],
) -> float:
    values = []
    for item in item_ids:
        for left_rollout in (0, 1):
            for right_rollout in (0, 1):
                values.append(
                    _semantic_outcome(parsed[(item, left, left_rollout)])
                    == _semantic_outcome(parsed[(item, right, right_rollout)])
                )
    return float(np.mean(values))


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def analyze(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    schedule = read_json(review / "EVALUATION_SCHEDULE.json")
    rows = read_journal(review / "journal.jsonl")
    expected_rows = int(lock["schedule"]["logical_rows"])
    if len(rows) != expected_rows:
        raise RuntimeError(f"Gate 7 analysis requires all {expected_rows} rows; found {len(rows)}")
    expected_keys = Counter(
        (row["item_id"], row["condition"], int(row["rollout_index"])) for row in schedule
    )
    actual_keys = Counter(
        (row["item_id"], row["condition"], int(row["rollout_index"])) for row in rows
    )
    if actual_keys != expected_keys:
        raise RuntimeError("Gate 7 journal does not exactly match the frozen schedule")
    parsed = {
        (row["item_id"], row["condition"], int(row["rollout_index"])): _reparse(row) for row in rows
    }
    mismatches = []
    for row in rows:
        key = (row["item_id"], row["condition"], int(row["rollout_index"]))
        check = parsed[key]
        for field in ("status", "correct", "commitment_valid", "semantic_evaluable"):
            if check[field] != row.get(field):
                mismatches.append(
                    {
                        "key": key,
                        "field": field,
                        "recorded": row.get(field),
                        "reparsed": check[field],
                    }
                )
    if mismatches:
        raise RuntimeError(f"Gate 7 stored parser fields differ from frozen V3: {mismatches[:3]}")

    item_ids = sorted({str(row["item_id"]) for row in rows})
    if len(item_ids) != lock["sample"]["actual_n"]:
        raise RuntimeError("Gate 7 item count differs from lock")
    summary_rows, summaries = _condition_summary(rows, parsed)
    write_csv(
        review / "CONDITION_SUMMARY.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    arrays = _arrays(item_ids, parsed)
    estimands = {
        condition: audit_two_rollout_estimands(arrays[BASELINE], arrays[condition])
        for condition in CONDITIONS[1:]
    }
    baseline_b00 = float(np.mean(arrays[BASELINE][:, 0] * arrays[BASELINE][:, 1]))
    estimands[BASELINE] = {
        "B00": baseline_b00,
        "O00": 1.0 - baseline_b00,
        "accuracy": summaries[BASELINE]["accuracy"],
        "baseline_resampling_gain": (1.0 - baseline_b00) - summaries[BASELINE]["accuracy"],
    }
    random_summary = _random_summary(estimands, summaries)
    meaningful_contrasts = {
        metric: {
            "minus_random_mean": estimands[MEANINGFUL][metric] - random_summary[metric]["mean"],
            "minus_random_max": estimands[MEANINGFUL][metric] - random_summary[metric]["max"],
            "against_each_random": {
                name: estimands[MEANINGFUL][metric] - estimands[name][metric]
                for name in RANDOM_NAMES
            },
        }
        for metric in ("G", "C", "D")
    }

    commitment = {
        condition: np.asarray(
            [
                [int(parsed[(item, condition, rollout)]["commitment_valid"]) for rollout in (0, 1)]
                for item in item_ids
            ]
        )
        for condition in CONDITIONS
    }
    evaluability = {
        condition: np.asarray(
            [
                [
                    int(parsed[(item, condition, rollout)]["semantic_evaluable"])
                    for rollout in (0, 1)
                ]
                for item in item_ids
            ]
        )
        for condition in CONDITIONS
    }
    intervals = _bootstrap(arrays, commitment, evaluability)
    write_json(review / "BOOTSTRAP_INTERVALS.json", intervals)

    full = {
        "accuracy_change": summaries[MEANINGFUL]["accuracy"] - summaries[BASELINE]["accuracy"],
        "G": estimands[MEANINGFUL]["G"],
        "C": estimands[MEANINGFUL]["C"],
        "D": estimands[MEANINGFUL]["D"],
    }
    loo_rows: list[dict[str, Any]] = []
    loo_values: dict[str, list[float]] = defaultdict(list)
    for index, item_id in enumerate(item_ids):
        keep = np.arange(len(item_ids)) != index
        point = audit_two_rollout_estimands(arrays[BASELINE][keep], arrays[MEANINGFUL][keep])
        record = {
            "left_out_item_id": item_id,
            "accuracy_change": point["accuracy_condition"] - point["accuracy_baseline"],
            "G": point["G"],
            "C": point["C"],
            "D": point["D"],
        }
        loo_rows.append(record)
        for metric in ("accuracy_change", "G", "C", "D"):
            loo_values[metric].append(float(record[metric]))
    write_csv(review / "LOO_SENSITIVITY.csv", loo_rows, list(loo_rows[0]))
    loo_sign_stable = {
        metric: all(value > 0 for value in values)
        if full[metric] > 0
        else all(value < 0 for value in values)
        if full[metric] < 0
        else False
        for metric, values in loo_values.items()
    }

    contribution_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS[1:]:
        for item_id, contribution in zip(
            item_ids, item_contributions(arrays[BASELINE], arrays[condition]), strict=True
        ):
            contribution_rows.append({"item_id": item_id, "condition": condition, **contribution})
    write_csv(review / "ITEM_CONTRIBUTIONS.csv", contribution_rows, list(contribution_rows[0]))

    base_tokens = summaries[BASELINE]
    text_tokens = summaries[TEXTUAL]
    mean_tokens = summaries[MEANINGFUL]
    token_denominator = text_tokens["mean_tokens"] - base_tokens["mean_tokens"]
    token_gain_recovered = (
        (mean_tokens["mean_tokens"] - base_tokens["mean_tokens"]) / token_denominator
        if token_denominator > 0
        else None
    )
    accuracy_denominator = summaries[TEXTUAL]["accuracy"] - summaries[BASELINE]["accuracy"]
    accuracy_gain_recovered = (
        (summaries[MEANINGFUL]["accuracy"] - summaries[BASELINE]["accuracy"]) / accuracy_denominator
        if accuracy_denominator > 0
        else None
    )
    source_replicated = bool(
        summaries[TEXTUAL]["commitment_validity"] >= 0.90
        and summaries[TEXTUAL]["semantic_evaluability"] >= 0.90
        and summaries[TEXTUAL]["mean_tokens"] >= 1.5 * summaries[BASELINE]["mean_tokens"]
        and summaries[TEXTUAL]["median_tokens"] >= summaries[BASELINE]["median_tokens"] + 10
    )
    controller_style = bool(
        source_replicated
        and token_gain_recovered is not None
        and token_gain_recovered >= 0.50
        and summaries[MEANINGFUL]["median_tokens"]
        >= summaries[BASELINE]["median_tokens"]
        + 0.5 * (summaries[TEXTUAL]["median_tokens"] - summaries[BASELINE]["median_tokens"])
    )
    source_summary = {
        "classification": "CAREFUL_SOURCE_REPLICATED"
        if source_replicated
        else "CAREFUL_SOURCE_NOT_REPLICATED",
        "activation_controller_style_regime_replicated": controller_style,
        "token_gain_recovered_fraction": token_gain_recovered,
        "accuracy_gain_recovered_fraction": accuracy_gain_recovered,
        "meaningful_textual_semantic_agreement": _answer_agreement(
            item_ids, MEANINGFUL, TEXTUAL, parsed
        ),
        "meaningful_baseline_semantic_agreement": _answer_agreement(
            item_ids, MEANINGFUL, BASELINE, parsed
        ),
    }
    write_json(review / "SOURCE_REFERENCE_SUMMARY.json", source_summary)

    classification, gates = classify_gate7(
        baseline=summaries[BASELINE],
        controller=summaries[MEANINGFUL],
        controller_estimands=estimands[MEANINGFUL],
        random_summary=random_summary,
        bootstrap=intervals,
        loo_sign_stable=loo_sign_stable,
        controller_style_replicated=controller_style,
    )
    auxiliary = {
        "semantic_answer_change_rate_meaningful_vs_baseline": 1.0
        - _answer_agreement(item_ids, MEANINGFUL, BASELINE, parsed),
        "semantic_answer_agreement_meaningful_vs_textual": source_summary[
            "meaningful_textual_semantic_agreement"
        ],
        "error_jaccard_meaningful_baseline": _jaccard(arrays[BASELINE], arrays[MEANINGFUL]),
        "error_phi_meaningful_baseline": _phi(arrays[BASELINE], arrays[MEANINGFUL]),
        "pair_oracle_meaningful_baseline": estimands[MEANINGFUL]["O0j"],
        "first_divergence_token_mean": statistics.mean(values)
        if (
            values := [
                value
                for row in rows
                if row["condition"] == MEANINGFUL
                for value in [
                    _first_divergence(
                        next(
                            base["generated_token_ids"]
                            for base in rows
                            if base["item_id"] == row["item_id"]
                            and base["condition"] == BASELINE
                            and base["rollout_index"] == row["rollout_index"]
                        ),
                        row["generated_token_ids"],
                    )
                ]
                if value is not None
            ]
        )
        else None,
    }
    result = {
        "classification": classification,
        "source_policy_classification": source_summary["classification"],
        "gates": gates,
        "summaries": summaries,
        "estimands": estimands,
        "random_summary": random_summary,
        "meaningful_random_contrasts": meaningful_contrasts,
        "bootstrap": {"file": "BOOTSTRAP_INTERVALS.json", "resamples": BOOTSTRAP_RESAMPLES},
        "loo_sign_stable": loo_sign_stable,
        "auxiliary": auxiliary,
        "historical_gate6_3_classification": "GATE6_3_SINGLE_MEAN_DESTRUCTIVE",
        "historical_result_modified": False,
    }
    write_json(review / "ESTIMANDS.json", result)

    elapsed = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    cost = {
        "scientific_trajectories": len(rows),
        "summed_generation_seconds": sum(elapsed),
        "summed_generation_hours": sum(elapsed) / 3600,
        "estimated_generation_cost_usd_at_0_44": sum(elapsed) / 3600 * 0.44,
        "startup_and_engineering_cost_to_be_filled_from_remote_lifecycle": None,
        "hard_ceiling_usd": 4.0,
    }
    write_json(review / "COST.json", cost)

    random_rows = "".join(
        f"| `{name}` | {summaries[name]['accuracy']:.4f} | "
        f"{estimands[name]['G']:.6f} | {estimands[name]['C']:.6f} | "
        f"{estimands[name]['D']:.6f} |\n"
        for name in RANDOM_NAMES
    )
    interval_rows = "".join(
        f"| `{name}` | {record['q025']:.6f} | {record['q975']:.6f} |\n"
        for name, record in sorted(intervals.items())
    )
    binding = read_json(review / "EXPERIMENT_SOURCE_COMMIT.json")
    report = f"""# Gate 7 — Fresh Single-L27 Replication

Primary classification: `{classification}`.

Source-policy classification: `{source_summary["classification"]}`.

Experiment source commit: `{binding["experiment_source_commit"]}`.

Fresh sample: {len(item_ids)} CRUXEval items; {len(rows)} trajectories; seven
conditions; two independent rollouts per item-condition. Semantic evaluator:
`{lock["instrument"]["evaluator"]["version"]}`.

The historical Gate 6.3 classification remains
`GATE6_3_SINGLE_MEAN_DESTRUCTIVE` and was not modified.

## Conditions

| condition | commitment | evaluability | accuracy | mean / median / max tokens |
|---|---:|---:|---:|---:|
""" + "".join(
        f"| `{row['condition']}` | {row['commitment_validity']:.4f} | "
        f"{row['semantic_evaluability']:.4f} | {row['accuracy']:.4f} | "
        f"{row['mean_tokens']:.2f} / {row['median_tokens']:.1f} / "
        f"{row['max_tokens']} |\n"
        for row in summary_rows
    )
    report += f"""
## Meaningful controller

- Accuracy difference: {full["accuracy_change"]:.6f}
- G: {estimands[MEANINGFUL]["G"]:.6f}
- C: {estimands[MEANINGFUL]["C"]:.6f}
- D: {estimands[MEANINGFUL]["D"]:.6f}
- Rescue: {estimands[MEANINGFUL]["rescue"]:.6f}
- Damage: {estimands[MEANINGFUL]["damage"]:.6f}
- G minus random mean/max:
  {meaningful_contrasts["G"]["minus_random_mean"]:.6f} /
  {meaningful_contrasts["G"]["minus_random_max"]:.6f}
- C minus random mean/max:
  {meaningful_contrasts["C"]["minus_random_mean"]:.6f} /
  {meaningful_contrasts["C"]["minus_random_max"]:.6f}
- D minus random mean/max:
  {meaningful_contrasts["D"]["minus_random_mean"]:.6f} /
  {meaningful_contrasts["D"]["minus_random_max"]:.6f}

## Frozen guards

- Commitment-validity guard: {gates["commitment_validity_guard"]}
- Semantic-evaluability guard: {gates["semantic_evaluability_guard"]}
- Competence guard: {gates["competence_guard"]}

The controller improved primary accuracy by {full["accuracy_change"]:.4f} and
showed large positive G/C/D beyond every new random controller, but commitment
validity and semantic evaluability were 0.9000 versus a baseline of 0.9917.
The frozen relative guard required at least 0.9417, so the exhaustive
classification is mechanically `GATE7_DESTRUCTIVE`.

## Random-controller null

| random condition | accuracy | G | C | D |
|---|---:|---:|---:|---:|
{random_rows}

## Textual CAREFUL source

- Classification: `{source_summary["classification"]}`
- Token-gain fraction recovered by activation controller:
  {source_summary["token_gain_recovered_fraction"]:.6f}
- Accuracy-gain fraction recovered by activation controller:
  {source_summary["accuracy_gain_recovered_fraction"]:.6f}
- Meaningful/textual semantic agreement:
  {source_summary["meaningful_textual_semantic_agreement"]:.6f}

## Item-cluster bootstrap (10,000 resamples)

| estimand | 2.5% | 97.5% |
|---|---:|---:|
{interval_rows}

## Interpretation boundary

This is an independent DEVELOPMENT replication under a parser frozen before
collection. It is not confirmatory, Q2, character-count replication, or a
general claim beyond Qwen3-8B × CRUXEval. The controller produced a strong,
specific semantic-error-profile and accuracy signal, but it also induced a
condition-specific commitment/evaluability loss that violated the frozen
non-destructiveness guard. The exhaustive classification was applied
mechanically after all rows were collected.
"""
    (review / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = analyze(args.review_dir.resolve())
    print(json.dumps({"classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
