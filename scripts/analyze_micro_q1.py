#!/usr/bin/env python3
"""Offline Gate-4 journal validation and item-cluster analysis."""

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

from epistemic_geometry.experiments.micro_q1 import (  # noqa: E402
    all_pair_estimands,
    bootstrap_pair_estimands,
)

CONDITIONS = ("BASELINE", "CPLUS", "CMINUS", "CRANDOM")


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _error(record: dict[str, Any]) -> int:
    return int(record["status"] != "VALID_CORRECT")


def _validate(
    rows: list[dict[str, Any]],
) -> tuple[list[str], dict[tuple[str, str, int], dict[str, Any]]]:
    keyed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        if key in keyed:
            raise ValueError(f"duplicate trajectory key: {key}")
        if row["condition"] not in CONDITIONS:
            raise ValueError(f"unknown condition: {row['condition']}")
        keyed[key] = row
    ids = sorted({key[0] for key in keyed})
    expected = {
        (item_id, condition, rollout)
        for item_id in ids
        for condition in CONDITIONS
        for rollout in (0, 1)
    }
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        raise ValueError(f"incomplete trajectory table; missing={missing[:3]} extra={extra[:3]}")
    return ids, keyed


def _matrix(
    ids: list[str], keyed: dict[tuple[str, str, int], dict[str, Any]], condition: str
) -> np.ndarray:
    return np.asarray(
        [[_error(keyed[(item_id, condition, rollout)]) for rollout in (0, 1)] for item_id in ids],
        dtype=np.int8,
    )


def _summary(
    matrix: np.ndarray,
    keyed: dict[tuple[str, str, int], dict[str, Any]],
    ids: list[str],
    condition: str,
) -> dict[str, Any]:
    rows = [keyed[(item_id, condition, rollout)] for item_id in ids for rollout in (0, 1)]
    status = Counter(str(row["status"]) for row in rows)
    tokens = [int(row["generated_token_count"]) for row in rows]
    validity = sum(status[value] for value in ("VALID_CORRECT", "VALID_WRONG")) / len(rows)
    return {
        "condition": condition,
        "n": len(rows),
        "valid": status["VALID_CORRECT"] + status["VALID_WRONG"],
        "correct": status["VALID_CORRECT"],
        "wrong": status["VALID_WRONG"],
        "invalid": status["INVALID_FORMAT"],
        "truncated": status["TRUNCATED"],
        "runtime_error": status["RUNTIME_ERROR"],
        "validity": validity,
        "accuracy": 1.0 - float(matrix.mean()),
        "mean_tokens": float(np.mean(tokens)),
        "median_tokens": float(np.median(tokens)),
        "max_tokens": max(tokens),
        "status_counts": dict(status),
    }


def _classify(
    summaries: dict[str, dict[str, Any]], estimands: dict[str, dict[str, float]]
) -> tuple[str, dict[str, Any]]:
    baseline = summaries["BASELINE"]
    decisions: dict[str, Any] = {}
    for condition, name in (("CPLUS", "plus"), ("CMINUS", "minus")):
        summary = summaries[condition]
        values = estimands[name]
        valid_guard = (
            summary["validity"] >= 0.90 and summary["validity"] >= baseline["validity"] - 0.05
        )
        competence = summary["accuracy"] >= baseline["accuracy"] - 0.10
        movement = valid_guard and competence and values["D"] >= 0.05 and values["Delta_D"] >= 0.05
        useful = (
            movement and values["G"] >= 0.03 and values["C"] >= 0.03 and values["Delta_C"] >= 0.05
        )
        decisions[condition] = {
            "validity_guard": valid_guard,
            "competence_preserving": competence,
            "movement": movement,
            "useful": useful,
        }
    if all(
        not decisions[c]["competence_preserving"] or not decisions[c]["validity_guard"]
        for c in ("CPLUS", "CMINUS")
    ):
        return "MICRO_Q1_DESTRUCTIVE", decisions
    if any(decisions[c]["useful"] for c in ("CPLUS", "CMINUS")):
        return "MICRO_Q1_USEFUL_COMPLEMENTARITY_SIGNAL", decisions
    if any(decisions[c]["movement"] for c in ("CPLUS", "CMINUS")):
        return "MICRO_Q1_ERROR_PROFILE_MOVEMENT_ONLY", decisions
    return "MICRO_Q1_NO_DETECTABLE_SIGNAL", decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _read(args.journal)
    ids, keyed = _validate(rows)
    matrices = {condition: _matrix(ids, keyed, condition) for condition in CONDITIONS}
    summaries = {
        condition: _summary(matrices[condition], keyed, ids, condition) for condition in CONDITIONS
    }
    estimands = all_pair_estimands(
        matrices["BASELINE"], matrices["CPLUS"], matrices["CMINUS"], matrices["CRANDOM"]
    )
    bootstrap = bootstrap_pair_estimands(
        matrices["BASELINE"],
        {"plus": matrices["CPLUS"], "minus": matrices["CMINUS"], "random": matrices["CRANDOM"]},
        resamples=5000,
        seed=20260819,
    )
    classification, decisions = _classify(summaries, estimands)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "TRAJECTORY_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "item_id",
            "condition",
            "rollout_index",
            "seed",
            "status",
            "correct",
            "parsed_answer",
            "reference_answer",
            "generated_token_count",
            "prompt_hash",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    with (output / "CONDITION_SUMMARY.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: summary[field] for field in fields} for summary in summaries.values()
        )
    (output / "ESTIMANDS.json").write_text(
        json.dumps(estimands, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "BOOTSTRAP_INTERVALS.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "FIRST ORIGINAL MICRO-Q1",
        "======================================================================",
        "",
        "This is a DEVELOPMENT kill-test under the frozen Gate-4 protocol.",
        "The primary error is one for every non-VALID_CORRECT model outcome;",
        "infrastructure errors are not scientific outcomes.",
        "",
        "## Condition results",
        "",
        "| Condition | Valid/100 | Correct | Wrong | Invalid | Truncated | "
        "Accuracy | Validity | Mean tokens | Median tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        summary = summaries[condition]
        report.append(
            f"| {condition} | {summary['valid']}/100 | {summary['correct']} | "
            f"{summary['wrong']} | {summary['invalid']} | {summary['truncated']} | "
            f"{summary['accuracy']:.3f} | {summary['validity']:.3f} | "
            f"{summary['mean_tokens']:.1f} | {summary['median_tokens']:.1f} |"
        )
    report += [
        "",
        "## Estimands",
        "",
        "```json",
        json.dumps(estimands, indent=2, sort_keys=True),
        "```",
        "",
        "## Bootstrap",
        "",
        "```json",
        json.dumps(bootstrap, indent=2, sort_keys=True),
        "```",
        "",
        f"## Classification\n\n`{classification}`",
        "",
        "## Guard decisions",
        "",
        "```json",
        json.dumps(decisions, indent=2, sort_keys=True),
        "```",
        "",
        "No character-count replication, Q2, geometry, or confirmatory holdout was run.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "FINAL_CLASSIFICATION.json").write_text(
        json.dumps(
            {"classification": classification, "decisions": decisions, "n_items": len(ids)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"classification": classification, "n_items": len(ids), "rows": len(rows)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
