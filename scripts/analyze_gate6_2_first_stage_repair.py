#!/usr/bin/env python3
"""Analyze the completed Gate 6.2 matched-coupling first-stage gate.

This is deliberately outcome-independent with respect to continuation: it
reads the complete manipulation journal, computes the frozen gate metrics, and
never launches the 60-item evaluation phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

MEANINGFUL = (
    "BEST_SINGLE_MEAN_PLUS",
    "MULTILAYER_MEAN_PLUS",
    "MULTILAYER_MEAN_MINUS",
)
RANDOM = tuple(f"MULTILAYER_RANDOM_MEAN_R{i}" for i in range(4))
VALID = {"VALID_CORRECT", "VALID_WRONG"}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 200:
        raise RuntimeError(f"expected 200 manipulation rows, found {len(rows)}")
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate manipulation logical row")
    return rows


def semantic_outcome(row: dict[str, Any]) -> tuple[str, str]:
    status = str(row["status"])
    if status in VALID:
        return ("VALID", str(row.get("parsed_answer")))
    return ("MECHANICAL", status)


def sequence_change(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return list(left.get("generated_token_ids", [])) != list(right.get("generated_token_ids", []))


def first_divergence(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_ids = list(left.get("generated_token_ids", []))
    right_ids = list(right.get("generated_token_ids", []))
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids, strict=False)):
        if left_id != right_id:
            return index
    if len(left_ids) != len(right_ids):
        return min(len(left_ids), len(right_ids))
    return None


def summarize(
    condition: str,
    rows: list[dict[str, Any]],
    baseline_by_item: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    statuses = {
        status: sum(str(row["status"]) == status for row in rows)
        for status in sorted({str(row["status"]) for row in rows})
    }
    divergences = [
        first_divergence(baseline_by_item[str(row["item_id"])], row)
        for row in rows
    ]
    defined_divergences = [value for value in divergences if value is not None]
    valid_count = sum(str(row["status"]) in VALID for row in rows)
    q = sum(
        semantic_outcome(baseline_by_item[str(row["item_id"])]) != semantic_outcome(row)
        for row in rows
    ) / len(rows)
    raw_change = sum(
        sequence_change(baseline_by_item[str(row["item_id"])], row) for row in rows
    ) / len(rows)
    tokens = [int(row["generated_token_count"]) for row in rows]
    return {
        "condition": condition,
        "n": len(rows),
        "valid": valid_count,
        "validity": valid_count / len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "wrong": sum(str(row["status"]) == "VALID_WRONG" for row in rows),
        "invalid_format": sum(str(row["status"]) == "INVALID_FORMAT" for row in rows),
        "truncated": sum(str(row["status"]) == "TRUNCATED" for row in rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "semantic_change_rate": q,
        "raw_token_sequence_change_rate": raw_change,
        "mean_first_divergence_token": (
            sum(defined_divergences) / len(defined_divergences) if defined_divergences else None
        ),
        "mean_tokens": statistics.mean(tokens),
        "median_tokens": statistics.median(tokens),
        "max_tokens": max(tokens),
        "statuses": statuses,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("review/gate6_2_first_stage_repair_mean_bridge"),
    )
    args = parser.parse_args()
    review = args.review_dir.resolve()
    rows = load_rows(review / "journal.jsonl")
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_condition[str(row["condition"])].append(row)
        if str(row["condition"]) == "BASELINE":
            by_item[str(row["item_id"])] = row
    if len(by_item) != 20 or any(len(value) != 20 for value in by_condition.values()):
        raise RuntimeError("manipulation manifest does not contain 20 rows per condition")
    if any(
        row["seed"] != by_item[str(row["item_id"])]["seed"]
        for row in rows
        if str(row["condition"]) != "BASELINE"
    ):
        raise RuntimeError("matched-coupling seeds are not shared by item")

    summaries = {
        condition: summarize(condition, by_condition[condition], by_item)
        for condition in sorted(by_condition)
    }
    random_mean_q = statistics.mean(summaries[name]["semantic_change_rate"] for name in RANDOM)
    random_max_q = max(summaries[name]["semantic_change_rate"] for name in RANDOM)
    gates: dict[str, dict[str, Any]] = {}
    for condition in MEANINGFUL:
        row = summaries[condition]
        gates[condition] = {
            "validity_pass": row["validity"] >= 0.85,
            "semantic_change_pass": row["semantic_change_rate"] >= 0.15,
            "meaningful_minus_random_mean_pass": (
                row["semantic_change_rate"] - random_mean_q >= 0.05
            ),
            "random_mean_q": random_mean_q,
            "random_max_q": random_max_q,
        }
        gates[condition]["pass"] = all(
            gates[condition][key]
            for key in (
                "validity_pass",
                "semantic_change_pass",
                "meaningful_minus_random_mean_pass",
            )
        )
    passed = [condition for condition, gate in gates.items() if gate["pass"]]
    classification = "GATE6_2_FIRST_STAGE_PASS" if passed else "GATE6_2_NO_BEHAVIORAL_FIRST_STAGE"

    with (review / "MANIPULATION_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "condition", "n", "valid", "validity", "correct", "wrong",
            "invalid_format", "truncated",
            "accuracy", "semantic_change_rate", "raw_token_sequence_change_rate",
            "mean_first_divergence_token", "mean_tokens", "median_tokens", "max_tokens", "statuses",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries.values():
            writer.writerow({**row, "statuses": json.dumps(row["statuses"], sort_keys=True)})
    (review / "CONDITION_SUMMARY.csv").write_bytes(
        (review / "MANIPULATION_RESULTS.csv").read_bytes()
    )

    estimands = {
        "phase": "MANIPULATION",
        "n_items": 20,
        "seed_regime": "MATCHED_COUPLING_SECONDARY",
        "baseline_condition": "BASELINE",
        "random_conditions": list(RANDOM),
        "random_mean_semantic_change_rate": random_mean_q,
        "random_max_semantic_change_rate": random_max_q,
        "conditions": summaries,
        "frozen_gate": {
            "validity_minimum": 0.85,
            "semantic_change_minimum": 0.15,
            "meaningful_minus_random_mean_minimum": 0.05,
            "duration_not_applicable": True,
        },
        "gate_checks": gates,
        "passed_meaningful_conditions": passed,
        "classification": classification,
        "evaluation_executed": False,
    }
    (review / "MANIPULATION_ESTIMANDS.json").write_text(
        json.dumps(estimands, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (review / "ESTIMANDS.json").write_bytes((review / "MANIPULATION_ESTIMANDS.json").read_bytes())
    (review / "SOURCE_PHASE_DECISION_FINAL.json").write_text(
        json.dumps(
            {
                "source_phase_decision": json.loads(
                    (review / "SOURCE_PHASE_DECISION_CORRECTED.json").read_text()
                ),
                "manipulation_gate": classification,
                "passed_meaningful_conditions": passed,
                "evaluation_executed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Gate 6.2 — First-stage repair and paired-mean bridge",
        "",
        f"Classification: `{classification}`.",
        "",
        "The source-only controller selection passed for paired-mean prompt L27",
        "(single) and prompt L22/L27/L32 (multilayer). The RFM hierarchy selected",
        "no controller source because no RFM source group had at least two passing",
        "layers.",
        "",
        "## Matched manipulation gate",
        "",
        "| condition | valid/20 | accuracy | semantic Q | raw token change | mean tokens | gate |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for condition, row in summaries.items():
        gate_value = gates.get(condition, {}).get("pass", "reference/random")
        lines.append(
            f"| {condition} | {row['valid']}/20 | {row['accuracy']:.3f} | "
            f"{row['semantic_change_rate']:.3f} | {row['raw_token_sequence_change_rate']:.3f} | "
            f"{row['mean_tokens']:.1f} | {gate_value} |"
        )
    lines += [
        "",
        f"Random mean Q: `{random_mean_q:.6f}`; random maximum Q: `{random_max_q:.6f}`.",
        "",
        "The duration contrast is not applicable in this frozen Gate 6.2 protocol.",
        "The meaningful plus controllers produced high matched changes but failed",
        "the 0.85 validity guard. The meaningful minus controller preserved validity",
        "but did not exceed the random-mean semantic-change null by 0.05.",
        "Therefore the 60-item evaluation phase was not executed.",
        "",
        "Scientific firewall: character count NOT RUN; Q2 NOT RUN; confirmatory",
        "holdout UNTOUCHED.",
    ]
    (review / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashed_files = (
        "MANIPULATION_MANIFEST.json",
        "EVALUATION_MANIFEST.json",
        "journal.jsonl",
        "MANIPULATION_RESULTS.csv",
        "MANIPULATION_ESTIMANDS.json",
        "REPORT.md",
        "COST.json",
    )
    hashes = {
        name: hashlib.sha256((review / name).read_bytes()).hexdigest()
        for name in hashed_files
        if (review / name).exists()
    }
    (review / "ARTIFACT_HASHES.json").write_text(
        json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"classification": classification, "passed": passed, "random_mean_q": random_mean_q},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
