#!/usr/bin/env python3
"""Build the model-free review report for completion-length diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

HOURLY_COST = 0.44
Q1_SMOKE_ITEMS = 20


def _load(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    recommendation = json.loads((path / "cap_recommendation.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (path / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return manifest, recommendation, rows


def _summary(path: Path) -> dict[str, Any]:
    manifest, recommendation, rows = _load(path)
    required_caps = next(
        row["item_required_caps"]
        for row in recommendation["caps"]
        if row["cap"] == recommendation["proposed_cap"]
    )
    completion_rows = {
        item_id: next(
            row
            for row in rows
            if row["item_id"] == item_id
            and int(row["metadata"]["diagnostic_cap"]) == required_cap
        )
        for item_id, required_cap in required_caps.items()
    }
    durations = [float(row["metadata"]["generation_seconds"]) for row in completion_rows.values()]
    projected_seconds = statistics.mean(durations) * Q1_SMOKE_ITEMS
    return {
        "candidate": manifest["identity"]["candidate"],
        "diagnostic_items": len(required_caps),
        "diagnostic_rows": len(rows),
        "proposed_cap": recommendation["proposed_cap"],
        "high_cap_warning": recommendation["high_cap_warning"],
        "required_caps": required_caps,
        "natural_tokens_min": min(int(row["token_count"]) for row in completion_rows.values()),
        "natural_tokens_median": statistics.median(
            int(row["token_count"]) for row in completion_rows.values()
        ),
        "natural_tokens_max": max(int(row["token_count"]) for row in completion_rows.values()),
        "diagnostic_seconds": sum(float(row["metadata"]["generation_seconds"]) for row in rows),
        "mean_seconds_per_completed_item": statistics.mean(durations),
        "projected_q1_smoke_seconds": projected_seconds,
        "projected_q1_smoke_hours": projected_seconds / 3600,
        "projected_q1_smoke_cost_usd": projected_seconds / 3600 * HOURLY_COST,
        "accuracy_used_for_cap": False,
        "scientific_qualification_run": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, help="candidate=/path/to/diagnostic"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for argument in args.run:
        candidate, separator, raw_path = argument.partition("=")
        if not separator or not candidate or not raw_path:
            raise ValueError("--run must have the form candidate=/path/to/diagnostic")
        summary = _summary(Path(raw_path))
        summary["candidate"] = candidate
        rows.append(summary)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "COMPLETION_DIAGNOSTICS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "candidate",
            "diagnostic_items",
            "diagnostic_rows",
            "proposed_cap",
            "high_cap_warning",
            "required_caps",
            "natural_tokens_min",
            "natural_tokens_median",
            "natural_tokens_max",
            "diagnostic_seconds",
            "mean_seconds_per_completed_item",
            "projected_q1_smoke_seconds",
            "projected_q1_smoke_hours",
            "projected_q1_smoke_cost_usd",
            "accuracy_used_for_cap",
            "scientific_qualification_run",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# Completion-budget diagnostic report",
        "",
        "Classification: DEVELOPMENT-ONLY OPERATIONAL DIAGNOSTIC.",
        "No Q1 smoke, semantic qualification, steering, or geometry run was performed.",
        "Caps were selected from natural completion length only; accuracy was not used.",
        "",
        "| Candidate | Proposed cap | Natural token range | High-cap warning | "
        "Projected 20-item smoke | Projected cost |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    for row in rows:
        token_range = (
            f"{row['natural_tokens_min']}–{row['natural_tokens_max']}"
        )
        report.append(
            f"| {row['candidate']} | {row['proposed_cap']} | {token_range} | "
            f"{'YES' if row['high_cap_warning'] else 'NO'} | "
            f"{row['projected_q1_smoke_hours']:.2f} h | "
            f"US${row['projected_q1_smoke_cost_usd']:.2f} |"
        )
    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "- CRUXEval is the cheapest operational candidate in this diagnostic, "
            "with a prospective cap of 8,192.",
            "- LiveBench requires a prospective cap of 16,384 and remains operationally heavier.",
            "- LiveCodeBench requires 32,768 because one diagnostic item completed at "
            "18,884 tokens; this is an explicit high-cost warning, not a semantic rejection.",
            "- The 2048-token runs remain `LOW_CAP_DIAGNOSTIC` and are excluded from "
            "qualification.",
            "- The next authorized action is a fresh 20-item × 1-seed smoke only after "
            "principal review of this cost report.",
        ]
    )
    (args.output / "COMPLETION_DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (args.output / "COMPLETION_DIAGNOSTICS.json").write_text(
        json.dumps({"candidates": rows, "hourly_cost_usd": HOURLY_COST}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
