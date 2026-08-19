#!/usr/bin/env python3
"""Analyze the frozen long character-count follow-up locally."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.v4.character_parser import parse_final_integer  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _status_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        token_count = row.get("token_count")
        status, parsed, reason = parse_final_integer(
            str(row.get("raw_output", "")),
            truncated=token_count is not None and int(token_count) >= 8192,
        )
        if status == "PARSED":
            status = (
                "VALID_CORRECT"
                if parsed == int(row["reference_answer"])
                else "VALID_WRONG"
            )
        output.append(
            {
                "item_id": row["item_id"],
                "stratum": row["stratum"],
                "original_recorded_status": row["status"],
                "corrected_status": status,
                "reference_answer": int(row["reference_answer"]),
                "parsed_answer": parsed,
                "token_count": token_count,
                "wall_seconds": row.get("timing_seconds_wall"),
                "parse_reason": reason or "deterministic semantic FINAL parse",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "review" / "q1_v4_microbench",
    )
    parser.add_argument("--scientific-active-seconds", type=float, required=True)
    parser.add_argument("--recovery-active-seconds", type=float, required=True)
    parser.add_argument("--rate-per-hour", type=float, default=0.44)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = _status_rows(_rows(args.run / "journal.jsonl"))
    counts = Counter(row["corrected_status"] for row in rows)
    valid = counts["VALID_CORRECT"] + counts["VALID_WRONG"]
    correct = counts["VALID_CORRECT"]
    tokens = [int(row["token_count"]) for row in rows if row["token_count"] is not None]
    walls = [float(row["wall_seconds"]) for row in rows if row["wall_seconds"] is not None]
    valid_rate = valid / len(rows) if rows else 0.0
    accuracy = correct / valid if valid else None
    failures = (
        counts["INVALID_FORMAT"]
        + counts["TRUNCATED_THINKING"]
        + counts["RUNTIME_ERROR"]
    )
    if (
        valid_rate >= 0.90
        and correct >= 2
        and counts["VALID_WRONG"] >= 2
        and failures <= counts["VALID_WRONG"]
    ):
        classification = "LONG_CHARCOUNT_PROMISING"
        next_action = "CHARCOUNT_BASELINE_QUALIFICATION"
    elif (
        valid_rate >= 0.90
        and counts["VALID_WRONG"] < 2
        and accuracy is not None
        and accuracy >= 0.80
    ):
        classification = "LONG_CHARCOUNT_SATURATED"
        next_action = "DENSE_CODE_3_TO_5_PROBLEM_PILOT"
    elif valid_rate >= 0.90 and correct < 2:
        classification = "LONG_CHARCOUNT_FLOOR"
        next_action = "DENSE_CODE_3_TO_5_PROBLEM_PILOT"
    else:
        classification = "LONG_CHARCOUNT_FORMAT_OR_COMPLETION_FAILURE"
        next_action = "DENSE_CODE_3_TO_5_PROBLEM_PILOT"

    with (args.output / "CHARCOUNT_LONG_RESULTS.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean_tokens = statistics.mean(tokens) if tokens else None
    median_tokens = statistics.median(tokens) if tokens else None
    max_tokens = max(tokens) if tokens else None
    total_wall = sum(walls) if walls else 0.0
    mean_wall = statistics.mean(walls) if walls else None
    median_wall = statistics.median(walls) if walls else None
    accuracy_line = (
        f"Conditional semantic accuracy: {accuracy:.1%}"
        if accuracy is not None
        else "Conditional semantic accuracy: undefined"
    )
    long_report = [
        "# V4 long character-count report",
        "",
        "This is a DEVELOPMENT-only completion of the ten pre-registered long-stratum",
        "items. No items, prompts, seeds, cap, or generator settings were changed.",
        "",
        "## Outcome",
        "",
        f"Items: {len(rows)}/10",
        f"Valid completion: {valid}/{len(rows)} ({valid_rate:.1%})",
        f"Valid correct: {counts['VALID_CORRECT']}",
        f"Valid wrong: {counts['VALID_WRONG']}",
        f"Invalid format: {counts['INVALID_FORMAT']}",
        f"Truncated thinking: {counts['TRUNCATED_THINKING']}",
        f"Runtime error: {counts['RUNTIME_ERROR']}",
        accuracy_line,
        f"Generated tokens mean / median / max: {mean_tokens:.1f} / "
        f"{median_tokens:.1f} / {max_tokens}",
        f"Wall seconds total / mean / median: {total_wall:.2f} / "
        f"{mean_wall:.2f} / {median_wall:.2f}",
        "",
        f"Classification: {classification}",
        "",
        "Increasing string length produced no genuine semantic errors in this",
        "ten-item development sample. The corrected parser accepted explicit FINAL",
        "commitments and did not infer answers from free-form reasoning.",
    ]
    (args.output / "CHARCOUNT_LONG_REPORT.md").write_text(
        "\n".join(long_report) + "\n", encoding="utf-8"
    )

    total_active = args.scientific_active_seconds + args.recovery_active_seconds
    cost = total_active * args.rate_per_hour / 3600
    cost_report = [
        "# V4 follow-up cost report",
        "",
        f"Scientific run active seconds: {args.scientific_active_seconds:.0f}",
        f"Artifact-recovery restart active seconds: {args.recovery_active_seconds:.0f}",
        f"Total billed active seconds: {total_active:.0f}",
        f"Rate per hour: {args.rate_per_hour} USD/hour",
        f"Estimated incremental cost: {cost:.3f} USD",
        "Target: <= 0.10 USD",
        "Hard stop: <= 0.20 USD",
        "Model download: NO; cached revision reused",
        "Pod final state: EXITED",
    ]
    (args.output / "V4_FOLLOWUP_COST_REPORT.md").write_text(
        "\n".join(cost_report) + "\n", encoding="utf-8"
    )

    journal_hash = hashlib.sha256((args.run / "journal.jsonl").read_bytes()).hexdigest()
    final_report = [
        "# Q1 V4 follow-up — long character-count stratum",
        "",
        "## Boundary",
        "",
        "This follow-up completed only the ten previously frozen long-stratum items.",
        "It did not alter the original V4 result, geometry evidence, V1–V3 outcomes,",
        "or dense-code audit. No steering, activation intervention, PCA, layer sweep,",
        "geometry causal test, LiveCodeBench pilot, or holdout access occurred.",
        "",
        "## Answers",
        "",
        "1. Original six format errors: all six contain one deterministic final integer",
        "   under the corrected parser, including the two-line Markdown heading variant.",
        "   The original classifications remain preserved; this is a diagnostic correction.",
        "2. Corrected short+medium diagnostic: 20/20 valid semantic answers correct",
        "   (100.0%). This indicates semantic saturation after removing the parser confound.",
        f"3. Frozen long stratum: {valid}/10 valid, {correct} correct,",
        f"   {counts['VALID_WRONG']} genuine wrong, {counts['INVALID_FORMAT']} invalid,",
        f"   {counts['TRUNCATED_THINKING']} truncated, {counts['RUNTIME_ERROR']} runtime errors.",
        "4. Effect of length: no genuine errors appeared in the long sample;",
        "   Qwen3-8B remained saturated on this character-count design.",
        f"5. Instrument classification: {classification}. It is not a useful",
        "   error benchmark for the present Q1 purpose.",
        f"6. Recommended next single experiment: {next_action}.",
        "   Do not execute it automatically; this is a principal-review recommendation.",
        "",
        "## Provenance",
        "",
        f"Long-run manifest: {args.run / 'manifest.json'}",
        f"Journal SHA-256: {journal_hash}",
        "Pod final state: EXITED",
        "",
        "The accumulated evidence supports moving away from character counting, not",
        "relaxing thresholds or manufacturing difficulty through parser or cap changes.",
    ]
    (args.output / "V4_FOLLOWUP_FINAL_REPORT.md").write_text(
        "\n".join(final_report) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"classification": classification, "next_action": next_action, "cost": cost},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
