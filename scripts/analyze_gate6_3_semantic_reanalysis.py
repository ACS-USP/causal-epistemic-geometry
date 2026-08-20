#!/usr/bin/env python3
"""Independently reanalyse the immutable Gate 6.2 manipulation journal.

This script is intentionally offline.  It never loads a model, never reruns a
trajectory, and does not call the historical Gate 6.2 analysis script.  It
uses the prospectively registered ``external-semantic-v2`` parser uniformly
for every preserved condition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.external.base import (
    ExternalStatus,
    evaluate_external_answer,
)
from epistemic_geometry.benchmarks.external.semantic_v2 import (
    PARSER_VERSION,
    parse_external_answer_v2,
)

VALID = {ExternalStatus.VALID_CORRECT.value, ExternalStatus.VALID_WRONG.value}
MEANINGFUL = (
    "BEST_SINGLE_MEAN_PLUS",
    "MULTILAYER_MEAN_PLUS",
    "MULTILAYER_MEAN_MINUS",
)
RANDOM = tuple(f"MULTILAYER_RANDOM_MEAN_R{i}" for i in range(4))
EXPECTED_CONDITIONS = {
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "TEXTUAL_DIRECT_REFERENCE",
    *MEANINGFUL,
    *RANDOM,
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != 200:
        raise RuntimeError(f"expected exactly 200 preserved rows, found {len(rows)}")
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate Gate 6.2 logical row")
    conditions = {str(row["condition"]) for row in rows}
    if conditions != EXPECTED_CONDITIONS:
        raise RuntimeError(f"unexpected condition set: {sorted(conditions)}")
    counts = Counter(str(row["condition"]) for row in rows)
    if set(counts.values()) != {20}:
        raise RuntimeError(f"expected 20 rows per condition, found {counts}")
    return rows


def _historically_truncated(row: dict[str, Any]) -> bool:
    """Preserve a historical truncation label instead of repairing it."""

    return str(row.get("status")) == "TRUNCATED"


def _v2_result(row: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_external_answer_v2(
        str(row.get("raw_output", "")), truncated=_historically_truncated(row)
    )
    status = parsed.status.value if parsed.status is not None else None
    correct = False
    evaluator_reason = None
    if status is None:
        try:
            correct = evaluate_external_answer(
                parsed.answer_text or "",
                str(row["reference_answer"]),
                str(row["evaluator"]),
            )
            status = (
                ExternalStatus.VALID_CORRECT.value if correct else ExternalStatus.VALID_WRONG.value
            )
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError) as exc:
            status = ExternalStatus.INVALID_FORMAT.value
            evaluator_reason = f"typed evaluator rejected answer: {type(exc).__name__}"
    return {
        "status": status,
        "parsed_answer": parsed.answer_text,
        "correct": bool(correct),
        "parse_reason": evaluator_reason or parsed.parse_reason,
    }


def _semantic_outcome(status: str, parsed_answer: str | None) -> str:
    if status in VALID:
        return f"VALID::{parsed_answer}"
    return f"MECHANICAL::{status}"


def _sequence_change(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return list(left.get("generated_token_ids", [])) != list(right.get("generated_token_ids", []))


def _first_divergence(left: dict[str, Any], right: dict[str, Any]) -> int | None:
    left_ids = list(left.get("generated_token_ids", []))
    right_ids = list(right.get("generated_token_ids", []))
    for index, (left_id, right_id) in enumerate(zip(left_ids, right_ids, strict=False)):
        if left_id != right_id:
            return index
    if len(left_ids) != len(right_ids):
        return min(len(left_ids), len(right_ids))
    return None


def disagreement_reason(row: dict[str, Any], v2: dict[str, Any]) -> str:
    original_status = str(row["status"])
    v2_status = str(v2["status"])
    original_answer = row.get("parsed_answer")
    v2_answer = v2.get("parsed_answer")
    if original_status == v2_status and original_answer == v2_answer:
        return "unchanged"
    if original_status != v2_status and v2_status in VALID:
        return "V2 accepted one unique FINAL commitment with allowed formatting"
    if original_status != v2_status:
        return f"V2 status changed {original_status} -> {v2_status}"
    if original_answer != v2_answer:
        return "V2 extracted a different final payload"
    return str(v2.get("parse_reason") or "V2 semantic reanalysis changed the row")


def summarize(
    condition: str,
    rows: list[dict[str, Any]],
    baseline_by_item: dict[str, dict[str, Any]],
    v2_by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    statuses = [
        str(v2_by_key[(str(row["item_id"]), condition, int(row["rollout_index"]))]["status"])
        for row in rows
    ]
    valid_count = sum(status in VALID for status in statuses)
    correct = sum(
        bool(v2_by_key[(str(row["item_id"]), condition, int(row["rollout_index"]))]["correct"])
        for row in rows
    )
    semantic_changes = []
    raw_changes = []
    divergence = []
    for row in rows:
        key = (str(row["item_id"]), condition, int(row["rollout_index"]))
        current_v2 = v2_by_key[key]
        base = baseline_by_item[str(row["item_id"])]
        base_v2 = v2_by_key[(str(base["item_id"]), "BASELINE", int(base["rollout_index"]))]
        semantic_changes.append(
            _semantic_outcome(str(base_v2["status"]), base_v2["parsed_answer"])
            != _semantic_outcome(str(current_v2["status"]), current_v2["parsed_answer"])
        )
        raw_changes.append(_sequence_change(base, row))
        divergence.append(_first_divergence(base, row))
    tokens = [int(row.get("generated_token_count", 0)) for row in rows]
    defined = [value for value in divergence if value is not None]
    original_statuses = Counter(str(row["status"]) for row in rows)
    return {
        "condition": condition,
        "n": len(rows),
        "valid": valid_count,
        "validity": valid_count / len(rows),
        "correct": correct,
        "wrong": sum(status == ExternalStatus.VALID_WRONG.value for status in statuses),
        "invalid_format": sum(status == ExternalStatus.INVALID_FORMAT.value for status in statuses),
        "truncated": sum(status == ExternalStatus.TRUNCATED_THINKING.value for status in statuses),
        "accuracy": correct / len(rows),
        "semantic_change_rate": sum(semantic_changes) / len(rows),
        "raw_token_sequence_change_rate": sum(raw_changes) / len(rows),
        "mean_first_divergence_token": statistics.mean(defined) if defined else None,
        "mean_tokens": statistics.mean(tokens),
        "median_tokens": statistics.median(tokens),
        "max_tokens": max(tokens),
        "original_statuses": dict(sorted(original_statuses.items())),
        "v2_statuses": dict(sorted(Counter(statuses).items())),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("review/gate6_2_first_stage_repair_mean_bridge/journal.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("review/gate6_3_single_mean_semantic_evaluation"),
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input.resolve())

    v2_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    row_reanalysis: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        v2 = _v2_result(row)
        v2_by_key[key] = v2
        row_reanalysis.append(
            {
                "item_id": str(row["item_id"]),
                "condition": str(row["condition"]),
                "rollout_index": int(row["rollout_index"]),
                "original_status": str(row["status"]),
                "v2_status": v2["status"],
                "original_parsed_answer": row.get("parsed_answer"),
                "v2_parsed_answer": v2.get("parsed_answer"),
                "reference_answer": str(row["reference_answer"]),
                "v2_correct": v2["correct"],
                "v2_parse_reason": v2.get("parse_reason"),
                "disagreement_reason": disagreement_reason(row, v2),
            }
        )
    _write_csv(
        output / "SEMANTIC_V2_ROW_REANALYSIS.csv",
        row_reanalysis,
        [
            "item_id",
            "condition",
            "rollout_index",
            "original_status",
            "v2_status",
            "original_parsed_answer",
            "v2_parsed_answer",
            "reference_answer",
            "v2_correct",
            "v2_parse_reason",
            "disagreement_reason",
        ],
    )

    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    baseline_by_item: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row["condition"])
        by_condition[condition].append(row)
        if condition == "BASELINE":
            baseline_by_item[str(row["item_id"])] = row
    if len(baseline_by_item) != 20:
        raise RuntimeError("baseline item index is not exactly 20 items")
    summaries = {
        condition: summarize(condition, by_condition[condition], baseline_by_item, v2_by_key)
        for condition in sorted(by_condition)
    }
    summary_rows = []
    for condition, row in summaries.items():
        summary_rows.append(
            {
                **row,
                "original_valid": sum(
                    str(value["status"]) in {"VALID_CORRECT", "VALID_WRONG"}
                    for value in by_condition[condition]
                ),
                "original_accuracy": sum(
                    bool(value["correct"]) for value in by_condition[condition]
                )
                / len(by_condition[condition]),
                "original_statuses": json.dumps(row["original_statuses"], sort_keys=True),
                "v2_statuses": json.dumps(row["v2_statuses"], sort_keys=True),
            }
        )
    _write_csv(
        output / "SEMANTIC_V2_CONDITION_SUMMARY.csv",
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
            "accuracy",
            "semantic_change_rate",
            "raw_token_sequence_change_rate",
            "mean_first_divergence_token",
            "mean_tokens",
            "median_tokens",
            "max_tokens",
            "original_valid",
            "original_accuracy",
            "original_statuses",
            "v2_statuses",
        ],
    )

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
            "meaningful_minus_random_max": row["semantic_change_rate"] - random_max_q,
            "textual_careful_q": summaries["TEXTUAL_CAREFUL_REFERENCE"]["semantic_change_rate"],
            "textual_direct_q": summaries["TEXTUAL_DIRECT_REFERENCE"]["semantic_change_rate"],
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

    all_passed = [condition for condition, gate in gates.items() if gate["pass"]]
    if gates["BEST_SINGLE_MEAN_PLUS"]["pass"]:
        classification = "GATE6_2A_PARSER_REANALYSIS_SINGLE_MEAN_PASS"
    elif all_passed:
        classification = "GATE6_2A_PARSER_REANALYSIS_AMBIGUOUS"
    else:
        classification = "GATE6_2A_PARSER_REANALYSIS_NO_CONTROLLER_PASS"

    estimands = {
        "parser_version": PARSER_VERSION,
        "input_journal": str(args.input.resolve()),
        "n_rows": len(rows),
        "n_items": len(baseline_by_item),
        "conditions": summaries,
        "random_mean_semantic_change_rate": random_mean_q,
        "random_max_semantic_change_rate": random_max_q,
        "gate_thresholds": {
            "validity_minimum": 0.85,
            "semantic_change_minimum": 0.15,
            "meaningful_minus_random_mean_minimum": 0.05,
        },
        "gate_checks": gates,
        "passed_meaningful_conditions": all_passed,
        "promotable_controller": "BEST_SINGLE_MEAN_PLUS",
        "promotable_controller_pass": gates["BEST_SINGLE_MEAN_PLUS"]["pass"],
        "classification": classification,
        "evaluation_authorized_by_reanalysis": gates["BEST_SINGLE_MEAN_PLUS"]["pass"],
    }
    (output / "SEMANTIC_V2_MANIPULATION_ESTIMANDS.json").write_text(
        json.dumps(estimands, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parser_spec = {
        "parser_version": PARSER_VERSION,
        "contract": "exactly one FINAL commitment; wrappers/fences only; no substantive suffix",
        "source_journal_sha256": hashlib.sha256(args.input.resolve().read_bytes()).hexdigest(),
        "row_count": len(rows),
        "status_counts_original": dict(sorted(Counter(str(row["status"]) for row in rows).items())),
        "status_counts_v2": dict(
            sorted(Counter(str(v2["status"]) for v2 in v2_by_key.values()).items())
        ),
        "historical_result_preserved": True,
    }
    (output / "parser_version.json").write_text(
        json.dumps(parser_spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    changed = [row for row in row_reanalysis if row["disagreement_reason"] != "unchanged"]
    lines = [
        "# Gate 6.3 — Semantic V2 forensic reanalysis",
        "",
        f"Parser: `{PARSER_VERSION}`.",
        "",
        "The historical Gate 6.2 journal and report are unchanged.  This report",
        "recomputes all 200 rows offline from raw outputs with one parser applied",
        "uniformly to every condition.",
        "",
        f"Rows reclassified or otherwise changed: **{len(changed)} / {len(rows)}**.",
        "",
        "## Condition summary",
        "",
        "| condition | V2 valid/20 | V2 correct | V2 wrong | V2 invalid | "
        "V2 truncated | V2 Q | gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for condition, row in summaries.items():
        gate = gates.get(condition, {}).get("pass", "reference/random")
        lines.append(
            f"| {condition} | {row['valid']}/20 | {row['correct']} | {row['wrong']} | "
            f"{row['invalid_format']} | {row['truncated']} | "
            f"{row['semantic_change_rate']:.3f} | {gate} |"
        )
    lines += [
        "",
        f"Random mean Q: `{random_mean_q:.6f}`; random maximum Q: `{random_max_q:.6f}`.",
        f"BEST_SINGLE_MEAN_PLUS gate: `{gates['BEST_SINGLE_MEAN_PLUS']['pass']}`.",
        "",
        f"## Offline classification: `{classification}`",
        "",
        "Only BEST_SINGLE_MEAN_PLUS is promotable by the authorized continuation.",
        "If it does not pass, no RunPod phase is authorized by this reanalysis.",
        "The original Gate 6.2 classification remains GATE6_2_NO_BEHAVIORAL_FIRST_STAGE.",
    ]
    (output / "SEMANTIC_V2_REANALYSIS_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"classification": classification, "changed_rows": len(changed), "gates": gates},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
