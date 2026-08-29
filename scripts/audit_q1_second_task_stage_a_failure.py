#!/usr/bin/env python3
"""Post-closeout diagnostic audit of the sealed Q1 second-task Stage A.

This script never changes the historical Stage-A classification.  It records
row-level structural classifications without copying raw outputs or benchmark
references into repository artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_a as stage_a  # noqa: E402
from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (  # noqa: E402
    TAXONOMY,
    classify_output,
    conservative_repair_candidate,
)

EXPECTED_JOURNAL_SHA256 = "5b0fec6960ac414f56995d91a43c3b41c49a06b5fb868156a8e24d037b9281b1"
EXPECTED_HISTORICAL = {
    "BASELINE": {"commitment_valid": 53, "semantic_evaluable": 52, "correct": 39},
    "TEXTUAL_CAREFUL": {
        "commitment_valid": 62,
        "semantic_evaluable": 61,
        "correct": 50,
    },
}
POSTHOC_LABEL = "POST_HOC_DIAGNOSTIC_NOT_STAGE_A1_RECLASSIFICATION"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_schedule(raw: list[dict[str, Any]], schedule_path: Path) -> None:
    schedule = read_json(schedule_path)
    stage_a.validate_schedule(schedule)
    expected = {stage_a.logical_key(row): row for row in schedule}
    observed = {stage_a.logical_key(row): row for row in raw}
    if len(raw) != 128 or len(observed) != 128 or set(observed) != set(expected):
        raise RuntimeError("sealed Stage-A journal does not match the 128-row schedule")
    for key, row in observed.items():
        locked = expected[key]
        for field in ("family_id", "item_id", "item_sha256", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"schedule mismatch for {field}: {key}")
        if row.get("intervention") != "NONE" or row.get("activation_hook_active"):
            raise RuntimeError("activation intervention found in Stage A")


def score_and_classify(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    scored = q1s.evaluate_livecodebench_output(
        row["raw_output"],
        row["reference_answer"],
        truncated=bool(row["truncated"]),
    )
    expected = q1s.parse_safe_literal(row["reference_answer"])
    expected_type = str(expected[0])
    frozen_type = None
    if scored["canonical_value"] is not None:
        frozen_type = str(json.loads(scored["canonical_value"])[0])
    classification = classify_output(
        row["raw_output"],
        row["generated_token_ids"],
        truncated=bool(row["truncated"]),
        frozen_commitment_valid=bool(scored["commitment_valid"]),
        frozen_evaluable=bool(scored["semantic_evaluable"]),
        frozen_value_type=frozen_type,
        expected_type=expected_type,
    )
    candidate = classification.candidate
    candidate_matches_reference = (
        candidate is not None and candidate.canonical_json == canonical_json(expected)
    )
    repair = conservative_repair_candidate(classification)
    if scored["semantic_evaluable"]:
        repaired_commitment = bool(scored["commitment_valid"])
        repaired_evaluable = True
        repaired_correct = bool(scored["correct"])
    elif repair is not None:
        repaired_commitment = True
        repaired_evaluable = True
        repaired_correct = repair.canonical_json == canonical_json(expected)
    else:
        repaired_commitment = bool(scored["commitment_valid"])
        repaired_evaluable = False
        repaired_correct = False
    public_row = {
        "condition": row["condition"],
        "family_id": row["family_id"],
        "item_id": row["item_id"],
        "rollout_index": int(row["rollout_index"]),
        "seed": int(row["seed"]),
        "output_sha256": row["output_sha256"],
        "generation_token_count": int(row["generated_token_count"]),
        "truncated": bool(row["truncated"]),
        "terminal_reason": scored["status"],
        "frozen_failure_reason": scored["failure_reason"],
        "frozen_commitment_valid": bool(scored["commitment_valid"]),
        "frozen_semantic_evaluable": bool(scored["semantic_evaluable"]),
        "frozen_correct": bool(scored["correct"]),
        "frozen_value_type": frozen_type,
        "expected_reference_type": expected_type,
        "exactly_one_mechanical_candidate": candidate is not None,
        "candidate_has_expected_type": candidate is not None
        and candidate.value_type == expected_type,
        "candidate_matches_reference": candidate_matches_reference,
        "recovery_requires_semantic_judgment": classification.requires_semantic_judgment,
        "failure_category": classification.category,
        "mechanically_repetitive": classification.mechanically_repetitive,
        "unfinished_reasoning": classification.unfinished,
        "candidate_parser_a_eligible": repair is not None,
        "candidate_parser_a_commitment_valid": repaired_commitment,
        "candidate_parser_a_semantic_evaluable": repaired_evaluable,
        "candidate_parser_a_correct": repaired_correct,
        "raw_output_persisted_only_in_sealed_journal": True,
    }
    private_row = {
        **public_row,
        "raw_output": row["raw_output"],
        "reference_answer": row["reference_answer"],
        "candidate_payload": candidate.payload if candidate is not None else None,
        "candidate_canonical_json": candidate.canonical_json if candidate is not None else None,
    }
    return public_row, private_row


def condition_summary(rows: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    selected = [row for row in rows if row["condition"] == condition]
    invalid = [row for row in selected if not row["frozen_semantic_evaluable"]]
    categories = Counter(row["failure_category"] for row in selected)
    old = {
        "commitment_valid": sum(row["frozen_commitment_valid"] for row in selected),
        "semantic_evaluable": sum(row["frozen_semantic_evaluable"] for row in selected),
        "correct": sum(row["frozen_correct"] for row in selected),
    }
    repaired = {
        "commitment_valid": sum(
            row["candidate_parser_a_commitment_valid"] for row in selected
        ),
        "semantic_evaluable": sum(
            row["candidate_parser_a_semantic_evaluable"] for row in selected
        ),
        "correct": sum(row["candidate_parser_a_correct"] for row in selected),
    }
    expected = EXPECTED_HISTORICAL[condition]
    if old != expected:
        raise RuntimeError(f"historical metric mismatch for {condition}: {old} != {expected}")
    n = len(selected)
    return {
        "rows": n,
        "valid_as_frozen": old["commitment_valid"],
        "evaluable_as_frozen": old["semantic_evaluable"],
        "correct_as_frozen": old["correct"],
        "invalid_or_nonevaluable": len(invalid),
        "failure_categories": {category: categories.get(category, 0) for category in TAXONOMY},
        "token_cap": sum(row["truncated"] for row in selected),
        "mechanical_repetition": sum(row["mechanically_repetitive"] for row in selected),
        "unique_mechanically_recoverable_invalid": sum(
            row["exactly_one_mechanical_candidate"] for row in invalid
        ),
        "candidate_parser_a_eligible_invalid": sum(
            row["candidate_parser_a_eligible"] for row in invalid
        ),
        "semantically_ambiguous_invalid": sum(
            row["failure_category"] in {"AMBIGUOUS_OUTPUT", "REFERENCE_OR_PROMPT_AMBIGUITY"}
            for row in invalid
        ),
        "unrecoverable_invalid": sum(
            not row["exactly_one_mechanical_candidate"] for row in invalid
        ),
        "historical": {
            **old,
            "commitment_validity": old["commitment_valid"] / n,
            "semantic_evaluability": old["semantic_evaluable"] / n,
            "accuracy": old["correct"] / n,
        },
        "candidate_parser_a": {
            **repaired,
            "commitment_validity": repaired["commitment_valid"] / n,
            "semantic_evaluability": repaired["semantic_evaluable"] / n,
            "accuracy": repaired["correct"] / n,
            "crosses_commitment_threshold_0_95": repaired["commitment_valid"] / n >= 0.95,
            "crosses_evaluability_threshold_0_95": repaired["semantic_evaluable"] / n
            >= 0.95,
        },
    }


def decompose_correct_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {
        (row["family_id"], row["rollout_index"], row["condition"]): row for row in rows
    }
    both_evaluable = 0
    invalid_union = 0
    pair_count = 0
    both_evaluable_pairs = 0
    invalid_union_pairs = 0
    for family_id, rollout in sorted(
        {(row["family_id"], row["rollout_index"]) for row in rows}
    ):
        baseline = by_key[(family_id, rollout, "BASELINE")]
        textual = by_key[(family_id, rollout, "TEXTUAL_CAREFUL")]
        delta = int(textual["frozen_correct"]) - int(baseline["frozen_correct"])
        pair_count += 1
        if baseline["frozen_semantic_evaluable"] and textual["frozen_semantic_evaluable"]:
            both_evaluable += delta
            both_evaluable_pairs += 1
        else:
            invalid_union += delta
            invalid_union_pairs += 1
    total = both_evaluable + invalid_union
    if total != 11 or pair_count != 64:
        raise RuntimeError("paired correct-row decomposition failed to recover +11")
    return {
        "label": POSTHOC_LABEL,
        "matched_family_rollout_pairs": pair_count,
        "total_correct_row_difference_textual_minus_baseline": total,
        "fractional_accuracy_difference": total / 64,
        "pairs_where_both_conditions_evaluable": both_evaluable_pairs,
        "correct_row_difference_among_both_evaluable": both_evaluable,
        "pairs_with_at_least_one_invalid_or_nonevaluable": invalid_union_pairs,
        "correct_row_difference_associated_with_invalidity_union": invalid_union,
        "unresolved_contribution": total - both_evaluable - invalid_union,
        "conditional_on_valid_is_not_primary": True,
        "new_hypothesis_test": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--public-output-dir", type=Path, required=True)
    parser.add_argument("--private-output-dir", type=Path, required=True)
    args = parser.parse_args()
    observed_hash = sha256(args.journal)
    if observed_hash != EXPECTED_JOURNAL_SHA256:
        raise RuntimeError(f"journal SHA mismatch: {observed_hash}")
    raw = read_jsonl(args.journal)
    validate_schedule(raw, args.schedule)
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for row in raw:
        public, private = score_and_classify(row)
        public_rows.append(public)
        private_rows.append(private)
    public_rows.sort(
        key=lambda row: (row["condition"], row["family_id"], row["rollout_index"])
    )
    private_rows.sort(
        key=lambda row: (row["condition"], row["family_id"], row["rollout_index"])
    )
    summaries = {
        condition: condition_summary(public_rows, condition)
        for condition in ("BASELINE", "TEXTUAL_CAREFUL")
    }
    decomposition = decompose_correct_delta(public_rows)
    write_jsonl(args.public_output_dir / "ROW_FAILURE_AUDIT.jsonl", public_rows)
    write_jsonl(args.private_output_dir / "ROW_FAILURE_AUDIT_WITH_RAW.jsonl", private_rows)
    write_json(
        args.public_output_dir / "FAILURE_TAXONOMY_SUMMARY.json",
        {
            "schema_version": 1,
            "status": POSTHOC_LABEL,
            "journal_sha256": observed_hash,
            "historical_classification": "Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED",
            "historical_classification_modified": False,
            "conditions": summaries,
            "raw_outputs_committed": False,
            "private_row_audit_sha256": sha256(
                args.private_output_dir / "ROW_FAILURE_AUDIT_WITH_RAW.jsonl"
            ),
        },
    )
    write_json(
        args.public_output_dir / "SEMANTIC_ANSWER_CHANNEL_DECOMPOSITION.json",
        decomposition,
    )
    write_json(
        args.public_output_dir / "POSTHOC_PARSER_A_RESCORE.json",
        {
            "status": POSTHOC_LABEL,
            "historical_classification": "Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED",
            "historical_classification_modified": False,
            "conditions": {
                condition: {
                    "historical": summary["historical"],
                    "candidate_parser_a": summary["candidate_parser_a"],
                }
                for condition, summary in summaries.items()
            },
            "selection_used_correctness": False,
            "condition_symmetric": True,
            "fuzzy_or_semantic_matching": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
