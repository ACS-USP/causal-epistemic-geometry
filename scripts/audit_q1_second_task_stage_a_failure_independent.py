#!/usr/bin/env python3
"""Independent low-level crosscheck of the Stage-A1 failure audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.base import _visible_text  # noqa: E402
from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    _marker,
    extract_final_commitment,
)

EXPECTED_SHA = "5b0fec6960ac414f56995d91a43c3b41c49a06b5fb868156a8e24d037b9281b1"
FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)```", re.DOTALL)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> Any:
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [canonical(entry) for entry in value]]
    if value is None:
        return ["none", None]
    if isinstance(value, float):
        return ["float", value]
    raise ValueError(type(value).__name__)


def parse(value: str) -> Any:
    try:
        parsed = ast.literal_eval(value.strip())
    except (SyntaxError, ValueError, TypeError):
        parsed = json.loads(value)
    return canonical(parsed)


def frozen_score(row: dict[str, Any]) -> dict[str, Any]:
    commitment = extract_final_commitment(row["raw_output"], truncated=row["truncated"])
    if not commitment.valid or commitment.payload is None:
        return {"commitment": False, "evaluable": False, "correct": False}
    try:
        actual = parse(commitment.payload)
    except (ValueError, json.JSONDecodeError, TypeError):
        return {"commitment": True, "evaluable": False, "correct": False}
    expected = canonical(json.loads(row["reference_answer"]))
    return {"commitment": True, "evaluable": True, "correct": actual == expected}


def repair_candidate(row: dict[str, Any]) -> Any | None:
    if row["truncated"]:
        return None
    visible, unclosed = _visible_text(row["raw_output"])
    if unclosed:
        return None
    lines = visible.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    markers = [
        (index, marker)
        for index, line in enumerate(lines)
        if (marker := _marker(line)) is not None
    ]
    if len(markers) < 2 or not nonempty:
        return None
    terminal_index, terminal = markers[-1]
    kind, payload, closer = terminal
    if terminal_index != nonempty[-1] or kind != "FINAL" or closer or not payload.strip():
        return None
    try:
        candidate = parse(payload)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
    expected = canonical(json.loads(row["reference_answer"]))
    if candidate[0] != expected[0]:
        return None
    if any(
        prior_kind not in {"FINAL_ANSWER", "FINAL_SECTION"} or prior_payload.strip()
        for _index, (prior_kind, prior_payload, _closer) in markers[:-1]
    ):
        return None
    outside: list[Any] = []
    for line in lines:
        if _marker(line) is not None or line.strip().startswith("```") or not line.strip():
            continue
        try:
            outside.append(parse(line.strip().strip("`")))
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
    for match in FENCE.finditer(visible):
        body = match.group("body").strip()
        try:
            outside.append(parse(body))
        except (ValueError, json.JSONDecodeError, TypeError):
            pass
    if any(value != candidate for value in outside):
        return None
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--primary-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.journal) != EXPECTED_SHA:
        raise RuntimeError("journal hash mismatch")
    raw = [json.loads(line) for line in args.journal.read_text().splitlines() if line]
    primary = [
        json.loads(line) for line in args.primary_rows.read_text().splitlines() if line
    ]
    by_key = {
        (row["condition"], row["family_id"], row["rollout_index"]): row for row in primary
    }
    summaries: dict[str, Any] = {}
    discrepancies = 0
    for condition in ("BASELINE", "TEXTUAL_CAREFUL"):
        selected = [row for row in raw if row["condition"] == condition]
        frozen = [frozen_score(row) for row in selected]
        repaired = []
        eligible = 0
        for row, score in zip(selected, frozen, strict=True):
            candidate = None if score["evaluable"] else repair_candidate(row)
            if candidate is not None:
                eligible += 1
                expected = canonical(json.loads(row["reference_answer"]))
                value = {"commitment": True, "evaluable": True, "correct": candidate == expected}
            else:
                value = score
            repaired.append(value)
            public = by_key[(condition, row["family_id"], row["rollout_index"])]
            discrepancies += int(public["candidate_parser_a2_eligible"] != (candidate is not None))
            discrepancies += int(public["candidate_parser_a2_correct"] != value["correct"])
        summaries[condition] = {
            "rows": len(selected),
            "frozen": {
                "commitment_valid": sum(row["commitment"] for row in frozen),
                "semantic_evaluable": sum(row["evaluable"] for row in frozen),
                "correct": sum(row["correct"] for row in frozen),
            },
            "repair_eligible_invalid": eligible,
            "repaired": {
                "commitment_valid": sum(row["commitment"] for row in repaired),
                "semantic_evaluable": sum(row["evaluable"] for row in repaired),
                "correct": sum(row["correct"] for row in repaired),
            },
        }
    result = {
        "classification": (
            "Q1_SECOND_TASK_STAGE_A_FAILURE_AUDIT_FORENSIC_CLEAN"
            if discrepancies == 0
            else "Q1_SECOND_TASK_STAGE_A_FAILURE_AUDIT_FORENSIC_DISAGREEMENT"
        ),
        "journal_sha256": EXPECTED_SHA,
        "rows": len(raw),
        "conditions": summaries,
        "primary_row_field_discrepancies": discrepancies,
        "raw_output_reproduced_in_artifact": False,
        "historical_stage_a1_classification_modified": False,
        "new_model_inference": 0,
        "q2_outputs_inspected": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if discrepancies:
        raise RuntimeError(result["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
