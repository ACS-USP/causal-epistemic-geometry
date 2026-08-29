#!/usr/bin/env python3
"""Independent low-level forensic recomputation of Q1 second-task Stage A2."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    extract_final_commitment,
)

AUDIT = (
    ROOT
    / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
    / "stage_a_failure_audit"
)
SCHEDULE = AUDIT / "STAGE_A2_SCHEDULE.json"
FINAL = re.compile(r"^\s*(?:#{1,6}\s*)?(?:\*\*)?final\s*:\s*(.*?)\s*(?:\*\*)?\s*$", re.I)
EMPTY_FINAL = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?final(?:\s+answer)?\s*:\s*(?:\*\*)?\s*$",
    re.I,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def canonical(value: Any) -> Any:
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, list):
        return ["list", [canonical(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [canonical(item) for item in value]]
    if isinstance(value, dict):
        records = [[canonical(key), canonical(item)] for key, item in value.items()]
        return ["dict", sorted(records, key=lambda record: json.dumps(record[0], sort_keys=True))]
    if isinstance(value, set):
        records = [canonical(item) for item in value]
        return ["set", sorted(records, key=lambda record: json.dumps(record, sort_keys=True))]
    raise ValueError(type(value).__name__)


def parse_payload(payload: str) -> Any:
    try:
        return canonical(ast.literal_eval(payload))
    except (SyntaxError, ValueError, TypeError):
        return canonical(json.loads(payload))


def visible_text(raw: str) -> tuple[str, bool]:
    lowered = raw.lower()
    if "<think>" not in lowered:
        return raw, False
    pieces: list[str] = []
    cursor = 0
    while True:
        start = lowered.find("<think>", cursor)
        if start < 0:
            pieces.append(raw[cursor:])
            return "".join(pieces), False
        pieces.append(raw[cursor:start])
        end = lowered.find("</think>", start + 7)
        if end < 0:
            return "".join(pieces), True
        cursor = end + 8


def direct_final_payload(raw: str, truncated: bool) -> str | None:
    commitment = extract_final_commitment(raw, truncated=truncated)
    return commitment.payload if commitment.valid else None


def repair_payload(raw: str, truncated: bool) -> str | None:
    visible, unclosed = visible_text(raw)
    if truncated or unclosed:
        return None
    lines = visible.splitlines()
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    markers = [
        index
        for index, line in enumerate(lines)
        if EMPTY_FINAL.fullmatch(line) or FINAL.fullmatch(line)
    ]
    if len(markers) < 2 or not nonempty or markers[-1] != nonempty[-1]:
        return None
    terminal_match = FINAL.fullmatch(lines[markers[-1]])
    if terminal_match is None or not terminal_match.group(1).strip():
        return None
    if any(not EMPTY_FINAL.fullmatch(lines[index]) for index in markers[:-1]):
        return None
    payload = terminal_match.group(1).strip()
    try:
        terminal = parse_payload(payload)
    except (ValueError, json.JSONDecodeError, TypeError):
        return None
    # Fail closed if any other standalone Python/JSON literal competes.
    for index, line in enumerate(lines):
        candidate = line.strip().strip("`").strip()
        if not candidate or index == markers[-1] or EMPTY_FINAL.fullmatch(line):
            continue
        try:
            other = parse_payload(candidate)
        except (ValueError, json.JSONDecodeError, TypeError):
            continue
        if other != terminal:
            return None
    return payload


def score(row: dict[str, Any]) -> dict[str, bool]:
    payload = direct_final_payload(row["raw_output"], bool(row["truncated"]))
    if payload is None:
        payload = repair_payload(row["raw_output"], bool(row["truncated"]))
    if payload is None:
        return {"commitment_valid": False, "semantic_evaluable": False, "correct": False}
    try:
        actual = parse_payload(payload)
    except (ValueError, json.JSONDecodeError, TypeError):
        return {"commitment_valid": True, "semantic_evaluable": False, "correct": False}
    expected = canonical(json.loads(row["reference_answer"]))
    if actual[0] != expected[0]:
        return {"commitment_valid": False, "semantic_evaluable": False, "correct": False}
    return {"commitment_valid": True, "semantic_evaluable": True, "correct": actual == expected}


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == condition]
    tokens = [int(row["generated_token_count"]) for row in selected]
    return {
        "commitment_validity": sum(bool(row["commitment_valid"]) for row in selected)
        / len(selected),
        "semantic_evaluability": sum(bool(row["semantic_evaluable"]) for row in selected)
        / len(selected),
        "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
        "mean_generated_tokens": sum(tokens) / len(tokens),
        "median_generated_tokens": float(median(tokens)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    journal = args.raw_dir / "journal.jsonl"
    raw = read_jsonl(journal)
    schedule = read_json(SCHEDULE)

    def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
        return (row["stage"], row["family_id"], row["condition"], int(row["rollout_index"]))

    expected = {key(row): row for row in schedule}
    observed = {key(row): row for row in raw}
    if len(raw) != 80 or len(observed) != 80 or set(observed) != set(expected):
        raise RuntimeError("Stage-A2 forensic schedule completeness failure")
    for logical_key, row in observed.items():
        locked = expected[logical_key]
        for field in ("family_id", "item_id", "item_sha256", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"Stage-A2 forensic lock mismatch: {field}")
    parsed = [{**row, **score(row)} for row in raw]
    baseline = summarize(parsed, "BASELINE")
    textual = summarize(parsed, "TEXTUAL_CAREFUL")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        if row["condition"] == "BASELINE":
            groups[row["family_id"]].append(row)
    wrong_both = sum(all(not row["correct"] for row in values) for values in groups.values())
    correct_once = sum(any(row["correct"] for row in values) for values in groups.values())
    b00 = wrong_both / 20
    delta = textual["accuracy"] - baseline["accuracy"]
    mean_ratio = textual["mean_generated_tokens"] / baseline["mean_generated_tokens"]
    median_delta = textual["median_generated_tokens"] - baseline["median_generated_tokens"]
    manifestations = {
        "TEXTUAL_ACCURACY_GAIN_GE_0_03": delta >= 0.03,
        "TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5": mean_ratio >= 1.5,
        "TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10": median_delta >= 10,
    }
    gates = {
        "baseline_commitment_validity": baseline["commitment_validity"] >= 0.95,
        "baseline_semantic_evaluability": baseline["semantic_evaluability"] >= 0.95,
        "baseline_accuracy": 0.25 <= baseline["accuracy"] <= 0.90,
        "baseline_B00": b00 >= 0.05,
        "baseline_wrong_both": wrong_both >= 2,
        "baseline_correct_once": correct_once >= 4,
        "textual_commitment_validity": textual["commitment_validity"] >= 0.95,
        "textual_semantic_evaluability": textual["semantic_evaluability"] >= 0.95,
        "textual_nonharm": textual["accuracy"] >= baseline["accuracy"] - 0.03,
        "textual_manifestation": any(manifestations.values()),
    }
    classification = (
        "Q1_SECOND_TASK_STAGE_A2_QUALIFIED"
        if all(gates.values())
        else "Q1_SECOND_TASK_STAGE_A2_NOT_QUALIFIED"
    )
    primary = read_json(args.analysis_dir / "PRIMARY_STAGE_A2_RESULTS.json")
    audit_values = {
        "baseline.commitment_validity": baseline["commitment_validity"],
        "baseline.semantic_evaluability": baseline["semantic_evaluability"],
        "baseline.accuracy": baseline["accuracy"],
        "baseline.B00": b00,
        "baseline.families_wrong_both_rollouts": float(wrong_both),
        "baseline.families_correct_at_least_once": float(correct_once),
        "textual_careful.commitment_validity": textual["commitment_validity"],
        "textual_careful.semantic_evaluability": textual["semantic_evaluability"],
        "textual_careful.accuracy": textual["accuracy"],
        "textual_careful.textual_accuracy_delta": delta,
        "textual_careful.textual_mean_token_ratio": mean_ratio,
        "textual_careful.textual_median_token_delta": median_delta,
    }
    differences = {}
    for dotted, value in audit_values.items():
        section, field = dotted.split(".", 1)
        differences[dotted] = abs(float(primary[section][field]) - float(value))
    maximum_difference = max(differences.values(), default=0.0)
    clean = (
        classification == primary["classification"]
        and manifestations == primary["textual_careful"]["manifestations"]
        and gates == primary["gates"]
        and maximum_difference <= 1e-12
    )
    result = {
        "classification": (
            "Q1_SECOND_TASK_STAGE_A2_FORENSIC_CLEAN"
            if clean
            else "Q1_SECOND_TASK_STAGE_A2_FORENSIC_DISAGREEMENT"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "journal_sha256": sha256(journal),
        "schedule_coverage": 80,
        "duplicates": 0,
        "missing": 0,
        "families": 20,
        "selected_rows_exact": True,
        "seeds_exact": all(
            observed[value]["seed"] == row["seed"] for value, row in expected.items()
        ),
        "primary_classification": primary["classification"],
        "audit_classification": classification,
        "maximum_metric_difference": maximum_difference,
        "metric_differences": differences,
        "audit_gate_booleans": gates,
        "audit_manifestations": manifestations,
        "meaningful_controller_livecodebench_trajectories": 0,
        "activation_null_livecodebench_trajectories": 0,
    }
    write_json(args.analysis_dir / "FORENSIC_AUDIT_STAGE_A2.json", result)
    if not clean:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A2_FORENSIC_DISAGREEMENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
