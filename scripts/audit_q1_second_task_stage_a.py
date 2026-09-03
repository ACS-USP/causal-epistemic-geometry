#!/usr/bin/env python3
"""Independent low-level forensic recomputation of Q1 second-task Stage A."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
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

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"


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


def score(row: dict[str, Any]) -> dict[str, bool]:
    commitment = extract_final_commitment(row["raw_output"], truncated=bool(row["truncated"]))
    if not commitment.valid or commitment.payload is None:
        return {"commitment_valid": False, "semantic_evaluable": False, "correct": False}
    try:
        actual = parse_payload(commitment.payload)
    except (ValueError, json.JSONDecodeError, TypeError):
        return {"commitment_valid": True, "semantic_evaluable": False, "correct": False}
    expected = canonical(json.loads(row["reference_answer"]))
    return {"commitment_valid": True, "semantic_evaluable": True, "correct": actual == expected}


def summarize(rows: list[dict[str, Any]], condition: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == condition]
    tokens = [row["generated_token_count"] for row in selected]
    return {
        "commitment_validity": sum(row["commitment_valid"] for row in selected) / len(selected),
        "semantic_evaluability": sum(row["semantic_evaluable"] for row in selected) / len(selected),
        "accuracy": sum(row["correct"] for row in selected) / len(selected),
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
    schedule = read_json(REVIEW / "STAGE_A_SCHEDULE.json")
    def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
        return (row["stage"], row["item_id"], row["condition"], row["rollout_index"])

    expected = {key(row): row for row in schedule}
    observed = {key(row): row for row in raw}
    if len(raw) != len(observed) or set(observed) != set(expected) or len(raw) != 128:
        raise RuntimeError("forensic schedule completeness failure")
    parsed = [{**row, **score(row)} for row in raw]
    baseline = summarize(parsed, "BASELINE")
    textual = summarize(parsed, "TEXTUAL_CAREFUL")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        if row["condition"] == "BASELINE":
            groups[row["family_id"]].append(row)
    wrong_both = sum(all(not row["correct"] for row in values) for values in groups.values())
    correct_once = sum(any(row["correct"] for row in values) for values in groups.values())
    b00 = wrong_both / 32
    accuracy_delta = textual["accuracy"] - baseline["accuracy"]
    mean_ratio = textual["mean_generated_tokens"] / baseline["mean_generated_tokens"]
    median_delta = textual["median_generated_tokens"] - baseline["median_generated_tokens"]
    manifestations = {
        "TEXTUAL_ACCURACY_GAIN_GE_0_03": accuracy_delta >= 0.03,
        "TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5": mean_ratio >= 1.5,
        "TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10": median_delta >= 10,
    }
    gates = {
        "baseline_validity": baseline["commitment_validity"] >= 0.95,
        "baseline_evaluability": baseline["semantic_evaluability"] >= 0.95,
        "baseline_accuracy": 0.25 <= baseline["accuracy"] <= 0.90,
        "baseline_B00": b00 >= 0.05,
        "baseline_wrong_both": wrong_both >= 4,
        "baseline_correct_once": correct_once >= 7,
        "textual_validity": textual["commitment_validity"] >= 0.95,
        "textual_evaluability": textual["semantic_evaluability"] >= 0.95,
        "textual_nonharm": textual["accuracy"] >= baseline["accuracy"] - 0.03,
        "textual_manifestation": any(manifestations.values()),
    }
    classification = (
        "Q1_SECOND_TASK_STAGE_A_QUALIFIED"
        if all(gates.values())
        else "Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED"
    )
    primary = read_json(args.analysis_dir / "PRIMARY_STAGE_A_RESULTS.json")
    audit_values = {
        "baseline.commitment_validity": baseline["commitment_validity"],
        "baseline.semantic_evaluability": baseline["semantic_evaluability"],
        "baseline.accuracy": baseline["accuracy"],
        "baseline.B00": b00,
        "textual_careful.commitment_validity": textual["commitment_validity"],
        "textual_careful.semantic_evaluability": textual["semantic_evaluability"],
        "textual_careful.accuracy": textual["accuracy"],
        "textual_careful.textual_accuracy_delta": accuracy_delta,
        "textual_careful.textual_mean_token_ratio": mean_ratio,
        "textual_careful.textual_median_token_delta": median_delta,
    }
    differences = {}
    for dotted, value in audit_values.items():
        section, field = dotted.split(".", 1)
        differences[dotted] = abs(float(primary[section][field]) - float(value))
    maximum_difference = max(differences.values(), default=0.0)
    clean = classification == primary["classification"] and maximum_difference <= 1e-12
    result = {
        "classification": (
            "Q1_SECOND_TASK_STAGE_A_FORENSIC_CLEAN"
            if clean
            else "Q1_SECOND_TASK_STAGE_A_FORENSIC_DISAGREEMENT"
        ),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "journal_sha256": sha256(journal),
        "schedule_coverage": 128,
        "duplicates": 0,
        "missing": 0,
        "families": 32,
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
    write_json(args.analysis_dir / "FORENSIC_AUDIT.json", result)
    if not clean:
        raise RuntimeError("Q1_SECOND_TASK_STAGE_A_FORENSIC_DISAGREEMENT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
