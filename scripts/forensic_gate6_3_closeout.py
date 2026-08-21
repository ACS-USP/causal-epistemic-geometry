#!/usr/bin/env python3
"""Independent offline forensic closeout for Gate 6.3.

This intentionally does not call the Gate 6.3 high-level analyzer or the
canonical estimand helper.  It validates the frozen schedules and recomputes
the primary two-rollout quantities directly from the raw journal.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.benchmarks.external.base import ExternalStatus, evaluate_external_answer
from epistemic_geometry.benchmarks.external.semantic_v2 import parse_external_answer_v2

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/gate6_3_single_mean_semantic_evaluation"
JOURNAL = REVIEW / "journal.jsonl"
MATCHED = REVIEW / "MATCHED_RANDOM_SCHEDULE.json"
EVALUATION = REVIEW / "EVALUATION_SCHEDULE.json"
LOCK = REVIEW / "PROTOCOL_LOCK.json"
VALID = {ExternalStatus.VALID_CORRECT.value, ExternalStatus.VALID_WRONG.value}
RANDOM = tuple(f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
EVAL_CONDITIONS = ("BASELINE", "TEXTUAL_CAREFUL_REFERENCE", "BEST_SINGLE_MEAN_PLUS", *RANDOM)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["phase"]),
        str(row["item_id"]),
        str(row["condition"]),
        int(row["rollout_index"]),
    )


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    raw = str(row.get("raw_output", ""))
    token_count = int(row.get("generated_token_count", 0))
    truncated = str(row.get("status")) == "TRUNCATED" or token_count >= 4096
    parsed = parse_external_answer_v2(raw, truncated=truncated)
    if parsed.status is not None:
        return {
            "status": parsed.status.value.replace("TRUNCATED_THINKING", "TRUNCATED"),
            "answer": parsed.answer_text,
            "correct": False,
        }
    try:
        correct = evaluate_external_answer(
            parsed.answer_text or "", str(row["reference_answer"]), str(row["evaluator"])
        )
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
        return {"status": "INVALID_FORMAT", "answer": parsed.answer_text, "correct": False}
    return {
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "answer": parsed.answer_text,
        "correct": bool(correct),
    }


def errors(
    rows: dict[tuple[str, int], dict[str, Any]], items: list[str], condition: str
) -> list[float]:
    result = []
    for item in items:
        first = rows[(item, 0)]
        second = rows[(item, 1)]
        result.extend(
            [
                0.0 if first["status"] == "VALID_CORRECT" else 1.0,
                0.0 if second["status"] == "VALID_CORRECT" else 1.0,
            ]
        )
    return result


def item_errors(
    rows: dict[tuple[str, int], dict[str, Any]], items: list[str]
) -> list[tuple[float, float]]:
    return [
        (
            0.0 if rows[(item, 0)]["status"] == "VALID_CORRECT" else 1.0,
            0.0 if rows[(item, 1)]["status"] == "VALID_CORRECT" else 1.0,
        )
        for item in items
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def estimands(
    base: list[tuple[float, float]], condition: list[tuple[float, float]]
) -> dict[str, float]:
    n = len(base)
    b00 = mean([a * b for a, b in base])
    b0j = mean(
        [
            (a * c + a * d + b * c + b * d) / 4.0
            for (a, b), (c, d) in zip(base, condition, strict=True)
        ]
    )
    q0 = [(a + b) / 2.0 for a, b in base]
    qj = [(a + b) / 2.0 for a, b in condition]
    cross0 = [q0[i] * q0[j] for i in range(n) for j in range(n) if i != j]
    crossj = [q0[i] * qj[j] for i in range(n) for j in range(n) if i != j]
    u00 = mean(cross0)
    u0j = mean(crossj)
    d = mean(
        [
            a * b + c * d - a * d - b * c
            for (a, b), (c, d) in zip(base, condition, strict=True)
        ]
    )
    rescue = mean(
        [
            (a * (1 - c) + a * (1 - d) + b * (1 - c) + b * (1 - d)) / 4.0
            for (a, b), (c, d) in zip(base, condition, strict=True)
        ]
    )
    damage = mean(
        [
            ((1 - a) * c + (1 - a) * d + (1 - b) * c + (1 - b) * d) / 4.0
            for (a, b), (c, d) in zip(base, condition, strict=True)
        ]
    )
    return {
        "B00": b00,
        "B0j": b0j,
        "O00": 1.0 - b00,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": d,
        "rescue": rescue,
        "damage": damage,
    }


def closeout() -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in JOURNAL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matched_schedule = load_json(MATCHED)
    evaluation_schedule = load_json(EVALUATION)
    expected = matched_schedule + evaluation_schedule
    actual_keys = [key(row) for row in rows]
    expected_keys = [
        (str(row["phase"]), str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        for row in expected
    ]
    schedule_exact = Counter(actual_keys) == Counter(expected_keys)
    parsed = {key(row): parse_row(row) for row in rows}
    recorded_reanalysis_mismatches = [
        key(row)
        for row in rows
        if parsed[key(row)]["status"] != str(row["status"])
        or parsed[key(row)]["answer"] != row.get("parsed_answer")
        or parsed[key(row)]["correct"] != bool(row.get("correct", False))
    ]

    matched_rows = [row for row in rows if row["phase"] == "GATE6_3_MATCHED_RANDOM_SUPPLEMENT"]
    evaluation_rows = [row for row in rows if row["phase"] == "GATE6_3_PRIMARY_EVALUATION"]
    matched_items = sorted({str(row["item_id"]) for row in matched_rows})
    evaluation_items = sorted({str(row["item_id"]) for row in evaluation_rows})
    by_eval: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for condition in EVAL_CONDITIONS:
        by_eval[condition] = {
            (str(row["item_id"]), int(row["rollout_index"])): parsed[key(row)]
            for row in evaluation_rows
            if row["condition"] == condition
        }
    baseline = item_errors(by_eval["BASELINE"], evaluation_items)
    independent = {
        condition: estimands(baseline, item_errors(by_eval[condition], evaluation_items))
        for condition in EVAL_CONDITIONS[1:]
    }
    independent["BASELINE"] = {
        "B00": mean([a * b for a, b in baseline]),
        "O00": 1.0 - mean([a * b for a, b in baseline]),
    }
    main_estimands = load_json(REVIEW / "ESTIMANDS.json").get("estimands", {})
    estimator_mismatches = []
    for condition, values in independent.items():
        recorded = main_estimands.get(condition, {})
        for metric, value in values.items():
            if metric in recorded and not math.isclose(
                float(value), float(recorded[metric]), rel_tol=0.0, abs_tol=1e-12
            ):
                estimator_mismatches.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "independent": value,
                        "recorded": recorded[metric],
                    }
                )
    lock = load_json(LOCK)
    metadata = [row.get("metadata", {}).get("stop_metadata", {}) for row in rows]
    model_revisions = sorted({str(meta.get("model_revision")) for meta in metadata})
    alphas = sorted({float(meta["eta0"]) for meta in metadata if "eta0" in meta})
    parser_versions = sorted({str(row.get("parser_version")) for row in rows})
    intervention_metadata = {
        condition: sorted(
            {
                str(row.get("metadata", {}).get("stop_metadata", {}).get("intervention"))
                for row in rows
                if str(row["condition"]) == condition
            }
        )
        for condition in sorted({str(row["condition"]) for row in rows})
    }
    runtime_commits = Counter(str(row.get("runtime_source_commit")) for row in rows)
    seed_groups: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for row in matched_rows:
        seed_groups[str(row["item_id"])][str(row["condition"])].add(int(row["seed"]))
    matched_seed_coupling = all(
        len(set().union(*condition_seeds.values())) == 1 for condition_seeds in seed_groups.values()
    )
    evaluation_seed_unique = len({int(row["seed"]) for row in evaluation_rows}) == len(
        evaluation_rows
    )
    evaluation_item_condition_rollouts = all(
        len(
            [
                row
                for row in evaluation_rows
                if row["item_id"] == item and row["condition"] == condition
            ]
        )
        == 2
        for item in evaluation_items
        for condition in EVAL_CONDITIONS
    )
    vector_hashes = {
        condition: sorted(
            {
                str(value)
                for row in rows
                if row["condition"] == condition
                for value in row.get("metadata", {})
                .get("stop_metadata", {})
                .get("intervention_vector_hashes", {})
                .values()
            }
        )
        for condition in sorted({str(row["condition"]) for row in rows})
    }
    forward_count_failures = [
        key(row)
        for row in rows
        if row["condition"] in (*RANDOM, "BEST_SINGLE_MEAN_PLUS")
        and row.get("metadata", {})
        .get("stop_metadata", {})
        .get("intervention_forward_trace", {})
        .get("forward_count", 0)
        != len(
            row.get("metadata", {})
            .get("stop_metadata", {})
            .get("intervention_forward_trace", {})
            .get("applications", [])
        )
    ]
    bootstrap = load_json(REVIEW / "BOOTSTRAP_INTERVALS.json")
    closeout = {
        "classification": "GATE6_3_FORENSIC_CLEAN",
        "raw_integrity": {
            "journal_sha256": sha256(JOURNAL),
            "rows": len(rows),
            "expected_rows": 920,
            "unique_logical_rows": len(set(actual_keys)),
            "duplicate_logical_rows": len(actual_keys) - len(set(actual_keys)),
            "schedule_exact": schedule_exact,
            "matched_rows": len(matched_rows),
            "evaluation_rows": len(evaluation_rows),
            "matched_items": len(matched_items),
            "evaluation_items": len(evaluation_items),
            "matched_evaluation_item_overlap": sorted(set(matched_items) & set(evaluation_items)),
        },
        "provenance": {
            "model_revisions": model_revisions,
            "expected_model_revision": lock["model"]["revision"],
            "parser_versions": parser_versions,
            "expected_parser": lock["parser"]["version"],
            "eta0_values": alphas,
            "expected_eta0": lock["controller"]["eta0"],
            "runtime_source_commits": dict(runtime_commits),
            "protocol_lock_commit": lock["source_commit"],
            "vector_hashes_by_condition": vector_hashes,
            "intervention_metadata_by_condition": intervention_metadata,
        },
        "seed_and_schedule": {
            "matched_seed_coupling_exact": matched_seed_coupling,
            "evaluation_seed_values_unique": evaluation_seed_unique,
            "evaluation_two_rollouts_per_item_condition": evaluation_item_condition_rollouts,
            "seed_regimes": dict(Counter(str(row["seed_regime"]) for row in rows)),
            "parser_reanalysis_mismatches": len(recorded_reanalysis_mismatches),
            "forward_count_metadata_failures": len(forward_count_failures),
        },
        "independent_estimands": independent,
        "comparison": {
            "main_estimator_file": "ESTIMANDS.json",
            "primary_values_recomputed_independently": True,
            "main_estimator_mismatches": estimator_mismatches,
            "bootstrap_entries": len(bootstrap),
            "bootstrap_resamples": sorted(
                {int(value["resamples"]) for value in bootstrap.values()}
            ),
        },
    }
    if not (
        len(rows) == 920
        and schedule_exact
        and len(set(actual_keys)) == 920
        and not closeout["raw_integrity"]["matched_evaluation_item_overlap"]
        and model_revisions == [lock["model"]["revision"]]
        and parser_versions == [lock["parser"]["version"]]
        and len(recorded_reanalysis_mismatches) == 0
        and matched_seed_coupling
        and evaluation_seed_unique
        and evaluation_item_condition_rollouts
        and not forward_count_failures
        and not estimator_mismatches
    ):
        closeout["classification"] = "GATE6_3_FORENSIC_INTEGRITY_CONCERN"
    return closeout


if __name__ == "__main__":
    result = closeout()
    (REVIEW / "FORENSIC_CLOSEOUT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
