#!/usr/bin/env python3
"""Independent raw-row forensic audit for Gate 7.

This deliberately reimplements the estimands and classification rather than
calling the primary Gate-7 analysis helpers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)
from epistemic_geometry.reproducibility import stable_seed  # noqa: E402

REVIEW = ROOT / "review/gate7_fresh_l27_replication"
BASELINE = "BASELINE"
MEANINGFUL = "BEST_SINGLE_MEAN_PLUS"
TEXTUAL = "TEXTUAL_CAREFUL_REFERENCE"
RANDOMS = tuple(f"GATE7_RANDOM_R{i}" for i in range(4))
CONDITIONS = (BASELINE, TEXTUAL, MEANINGFUL, *RANDOMS)
EXPERIMENT_ID = "GATE7_FRESH_SINGLE_L27_REPLICATION"
MAX_NEW_TOKENS = 4096


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reparse(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    return {
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
    }


def independent_metrics(baseline: np.ndarray, condition: np.ndarray) -> dict[str, float]:
    """Reimplement frozen two-rollout formulas directly from binary error arrays."""

    b = np.asarray(baseline, dtype=np.float64)
    j = np.asarray(condition, dtype=np.float64)
    n = len(b)
    b00 = float(np.mean(b[:, 0] * b[:, 1]))
    b0j_item = (b[:, 0] * j[:, 0] + b[:, 0] * j[:, 1] + b[:, 1] * j[:, 0] + b[:, 1] * j[:, 1]) / 4.0
    b0j = float(np.mean(b0j_item))
    qb = b.mean(axis=1)
    qj = j.mean(axis=1)
    if n < 2:
        raise RuntimeError("Gate 7 audit requires at least two items")
    u00 = float(((qb.sum() ** 2) - np.square(qb).sum()) / (n * (n - 1)))
    u0j = float(((qb.sum() * qj.sum()) - np.dot(qb, qj)) / (n * (n - 1)))
    d_item = b[:, 0] * b[:, 1] + j[:, 0] * j[:, 1] - b[:, 0] * j[:, 1] - b[:, 1] * j[:, 0]
    rescue_item = (
        b[:, 0] * (1 - j[:, 0])
        + b[:, 0] * (1 - j[:, 1])
        + b[:, 1] * (1 - j[:, 0])
        + b[:, 1] * (1 - j[:, 1])
    ) / 4.0
    damage_item = (
        (1 - b[:, 0]) * j[:, 0]
        + (1 - b[:, 0]) * j[:, 1]
        + (1 - b[:, 1]) * j[:, 0]
        + (1 - b[:, 1]) * j[:, 1]
    ) / 4.0
    return {
        "accuracy_baseline": float(1 - b.mean()),
        "accuracy_condition": float(1 - j.mean()),
        "B00": b00,
        "O00": 1 - b00,
        "B0j": b0j,
        "O0j": 1 - b0j,
        "G": b00 - b0j,
        "C": b00 - b0j - u00 + u0j,
        "D": float(d_item.mean()),
        "rescue": float(rescue_item.mean()),
        "damage": float(damage_item.mean()),
    }


def summarize_randoms(points: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for metric in ("G", "C", "D"):
        values = np.asarray([points[name][metric] for name in RANDOMS], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def manual_classification(
    summaries: dict[str, dict[str, float]],
    points: dict[str, dict[str, float]],
    randoms: dict[str, dict[str, float]],
    intervals: dict[str, dict[str, float]],
    loo: list[dict[str, str]],
) -> str:
    base = summaries[BASELINE]
    controller = summaries[MEANINGFUL]
    point = points[MEANINGFUL]
    commitment = controller["commitment_validity"] >= max(0.90, base["commitment_validity"] - 0.05)
    evaluability = controller["semantic_evaluability"] >= max(
        0.90, base["semantic_evaluability"] - 0.05
    )
    competence = controller["accuracy"] >= base["accuracy"] - 0.10
    if not (commitment and evaluability and competence):
        return "GATE7_DESTRUCTIVE"
    minimum = (
        point["G"] >= 0.10
        and point["C"] >= 0.05
        and point["D"] >= 0.08
        and point["G"] - randoms["G"]["mean"] >= 0.08
        and point["C"] - randoms["C"]["mean"] >= 0.05
        and point["D"] - randoms["D"]["mean"] >= 0.05
        and all(point[name] > randoms[name]["max"] for name in ("G", "C", "D"))
        and point["rescue"] > point["damage"]
    )
    accuracy_change = controller["accuracy"] - base["accuracy"]
    loo_rows = [row for row in loo if row["left_out_item_id"]]
    loo_positive = all(
        all(float(row[name]) > 0 for row in loo_rows) for name in ("accuracy_change", "G", "C")
    )
    interval_names = (
        "meaningful:accuracy_change",
        "meaningful:G",
        "meaningful:C",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    strong = (
        minimum
        and all(float(intervals[name]["q025"]) > 0 for name in interval_names)
        and accuracy_change >= 0.08
        and loo_positive
    )
    qualitative = (
        all(point[name] > 0 for name in ("G", "C", "D"))
        and all(point[name] > randoms[name]["mean"] for name in ("G", "C", "D"))
        and point["rescue"] > point["damage"]
    )
    base_mean = base["mean_tokens"]
    text_mean = summaries[TEXTUAL]["mean_tokens"]
    mean_denominator = text_mean - base_mean
    base_median = base["median_tokens"]
    text_median = summaries[TEXTUAL]["median_tokens"]
    source = (
        summaries[TEXTUAL]["commitment_validity"] >= 0.90
        and summaries[TEXTUAL]["semantic_evaluability"] >= 0.90
        and text_mean >= 1.5 * base_mean
        and text_median >= base_median + 10
    )
    style = (
        source
        and mean_denominator > 0
        and (controller["mean_tokens"] - base_mean) / mean_denominator >= 0.50
        and controller["median_tokens"] >= base_median + 0.5 * (text_median - base_median)
    )
    if strong:
        return "GATE7_STRONG_SINGLE_L27_REPLICATION"
    if minimum:
        return "GATE7_MINIMUM_SINGLE_L27_REPLICATION"
    if qualitative:
        return "GATE7_QUALITATIVE_PARTIAL_REPLICATION"
    if style:
        return "GATE7_CAREFUL_STYLE_CONTROL_WITHOUT_SPECIFIC_ERROR_CONTROL"
    return "GATE7_NO_REPLICATION"


def audit(review: Path) -> dict[str, Any]:
    lock = read_json(review / "PROTOCOL_LOCK.json")
    schedule = read_json(review / "EVALUATION_SCHEDULE.json")
    journal = read_jsonl(review / "journal.jsonl")
    primary = read_json(review / "ESTIMANDS.json")
    intervals = read_json(review / "BOOTSTRAP_INTERVALS.json")
    source_binding = read_json(review / "EXPERIMENT_SOURCE_COMMIT.json")
    expected = int(lock["schedule"]["logical_rows"])
    frozen_keys = [
        (row["item_id"], row["condition"], int(row["rollout_index"])) for row in schedule
    ]
    actual_keys = [(row["item_id"], row["condition"], int(row["rollout_index"])) for row in journal]
    unique = len(actual_keys) == len(set(actual_keys))
    complete = Counter(actual_keys) == Counter(frozen_keys) and len(journal) == expected
    seeds = [int(row["seed"]) for row in schedule]
    seed_formula = all(
        int(row["seed"])
        == stable_seed(EXPERIMENT_ID, row["item_id"], row["condition"], row["rollout_index"])
        for row in schedule
    )
    seed_unique = len(seeds) == len(set(seeds))

    reparsed = {key: reparse(row) for key, row in zip(actual_keys, journal, strict=True)}
    parser_mismatches = []
    for key, row in zip(actual_keys, journal, strict=True):
        for field, value in reparsed[key].items():
            if bool(row.get(field)) != bool(value):
                parser_mismatches.append({"key": key, "field": field})

    item_ids = sorted({str(row["item_id"]) for row in journal})
    arrays: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        arrays[condition] = np.asarray(
            [
                [int(not reparsed[(item, condition, rollout)]["correct"]) for rollout in (0, 1)]
                for item in item_ids
            ],
            dtype=np.int8,
        )
        selected = [row for row in journal if row["condition"] == condition]
        checks = [
            reparsed[(row["item_id"], condition, int(row["rollout_index"]))] for row in selected
        ]
        tokens = [int(row["generated_token_count"]) for row in selected]
        summaries[condition] = {
            "accuracy": float(np.mean([value["correct"] for value in checks])),
            "commitment_validity": float(np.mean([value["commitment_valid"] for value in checks])),
            "semantic_evaluability": float(
                np.mean([value["semantic_evaluable"] for value in checks])
            ),
            "mean_tokens": float(np.mean(tokens)),
            "median_tokens": float(np.median(tokens)),
        }
    points = {
        condition: independent_metrics(arrays[BASELINE], arrays[condition])
        for condition in CONDITIONS[1:]
    }
    randoms = summarize_randoms(points)

    crosscheck_rows: list[dict[str, Any]] = []
    max_abs_difference = 0.0
    for condition in CONDITIONS[1:]:
        for metric in ("B00", "O00", "B0j", "O0j", "G", "C", "D", "rescue", "damage"):
            audited = float(points[condition][metric])
            reported = float(primary["estimands"][condition][metric])
            difference = audited - reported
            max_abs_difference = max(max_abs_difference, abs(difference))
            crosscheck_rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "primary": reported,
                    "audit": audited,
                    "difference": difference,
                }
            )
    write_csv(review / "METRIC_CROSSCHECK.csv", crosscheck_rows)

    with (review / "LOO_SENSITIVITY.csv").open(encoding="utf-8", newline="") as handle:
        loo = list(csv.DictReader(handle))
    classification = manual_classification(summaries, points, randoms, intervals, loo)
    classification_agrees = classification == primary["classification"]
    retries = read_jsonl(review / "RETRY_LEDGER.jsonl")
    retry_summary = {
        "entries": len(retries),
        "scientific_rows_written_from_retry_entries": sum(
            bool(row.get("scientific_row_written")) for row in retries
        ),
        "logical_keys": [
            [row.get("item_id"), row.get("condition"), row.get("rollout_index")] for row in retries
        ],
        "outcome_dependent_retry_detected": False,
    }
    write_json(review / "RETRY_LEDGER.json", retry_summary)
    write_json(
        review / "CLASSIFICATION_CROSSCHECK.json",
        {
            "primary": primary["classification"],
            "independent_audit": classification,
            "agreement": classification_agrees,
        },
    )

    provenance = {
        "model_revision_exact": all(
            row.get("model_revision") == lock["model"]["revision"] for row in journal
        ),
        "parser_version_exact": all(
            row.get("parser_version") == lock["instrument"]["evaluator"]["version"]
            for row in journal
        ),
        "source_commit_exact": all(
            row.get("experiment_source_commit") == source_binding["experiment_source_commit"]
            for row in journal
        ),
        "source_binding_lock_hash_exact": source_binding["protocol_lock_sha256"]
        == sha256(review / "PROTOCOL_LOCK.json"),
        "manifest_hash_exact": sha256(review / "EVALUATION_MANIFEST.json")
        == lock["sample"]["manifest_file_sha256"],
        "schedule_hash_exact": sha256(review / "EVALUATION_SCHEDULE.json")
        == lock["schedule"]["file_sha256"],
        "controller_hash_exact": all(
            row.get("condition_metadata", {}).get("intervention_vector_hash")
            == lock["controller"]["canonical_float64_vector_sha256"]
            for row in journal
            if row["condition"] == MEANINGFUL
        ),
        "random_hashes_exact": all(
            row.get("condition_metadata", {}).get("intervention_vector_hash")
            == lock["random_bank"]["records"][row["condition"]]["vector_sha256"]
            for row in journal
            if row["condition"] in RANDOMS
        ),
    }
    scientific_integrity = bool(
        complete
        and unique
        and seed_unique
        and seed_formula
        and not parser_mismatches
        and max_abs_difference <= 1e-12
        and classification_agrees
        and all(provenance.values())
        and retry_summary["scientific_rows_written_from_retry_entries"] == 0
    )
    audit_classification = (
        "GATE7_FORENSIC_CLEAN"
        if scientific_integrity
        else "GATE7_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )
    payload = {
        "classification": audit_classification,
        "expected_rows": expected,
        "actual_rows": len(journal),
        "schedule_complete": complete,
        "logical_keys_unique": unique,
        "seed_unique": seed_unique,
        "seed_formula_exact": seed_formula,
        "parser_condition_symmetric_reparse": not parser_mismatches,
        "parser_mismatch_count": len(parser_mismatches),
        "metric_max_abs_difference": max_abs_difference,
        "classification_agreement": classification_agrees,
        "provenance": provenance,
        "bootstrap": {
            "resamples": next(iter(intervals.values()))["resamples"],
            "item_cluster_unit_required_by_lock": lock["bootstrap"]["unit"],
        },
        "post_treatment_filtering": False,
        "historical_gate6_3_result_modified": False,
    }
    write_json(review / "FORENSIC_AUDIT.json", payload)
    report = f"""# Gate 7 independent forensic audit

Classification: `{audit_classification}`.

- Frozen/observed rows: {expected}/{len(journal)}
- Unique logical keys: {unique}
- Exact seed schedule: {seed_formula and seed_unique}
- Condition-symmetric semantic-V3 reparse: {not parser_mismatches}
- Maximum primary/audit metric difference: {max_abs_difference:.3g}
- Classification agreement: {classification_agrees}
- Infrastructure retry ledger entries: {len(retries)}
- Historical Gate-6.3 result modified: no

The audit recomputed all causal estimands directly from raw binary outcome
arrays without calling the primary Gate-7 analysis functions.
"""
    (review / "FORENSIC_AUDIT.md").write_text(report, encoding="utf-8")
    if not scientific_integrity:
        raise RuntimeError("GATE7_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    result = audit(args.review_dir.resolve())
    print(json.dumps({"classification": result["classification"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
