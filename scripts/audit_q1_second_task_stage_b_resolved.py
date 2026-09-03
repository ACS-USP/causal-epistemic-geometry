#!/usr/bin/env python3
"""Resolved independent read-only forensic audit for Q1 Stage B.

This additive audit preserves the historical first audit and replaces only its
non-equivalent parser implementation.  The parser used here is independently
implemented and specification-equivalent; it does not import or invoke the
primary Stage-A2 parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_q1_second_task_stage_b import (  # noqa: E402
    independent_bootstrap,
    independent_estimands,
    key,
    read_json,
    read_jsonl,
    scalar_differences,
    sha256,
    summarize,
    write_json,
)

from epistemic_geometry.experiments import q1_second_task_stage_b as stage_b  # noqa: E402
from epistemic_geometry.experiments.q1_second_task_stage_b_forensic import (  # noqa: E402
    independent_score,
)

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
SCHEDULE = REVIEW / "STAGE_B_SCHEDULE.json"
MANIFEST = REVIEW / "STAGE_B_FAMILY_MANIFEST.json"
FIELDS = ("commitment_valid", "semantic_evaluable", "correct")


def _errors(parsed: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    family_order = [row["family_id"] for row in read_json(MANIFEST)["ordered_families"]]
    result: dict[str, np.ndarray] = {}
    for condition in stage_b.CONDITIONS:
        lookup = {
            (row["family_id"], int(row["rollout_index"])): float(not row["correct"])
            for row in parsed
            if row["condition"] == condition
        }
        result[condition] = np.asarray(
            [[lookup[(family, rollout)] for rollout in range(4)] for family in family_order],
            dtype=np.float64,
        )
    return result


def _split_halves(
    baseline: np.ndarray, errors: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for half, rollout_ids in {"A": (0, 1), "B": (2, 3)}.items():
        meaningful = independent_estimands(
            baseline[:, rollout_ids],
            errors["MEANINGFUL_FIXED_QWEN_L27_D75"][:, rollout_ids],
        )["C"]
        nulls = {
            name: independent_estimands(
                baseline[:, rollout_ids], errors[name][:, rollout_ids]
            )["C"]
            for name in stage_b.RANDOM_NAMES
        }
        null_mean = float(np.mean(list(nulls.values())))
        checks = {
            "C_meaningful_gt_zero": meaningful > 0,
            "delta_C_nullmean_gt_zero": meaningful - null_mean > 0,
            "C_meaningful_gt_mean_nulls": meaningful > null_mean,
        }
        result[half] = {
            "C_meaningful": meaningful,
            "null_C_values": nulls,
            "null_C_mean": null_mean,
            "delta_C_nullmean": meaningful - null_mean,
            "checks": checks,
            "passes": all(checks.values()),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = read_jsonl(args.raw_dir / "journal.jsonl")
    schedule = read_json(SCHEDULE)
    expected = {key(row): row for row in schedule}
    observed = {key(row): row for row in raw}
    if len(raw) != 5720 or len(observed) != 5720 or set(observed) != set(expected):
        raise RuntimeError("Stage-B resolved forensic schedule completeness failure")
    for logical, row in observed.items():
        locked = expected[logical]
        for field in ("family_id", "item_id", "item_sha256", "seed"):
            if row[field] != locked[field]:
                raise RuntimeError(f"Stage-B resolved forensic lock mismatch: {field}")

    parsed = [{**row, **independent_score(row)} for row in raw]
    primary = read_json(args.analysis_dir / "PRIMARY_STAGE_B_RESULTS.json")
    primary_parsed = read_jsonl(args.analysis_dir / "PARSED_STAGE_B_RECORDS.jsonl")
    primary_by_key = {key(row): row for row in primary_parsed}
    parser_disagreements = [
        {
            "logical_key": list(logical),
            "fields": [
                field
                for field in FIELDS
                if bool(row[field]) != bool(primary_by_key[logical][field])
            ],
        }
        for logical, row in ((key(entry), entry) for entry in parsed)
        if any(bool(row[field]) != bool(primary_by_key[logical][field]) for field in FIELDS)
    ]

    summaries = {
        condition: summarize([row for row in parsed if row["condition"] == condition])
        for condition in stage_b.CONDITIONS
    }
    errors = _errors(parsed)
    baseline = errors["BASELINE"]
    estimands = {
        condition: independent_estimands(baseline, errors[condition])
        for condition in stage_b.CONDITIONS
        if condition != "BASELINE"
    }
    primary_names = ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    intervals = independent_bootstrap(
        baseline, {name: errors[name] for name in primary_names}
    )
    split_halves = _split_halves(baseline, errors)

    meaningful_c = estimands["MEANINGFUL_FIXED_QWEN_L27_D75"]["C"]
    null_cs = [estimands[name]["C"] for name in stage_b.RANDOM_NAMES]
    scientific_checks = {
        "P1_C_lower_gt_zero": intervals["C_meaningful"]["q025"] > 0,
        "P2a_delta_C_nullmean_lower_gt_zero": intervals["delta_C_nullmean"]["q025"] > 0,
        "P2b_C_meaningful_gt_every_null": meaningful_c > max(null_cs),
        "split_half_A": split_halves["A"]["passes"],
        "split_half_B": split_halves["B"]["passes"],
    }
    safety_checks = {
        "commitment_validity": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"][
            "commitment_validity"
        ]
        >= summaries["BASELINE"]["commitment_validity"] - 0.05,
        "semantic_evaluability": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"][
            "semantic_evaluability"
        ]
        >= summaries["BASELINE"]["semantic_evaluability"] - 0.05,
        "accuracy": summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]["accuracy"]
        >= summaries["BASELINE"]["accuracy"] - 0.10,
    }
    if all(scientific_checks.values()) and all(safety_checks.values()):
        classification = "Q1_SECOND_TASK_FIXED_CONTROLLER_PASS"
    elif all(scientific_checks.values()):
        classification = "Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL"
    else:
        classification = "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY"

    comparisons = scalar_differences(
        {
            "summaries": summaries,
            "estimands": estimands,
            "intervals": intervals,
            "split_halves": split_halves,
        },
        {
            "summaries": primary["summaries"],
            "estimands": primary["estimands"],
            "intervals": {
                "C_meaningful": primary["primary"]["C_meaningful_95_percentile_CI"],
                "delta_C_nullmean": primary["primary"][
                    "delta_C_nullmean_95_percentile_CI"
                ],
            },
            "split_halves": primary["split_halves"],
        },
    )
    max_difference = max((value for _, value in comparisons), default=0.0)
    clean = (
        not parser_disagreements
        and max_difference <= 1e-12
        and classification == primary["decision"]["classification"]
        and scientific_checks == primary["decision"]["scientific_checks"]
        and safety_checks == primary["decision"]["safety_checks"]
    )
    audit = {
        "schema_version": 1,
        "resolution": "AUDIT_IMPLEMENTATION_NON_EQUIVALENT",
        "historical_audit_preserved": True,
        "journal_sha256": sha256(args.raw_dir / "journal.jsonl"),
        "rows_recomputed": len(parsed),
        "parser_disagreements": parser_disagreements,
        "summaries": summaries,
        "estimands": estimands,
        "intervals": intervals,
        "split_halves": split_halves,
        "scientific_checks": scientific_checks,
        "safety_checks": safety_checks,
        "classification": classification,
        "maximum_primary_audit_metric_difference": max_difference,
        "classification_agreement": classification == primary["decision"]["classification"],
        "forensic_classification": (
            "Q1_SECOND_TASK_STAGE_B_FORENSIC_RESOLVED_PRIMARY_CONFIRMED"
            if clean
            else "Q1_SECOND_TASK_STAGE_B_FORENSIC_RESOLUTION_BLOCKED"
        ),
        "primary_artifacts_modified": False,
    }
    write_json(args.output, audit)
    return 0 if clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
