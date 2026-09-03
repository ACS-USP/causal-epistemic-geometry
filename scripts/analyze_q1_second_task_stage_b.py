#!/usr/bin/env python3
"""Frozen primary analysis for the completed Q1 LiveCodeBench Stage B campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_b as stage_b  # noqa: E402
from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (  # noqa: E402
    evaluate_livecodebench_output_stage_a2,
)

REVIEW = ROOT / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
SCHEDULE = REVIEW / "STAGE_B_SCHEDULE.json"
MANIFEST = REVIEW / "STAGE_B_FAMILY_MANIFEST.json"


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


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return stage_b.logical_key(row)


def validate_complete(
    raw: list[dict[str, Any]], schedule: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], dict[str, Any]]:
    expected = {key(row): row for row in schedule}
    observed: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    duplicates = 0
    unexpected = 0
    mismatches: list[str] = []
    for row in raw:
        logical = key(row)
        if logical in observed:
            duplicates += 1
            continue
        if logical not in expected:
            unexpected += 1
            continue
        locked = expected[logical]
        for field in ("family_id", "item_id", "item_sha256", "condition", "rollout_index", "seed"):
            if row[field] != locked[field]:
                mismatches.append(f"{logical}:{field}")
        if not row.get("terminal_model_output", False):
            mismatches.append(f"{logical}:terminal_model_output")
        observed[logical] = row
    missing = len(set(expected) - set(observed))
    report = {
        "expected": len(expected),
        "observed_rows": len(raw),
        "unique_expected_keys_observed": len(observed),
        "duplicates": duplicates,
        "unexpected": unexpected,
        "missing": missing,
        "metadata_mismatches": mismatches,
    }
    if report != {
        "expected": 5720,
        "observed_rows": 5720,
        "unique_expected_keys_observed": 5720,
        "duplicates": 0,
        "unexpected": 0,
        "missing": 0,
        "metadata_mismatches": [],
    }:
        raise RuntimeError(f"Q1_SECOND_TASK_EXECUTION_INCOMPLETE: {report}")
    return observed, report


def seal_raw(
    raw_dir: Path,
    analysis_dir: Path,
    raw: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> dict[str, Any]:
    journal = raw_dir / "journal.jsonl"
    output_hashes = [str(row["output_sha256"]) for row in raw]
    campaign_manifest = {
        "experiment": stage_b.EXPERIMENT_ID,
        "stage": stage_b.STAGE,
        "families": 130,
        "conditions": list(stage_b.CONDITIONS),
        "rollouts": 4,
        "logical_rows": 5720,
        "schedule_sha256": sha256(SCHEDULE),
        "family_manifest_sha256": sha256(MANIFEST),
        "coverage": coverage,
    }
    write_json(analysis_dir / "CAMPAIGN_MANIFEST.json", campaign_manifest)
    seal = {
        "classification": "Q1_SECOND_TASK_STAGE_B_RAW_DATA_SEALED_PRE_ANALYSIS",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "journal_sha256": sha256(journal),
        "raw_output_hash_bundle_sha256": stable_digest(output_hashes),
        "campaign_manifest_sha256": sha256(analysis_dir / "CAMPAIGN_MANIFEST.json"),
        "raw_rows": len(raw),
        "scientific_analysis_before_seal": False,
    }
    write_json(analysis_dir / "RAW_DATA_SEAL.json", seal)
    return seal


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    tokens = [int(row["generated_token_count"]) for row in rows]
    n = len(rows)
    return {
        "n": n,
        "commitment_validity": sum(bool(row["commitment_valid"]) for row in rows) / n,
        "semantic_evaluability": sum(bool(row["semantic_evaluable"]) for row in rows) / n,
        "accuracy": sum(bool(row["correct"]) for row in rows) / n,
        "generated_tokens_mean": float(np.mean(tokens)),
        "generated_tokens_median": float(median(tokens)),
        "generated_tokens_p90": float(np.quantile(tokens, 0.90)),
        "generated_tokens_p95": float(np.quantile(tokens, 0.95)),
        "generated_tokens_max": int(max(tokens)),
    }


def textual_annotation(baseline: dict[str, float], textual: dict[str, float]) -> dict[str, Any]:
    accuracy_delta = textual["accuracy"] - baseline["accuracy"]
    mean_ratio = textual["generated_tokens_mean"] / baseline["generated_tokens_mean"]
    median_delta = textual["generated_tokens_median"] - baseline["generated_tokens_median"]
    manifestations = {
        "accuracy_gain_ge_0_03": accuracy_delta >= 0.03,
        "mean_token_ratio_ge_1_5": mean_ratio >= 1.5,
        "median_token_gain_ge_10": median_delta >= 10.0,
    }
    nonharm = textual["accuracy"] >= baseline["accuracy"] - 0.03
    if manifestations["accuracy_gain_ge_0_03"]:
        label = "TEXTUAL_CAREFUL_ACCURACY_BENEFIT_PRESENT"
    elif nonharm and any(manifestations.values()):
        label = "TEXTUAL_CAREFUL_NONHARMFUL_COMPUTE_MANIFESTATION"
    else:
        label = "TEXTUAL_CAREFUL_NO_QUALIFYING_MANIFESTATION"
    return {
        "accuracy_delta": accuracy_delta,
        "mean_token_ratio": mean_ratio,
        "median_token_delta": median_delta,
        "nonharm": nonharm,
        "manifestations": manifestations,
        "descriptive_label": label,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    journal = args.raw_dir / "journal.jsonl"
    raw = read_jsonl(journal)
    schedule = read_json(SCHEDULE)
    stage_b.validate_schedule(schedule)
    observed, coverage = validate_complete(raw, schedule)
    seal = seal_raw(args.raw_dir, args.analysis_dir, raw, coverage)

    parsed: list[dict[str, Any]] = []
    for locked in schedule:
        row = observed[key(locked)]
        score = evaluate_livecodebench_output_stage_a2(
            row["raw_output"],
            row["reference_answer"],
            row["generated_token_ids"],
            truncated=bool(row["truncated"]),
        )
        parsed.append(
            {
                **{
                    field: row[field]
                    for field in (
                        "stage",
                        "family_id",
                        "item_id",
                        "item_sha256",
                        "condition",
                        "rollout_index",
                        "seed",
                        "generated_token_count",
                        "truncated",
                        "output_sha256",
                    )
                },
                **score,
            }
        )
    write_jsonl(args.analysis_dir / "PARSED_STAGE_B_RECORDS.jsonl", parsed)

    by_condition = {
        condition: [row for row in parsed if row["condition"] == condition]
        for condition in stage_b.CONDITIONS
    }
    summaries = {name: summarize(rows) for name, rows in by_condition.items()}
    family_order = [row["family_id"] for row in read_json(MANIFEST)["ordered_families"]]
    errors: dict[str, np.ndarray] = {}
    for condition in stage_b.CONDITIONS:
        lookup = {
            (row["family_id"], int(row["rollout_index"])): float(not row["correct"])
            for row in by_condition[condition]
        }
        errors[condition] = np.asarray(
            [[lookup[(family, rollout)] for rollout in range(4)] for family in family_order],
            dtype=np.float64,
        )
    baseline = errors["BASELINE"]
    estimands = {
        condition: q1s.r_rollout_estimands(baseline, values)
        for condition, values in errors.items()
        if condition != "BASELINE"
    }
    primary_conditions = {
        name: errors[name] for name in ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    }
    intervals = stage_b.primary_bootstrap(baseline, primary_conditions)
    split_halves = stage_b.split_half_checks(baseline, primary_conditions)
    decision = stage_b.classify(
        summaries=summaries,
        estimands=estimands,
        intervals=intervals,
        split_halves=split_halves,
    )
    null_values = {name: estimands[name]["C"] for name in stage_b.RANDOM_NAMES}
    meaningful_c = estimands["MEANINGFUL_FIXED_QWEN_L27_D75"]["C"]
    result = {
        "schema_version": 1,
        "analysis_timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_data_seal_sha256": sha256(args.analysis_dir / "RAW_DATA_SEAL.json"),
        "journal_sha256": seal["journal_sha256"],
        "parsed_records_sha256": sha256(args.analysis_dir / "PARSED_STAGE_B_RECORDS.jsonl"),
        "coverage": coverage,
        "summaries": summaries,
        "unstratified_token_summary": summarize(parsed),
        "estimands": estimands,
        "primary": {
            "C_meaningful": meaningful_c,
            "C_meaningful_95_percentile_CI": intervals["C_meaningful"],
            "null_C_values": null_values,
            "null_C_mean": float(np.mean(list(null_values.values()))),
            "delta_C_nullmean": meaningful_c - float(np.mean(list(null_values.values()))),
            "delta_C_nullmean_95_percentile_CI": intervals["delta_C_nullmean"],
            "C_meaningful_gt_every_null": meaningful_c > max(null_values.values()),
            "bootstrap_resamples": stage_b.BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": stage_b.BOOTSTRAP_SEED,
            "bootstrap_unit": "QUESTION_FAMILY",
        },
        "split_halves": split_halves,
        "decision": decision,
        "textual_careful": textual_annotation(summaries["BASELINE"], summaries["TEXTUAL_CAREFUL"]),
        "stage_a1_preserved": "Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED",
        "stage_a2_preserved": "Q1_SECOND_TASK_STAGE_A2_QUALIFIED",
        "q2_outputs_inspected": False,
    }
    write_json(args.analysis_dir / "PRIMARY_STAGE_B_RESULTS.json", result)
    write_json(
        args.analysis_dir / "PRIMARY_STAGE_B_CLASSIFICATION.json",
        {
            "classification": decision["classification"],
            "primary_results_sha256": sha256(args.analysis_dir / "PRIMARY_STAGE_B_RESULTS.json"),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
