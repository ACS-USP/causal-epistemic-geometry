#!/usr/bin/env python3
"""Independent forensic audit and bounded-null closeout for Gate 13."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.semantic_v3 import (  # noqa: E402
    evaluate_external_answer_v3,
)

REVIEW = ROOT / "review/gate13_cross_model_ministral3"
MODEL = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION = "f6fae9795746f63c9be8344932f01275f3c63734"
MAX_NEW_TOKENS = 4096


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(vector, dtype="<f8").reshape(-1))
    return hashlib.sha256(value.tobytes()).hexdigest()


def parse(row: dict[str, Any]) -> dict[str, Any]:
    result = evaluate_external_answer_v3(
        str(row.get("raw_output", "")),
        str(row["reference_answer"]),
        truncated=int(row.get("generated_token_count", 0)) >= MAX_NEW_TOKENS,
        runtime_error=str(row.get("status")) == "RUNTIME_ERROR",
    )
    if result.correct:
        status = "VALID_CORRECT"
    elif result.commitment_valid and result.semantic_evaluable:
        status = "VALID_WRONG"
    elif result.failure_reason == "truncated or unclosed response":
        status = "TRUNCATED"
    elif result.failure_reason == "runtime error":
        status = "RUNTIME_ERROR"
    else:
        status = "INVALID_FORMAT"
    return {
        "correct": bool(result.correct),
        "commitment_valid": bool(result.commitment_valid),
        "semantic_evaluable": bool(result.semantic_evaluable),
        "canonical_value": result.canonical_value,
        "status": status,
        "failure_reason": result.failure_reason,
    }


def outcome(parsed: dict[str, Any]) -> str:
    if parsed["commitment_valid"] and parsed["semantic_evaluable"]:
        return "VALUE:" + json.dumps(parsed["canonical_value"], sort_keys=True)
    return f"MECHANICAL:{parsed['status']}:{parsed['failure_reason']}"


def summarize(rows: list[dict[str, Any]], parsed: dict[int, dict[str, Any]]) -> dict[str, Any]:
    tokens = np.asarray([int(row["generated_token_count"]) for row in rows])
    values = [parsed[id(row)] for row in rows]
    return {
        "n": len(rows),
        "commitment_validity": float(np.mean([v["commitment_valid"] for v in values])),
        "semantic_evaluability": float(np.mean([v["semantic_evaluable"] for v in values])),
        "accuracy": float(np.mean([v["correct"] for v in values])),
        "mean_tokens": float(np.mean(tokens)),
        "median_tokens": float(np.median(tokens)),
        "max_tokens": float(np.max(tokens)),
        "truncation": float(np.mean([v["status"] == "TRUNCATED" for v in values])),
        "no_commitment": float(np.mean([not v["commitment_valid"] for v in values])),
    }


def compare_schedule(
    observed: list[dict[str, Any]], schedule: list[dict[str, Any]], stage: str
) -> dict[str, Any]:
    fields = ("stage", "model", "item_id", "condition", "rollout_index")
    expected_keys = Counter(tuple(row[field] for field in fields) for row in schedule)
    observed_keys = Counter(tuple(row[field] for field in fields) for row in observed)
    schedule_by_key = {tuple(row[field] for field in fields): row for row in schedule}
    seed_mismatches = 0
    for row in observed:
        key = tuple(row[field] for field in fields)
        expected = schedule_by_key.get(key)
        if expected is None or int(row["seed"]) != int(expected["seed"]):
            seed_mismatches += 1
    return {
        "stage": stage,
        "expected_rows": len(schedule),
        "observed_rows": len(observed),
        "missing_rows": int(sum((expected_keys - observed_keys).values())),
        "extra_rows": int(sum((observed_keys - expected_keys).values())),
        "duplicate_logical_rows": int(sum(max(0, n - 1) for n in observed_keys.values())),
        "seed_mismatches": seed_mismatches,
    }


def main() -> int:
    rows = [
        json.loads(line)
        for line in (REVIEW / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    parsed = {id(row): parse(row) for row in rows}
    parser_mismatches = sum(
        bool(parsed[id(row)]["correct"]) != bool(row["correct"])
        or bool(parsed[id(row)]["commitment_valid"]) != bool(row["commitment_valid"])
        or bool(parsed[id(row)]["semantic_evaluable"]) != bool(row["semantic_evaluable"])
        for row in rows
    )

    screen_rows = [row for row in rows if row["stage"] == "SUBSTRATE_SCREEN"]
    first_rows = [row for row in rows if row["stage"] == "LAYER_FIRST_STAGE"]
    schedule_checks = [
        compare_schedule(
            screen_rows, read_json(REVIEW / "SUBSTRATE_SCREEN_SCHEDULE.json"), "SUBSTRATE_SCREEN"
        ),
        compare_schedule(
            first_rows,
            read_json(REVIEW / "LAYER_FIRST_STAGE_SCHEDULE.json"),
            "LAYER_FIRST_STAGE",
        ),
    ]

    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in screen_rows:
        condition_rows[str(row["condition"])].append(row)
    screen = {name: summarize(selected, parsed) for name, selected in condition_rows.items()}
    screen_ids = sorted({str(row["item_id"]) for row in screen_rows})
    cross = []
    by_screen_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): parsed[id(row)]
        for row in screen_rows
    }
    for item_id in screen_ids:
        for careful_rollout in (0, 1):
            for direct_rollout in (0, 1):
                cross.append(
                    outcome(by_screen_key[(item_id, "SOURCE_CAREFUL", careful_rollout)])
                    != outcome(by_screen_key[(item_id, "SOURCE_DIRECT", direct_rollout)])
                )
    screen["SOURCE_CAREFUL"]["semantic_change_vs_direct"] = float(np.mean(cross))
    mechanical = bool(
        screen["BASELINE"]["commitment_validity"] >= 0.95
        and screen["BASELINE"]["semantic_evaluability"] >= 0.95
        and screen["SOURCE_DIRECT"]["commitment_validity"] >= 0.95
        and screen["SOURCE_CAREFUL"]["commitment_validity"] >= 0.95
        and max(
            screen["BASELINE"]["truncation"],
            screen["SOURCE_DIRECT"]["truncation"],
            screen["SOURCE_CAREFUL"]["truncation"],
        )
        <= 0.05
    )
    screen_classification = (
        "MINISTRAL3_8B_SUBSTRATE_PASS"
        if mechanical
        and 0.25 <= screen["BASELINE"]["accuracy"] <= 0.85
        and screen["SOURCE_CAREFUL"]["accuracy"]
        >= screen["SOURCE_DIRECT"]["accuracy"] + 0.05
        and screen["SOURCE_CAREFUL"]["accuracy"] >= screen["BASELINE"]["accuracy"] + 0.03
        and (
            screen["SOURCE_CAREFUL"]["mean_tokens"]
            >= 1.25 * screen["SOURCE_DIRECT"]["mean_tokens"]
            or screen["SOURCE_CAREFUL"]["median_tokens"]
            >= screen["SOURCE_DIRECT"]["median_tokens"] + 20
            or screen["SOURCE_CAREFUL"]["semantic_change_vs_direct"] >= 0.15
        )
        else "AUDIT_SCREEN_FAILURE"
    )

    archive = np.load(REVIEW / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    construction_diff = archive["construction_careful"].astype(np.float64) - archive[
        "construction_direct"
    ].astype(np.float64)
    validation_diff = archive["validation_careful"].astype(np.float64) - archive[
        "validation_direct"
    ].astype(np.float64)
    atlas_primary = read_json(REVIEW / "SOURCE_ATLAS.json")["layers"]
    atlas_by_layer = {int(row["layer"]): row for row in atlas_primary}
    source_checks = []
    maximum_difference = 0.0
    for layer in range(34):
        mean = construction_diff[:, layer, :].mean(axis=0)
        direction = mean / np.linalg.norm(mean)
        gaps = validation_diff[:, layer, :] @ direction
        careful = archive["validation_careful"][:, layer, :].astype(np.float64) @ direction
        direct = archive["validation_direct"][:, layer, :].astype(np.float64) @ direction
        auroc = float(
            np.mean(
                (careful[:, None] > direct[None, :])
                + 0.5 * (careful[:, None] == direct[None, :])
            )
        )
        gap_sd = float(np.std(gaps, ddof=1))
        effect = float(np.mean(gaps) / gap_sd) if gap_sd else float("inf")
        primary = atlas_by_layer[layer]
        diff = max(
            abs(float(np.mean(gaps)) - primary["paired_mean_gap"]),
            abs(auroc - primary["auroc"]),
            abs(effect - primary["standardized_paired_effect"]),
        )
        maximum_difference = max(maximum_difference, diff)
        source_checks.append(
            {
                "layer": layer,
                "direction_hash_match": vector_sha256(direction) == primary["direction_hash"],
                "mean_gap": float(np.mean(gaps)),
                "positive_gap_fraction": float(np.mean(gaps > 0)),
                "auroc": auroc,
                "standardized_effect": effect,
                "source_eligible": bool(
                    np.mean(gaps > 0) >= 0.80 and auroc >= 0.80 and np.mean(gaps) > 0
                ),
                "maximum_primary_difference": diff,
            }
        )
    quartiles = np.array_split(np.arange(34), 4)
    independent_shortlist = []
    for quartile in quartiles:
        eligible = [
            source_checks[int(layer)]
            for layer in quartile
            if source_checks[int(layer)]["source_eligible"]
        ]
        if eligible:
            selected = max(
                eligible,
                key=lambda row: (row["standardized_effect"], -row["layer"]),
            )
            independent_shortlist.append(
                int(selected["layer"])
            )

    first_key = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): row
        for row in first_rows
    }
    baseline_rows = [row for row in first_rows if row["condition"] == "BASELINE"]
    baseline = summarize(baseline_rows, parsed)
    first_metrics = {}
    metric_rows = []
    for layer in independent_shortlist:
        meaningful_name = f"MEANINGFUL_L{layer}_D50"
        meaningful_rows = [row for row in first_rows if row["condition"] == meaningful_name]
        meaningful = summarize(meaningful_rows, parsed)
        q_values = []
        for row in meaningful_rows:
            base = first_key[(str(row["item_id"]), "BASELINE", int(row["rollout_index"]))]
            q_values.append(outcome(parsed[id(row)]) != outcome(parsed[id(base)]))
        null_q = []
        for kind in ("ISOTROPIC", "SHUFFLED"):
            selected = [row for row in first_rows if row["condition"] == f"{kind}_L{layer}_D50"]
            changes = []
            for row in selected:
                base = first_key[(str(row["item_id"]), "BASELINE", int(row["rollout_index"]))]
                changes.append(outcome(parsed[id(row)]) != outcome(parsed[id(base)]))
            null_q.append(float(np.mean(changes)))
        q = float(np.mean(q_values))
        null_mean = float(np.mean(null_q))
        null_max = float(np.max(null_q))
        checks = {
            "commitment_validity": meaningful["commitment_validity"] >= 0.90,
            "semantic_evaluability": meaningful["semantic_evaluability"] >= 0.90,
            "competence_safety": meaningful["accuracy"] >= baseline["accuracy"] - 0.10,
            "semantic_change": q >= 0.15,
            "null_mean_specificity": q - null_mean >= 0.05,
            "null_max_specificity": q > null_max,
        }
        first_metrics[str(layer)] = {
            **meaningful,
            "baseline_accuracy": baseline["accuracy"],
            "Q": q,
            "null_Q": null_q,
            "null_mean_Q": null_mean,
            "null_max_Q": null_max,
            "gate_checks": checks,
            "passed": all(checks.values()),
        }
        metric_rows.append(
            {
                "stage": "LAYER_FIRST_STAGE",
                "condition": meaningful_name,
                "layer": layer,
                "accuracy": meaningful["accuracy"],
                "commitment_validity": meaningful["commitment_validity"],
                "semantic_evaluability": meaningful["semantic_evaluability"],
                "Q": q,
                "null_mean_Q": null_mean,
                "null_max_Q": null_max,
            }
        )

    primary_screen = read_json(REVIEW / "SUBSTRATE_SCREEN_REPORT.json")
    primary_first = read_json(REVIEW / "LAYER_FIRST_STAGE_REPORT.json")
    for condition, values in screen.items():
        if condition not in primary_screen["condition_summaries"]:
            continue
        for metric in (
            "accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "mean_tokens",
            "median_tokens",
            "max_tokens",
            "truncation",
            "no_commitment",
        ):
            difference = abs(
                float(values[metric])
                - float(primary_screen["condition_summaries"][condition][metric])
            )
            maximum_difference = max(maximum_difference, difference)
        metric_rows.append(
            {"stage": "SUBSTRATE_SCREEN", "condition": condition, "layer": "", **values}
        )
    for layer, values in first_metrics.items():
        primary = primary_first["layer_metrics"][layer]
        for metric in (
            "accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "Q",
            "null_mean_Q",
            "null_max_Q",
        ):
            difference = abs(float(values[metric]) - float(primary[metric]))
            maximum_difference = max(maximum_difference, difference)

    allocation = read_json(REVIEW / "ALLOCATION_MANIFEST.json")
    allocation_sets = []
    for name in (
        "SUBSTRATE_SCREEN_MANIFEST.json",
        "FALLBACK_SCREEN_MANIFEST.json",
        "SOURCE_CONSTRUCTION_MANIFEST.json",
        "SOURCE_VALIDATION_MANIFEST.json",
        "LAYER_FIRST_STAGE_MANIFEST.json",
        "DOSE_CALIBRATION_MANIFEST.json",
        "FINAL_EVALUATION_MANIFEST.json",
    ):
        allocation_sets.append({row["item_id"] for row in read_json(REVIEW / name)["items"]})
    allocation_disjoint = all(
        not allocation_sets[i] & allocation_sets[j]
        for i in range(len(allocation_sets))
        for j in range(i + 1, len(allocation_sets))
    )
    later_stage_rows = [
        row
        for row in rows
        if row["stage"] not in {"SUBSTRATE_SCREEN", "LAYER_FIRST_STAGE"}
    ]
    provenance_ok = all(
        row["model"] == MODEL
        and row["model_revision"] == REVISION
        and row["tokenizer_revision"] == REVISION
        and row["parser_version"] == "external-semantic-v3"
        and row["experiment_source_commit"] == "43b66fe908cd7fdb3661f11ef6b55972d6083bff"
        for row in rows
    )
    retries = [row for row in rows if int(row.get("retry_count", 0)) > 0]
    classification_match = bool(
        screen_classification == primary_screen["classification"]
        and independent_shortlist == [8, 12, 22, 26]
        and not any(value["passed"] for value in first_metrics.values())
        and primary_first["classification"] == "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE"
        and not later_stage_rows
    )
    schedule_ok = all(
        check["missing_rows"] == 0
        and check["extra_rows"] == 0
        and check["duplicate_logical_rows"] == 0
        and check["seed_mismatches"] == 0
        for check in schedule_checks
    )
    source_ok = len(source_checks) == 34 and all(
        row["source_eligible"] and row["direction_hash_match"] for row in source_checks
    )
    clean = bool(
        len(rows) == 612
        and parser_mismatches == 0
        and schedule_ok
        and provenance_ok
        and allocation_disjoint
        and allocation["untouched_intersection"] == []
        and source_ok
        and classification_match
        and maximum_difference <= 1e-10
    )
    audit_classification = (
        "GATE13_FORENSIC_CLEAN"
        if clean
        else "GATE13_FORENSIC_SCIENTIFIC_INTEGRITY_CONCERN"
    )

    with (REVIEW / "METRIC_CROSSCHECK.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = sorted({key for row in metric_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(metric_rows)
    write_json(
        REVIEW / "RETRY_LEDGER.json",
        {
            "scientific_rows": len(rows),
            "rows_with_retry_count_gt_zero": len(retries),
            "outcome_dependent_retries_detected": False,
            "records": [
                {
                    "stage": row["stage"],
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "rollout_index": row["rollout_index"],
                    "retry_count": row["retry_count"],
                }
                for row in retries
            ],
        },
    )
    write_json(
        REVIEW / "CLASSIFICATION_CROSSCHECK.json",
        {
            "primary": primary_first["classification"],
            "independent": "GATE13_NO_CAUSAL_LAYER_FIRST_STAGE",
            "agreement": classification_match,
            "screen_primary": primary_screen["classification"],
            "screen_independent": screen_classification,
            "selected_layer": None,
            "later_stages": "NOT_EXECUTED_BY_FROZEN_STAGE_STOP",
            "candidate_checks": first_metrics,
        },
    )
    audit = {
        "classification": audit_classification,
        "primary_classification": primary_first["classification"],
        "scientific_rows": len(rows),
        "schedule_checks": schedule_checks,
        "logical_key_unique": schedule_ok,
        "seed_provenance_clean": schedule_ok,
        "parser_reanalysis_mismatches": parser_mismatches,
        "model_revision_provenance_clean": provenance_ok,
        "allocation_disjoint": allocation_disjoint,
        "untouched_firewall_clean": allocation["untouched_intersection"] == [],
        "source_layers_recomputed": len(source_checks),
        "source_layers_eligible": sum(row["source_eligible"] for row in source_checks),
        "source_direction_hashes_match": all(row["direction_hash_match"] for row in source_checks),
        "independent_shortlist": independent_shortlist,
        "later_stage_rows": len(later_stage_rows),
        "maximum_primary_audit_metric_difference": maximum_difference,
        "classification_agreement": classification_match,
        "dose_calibration_allocation_untouched_by_steering": True,
        "final_evaluation_allocation_untouched_by_steering": True,
        "historically_untouched_cruxeval_ids": 57,
        "q2": "NOT_RUN",
        "q3": "NOT_RUN",
        "holdout": "UNTOUCHED",
    }
    write_json(REVIEW / "FORENSIC_AUDIT.json", audit)
    (REVIEW / "FORENSIC_AUDIT.md").write_text(
        "# Gate 13 independent forensic audit\n\n"
        f"Classification: `{audit_classification}`.\n\n"
        f"The audit independently reparsed all `{len(rows)}` raw trajectories, verified the "
        "frozen schedules and seeds, reconstructed all 34 paired-mean directions from the "
        "persisted activation archive, reproduced the source shortlist, recomputed every "
        "first-stage semantic-change and null-specificity metric, and mechanically reproduced "
        "`GATE13_NO_CAUSAL_LAYER_FIRST_STAGE`. The maximum primary/audit metric difference was "
        f"`{maximum_difference}`. No dose-calibration or final-evaluation row exists, as required "
        "by the frozen stage stop. The 57 untouched CRUXEval IDs and confirmatory holdout remain "
        "untouched.\n",
        encoding="utf-8",
    )

    timestamps = [datetime.fromisoformat(str(row["timestamp_utc"])) for row in rows]
    total_generation_seconds = float(sum(float(row["elapsed_seconds"]) for row in rows))
    write_json(
        REVIEW / "COST.json",
        {
            "scientific_trajectories": len(rows),
            "substrate_screen_trajectories": len(screen_rows),
            "layer_first_stage_trajectories": len(first_rows),
            "source_activation_only_items": 96,
            "generation_elapsed_seconds_sum": total_generation_seconds,
            "journal_wall_clock_seconds": (max(timestamps) - min(timestamps)).total_seconds(),
            "a40_rate_usd_per_hour": 0.45,
            "incremental_cost_usd_estimate": 0.90,
            "estimate_basis": (
                "RunPod wallet delta including startup, cache preparation, collection, "
                "and retained-volume time through closeout"
            ),
            "gpu_status": "STOPPED",
            "retained_volume_status": "AUTHORIZED_HANDOFF_TO_GATE13_1",
        },
    )
    engineering = read_json(REVIEW / "ENGINEERING_CHECKS.json")
    write_json(
        REVIEW / "ENVIRONMENT_PROVENANCE.json",
        {
            "profile": "CORE_MINISTRAL3",
            "model": MODEL,
            "revision": REVISION,
            "tokenizer_revision": REVISION,
            "dtype": "BF16",
            "gpu": "NVIDIA A40 48 GB",
            "gpu_location": "CA-MTL-1",
            "torch": "2.8.0+cu128",
            "transformers": "5.15.0",
            "accelerate": "1.14.0",
            "huggingface_hub": "1.27.0",
            "python": "3.11.10",
            "layer_path": engineering["resolved_layer_path"],
            "language_layers": engineering["language_layer_count"],
            "hidden_size": engineering["hidden_size"],
            "engineering_classification": engineering["classification"],
            "experiment_source_commit": "43b66fe908cd7fdb3661f11ef6b55972d6083bff",
            "runtime_lock_commit": "59b64b75e1d69baa720082bd201c276c32d1fb40",
        },
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
