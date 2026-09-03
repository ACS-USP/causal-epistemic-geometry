#!/usr/bin/env python3
"""Prepare the prospective, never-executed Q1 second-task Stage-A2 lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_stage_a2 as a2  # noqa: E402
from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (  # noqa: E402
    evaluate_livecodebench_output_stage_a2,
)

REVIEW = ROOT / "review/q1_second_task_spark2_design"
AMENDMENT1 = REVIEW / "amendment1_hierarchical_unit"
STAGE_A1_CLOSEOUT = "00ab87c386c72a7c88fac438e94240619555d629"
JOURNAL_SHA256 = "5b0fec6960ac414f56995d91a43c3b41c49a06b5fb868156a8e24d037b9281b1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def selected_power_rows() -> dict[str, dict[str, dict[str, float]]]:
    selected: dict[str, dict[str, dict[str, float]]] = {"120": {}, "130": {}}
    with (AMENDMENT1 / "DEPENDENCE_AWARE_POWER_GRID.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["design"] != "C_ONE_ROW_PER_FAMILY":
                continue
            units = row["independent_units"]
            transfer = row["transfer_fraction"]
            if units not in selected or transfer not in {"0.0", "0.75", "1.0"}:
                continue
            selected[units][transfer] = {
                "expected_c_ci_width": float(row["expected_c_ci_width"]),
                "expected_delta_c_ci_width": float(row["expected_delta_c_ci_width"]),
                "power_c_lower_gt_zero": float(row["power_c_lower_gt_zero"]),
                "power_delta_c_lower_gt_zero": float(row["power_delta_c_lower_gt_zero"]),
                "probability_c_above_every_random": float(
                    row["probability_c_above_every_random"]
                ),
                "split_half_sign_consistency": float(row["split_half_sign_consistency"]),
                "joint_frozen_rule_probability": float(row["joint_frozen_rule_probability"]),
                "monte_carlo_se": float(row["joint_probability_monte_carlo_se"]),
            }
    if any(set(values) != {"0.0", "0.75", "1.0"} for values in selected.values()):
        raise RuntimeError("required N=120/130 power planning rows are missing")
    return selected


def option_comparison(power: dict[str, Any]) -> dict[str, Any]:
    observed_runtime_seconds = 4593.385388884984
    observed_journal_bytes = 619809
    observed_mean_tokens = (368.484375 + 460.359375) / 2

    def option(name: str, families: int, stage_b_families: int) -> dict[str, Any]:
        trajectories = families * 2 * 2
        scaling = trajectories / 128
        planning_128_hours = 0.49348558455200525 * scaling
        return {
            "option": name,
            "stage_a2_families": families,
            "stage_a2_trajectories": trajectories,
            "condition_rows": families * 2,
            "validity_resolution": 1 / (families * 2),
            "maximum_invalid_rows_at_0_95": int((families * 2) * 0.05 + 1e-12),
            "wrong_both_count_min": (families + 9) // 10,
            "correct_at_least_once_count_min": (families + 4) // 5,
            "stage_b_families_remaining": stage_b_families,
            "stage_b_trajectories": stage_b_families * 11 * 4,
            "untouched_reserve_families_after_stage_a2": 0,
            "runtime_hours_with_25pct_margin": {
                "mean_tokens_128": planning_128_hours,
                "mean_tokens_256": planning_128_hours * 2,
                "mean_tokens_512": planning_128_hours * 4,
                "stage_a1_empirical_mix": observed_runtime_seconds
                * scaling
                * 1.25
                / 3600,
            },
            "stage_a1_observed_mean_tokens_for_empirical_projection": observed_mean_tokens,
            "estimated_raw_journal_bytes": round(observed_journal_bytes * scaling),
            "stage_b_power": power[str(stage_b_families)],
        }

    return {
        "planning_only": True,
        "livecodebench_stage_a2_outcomes": 0,
        "options": [
            option("OPTION_1_RESERVE_ONLY", 20, 130),
            option("OPTION_2_RESERVE_PLUS_TEN_STAGE_B", 30, 120),
        ],
        "recommendation": "OPTION_1_RESERVE_ONLY",
        "rationale": [
            "The parser repair directly addresses the observed answer-channel defect.",
            "Twenty families allow two invalid rows per condition at the unchanged 0.95 guard.",
            "The task-opportunity gates were already comfortably passed in Stage A1.",
            "Option 1 preserves all 130 Stage-B families and its frozen planning power.",
            "Option 2 consumes ten Stage-B families and lowers full-transfer "
            "joint planning power from 0.81514 to 0.78552.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reserve_path = AMENDMENT1 / "RESERVE_FAMILY_MANIFEST.json"
    stage_a1_schedule_path = AMENDMENT1 / "STAGE_A_SCHEDULE.json"
    stage_b_schedule_path = AMENDMENT1 / "STAGE_B_SCHEDULE.json"
    reserve = read_json(reserve_path)
    selected = a2.selected_reserve_families(reserve)
    schedule = a2.build_schedule(selected)
    a1_schedule = read_json(stage_a1_schedule_path)
    b_schedule = read_json(stage_b_schedule_path)
    a2_families = {row["family_id"] for row in schedule}
    if a2_families & {row["family_id"] for row in a1_schedule}:
        raise RuntimeError("Stage-A2/Stage-A1 family collision")
    if a2_families & {row["family_id"] for row in b_schedule}:
        raise RuntimeError("Stage-A2/Stage-B family collision")
    old_seeds = {int(row["seed"]) for row in [*a1_schedule, *b_schedule]}
    if old_seeds & {int(row["seed"]) for row in schedule}:
        raise RuntimeError("Stage-A2 seed collision with an earlier schedule")
    manifest = {
        "schema_version": 1,
        "status": "DRAFT_AWAITING_PRINCIPAL_RESEARCHER_FREEZE_NOT_EXECUTED",
        "experiment_id": a2.EXPERIMENT_ID,
        "scientific_unit": "QUESTION_FAMILY",
        "n_families": 20,
        "n_selected_rows": 20,
        "source_reserve_manifest": str(reserve_path.relative_to(ROOT)),
        "source_reserve_manifest_sha256": sha256(reserve_path),
        "representative_row_rule": "Amendment-1 stable-digest rule reused without change",
        "family_order": "frozen reserve-manifest order",
        "ordered_families": selected,
        "fresh_family_correctness_inspected": False,
        "stage_a2_outcomes": 0,
    }
    manifest_path = args.output_dir / "STAGE_A2_FAMILY_MANIFEST.json"
    schedule_path = args.output_dir / "STAGE_A2_SCHEDULE.json"
    write_json(manifest_path, manifest)
    write_json(schedule_path, schedule)
    power = selected_power_rows()
    comparison = option_comparison(power)
    comparison_path = args.output_dir / "STAGE_A2_OPTION_COMPARISON.json"
    write_json(comparison_path, comparison)

    parser_file = ROOT / "src/epistemic_geometry/experiments/q1_second_task_stage_a_failure.py"
    stage_a2_file = ROOT / "src/epistemic_geometry/experiments/q1_second_task_stage_a2.py"
    prompt_builder_source = inspect.getsource(q1s.build_livecodebench_prompt)
    evaluator_source = inspect.getsource(evaluate_livecodebench_output_stage_a2)
    protocol_path = REVIEW / "PROTOCOL_LOCK.json"
    controller_path = REVIEW / "CONTROLLER_PROVENANCE_LOCK.json"
    controller = read_json(controller_path)
    instrument_audit = {
        "classification": "PARSER_LIMITATION_PRIMARY_WITH_ONE_FAIL_CLOSED_AMBIGUITY",
        "historical_stage_a1_classification": "Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED",
        "historical_stage_a1_modified": False,
        "baseline_prompt": {
            "system_prompt": None,
            "user_prompt_builder": "q1_second_task.build_livecodebench_prompt",
            "user_prompt_builder_source_sha256": text_sha256(prompt_builder_source),
            "stage_a2_change": "NONE",
        },
        "textual_careful_prompt": {
            "system_prompt_sha256": controller["textual_careful_sha256"],
            "stage_a2_change": "NONE",
        },
        "prompt_difference_in_stage_a1": (
            "TEXTUAL_CAREFUL alone added meticulous tracing, verification, and a "
            "repeated FINAL instruction; causal attribution among those components "
            "is not possible from Stage A1."
        ),
        "selected_repair": "parser-only condition-symmetric terminal typed literal extension",
        "prompt_repair_selected": False,
        "combined_repair_selected": False,
        "parser_file_sha256": sha256(parser_file),
        "evaluator_function_source_sha256": text_sha256(evaluator_source),
        "stage_a2_gate_file_sha256": sha256(stage_a2_file),
        "typed_exact_comparison_preserved": True,
        "llm_judge": False,
        "fuzzy_matching": False,
    }
    instrument_path = args.output_dir / "CURRENT_INSTRUMENT_AND_PROMPT_AUDIT.json"
    write_json(instrument_path, instrument_audit)

    amendment = {
        "schema_version": 1,
        "classification": "Q1_SECOND_TASK_STAGE_A2_PROSPECTIVE_AMENDMENT_DRAFT",
        "status": "DRAFT_AWAITING_PRINCIPAL_RESEARCHER_FREEZE_NOT_EXECUTED",
        "parent_stage_a1_closeout_commit": STAGE_A1_CLOSEOUT,
        "parent_stage_a1_journal_sha256": JOURNAL_SHA256,
        "stage_a1_result_preserved": "Q1_SECOND_TASK_STAGE_A_NOT_QUALIFIED",
        "stage_a1_schedule_status": "EXECUTED_AND_FAILED_AS_FROZEN",
        "stage_a1_data_use": "POST_HOC_INSTRUMENT_DEVELOPMENT_ONLY",
        "reason": (
            "A generic answer-channel parser limitation rejected a unique terminal "
            "typed literal when an earlier empty nonliteral final-style heading was present."
        ),
        "selected_repair_class": "REPAIR_A_CONDITION_SYMMETRIC_PARSER_EXTENSION",
        "selected_repair": "TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1",
        "parser_sha256": sha256(parser_file),
        "prompt_builder_source_sha256": text_sha256(prompt_builder_source),
        "prompt_changed": False,
        "evaluator_function_source_sha256": text_sha256(evaluator_source),
        "selected_option": "OPTION_1_RESERVE_ONLY",
        "scientific_unit": "QUESTION_FAMILY",
        "families": 20,
        "one_row_per_family": True,
        "conditions": ["BASELINE", "TEXTUAL_CAREFUL"],
        "rollouts": 2,
        "logical_rows": 80,
        "manifest_sha256": sha256(manifest_path),
        "schedule_sha256": sha256(schedule_path),
        "unique_seeds": 80,
        "thresholds": {
            "baseline_commitment_validity_min": 0.95,
            "baseline_semantic_evaluability_min": 0.95,
            "baseline_accuracy_range": [0.25, 0.90],
            "baseline_B00_min": 0.05,
            "baseline_wrong_both_min": 2,
            "baseline_correct_at_least_once_min": 4,
            "textual_commitment_validity_min": 0.95,
            "textual_semantic_evaluability_min": 0.95,
            "textual_nonharm_margin": 0.03,
            "manifestation_or": {
                "accuracy_gain_min": 0.03,
                "mean_token_ratio_min": 1.5,
                "median_token_gain_min": 10
            },
        },
        "model": q1s.MODEL,
        "model_revision": q1s.MODEL_REVISION,
        "tokenizer_revision": q1s.MODEL_REVISION,
        "environment_fingerprint": (
            "306d65af9643cc1144d344ae57141ac96ffbbcf70520f67e9276a907d29660bc"
        ),
        "generation": read_json(protocol_path)["generation"],
        "retry_resume": (
            "unchanged from frozen Stage-A1 operational policy; retry only "
            "pre-persistence operational failures with identical logical key/seed; "
            "terminal rows never retry"
        ),
        "stage_a2_outcomes": 0,
        "stage_b_outcomes": 0,
        "fresh_family_correctness_inspected": False,
        "stage_b_status": "CLOSED_NOT_AUTHORIZED",
        "meaningful_controller_status": "SEALED_NOT_OPENED",
        "activation_null_status": "SEALED_NOT_OPENED",
        "meaningful_controller_livecodebench_trajectories": 0,
        "activation_null_livecodebench_trajectories": 0,
        "q2_outputs_inspected": False,
        "stage_a2_execution_authorized": False,
    }
    amendment_path = args.output_dir / "AMENDMENT2_LOCK_DRAFT.json"
    write_json(amendment_path, amendment)

    audit = {
        "classification": "Q1_SECOND_TASK_STAGE_A2_DESIGN_INTEGRITY_PASS",
        "manifest_rows": 20,
        "schedule_rows": len(schedule),
        "logical_keys_unique": len({a2.logical_key(row) for row in schedule}) == 80,
        "seeds_unique": len({row["seed"] for row in schedule}) == 80,
        "seeds_disjoint_from_stage_a1_and_stage_b": True,
        "stage_a1_family_overlap": 0,
        "stage_b_family_overlap": 0,
        "all_twenty_reserve_families_used": len(a2_families) == 20,
        "stage_b_manifest_unchanged_sha256": sha256(
            AMENDMENT1 / "STAGE_B_FAMILY_MANIFEST.json"
        ),
        "stage_b_schedule_unchanged_sha256": sha256(stage_b_schedule_path),
        "activation_conditions_present": False,
        "stage_a2_outcomes": 0,
        "new_model_inference": 0,
        "q2_outputs_inspected": False,
    }
    audit_path = args.output_dir / "STAGE_A2_DESIGN_AUDIT.json"
    write_json(audit_path, audit)

    artifact_paths = [
        manifest_path,
        schedule_path,
        comparison_path,
        instrument_path,
        amendment_path,
        audit_path,
    ]
    write_json(
        args.output_dir / "stage_a2_artifact_hashes.json",
        {str(path.relative_to(args.output_dir)): sha256(path) for path in artifact_paths},
    )
    write_json(
        args.output_dir / "artifact_hashes.json",
        {
            path.name: sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "artifact_hashes.json"
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
