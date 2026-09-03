#!/usr/bin/env python3
"""Prepare the model-free Q1 second-task hierarchical-unit Amendment 1."""

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

from epistemic_geometry.experiments import q1_second_task as base  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_hierarchical as h  # noqa: E402
from epistemic_geometry.experiments import q1_second_task_hierarchical_power as hp  # noqa: E402
from epistemic_geometry.reproducibility import stable_digest  # noqa: E402

PARENT = ROOT / "review/q1_second_task_spark2_design"
REVIEW = PARENT / "amendment1_hierarchical_unit"
PARENT_COMMIT = "03098d80f73b1ee1a4cfb5fb56504c409aad6286"
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TRANSFER_GRID = (0.0, 0.5, 0.75, 1.0)
UNIT_GRID = (80, 100, 120, 130, 140)
REPLICATES = 100_000
POWER_SEED = 2026083001


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pages(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(
        directory.glob("rows_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    rows = [entry for path in paths for entry in read_json(path)["rows"]]
    rows.sort(key=lambda entry: int(entry["row_idx"]))
    return [dict(entry["row"]) for entry in rows]


def quantiles(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": max(values),
    }


def family_program_hash(rows: list[base.LiveCodeBenchItem]) -> str:
    first = rows[0]
    payload = json.dumps(
        {
            "question_content": first.question_content,
            "starter_code": first.starter_code,
            "question_id": first.question_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def family_record(
    family_id: str,
    rows: list[base.LiveCodeBenchItem],
    *,
    selected: base.LiveCodeBenchItem | None,
) -> dict[str, Any]:
    return {
        "family_id": family_id,
        "family_size": len(rows),
        "family_program_sha256": family_program_hash(rows),
        "all_item_ids": [row.item_id for row in rows],
        "all_item_sha256": [row.item_sha256 for row in rows],
        "selected_item": selected.public_manifest_record() if selected else None,
        "selection_digest": (
            stable_digest(h.EXPERIMENT_ID, "REPRESENTATIVE_ROW", family_id, selected.item_id)
            if selected
            else None
        ),
    }


def split_audit(name: str, manifest_name: str, item_to_family: dict[str, str]) -> dict[str, Any]:
    manifest = read_json(PARENT / manifest_name)
    counts = Counter(item_to_family[row["item_id"]] for row in manifest["ordered_records"])
    sizes = sorted(counts.values())
    row_count = len(manifest["ordered_records"])
    family_count = len(sizes)
    equal_weight = 1.0 / family_count
    maximum_weight = max(sizes) / row_count
    records = []
    for rho in RHO_GRID:
        effect = hp.unequal_cluster_design_effect(sizes, rho)
        records.append(
            {
                "rho": rho,
                "design_effect": effect,
                "effective_independent_units": row_count / effect,
            }
        )
    return {
        "split": name,
        "rows": row_count,
        "families": family_count,
        "family_size_counts": dict(sorted(Counter(sizes).items())),
        "family_size_summary": quantiles(sizes),
        "equal_family_weight": equal_weight,
        "minimum_row_level_family_weight": min(sizes) / row_count,
        "maximum_row_level_family_weight": maximum_weight,
        "maximum_to_minimum_family_weight_ratio": max(sizes) / min(sizes),
        "row_bootstrap_sensitivity": records,
    }


def power_grid(
    all_groups: dict[str, list[base.LiveCodeBenchItem]],
    current_stage_b_sizes: list[int],
) -> list[dict[str, Any]]:
    ordered = sorted(
        all_groups, key=lambda family_id: stable_digest(h.EXPERIMENT_ID, "FAMILY_ORDER", family_id)
    )
    stage_b_candidates = ordered[h.STAGE_A_FAMILIES :]
    records: list[dict[str, Any]] = []
    index = 0
    for units in UNIT_GRID:
        sizes = [len(all_groups[family_id]) for family_id in stage_b_candidates[:units]]
        for fraction in TRANSFER_GRID:
            records.append(
                hp.simulate_one_row_per_family(
                    units,
                    transfer_fraction=fraction,
                    replicates=REPLICATES,
                    seed=POWER_SEED + index,
                )
            )
            index += 1
        for rho in RHO_GRID:
            for fraction in TRANSFER_GRID:
                records.append(
                    hp.simulate_family_balanced(
                        sizes,
                        rho=rho,
                        transfer_fraction=fraction,
                        replicates=REPLICATES,
                        seed=POWER_SEED + index,
                    )
                )
                index += 1
    for rho in RHO_GRID:
        for fraction in TRANSFER_GRID:
            records.append(
                hp.simulate_row_weighted(
                    current_stage_b_sizes,
                    rho=rho,
                    transfer_fraction=fraction,
                    replicates=REPLICATES,
                    seed=POWER_SEED + index,
                )
            )
            index += 1
    return records


def runtime_record(name: str, stage_a_rows: int, stage_b_rows: int) -> dict[str, Any]:
    tokens_per_second = 11.527973798978062
    safety = 1.25

    def hours(rows: int, tokens: int) -> float:
        return rows * tokens / tokens_per_second / 3600.0 * safety

    return {
        "design": name,
        "stage_a_trajectories": stage_a_rows,
        "stage_b_trajectories": stage_b_rows,
        "stage_a_hours_with_25pct_margin": {
            str(tokens): hours(stage_a_rows, tokens) for tokens in (128, 256, 512)
        },
        "stage_b_hours_with_25pct_margin": {
            str(tokens): hours(stage_b_rows, tokens) for tokens in (128, 256, 512)
        },
        "combined_hours_with_25pct_margin": {
            str(tokens): hours(stage_a_rows + stage_b_rows, tokens)
            for tokens in (128, 256, 512)
        },
        "storage_reservation_gb": max(1.0, stage_b_rows / 6600.0 * 1.5),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-pages", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.parquet) != base.LIVECODEBENCH_PARQUET_SHA256:
        raise RuntimeError("pinned LiveCodeBench parquet mismatch")
    parent_lock = read_json(PARENT / "PRINCIPAL_REVIEW_LOCK.json")
    if parent_lock["scientific_benchmark_outcomes"] != 0 or parent_lock["correctness_inspected"]:
        raise RuntimeError("parent design no longer has a clean pre-outcome firewall")

    raw_rows = load_pages(args.dataset_pages)
    items = [base.normalize_livecodebench_row(row, index) for index, row in enumerate(raw_rows)]
    groups = h.group_families(items)
    if len(items) != 442 or len(groups) != 182:
        raise RuntimeError("official LiveCodeBench structure mismatch")
    raw_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        raw_by_family.setdefault(str(row["question_id"]), []).append(row)
    shared_fields = (
        "question_content",
        "starter_code",
        "function_name",
        "question_title",
        "contest_id",
        "contest_date",
        "difficulty",
    )
    field_checks = {
        field: all(
            len({json.dumps(row[field], sort_keys=True) for row in rows}) == 1
            for rows in raw_by_family.values()
        )
        for field in shared_fields
    }
    tests_differ = all(
        len({row["test"] for row in rows}) == len(rows)
        for rows in raw_by_family.values()
    )
    sizes = sorted(len(rows) for rows in groups.values())
    family_structure = {
        "classification": "QUESTION_FAMILY_IS_PRIMARY_STRUCTURAL_UNIT",
        "rows": len(items),
        "families": len(groups),
        "family_size_counts": dict(sorted(Counter(sizes).items())),
        "family_size_summary": quantiles(sizes),
        "shared_fields_identical_within_every_family": field_checks,
        "test_id_unique_within_every_family": all(
            len({row["test_id"] for row in rows}) == len(rows)
            for rows in raw_by_family.values()
        ),
        "test_input_and_reference_differ_within_every_family": tests_differ,
        "rows_from_three_or_more_row_families": sum(value for value in sizes if value >= 3),
        "fraction_rows_from_three_or_more_row_families": sum(
            value for value in sizes if value >= 3
        )
        / len(items),
        "scientific_interpretation": (
            "Rows within a family share the complete program, statement, function, and latent "
            "reasoning structure; they differ by test input and exact output. Positive dependence "
            "is structurally plausible, so row-level independence is not justified."
        ),
        "model_inference": False,
        "correctness_inspected": False,
    }
    write_json(REVIEW / "FAMILY_STRUCTURE_AUDIT.json", family_structure)

    item_to_family = {item.item_id: item.question_id for item in items}
    current_a = split_audit("STAGE_A", "STAGE_A_MANIFEST.json", item_to_family)
    current_b = split_audit("STAGE_B", "STAGE_B_HOLDOUT_MANIFEST.json", item_to_family)
    current_r = split_audit("RESERVE", "RESERVE_MANIFEST.json", item_to_family)
    current_split = {
        "classification": "CURRENT_ROW_BOOTSTRAP_STRUCTURALLY_ANTI_CONSERVATIVE_IF_RHO_POSITIVE",
        "splits": [current_a, current_b, current_r],
        "ruling": (
            "Family sizes are bounded, so unequal weighting is modest; nevertheless row bootstrap "
            "treats shared-program tests as independent and can materially understate uncertainty."
        ),
    }
    write_json(REVIEW / "CURRENT_SPLIT_FAMILY_AUDIT.json", current_split)

    stage_a, stage_b, reserve = h.split_families(items)
    selected_by_family = {item.question_id: item for item in [*stage_a, *stage_b]}
    stage_a_families = {item.question_id for item in stage_a}
    stage_b_families = {item.question_id for item in stage_b}
    reserve_families = set(reserve)
    if stage_a_families & stage_b_families or stage_a_families & reserve_families:
        raise RuntimeError("amended family split is not disjoint")

    def manifest(role: str, selected: list[base.LiveCodeBenchItem]) -> dict[str, Any]:
        return {
            "role": role,
            "scientific_unit": "QUESTION_FAMILY",
            "representative_rule": "minimum frozen stable digest within family",
            "n_families": len(selected),
            "n_selected_rows": len(selected),
            "ordered_families": [
                family_record(item.question_id, groups[item.question_id], selected=item)
                for item in selected
            ],
        }

    write_json(REVIEW / "STAGE_A_FAMILY_MANIFEST.json", manifest("STAGE_A", stage_a))
    write_json(REVIEW / "STAGE_B_FAMILY_MANIFEST.json", manifest("STAGE_B", stage_b))
    write_json(
        REVIEW / "RESERVE_FAMILY_MANIFEST.json",
        {
            "role": "UNALLOCATED_FAMILY_RESERVE",
            "n_families": len(reserve),
            "n_raw_rows": sum(len(rows) for rows in reserve.values()),
            "families": [
                family_record(family_id, rows, selected=None)
                for family_id, rows in reserve.items()
            ],
        },
    )
    siblings = []
    for stage, family_ids in (("STAGE_A", stage_a_families), ("STAGE_B", stage_b_families)):
        for family_id in sorted(family_ids):
            selected = selected_by_family[family_id]
            for item in groups[family_id]:
                if item.item_id != selected.item_id:
                    siblings.append(
                        {
                            "stage_family": stage,
                            "family_id": family_id,
                            "item": item.public_manifest_record(),
                            "status": "STRUCTURAL_SIBLING_EXCLUDED_PRE_OUTCOME",
                        }
                    )
    write_json(
        REVIEW / "EXCLUDED_SIBLING_ROWS_MANIFEST.json",
        {"n_rows": len(siblings), "records": siblings},
    )

    schedule_a = h.build_schedule(
        stage_a,
        stage="STAGE_A",
        conditions=h.STAGE_A_CONDITIONS,
        rollouts=h.STAGE_A_ROLLOUTS,
    )
    schedule_b = h.build_schedule(
        stage_b,
        stage="STAGE_B",
        conditions=h.STAGE_B_CONDITIONS,
        rollouts=h.STAGE_B_ROLLOUTS,
    )
    if len({row["seed"] for row in [*schedule_a, *schedule_b]}) != (
        len(schedule_a) + len(schedule_b)
    ):
        raise RuntimeError("cross-stage amended seed collision")
    write_json(REVIEW / "STAGE_A_SCHEDULE.json", schedule_a)
    write_json(REVIEW / "STAGE_B_SCHEDULE.json", schedule_b)

    current_stage_b_records = read_json(PARENT / "STAGE_B_HOLDOUT_MANIFEST.json")[
        "ordered_records"
    ]
    current_stage_b_counts = Counter(
        item_to_family[row["item_id"]] for row in current_stage_b_records
    )
    current_stage_b_sizes = [int(value) for value in current_stage_b_counts.values()]
    powers = power_grid(groups, current_stage_b_sizes)
    with (REVIEW / "DEPENDENCE_AWARE_POWER_GRID.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(powers[0]))
        writer.writeheader()
        writer.writerows(powers)
    write_json(
        REVIEW / "POWER_METHOD.json",
        {
            "replicates_per_cell": REPLICATES,
            "seed": POWER_SEED,
            "unit_grid": list(UNIT_GRID),
            "rho_grid": list(RHO_GRID),
            "transfer_grid": list(TRANSFER_GRID),
            "random_controls": 8,
            "rollouts": 4,
            "historical_outcomes_used_only_as_planning_inputs": True,
            "livecodebench_outcomes_used": False,
        },
    )

    ordered_family_ids = sorted(
        groups, key=lambda family_id: stable_digest(h.EXPERIMENT_ID, "FAMILY_ORDER", family_id)
    )
    b130_sizes = [len(groups[family_id]) for family_id in ordered_family_ids[32:162]]
    a32_sizes = [len(groups[family_id]) for family_id in ordered_family_ids[:32]]
    runtimes = [
        runtime_record("A_CURRENT_ROW_LEVEL", 200, 6600),
        runtime_record(
            "B_EQUAL_FAMILY_ALL_ROWS_32_130",
            sum(a32_sizes) * 2 * 2,
            sum(b130_sizes) * 11 * 4,
        ),
        runtime_record("C_ONE_ROW_PER_FAMILY_32_130", 32 * 2 * 2, 130 * 11 * 4),
    ]
    write_json(REVIEW / "RUNTIME_STORAGE_COMPARISON.json", runtimes)

    old_names = (
        "PROTOCOL_LOCK.json",
        "PRINCIPAL_REVIEW_LOCK.json",
        "STAGE_A_MANIFEST.json",
        "STAGE_B_HOLDOUT_MANIFEST.json",
        "RESERVE_MANIFEST.json",
        "STAGE_A_SCHEDULE.json",
        "STAGE_B_SCHEDULE.json",
    )
    write_json(
        REVIEW / "PREOUTCOME_SUPERSESSION.json",
        {
            "classification": "OLD_ROW_DESIGN_SUPERSEDED_PRE_OUTCOME_NEVER_EXECUTED",
            "parent_commit": PARENT_COMMIT,
            "benchmark_outcomes_before_amendment": 0,
            "correctness_inspected": False,
            "old_artifacts": {
                name: {
                    "sha256": sha256(PARENT / name),
                    "status": "SUPERSEDED_PRE_OUTCOME_NEVER_EXECUTED",
                }
                for name in old_names
            },
        },
    )

    careful = {
        "classification": "TEXTUAL_CAREFUL_GATE_RETAINED",
        "historical_source": "scripts/analyze_gate10_cross_domain_charcount.py:135-144",
        "commitment_validity_min": 0.95,
        "semantic_evaluability_min": 0.95,
        "competence_floor": "textual accuracy >= baseline accuracy - 0.03",
        "behavioral_or": [
            "accuracy gain >= 0.03",
            "mean tokens >= 1.5 * baseline",
            "median tokens >= baseline + 10",
        ],
        "interpretation": (
            "The gate tests source-policy manifestation plus absence of material competence harm; "
            "token increase alone is not called an accuracy benefit. Gate 10 failed because its "
            "accuracy loss exceeded three points, exactly as this rule requires."
        ),
        "livecodebench_outcomes_used": False,
    }
    write_json(REVIEW / "TEXTUAL_CAREFUL_GATE_REVIEW.json", careful)

    estimators = {
        "error_convention": (
            "e[f,t,r]=1 for wrong or terminal invalid/unevaluable; 0 only for exact correct"
        ),
        "design_a": {
            "estimand": "row-weighted average over test-output rows",
            "bootstrap": "row bootstrap",
            "ruling": "REJECTED_AS_PRIMARY",
            "reason": "shared-program rows are not defensibly independent",
        },
        "design_b": {
            "estimand": "equal-weight average over question families",
            "family_rollout_error": "ebar[f,r]=(1/m_f)*sum_t e[f,t,r]",
            "B00": (
                "mean_f sum_(r!=s) bbar[f,r]*bbar[f,s]/(R*(R-1))"
            ),
            "B0x": "mean_f mean_r(bbar[f,r])*mean_r(xbar[f,r])",
            "G": "B00-B0x",
            "C": "B00-B0x-U00+U0x; U terms exclude identical families",
            "D": (
                "mean_f sum_(r!=s)(bbar[f,r]-xbar[f,r])"
                "*(bbar[f,s]-xbar[f,s])/(R*(R-1))"
            ),
            "rescue": "mean_f mean_r(bbar[f,r])*(1-mean_r(xbar[f,r]))",
            "damage": "mean_f (1-mean_r(bbar[f,r]))*mean_r(xbar[f,r])",
            "bootstrap": "resample families; move every row, condition, and rollout together",
            "ruling": "VALID_BUT_NOT_RECOMMENDED",
        },
        "design_c": {
            "estimand": "equal-weight average over question families",
            "rows_per_family": 1,
            "estimators": "canonical pooled-R4 equations apply without algebraic change",
            "bootstrap": "family bootstrap; equivalent to selected-row bootstrap by construction",
            "ruling": "RECOMMENDED",
        },
        "r4_invariance": {
            "within_condition_products_exclude_self_pairs": True,
            "between_family_products_exclude_same_family": True,
            "r2_reduction_exact": True,
            "negative_D_retained": True,
            "bootstrap_resamples": 50_000,
            "split_halves": {"A": [0, 1], "B": [2, 3]},
        },
    }
    write_json(REVIEW / "HIERARCHICAL_ESTIMATOR_SPEC.json", estimators)

    design_comparison = {
        "design_a": {
            "stage_b": "150 rows / 62 families",
            "scientific_unit": "test-output row",
            "family_weighting": "proportional to family row count",
            "ruling": "REJECTED",
        },
        "design_b": {
            "stage_a": f"{sum(a32_sizes)} rows / 32 families",
            "stage_b": f"{sum(b130_sizes)} rows / 130 families",
            "scientific_unit": "question family",
            "family_weighting": "equal after within-family reduction",
            "ruling": "VALID_BUT_DOMINATED_BY_DESIGN_C",
        },
        "design_c": {
            "stage_a": "32 rows / 32 families",
            "stage_b": "130 rows / 130 families",
            "reserve": "20 complete families",
            "scientific_unit": "question family",
            "family_weighting": "equal",
            "representative_rule": (
                "minimum stable digest of namespace, family ID, and stable item ID"
            ),
            "ruling": "SELECTED_PROSPECTIVELY",
            "selection_rationale": (
                "meets the inherited approximately-80-percent full-transfer planning target, "
                "is invariant to within-family dependence, retains a 20-family reserve, and "
                "uses fewer trajectories than the old row-level design"
            ),
        },
        "selection_used_livecodebench_outcomes": False,
    }
    write_json(REVIEW / "DESIGN_COMPARISON.json", design_comparison)

    amendment = {
        "schema_version": 1,
        "classification": "Q1_SECOND_TASK_HIERARCHICAL_DESIGN_READY_FOR_PRINCIPAL_REVIEW",
        "parent_design_commit": PARENT_COMMIT,
        "reason": "hierarchical dependence discovered during principal design review",
        "benchmark_outcomes_before_amendment": 0,
        "correctness_inspected": False,
        "old_design": {
            "stage_a": "50 rows / 21 families / row bootstrap",
            "stage_b": "150 rows / 62 families / row bootstrap",
        },
        "new_design": {
            "scientific_unit": "QUESTION_FAMILY",
            "estimand_weighting": "equal family weight",
            "representative_rows_per_family": 1,
            "stage_a": {"families": 32, "rows": 32, "rollouts": 2, "trajectories": 128},
            "stage_b": {"families": 130, "rows": 130, "rollouts": 4, "trajectories": 5720},
            "reserve": {"families": 20, "raw_rows": sum(len(rows) for rows in reserve.values())},
            "bootstrap_unit": "QUESTION_FAMILY",
        },
        "stage_a_gate": {
            "baseline_commitment_validity_min": 0.95,
            "baseline_semantic_evaluability_min": 0.95,
            "baseline_accuracy_range_inclusive": [0.25, 0.90],
            "baseline_B00_min": 0.05,
            "families_wrong_both_rollouts_min": 4,
            "families_correct_at_least_once_min": 7,
            "count_translation": (
                "ceil of old row fractions: wrong-both 5/50 -> 4/32; "
                "correct-at-least-once 10/50 -> 7/32"
            ),
            "textual_gate": careful,
        },
        "stage_b_analysis": {
            "pooled_r4_estimators": "unchanged algebra; each selected row is one family",
            "split_halves": {"A": [0, 1], "B": [2, 3]},
            "bootstrap": "50,000 family resamples; all conditions and rollouts move together",
            "primary_and_safety_rules": "unchanged from parent ESTIMATOR_AND_DECISION_LOCK.json",
        },
        "controller_and_null_bank": "unchanged exact parent identities",
        "inherited_lock_hashes": {
            name: sha256(PARENT / name)
            for name in (
                "CONTROLLER_PROVENANCE_LOCK.json",
                "RANDOM_BANK_LOCK.json",
                "ESTIMATOR_AND_DECISION_LOCK.json",
                "SPARK2_ENGINE_QUALIFICATION.json",
                "MODEL_FREE_INSTRUMENT_AUDIT.json",
            )
        },
        "dataset": {
            "repository_revision": base.LIVECODEBENCH_DATASET_REVISION,
            "parquet_sha256": base.LIVECODEBENCH_PARQUET_SHA256,
            "rows": 442,
            "families": 182,
        },
        "spark2_engine": "unchanged qualified environment fingerprint",
        "stage_a_authorized": False,
        "stage_b_authorized": False,
        "q2_outputs_inspected": False,
        "q2_process_modified": False,
        "spark1_used": False,
        "spark2_scientific_inference": False,
    }
    core_names = (
        "FAMILY_STRUCTURE_AUDIT.json",
        "CURRENT_SPLIT_FAMILY_AUDIT.json",
        "STAGE_A_FAMILY_MANIFEST.json",
        "STAGE_B_FAMILY_MANIFEST.json",
        "RESERVE_FAMILY_MANIFEST.json",
        "EXCLUDED_SIBLING_ROWS_MANIFEST.json",
        "STAGE_A_SCHEDULE.json",
        "STAGE_B_SCHEDULE.json",
        "DEPENDENCE_AWARE_POWER_GRID.csv",
        "POWER_METHOD.json",
        "RUNTIME_STORAGE_COMPARISON.json",
        "PREOUTCOME_SUPERSESSION.json",
        "TEXTUAL_CAREFUL_GATE_REVIEW.json",
        "HIERARCHICAL_ESTIMATOR_SPEC.json",
        "DESIGN_COMPARISON.json",
    )
    amendment["artifact_hashes"] = {name: sha256(REVIEW / name) for name in core_names}
    write_json(REVIEW / "AMENDMENT_LOCK.json", amendment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
