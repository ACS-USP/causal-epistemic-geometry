"""Prospective Stage-A2 lock for the Q1 LiveCodeBench second-task instrument."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from epistemic_geometry.experiments import q1_second_task as base
from epistemic_geometry.experiments import q1_second_task_hierarchical as amendment1
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "Q1_SECOND_TASK_LIVECODEBENCH_SPARK2_AMENDMENT2_STAGE_A2"
STAGE = "STAGE_A2"
FAMILIES = 20
ROLLOUTS = 2
CONDITIONS = base.STAGE_A_CONDITIONS
LOGICAL_ROWS = FAMILIES * ROLLOUTS * len(CONDITIONS)

BASELINE_VALIDITY_MIN = 0.95
BASELINE_EVALUABILITY_MIN = 0.95
BASELINE_ACCURACY_RANGE = (0.25, 0.90)
BASELINE_B00_MIN = 0.05
BASELINE_WRONG_BOTH_MIN = 2
BASELINE_CORRECT_ONCE_MIN = 4
TEXTUAL_VALIDITY_MIN = 0.95
TEXTUAL_EVALUABILITY_MIN = 0.95
TEXTUAL_NONHARM_MARGIN = 0.03
TEXTUAL_ACCURACY_GAIN_MIN = 0.03
TEXTUAL_MEAN_TOKEN_RATIO_MIN = 1.5
TEXTUAL_MEDIAN_TOKEN_GAIN_MIN = 10.0


def selected_reserve_families(reserve: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Apply the already-frozen Amendment-1 representative-row rule to reserve."""

    selected: list[dict[str, Any]] = []
    for family in reserve["families"]:
        pairs = list(zip(family["all_item_ids"], family["all_item_sha256"], strict=True))
        item_id, item_sha256 = min(
            pairs,
            key=lambda pair: stable_digest(
                amendment1.EXPERIMENT_ID,
                "REPRESENTATIVE_ROW",
                family["family_id"],
                pair[0],
            ),
        )
        selected.append(
            {
                **family,
                "selected_item": {"item_id": item_id, "item_sha256": item_sha256},
                "selection_digest": stable_digest(
                    amendment1.EXPERIMENT_ID,
                    "REPRESENTATIVE_ROW",
                    family["family_id"],
                    item_id,
                ),
            }
        )
    if len(selected) != FAMILIES or len({row["family_id"] for row in selected}) != FAMILIES:
        raise ValueError("Stage-A2 requires exactly 20 unique reserve families")
    return selected


def build_schedule(families: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        item = family["selected_item"]
        for rollout in range(ROLLOUTS):
            conditions = sorted(
                CONDITIONS,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID,
                    STAGE,
                    "CONDITION_ORDER",
                    item["item_id"],
                    rollout,
                    condition,
                ),
            )
            for condition_order, condition in enumerate(conditions):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "stage": STAGE,
                        "family_id": family["family_id"],
                        "item_id": item["item_id"],
                        "item_sha256": item["item_sha256"],
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": condition_order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, STAGE, item["item_id"], condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                        "scientific_unit": "QUESTION_FAMILY",
                    }
                )
    validate_schedule(rows)
    return rows


def logical_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["stage"]),
        str(row["family_id"]),
        str(row["condition"]),
        int(row["rollout_index"]),
    )


def validate_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [logical_key(row) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != LOGICAL_ROWS or len(keys) != len(set(keys)):
        raise ValueError("Stage-A2 schedule must contain 80 unique logical rows")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Stage-A2 schedule contains duplicate seeds")
    if {str(row["stage"]) for row in rows} != {STAGE}:
        raise ValueError("Stage-A2 schedule stage mismatch")
    if {str(row["condition"]) for row in rows} != set(CONDITIONS):
        raise ValueError("Stage-A2 schedule condition mismatch")
    if len({str(row["family_id"]) for row in rows}) != FAMILIES:
        raise ValueError("Stage-A2 schedule family mismatch")


def _summary(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == condition]
    if len(selected) != FAMILIES * ROLLOUTS:
        raise ValueError(f"incomplete Stage-A2 condition: {condition}")
    tokens = [int(row["generated_token_count"]) for row in selected]
    return {
        "n": len(selected),
        "commitment_validity": sum(bool(row["commitment_valid"]) for row in selected)
        / len(selected),
        "semantic_evaluability": sum(bool(row["semantic_evaluable"]) for row in selected)
        / len(selected),
        "accuracy": sum(bool(row["correct"]) for row in selected) / len(selected),
        "mean_generated_tokens": sum(tokens) / len(tokens),
        "median_generated_tokens": float(median(tokens)),
    }


def stage_a2_gate(parsed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validate_schedule(parsed_rows)
    baseline = _summary(parsed_rows, "BASELINE")
    textual = _summary(parsed_rows, "TEXTUAL_CAREFUL")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        if row["condition"] == "BASELINE":
            by_family[str(row["family_id"])].append(row)
    if set(map(len, by_family.values())) != {2}:
        raise ValueError("Stage-A2 baseline requires two rollouts per family")
    wrong_both = sum(all(not bool(row["correct"]) for row in rows) for rows in by_family.values())
    correct_once = sum(any(bool(row["correct"]) for row in rows) for rows in by_family.values())
    b00 = wrong_both / FAMILIES
    delta = textual["accuracy"] - baseline["accuracy"]
    mean_ratio = textual["mean_generated_tokens"] / baseline["mean_generated_tokens"]
    median_delta = textual["median_generated_tokens"] - baseline["median_generated_tokens"]
    manifestations = {
        "TEXTUAL_ACCURACY_GAIN_GE_0_03": delta >= TEXTUAL_ACCURACY_GAIN_MIN,
        "TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5": mean_ratio >= TEXTUAL_MEAN_TOKEN_RATIO_MIN,
        "TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10": median_delta >= TEXTUAL_MEDIAN_TOKEN_GAIN_MIN,
    }
    gates = {
        "baseline_commitment_validity": baseline["commitment_validity"]
        >= BASELINE_VALIDITY_MIN,
        "baseline_semantic_evaluability": baseline["semantic_evaluability"]
        >= BASELINE_EVALUABILITY_MIN,
        "baseline_accuracy": BASELINE_ACCURACY_RANGE[0]
        <= baseline["accuracy"]
        <= BASELINE_ACCURACY_RANGE[1],
        "baseline_B00": b00 >= BASELINE_B00_MIN,
        "baseline_wrong_both": wrong_both >= BASELINE_WRONG_BOTH_MIN,
        "baseline_correct_once": correct_once >= BASELINE_CORRECT_ONCE_MIN,
        "textual_commitment_validity": textual["commitment_validity"]
        >= TEXTUAL_VALIDITY_MIN,
        "textual_semantic_evaluability": textual["semantic_evaluability"]
        >= TEXTUAL_EVALUABILITY_MIN,
        "textual_nonharm": textual["accuracy"] >= baseline["accuracy"] - TEXTUAL_NONHARM_MARGIN,
        "textual_manifestation": any(manifestations.values()),
    }
    return {
        "classification": (
            "Q1_SECOND_TASK_STAGE_A2_QUALIFIED"
            if all(gates.values())
            else "Q1_SECOND_TASK_STAGE_A2_NOT_QUALIFIED"
        ),
        "baseline": {
            **baseline,
            "B00": b00,
            "families_wrong_both_rollouts": wrong_both,
            "families_correct_at_least_once": correct_once,
        },
        "textual_careful": {
            **textual,
            "textual_accuracy_delta": delta,
            "textual_mean_token_ratio": mean_ratio,
            "textual_median_token_delta": median_delta,
            "manifestations": manifestations,
        },
        "gates": gates,
        "stage_b_status": "CLOSED_NOT_AUTHORIZED",
    }


__all__ = [
    "CONDITIONS",
    "EXPERIMENT_ID",
    "FAMILIES",
    "LOGICAL_ROWS",
    "ROLLOUTS",
    "STAGE",
    "build_schedule",
    "logical_key",
    "selected_reserve_families",
    "stage_a2_gate",
    "validate_schedule",
]
