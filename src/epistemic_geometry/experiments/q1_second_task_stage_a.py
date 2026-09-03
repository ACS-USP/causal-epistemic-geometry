"""Frozen Stage-A gate for the Q1 LiveCodeBench second-task instrument.

This module contains only the predeclared BASELINE/TEXTUAL_CAREFUL gate.  It
has no activation-controller or null-vector execution path.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

STAGE_A_CONDITIONS = ("BASELINE", "TEXTUAL_CAREFUL")
STAGE_A_FAMILIES = 32
STAGE_A_ROLLOUTS = 2
STAGE_A_LOGICAL_ROWS = 128

BASELINE_VALIDITY_MIN = 0.95
BASELINE_EVALUABILITY_MIN = 0.95
BASELINE_ACCURACY_RANGE = (0.25, 0.90)
BASELINE_B00_MIN = 0.05
BASELINE_WRONG_BOTH_MIN = 4
BASELINE_CORRECT_ONCE_MIN = 7
TEXTUAL_VALIDITY_MIN = 0.95
TEXTUAL_EVALUABILITY_MIN = 0.95
TEXTUAL_NONHARM_MARGIN = 0.03
TEXTUAL_ACCURACY_GAIN_MIN = 0.03
TEXTUAL_MEAN_TOKEN_RATIO_MIN = 1.5
TEXTUAL_MEDIAN_TOKEN_GAIN_MIN = 10.0


def logical_key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["stage"]),
        str(row["item_id"]),
        str(row["condition"]),
        int(row["rollout_index"]),
    )


def validate_schedule(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != STAGE_A_LOGICAL_ROWS:
        raise ValueError("Stage-A schedule must contain exactly 128 rows")
    if {str(row["condition"]) for row in rows} != set(STAGE_A_CONDITIONS):
        raise ValueError("Stage-A schedule contains an unauthorized condition")
    if {str(row["stage"]) for row in rows} != {"STAGE_A"}:
        raise ValueError("Stage-A schedule stage mismatch")
    if {int(row["rollout_index"]) for row in rows} != {0, 1}:
        raise ValueError("Stage-A rollout mismatch")
    keys = [logical_key(row) for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    families = {str(row["family_id"]) for row in rows}
    if len(keys) != len(set(keys)) or len(seeds) != len(set(seeds)):
        raise ValueError("Stage-A schedule has duplicate keys or seeds")
    if len(families) != STAGE_A_FAMILIES:
        raise ValueError("Stage-A schedule must contain exactly 32 families")


def _condition_summary(rows: Sequence[Mapping[str, Any]], condition: str) -> dict[str, float]:
    selected = [row for row in rows if row["condition"] == condition]
    if len(selected) != STAGE_A_FAMILIES * STAGE_A_ROLLOUTS:
        raise ValueError(f"incomplete condition: {condition}")
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
        "minimum_generated_tokens": min(tokens),
        "maximum_generated_tokens": max(tokens),
    }


def stage_a_gate(parsed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the prospectively frozen Stage-A decision mechanically."""

    schedule_projection = [
        {
            "stage": row["stage"],
            "item_id": row["item_id"],
            "family_id": row["family_id"],
            "condition": row["condition"],
            "rollout_index": row["rollout_index"],
            "seed": row["seed"],
        }
        for row in parsed_rows
    ]
    validate_schedule(schedule_projection)
    if len({logical_key(row) for row in parsed_rows}) != STAGE_A_LOGICAL_ROWS:
        raise ValueError("parsed Stage-A rows are not unique")

    baseline = _condition_summary(parsed_rows, "BASELINE")
    textual = _condition_summary(parsed_rows, "TEXTUAL_CAREFUL")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in parsed_rows:
        if row["condition"] == "BASELINE":
            by_family[str(row["family_id"])].append(row)
    if set(map(len, by_family.values())) != {2}:
        raise ValueError("baseline does not contain two rollouts per family")
    wrong_both = sum(
        all(not bool(row["correct"]) for row in values) for values in by_family.values()
    )
    correct_once = sum(any(bool(row["correct"]) for row in values) for values in by_family.values())
    b00 = sum(
        (not bool(values[0]["correct"])) * (not bool(values[1]["correct"]))
        for values in by_family.values()
    ) / len(by_family)

    accuracy_delta = textual["accuracy"] - baseline["accuracy"]
    mean_ratio = (
        textual["mean_generated_tokens"] / baseline["mean_generated_tokens"]
        if baseline["mean_generated_tokens"] > 0
        else float("inf")
    )
    median_delta = textual["median_generated_tokens"] - baseline["median_generated_tokens"]
    manifestations = {
        "TEXTUAL_ACCURACY_GAIN_GE_0_03": accuracy_delta >= TEXTUAL_ACCURACY_GAIN_MIN,
        "TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5": mean_ratio >= TEXTUAL_MEAN_TOKEN_RATIO_MIN,
        "TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10": median_delta >= TEXTUAL_MEDIAN_TOKEN_GAIN_MIN,
    }
    baseline_gates = {
        "commitment_validity_ge_0_95": baseline["commitment_validity"]
        >= BASELINE_VALIDITY_MIN,
        "semantic_evaluability_ge_0_95": baseline["semantic_evaluability"]
        >= BASELINE_EVALUABILITY_MIN,
        "accuracy_in_0_25_0_90": BASELINE_ACCURACY_RANGE[0]
        <= baseline["accuracy"]
        <= BASELINE_ACCURACY_RANGE[1],
        "B00_ge_0_05": b00 >= BASELINE_B00_MIN,
        "families_wrong_both_ge_4": wrong_both >= BASELINE_WRONG_BOTH_MIN,
        "families_correct_once_ge_7": correct_once >= BASELINE_CORRECT_ONCE_MIN,
    }
    textual_gates = {
        "commitment_validity_ge_0_95": textual["commitment_validity"]
        >= TEXTUAL_VALIDITY_MIN,
        "semantic_evaluability_ge_0_95": textual["semantic_evaluability"]
        >= TEXTUAL_EVALUABILITY_MIN,
        "accuracy_nonharm": textual["accuracy"] >= baseline["accuracy"] - TEXTUAL_NONHARM_MARGIN,
        "manifestation_or": any(manifestations.values()),
    }
    qualified = all(baseline_gates.values()) and all(textual_gates.values())
    if manifestations["TEXTUAL_ACCURACY_GAIN_GE_0_03"]:
        manifestation_class = "TEXTUAL_CAREFUL_ACCURACY_BENEFIT_PRESENT"
    elif textual_gates["accuracy_nonharm"] and (
        manifestations["TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5"]
        or manifestations["TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10"]
    ):
        manifestation_class = "TEXTUAL_CAREFUL_NONHARMFUL_COMPUTE_MANIFESTATION"
    else:
        manifestation_class = "TEXTUAL_CAREFUL_NO_QUALIFYING_MANIFESTATION"
    return {
        "classification": (
            "Q1_SECOND_TASK_STAGE_A_QUALIFIED"
            if qualified
            else "Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED"
        ),
        "baseline": {
            **baseline,
            "B00": b00,
            "families_wrong_both_rollouts": wrong_both,
            "families_correct_at_least_once": correct_once,
            "gates": baseline_gates,
        },
        "textual_careful": {
            **textual,
            "textual_accuracy_delta": accuracy_delta,
            "textual_mean_token_ratio": mean_ratio,
            "textual_median_token_delta": median_delta,
            "manifestation_booleans": manifestations,
            "gates": textual_gates,
            "descriptive_manifestation_classification": manifestation_class,
        },
        "stage_b_status": "NOT_AUTHORIZED_NOT_OPENED",
    }


__all__ = [
    "STAGE_A_CONDITIONS",
    "STAGE_A_FAMILIES",
    "STAGE_A_LOGICAL_ROWS",
    "STAGE_A_ROLLOUTS",
    "logical_key",
    "stage_a_gate",
    "validate_schedule",
]
