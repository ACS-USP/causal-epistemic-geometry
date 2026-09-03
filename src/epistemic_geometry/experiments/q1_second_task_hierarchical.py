"""Prospective family-unit Amendment 1 for the Q1 LiveCodeBench design.

This module is model-free. It preserves the original fixed controller,
conditions, rollout policy, and estimands while replacing test-row sampling by
one deterministic representative row per independent question family.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from epistemic_geometry.experiments import q1_second_task as base
from epistemic_geometry.reproducibility import stable_digest, stable_seed

EXPERIMENT_ID = "Q1_SECOND_TASK_LIVECODEBENCH_SPARK2_AMENDMENT1"
STAGE_A_FAMILIES = 32
STAGE_B_FAMILIES = 130
RESERVE_FAMILIES = 20
STAGE_A_ROLLOUTS = 2
STAGE_B_ROLLOUTS = 4
STAGE_A_CONDITIONS = base.STAGE_A_CONDITIONS
STAGE_B_CONDITIONS = base.STAGE_B_CONDITIONS


def group_families(
    items: Sequence[base.LiveCodeBenchItem],
) -> dict[str, list[base.LiveCodeBenchItem]]:
    groups: dict[str, list[base.LiveCodeBenchItem]] = {}
    for item in items:
        groups.setdefault(item.question_id, []).append(item)
    for family in groups.values():
        family.sort(key=lambda item: item.item_id)
    return groups


def representative_row(
    family_id: str, rows: Sequence[base.LiveCodeBenchItem]
) -> base.LiveCodeBenchItem:
    """Select one row with one frozen outcome-independent hash rule."""

    return min(
        rows,
        key=lambda item: stable_digest(
            EXPERIMENT_ID, "REPRESENTATIVE_ROW", family_id, item.item_id
        ),
    )


def split_families(
    items: Sequence[base.LiveCodeBenchItem],
) -> tuple[
    list[base.LiveCodeBenchItem],
    list[base.LiveCodeBenchItem],
    dict[str, list[base.LiveCodeBenchItem]],
]:
    groups = group_families(items)
    if len(groups) != STAGE_A_FAMILIES + STAGE_B_FAMILIES + RESERVE_FAMILIES:
        raise RuntimeError("official pool must contain exactly 182 question families")
    ordered_families = sorted(
        groups, key=lambda family_id: stable_digest(EXPERIMENT_ID, "FAMILY_ORDER", family_id)
    )
    stage_a_ids = ordered_families[:STAGE_A_FAMILIES]
    stage_b_ids = ordered_families[
        STAGE_A_FAMILIES : STAGE_A_FAMILIES + STAGE_B_FAMILIES
    ]
    reserve_ids = ordered_families[STAGE_A_FAMILIES + STAGE_B_FAMILIES :]

    def selected(family_ids: Sequence[str], stage: str) -> list[base.LiveCodeBenchItem]:
        values = [representative_row(family_id, groups[family_id]) for family_id in family_ids]
        return sorted(
            values,
            key=lambda item: stable_digest(EXPERIMENT_ID, stage, "ITEM_ORDER", item.item_id),
        )

    stage_a = selected(stage_a_ids, "STAGE_A")
    stage_b = selected(stage_b_ids, "STAGE_B")
    reserve = {family_id: groups[family_id] for family_id in reserve_ids}
    return stage_a, stage_b, reserve


def build_schedule(
    items: Sequence[base.LiveCodeBenchItem],
    *,
    stage: str,
    conditions: Sequence[str],
    rollouts: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        for rollout in range(rollouts):
            ordered_conditions = sorted(
                conditions,
                key=lambda condition: stable_digest(
                    EXPERIMENT_ID,
                    stage,
                    "CONDITION_ORDER",
                    item.item_id,
                    rollout,
                    condition,
                ),
            )
            for condition_order, condition in enumerate(ordered_conditions):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "stage": stage,
                        "family_id": item.question_id,
                        "item_id": item.item_id,
                        "item_sha256": item.item_sha256,
                        "condition": condition,
                        "rollout_index": rollout,
                        "condition_order": condition_order,
                        "seed": stable_seed(
                            EXPERIMENT_ID, stage, item.item_id, condition, rollout
                        ),
                        "seed_regime": "INDEPENDENT_PRIMARY",
                        "scientific_unit": "QUESTION_FAMILY",
                    }
                )
    validate_schedule(
        rows, items, stage=stage, conditions=conditions, rollouts=rollouts
    )
    return rows


def validate_schedule(
    rows: Sequence[Mapping[str, Any]],
    items: Sequence[base.LiveCodeBenchItem],
    *,
    stage: str,
    conditions: Sequence[str],
    rollouts: int,
) -> None:
    expected = len(items) * len(conditions) * rollouts
    keys = [
        (
            str(row["stage"]),
            str(row["family_id"]),
            str(row["condition"]),
            int(row["rollout_index"]),
        )
        for row in rows
    ]
    seeds = [int(row["seed"]) for row in rows]
    if len(rows) != expected or len(keys) != len(set(keys)):
        raise RuntimeError("Amendment-1 schedule has a row/key mismatch")
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("Amendment-1 schedule has seed collisions")
    if {str(row["stage"]) for row in rows} != {stage}:
        raise RuntimeError("Amendment-1 schedule stage mismatch")
    if {str(row["condition"]) for row in rows} != set(conditions):
        raise RuntimeError("Amendment-1 schedule condition mismatch")
    if len({item.question_id for item in items}) != len(items):
        raise RuntimeError("more than one selected row belongs to a family")


__all__ = [
    "EXPERIMENT_ID",
    "RESERVE_FAMILIES",
    "STAGE_A_CONDITIONS",
    "STAGE_A_FAMILIES",
    "STAGE_A_ROLLOUTS",
    "STAGE_B_CONDITIONS",
    "STAGE_B_FAMILIES",
    "STAGE_B_ROLLOUTS",
    "build_schedule",
    "group_families",
    "representative_row",
    "split_families",
    "validate_schedule",
]
