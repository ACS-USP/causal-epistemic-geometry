"""Frozen Stage A/B reasoning-instrument calibration rules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .families import FAMILY_CELLS
from .splits import STAGE_A, STAGE_B, ReasoningSplit, generate_split

REASONING_BUDGETS = (512, 1024, 2048)
STAGE_A_ITEMS = 60
STAGE_A_ROLLOUTS = 2
STAGE_B_ITEMS = 200
STAGE_B_ROLLOUTS = 4
STAGE_A_MIN_ACCURACY = 0.20
STAGE_A_MAX_ACCURACY = 0.90
STAGE_A_MIN_PARSE_SUCCESS = 0.98
STAGE_A_MAX_SEED_GAP = 0.15
STAGE_B_MIN_ACCURACY = 0.30
STAGE_B_MAX_ACCURACY = 0.80
STAGE_B_MIN_PARSE_SUCCESS = 0.99
STAGE_B_MAX_TWIN_ACCURACY_GAP = 0.07
STAGE_B_MIN_TWIN_AGREEMENT = 0.70
STAGE_B_MAX_SEED_SD = 0.07


def eligible_cells_from_gate(gate: dict[str, Any]) -> list[tuple[str, str]]:
    if gate.get("status") != "PASS":
        raise ValueError("Q1 V3 structural gate did not pass")
    failures = set(gate.get("shortcut_failures", [])) | set(
        gate.get("answer_collapse_failures", [])
    )
    all_cells = [(family, cell) for family, cells in FAMILY_CELLS.items() for cell in cells]
    return [(family, cell) for family, cell in all_cells if f"{family}/{cell}" not in failures]


def generate_stage_a_manifests(
    eligible_cells: Iterable[tuple[str, str]], *, seed: int
) -> tuple[ReasoningSplit, ...]:
    manifests: list[ReasoningSplit] = []
    for family, cell in sorted(eligible_cells):
        for budget in REASONING_BUDGETS:
            manifests.append(
                generate_split(
                    family,
                    cell,
                    STAGE_A,
                    seed=seed + budget,
                    n_items=STAGE_A_ITEMS,
                    reasoning_budget=budget,
                )
            )
    return tuple(manifests)


def stage_a_qualifies(outcome: dict[str, Any]) -> bool:
    seed_accuracy = [float(value) for value in outcome["seed_accuracy"]]
    return (
        STAGE_A_MIN_ACCURACY <= float(outcome["mean_accuracy"] or 0.0) <= STAGE_A_MAX_ACCURACY
        and float(outcome["parse_success"]) >= STAGE_A_MIN_PARSE_SUCCESS
        and max(seed_accuracy) - min(seed_accuracy) <= STAGE_A_MAX_SEED_GAP
    )


def select_stage_b_cells(outcomes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    eligible = [outcome for outcome in outcomes if stage_a_qualifies(outcome)]
    selected: dict[str, dict[str, Any]] = {}
    for family in FAMILY_CELLS:
        candidates = [outcome for outcome in eligible if outcome["family"] == family]
        if not candidates:
            continue
        chosen = min(
            candidates,
            key=lambda outcome: (
                abs(float(outcome["mean_accuracy"]) - 0.55),
                int(outcome["reasoning_budget"]),
                _difficulty_rank(str(outcome["cell"])),
            ),
        )
        selected[family] = dict(chosen)
    return selected


def generate_stage_b_manifests(
    selected: dict[str, dict[str, Any]], *, seed: int
) -> tuple[ReasoningSplit, ...]:
    return tuple(
        generate_split(
            family,
            str(outcome["cell"]),
            STAGE_B,
            seed=seed,
            n_items=STAGE_B_ITEMS,
            reasoning_budget=int(outcome["reasoning_budget"]),
        )
        for family, outcome in sorted(selected.items())
    )


def stage_b_qualifies(outcome: dict[str, Any]) -> bool:
    return (
        STAGE_B_MIN_ACCURACY <= float(outcome["mean_accuracy"]) <= STAGE_B_MAX_ACCURACY
        and float(outcome["parse_success"]) >= STAGE_B_MIN_PARSE_SUCCESS
        and abs(float(outcome["canonical_accuracy"]) - float(outcome["twin_accuracy"]))
        <= STAGE_B_MAX_TWIN_ACCURACY_GAP
        and float(outcome["twin_agreement"]) >= STAGE_B_MIN_TWIN_AGREEMENT
        and float(outcome["seed_accuracy_sd"]) <= STAGE_B_MAX_SEED_SD
    )


def select_qualified_families(outcomes: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if stage_b_qualifies(outcome):
            selected[str(outcome["family"])] = dict(outcome)
    return selected


def _difficulty_rank(cell: str) -> int:
    digits = "".join(character for character in cell if character.isdigit())
    return int(digits or "0")
