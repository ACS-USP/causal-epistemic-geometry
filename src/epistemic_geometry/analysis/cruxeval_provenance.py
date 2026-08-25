"""Item-level provenance primitives for the finite CRUXEval repository pool."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

CLASSES = ("A", "B", "C", "D", "UNRESOLVED")


def classify_item(events: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Classify one item using exposure chronology, never outcome values."""

    if not events:
        return "A", "no scientific inference or design exposure found"
    if any(bool(event.get("q2_geometry_discovery")) for event in events):
        return "D", "behavioral geometry directly informed Q2 V2/V3 redesign"
    if any(
        bool(event.get("semantic_outcome_inspected"))
        or bool(event.get("behavioral_outcome_inspected"))
        for event in events
    ):
        return "C", "historical behavioral outcome inspected outside Q2 geometry discovery"
    if any(
        bool(event.get(field))
        for event in events
        for field in (
            "activation_only",
            "source_axis_construction",
            "covariance_geometry_calibration",
            "label_free_generation",
            "reserved_or_allocated",
        )
    ):
        return "B", "representation-only, label-free, or allocation exposure"
    return "UNRESOLVED", "appearance found but exposure semantics are unresolved"


def eligibility_claim(provenance_class: str) -> str:
    return {
        "A": "new-controller/new-item prospective validation",
        "B": "prospective-controller evidence with representation/allocation exposure disclosed",
        "C": "historical-item/prospective-controller validation",
        "D": "retrospective diagnostic only; exclude from primary Q2 V3",
        "UNRESOLVED": "ineligible pending provenance resolution",
    }[provenance_class]


def deterministic_panel(
    rows: Iterable[Mapping[str, Any]], *, allowed_classes: set[str], count: int | None = None
) -> list[dict[str, Any]]:
    """Select by frozen provenance ordering, never by historical performance."""

    eligible = [dict(row) for row in rows if row["provenance_class"] in allowed_classes]
    eligible.sort(key=lambda row: (row["selection_rank"], row["item_id"]))
    return eligible if count is None else eligible[:count]


__all__ = ["CLASSES", "classify_item", "deterministic_panel", "eligibility_claim"]
