"""Tiny known-structure geometry prompts for the V4 Bench G screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class GeometryItem:
    item_id: str
    domain: str
    entity: str
    offset: int
    answer: str
    conceptual_index: int
    prompt: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["prompt_hash"] = stable_digest("V4-GEOMETRY-PROMPT", self.prompt)
        record["item_hash"] = stable_digest("V4-GEOMETRY-ITEM", canonical_json(record))
        return record


def _weekday_items() -> list[GeometryItem]:
    items: list[GeometryItem] = []
    for index, weekday in enumerate(WEEKDAYS):
        for offset in range(1, 8):
            answer_index = (index + offset) % len(WEEKDAYS)
            prompt = (
                f"What day is {offset} days after {weekday}?\n"
                "Answer with exactly one weekday name."
            )
            items.append(
                GeometryItem(
                    item_id=f"geometry_weekday_{index}_{offset}",
                    domain="WEEKDAYS",
                    entity=weekday,
                    offset=offset,
                    answer=WEEKDAYS[answer_index],
                    conceptual_index=answer_index,
                    prompt=prompt,
                )
            )
    return items


def _letter_items() -> list[GeometryItem]:
    items: list[GeometryItem] = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for index in range(2, len(alphabet)):
        for offset in (1, 2):
            answer_index = index + offset
            if answer_index >= len(alphabet):
                continue
            entity = alphabet[index]
            prompt = (
                "Consider letters in the alphabet. "
                f"Starting at letter {entity}, increment by {offset}. "
                "The result is letter ...\nAnswer with exactly one uppercase letter."
            )
            items.append(
                GeometryItem(
                    item_id=f"geometry_letter_{entity}_{offset}",
                    domain="LETTERS",
                    entity=entity,
                    offset=offset,
                    answer=alphabet[answer_index],
                    conceptual_index=answer_index,
                    prompt=prompt,
                )
            )
    return items


def generate_geometry_manifest() -> dict[str, Any]:
    """Return the fixed weekday and letter prompt set with exact oracles."""

    items = [item.to_record() for item in (*_weekday_items(), *_letter_items())]
    return {
        "suite": "Q1_V4_MICROBENCH",
        "instrument": "GEOMETRY",
        "generator_version": "v4-geometry-1",
        "preset_layer": 31,
        "domains": {
            "WEEKDAYS": {"structure": "cyclic", "size": 7, "items": 49},
            "LETTERS": {"structure": "sequential", "size": 26, "items": 45},
        },
        "items": items,
        "manifest_hash": stable_digest("V4-GEOMETRY-MANIFEST", canonical_json(items)),
    }


def conceptual_distance(domain: str, left: int, right: int) -> int:
    """Compute the frozen conceptual distance for one domain."""

    if domain == "WEEKDAYS":
        return min(abs(left - right), 7 - abs(left - right))
    if domain == "LETTERS":
        return abs(left - right)
    raise ValueError(f"unknown geometry domain: {domain}")
