"""Fresh namespaces and deterministic non-balanced split generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from epistemic_geometry.reproducibility import stable_seed

from .base import GENERATOR_VERSION, SUITE_VERSION, ReasoningItem
from .families import generate_item

REASONING_CALIBRATION = "REASONING_INSTRUMENT_CALIBRATION"
STAGE_A = "REASONING_STAGE_A_SCREEN"
STAGE_B = "REASONING_STAGE_B_CALIBRATION"
GEOMETRY_CALIBRATION = "GEOMETRY_CALIBRATION"
STEERING_DEVELOPMENT = "STEERING_DEVELOPMENT"
CONFIRMATORY_HOLDOUT = "CONFIRMATORY_HOLDOUT"
DEVELOPMENT_SPLITS = frozenset(
    {
        REASONING_CALIBRATION,
        STAGE_A,
        STAGE_B,
        GEOMETRY_CALIBRATION,
        STEERING_DEVELOPMENT,
    }
)


@dataclass(frozen=True)
class ReasoningSplit:
    split_name: str
    family: str
    cell: str
    seed: int
    items: tuple[ReasoningItem, ...]
    metadata: dict[str, Any]
    reasoning_budget: int | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "split_name": self.split_name,
            "family": self.family,
            "cell": self.cell,
            "suite_version": SUITE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "reasoning_budget": self.reasoning_budget,
            "items": [item.to_record() for item in self.items],
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any], *, development: bool = True) -> ReasoningSplit:
        split_name = str(record["split_name"])
        if development:
            assert_development_access(split_name)
        items = tuple(ReasoningItem.from_record(item) for item in record["items"])
        family = str(record["family"])
        cell = str(record["cell"])
        if any(item.family != family or item.cell != cell for item in items):
            raise ValueError("split contains an item from a different family or cell")
        return cls(
            split_name=split_name,
            family=family,
            cell=cell,
            seed=int(record["seed"]),
            items=items,
            metadata=dict(record.get("metadata", {})),
            reasoning_budget=(
                int(record["reasoning_budget"])
                if record.get("reasoning_budget") is not None
                else None
            ),
        )


def generate_split(
    family: str,
    cell: str,
    split_name: str,
    *,
    seed: int,
    n_items: int,
    reasoning_budget: int | None = None,
) -> ReasoningSplit:
    if split_name not in {
        REASONING_CALIBRATION,
        STAGE_A,
        STAGE_B,
        GEOMETRY_CALIBRATION,
        STEERING_DEVELOPMENT,
        CONFIRMATORY_HOLDOUT,
    }:
        raise ValueError(f"unknown reasoning split: {split_name}")
    items: list[ReasoningItem] = []
    seen: set[str] = set()
    for index in range(n_items):
        item_seed = stable_seed(SUITE_VERSION, split_name, family, cell, seed, index)
        item = generate_item(family, cell, item_seed)
        if item.latent_id in seen:
            raise ValueError(f"duplicate latent ID generated: {item.latent_id}")
        seen.add(item.latent_id)
        items.append(item)
    return ReasoningSplit(
        split_name=split_name,
        family=family,
        cell=cell,
        seed=seed,
        items=tuple(items),
        metadata={
            "development_access": split_name != CONFIRMATORY_HOLDOUT,
            "target_balancing": "not_applied",
            "seed_namespace": f"{SUITE_VERSION}:{split_name}",
        },
        reasoning_budget=reasoning_budget,
    )


def assert_development_access(split_name: str) -> None:
    if split_name == CONFIRMATORY_HOLDOUT:
        raise PermissionError(
            f"ordinary development code cannot access {split_name}; confirmatory firewall active"
        )
    if split_name not in DEVELOPMENT_SPLITS:
        raise ValueError(f"unknown or non-development reasoning split: {split_name}")


def assert_split_disjoint(splits: tuple[ReasoningSplit, ...]) -> None:
    seen: dict[str, str] = {}
    for split in splits:
        for item in split.items:
            previous = seen.setdefault(item.latent_id, split.split_name)
            if previous != split.split_name:
                raise ValueError(
                    f"latent {item.latent_id} overlaps {previous} and {split.split_name}"
                )


def generate_fresh_scientific_splits(
    selected: dict[str, dict[str, Any]], *, seed: int
) -> tuple[ReasoningSplit, ...]:
    """Generate manifests only after explicit instrument qualification."""

    result: list[ReasoningSplit] = []
    for family, choice in sorted(selected.items()):
        cell = choice["cell"]
        for split_name, count in (
            (GEOMETRY_CALIBRATION, 400),
            (STEERING_DEVELOPMENT, 400),
            (CONFIRMATORY_HOLDOUT, 800),
        ):
            result.append(
                generate_split(
                    family,
                    cell,
                    split_name,
                    seed=stable_seed(seed, family, cell, split_name),
                    n_items=count,
                )
            )
    assert_split_disjoint(tuple(result))
    return tuple(result)
