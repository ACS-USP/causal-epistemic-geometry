"""Outcome-free manifests and schedules for a new reasoning coverage study.

This module is deliberately separate from :mod:`.splits`.  The names in that
module describe the existing Q1 V3 calibration and scientific split protocol;
they must not be repurposed for a later diversification study.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

from .base import GENERATOR_VERSION, SUITE_VERSION, ReasoningItem
from .families import FAMILY_CELLS, generate_item
from .splits import (
    CONFIRMATORY_HOLDOUT,
    GEOMETRY_CALIBRATION,
    REASONING_CALIBRATION,
    STAGE_A,
    STAGE_B,
    STEERING_DEVELOPMENT,
)

DIVERSIFICATION_COVERAGE_SPLIT = "DIVERSIFICATION_COVERAGE"
DIVERSIFICATION_COVERAGE_NAMESPACE = "Q1-V3-DIVERSIFICATION-COVERAGE-V1"
DIVERSIFICATION_CONDITIONS = ("A", "B", "T")
ROLLOUTS = 2

# These names are protocol history.  Keeping this set private to the new
# constructor makes accidental reuse fail before any item is generated.
HISTORICAL_REASONING_SPLIT_NAMES = frozenset(
    {
        REASONING_CALIBRATION,
        STAGE_A,
        STAGE_B,
        GEOMETRY_CALIBRATION,
        STEERING_DEVELOPMENT,
        CONFIRMATORY_HOLDOUT,
    }
)


def _study_namespace(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("study namespace must be a non-empty string")
    value = value.strip()
    if value in HISTORICAL_REASONING_SPLIT_NAMES:
        raise ValueError(f"historical reasoning split name cannot be a study namespace: {value}")
    return value


def _excluded_ids(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError("excluded_latent_ids must be an iterable of IDs")
    try:
        result = frozenset(str(value) for value in values)
    except TypeError as exc:
        raise ValueError("excluded_latent_ids must be an iterable of IDs") from exc
    if any(not value for value in result):
        raise ValueError("excluded_latent_ids cannot contain empty IDs")
    return result


@dataclass(frozen=True)
class DiversificationManifest:
    """A deterministic, outcome-free manifest in a named new namespace."""

    namespace: str
    family: str
    cell: str
    seed: int
    items: tuple[ReasoningItem, ...]
    excluded_latent_ids: frozenset[str] = frozenset()

    @property
    def split_name(self) -> str:
        return DIVERSIFICATION_COVERAGE_SPLIT

    def __post_init__(self) -> None:
        namespace = _study_namespace(self.namespace)
        excluded = _excluded_ids(self.excluded_latent_ids)
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "excluded_latent_ids", excluded)
        if self.family not in FAMILY_CELLS or self.cell not in FAMILY_CELLS[self.family]:
            raise ValueError(f"unknown reasoning family/cell: {self.family}/{self.cell}")
        if not self.items:
            raise ValueError("diversification manifest must contain at least one item")
        ids = [item.latent_id for item in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError("diversification manifest contains duplicate latent IDs")
        if set(ids) & excluded:
            raise ValueError("diversification manifest collides with excluded latent IDs")
        if any(item.family != self.family or item.cell != self.cell for item in self.items):
            raise ValueError("diversification manifest contains an item from another family/cell")

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "split_name": self.split_name,
            "namespace": self.namespace,
            "family": self.family,
            "cell": self.cell,
            "suite_version": SUITE_VERSION,
            "generator_version": GENERATOR_VERSION,
            "seed": self.seed,
            "excluded_latent_ids": sorted(self.excluded_latent_ids),
            "items": [item.to_record() for item in self.items],
        }
        record["manifest_hash"] = stable_digest(
            "Q1-V3-DIVERSIFICATION-COVERAGE-MANIFEST", canonical_json(record)
        )
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> DiversificationManifest:
        if str(record.get("split_name", "")) != DIVERSIFICATION_COVERAGE_SPLIT:
            raise ValueError("record is not a diversification coverage manifest")
        expected_hash = stable_digest(
            "Q1-V3-DIVERSIFICATION-COVERAGE-MANIFEST",
            canonical_json({key: value for key, value in record.items() if key != "manifest_hash"}),
        )
        if str(record.get("manifest_hash", "")) != expected_hash:
            raise ValueError("diversification manifest hash mismatch")
        if str(record.get("suite_version", SUITE_VERSION)) != SUITE_VERSION:
            raise ValueError("unsupported reasoning suite version")
        if str(record.get("generator_version", GENERATOR_VERSION)) != GENERATOR_VERSION:
            raise ValueError("unsupported reasoning generator version")
        return cls(
            namespace=str(record["namespace"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            seed=int(record["seed"]),
            excluded_latent_ids=_excluded_ids(record.get("excluded_latent_ids", ())),
            items=tuple(ReasoningItem.from_record(item) for item in record["items"]),
        )


def generate_diversification_manifest(
    family: str,
    cell: str,
    *,
    seed: int,
    n_items: int,
    namespace: str = DIVERSIFICATION_COVERAGE_NAMESPACE,
    excluded_latent_ids: Iterable[str] = (),
    max_attempts: int | None = None,
) -> DiversificationManifest:
    """Construct a deterministic fresh manifest without consulting outcomes.

    Candidate IDs in ``excluded_latent_ids`` are rejected and the generator
    advances its deterministic candidate stream.  A duplicate among accepted
    candidates is an error, since silently dropping it would hide a generator
    or namespace defect.
    """

    namespace = _study_namespace(namespace)
    excluded = _excluded_ids(excluded_latent_ids)
    if family not in FAMILY_CELLS or cell not in FAMILY_CELLS[family]:
        raise ValueError(f"unknown reasoning family/cell: {family}/{cell}")
    if n_items <= 0:
        raise ValueError("n_items must be positive")
    limit = max_attempts if max_attempts is not None else n_items * 100
    if limit <= 0:
        raise ValueError("max_attempts must be positive")

    items: list[ReasoningItem] = []
    seen: set[str] = set()
    rejected = 0
    for candidate_index in range(limit):
        item_seed = stable_seed(namespace, "ITEM", family, cell, seed, candidate_index)
        item = generate_item(family, cell, item_seed)
        if item.latent_id in excluded:
            rejected += 1
            continue
        if item.latent_id in seen:
            raise ValueError(f"duplicate latent ID generated in new namespace: {item.latent_id}")
        seen.add(item.latent_id)
        items.append(item)
        if len(items) == n_items:
            manifest = DiversificationManifest(
                namespace=namespace,
                family=family,
                cell=cell,
                seed=seed,
                items=tuple(items),
                excluded_latent_ids=excluded,
            )
            return manifest
    raise RuntimeError(
        f"could not construct {n_items} novel latents after {limit} attempts; "
        f"excluded collisions={rejected}"
    )


def build_diversification_schedule(
    manifest: DiversificationManifest,
    *,
    base_seed: int | None = None,
    conditions: Sequence[str] = DIVERSIFICATION_CONDITIONS,
    n_rollouts: int = ROLLOUTS,
) -> list[dict[str, Any]]:
    """Build and validate a deterministic independent rollout schedule."""

    if not isinstance(manifest, DiversificationManifest):
        raise TypeError("manifest must be a DiversificationManifest")
    condition_ids = tuple(str(condition) for condition in conditions)
    if not condition_ids or len(set(condition_ids)) != len(condition_ids):
        raise ValueError("conditions must be non-empty and unique")
    if any(not condition for condition in condition_ids):
        raise ValueError("conditions must be non-empty")
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be positive")
    seed_root = manifest.seed if base_seed is None else int(base_seed)
    rows: list[dict[str, Any]] = []
    for item in manifest.items:
        for condition in condition_ids:
            for rollout_index in range(n_rollouts):
                rows.append(
                    {
                        "namespace": manifest.namespace,
                        "split_name": manifest.split_name,
                        "family": manifest.family,
                        "cell": manifest.cell,
                        "latent_id": item.latent_id,
                        "condition": condition,
                        "rollout_index": rollout_index,
                        "seed": stable_seed(
                            manifest.namespace,
                            "ROLLOUT",
                            seed_root,
                            item.latent_id,
                            condition,
                            rollout_index,
                        ),
                        "seed_regime": "independent",
                    }
                )
    keys = [
        (row["latent_id"], row["condition"], row["rollout_index"])
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("diversification schedule contains duplicate logical keys")
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("diversification schedule contains a seed collision")
    return rows


__all__ = [
    "DIVERSIFICATION_CONDITIONS",
    "DIVERSIFICATION_COVERAGE_NAMESPACE",
    "DIVERSIFICATION_COVERAGE_SPLIT",
    "DiversificationManifest",
    "HISTORICAL_REASONING_SPLIT_NAMES",
    "build_diversification_schedule",
    "generate_diversification_manifest",
]
