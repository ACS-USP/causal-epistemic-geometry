"""Deterministic, oracle-only balanced splits for E3-10."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from epistemic_geometry.reproducibility import canonical_json, stable_seed

from . import fsm10, modreg10, reachcount10, satcount10
from .base import DIGITS, GENERATOR_VERSION, SUITE_VERSION, LatentItem

FAMILY_CELLS: dict[str, tuple[str, ...]] = {
    "MODREG10": ("depth_4", "depth_8", "depth_12", "depth_16"),
    "FSM10": ("length_4", "length_8", "length_12", "length_16"),
    "REACHCOUNT10": ("H1_p010", "H2_p010", "H2_p018", "H3_p015"),
    "SATCOUNT10": ("vars4_clauses4", "vars4_clauses6", "vars5_clauses8", "vars6_clauses10"),
}
CALIBRATION_SPLIT = "INSTRUMENT_CALIBRATION"
GEOMETRY_SPLIT = "GEOMETRY_CALIBRATION"
DEV_SPLIT = "DEV_EVALUATION"
HOLDOUT_SPLIT = "CONFIRMATORY_HOLDOUT"
# Public descriptive aliases used by manifests and review tooling.
GEOMETRY_CALIBRATION = GEOMETRY_SPLIT
CONFIRMATORY_HOLDOUT = HOLDOUT_SPLIT
DEVELOPMENT_SPLITS = frozenset({CALIBRATION_SPLIT, GEOMETRY_SPLIT, DEV_SPLIT})


@dataclass(frozen=True)
class SplitManifest:
    """A serializable split manifest with an explicit holdout firewall."""

    split_name: str
    suite_version: str
    generator_version: str
    seed: int
    items: tuple[LatentItem, ...]
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        if self.split_name == HOLDOUT_SPLIT and self.metadata.get("development_access", True):
            raise ValueError("confirmatory holdout must not be marked development-accessible")

    def to_record(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "suite_version": self.suite_version,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "items": [item.to_record() for item in self.items],
            "metadata": self.metadata,
        }


def generate_latent(family: str, cell: str, seed: int) -> LatentItem:
    """Dispatch to a family generator using no model-dependent state."""

    if family == "MODREG10":
        return modreg10.generate(seed, cell)
    if family == "FSM10":
        return fsm10.generate(seed, cell)
    if family == "REACHCOUNT10":
        return reachcount10.generate(seed, cell)
    if family == "SATCOUNT10":
        return satcount10.generate(seed, cell)
    raise ValueError(f"unknown E3-10 family: {family}")


def _validate_family_cell(family: str, cell: str) -> None:
    if family not in FAMILY_CELLS:
        raise ValueError(f"unknown E3-10 family: {family}")
    if cell not in FAMILY_CELLS[family]:
        raise ValueError(f"unknown {family} cell: {cell}")


def generate_balanced_items(
    family: str,
    cell: str,
    n_items: int,
    seed: int,
    *,
    split_name: str = CALIBRATION_SPLIT,
    max_attempts: int | None = None,
) -> tuple[LatentItem, ...]:
    """Generate exactly balanced targets using oracle-only rejection sampling.

    Acceptance depends only on the procedural target and latent uniqueness.  No
    model output, confidence, activation, or steering result is consulted.
    """

    _validate_family_cell(family, cell)
    if n_items <= 0 or n_items % 10:
        raise ValueError("n_items must be positive and divisible by 10")
    quotas = {digit: n_items // 10 for digit in DIGITS}
    counts = {digit: 0 for digit in DIGITS}
    accepted: list[LatentItem] = []
    seen: set[str] = set()
    limit = max_attempts or n_items * 20_000
    for attempt in range(limit):
        candidate_seed = stable_seed("E3-10", split_name, family, cell, seed, attempt)
        item = generate_latent(family, cell, candidate_seed)
        if item.latent_hash in seen or counts[item.target] >= quotas[item.target]:
            continue
        accepted.append(item)
        seen.add(item.latent_hash)
        counts[item.target] += 1
        if len(accepted) == n_items:
            return tuple(accepted)
    raise RuntimeError(
        f"could not balance {family}/{cell} after {limit} deterministic attempts; counts={counts}"
    )


def generate_calibration_manifest(
    family: str,
    cell: str,
    *,
    seed: int,
    n_items: int = 200,
) -> SplitManifest:
    items = generate_balanced_items(family, cell, n_items, seed, split_name=CALIBRATION_SPLIT)
    return SplitManifest(
        split_name=CALIBRATION_SPLIT,
        suite_version=SUITE_VERSION,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        items=items,
        metadata={
            "development_access": True,
            "target_balance": {str(d): n_items // 10 for d in DIGITS},
        },
    )


def generate_fresh_split_manifest(
    family: str,
    cell: str,
    split_name: str,
    *,
    seed: int,
    n_items: int,
) -> SplitManifest:
    if split_name not in {GEOMETRY_SPLIT, DEV_SPLIT, HOLDOUT_SPLIT}:
        raise ValueError(f"unsupported fresh split: {split_name}")
    items = generate_balanced_items(family, cell, n_items, seed, split_name=split_name)
    return SplitManifest(
        split_name=split_name,
        suite_version=SUITE_VERSION,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        items=items,
        metadata={
            "development_access": split_name != HOLDOUT_SPLIT,
            "target_balance": {str(d): n_items // 10 for d in DIGITS},
            "firewall": "development code must not load CONFIRMATORY_HOLDOUT",
        },
    )


def assert_split_disjoint(manifests: Iterable[SplitManifest]) -> None:
    seen: dict[str, str] = {}
    for manifest in manifests:
        for item in manifest.items:
            previous = seen.setdefault(item.latent_id, manifest.split_name)
            if previous != manifest.split_name:
                raise ValueError(
                    f"latent item {item.latent_id} appears in {previous} and {manifest.split_name}"
                )


def assert_development_access(split_name: str) -> None:
    if split_name not in DEVELOPMENT_SPLITS:
        raise PermissionError(
            f"development code cannot access {split_name}; confirmatory holdout is firewalled"
        )


def manifest_hash(manifest: SplitManifest) -> str:
    from epistemic_geometry.reproducibility import stable_digest

    return stable_digest("manifest", canonical_json(manifest.to_record()))
