"""Outcome-free manifest and logical schedule for Q1 finalization timing.

This module deliberately owns only prospective design information.  It does not
read historical files, materialize reasoning splits, or record model outcomes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from epistemic_geometry.reproducibility import stable_digest, stable_seed

from .families import FAMILY_CELLS, generate_item

NAMESPACE = "Q1-V3-FINALIZATION-TIMING-V1"
EXPERIMENT_ID = NAMESPACE
DEFAULT_LATENTS_PER_CELL = 12
CAPS = (4096,)
REASONING_CAPS = CAPS
CONDITIONS = ("BASELINE", "D75")
ROLLOUT_INDICES = (0, 1)
ROLLOUTS = ROLLOUT_INDICES
SEED_REGIME = "MATCHED"


def _record_sequence(value: Any, *, field: str) -> tuple[Mapping[str, Any], ...]:
    """Return records from a schema container without interpreting its source."""

    if isinstance(value, Mapping):
        records = tuple(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records = tuple(value)
    else:
        raise TypeError(f"{field} must be a mapping or sequence")
    if not records:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError(f"{field} entries must be mappings")
    return records  # type: ignore[return-value]


def _latent_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must contain non-empty strings")
    return value


def _manifest_latent_ids(value: Any) -> frozenset[str]:
    manifests = _record_sequence(value, field="manifests")
    extracted: set[str] = set()
    for manifest in manifests:
        if "items" not in manifest:
            raise ValueError("manifest is missing items")
        items = _record_sequence(manifest["items"], field="manifest items")
        ids: list[str] = []
        for item in items:
            if "latent_id" not in item:
                raise ValueError("manifest item is missing latent_id")
            ids.append(_latent_id(item["latent_id"], field="latent_id"))
        if len(set(ids)) != len(ids):
            raise ValueError("manifest items contain duplicate latent IDs")
        extracted.update(ids)
    return frozenset(extracted)


def _paired_group_latent_ids(value: Any) -> frozenset[str]:
    groups = _record_sequence(value, field="paired_budget_groups")
    extracted: set[str] = set()
    for group in groups:
        if "latent_ids" not in group:
            raise ValueError("paired budget group is missing latent_ids")
        latent_ids = group["latent_ids"]
        if not isinstance(latent_ids, Sequence) or isinstance(latent_ids, (str, bytes)):
            raise TypeError("paired_budget_group latent_ids must be a sequence")
        if not latent_ids:
            raise ValueError("paired_budget_group latent_ids must not be empty")
        current = [
            _latent_id(value, field="paired_budget_group latent_ids") for value in latent_ids
        ]
        if len(set(current)) != len(current):
            raise ValueError("paired budget group contains duplicate latent IDs")
        extracted.update(current)
    return frozenset(extracted)


def extract_historical_latent_ids(
    manifest: Mapping[str, Any] | Sequence[Any],
) -> frozenset[str]:
    """Extract latent IDs from an already-loaded Stage A manifest object.

    The extractor is deliberately a pure boundary: callers provide the loaded
    mapping or sequence, and this function performs no external record or
    result access.  It accepts the item-oriented ``manifests`` shape, the
    ``paired_budget_groups`` shape, or both.  IDs are conservatively unioned
    within each representation.  When both are present they must describe the
    same total ID set.
    """

    if isinstance(manifest, Mapping):
        has_manifests = "manifests" in manifest
        has_groups = "paired_budget_groups" in manifest
        if not has_manifests and not has_groups:
            raise ValueError(
                "manifest must contain manifests or paired_budget_groups"
            )
        representations: list[frozenset[str]] = []
        if has_manifests:
            representations.append(_manifest_latent_ids(manifest["manifests"]))
        if has_groups:
            representations.append(_paired_group_latent_ids(manifest["paired_budget_groups"]))
    elif isinstance(manifest, Sequence) and not isinstance(manifest, (str, bytes)):
        representations = [_manifest_latent_ids(manifest)]
    else:
        raise TypeError("manifest must be a mapping or sequence")

    if any(not ids for ids in representations):
        raise ValueError("manifest contains no latent IDs")
    if len(representations) == 2 and representations[0] != representations[1]:
        raise ValueError("manifest representations contain inconsistent latent IDs")
    return representations[0]


def _positive_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("n_per_cell must be a positive integer")
    return value


def _excluded_ids(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise TypeError("excluded_latent_ids must be an iterable of latent ID strings")
    try:
        ids = tuple(values)
    except TypeError as exc:
        raise TypeError("excluded_latent_ids must be an iterable of latent ID strings") from exc
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("excluded_latent_ids must contain non-empty strings")
    if len(set(ids)) != len(ids):
        raise ValueError("excluded_latent_ids contains duplicates")
    return frozenset(ids)


def _candidate_record(family: str, cell: str, seed: int) -> dict[str, Any]:
    """Generate one latent and retain identity metadata only."""

    item = generate_item(family, cell, seed)
    identity_hash = stable_digest(
        NAMESPACE, "LATENT", family, cell, item.latent_id, item.latent_hash, seed
    )
    return {
        "namespace": NAMESPACE,
        "family": family,
        "cell": cell,
        "latent_index": 0,  # replaced while the manifest is assembled
        "generator_seed": seed,
        "latent_seed": seed,
        "latent_id": item.latent_id,
        "latent_hash": item.latent_hash,
        "identity_hash": identity_hash,
        "outcome_free": True,
    }


def build_manifest(
    excluded_latent_ids: Iterable[str] | None = None, *, n_per_cell: int = DEFAULT_LATENTS_PER_CELL
) -> list[dict[str, Any]]:
    """Materialize a deterministic candidate plan with fresh latents per cell.

    ``excluded_latent_ids`` is intentionally caller supplied.  No historical
    split or journal is loaded by this function.  ``n_per_cell`` exists for
    focused tests and must remain positive.
    """

    count = _positive_count(n_per_cell)
    excluded = _excluded_ids(excluded_latent_ids)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    # A finite guard keeps a pathological generator or a fully blocked history
    # from turning a malformed request into an unbounded loop.
    max_candidates = max(10_000, count * 100)
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            selected = 0
            for candidate_index in range(max_candidates):
                seed = stable_seed(NAMESPACE, "LATENT-SEED", family, cell, candidate_index)
                record = _candidate_record(family, cell, seed)
                latent_id = record["latent_id"]
                if latent_id in excluded:
                    continue
                if latent_id in seen:
                    raise ValueError(f"generated latent ID collision: {latent_id}")
                record["latent_index"] = selected
                rows.append(record)
                seen.add(latent_id)
                selected += 1
                if selected == count:
                    break
            if selected != count:
                raise ValueError(
                    f"could not materialize {count} fresh latents for {family}/{cell}"
                )

    manifest_hash = stable_digest(
        NAMESPACE, "MANIFEST", *(record["identity_hash"] for record in rows)
    )
    for record in rows:
        record["manifest_hash"] = manifest_hash
    validate_manifest(rows, n_per_cell=count, excluded_latent_ids=excluded)
    return rows


def materialize_manifest(
    excluded_latent_ids: Iterable[str] | None = None, *, n_per_cell: int = DEFAULT_LATENTS_PER_CELL
) -> list[dict[str, Any]]:
    """Public descriptive alias for :func:`build_manifest`."""

    return build_manifest(excluded_latent_ids, n_per_cell=n_per_cell)


def validate_manifest(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_per_cell: int = DEFAULT_LATENTS_PER_CELL,
    excluded_latent_ids: Iterable[str] | None = None,
) -> None:
    """Validate manifest identity, family/cell coverage, and uniqueness."""

    count = _positive_count(n_per_cell)
    excluded = _excluded_ids(excluded_latent_ids)
    if isinstance(rows, (str, bytes)):
        raise TypeError("manifest must be a sequence of records")
    expected = sum(len(cells) for cells in FAMILY_CELLS.values()) * count
    if len(rows) != expected:
        raise ValueError(f"manifest has {len(rows)} rows, expected {expected}")
    keys: set[str] = set()
    per_cell: dict[tuple[str, str], int] = {}
    hashes: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("manifest rows must be mappings")
        try:
            family, cell = str(row["family"]), str(row["cell"])
            latent_id = row["latent_id"]
            seed = row["generator_seed"]
            latent_hash = row["latent_hash"]
            identity_hash = row["identity_hash"]
        except KeyError as exc:
            raise ValueError(f"manifest row is missing {exc.args[0]}") from exc
        if family not in FAMILY_CELLS or cell not in FAMILY_CELLS[family]:
            raise ValueError(f"unknown reasoning family/cell: {family}/{cell}")
        if not isinstance(latent_id, str) or not latent_id:
            raise ValueError("manifest latent_id must be a non-empty string")
        if latent_id in excluded:
            raise ValueError(f"manifest includes excluded latent ID: {latent_id}")
        if latent_id in keys:
            raise ValueError(f"manifest contains duplicate latent ID: {latent_id}")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("manifest generator_seed must be an integer")
        regenerated = generate_item(family, cell, seed)
        if latent_id != regenerated.latent_id or latent_hash != regenerated.latent_hash:
            raise ValueError(f"manifest latent identity does not match generator seed: {latent_id}")
        if row.get("latent_seed") != regenerated.latent_seed:
            raise ValueError(f"manifest latent_seed does not match generator seed: {latent_id}")
        expected_identity = stable_digest(
            NAMESPACE, "LATENT", family, cell, latent_id, latent_hash, seed
        )
        if identity_hash != expected_identity:
            raise ValueError(f"manifest identity hash mismatch for {latent_id}")
        if row.get("namespace") != NAMESPACE or row.get("outcome_free") is not True:
            raise ValueError("manifest row has the wrong namespace or is not outcome-free")
        keys.add(latent_id)
        hashes.append(str(identity_hash))
        per_cell[(family, cell)] = per_cell.get((family, cell), 0) + 1
    expected_cells = {
        (family, cell) for family, cells in FAMILY_CELLS.items() for cell in cells
    }
    if set(per_cell) != expected_cells:
        raise ValueError("manifest does not cover the frozen family/cell map")
    if set(per_cell.values()) != {count}:
        raise ValueError("manifest has unequal family/cell counts")
    manifest_hashes = {row.get("manifest_hash") for row in rows}
    expected_manifest_hash = stable_digest(NAMESPACE, "MANIFEST", *hashes)
    if manifest_hashes != {expected_manifest_hash}:
        raise ValueError("manifest hash mismatch")


def _manifest_for_schedule(manifest: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    if isinstance(manifest, (str, bytes)):
        raise TypeError("manifest must be a sequence of records")
    rows = tuple(manifest)
    n_cells = sum(len(cells) for cells in FAMILY_CELLS.values())
    if not rows or len(rows) % n_cells:
        raise ValueError("manifest length must be a positive multiple of the frozen cell count")
    validate_manifest(rows, n_per_cell=len(rows) // n_cells)
    return rows


def _rollout_seed(latent_id: str, cap: int, rollout_index: int) -> int:
    return stable_seed(NAMESPACE, "ROLLOUT-SEED", latent_id, cap, rollout_index)


def build_schedule(
    manifest: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the canonical finalization-timing schedule.

    Conditions share a seed for each latent/rollout (matched random numbers).
    The fixed cap and rollout index remain in seed derivation for an auditable
    schedule identity.
    """

    rows = _manifest_for_schedule(build_manifest() if manifest is None else manifest)
    schedule: list[dict[str, Any]] = []
    for latent in rows:
        for cap in CAPS:
            for rollout_index in ROLLOUT_INDICES:
                seed = _rollout_seed(latent["latent_id"], cap, rollout_index)
                for condition in CONDITIONS:
                    schedule_key = (latent["latent_id"], cap, condition, rollout_index)
                    schedule_identity_hash = stable_digest(NAMESPACE, "SCHEDULE", *schedule_key)
                    schedule.append(
                        {
                            "namespace": NAMESPACE,
                            "surface": "canonical",
                            "latent_id": latent["latent_id"],
                            "family": latent["family"],
                            "cell": latent["cell"],
                            "latent_seed": latent["latent_seed"],
                            "latent_hash": latent["latent_hash"],
                            "manifest_hash": latent["manifest_hash"],
                            "cap": cap,
                            "reasoning_budget": cap,
                            "condition": condition,
                            "rollout_index": rollout_index,
                            "seed": seed,
                            "sampling_seed": seed,
                            "seed_regime": SEED_REGIME,
                            "schedule_identity_hash": schedule_identity_hash,
                        }
                    )
    validate_schedule(schedule, rows)
    return schedule


def validate_schedule(
    rows: Sequence[Mapping[str, Any]], manifest: Sequence[Mapping[str, Any]]
) -> None:
    """Validate complete paired timing schedule and seed contract."""

    if isinstance(rows, (str, bytes)):
        raise TypeError("schedule must be a sequence of records")
    manifest_rows = _manifest_for_schedule(manifest)
    expected = len(manifest_rows) * len(CAPS) * len(CONDITIONS) * len(ROLLOUT_INDICES)
    if len(rows) != expected:
        raise ValueError(f"schedule has {len(rows)} rows, expected {expected}")
    keys: set[tuple[str, int, str, int]] = set()
    seeds: dict[tuple[str, int, int], int] = {}
    seed_owners: dict[int, tuple[str, int, int]] = {}
    allowed = {row["latent_id"] for row in manifest_rows}
    for row in rows:
        if row.get("namespace") != NAMESPACE or row.get("surface") != "canonical":
            raise ValueError("schedule row has the wrong namespace or surface")
        key = (row.get("latent_id"), row.get("cap"), row.get("condition"), row.get("rollout_index"))
        if key in keys:
            raise ValueError("schedule contains duplicate logical keys")
        keys.add(key)
        latent_id, cap, condition, rollout = key
        if latent_id not in allowed or cap not in CAPS or condition not in CONDITIONS:
            raise ValueError("schedule contains an unknown logical value")
        if rollout not in ROLLOUT_INDICES:
            raise ValueError("schedule contains an unknown rollout index")
        seed = row.get("seed")
        expected_seed = _rollout_seed(latent_id, cap, rollout)
        if seed != expected_seed or row.get("sampling_seed") != seed:
            raise ValueError("schedule seed does not match the frozen deterministic derivation")
        seed_key = (latent_id, cap, rollout)
        previous = seeds.setdefault(seed_key, seed)
        if previous != seed:
            raise ValueError("schedule has a seed collision")
        owner = seed_owners.setdefault(seed, seed_key)
        if owner != seed_key:
            raise ValueError("schedule has a seed collision across caps or rollouts")
        if row.get("seed_regime") != SEED_REGIME:
            raise ValueError("schedule must use MATCHED seeds")
    expected_keys = {
        (row["latent_id"], cap, condition, rollout)
        for row in manifest_rows
        for cap in CAPS
        for condition in CONDITIONS
        for rollout in ROLLOUT_INDICES
    }
    if keys != expected_keys:
        raise ValueError("schedule is incomplete or contains duplicate logical keys")


__all__ = [
    "CAPS",
    "CONDITIONS",
    "DEFAULT_LATENTS_PER_CELL",
    "EXPERIMENT_ID",
    "NAMESPACE",
    "REASONING_CAPS",
    "ROLLOUT_INDICES",
    "ROLLOUTS",
    "SEED_REGIME",
    "build_manifest",
    "build_schedule",
    "extract_historical_latent_ids",
    "materialize_manifest",
    "validate_manifest",
    "validate_schedule",
]
