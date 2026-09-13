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
# Phase namespaces are intentionally derived from the single study namespace.
# Keeping this suffix stable makes phase identity explicit while preventing a
# qualification item from being silently reused as an evaluation item.
DIVERSIFICATION_QUALIFICATION_NAMESPACE = f"{DIVERSIFICATION_COVERAGE_NAMESPACE}:QUALIFICATION"
DIVERSIFICATION_EVALUATION_NAMESPACE = f"{DIVERSIFICATION_COVERAGE_NAMESPACE}:EVALUATION"
DIVERSIFICATION_QUALIFICATION_PHASE = "qualification"
DIVERSIFICATION_EVALUATION_PHASE = "evaluation"
DIVERSIFICATION_CONDITIONS = ("A", "B", "T")
DIVERSIFICATION_QUALIFICATION_CONDITIONS = ("A", "T")
DIVERSIFICATION_EVALUATION_CONDITIONS = DIVERSIFICATION_CONDITIONS
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


@dataclass(frozen=True)
class DiversificationNoveltyReport:
    """Deterministic result of an outcome-free historical ID audit."""

    candidate_namespace: str
    candidate_split_name: str
    candidate_latent_ids: tuple[str, ...]
    historical_manifest_names: tuple[str, ...]
    historical_latent_ids: tuple[str, ...]
    historical_duplicate_ids: tuple[str, ...]
    historical_collisions: tuple[dict[str, Any], ...]
    candidate_collisions: tuple[dict[str, Any], ...]
    candidate_disjoint: bool
    passed: bool

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible, deterministic report record."""

        return {
            "candidate_namespace": self.candidate_namespace,
            "candidate_split_name": self.candidate_split_name,
            "candidate_latent_ids": list(self.candidate_latent_ids),
            "historical_manifest_names": list(self.historical_manifest_names),
            "historical_latent_ids": list(self.historical_latent_ids),
            "historical_duplicate_ids": list(self.historical_duplicate_ids),
            "historical_collisions": [dict(collision) for collision in self.historical_collisions],
            "candidate_collisions": [dict(collision) for collision in self.candidate_collisions],
            "candidate_disjoint": self.candidate_disjoint,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class DiversificationCoveragePlan:
    """Pure, deterministic qualification/evaluation coverage plan.

    The plan contains only procedurally constructed items and schedule
    metadata.  It does not read historical files or outcomes: callers provide
    historical latent IDs explicitly through ``historical_excluded_latent_ids``.
    ``cell_counts`` retains the caller's family/cell order as a tuple.
    """

    study_seed: int
    cell_counts: tuple[tuple[str, str, int, int], ...]
    historical_excluded_latent_ids: frozenset[str]
    qualification_manifests: tuple[DiversificationManifest, ...]
    evaluation_manifests: tuple[DiversificationManifest, ...]
    schedule_counts: dict[str, int]
    schedule_keys: dict[str, tuple[tuple[str, str, int], ...]]

    @property
    def phase_namespaces(self) -> dict[str, str]:
        """Return the explicit namespace used by each phase."""

        return {
            DIVERSIFICATION_QUALIFICATION_PHASE: DIVERSIFICATION_QUALIFICATION_NAMESPACE,
            DIVERSIFICATION_EVALUATION_PHASE: DIVERSIFICATION_EVALUATION_NAMESPACE,
        }

    @property
    def phases(self) -> dict[str, tuple[DiversificationManifest, ...]]:
        """Return phase manifests keyed by their stable phase names."""

        return {
            DIVERSIFICATION_QUALIFICATION_PHASE: self.qualification_manifests,
            DIVERSIFICATION_EVALUATION_PHASE: self.evaluation_manifests,
        }

    @property
    def qualification_latent_ids(self) -> tuple[str, ...]:
        return tuple(
            item.latent_id
            for manifest in self.qualification_manifests
            for item in manifest.items
        )

    @property
    def evaluation_latent_ids(self) -> tuple[str, ...]:
        return tuple(
            item.latent_id
            for manifest in self.evaluation_manifests
            for item in manifest.items
        )

    @property
    def all_latent_ids(self) -> tuple[str, ...]:
        return self.qualification_latent_ids + self.evaluation_latent_ids

    @property
    def passed(self) -> bool:
        """Whether both phase sets are disjoint and exclude history."""

        history = set(self.historical_excluded_latent_ids)
        qualification = set(self.qualification_latent_ids)
        evaluation = set(self.evaluation_latent_ids)
        return not (
            qualification & history
            or evaluation & history
            or qualification & evaluation
        )

    def to_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible plan/report record."""

        record: dict[str, Any] = {
            "study_seed": self.study_seed,
            "namespace": DIVERSIFICATION_COVERAGE_NAMESPACE,
            "split_name": DIVERSIFICATION_COVERAGE_SPLIT,
            "phase_namespaces": dict(self.phase_namespaces),
            "cell_counts": [
                {
                    "family": family,
                    "cell": cell,
                    "qualification": qualification_count,
                    "evaluation": evaluation_count,
                }
                for family, cell, qualification_count, evaluation_count in self.cell_counts
            ],
            "historical_excluded_latent_ids": sorted(self.historical_excluded_latent_ids),
            "qualification_manifests": [
                manifest.to_record() for manifest in self.qualification_manifests
            ],
            "evaluation_manifests": [
                manifest.to_record() for manifest in self.evaluation_manifests
            ],
            "qualification_latent_ids": list(self.qualification_latent_ids),
            "evaluation_latent_ids": list(self.evaluation_latent_ids),
            "all_latent_ids": list(self.all_latent_ids),
            "schedule_counts": dict(self.schedule_counts),
            "schedule_keys": {
                phase: [list(key) for key in keys]
                for phase, keys in self.schedule_keys.items()
            },
            "passed": self.passed,
        }
        record["plan_hash"] = stable_digest(
            "Q1-V3-DIVERSIFICATION-COVERAGE-PLAN", canonical_json(record)
        )
        return record


def _count_pair(value: Any, *, family: str, cell: str) -> tuple[int, int]:
    """Normalize one cell count to ``(qualification, evaluation)``."""

    if isinstance(value, Mapping):
        aliases = {
            "qualification": "qualification",
            "qualification_count": "qualification",
            "n_qualification": "qualification",
            "evaluation": "evaluation",
            "evaluation_count": "evaluation",
            "n_evaluation": "evaluation",
        }
        counts: dict[str, Any] = {}
        for key, count in value.items():
            phase = aliases.get(str(key))
            if phase is None:
                raise ValueError(
                    f"unknown count field for reasoning family/cell: {family}/{cell}: {key}"
                )
            if phase in counts:
                raise ValueError(
                    f"duplicate {phase} count for reasoning family/cell: {family}/{cell}"
                )
            counts[phase] = count
        if set(counts) != {"qualification", "evaluation"}:
            raise ValueError(
                f"counts for reasoning family/cell must specify qualification and evaluation: "
                f"{family}/{cell}"
            )
        value = (counts["qualification"], counts["evaluation"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise ValueError(
                "counts for reasoning family/cell must be an int or a two-item pair: "
                f"{family}/{cell}"
            )
    else:
        value = (value, value)

    result: list[int] = []
    for count in value:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(
                f"counts must be positive integers for reasoning family/cell: {family}/{cell}"
            )
        result.append(count)
    return result[0], result[1]


def _ordered_cell_counts(
    family_cell_counts: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str, int, int], ...]:
    """Validate a nested ordered ``family -> cell -> counts`` mapping."""

    if isinstance(family_cell_counts, (str, bytes)) or not isinstance(family_cell_counts, Mapping):
        raise ValueError("family_cell_counts must be a non-empty family-to-cell mapping")
    if not family_cell_counts:
        raise ValueError("family_cell_counts must be non-empty")

    result: list[tuple[str, str, int, int]] = []
    for family, cells in family_cell_counts.items():
        if not isinstance(family, str) or not family:
            raise ValueError("reasoning family names must be non-empty strings")
        if family in HISTORICAL_REASONING_SPLIT_NAMES:
            raise ValueError(
                f"historical reasoning split name cannot be used as a family: {family}"
            )
        if family not in FAMILY_CELLS:
            raise ValueError(f"unknown reasoning family: {family}")
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Mapping) or not cells:
            raise ValueError(f"cells for reasoning family must be a non-empty mapping: {family}")
        for cell, counts in cells.items():
            if not isinstance(cell, str) or not cell:
                raise ValueError("reasoning cell names must be non-empty strings")
            if cell not in FAMILY_CELLS[family]:
                raise ValueError(f"unknown reasoning family/cell: {family}/{cell}")
            qualification_count, evaluation_count = _count_pair(
                counts, family=family, cell=cell
            )
            result.append((family, cell, qualification_count, evaluation_count))
    if not result:
        raise ValueError("family_cell_counts must contain at least one cell")
    return tuple(result)


def plan_diversification_coverage(
    family_cell_counts: Mapping[str, Mapping[str, Any]],
    *,
    seed: int | None = None,
    study_seed: int | None = None,
    historical_excluded_latent_ids: Iterable[str] = (),
    qualification_conditions: Sequence[str] = DIVERSIFICATION_QUALIFICATION_CONDITIONS,
    evaluation_conditions: Sequence[str] = DIVERSIFICATION_EVALUATION_CONDITIONS,
    n_rollouts: int = ROLLOUTS,
) -> DiversificationCoveragePlan:
    """Plan disjoint qualification and evaluation manifests by family/cell.

    ``family_cell_counts`` is an ordered nested mapping.  Each value may be a
    positive integer (used for both phases), a ``(qualification, evaluation)``
    pair, or a mapping with explicit ``qualification`` and ``evaluation``
    fields.  The only exclusion inputs are the caller-provided historical IDs;
    no files, journals, outcomes, model inference, or remote services are
    consulted.  Qualification schedules default to A/T; evaluation schedules
    default to A/B/T and each phase can be configured independently.
    """

    if seed is None and study_seed is None:
        raise ValueError("study seed is required")
    if seed is not None and study_seed is not None and int(seed) != int(study_seed):
        raise ValueError("seed and study_seed must agree when both are supplied")
    study_seed_value = int(seed if seed is not None else study_seed)
    ordered_counts = _ordered_cell_counts(family_cell_counts)
    historical = _excluded_ids(historical_excluded_latent_ids)
    if n_rollouts <= 0:
        raise ValueError("n_rollouts must be positive")
    for phase, conditions_for_phase in (
        (DIVERSIFICATION_QUALIFICATION_PHASE, qualification_conditions),
        (DIVERSIFICATION_EVALUATION_PHASE, evaluation_conditions),
    ):
        condition_ids = tuple(str(condition) for condition in conditions_for_phase)
        if not condition_ids or len(set(condition_ids)) != len(condition_ids):
            raise ValueError(f"{phase} conditions must be non-empty and unique")
        if any(not condition for condition in condition_ids):
            raise ValueError(f"{phase} conditions must be non-empty")

    qualification_seed = stable_seed(
        DIVERSIFICATION_COVERAGE_NAMESPACE,
        DIVERSIFICATION_QUALIFICATION_PHASE,
        study_seed_value,
    )
    evaluation_seed = stable_seed(
        DIVERSIFICATION_COVERAGE_NAMESPACE,
        DIVERSIFICATION_EVALUATION_PHASE,
        study_seed_value,
    )

    qualification: list[DiversificationManifest] = []
    qualification_ids: set[str] = set()
    for family, cell, qualification_count, _evaluation_count in ordered_counts:
        manifest = generate_diversification_manifest(
            family,
            cell,
            seed=qualification_seed,
            n_items=qualification_count,
            namespace=DIVERSIFICATION_QUALIFICATION_NAMESPACE,
            excluded_latent_ids=historical,
        )
        qualification.append(manifest)
        qualification_ids.update(item.latent_id for item in manifest.items)

    evaluation_excluded = historical | qualification_ids
    evaluation: list[DiversificationManifest] = []
    for family, cell, _qualification_count, evaluation_count in ordered_counts:
        manifest = generate_diversification_manifest(
            family,
            cell,
            seed=evaluation_seed,
            n_items=evaluation_count,
            namespace=DIVERSIFICATION_EVALUATION_NAMESPACE,
            excluded_latent_ids=evaluation_excluded,
        )
        evaluation.append(manifest)

    # Building schedules here validates logical keys and independent rollout
    # seeds while returning only compact counts and keys in the plan.
    schedules = {
        DIVERSIFICATION_QUALIFICATION_PHASE: [
            row
            for manifest in qualification
            for row in build_diversification_schedule(
                manifest,
                conditions=qualification_conditions,
                n_rollouts=n_rollouts,
            )
        ],
        DIVERSIFICATION_EVALUATION_PHASE: [
            row
            for manifest in evaluation
            for row in build_diversification_schedule(
                manifest,
                conditions=evaluation_conditions,
                n_rollouts=n_rollouts,
            )
        ],
    }
    schedule_counts = {phase: len(rows) for phase, rows in schedules.items()}
    schedule_keys = {
        phase: tuple(
            (str(row["latent_id"]), str(row["condition"]), int(row["rollout_index"]))
            for row in rows
        )
        for phase, rows in schedules.items()
    }
    plan = DiversificationCoveragePlan(
        study_seed=study_seed_value,
        cell_counts=ordered_counts,
        historical_excluded_latent_ids=historical,
        qualification_manifests=tuple(qualification),
        evaluation_manifests=tuple(evaluation),
        schedule_counts=schedule_counts,
        schedule_keys=schedule_keys,
    )
    if not plan.passed:
        raise RuntimeError("diversification coverage plan failed phase disjointness checks")
    return plan


# Descriptive aliases keep the constructor discoverable for callers that use
# "construct" or "build" for pure planning APIs.
construct_diversification_coverage_plan = plan_diversification_coverage
build_diversification_coverage_plan = plan_diversification_coverage


def _historical_records(records: Iterable[Mapping[str, Any]]) -> list[tuple[str, tuple[str, ...]]]:
    """Validate and extract IDs from explicitly supplied historical records."""

    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("historical_manifest_records must be an iterable of mappings")
    try:
        raw_records = list(records)
    except TypeError as exc:
        raise ValueError("historical_manifest_records must be an iterable of mappings") from exc

    normalized: list[tuple[str, str, tuple[str, ...]]] = []
    for record_index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"historical manifest record {record_index} must be a mapping")
        items = record.get("items")
        if isinstance(items, (str, bytes, Mapping)) or not isinstance(items, Sequence):
            raise ValueError(f"historical manifest record {record_index} has malformed items")
        if not items:
            raise ValueError(f"historical manifest record {record_index} must contain items")

        ids: list[str] = []
        for item_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(
                    f"historical manifest record {record_index} item {item_index} must be a mapping"
                )
            latent_id = item.get("latent_id")
            if not isinstance(latent_id, str) or not latent_id:
                raise ValueError(
                    f"historical manifest record {record_index} item {item_index} "
                    "has a malformed latent_id"
                )
            ids.append(latent_id)

        label_value = record.get("split_name", record.get("namespace"))
        label = label_value if isinstance(label_value, str) and label_value else ""
        canonical = canonical_json(record)
        if not label:
            digest = stable_digest("Q1-V3-HISTORICAL-MANIFEST", canonical)[:16]
            label = f"historical-{digest}"
        normalized.append((label, canonical, tuple(ids)))

    normalized.sort(key=lambda row: (row[0], row[1]))
    result: list[tuple[str, tuple[str, ...]]] = []
    label_counts: dict[str, int] = {}
    for label, _canonical, ids in normalized:
        count = label_counts.get(label, 0) + 1
        label_counts[label] = count
        result.append((label if count == 1 else f"{label}#{count}", ids))
    return result


def audit_diversification_novelty(
    historical_manifest_records: Iterable[Mapping[str, Any]],
    candidate: DiversificationManifest | Mapping[str, Any],
) -> DiversificationNoveltyReport:
    """Audit candidate latent IDs against explicit historical manifest records.

    Historical records are trusted, caller-provided mappings.  Their only
    scientific content used here is each item's ``latent_id``.  A candidate
    record is fully reconstructed so a stale or tampered manifest hash fails
    closed before any candidate IDs are considered.
    """

    historical = _historical_records(historical_manifest_records)
    if isinstance(candidate, DiversificationManifest):
        candidate_manifest = candidate
    elif isinstance(candidate, Mapping):
        try:
            candidate_manifest = DiversificationManifest.from_record(candidate)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"candidate diversification manifest record is malformed: {exc}"
            ) from exc
    else:
        raise TypeError("candidate must be a DiversificationManifest or manifest record")

    candidate_ids = tuple(item.latent_id for item in candidate_manifest.items)
    occurrences: dict[str, list[tuple[str, int]]] = {}
    for manifest_name, ids in historical:
        for item_index, latent_id in enumerate(ids):
            occurrences.setdefault(latent_id, []).append((manifest_name, item_index))

    historical_duplicate_ids = tuple(
        sorted(latent_id for latent_id, places in occurrences.items() if len(places) > 1)
    )
    historical_collisions: list[dict[str, Any]] = []
    for latent_id in historical_duplicate_ids:
        places = occurrences[latent_id]
        manifest_names = tuple(dict.fromkeys(name for name, _index in places))
        kind = "within_manifest" if len(manifest_names) == 1 else "across_manifests"
        historical_collisions.append(
            {
                "kind": kind,
                "latent_id": latent_id,
                "manifest_names": list(manifest_names),
                "occurrences": [
                    {"manifest_name": name, "item_index": index} for name, index in places
                ],
            }
        )

    historical_id_set = set(occurrences)
    candidate_collisions = tuple(
        {
            "kind": "candidate_historical",
            "latent_id": latent_id,
            "historical_manifest_names": [
                name for name, ids in historical if latent_id in ids
            ],
        }
        for latent_id in sorted(set(candidate_ids) & historical_id_set)
    )

    historical_latent_ids = tuple(sorted(historical_id_set))
    candidate_disjoint = not bool(set(candidate_ids) & historical_id_set)
    passed = candidate_disjoint and not historical_duplicate_ids
    return DiversificationNoveltyReport(
        candidate_namespace=candidate_manifest.namespace,
        candidate_split_name=candidate_manifest.split_name,
        candidate_latent_ids=tuple(candidate_ids),
        historical_manifest_names=tuple(name for name, _ids in historical),
        historical_latent_ids=historical_latent_ids,
        historical_duplicate_ids=historical_duplicate_ids,
        historical_collisions=tuple(historical_collisions),
        candidate_collisions=tuple(candidate_collisions),
        candidate_disjoint=candidate_disjoint,
        passed=passed,
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
    "DIVERSIFICATION_EVALUATION_NAMESPACE",
    "DIVERSIFICATION_EVALUATION_CONDITIONS",
    "DIVERSIFICATION_EVALUATION_PHASE",
    "DIVERSIFICATION_QUALIFICATION_CONDITIONS",
    "DIVERSIFICATION_QUALIFICATION_NAMESPACE",
    "DIVERSIFICATION_QUALIFICATION_PHASE",
    "DiversificationCoveragePlan",
    "DiversificationManifest",
    "DiversificationNoveltyReport",
    "HISTORICAL_REASONING_SPLIT_NAMES",
    "audit_diversification_novelty",
    "build_diversification_coverage_plan",
    "build_diversification_schedule",
    "construct_diversification_coverage_plan",
    "generate_diversification_manifest",
    "plan_diversification_coverage",
]
