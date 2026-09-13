"""Pure A/B/T coverage endpoint and its planning bootstrap.

The endpoint treats a latent as the sampling unit.  A, B, and T each have
two independent correctness rollouts.  Coverage is the probability that at
least one of the relevant rollouts is correct, estimated from binary
correctness indicators.  Cross-condition coverage averages the four ordered
rollout pairs before taking the complement.

This module deliberately has no model, network, Spark, or outcome
dependencies.  It only consumes already-computed correctness indicators.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import floor
from numbers import Integral
from random import Random
from types import MappingProxyType
from typing import Any

CONDITIONS = ("A", "B", "T")
VALUE_NAMES = ("V_AA", "V_BB", "V_TT", "V_AB", "V_AT")
DEFAULT_LOWER_PERCENTILE = 2.5
DEFAULT_FAMILY_CELLS = MappingProxyType(
    {
        "MODREG-R": ("depth_4", "depth_8", "depth_12", "depth_16"),
        "FSM-R": ("length_4", "length_8", "length_12", "length_16"),
        "SATCOUNT-R": (
            "vars4_clauses4",
            "vars4_clauses6",
            "vars5_clauses8",
            "vars6_clauses10",
        ),
    }
)


def _binary(value: Any, name: str) -> int:
    """Coerce a strict binary correctness indicator."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral) and int(value) in (0, 1):
        return int(value)
    raise ValueError(f"{name} must be binary (0 or 1)")


def _identity(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, Integral)):
        raise ValueError(f"{name} must be a string or integer")
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _find(row: Mapping[str, Any], names: Sequence[str], description: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"row is missing {description}")


def _rollout(row: Mapping[str, Any], condition: str, index: int) -> Any:
    lower = condition.lower()
    upper = condition.upper()
    names = (
        f"{lower}{index}",
        f"{upper}{index}",
        f"{lower}_rollout_{index}",
        f"{upper}_rollout_{index}",
        f"{lower}_response_{index}",
        f"{upper}_response_{index}",
        f"{lower}_{index}",
        f"{upper}_{index}",
    )
    for name in names:
        if name in row:
            return row[name]
    # A nested [rollout_1, rollout_2] or {1: ..., 2: ...} representation is
    # convenient for callers that already group a condition's rollouts.
    for container_name in (condition, upper, lower, f"{lower}_rollouts", f"{upper}_rollouts"):
        if container_name not in row:
            continue
        values = row[container_name]
        if isinstance(values, Mapping):
            for nested_name in ("correct", "correctness", "rollouts", "responses"):
                if nested_name in values:
                    nested_values = values[nested_name]
                    if isinstance(nested_values, Sequence) and not isinstance(
                        nested_values, (str, bytes)
                    ) and len(nested_values) == 2:
                        return nested_values[index - 1]
            for key in (index, str(index), f"rollout_{index}"):
                if key in values:
                    return values[key]
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            if len(values) == 2:
                return values[index - 1]
    raise ValueError(f"row is missing {condition} rollout {index}")


@dataclass(frozen=True)
class ABTObservation:
    """One latent with two correctness rollouts for each A/B/T condition."""

    family: str
    cell: str
    latent: str
    a1: int
    a2: int
    b1: int
    b2: int
    t1: int
    t2: int

    def __post_init__(self) -> None:
        for field in ("a1", "a2", "b1", "b2", "t1", "t2"):
            value = _binary(getattr(self, field), field)
            object.__setattr__(self, field, value)
        for field in ("family", "cell", "latent"):
            object.__setattr__(self, field, _identity(getattr(self, field), field))


# A descriptive alias makes the type easy to discover without colliding with
# the legacy A/T-only CoverageObservation in coverage_qualification.py.
CoverageEndpointObservation = ABTObservation


@dataclass(frozen=True)
class CompositionValidation:
    """Result of validating the frozen family/cell/latent composition."""

    passed: bool
    reasons: tuple[str, ...]
    n_rows: int
    families: tuple[str, ...]
    cells_by_family: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "n_rows": self.n_rows,
            "families": list(self.families),
            "cells_by_family": {
                family: list(cells) for family, cells in self.cells_by_family.items()
            },
        }


def _coerce(row: ABTObservation | Mapping[str, Any]) -> ABTObservation:
    if isinstance(row, ABTObservation):
        return row
    if not isinstance(row, Mapping):
        raise ValueError("each row must be an ABTObservation or mapping")
    return ABTObservation(
        family=_find(row, ("family", "family_id"), "family"),
        cell=_find(row, ("cell", "cell_id"), "cell"),
        latent=_find(row, ("latent", "latent_id", "latent_index", "id"), "latent"),
        a1=_rollout(row, "A", 1),
        a2=_rollout(row, "A", 2),
        b1=_rollout(row, "B", 1),
        b2=_rollout(row, "B", 2),
        t1=_rollout(row, "T", 1),
        t2=_rollout(row, "T", 2),
    )


def _normalise(rows: Iterable[ABTObservation | Mapping[str, Any]]) -> list[ABTObservation]:
    if isinstance(rows, (str, bytes)):
        raise ValueError("rows must be an iterable of observations")
    try:
        result = [_coerce(row) for row in rows]
    except TypeError as exc:
        raise ValueError("rows must be an iterable of observations") from exc
    return sorted(result, key=lambda row: (row.family, row.cell, row.latent))


def _expected_map(
    expected_cells: Mapping[str, Sequence[str]] | Sequence[str] | None,
    families: Sequence[str] | None,
    cells_per_family: int | None,
) -> dict[str, tuple[str, ...]]:
    if expected_cells is None:
        result = {family: tuple(cells) for family, cells in DEFAULT_FAMILY_CELLS.items()}
    elif isinstance(expected_cells, Mapping):
        result = {
            str(family): tuple(str(cell) for cell in cells)
            for family, cells in expected_cells.items()
        }
    else:
        if families is None:
            raise ValueError("families is required when expected_cells is a shared sequence")
        result = {str(family): tuple(str(cell) for cell in expected_cells) for family in families}
    if families is not None:
        requested = tuple(str(family) for family in families)
        if len(set(requested)) != len(requested):
            raise ValueError("families must be unique")
        result = {family: result.get(family, ()) for family in requested}
    if cells_per_family is not None:
        if cells_per_family <= 0:
            raise ValueError("cells_per_family must be positive")
        if any(len(cells) != cells_per_family for cells in result.values()):
            raise ValueError("expected cell composition does not match cells_per_family")
    if not result or any(not cells for cells in result.values()):
        raise ValueError("expected family/cell composition must be non-empty")
    if any(len(set(cells)) != len(cells) for cells in result.values()):
        raise ValueError("expected cells must be unique within each family")
    return result


def validate_abt_composition(
    rows: Iterable[ABTObservation | Mapping[str, Any]],
    *,
    expected_cells: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    n_latents_per_cell: int | None = None,
    cells_per_family: int | None = None,
) -> CompositionValidation:
    """Check explicit families, cells, and unique latent identities.

    Duplicate identities and malformed rows raise ``ValueError``.  Missing or
    extra rows are represented by a failed result so callers can inspect all
    composition diagnostics at once.
    """

    normalised = _normalise(rows)
    expected = _expected_map(expected_cells, families, cells_per_family)
    reasons: list[str] = []
    identities = [(row.family, row.cell, row.latent) for row in normalised]
    seen: set[tuple[str, str, str]] = set()
    duplicates: list[tuple[str, str, str]] = []
    for identity in identities:
        if identity in seen and identity not in duplicates:
            duplicates.append(identity)
        seen.add(identity)
    if duplicates:
        raise ValueError(f"duplicate latent identities: {duplicates[:3]}")

    found_families = tuple(sorted({row.family for row in normalised}))
    expected_families = tuple(sorted(expected))
    if found_families != expected_families:
        reasons.append(f"expected families {list(expected_families)}, found {list(found_families)}")
    cells_by_family: dict[str, tuple[str, ...]] = {}
    for family in found_families:
        found_cells = tuple(sorted({row.cell for row in normalised if row.family == family}))
        cells_by_family[family] = found_cells
        required_cells = tuple(sorted(expected.get(family, ())))
        if found_cells != required_cells:
            reasons.append(
                f"family {family!r} expected cells {list(required_cells)}, "
                f"found {list(found_cells)}"
            )
        for cell in found_cells:
            count = sum(row.family == family and row.cell == cell for row in normalised)
            if n_latents_per_cell is not None and count != n_latents_per_cell:
                reasons.append(
                    f"cell {family!r}/{cell!r} expected {n_latents_per_cell} latents, found {count}"
                )
    if n_latents_per_cell is not None:
        if n_latents_per_cell <= 0:
            raise ValueError("n_latents_per_cell must be positive")
        expected_rows = sum(len(cells) for cells in expected.values()) * n_latents_per_cell
        if len(normalised) != expected_rows:
            reasons.append(f"expected {expected_rows} rows, found {len(normalised)}")
    return CompositionValidation(
        passed=not reasons,
        reasons=tuple(reasons),
        n_rows=len(normalised),
        families=found_families,
        cells_by_family=cells_by_family,
    )


def _metric_values(row: ABTObservation) -> dict[str, float]:
    errors = {
        "A": (1 - row.a1, 1 - row.a2),
        "B": (1 - row.b1, 1 - row.b2),
        "T": (1 - row.t1, 1 - row.t2),
    }
    same = {
        "V_AA": 1.0 - errors["A"][0] * errors["A"][1],
        "V_BB": 1.0 - errors["B"][0] * errors["B"][1],
        "V_TT": 1.0 - errors["T"][0] * errors["T"][1],
    }
    cross = {
        "V_AB": 1.0
        - sum(a_error * b_error for a_error in errors["A"] for b_error in errors["B"]) / 4.0,
        "V_AT": 1.0
        - sum(a_error * t_error for a_error in errors["A"] for t_error in errors["T"]) / 4.0,
    }
    return {**same, **cross}


def _mean(rows: Sequence[ABTObservation]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty group")
    values = [_metric_values(row) for row in rows]
    return {name: sum(item[name] for item in values) / len(values) for name in VALUE_NAMES}


def _aggregate(rows: Sequence[ABTObservation]) -> tuple[dict[str, float], dict[str, Any]]:
    groups: dict[tuple[str, str], list[ABTObservation]] = {}
    for row in rows:
        groups.setdefault((row.family, row.cell), []).append(row)
    cell_values = {
        f"{family}/{cell}": _mean(group)
        for (family, cell), group in sorted(groups.items())
    }
    family_values: dict[str, dict[str, float]] = {}
    for family in sorted({row.family for row in rows}):
        cells = [cell_values[f"{family}/{cell}"] for cell in sorted(
            cell for fam, cell in groups if fam == family
        )]
        family_values[family] = {
            name: sum(value[name] for value in cells) / len(cells) for name in VALUE_NAMES
        }
    families = [family_values[family] for family in sorted(family_values)]
    overall = {name: sum(value[name] for value in families) / len(families) for name in VALUE_NAMES}
    family_names = sorted(family_values)
    cell_weights = {
        family: {
            cell: 1.0 / sum(fam == family for fam, _ in groups)
            for fam, cell in sorted(groups)
            if fam == family
        }
        for family in family_names
    }
    metadata = {
        "n_latents": len(rows),
        "n_cells": len(cell_values),
        "n_families": len(family_values),
        "weights": {
            "latent_within_cell": "equal",
            "cell_within_family": cell_weights,
            "family_global": {family: 1.0 / len(family_names) for family in family_names},
        },
        "cell_values": cell_values,
        "family_values": family_values,
    }
    return overall, metadata


def _contrasts(values: Mapping[str, float]) -> dict[str, float]:
    control_mean = (values["V_AA"] + values["V_BB"]) / 2.0
    return {
        "AB_minus_AA": values["V_AB"] - values["V_AA"],
        "AB_minus_BB": values["V_AB"] - values["V_BB"],
        "AB_minus_AT": values["V_AB"] - values["V_AT"],
        "AB_minus_TT": values["V_AB"] - values["V_TT"],
        "AB_minus_control": values["V_AB"] - control_mean,
    }


@dataclass(frozen=True)
class BootstrapReport:
    """Planning bootstrap summary using latent clusters stratified by cell."""

    n_replicates: int
    seed: int | None
    lower_percentile: float
    lower_bounds: dict[str, float]
    estimate: dict[str, float]
    note: str = (
        "Planning implementation: percentile lower bounds are descriptive and are not a final "
        "inferential guarantee."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "lower_percentile": self.lower_percentile,
            "lower_bounds": dict(self.lower_bounds),
            "estimate": dict(self.estimate),
            "note": self.note,
        }


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def bootstrap_coverage(
    rows: Iterable[ABTObservation | Mapping[str, Any]],
    *,
    n_replicates: int = 2000,
    seed: int | None = None,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
    expected_cells: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    n_latents_per_cell: int | None = None,
    cells_per_family: int | None = None,
) -> BootstrapReport:
    """Resample complete latent rows within each cell and reaggregate each draw."""

    if n_replicates <= 0:
        raise ValueError("n_replicates must be positive")
    if not 0 <= lower_percentile <= 100:
        raise ValueError("lower_percentile must be between 0 and 100")
    normalised = _normalise(rows)
    composition = validate_abt_composition(
        normalised,
        expected_cells=expected_cells,
        families=families,
        n_latents_per_cell=n_latents_per_cell,
        cells_per_family=cells_per_family,
    )
    if not composition.passed:
        raise ValueError("incomplete composition: " + "; ".join(composition.reasons))
    groups: dict[tuple[str, str], list[ABTObservation]] = {}
    for row in normalised:
        groups.setdefault((row.family, row.cell), []).append(row)
    rng = Random(seed)
    replicate_values = {name: [] for name in VALUE_NAMES}
    for _ in range(n_replicates):
        sample = [rng.choice(group) for _, group in sorted(groups.items()) for _ in group]
        values, _ = _aggregate(sample)
        for name in VALUE_NAMES:
            replicate_values[name].append(values[name])
    estimate, _ = _aggregate(normalised)
    lower_bounds = {
        name: _percentile(values, lower_percentile)
        for name, values in replicate_values.items()
    }
    # G and contrasts are derived after aggregation in every replicate.  This
    # is the key protection against accidentally taking a latent-level max.
    derived = {
        "G": [],
        **{name: [] for name in _contrasts(estimate)},
    }
    for index in range(n_replicates):
        values = {name: replicate_values[name][index] for name in VALUE_NAMES}
        derived["G"].append(
            values["V_AB"]
            - max(values[name] for name in ("V_AA", "V_BB", "V_AT", "V_TT"))
        )
        for name, value in _contrasts(values).items():
            derived[name].append(value)
    lower_bounds.update(
        {name: _percentile(values, lower_percentile) for name, values in derived.items()}
    )
    return BootstrapReport(n_replicates, seed, lower_percentile, lower_bounds, estimate)


@dataclass(frozen=True)
class CoverageEndpointReport:
    """Structured A/B/T endpoint report."""

    values: dict[str, float]
    contrasts: dict[str, float]
    g: float
    aggregation: dict[str, Any]
    composition: CompositionValidation
    per_latent: tuple[dict[str, Any], ...]
    bootstrap: BootstrapReport | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": dict(self.values),
            "Vs": dict(self.values),
            **self.values,
            "contrasts": dict(self.contrasts),
            "AB_contrasts": dict(self.contrasts),
            "AB_control": self.contrasts["AB_minus_control"],
            "G": self.g,
            "aggregation": dict(self.aggregation),
            "weights": self.aggregation["weights"],
            "composition": self.composition.to_dict(),
            "per_latent": [dict(row) for row in self.per_latent],
            "bootstrap": None if self.bootstrap is None else self.bootstrap.to_dict(),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    @property
    def G(self) -> float:
        """Uppercase spelling used by the study notation."""

        return self.g

    def __getattr__(self, name: str) -> Any:
        if name in VALUE_NAMES:
            return self.values[name]
        raise AttributeError(name)


def calculate_coverage_endpoint(
    rows: Iterable[ABTObservation | Mapping[str, Any]],
    *,
    expected_cells: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    n_latents_per_cell: int | None = None,
    cells_per_family: int | None = None,
    bootstrap_replicates: int | None = None,
    n_bootstrap: int | None = None,
    seed: int | None = None,
    lower_percentile: float = DEFAULT_LOWER_PERCENTILE,
) -> CoverageEndpointReport:
    """Calculate the frozen A/B/T endpoint and optional planning bootstrap."""

    normalised = _normalise(rows)
    composition = validate_abt_composition(
        normalised,
        expected_cells=expected_cells,
        families=families,
        n_latents_per_cell=n_latents_per_cell,
        cells_per_family=cells_per_family,
    )
    if not composition.passed:
        raise ValueError("incomplete composition: " + "; ".join(composition.reasons))
    values, aggregation = _aggregate(normalised)
    contrasts = _contrasts(values)
    g = values["V_AB"] - max(values[name] for name in ("V_AA", "V_BB", "V_AT", "V_TT"))
    per_latent = tuple(
        {"family": row.family, "cell": row.cell, "latent": row.latent, **_metric_values(row)}
        for row in normalised
    )
    requested_bootstrap = bootstrap_replicates if bootstrap_replicates is not None else n_bootstrap
    bootstrap = None
    if requested_bootstrap is not None:
        bootstrap = bootstrap_coverage(
            normalised,
            n_replicates=requested_bootstrap,
            seed=seed,
            lower_percentile=lower_percentile,
            expected_cells=expected_cells,
            families=families,
            n_latents_per_cell=n_latents_per_cell,
            cells_per_family=cells_per_family,
        )
    return CoverageEndpointReport(
        values, contrasts, g, aggregation, composition, per_latent, bootstrap
    )


# Explicit aliases ease use from notebooks and keep the endpoint discoverable.
compute_coverage_endpoint = calculate_coverage_endpoint
calculate_abt_endpoint = calculate_coverage_endpoint
coverage_endpoint = calculate_coverage_endpoint
analyze_abt = calculate_coverage_endpoint
analyze_coverage = calculate_coverage_endpoint
bootstrap_endpoint = bootstrap_coverage
validate_composition = validate_abt_composition

__all__ = [
    "ABTObservation",
    "BootstrapReport",
    "CONDITIONS",
    "CoverageEndpointObservation",
    "CoverageEndpointReport",
    "DEFAULT_FAMILY_CELLS",
    "DEFAULT_LOWER_PERCENTILE",
    "VALUE_NAMES",
    "analyze_abt",
    "analyze_coverage",
    "bootstrap_coverage",
    "bootstrap_endpoint",
    "calculate_abt_endpoint",
    "calculate_coverage_endpoint",
    "compute_coverage_endpoint",
    "coverage_endpoint",
    "validate_abt_composition",
    "validate_composition",
]
