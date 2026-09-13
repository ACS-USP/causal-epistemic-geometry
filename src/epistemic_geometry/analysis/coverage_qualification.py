"""Small, outcome-only adequacy gate for paired A/T coverage.

The gate is a qualification check for an instrument, not the primary
inference or a power calculation.  It assumes one row per latent unit, with
two binary correctness outcomes for canonical view A, two for twin view T,
and an arbitrary hashable oracle answer.  The fixed composition is three
families, four cells per family, and eight latents per cell.  Rows are treated
as independent latent observations for Hoeffding bounds; the two rollouts
within a row are averaged and do not create extra statistical units.

``alpha`` is the total gate error budget.  Half is allocated to six
family/view lower bounds (epsilon ``alpha / 12`` each), and half to three
global opportunity-control upper bounds (epsilon ``alpha / 6`` each).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import log, sqrt
from types import MappingProxyType
from typing import Any, Literal

VIEWS = ("A", "T")
EXPECTED_FAMILIES = 3
EXPECTED_CELLS_PER_FAMILY = 4
EXPECTED_LATENTS_PER_CELL = 8
FAMILY_CELLS = MappingProxyType(
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
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"{name} must be binary (0 or 1)")


@dataclass(frozen=True)
class CoverageObservation:
    """One latent, carrying two correctness rollouts for each A/T view."""

    family: str
    cell: str
    latent: str
    a1: int
    a2: int
    t1: int
    t2: int
    answer: Any

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family:
            raise ValueError("family must be a non-empty string")
        for name, value in (
            ("a1", self.a1),
            ("a2", self.a2),
            ("t1", self.t1),
            ("t2", self.t2),
        ):
            _binary(value, name)
        try:
            hash(self.answer)
        except TypeError as exc:
            raise ValueError("answer must be hashable") from exc


@dataclass(frozen=True)
class CompositionReport:
    """Strict composition result with useful diagnostics."""

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


@dataclass(frozen=True)
class QualificationReport:
    """Structured PASS/FAIL result for the coverage adequacy gate."""

    passed: bool
    status: Literal["PASS", "FAIL"]
    reasons: tuple[str, ...]
    delta: float
    alpha: float
    composition: CompositionReport
    competence: dict[str, dict[str, dict[str, Any]]]
    opportunity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "delta": self.delta,
            "alpha": self.alpha,
            "composition": self.composition.to_dict(),
            "competence": {
                family: {view: dict(metric) for view, metric in views.items()}
                for family, views in self.competence.items()
            },
            "opportunity": dict(self.opportunity),
        }

    def __getitem__(self, key: str) -> Any:
        """Allow report access in the same style as its serialized form."""

        return self.to_dict()[key]


def _check_probability(name: str, value: float, *, allow_zero: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number")
    if (result <= 0 and not allow_zero) or result < 0 or result >= 1:
        lower = "0 <=" if allow_zero else "0 <"
        raise ValueError(f"{name} must satisfy {lower} {name} < 1")
    return result


def _value(row: Mapping[str, Any], names: Sequence[str], description: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValueError(f"row is missing {description}")


def _coerce(row: CoverageObservation | Mapping[str, Any]) -> CoverageObservation:
    if isinstance(row, CoverageObservation):
        return CoverageObservation(
            family=row.family,
            cell=str(row.cell),
            latent=str(row.latent),
            a1=_binary(row.a1, "a1"),
            a2=_binary(row.a2, "a2"),
            t1=_binary(row.t1, "t1"),
            t2=_binary(row.t2, "t2"),
            answer=row.answer,
        )
    if not isinstance(row, Mapping):
        raise ValueError("each coverage row must be a CoverageObservation or mapping")
    family = _value(row, ("family", "family_id"), "family")
    cell = _value(row, ("cell", "cell_id"), "cell")
    latent = _value(row, ("latent", "latent_id", "latent_index"), "latent")
    if not isinstance(family, str) or not family:
        raise ValueError("family must be a non-empty string")
    if not isinstance(cell, (str, int)) or isinstance(cell, bool):
        raise ValueError("cell must be a string or integer")
    if not isinstance(latent, (str, int)) or isinstance(latent, bool):
        raise ValueError("latent must be a string or integer")
    answer = _value(row, ("answer", "label", "oracle", "oracle_response", "target"), "answer")
    try:
        hash(answer)
    except TypeError as exc:
        raise ValueError("answer must be hashable") from exc
    return CoverageObservation(
        family=family,
        cell=str(cell),
        latent=str(latent),
        a1=_binary(_value(row, ("a1", "A1", "a_response_1"), "A rollout 1"), "a1"),
        a2=_binary(_value(row, ("a2", "A2", "a_response_2"), "A rollout 2"), "a2"),
        t1=_binary(_value(row, ("t1", "T1", "t_response_1"), "T rollout 1"), "t1"),
        t2=_binary(_value(row, ("t2", "T2", "t_response_2"), "T rollout 2"), "t2"),
        answer=answer,
    )


def _normalise_rows(
    rows: Iterable[CoverageObservation | Mapping[str, Any]],
) -> list[CoverageObservation]:
    if isinstance(rows, (str, bytes)):
        raise ValueError("rows must be an iterable of coverage observations")
    try:
        return [_coerce(row) for row in rows]
    except TypeError as exc:
        raise ValueError("rows must be an iterable of coverage observations") from exc


def validate_composition(
    rows: Iterable[CoverageObservation | Mapping[str, Any]],
    *,
    n_families: int = EXPECTED_FAMILIES,
    n_cells_per_family: int = EXPECTED_CELLS_PER_FAMILY,
    n_latents_per_cell: int = EXPECTED_LATENTS_PER_CELL,
    expected_cells: Mapping[str, Sequence[str]] | None = None,
) -> CompositionReport:
    """Validate exact, unique family/cell/latent composition.

    Malformed rows and duplicate ``(family, cell, latent)`` identities raise
    ``ValueError``.  A valid iterable with missing or extra rows returns a
    failed report.
    """

    normalised = _normalise_rows(rows)
    reasons: list[str] = []
    if n_families <= 0 or n_cells_per_family <= 0 or n_latents_per_cell <= 0:
        raise ValueError("composition sizes must be positive")
    identities = [(row.family, row.cell, row.latent) for row in normalised]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"duplicate latent identities: {duplicates[:3]}")

    expected_source = FAMILY_CELLS if expected_cells is None else expected_cells
    expected_map = {family: tuple(cells) for family, cells in expected_source.items()}
    families = tuple(sorted({row.family for row in normalised}))
    expected_family_names = tuple(sorted(expected_map))
    if set(families) != set(expected_family_names):
        reasons.append(
            f"expected families {list(expected_family_names)}, found {list(families)}"
        )
    cells_by_family: dict[str, tuple[str, ...]] = {}
    for family in families:
        cells = tuple(sorted({row.cell for row in normalised if row.family == family}))
        cells_by_family[family] = cells
        required_cells = expected_map.get(family, ())
        if cells != tuple(sorted(required_cells)):
            reasons.append(
                f"family {family!r} expected cells {list(required_cells)}, found {list(cells)}"
            )
        for cell in cells:
            count = sum(row.family == family and row.cell == cell for row in normalised)
            if count != n_latents_per_cell:
                reasons.append(
                    f"cell {family!r}/{cell!r} expected {n_latents_per_cell} latents, found {count}"
                )
    expected_rows = sum(len(cells) for cells in expected_map.values()) * n_latents_per_cell
    if len(normalised) != expected_rows:
        reasons.append(f"expected {expected_rows} rows, found {len(normalised)}")
    return CompositionReport(
        passed=not reasons,
        reasons=tuple(reasons),
        n_rows=len(normalised),
        families=families,
        cells_by_family=cells_by_family,
    )


def hoeffding_lower(mean: float, n: int, epsilon: float) -> float:
    """One-sided lower confidence bound for a [0, 1] mean."""

    if n <= 0:
        raise ValueError("n must be positive")
    _check_probability("epsilon", epsilon)
    return max(0.0, float(mean) - sqrt(log(1.0 / epsilon) / (2.0 * n)))


def hoeffding_upper(mean: float, n: int, epsilon: float) -> float:
    """One-sided upper confidence bound for a [0, 1] mean."""

    if n <= 0:
        raise ValueError("n must be positive")
    _check_probability("epsilon", epsilon)
    return min(1.0, float(mean) + sqrt(log(1.0 / epsilon) / (2.0 * n)))


def _competence(
    rows: list[CoverageObservation], alpha: float
) -> dict[str, dict[str, dict[str, Any]]]:
    epsilon = alpha / 12.0
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for family in sorted({row.family for row in rows}):
        family_rows = [row for row in rows if row.family == family]
        counts = Counter(row.answer for row in family_rows)
        constant_prediction, constant_count = min(
            counts.items(), key=lambda item: (-item[1], repr(item[0]))
        )
        constant_accuracy = constant_count / len(family_rows)
        result[family] = {}
        for view, attrs in (("A", ("a1", "a2")), ("T", ("t1", "t2"))):
            latent_scores = [
                (getattr(row, attrs[0]) + getattr(row, attrs[1])) / 2.0
                for row in family_rows
            ]
            accuracy = sum(latent_scores) / len(latent_scores)
            lower = hoeffding_lower(accuracy, len(latent_scores), epsilon)
            result[family][view] = {
                "n": len(latent_scores),
                "accuracy": accuracy,
                "accuracy_lower": lower,
                "constant_prediction": constant_prediction,
                "constant_accuracy": constant_accuracy,
                "margin_over_constant": accuracy - constant_accuracy,
                "epsilon": epsilon,
                "passed": lower > constant_accuracy,
            }
    return result


def _opportunity(rows: list[CoverageObservation], alpha: float, delta: float) -> dict[str, Any]:
    values: dict[str, list[float]] = {"V_AA": [], "V_AT": [], "V_TT": []}
    per_latent = []
    for row in rows:
        a_errors = (1 - row.a1, 1 - row.a2)
        t_errors = (1 - row.t1, 1 - row.t2)
        v_aa = 1 - a_errors[0] * a_errors[1]
        v_tt = 1 - t_errors[0] * t_errors[1]
        v_at = 1 - sum(a_error * t_error for a_error in a_errors for t_error in t_errors) / 4.0
        values["V_AA"].append(v_aa)
        values["V_AT"].append(v_at)
        values["V_TT"].append(v_tt)
        per_latent.append(
            {
                "family": row.family,
                "cell": row.cell,
                "latent": row.latent,
                "V_AA": v_aa,
                "V_AT": v_at,
                "V_TT": v_tt,
            }
        )
    epsilon = alpha / 6.0
    means = {name: sum(items) / len(items) for name, items in values.items()}
    uppers = {name: hoeffding_upper(mean, len(rows), epsilon) for name, mean in means.items()}
    h_lower = max(0.0, 1.0 - max(uppers.values()))
    return {
        "n": len(rows),
        "per_latent": per_latent,
        "means": means,
        "upper_bounds": uppers,
        "epsilon": epsilon,
        "H": 1.0 - max(means.values()),
        "H_lower": h_lower,
        "required_H_lower": delta,
        "passed": h_lower > delta,
    }


def qualify_coverage(
    rows: Iterable[CoverageObservation | Mapping[str, Any]], *, delta: float, alpha: float
) -> QualificationReport:
    """Return a conservative PASS/FAIL adequacy report for paired A/T coverage.

    ``delta`` is the minimum global opportunity headroom.  Competence is
    qualified when its lower bound strictly exceeds the label-derived best constant
    accuracy; it has no additional delta margin.  This gate is not the
    study's primary inferential result or a statement about statistical power.
    """

    delta = _check_probability("delta", delta, allow_zero=True)
    alpha = _check_probability("alpha", alpha)
    try:
        normalised = _normalise_rows(rows)
        composition = validate_composition(normalised)
    except ValueError as exc:
        composition = CompositionReport(False, (str(exc),), 0, (), {})
        return QualificationReport(
            False,
            "FAIL",
            (f"malformed data: {exc}",),
            delta,
            alpha,
            composition,
            {},
            {"passed": False, "H_lower": None, "required_H_lower": delta},
        )
    if not composition.passed:
        return QualificationReport(
            False,
            "FAIL",
            tuple(f"composition: {reason}" for reason in composition.reasons),
            delta,
            alpha,
            composition,
            {},
            {"passed": False, "H_lower": None, "required_H_lower": delta},
        )

    competence = _competence(normalised, alpha)
    opportunity = _opportunity(normalised, alpha, delta)
    reasons = []
    for family, views in competence.items():
        for view, metric in views.items():
            if not metric["passed"]:
                if metric["margin_over_constant"] <= 0:
                    reasons.append(
                        f"{family}/{view}: response is no better than constant predictor"
                    )
                else:
                    reasons.append(
                        f"{family}/{view}: accuracy lower bound {metric['accuracy_lower']:.4f} "
                        f"is below constant accuracy {metric['constant_accuracy']:.4f}"
                    )
    if not opportunity["passed"]:
        reasons.append(
            f"insufficient opportunity: H lower bound {opportunity['H_lower']:.4f} "
            f"is not above delta {delta:.4f}"
        )
    return QualificationReport(
        not reasons,
        "PASS" if not reasons else "FAIL",
        tuple(reasons),
        delta,
        alpha,
        composition,
        competence,
        opportunity,
    )


evaluate_coverage = qualify_coverage

__all__ = [
    "CoverageObservation",
    "CompositionReport",
    "FAMILY_CELLS",
    "QualificationReport",
    "evaluate_coverage",
    "hoeffding_lower",
    "hoeffding_upper",
    "qualify_coverage",
    "validate_composition",
]
