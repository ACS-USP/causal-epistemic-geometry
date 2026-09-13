"""Outcome-only estimands for the prospective D75 by token-cap study.

This module deliberately contains no prompt construction, model execution, or
file loading.  It consumes completed observations from the frozen canonical
prompt schedule and applies the pre-specified factorial validation and
equal-weight aggregation rules.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any, Final, TypeAlias

import numpy as np

TOKEN_CAPS: Final[tuple[int, int]] = (2048, 4096)
CONDITIONS: Final[tuple[str, str]] = ("BASELINE", "D75")
ROLLOUTS: Final[tuple[int, int]] = (0, 1)
CANONICAL_SURFACE: Final[str] = "canonical"
PARSE_STATUSES: Final[frozenset[str]] = frozenset(
    {"OK", "MISSING_FINAL", "THINKING_UNCLOSED", "TRUNCATED_NO_FINAL", "INVALID_FINAL"}
)
INCOMPLETE_STATUSES: Final[frozenset[str]] = frozenset(
    {"MISSING_FINAL", "THINKING_UNCLOSED", "TRUNCATED_NO_FINAL"}
)
CATEGORIES: Final[tuple[str, str, str, str]] = (
    "correct",
    "valid_wrong",
    "incomplete",
    "invalid",
)

Key: TypeAlias = tuple[str, str, str, int, str, int]
Stratum: TypeAlias = tuple[str, str]
Rates: TypeAlias = dict[str, float]
LatentKey: TypeAlias = tuple[str, str, str]

PRELOCK_DELTA: Final[float] = 0.10
PRELOCK_THRESHOLD: Final[float] = 60.0
PRELOCK_STATUS_TRIGGERED: Final[str] = "TRIGGERED"
PRELOCK_STATUS_NOT_TRIGGERED: Final[str] = "NOT_TRIGGERED"


@dataclass(frozen=True, slots=True)
class Observation:
    """One scored rollout from the canonical prompt schedule."""

    family: str
    cell: str
    latent: str
    cap: int
    condition: str
    rollout: int
    correct: bool
    parse_status: str
    stop_reason: str | None = None

    @property
    def category(self) -> str:
        """Return the mutually exclusive outcome category for this rollout."""

        return outcome_category(self.correct, self.parse_status)

    def key(self) -> Key:
        return (self.family, self.cell, self.latent, self.cap, self.condition, self.rollout)


def outcome_category(correct: bool, parse_status: str) -> str:
    """Classify one result without conditioning on valid completions."""

    if type(correct) is not bool:
        raise ValueError("correct must be a boolean")
    if not isinstance(parse_status, str) or parse_status not in PARSE_STATUSES:
        raise ValueError(f"unknown parse_status: {parse_status!r}")
    if correct:
        if parse_status != "OK":
            raise ValueError("a correct outcome must have parse_status='OK'")
        return "correct"
    if parse_status == "OK":
        return "valid_wrong"
    if parse_status in INCOMPLETE_STATUSES:
        return "incomplete"
    return "invalid"


def _as_observation(row: Observation | Mapping[str, Any]) -> Observation:
    if isinstance(row, Observation):
        return row
    if not isinstance(row, Mapping):
        raise TypeError("observations must be Observation objects or mappings")
    if "surface" in row and row["surface"] != CANONICAL_SURFACE:
        raise ValueError("observations must use the canonical prompt surface")
    required = (
        "family",
        "cell",
        "latent",
        "cap",
        "condition",
        "rollout",
        "correct",
        "parse_status",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"observation is missing fields: {', '.join(missing)}")
    observation = Observation(
        family=row["family"],
        cell=row["cell"],
        latent=row["latent"],
        cap=row["cap"],
        condition=row["condition"],
        rollout=row["rollout"],
        correct=row["correct"],
        parse_status=row["parse_status"],
        stop_reason=row.get("stop_reason"),
    )
    if "category" in row and row["category"] != observation.category:
        raise ValueError("observation category does not match correct/parse_status")
    return observation


def _validate_scalar_fields(observation: Observation) -> None:
    for name in ("family", "cell", "latent"):
        value = getattr(observation, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if type(observation.cap) is not int or observation.cap not in TOKEN_CAPS:
        raise ValueError(f"cap must be one of {TOKEN_CAPS}")
    if not isinstance(observation.condition, str) or observation.condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    if type(observation.rollout) is not int or observation.rollout not in ROLLOUTS:
        raise ValueError(f"rollout must be one of {ROLLOUTS}")
    # This also checks boolean type and the exact allowed status partition.
    outcome_category(observation.correct, observation.parse_status)


def validate_factorial_schedule(
    observations: Iterable[Observation | Mapping[str, Any]],
    *,
    expected_latents: Mapping[Stratum, Iterable[str]] | None = None,
    canonical_cells: Mapping[str, Iterable[str]] | None = None,
) -> tuple[Observation, ...]:
    """Validate the complete 2-cap x 2-condition x 2-rollout schedule.

    The observed latent union is used to detect missing logical rows.  A
    planned ``expected_latents`` mapping is available when the schedule must
    also detect a latent omitted entirely.  ``canonical_cells`` can pin the
    allowed family/cell prompt strata; the endpoint never creates prompts.
    """

    rows = tuple(_as_observation(row) for row in observations)
    if not rows:
        raise ValueError("at least one observation is required")
    for row in rows:
        _validate_scalar_fields(row)

    allowed_cells: dict[str, frozenset[str]] | None = None
    if canonical_cells is not None:
        allowed_cells = {
            family: frozenset(cells) for family, cells in canonical_cells.items()
        }
        if not allowed_cells or any(
            not family or not cells for family, cells in allowed_cells.items()
        ):
            raise ValueError("canonical_cells must contain non-empty family/cell sets")
        for family, cell in {(row.family, row.cell) for row in rows}:
            if family not in allowed_cells or cell not in allowed_cells[family]:
                raise ValueError(f"unknown canonical family/cell: {(family, cell)!r}")

    keys = [row.key() for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate observation key")

    observed_strata = {(row.family, row.cell) for row in rows}
    if allowed_cells is not None:
        required_strata = {
            (family, cell) for family, cells in allowed_cells.items() for cell in cells
        }
        if observed_strata != required_strata:
            missing = sorted(required_strata - observed_strata)
            extra = sorted(observed_strata - required_strata)
            raise ValueError(f"canonical strata mismatch; missing={missing}, extra={extra}")

    planned: dict[Stratum, frozenset[str]] = {}
    if expected_latents is not None:
        planned = {stratum: frozenset(latents) for stratum, latents in expected_latents.items()}
        if any(
            not family or not cell or not latents for (family, cell), latents in planned.items()
        ):
            raise ValueError("expected_latents must contain non-empty strata and latent sets")
        if set(planned) != observed_strata:
            raise ValueError("expected_latents strata must match observed canonical strata")

    latent_union: dict[Stratum, set[str]] = defaultdict(set)
    for row in rows:
        latent_union[(row.family, row.cell)].add(row.latent)
    for stratum, latent_ids in latent_union.items():
        expected = planned.get(stratum, frozenset(latent_ids))
        if latent_ids != expected:
            raise ValueError(f"latent set mismatch in {stratum!r}")
        expected_keys = {
            (latent, cap, condition, rollout)
            for latent in expected
            for cap in TOKEN_CAPS
            for condition in CONDITIONS
            for rollout in ROLLOUTS
        }
        actual_keys = {
            (row.latent, row.cap, row.condition, row.rollout)
            for row in rows
            if (row.family, row.cell) == stratum
        }
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"incomplete factorial schedule in {stratum!r}; "
                f"missing={missing}, extra={extra}"
            )
    return rows


def _latent_rates(rows: Sequence[Observation]) -> dict[tuple[str, str, str, int, str], Rates]:
    by_key: dict[tuple[str, str, str, int, str], list[Observation]] = defaultdict(list)
    for row in rows:
        by_key[(row.family, row.cell, row.latent, row.cap, row.condition)].append(row)
    result: dict[tuple[str, str, str, int, str], Rates] = {}
    for key, group in by_key.items():
        if len(group) != 2 or {row.rollout for row in group} != set(ROLLOUTS):
            raise ValueError(f"expected exactly two rollouts for {key!r}")
        counts = {
            category: sum(row.category == category for row in group) for category in CATEGORIES
        }
        result[key] = {category: count / 2.0 for category, count in counts.items()}
    return result


def _mean_rates(groups: Iterable[Rates]) -> Rates:
    values = tuple(groups)
    if not values:
        raise ValueError("cannot average an empty group")
    return {
        category: float(np.mean([value[category] for value in values]))
        for category in CATEGORIES
    }


def _empty_grid() -> dict[int, dict[str, Rates]]:
    return {
        cap: {condition: {category: 0.0 for category in CATEGORIES} for condition in CONDITIONS}
        for cap in TOKEN_CAPS
    }


def _hierarchical_from_latents(
    latent_values: Mapping[tuple[str, str, str, int, str], Rates],
    *,
    sampled_latents: Mapping[Stratum, Sequence[str]] | None = None,
) -> tuple[
    dict[Stratum, dict[int, dict[str, Rates]]],
    dict[str, dict[int, dict[str, Rates]]],
    dict[int, dict[str, Rates]],
]:
    strata = sorted({(family, cell) for family, cell, *_ in latent_values})
    cell_rates: dict[Stratum, dict[int, dict[str, Rates]]] = {}
    for family, cell in strata:
        latent_ids = (
            tuple(sampled_latents[(family, cell)])
            if sampled_latents is not None
            else tuple(
                sorted({
                    latent
                    for f, c, latent, *_ in latent_values
                    if (f, c) == (family, cell)
                })
            )
        )
        if not latent_ids:
            raise ValueError(f"empty latent stratum: {(family, cell)!r}")
        grid = _empty_grid()
        for cap in TOKEN_CAPS:
            for condition in CONDITIONS:
                grid[cap][condition] = _mean_rates(
                    latent_values[(family, cell, latent, cap, condition)] for latent in latent_ids
                )
        cell_rates[(family, cell)] = grid

    family_rates: dict[str, dict[int, dict[str, Rates]]] = {}
    for family in sorted({family for family, _ in strata}):
        cells = [cell_rates[(family, cell)] for fam, cell in strata if fam == family]
        grid = _empty_grid()
        for cap in TOKEN_CAPS:
            for condition in CONDITIONS:
                grid[cap][condition] = _mean_rates(
                    cell[cap][condition] for cell in cells
                )
        family_rates[family] = grid

    rates = _empty_grid()
    for cap in TOKEN_CAPS:
        for condition in CONDITIONS:
            rates[cap][condition] = _mean_rates(
                family_grid[cap][condition] for family_grid in family_rates.values()
            )
    return cell_rates, family_rates, rates


def aggregate_outcomes(
    observations: Iterable[Observation | Mapping[str, Any]],
    *,
    expected_latents: Mapping[Stratum, Iterable[str]] | None = None,
    canonical_cells: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Return latent, cell, family, and equal-family estimands for the study."""

    rows = validate_factorial_schedule(
        observations, expected_latents=expected_latents, canonical_cells=canonical_cells
    )
    latent = _latent_rates(rows)
    cell_rates, family_rates, rates = _hierarchical_from_latents(latent)
    delta_correct = {
        cap: rates[cap]["D75"]["correct"] - rates[cap]["BASELINE"]["correct"]
        for cap in TOKEN_CAPS
    }
    interaction = delta_correct[4096] - delta_correct[2048]
    return {
        "rates": rates,
        "category_rates": rates,
        "latent_rates": latent,
        "cell_rates": cell_rates,
        "family_rates": family_rates,
        "delta_correct": delta_correct,
        "interaction": float(interaction),
    }


def stratified_latent_bootstrap(
    observations: Iterable[Observation | Mapping[str, Any]],
    *,
    n_resamples: int = 2_000,
    seed: int = 0,
    expected_latents: Mapping[Stratum, Iterable[str]] | None = None,
    canonical_cells: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    """Resample latents within each family/cell for descriptive precision planning.

    The return value contains percentile summaries for the deltas and
    interaction.  It is intentionally labelled descriptive and does not make
    an inferential decision or attach a hypothesis-test interpretation.
    """

    if type(n_resamples) is not int or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    rows = validate_factorial_schedule(
        observations, expected_latents=expected_latents, canonical_cells=canonical_cells
    )
    latent = _latent_rates(rows)
    strata = sorted({(family, cell) for family, cell, *_ in latent})
    latent_ids = {
        stratum: tuple(
            sorted({
                item_latent
                for family, cell, item_latent, *_ in latent
                if (family, cell) == stratum
            })
        )
        for stratum in strata
    }
    rng = np.random.default_rng(seed)
    interactions: list[float] = []
    delta_samples = {cap: [] for cap in TOKEN_CAPS}
    for _ in range(n_resamples):
        sampled = {
            stratum: tuple(rng.choice(ids, size=len(ids), replace=True).tolist())
            for stratum, ids in latent_ids.items()
        }
        _, _, rates = _hierarchical_from_latents(latent, sampled_latents=sampled)
        delta = {
            cap: rates[cap]["D75"]["correct"] - rates[cap]["BASELINE"]["correct"]
            for cap in TOKEN_CAPS
        }
        delta_samples[2048].append(float(delta[2048]))
        delta_samples[4096].append(float(delta[4096]))
        interactions.append(float(delta[4096] - delta[2048]))

    def interval(values: Sequence[float]) -> tuple[float, float]:
        return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))

    return {
        "method": "stratified_latent_bootstrap_descriptive",
        "n_resamples": n_resamples,
        "seed": seed,
        "delta_correct_samples": {cap: tuple(values) for cap, values in delta_samples.items()},
        "interaction_samples": tuple(interactions),
        "delta_correct_percentile_95": {
            cap: interval(values) for cap, values in delta_samples.items()
        },
        "interaction_percentile_95": interval(interactions),
    }


def _hierarchical_weights(
    latent_rates: Mapping[tuple[str, str, str, int, str], Rates],
) -> dict[LatentKey, float]:
    """Return equal family, cell, and latent weights for observed latents."""

    strata = sorted({(family, cell) for family, cell, *_ in latent_rates})
    if not strata:
        raise ValueError("cannot weight an empty latent set")
    families = tuple(sorted({family for family, _ in strata}))
    cells_by_family = {
        family: tuple(sorted(cell for fam, cell in strata if fam == family))
        for family in families
    }
    latents_by_stratum = {
        stratum: tuple(
            sorted(
                {
                    latent
                    for family, cell, latent, *_ in latent_rates
                    if (family, cell) == stratum
                }
            )
        )
        for stratum in strata
    }
    weights: dict[LatentKey, float] = {}
    for family in families:
        family_weight = 1.0 / len(families)
        cell_weight = 1.0 / len(cells_by_family[family])
        for cell in cells_by_family[family]:
            latents = latents_by_stratum[(family, cell)]
            latent_weight = 1.0 / len(latents)
            for latent in latents:
                weights[(family, cell, latent)] = family_weight * cell_weight * latent_weight
    if not np.isclose(sum(weights.values()), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("hierarchical weights do not sum to one")
    return weights


def _fixed_e_value(
    values: Mapping[LatentKey, float],
    *,
    weights: Mapping[LatentKey, float],
    delta: float,
    direction: int,
) -> tuple[float, dict[LatentKey, float]]:
    """Evaluate one of the fixed pre-lock product e-value bounds.

    With ``a_i = N q_i`` and ``c = 1 / (2 max_i a_i)``, the bound is
    ``prod_i(1 + direction*c*a_i*(value_i-delta))``.  The construction is
    deterministic and finite-sample descriptive; it is not a power or
    posterior calculation.
    """

    if direction not in (-1, 1):
        raise ValueError("direction must be either +1 (Eplus) or -1 (Eminus)")
    if not isinstance(delta, (int, float)) or isinstance(delta, bool) or not np.isfinite(delta):
        raise ValueError("delta must be a finite real number")
    if not values or set(values) != set(weights):
        raise ValueError("values and weights must contain the same non-empty latent keys")
    n = len(values)
    a_values = {key: n * float(weights[key]) for key in sorted(weights)}
    max_a = max(a_values.values())
    if not np.isfinite(max_a) or max_a <= 0:
        raise ValueError("latent weights must be finite and positive")
    c = 1.0 / (2.0 * max_a)
    factors: dict[LatentKey, float] = {}
    for key in sorted(values):
        value = values[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError("contrast values must be finite real numbers")
        factor = 1.0 + direction * c * a_values[key] * (float(value) - float(delta))
        if factor < 0.0 or not np.isfinite(factor):
            raise ValueError("fixed e-value factor is outside its supported bounds")
        factors[key] = factor
    return float(prod(factors.values())), factors


def _weighted_hierarchy(
    values: Mapping[LatentKey, float],
) -> dict[str, Any]:
    """Aggregate a latent vector using equal latent-to-cell-to-family weights."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (family, cell, _latent), value in values.items():
        cells[(family, cell)].append(float(value))
    cell_values = {
        key: float(np.mean(sorted(values))) for key, values in sorted(cells.items())
    }
    family_values: dict[str, list[float]] = defaultdict(list)
    for (family, _cell), value in cell_values.items():
        family_values[family].append(value)
    family_means = {
        family: float(np.mean(sorted(values))) for family, values in sorted(family_values.items())
    }
    overall = float(np.mean(tuple(family_means.values())))
    return {
        "latent": {key: float(values[key]) for key in sorted(values)},
        "cell": cell_values,
        "family": family_means,
        "overall": overall,
    }


def fixed_rule_prelock_decisions(
    observations: Iterable[Observation | Mapping[str, Any]],
    *,
    expected_latents: Mapping[Stratum, Iterable[str]] | None = None,
    canonical_cells: Mapping[str, Iterable[str]] | None = None,
    delta: float = PRELOCK_DELTA,
    threshold: float = PRELOCK_THRESHOLD,
) -> dict[str, Any]:
    """Apply the three fixed e-value pre-lock candidate decision rules.

    Each latent receives equal weight within its cell, each cell equal weight
    within its family, and each family equal weight overall.  For each cap,
    ``d_iL`` is the paired mean over the two rollouts (correct D75 minus
    correct BASELINE), and ``j_i = d_i4096 - d_i2048``.  The returned product
    bounds use ``a_i=N*q_i`` and ``c=1/(2*max_i(a_i))``.

    The three statuses are candidates under a fixed pre-lock rule:
    relevant gain at either cap, 10 percentage-point gain excluded at both
    caps, and nonzero interaction (the two-sided mixture).  They are not power
    calculations, confidence statements, or posterior probabilities.
    """

    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise ValueError("threshold must be a positive finite real number")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite real number")
    if (
        not isinstance(delta, (int, float))
        or isinstance(delta, bool)
        or not np.isfinite(delta)
        or float(delta) != PRELOCK_DELTA
    ):
        raise ValueError(f"delta is fixed at {PRELOCK_DELTA} for the pre-lock rules")

    rows = validate_factorial_schedule(
        observations, expected_latents=expected_latents, canonical_cells=canonical_cells
    )
    latent_rates = _latent_rates(rows)
    weights = _hierarchical_weights(latent_rates)
    n_latents = len(weights)
    if n_latents <= 0:
        raise ValueError("factorial observations must contain at least one latent")
    a_values = {key: n_latents * weight for key, weight in sorted(weights.items())}
    max_a = max(a_values.values())
    c = 1.0 / (2.0 * max_a)

    d_by_cap: dict[int, dict[LatentKey, float]] = {
        cap: {
            (family, cell, latent): float(
                latent_rates[(family, cell, latent, cap, "D75")]["correct"]
                - latent_rates[(family, cell, latent, cap, "BASELINE")]["correct"]
            )
            for family, cell, latent in sorted(weights)
        }
        for cap in TOKEN_CAPS
    }
    interaction = {
        key: float(d_by_cap[4096][key] - d_by_cap[2048][key]) for key in sorted(weights)
    }
    d_hierarchy = {cap: _weighted_hierarchy(d_by_cap[cap]) for cap in TOKEN_CAPS}
    interaction_hierarchy = _weighted_hierarchy(interaction)

    gain_eplus: dict[int, float] = {}
    gain_eminus: dict[int, float] = {}
    gain_plus_factors: dict[int, dict[LatentKey, float]] = {}
    gain_minus_factors: dict[int, dict[LatentKey, float]] = {}
    for cap in TOKEN_CAPS:
        gain_eplus[cap], gain_plus_factors[cap] = _fixed_e_value(
            d_by_cap[cap], weights=weights, delta=PRELOCK_DELTA, direction=1
        )
        gain_eminus[cap], gain_minus_factors[cap] = _fixed_e_value(
            d_by_cap[cap], weights=weights, delta=PRELOCK_DELTA, direction=-1
        )
    interaction_eplus, interaction_plus_factors = _fixed_e_value(
        {key: value / 2.0 for key, value in interaction.items()},
        weights=weights,
        delta=0.0,
        direction=1,
    )
    interaction_eminus, interaction_minus_factors = _fixed_e_value(
        {key: value / 2.0 for key, value in interaction.items()},
        weights=weights,
        delta=0.0,
        direction=-1,
    )

    average_gain_eplus = float(np.mean([gain_eplus[cap] for cap in TOKEN_CAPS]))
    minimum_gain_eminus = float(min(gain_eminus.values()))
    status_gain = average_gain_eplus >= float(threshold)
    status_excluded = minimum_gain_eminus >= float(threshold)
    interaction_mixture = float((interaction_eplus + interaction_eminus) / 2.0)
    status_interaction = interaction_mixture >= float(threshold)

    def decision(status: bool, statistic: float, criterion: str) -> dict[str, Any]:
        return {
            "status": PRELOCK_STATUS_TRIGGERED if status else PRELOCK_STATUS_NOT_TRIGGERED,
            "triggered": bool(status),
            "statistic": float(statistic),
            "threshold": float(threshold),
            "criterion": criterion,
        }

    decisions = {
        "relevant_gain_at_either_cap": decision(
            status_gain,
            average_gain_eplus,
            "average Eplus(delta, dL) >= threshold",
        ),
        "ten_pp_gain_excluded_at_both_caps": decision(
            status_excluded,
            minimum_gain_eminus,
            "min Eminus(delta, dL) >= threshold",
        ),
        "interaction_nonzero": decision(
            status_interaction,
            interaction_mixture,
            "(Eplus(0, j/2) + Eminus(0, j/2)) / 2 >= threshold",
        ),
    }
    statuses = {name: details["status"] for name, details in decisions.items()}

    # Include both descriptive names and compact aliases so downstream reports
    # can consume the result without reconstructing any statistic.
    return {
        "method": "fixed_rule_prelock_candidates",
        "power_calculation": False,
        "delta": PRELOCK_DELTA,
        "threshold": float(threshold),
        "n_latents": n_latents,
        "weights": {"q_i": weights, "a_i": a_values, "sum_q_i": float(sum(weights.values()))},
        "c": float(c),
        "d_iL": d_by_cap,
        "d_by_cap": d_by_cap,
        "j_i": interaction,
        "interaction_by_latent": interaction,
        "hierarchy": {"d_by_cap": d_hierarchy, "interaction": interaction_hierarchy},
        "weighted_delta": {cap: d_hierarchy[cap]["overall"] for cap in TOKEN_CAPS},
        "weighted_interaction": interaction_hierarchy["overall"],
        "e_values": {
            "gain": {
                "Eplus": gain_eplus,
                "Eminus": gain_eminus,
                "Eplus_factors": gain_plus_factors,
                "Eminus_factors": gain_minus_factors,
                "average_Eplus": average_gain_eplus,
                "minimum_Eminus": minimum_gain_eminus,
            },
            "interaction": {
                "Eplus": interaction_eplus,
                "Eminus": interaction_eminus,
                "Eplus_factors": interaction_plus_factors,
                "Eminus_factors": interaction_minus_factors,
                "mixture": interaction_mixture,
                "minimum_directional_E": min(interaction_eplus, interaction_eminus),
            },
        },
        "decisions": decisions,
        "statuses": statuses,
        "fixed_rule_candidate": True,
    }


# Short aliases make the endpoint convenient while preserving the explicit API.
analyze_outcomes = aggregate_outcomes
bootstrap_latents = stratified_latent_bootstrap
prelock_decisions = fixed_rule_prelock_decisions
evaluate_prelock_decisions = fixed_rule_prelock_decisions
apply_prelock_decision_rules = fixed_rule_prelock_decisions


__all__ = [
    "CANONICAL_SURFACE",
    "CATEGORIES",
    "CONDITIONS",
    "INCOMPLETE_STATUSES",
    "Observation",
    "PARSE_STATUSES",
    "PRELOCK_DELTA",
    "PRELOCK_STATUS_NOT_TRIGGERED",
    "PRELOCK_STATUS_TRIGGERED",
    "PRELOCK_THRESHOLD",
    "ROLLOUTS",
    "TOKEN_CAPS",
    "aggregate_outcomes",
    "apply_prelock_decision_rules",
    "analyze_outcomes",
    "bootstrap_latents",
    "evaluate_prelock_decisions",
    "fixed_rule_prelock_decisions",
    "outcome_category",
    "prelock_decisions",
    "stratified_latent_bootstrap",
    "validate_factorial_schedule",
]
