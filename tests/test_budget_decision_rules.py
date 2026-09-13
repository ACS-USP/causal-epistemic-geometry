from __future__ import annotations

from dataclasses import replace

import pytest

from epistemic_geometry.analysis.budget_interaction import (
    Observation,
    fixed_rule_prelock_decisions,
)


def _observations(
    n_latents: int = 96,
    *,
    d2048: float = 0.0,
    d4096: float = 0.0,
    family: str = "F",
    cell: str = "C",
) -> list[Observation]:
    """Build complete paired Bernoulli rows for deterministic rule tests."""

    if d2048 not in (-1.0, 0.0, 1.0) or d4096 not in (-1.0, 0.0, 1.0):
        raise ValueError("focused fixtures use integral paired contrasts")
    rows: list[Observation] = []
    for index in range(n_latents):
        latent = f"latent-{index:03d}"
        for cap, contrast in ((2048, d2048), (4096, d4096)):
            for condition in ("BASELINE", "D75"):
                correct = condition == "D75" and contrast == 1.0
                if contrast == -1.0:
                    correct = condition == "BASELINE"
                for rollout in (0, 1):
                    rows.append(
                        Observation(
                            family,
                            cell,
                            latent,
                            cap,
                            condition,
                            rollout,
                            correct,
                            "OK",
                        )
                    )
    return rows


def test_zero_contrasts_have_negative_evidence_above_fixed_threshold() -> None:
    result = fixed_rule_prelock_decisions(_observations())

    assert result["n_latents"] == 96
    assert result["c"] == pytest.approx(0.5)
    assert result["e_values"]["gain"]["minimum_Eminus"] > 60
    assert result["statuses"] == {
        "relevant_gain_at_either_cap": "NOT_TRIGGERED",
        "ten_pp_gain_excluded_at_both_caps": "TRIGGERED",
        "interaction_nonzero": "NOT_TRIGGERED",
    }


def test_positive_treatment_improvement_triggers_gain_candidate() -> None:
    result = fixed_rule_prelock_decisions(_observations(d2048=1.0, d4096=1.0))

    assert result["weighted_delta"] == {2048: 1.0, 4096: 1.0}
    assert result["e_values"]["gain"]["average_Eplus"] > 60
    assert result["decisions"]["relevant_gain_at_either_cap"]["triggered"] is True
    assert result["decisions"]["ten_pp_gain_excluded_at_both_caps"]["triggered"] is False


@pytest.mark.parametrize("mutator", [lambda rows: rows[:-1], lambda rows: rows + [rows[0]]])
def test_malformed_factorial_schedule_is_rejected(mutator) -> None:
    rows = _observations(n_latents=2)
    with pytest.raises(ValueError):
        fixed_rule_prelock_decisions(mutator(rows))


def test_invalid_factorial_value_is_rejected() -> None:
    rows = _observations(n_latents=2)
    bad = replace(rows[0], cap=1024)
    with pytest.raises(ValueError, match="cap"):
        fixed_rule_prelock_decisions([bad, *rows[1:]])


@pytest.mark.parametrize("d2048,d4096", [(0.0, 1.0), (0.0, -1.0)])
def test_interaction_mixture_is_direction_independent(d2048: float, d4096: float) -> None:
    result = fixed_rule_prelock_decisions(_observations(d2048=d2048, d4096=d4096))
    interaction = result["e_values"]["interaction"]

    assert interaction["mixture"] > 60
    assert result["decisions"]["interaction_nonzero"]["triggered"] is True
    # One side is intentionally tiny; the decision uses the bilateral mixture.
    assert min(interaction["Eplus"], interaction["Eminus"]) < 60


def test_output_is_explicitly_prelock_candidate_and_has_exactly_three_decisions() -> None:
    result = fixed_rule_prelock_decisions(_observations(n_latents=2))

    assert result["method"] == "fixed_rule_prelock_candidates"
    assert result["power_calculation"] is False
    assert set(result["decisions"]) == {
        "relevant_gain_at_either_cap",
        "ten_pp_gain_excluded_at_both_caps",
        "interaction_nonzero",
    }
