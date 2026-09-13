from __future__ import annotations

from dataclasses import asdict

import pytest

from epistemic_geometry.analysis.budget_interaction import (
    CATEGORIES,
    Observation,
    aggregate_outcomes,
    outcome_category,
    stratified_latent_bootstrap,
    validate_factorial_schedule,
)


def _rows() -> list[Observation]:
    rows: list[Observation] = []
    # Unequal latent/cell counts make both levels of equal weighting observable.
    strata = {"A": {"a1": ("a1", "a2"), "a2": ("a3",)}, "B": {"b1": ("b1",)}}
    for family, cells in strata.items():
        for cell, latents in cells.items():
            for latent in latents:
                for cap in (2048, 4096):
                    for condition in ("BASELINE", "D75"):
                        for rollout in (0, 1):
                            correct = condition == "D75" and cap == 4096 and cell == "a1"
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
                                    "eos",
                                )
                            )
    return rows


def test_category_partition_is_exact_and_mutually_exclusive() -> None:
    assert outcome_category(True, "OK") == "correct"
    assert outcome_category(False, "OK") == "valid_wrong"
    assert outcome_category(False, "MISSING_FINAL") == "incomplete"
    assert outcome_category(False, "THINKING_UNCLOSED") == "incomplete"
    assert outcome_category(False, "TRUNCATED_NO_FINAL") == "incomplete"
    assert outcome_category(False, "INVALID_FINAL") == "invalid"
    with pytest.raises(ValueError, match="unknown parse_status"):
        outcome_category(False, "UNKNOWN")
    with pytest.raises(ValueError, match="correct outcome"):
        outcome_category(True, "INVALID_FINAL")
    assert set(CATEGORIES) == {"correct", "valid_wrong", "incomplete", "invalid"}


def test_factorial_validation_rejects_missing_duplicate_and_unknown_rows() -> None:
    rows = _rows()
    validate_factorial_schedule(rows)

    with pytest.raises(ValueError, match="incomplete factorial"):
        validate_factorial_schedule(rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_factorial_schedule(rows + [rows[0]])
    bad_status = {**asdict(rows[0]), "parse_status": "MAYBE"}
    with pytest.raises(ValueError, match="unknown parse_status"):
        validate_factorial_schedule([bad_status, *rows[1:]])

    with pytest.raises(ValueError, match="latent set mismatch"):
        validate_factorial_schedule(
            rows,
            expected_latents={
                ("A", "a1"): ("a1", "a2", "missing"),
                ("A", "a2"): ("a3",),
                ("B", "b1"): ("b1",),
            },
        )


def test_aggregation_weights_latents_cells_and_families_equally() -> None:
    result = aggregate_outcomes(_rows())
    # A/a1 is 1.0 correct, A/a2 and B/b1 are 0.0. A gets equal cell weight:
    # (1 + 0) / 2 = .5; the final family average is (.5 + 0) / 2 = .25.
    assert result["rates"][4096]["D75"]["correct"] == pytest.approx(0.25)
    # The raw rollout mean would be 4 / 48 = 1/12.
    assert result["rates"][4096]["D75"]["correct"] != pytest.approx(1 / 12)
    for cap in (2048, 4096):
        for condition in ("BASELINE", "D75"):
            assert sum(result["rates"][cap][condition].values()) == pytest.approx(1.0)


def test_delta_and_interaction_are_returned_for_both_caps() -> None:
    result = aggregate_outcomes(_rows())
    assert result["delta_correct"][2048] == pytest.approx(0.0)
    assert result["delta_correct"][4096] == pytest.approx(0.25)
    assert result["interaction"] == pytest.approx(0.25)


def test_stratified_latent_bootstrap_is_reproducible_and_descriptive() -> None:
    first = stratified_latent_bootstrap(_rows(), n_resamples=100, seed=17)
    second = stratified_latent_bootstrap(_rows(), n_resamples=100, seed=17)
    assert first == second
    assert first["method"] == "stratified_latent_bootstrap_descriptive"
    assert len(first["interaction_samples"]) == 100
    assert set(first["delta_correct_samples"]) == {2048, 4096}
    assert all(len(interval) == 2 for interval in first["delta_correct_percentile_95"].values())
