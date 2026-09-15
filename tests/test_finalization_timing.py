from __future__ import annotations

import pytest

from epistemic_geometry.analysis.finalization_timing import (
    QWEN3_THINK_CLOSE_TOKEN_ID,
    TimingObservation,
    directional_evalues,
    equal_family_cell_mean,
    normalized_paired_shortening,
    paired_shortening_from_restricted_times,
    think_close_time,
    validate_timing_schedule,
)


def _rows() -> list[dict[str, object]]:
    rows = []
    for latent in ("a", "b"):
        for condition, tokens in (("BASELINE", [1, QWEN3_THINK_CLOSE_TOKEN_ID]), ("D75", [1])):
            for rollout in (0, 1):
                rows.append(
                    {
                        "family": "f",
                        "cell": "c",
                        "latent": latent,
                        "condition": condition,
                        "rollout": rollout,
                        "cap": 10,
                        "generated_token_ids": tokens,
                    }
                )
    return rows


def test_censoring_and_paired_shortening_are_structural() -> None:
    row = TimingObservation("f", "c", "a", "BASELINE", 0, 10, (1,))
    assert think_close_time(row) == 10
    contrasts = normalized_paired_shortening(_rows(), cap=10)
    assert contrasts == {("f", "c", "a"): -0.8, ("f", "c", "b"): -0.8}
    assert equal_family_cell_mean(contrasts) == pytest.approx(-0.8)


def test_schedule_refuses_incomplete_or_over_budget_rows() -> None:
    rows = _rows()[:-1]
    with pytest.raises(ValueError, match="incomplete"):
        validate_timing_schedule(rows, cap=10)
    bad = _rows()
    bad[0]["generated_token_ids"] = list(range(11))
    with pytest.raises(ValueError, match="exceeds"):
        validate_timing_schedule(bad, cap=10)


def test_directional_evalues_respect_the_sign() -> None:
    earlier = directional_evalues([0.5] * 21, delta=0.1)
    later = directional_evalues([-0.5] * 21, delta=0.1)
    assert earlier["d75_earlier_by_delta"] > 40
    assert later["d75_later_by_delta"] > 40


def test_structural_seal_records_cannot_carry_semantic_fields() -> None:
    records = []
    for condition, time in (("BASELINE", 10), ("D75", 2)):
        for rollout in (0, 1):
            records.append(
                {
                    "family": "f",
                    "cell": "c",
                    "latent": "a",
                    "condition": condition,
                    "rollout": rollout,
                    "cap": 10,
                    "restricted_think_close_time": time,
                    "think_close_observed": time < 10,
                }
            )
    assert paired_shortening_from_restricted_times(records, cap=10) == {
        ("f", "c", "a"): 0.8
    }
    records[0]["correct"] = False
    with pytest.raises(ValueError, match="unexpected schema"):
        paired_shortening_from_restricted_times(records, cap=10)
