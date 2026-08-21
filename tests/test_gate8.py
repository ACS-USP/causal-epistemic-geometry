from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments.gate8 import (
    CONDITIONS,
    allocate_calibration_items,
    build_schedule,
    classify_source,
    condition_spec,
    dose_eligibility,
    gate8_random_bank,
    select_dose,
)


def _candidates(n: int = 800) -> list[dict[str, str]]:
    return [
        {"id": f"sample_{i}", "code": "def f(x): return x", "input": str(i), "output": str(i)}
        for i in range(n)
    ]


def test_allocation_requires_150_and_preserves_100() -> None:
    selected, summary = allocate_calibration_items(
        _candidates(), [f"sample_{i}" for i in range(593)]
    )
    assert len(selected) == 50
    assert summary["eligible_before_allocation"] == 207
    assert summary["remaining_unallocated_n"] == 157
    assert summary["future_evaluation_ids_allocated"] is False
    with pytest.raises(RuntimeError, match="GATE8_BLOCKED_INSUFFICIENT_FRESH_ITEMS"):
        allocate_calibration_items(_candidates(742), [f"sample_{i}" for i in range(593)])


def test_schedule_has_matched_seeds_and_unique_keys() -> None:
    schedule = build_schedule(["sample_1", "sample_2"])
    assert len(schedule) == 2 * len(CONDITIONS) * 2
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in schedule}
    assert len(keys) == len(schedule)
    for item_id in ("sample_1", "sample_2"):
        for rollout in (0, 1):
            block = [
                row
                for row in schedule
                if row["item_id"] == item_id and row["rollout_index"] == rollout
            ]
            assert {row["condition"] for row in block} == set(CONDITIONS)
            assert len({row["seed"] for row in block}) == 1
    assert {
        row["seed"]
        for row in schedule
        if row["item_id"] == "sample_1" and row["rollout_index"] == 0
    } != {
        row["seed"]
        for row in schedule
        if row["item_id"] == "sample_1" and row["rollout_index"] == 1
    }


def test_condition_doses_are_exact() -> None:
    assert condition_spec("MEAN_D25")["eta"] == pytest.approx(3.2124759842840653)
    assert condition_spec("MEAN_D50")["eta"] == pytest.approx(6.4249519685681305)
    assert condition_spec("MEAN_D75")["eta"] == pytest.approx(9.637427952852196)
    assert condition_spec("MEAN_D100")["eta"] == pytest.approx(12.849903937136261)
    assert condition_spec("RANDOM_R3_D50")["vector"] == "GATE8_RANDOM_R3"


def test_random_bank_is_orthonormal_and_new() -> None:
    meaningful = np.zeros(4096, dtype=np.float64)
    meaningful[0] = 1.0
    bank, metadata = gate8_random_bank(meaningful)
    assert metadata["geometry"]["unit_norm_pass"]
    assert metadata["geometry"]["meaningful_orthogonality_pass"]
    assert metadata["geometry"]["random_pairwise_orthogonality_pass"]
    assert len({record["seed"] for record in metadata["records"].values()}) == 4
    assert all(abs(float(np.dot(meaningful, vector))) <= 1e-6 for vector in bank.values())


def test_source_and_lowest_eligible_selection() -> None:
    baseline = {
        "commitment_validity": 0.98,
        "semantic_evaluability": 0.98,
        "accuracy": 0.40,
        "mean_tokens": 10.0,
        "median_tokens": 8.0,
    }
    textual = {
        "commitment_validity": 0.98,
        "semantic_evaluability": 0.98,
        "mean_tokens": 100.0,
        "median_tokens": 40.0,
    }
    assert classify_source(baseline, textual) == "CAREFUL_SOURCE_REPLICATED"
    random_q = {"mean": 0.10, "max": 0.14}
    eligible = {
        "commitment_validity": 0.96,
        "semantic_evaluability": 0.96,
        "accuracy": 0.42,
        "Q": 0.22,
        "rho_tokens": 0.40,
    }
    gates = {
        dose: dose_eligibility(
            baseline=baseline, dose=eligible, random_q=random_q, source_replicated=True
        )
        for dose in ("D25", "D50", "D75", "D100")
    }
    gates["D25"]["eligible"] = False
    selected, classification = select_dose(gates)
    assert selected == "D50"
    assert classification == "GATE8_SAFE_LOWER_DOSE_SELECTED"


def test_tradeoff_and_inert_classifications() -> None:
    base = {
        dose: {
            "source_replicated": True,
            "commitment_validity": True,
            "semantic_evaluability": True,
            "competence_safety": True,
            "behavioral_first_stage": False,
            "eligible": False,
        }
        for dose in ("D25", "D50", "D75", "D100")
    }
    assert select_dose(base) == (None, "GATE8_LOWER_DOSES_NONSPECIFIC_OR_INERT")
    base["D100"].update(commitment_validity=False, behavioral_first_stage=True, eligible=False)
    assert select_dose(base) == (None, "GATE8_EFFECT_VALIDITY_TRADEOFF_CONFIRMED")
