from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments import gate13_1


def test_development_split_is_deterministic_and_disjoint() -> None:
    items = [{"item_id": f"sample_{index}"} for index in range(40)]
    sweep, qualification = gate13_1.split_development_items(items)
    assert len(sweep) == 12
    assert len(qualification) == 28
    assert {row["item_id"] for row in sweep}.isdisjoint(
        {row["item_id"] for row in qualification}
    )
    assert gate13_1.split_development_items(items) == (sweep, qualification)


def test_all_layer_sweep_schedule_is_matched_and_complete() -> None:
    schedule = gate13_1.build_sweep_schedule(["a", "b"])
    assert len(schedule) == 70
    for item_id in ("a", "b"):
        selected = [row for row in schedule if row["item_id"] == item_id]
        assert len({row["seed"] for row in selected}) == 1
        assert {row["condition"] for row in selected} == {
            "BASELINE",
            *(f"MEANINGFUL_L{layer}_D50" for layer in range(34)),
        }


def test_sweep_selection_uses_q_within_quartiles() -> None:
    metrics = {
        layer: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "Q": 0.10 + layer / 1000,
        }
        for layer in range(34)
    }
    candidates, eligible = gate13_1.select_sweep_candidates(metrics)
    assert candidates == [8, 17, 25, 33]
    assert all(eligible.values())


def test_stage_b_nulls_are_orthogonal_and_deterministic() -> None:
    rng = np.random.default_rng(13)
    meaningful = rng.normal(size=64)
    meaningful /= np.linalg.norm(meaningful)
    differences = rng.normal(size=(32, 64))
    first = gate13_1.stage_b_nulls(meaningful, differences, 7)
    second = gate13_1.stage_b_nulls(meaningful, differences, 7)
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert np.allclose(value, second[name])
        assert np.isclose(np.linalg.norm(value), 1.0)
        assert abs(np.dot(value, meaningful)) <= 1e-10
    assert abs(np.dot(first["ISOTROPIC_NULL"], first["SHUFFLED_NULL"])) <= 1e-10


def test_layer_dose_schedule_and_selection() -> None:
    schedule = gate13_1.build_layer_dose_schedule(["a"], [8, 22])
    assert len(schedule) == 25
    assert len({row["seed"] for row in schedule}) == 1
    metrics = {}
    for layer in (8, 22):
        for dose in gate13_1.DOSE_FRACTIONS:
            metrics[(layer, dose)] = {
                "commitment_validity": 1.0,
                "semantic_evaluability": 1.0,
                "accuracy": 0.5,
                "baseline_accuracy": 0.5,
                "Q": 0.10,
                "null_mean_Q": 0.05,
                "null_max_Q": 0.08,
            }
    metrics[(8, "D50")].update(Q=0.30, null_mean_Q=0.10, null_max_Q=0.15)
    metrics[(8, "D75")].update(Q=0.50, null_mean_Q=0.10, null_max_Q=0.15)
    metrics[(22, "D25")].update(Q=0.35, null_mean_Q=0.10, null_max_Q=0.15)
    selected, proof = gate13_1.select_layer_dose(metrics, {8: 1.0, 22: 2.0})
    assert selected == (22, "D25")
    assert proof["L8_D50"]["eligible"]
    assert proof["L22_D25"]["eligible"]


def test_final_null_bank_and_independent_schedule() -> None:
    rng = np.random.default_rng(131)
    meaningful = rng.normal(size=96)
    meaningful /= np.linalg.norm(meaningful)
    differences = rng.normal(size=(64, 96))
    bank, metadata = gate13_1.final_null_bank(meaningful, differences, 18)
    assert list(bank) == ["R0", "R1", "R2", "R3"]
    matrix = np.stack([meaningful, *bank.values()])
    assert np.max(np.abs(matrix @ matrix.T - np.eye(5))) <= 1e-10
    assert len(metadata["records"]) == 4
    schedule = gate13_1.build_final_schedule(["a", "b"])
    assert len(schedule) == 28
    assert len({row["seed"] for row in schedule}) == 28
