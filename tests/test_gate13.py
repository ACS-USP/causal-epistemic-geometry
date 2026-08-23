from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments import gate13


def _records(n: int) -> list[dict[str, str]]:
    return [
        {
            "item_id": f"sample_{index}",
            "prompt": f"prompt {index}",
            "reference_answer": str(index),
            "source_revision": gate13.DATASET_REVISION,
        }
        for index in range(n)
    ]


def test_reused_pool_allocation_is_disjoint_and_protects_untouched() -> None:
    pool = gate13.build_reused_development_pool(_records(400), ["sealed_1", "sealed_2"])
    allocations = gate13.allocate_pool(pool)
    expected = dict(gate13.ALLOCATION_COUNTS)
    assert {name: len(rows) for name, rows in allocations.items()} == expected
    ids = [row["item_id"] for rows in allocations.values() for row in rows]
    assert len(ids) == len(set(ids)) == 320
    with pytest.raises(RuntimeError, match="57 untouched"):
        gate13.build_reused_development_pool(_records(400), ["sample_3"])


def test_screen_schedule_uses_independent_unique_seeds() -> None:
    schedule = gate13.build_screen_schedule(["a", "b"], gate13.PRIMARY_MODEL)
    assert len(schedule) == 20
    assert len({row["seed"] for row in schedule}) == len(schedule)
    assert {row["condition"] for row in schedule} == set(gate13.SCREEN_CONDITIONS)


def test_substrate_classification_and_floor_rule() -> None:
    metrics = {
        condition: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.50,
            "mean_tokens": 20.0,
            "median_tokens": 18.0,
            "truncation": 0.0,
        }
        for condition in gate13.SCREEN_CONDITIONS
    }
    metrics["SOURCE_DIRECT"]["accuracy"] = 0.45
    metrics["SOURCE_CAREFUL"].update(
        {"accuracy": 0.60, "mean_tokens": 30.0, "semantic_change_vs_direct": 0.20}
    )
    classification, gates = gate13.classify_substrate(metrics)
    assert classification == "MINISTRAL3_8B_SUBSTRATE_PASS"
    assert gates["pass"]
    metrics["BASELINE"]["accuracy"] = 0.20
    classification, gates = gate13.classify_substrate(metrics)
    assert classification == "MINISTRAL3_8B_COMPETENCE_FLOOR"
    assert gates["source_accuracy"] and gates["source_behavior"]


def test_source_atlas_orientation_auroc_and_quartile_shortlist() -> None:
    rng = np.random.default_rng(13)
    train_direct = rng.normal(scale=0.01, size=(64, 8, 16))
    valid_direct = rng.normal(scale=0.01, size=(32, 8, 16))
    shifts = np.zeros((8, 16))
    shifts[:, 0] = np.linspace(1.0, 2.0, 8)
    directions, rows = gate13.source_atlas(
        train_direct + shifts,
        train_direct,
        valid_direct + shifts,
        valid_direct,
    )
    assert all(np.isclose(np.linalg.norm(value), 1.0) for value in directions.values())
    assert all(row["paired_mean_gap"] > 0 for row in rows)
    assert all(row["auroc"] == 1.0 for row in rows)
    shortlist = gate13.shortlist_layers(rows, 8)
    assert len(shortlist) == 4
    quartiles = ((0, 1), (2, 3), (4, 5), (6, 7))
    assert all(
        layer in quartile for layer, quartile in zip(shortlist, quartiles, strict=True)
    )


def test_first_stage_layer_selection_ignores_accuracy_as_ranker() -> None:
    metrics = {
        4: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.45,
            "baseline_accuracy": 0.50,
            "Q": 0.30,
            "null_mean_Q": 0.10,
            "null_max_Q": 0.15,
        },
        8: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.90,
            "baseline_accuracy": 0.50,
            "Q": 0.25,
            "null_mean_Q": 0.10,
            "null_max_Q": 0.15,
        },
    }
    selected, passed = gate13.select_first_stage_layer(metrics, {4: 0.5, 8: 2.0})
    assert all(passed.values())
    assert selected == 4


def test_first_stage_stops_when_no_candidate_passes() -> None:
    metrics = {
        4: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.5,
            "baseline_accuracy": 0.5,
            "Q": 0.10,
            "null_mean_Q": 0.08,
            "null_max_Q": 0.09,
        }
    }
    with pytest.raises(RuntimeError, match="GATE13_NO_CAUSAL_LAYER_FIRST_STAGE"):
        gate13.select_first_stage_layer(metrics, {4: 1.0})


def test_first_stage_schedule_is_matched() -> None:
    schedule = gate13.build_first_stage_schedule(["x", "y"], gate13.PRIMARY_MODEL, [2, 9])
    assert len(schedule) == 14
    for item_id in ("x", "y"):
        seeds = {row["seed"] for row in schedule if row["item_id"] == item_id}
        assert len(seeds) == 1


def test_final_null_bank_is_orthonormal_and_construction_matched() -> None:
    rng = np.random.default_rng(17)
    meaningful = rng.normal(size=64)
    meaningful /= np.linalg.norm(meaningful)
    differences = rng.normal(size=(64, 64))
    bank, metadata = gate13.final_null_bank(meaningful, differences, 12)
    assert set(bank) == {"R0", "R1", "R2", "R3"}
    assert metadata["records"]["R0"]["kind"] == "isotropic"
    assert metadata["records"]["R2"]["kind"] == "construction_matched_shuffled"
    assert max(abs(value) for value in metadata["cosines"].values()) <= 1e-6


def test_dose_schedule_matched_and_final_schedule_independent() -> None:
    dose = gate13.build_dose_schedule(["item"], gate13.PRIMARY_MODEL)
    assert len(dose) == 44
    for rollout in (0, 1):
        assert len({row["seed"] for row in dose if row["rollout_index"] == rollout}) == 1
    final = gate13.build_final_schedule(["item", "other"], gate13.PRIMARY_MODEL)
    assert len(final) == 28
    assert len({row["seed"] for row in final}) == len(final)


def test_lowest_eligible_dose_and_gate9_mapping() -> None:
    baseline = {
        "commitment_validity": 1.0,
        "semantic_evaluability": 1.0,
        "accuracy": 0.5,
        "mean_tokens": 10.0,
        "median_tokens": 8.0,
    }
    textual = {
        "commitment_validity": 1.0,
        "semantic_evaluability": 1.0,
        "accuracy": 0.6,
        "mean_tokens": 40.0,
        "median_tokens": 30.0,
    }
    doses = {
        name: {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.5,
            "Q": 0.30 if name != "D25" else 0.05,
            "rho_tokens": 0.50,
        }
        for name in gate13.DOSE_FRACTIONS
    }
    random_q = {name: {"mean": 0.10, "max": 0.15} for name in gate13.DOSE_FRACTIONS}
    selected, eligibility, classification = gate13.select_dose(
        baseline, textual, doses, random_q
    )
    assert selected == "D50"
    assert eligibility["D50"]["eligible"]
    assert classification == "GATE13_SAFE_DOSE_SELECTED"
    assert gate13.map_gate9_classification(
        "GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION"
    ) == "GATE13_STRONG_CROSS_MODEL_PROTOCOL_REPLICATION"
