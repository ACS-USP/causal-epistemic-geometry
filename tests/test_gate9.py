from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate9 import (
    CONDITIONS,
    ETA,
    RANDOM_NAMES,
    allocate_fresh_items,
    build_schedule,
    classify_gate9,
    gate9_random_bank,
)


def _rows(n: int = 220) -> list[dict[str, str]]:
    return [
        {
            "id": f"sample_{index}",
            "prompt": f"prompt {index}",
            "output": str(index),
        }
        for index in range(n)
    ]


def test_gate9_allocation_is_fresh_fixed_and_leaves_remainder() -> None:
    selected, summary = allocate_fresh_items(_rows(), [f"sample_{index}" for index in range(20)])
    assert len(selected) == 100
    assert summary["remaining_unallocated_n"] == 100
    assert not ({row["item_id"] for row in selected} & {f"sample_{i}" for i in range(20)})
    assert not ({row["item_id"] for row in selected} & set(summary["remaining_unallocated_ids"]))


def test_gate9_schedule_is_complete_independent_and_unique() -> None:
    schedule = build_schedule(["sample_1", "sample_2"])
    assert len(schedule) == 2 * len(CONDITIONS) * 2
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in schedule}
    assert len(keys) == len(schedule)
    assert len({row["seed"] for row in schedule}) == len(schedule)
    assert {row["seed_regime"] for row in schedule} == {"INDEPENDENT_PRIMARY"}


def test_gate9_random_bank_is_new_orthonormal_and_d75_matched() -> None:
    meaningful = np.zeros(4096, dtype=np.float64)
    meaningful[0] = 1.0
    bank, metadata = gate9_random_bank(meaningful)
    assert set(bank) == set(RANDOM_NAMES)
    assert metadata["geometry"]["unit_norm_pass"]
    assert metadata["geometry"]["meaningful_orthogonality_pass"]
    assert metadata["geometry"]["random_pairwise_orthogonality_pass"]
    assert len({metadata["records"][name]["seed"] for name in RANDOM_NAMES}) == 4
    assert ETA == 9.637427952852196


def _bootstrap(lower: float = 0.01) -> dict[str, dict[str, float]]:
    names = (
        "meaningful:accuracy_change",
        "meaningful:G",
        "meaningful:C",
        "meaningful:G_minus_random_mean",
        "meaningful:C_minus_random_mean",
    )
    return {name: {"q025": lower} for name in names}


def test_gate9_strong_classification_is_mechanical() -> None:
    classification, gates = classify_gate9(
        baseline={"commitment_validity": 0.99, "semantic_evaluability": 0.99, "accuracy": 0.40},
        controller={"commitment_validity": 0.97, "semantic_evaluability": 0.97, "accuracy": 0.50},
        controller_estimands={"G": 0.15, "C": 0.10, "D": 0.14, "rescue": 0.20, "damage": 0.10},
        random_summary={metric: {"mean": 0.02, "max": 0.04} for metric in ("G", "C", "D")},
        bootstrap=_bootstrap(),
        loo_sign_stable={"accuracy_change": True, "G": True, "C": True},
        controller_style_replicated=True,
        source_replicated=True,
    )
    assert classification == "GATE9_STRONG_SAFE_SELECTED_DOSE_REPLICATION"
    assert gates["strong_safe_error_control_replication"]


def test_gate9_source_failure_has_exhaustive_priority() -> None:
    classification, _ = classify_gate9(
        baseline={"commitment_validity": 1.0, "semantic_evaluability": 1.0, "accuracy": 0.4},
        controller={"commitment_validity": 1.0, "semantic_evaluability": 1.0, "accuracy": 0.6},
        controller_estimands={"G": 0.2, "C": 0.15, "D": 0.2, "rescue": 0.3, "damage": 0.1},
        random_summary={metric: {"mean": 0.0, "max": 0.01} for metric in ("G", "C", "D")},
        bootstrap=_bootstrap(),
        loo_sign_stable={"accuracy_change": True, "G": True, "C": True},
        controller_style_replicated=False,
        source_replicated=False,
    )
    assert classification == "GATE9_SOURCE_POLICY_NOT_REPLICATED"


def test_gate9_invalid_or_weak_effect_does_not_become_positive() -> None:
    classification, gates = classify_gate9(
        baseline={"commitment_validity": 0.99, "semantic_evaluability": 0.99, "accuracy": 0.5},
        controller={"commitment_validity": 0.90, "semantic_evaluability": 0.90, "accuracy": 0.55},
        controller_estimands={"G": 0.01, "C": 0.0, "D": 0.01, "rescue": 0.1, "damage": 0.05},
        random_summary={metric: {"mean": 0.02, "max": 0.03} for metric in ("G", "C", "D")},
        bootstrap=_bootstrap(-0.01),
        loo_sign_stable={"accuracy_change": True, "G": True, "C": False},
        controller_style_replicated=False,
        source_replicated=True,
    )
    assert classification == "GATE9_SELECTED_DOSE_DESTRUCTIVE"
    assert not gates["commitment_validity_guard"]
