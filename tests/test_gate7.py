from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate7 import (
    CONDITIONS,
    MEANINGFUL,
    RANDOM_NAMES,
    allocate_fresh_items,
    build_schedule,
    classify_gate7,
    gate7_random_bank,
    pseudo_replication_projection,
)


def _candidate(index: int) -> dict[str, str]:
    return {
        "id": f"sample_{index}",
        "code": "def f(x):\n    return x",
        "input": str(index),
        "output": str(index),
    }


def test_fresh_allocation_uses_120_then_100_contingency() -> None:
    candidates = [_candidate(index) for index in range(200)]
    selected, summary = allocate_fresh_items(candidates, [f"sample_{i}" for i in range(50)])
    assert len(selected) == summary["actual_n"] == 120
    assert not {row["item_id"] for row in selected} & {f"sample_{i}" for i in range(50)}
    assert all(
        row["reference_canonical_type"] == row["metadata"]["reference_canonical_type"]
        for row in selected
    )

    selected_100, summary_100 = allocate_fresh_items(
        [_candidate(index) for index in range(110)], []
    )
    assert len(selected_100) == summary_100["actual_n"] == 100


def test_schedule_is_complete_interleaved_and_seed_unique() -> None:
    schedule = build_schedule(["sample_1", "sample_2"])
    assert len(schedule) == 2 * len(CONDITIONS) * 2
    assert len({(r["item_id"], r["condition"], r["rollout_index"]) for r in schedule}) == len(
        schedule
    )
    assert len({r["seed"] for r in schedule}) == len(schedule)
    for item_id in ("sample_1", "sample_2"):
        for rollout in (0, 1):
            rows = [
                row
                for row in schedule
                if row["item_id"] == item_id and row["rollout_index"] == rollout
            ]
            assert {row["condition"] for row in rows} == set(CONDITIONS)
            assert sorted(row["condition_order"] for row in rows) == list(range(len(CONDITIONS)))


def test_random_bank_is_new_orthonormal_and_energy_matched() -> None:
    meaningful = np.arange(1, 17, dtype=np.float64)
    bank, metadata = gate7_random_bank(meaningful)
    assert set(bank) == set(RANDOM_NAMES)
    assert metadata["geometry"]["unit_norm_pass"]
    assert metadata["geometry"]["meaningful_orthogonality_pass"]
    assert metadata["geometry"]["random_pairwise_orthogonality_pass"]
    assert len({record["vector_sha256"] for record in metadata["records"].values()}) == 4
    delta_norms = [record["delta_norm"] for record in metadata["records"].values()]
    assert np.ptp(delta_norms) < 1e-12


def test_classification_strong_and_destructive_are_exhaustive() -> None:
    baseline = {"accuracy": 0.5, "commitment_validity": 0.98, "semantic_evaluability": 0.98}
    controller = {"accuracy": 0.7, "commitment_validity": 0.97, "semantic_evaluability": 0.97}
    estimands = {"G": 0.2, "C": 0.12, "D": 0.16, "rescue": 0.22, "damage": 0.02}
    random = {
        metric: {"mean": mean, "max": maximum}
        for metric, mean, maximum in (("G", 0.0, 0.02), ("C", 0.01, 0.03), ("D", 0.02, 0.05))
    }
    intervals = {
        name: {"q025": 0.01}
        for name in (
            "meaningful:accuracy_change",
            "meaningful:G",
            "meaningful:C",
            "meaningful:G_minus_random_mean",
            "meaningful:C_minus_random_mean",
        )
    }
    classification, gates = classify_gate7(
        baseline=baseline,
        controller=controller,
        controller_estimands=estimands,
        random_summary=random,
        bootstrap=intervals,
        loo_sign_stable={"accuracy_change": True, "G": True, "C": True},
        controller_style_replicated=False,
    )
    assert classification == "GATE7_STRONG_SINGLE_L27_REPLICATION"
    assert gates["strong_replication"]

    destructive, _ = classify_gate7(
        baseline=baseline,
        controller={**controller, "commitment_validity": 0.80},
        controller_estimands=estimands,
        random_summary=random,
        bootstrap=intervals,
        loo_sign_stable={"accuracy_change": True, "G": True, "C": True},
        controller_style_replicated=True,
    )
    assert destructive == "GATE7_DESTRUCTIVE"


def test_precision_projection_is_model_free_and_deterministic() -> None:
    baseline = np.asarray([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.int8)
    conditions = {
        MEANINGFUL: np.asarray([[0, 0], [1, 0], [0, 0], [0, 0]], dtype=np.int8),
        **{
            name: np.asarray([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.int8)
            for name in (f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
        },
    }
    first = pseudo_replication_projection(baseline, conditions, target_n=8, resamples=50, seed=11)
    second = pseudo_replication_projection(baseline, conditions, target_n=8, resamples=50, seed=11)
    assert first == second
    assert first["target_item_count"] == 8
