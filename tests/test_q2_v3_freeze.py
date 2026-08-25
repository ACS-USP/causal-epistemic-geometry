from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q2_v3 import (
    SHELL_TARGETS,
    condition_ids,
    deterministic_allocate,
    exact_family_qap_permutations,
    meaningful_controller_ids,
    null_controller_ids,
    ordered_id_hash,
    stable_seed,
)


def test_q2_v3_bank_cardinality_and_order() -> None:
    meaningful = meaningful_controller_ids()
    nulls = null_controller_ids()
    assert len(meaningful) == 20
    assert len(set(meaningful)) == 20
    assert len(nulls) == 4
    assert len(set(nulls)) == 4
    assert condition_ids() == ("BASELINE", *meaningful, *nulls)
    assert SHELL_TARGETS == {"MEDIUM": 0.25, "STRONG": 0.50}


def test_q2_v3_exact_family_qap_space() -> None:
    values = exact_family_qap_permutations()
    assert len(values) == 120 * 32
    assert len(set(values)) == len(values)


def test_q2_v3_deterministic_allocation_ignores_outcomes() -> None:
    rows = [
        {
            "item_id": f"sample_{index}",
            "provenance_class": "C",
            "accuracy": float(index % 2),
        }
        for index in range(20)
    ]
    selected = deterministic_allocate(
        rows, provenance_class="C", namespace="fixture", count=8
    )
    reversed_outcomes = [dict(row, accuracy=1.0 - row["accuracy"]) for row in rows]
    selected_reversed = deterministic_allocate(
        reversed_outcomes, provenance_class="C", namespace="fixture", count=8
    )
    assert [row["item_id"] for row in selected] == [
        row["item_id"] for row in selected_reversed
    ]


def test_q2_v3_seed_and_order_hash_are_stable() -> None:
    assert stable_seed("x", "sample_1", "BASELINE", 0) == stable_seed(
        "x", "sample_1", "BASELINE", 0
    )
    assert stable_seed("x", "sample_1", "BASELINE", 0) != stable_seed(
        "x", "sample_1", "BASELINE", 1
    )
    assert ordered_id_hash(["a", "b"]) != ordered_id_hash(["b", "a"])


def test_q2_v3_null_projection_fixture() -> None:
    rng = np.random.default_rng(20260826)
    meaningful = rng.normal(size=(10, 64))
    q, _ = np.linalg.qr(meaningful.T)
    candidate = rng.normal(size=64)
    projected = candidate - q @ (q.T @ candidate)
    projected /= np.linalg.norm(projected)
    assert np.max(np.abs(q.T @ projected)) <= 1e-12
