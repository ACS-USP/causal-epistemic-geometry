from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate5 import (
    classify_gate5,
    classify_manipulation,
    classify_source,
    independent_estimands,
    random_controller_bank,
    source_disagreement,
)


def test_gate5_random_bank_is_exactly_reproducible_and_orthogonal() -> None:
    meaningful = np.eye(8, dtype=np.float64)[0]
    r0 = np.eye(8, dtype=np.float64)[1]
    first = random_controller_bank(meaningful, r0)
    second = random_controller_bank(meaningful, r0)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    vectors = [meaningful, *first.values()]
    for left_index, left in enumerate(vectors):
        assert np.isclose(np.linalg.norm(left), 1.0)
        for right in vectors[left_index + 1 :]:
            assert abs(float(np.dot(left, right))) < 1e-12


def test_gate5_independent_estimands_and_rescue_damage_identity() -> None:
    baseline = [[1, 0], [1, 1], [0, 0], [0, 1]]
    condition = [[0, 0], [1, 0], [0, 1], [1, 1]]
    result = independent_estimands(baseline, condition)
    assert np.isclose(
        result["rescue"] - result["damage"],
        result["accuracy_condition"] - result["accuracy_baseline"],
    )
    assert np.isfinite(result["D"])


def test_gate5_source_disagreement_and_frozen_classification_rules() -> None:
    outcomes = {
        "ORDINARY": {"a": ("x", "x"), "b": ("x", "x")},
        "CAREFUL": {"a": ("care", "care"), "b": ("care", "other")},
        "DIRECT": {"a": ("direct", "direct"), "b": ("direct", "direct")},
    }
    result = source_disagreement(outcomes)
    assert result["X_cross_disagreement"] > 0
    metrics = {
        "careful_validity": 1.0,
        "direct_validity": 1.0,
        "X_cross_disagreement": 0.25,
        "S_excess": 0.20,
        "careful_mean_tokens": 4.0,
        "direct_mean_tokens": 2.0,
        "careful_median_tokens": 4.0,
        "direct_median_tokens": 2.0,
    }
    assert classify_source(metrics) == "SOURCE_SEMANTIC_BEHAVIOR_PASS"


def test_gate5_manipulation_and_primary_classification_are_outcome_independent() -> None:
    manipulation = {
        name: {"validity": 1.0, "semantic_change_rate": 0.0}
        for name in (
            "ONE_SHOT_PLUS",
            "ONE_SHOT_MINUS",
            "SUSTAINED_PLUS",
            "SUSTAINED_MINUS",
            "SUSTAINED_RANDOM_R0",
            "SUSTAINED_RANDOM_R1",
            "SUSTAINED_RANDOM_R2",
            "SUSTAINED_RANDOM_R3",
        )
    }
    manipulation["SUSTAINED_PLUS"]["semantic_change_rate"] = 0.25
    assert classify_manipulation(manipulation)
    estimands = {
        "BASELINE": {"validity": 1.0, "accuracy": 0.50},
        "ONE_SHOT_PLUS": {"D": 0.01},
        "ONE_SHOT_MINUS": {"D": 0.01},
        "SUSTAINED_PLUS": {"D": 0.01, "G": 0.0, "C": 0.0, "validity": 1.0, "accuracy": 0.50},
        "SUSTAINED_MINUS": {"D": 0.01, "G": 0.0, "C": 0.0, "validity": 1.0, "accuracy": 0.50},
    }
    estimands.update({f"SUSTAINED_RANDOM_R{i}": {"D": 0.0, "G": 0.0, "C": 0.0} for i in range(4)})
    assert (
        classify_gate5(estimands, engineering_pass=True, manipulation_pass=True)
        == "GATE5_NO_DURATION_EFFECT"
    )


def test_gate5_manipulation_accepts_exact_frozen_boundary_after_float_rounding() -> None:
    manipulation = {
        name: {"validity": 1.0, "semantic_change_rate": 0.0}
        for name in (
            "ONE_SHOT_PLUS",
            "ONE_SHOT_MINUS",
            "SUSTAINED_PLUS",
            "SUSTAINED_MINUS",
            "SUSTAINED_RANDOM_R0",
            "SUSTAINED_RANDOM_R1",
            "SUSTAINED_RANDOM_R2",
            "SUSTAINED_RANDOM_R3",
        )
    }
    manipulation["ONE_SHOT_PLUS"]["semantic_change_rate"] = 0.05
    manipulation["SUSTAINED_PLUS"]["semantic_change_rate"] = 0.15
    manipulation["SUSTAINED_RANDOM_R0"]["semantic_change_rate"] = 0.10
    manipulation["SUSTAINED_RANDOM_R1"]["semantic_change_rate"] = 0.05
    manipulation["SUSTAINED_RANDOM_R2"]["semantic_change_rate"] = 0.20
    manipulation["SUSTAINED_RANDOM_R3"]["semantic_change_rate"] = 0.05
    assert classify_manipulation(manipulation)
