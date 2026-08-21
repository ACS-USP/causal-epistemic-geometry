from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.gate6_3_v3 import (
    audit_two_rollout_estimands,
    classify_semantic_v3,
    item_contributions,
    random_metric_summary,
)


def test_independent_estimands_and_item_contributions_are_algebraically_exact() -> None:
    baseline = np.asarray([[1, 1], [1, 0], [0, 0], [1, 1]], dtype=np.int8)
    condition = np.asarray([[1, 0], [0, 0], [0, 1], [1, 1]], dtype=np.int8)
    result = audit_two_rollout_estimands(baseline, condition)
    contributions = item_contributions(baseline, condition)
    for metric in ("G", "C", "D", "rescue", "damage"):
        assert np.isclose(sum(row[metric] for row in contributions), result[metric])
    assert np.isclose(
        result["rescue"] - result["damage"],
        result["accuracy_condition"] - result["accuracy_baseline"],
    )


def test_random_summary_uses_all_controls() -> None:
    estimands = {
        f"R{index}": {"G": value, "C": value / 2, "D": value * 2, "accuracy_condition": 0.5}
        for index, value in enumerate((0.0, 0.1, 0.2, 0.3))
    }
    summary = random_metric_summary(estimands, ("R0", "R1", "R2", "R3"))
    assert np.isclose(summary["G"]["mean"], 0.15)
    assert np.isclose(summary["C"]["max"], 0.15)
    assert np.isclose(summary["D"]["min"], 0.0)


def test_frozen_v3_classification_rules_are_exhaustive() -> None:
    baseline = {"commitment_validity": 0.98, "semantic_evaluability": 0.98, "accuracy": 0.5}
    controller = {"commitment_validity": 0.95, "semantic_evaluability": 0.95, "accuracy": 0.6}
    estimands = {"G": 0.18, "C": 0.11, "D": 0.16}
    random = {
        "G": {"mean": 0.0, "max": 0.02},
        "C": {"mean": 0.01, "max": 0.03},
        "D": {"mean": 0.03, "max": 0.05},
    }
    classification, guards = classify_semantic_v3(
        baseline_summary=baseline,
        controller_summary=controller,
        controller_estimands=estimands,
        random_summary=random,
    )
    assert classification == "GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL"
    assert guards["movement"] and guards["useful_complementarity"]

    destructive = dict(controller)
    destructive["commitment_validity"] = 0.90
    classification, _guards = classify_semantic_v3(
        baseline_summary=baseline,
        controller_summary=destructive,
        controller_estimands=estimands,
        random_summary=random,
    )
    assert classification == "GATE6_3_V3_VALIDITY_COST_CONFIRMED"
