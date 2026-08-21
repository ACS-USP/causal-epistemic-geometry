"""Independent two-rollout estimands for the Gate 6.3 semantic-V3 audit."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def audit_two_rollout_estimands(
    baseline_errors: np.ndarray, condition_errors: np.ndarray
) -> dict[str, float]:
    """Recompute Gate estimands directly, without the historical analysis path."""

    baseline = np.asarray(baseline_errors, dtype=np.float64)
    condition = np.asarray(condition_errors, dtype=np.float64)
    if baseline.shape != condition.shape or baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("two-rollout audit arrays must both have shape (items, 2)")
    if len(baseline) < 2:
        raise ValueError("audit estimands require at least two items")

    b00 = float(np.mean(baseline[:, 0] * baseline[:, 1]))
    cross = (
        baseline[:, 0] * condition[:, 0]
        + baseline[:, 0] * condition[:, 1]
        + baseline[:, 1] * condition[:, 0]
        + baseline[:, 1] * condition[:, 1]
    ) / 4.0
    b0j = float(np.mean(cross))
    q0 = baseline.mean(axis=1)
    qj = condition.mean(axis=1)
    item_count = len(q0)
    denominator = item_count * (item_count - 1)
    u00 = float((q0.sum() ** 2 - np.square(q0).sum()) / denominator)
    u0j = float((q0.sum() * qj.sum() - np.dot(q0, qj)) / denominator)
    distance = float(
        np.mean(
            baseline[:, 0] * baseline[:, 1]
            + condition[:, 0] * condition[:, 1]
            - baseline[:, 0] * condition[:, 1]
            - baseline[:, 1] * condition[:, 0]
        )
    )
    rescue = float(
        np.mean(
            (
                baseline[:, 0] * (1 - condition[:, 0])
                + baseline[:, 0] * (1 - condition[:, 1])
                + baseline[:, 1] * (1 - condition[:, 0])
                + baseline[:, 1] * (1 - condition[:, 1])
            )
            / 4.0
        )
    )
    damage = float(
        np.mean(
            (
                (1 - baseline[:, 0]) * condition[:, 0]
                + (1 - baseline[:, 0]) * condition[:, 1]
                + (1 - baseline[:, 1]) * condition[:, 0]
                + (1 - baseline[:, 1]) * condition[:, 1]
            )
            / 4.0
        )
    )
    result = {
        "accuracy_baseline": float(1 - baseline.mean()),
        "accuracy_condition": float(1 - condition.mean()),
        "B00": b00,
        "O00": 1.0 - b00,
        "B0j": b0j,
        "O0j": 1.0 - b0j,
        "G": b00 - b0j,
        "U00": u00,
        "U0j": u0j,
        "C": b00 - b0j - u00 + u0j,
        "D": distance,
        "rescue": rescue,
        "damage": damage,
    }
    if not np.isclose(
        rescue - damage,
        result["accuracy_condition"] - result["accuracy_baseline"],
        atol=1e-12,
    ):
        raise AssertionError("rescue-damage identity failed")
    return result


def random_metric_summary(
    estimands: Mapping[str, Mapping[str, float]], random_names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Summarize the complete random-control bank without exemplar selection."""

    return {
        metric: {
            "mean": float(np.mean([estimands[name][metric] for name in random_names])),
            "median": float(np.median([estimands[name][metric] for name in random_names])),
            "min": float(np.min([estimands[name][metric] for name in random_names])),
            "max": float(np.max([estimands[name][metric] for name in random_names])),
        }
        for metric in ("G", "C", "D", "accuracy_condition")
    }


def classify_semantic_v3(
    *,
    baseline_summary: Mapping[str, float],
    controller_summary: Mapping[str, float],
    controller_estimands: Mapping[str, float],
    random_summary: Mapping[str, Mapping[str, float]],
) -> tuple[str, dict[str, bool | float]]:
    """Apply the frozen diagnostic guards and return one exhaustive class."""

    commitment_guard = bool(
        controller_summary["commitment_validity"] >= 0.90
        and controller_summary["commitment_validity"]
        >= baseline_summary["commitment_validity"] - 0.05
    )
    evaluability_guard = bool(
        controller_summary["semantic_evaluability"] >= 0.90
        and controller_summary["semantic_evaluability"]
        >= baseline_summary["semantic_evaluability"] - 0.05
    )
    competence_guard = bool(
        controller_summary["accuracy"] >= baseline_summary["accuracy"] - 0.10
    )
    movement = bool(
        controller_estimands["D"] >= 0.05
        and controller_estimands["D"] - random_summary["D"]["mean"] >= 0.05
        and controller_estimands["D"] > random_summary["D"]["max"]
    )
    useful = bool(
        controller_estimands["G"] >= 0.03
        and controller_estimands["C"] >= 0.03
        and controller_estimands["C"] - random_summary["C"]["mean"] >= 0.05
        and controller_estimands["C"] > random_summary["C"]["max"]
    )
    if not commitment_guard or not evaluability_guard:
        classification = "GATE6_3_V3_VALIDITY_COST_CONFIRMED"
    elif not competence_guard:
        classification = "GATE6_3_V3_AUDIT_INCONCLUSIVE"
    elif not movement:
        classification = "GATE6_3_V3_NO_SPECIFIC_MOVEMENT"
    elif competence_guard and useful:
        classification = "GATE6_3_V3_STRONG_SPECIFIC_CONTROL_SIGNAL"
    elif competence_guard:
        classification = "GATE6_3_V3_ERROR_PROFILE_MOVEMENT_ONLY"
    else:
        classification = "GATE6_3_V3_NO_SPECIFIC_MOVEMENT"
    return classification, {
        "commitment_validity_guard": commitment_guard,
        "semantic_evaluability_guard": evaluability_guard,
        "competence_guard": competence_guard,
        "movement": movement,
        "useful_complementarity": useful,
        "D_minus_random_mean": controller_estimands["D"] - random_summary["D"]["mean"],
        "C_minus_random_mean": controller_estimands["C"] - random_summary["C"]["mean"],
    }


def item_contributions(
    baseline_errors: np.ndarray, condition_errors: np.ndarray
) -> list[dict[str, float]]:
    """Return an exact additive per-item decomposition of G/C/D/rescue/damage."""

    baseline = np.asarray(baseline_errors, dtype=np.float64)
    condition = np.asarray(condition_errors, dtype=np.float64)
    if baseline.shape != condition.shape or baseline.ndim != 2 or baseline.shape[1] != 2:
        raise ValueError("two-rollout audit arrays must both have shape (items, 2)")
    item_count = len(baseline)
    q0 = baseline.mean(axis=1)
    qj = condition.mean(axis=1)
    denominator = item_count * (item_count - 1)
    records: list[dict[str, float]] = []
    for index in range(item_count):
        b00 = baseline[index, 0] * baseline[index, 1]
        b0j = float(
            (
                baseline[index, 0] * condition[index, 0]
                + baseline[index, 0] * condition[index, 1]
                + baseline[index, 1] * condition[index, 0]
                + baseline[index, 1] * condition[index, 1]
            )
            / 4.0
        )
        u00 = q0[index] * (q0.sum() - q0[index]) / denominator
        u0j = q0[index] * (qj.sum() - qj[index]) / denominator
        distance = (
            b00
            + condition[index, 0] * condition[index, 1]
            - baseline[index, 0] * condition[index, 1]
            - baseline[index, 1] * condition[index, 0]
        ) / item_count
        rescue = (
            baseline[index, 0] * (1 - condition[index, 0])
            + baseline[index, 0] * (1 - condition[index, 1])
            + baseline[index, 1] * (1 - condition[index, 0])
            + baseline[index, 1] * (1 - condition[index, 1])
        ) / (4 * item_count)
        damage = (
            (1 - baseline[index, 0]) * condition[index, 0]
            + (1 - baseline[index, 0]) * condition[index, 1]
            + (1 - baseline[index, 1]) * condition[index, 0]
            + (1 - baseline[index, 1]) * condition[index, 1]
        ) / (4 * item_count)
        records.append(
            {
                "G": (b00 - b0j) / item_count,
                "C": (b00 - b0j) / item_count - u00 + u0j,
                "D": float(distance),
                "rescue": float(rescue),
                "damage": float(damage),
            }
        )
    return records


__all__ = [
    "audit_two_rollout_estimands",
    "classify_semantic_v3",
    "item_contributions",
    "random_metric_summary",
]
