"""Paired baseline-versus-treatment metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np

from epistemic_geometry.metrics.errors import accuracy, double_fault, error_jaccard, phi_correlation
from epistemic_geometry.types import Prediction


def _paired_arrays(predictions: list[Prediction]) -> tuple[np.ndarray, np.ndarray, Counter[str]]:
    baseline = [item for item in predictions if item.condition == "baseline"]
    treatment = [item for item in predictions if item.condition == "steered"]
    if len(baseline) != len(treatment) or not baseline:
        raise ValueError("Need equal non-empty baseline and steered prediction sets")
    base_by_id = {item.item_id: item for item in baseline}
    treat_by_id = {item.item_id: item for item in treatment}
    if len(base_by_id) != len(baseline) or len(treat_by_id) != len(treatment):
        raise ValueError("Prediction item IDs must be unique within each condition")
    if set(base_by_id) != set(treat_by_id):
        raise ValueError("Baseline and steered predictions must contain the same item IDs")
    ordered_ids = [item.item_id for item in baseline]
    base_errors = np.array([not base_by_id[item_id].correct for item_id in ordered_ids], dtype=bool)
    treat_errors = np.array(
        [not treat_by_id[item_id].correct for item_id in ordered_ids], dtype=bool
    )
    counts = Counter(
        f"baseline_{'correct' if not base_error else 'wrong'}__"
        f"treatment_{'correct' if not treat_error else 'wrong'}"
        for base_error, treat_error in zip(base_errors, treat_errors, strict=True)
    )
    return base_errors, treat_errors, counts


def compute_paired_metrics(predictions: list[Prediction]) -> dict[str, float | dict[str, int]]:
    """Compute a small, accuracy-aware descriptive metric set."""

    base_errors, treat_errors, counts = _paired_arrays(predictions)
    base_correct = ~base_errors
    treat_correct = ~treat_errors
    baseline_accuracy = accuracy(base_correct)
    treatment_accuracy = accuracy(treat_correct)
    baseline_error_count = int(base_errors.sum())
    baseline_success_count = int(base_correct.sum())
    rescue_count = int(np.logical_and(base_errors, ~treat_errors).sum())
    damage_count = int(np.logical_and(base_correct, treat_errors).sum())
    pair_oracle_accuracy = float(np.logical_or(base_correct, treat_correct).mean())
    return {
        "n_items": float(base_errors.size),
        "parse_failure_count": float(
            sum(prediction.parse_status != "OK" for prediction in predictions)
        ),
        "parse_status_counts": dict(
            Counter(prediction.parse_status for prediction in predictions)
        ),
        "baseline_accuracy": baseline_accuracy,
        "treatment_accuracy": treatment_accuracy,
        "delta_accuracy": treatment_accuracy - baseline_accuracy,
        "error_correlation_phi": phi_correlation(base_errors, treat_errors),
        "error_correlation_phi_status": (
            "undefined_zero_variance"
            if np.std(base_errors) == 0 or np.std(treat_errors) == 0
            else "defined"
        ),
        "error_jaccard": error_jaccard(base_errors, treat_errors),
        "disagreement_rate": float(np.not_equal(base_errors, treat_errors).mean()),
        "double_fault": double_fault(base_errors, treat_errors),
        "rescue_rate": float(rescue_count / baseline_error_count)
        if baseline_error_count
        else float("nan"),
        "damage_rate": float(damage_count / baseline_success_count)
        if baseline_success_count
        else float("nan"),
        "pair_oracle_accuracy": pair_oracle_accuracy,
        "complementarity_headroom": pair_oracle_accuracy
        - max(baseline_accuracy, treatment_accuracy),
        "pair_counts": dict(counts),
        "paired_2x2": {
            "baseline_correct__treatment_correct": counts.get(
                "baseline_correct__treatment_correct", 0
            ),
            "baseline_correct__treatment_wrong": counts.get(
                "baseline_correct__treatment_wrong", 0
            ),
            "baseline_wrong__treatment_correct": counts.get(
                "baseline_wrong__treatment_correct", 0
            ),
            "baseline_wrong__treatment_wrong": counts.get(
                "baseline_wrong__treatment_wrong", 0
            ),
        },
    }
