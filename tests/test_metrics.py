import math

import numpy as np

from epistemic_geometry.metrics import compute_paired_metrics
from epistemic_geometry.metrics.errors import error_jaccard, phi_correlation
from epistemic_geometry.metrics.uncertainty import bootstrap_paired_metrics
from epistemic_geometry.types import Prediction


def _prediction(item_id: str, condition: str, correct: bool) -> Prediction:
    return Prediction(item_id, condition, "A", "A", "A", correct)


def test_paired_metrics_hand_constructed() -> None:
    predictions = [
        _prediction("1", "baseline", True),
        _prediction("1", "steered", True),
        _prediction("2", "baseline", False),
        _prediction("2", "steered", True),
        _prediction("3", "baseline", True),
        _prediction("3", "steered", False),
        _prediction("4", "baseline", False),
        _prediction("4", "steered", False),
    ]
    metrics = compute_paired_metrics(predictions)
    assert metrics["baseline_accuracy"] == 0.5
    assert metrics["treatment_accuracy"] == 0.5
    assert metrics["delta_accuracy"] == 0.0
    assert np.isclose(metrics["error_correlation_phi"], 0.0)
    assert np.isclose(metrics["error_jaccard"], 1 / 3)
    assert metrics["disagreement_rate"] == 0.5
    assert metrics["double_fault"] == 0.25
    assert metrics["rescue_rate"] == 0.5
    assert metrics["damage_rate"] == 0.5
    assert metrics["pair_oracle_accuracy"] == 0.75
    assert metrics["complementarity_headroom"] == 0.25
    assert metrics["paired_2x2"]["baseline_wrong__treatment_correct"] == 1


def test_disagreement_is_not_complementarity_by_itself() -> None:
    terrible = compute_paired_metrics(
        [
            _prediction("1", "baseline", True),
            _prediction("1", "steered", False),
            _prediction("2", "baseline", True),
            _prediction("2", "steered", False),
            _prediction("3", "baseline", True),
            _prediction("3", "steered", False),
            _prediction("4", "baseline", True),
            _prediction("4", "steered", False),
        ]
    )
    preserved = compute_paired_metrics(
        [
            _prediction("1", "baseline", True),
            _prediction("1", "steered", True),
            _prediction("2", "baseline", False),
            _prediction("2", "steered", True),
            _prediction("3", "baseline", False),
            _prediction("3", "steered", False),
            _prediction("4", "baseline", True),
            _prediction("4", "steered", False),
        ]
    )
    assert terrible["disagreement_rate"] > preserved["disagreement_rate"]
    assert terrible["treatment_accuracy"] < preserved["treatment_accuracy"]
    assert preserved["rescue_rate"] > 0


def test_bootstrap_is_deterministic_and_descriptive() -> None:
    predictions = [
        _prediction("1", "baseline", True),
        _prediction("1", "steered", True),
        _prediction("2", "baseline", False),
        _prediction("2", "steered", True),
        _prediction("3", "baseline", True),
        _prediction("3", "steered", False),
        _prediction("4", "baseline", False),
        _prediction("4", "steered", False),
    ]
    first = bootstrap_paired_metrics(predictions, seed=7, n_resamples=20)
    second = bootstrap_paired_metrics(predictions, seed=7, n_resamples=20)
    assert first == second
    assert first["method"] == "item_bootstrap_descriptive"


def test_degenerate_error_vectors_have_documented_conventions() -> None:
    zeros = np.array([False, False, False])
    ones = np.array([True, True, True])
    assert math.isnan(phi_correlation(zeros, zeros))
    assert math.isnan(phi_correlation(zeros, ones))
    assert error_jaccard(zeros, zeros) == 1.0
    metrics = compute_paired_metrics(
        [
            _prediction("1", "baseline", True),
            _prediction("1", "steered", True),
            _prediction("2", "baseline", True),
            _prediction("2", "steered", True),
        ]
    )
    assert math.isnan(metrics["rescue_rate"])
    assert metrics["damage_rate"] == 0.0
    assert metrics["error_correlation_phi_status"] == "undefined_zero_variance"
    all_errors = compute_paired_metrics(
        [
            _prediction("1", "baseline", False),
            _prediction("1", "steered", False),
            _prediction("2", "baseline", False),
            _prediction("2", "steered", False),
        ]
    )
    assert math.isnan(all_errors["damage_rate"])
