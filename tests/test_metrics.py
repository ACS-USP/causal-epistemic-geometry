import math

import numpy as np

from epistemic_geometry.metrics import compute_paired_metrics
from epistemic_geometry.metrics.errors import error_jaccard, phi_correlation
from epistemic_geometry.metrics.uncertainty import (
    bootstrap_paired_metrics,
    cluster_bootstrap_mean,
)
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
    assert metrics["rescue_fraction"] == 0.25
    assert metrics["damage_fraction"] == 0.25
    assert metrics["net_flip_fraction"] == metrics["delta_accuracy"] == 0.0
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
    assert first["net_flip_fraction_interval"] == first["delta_accuracy_interval"]
    assert first["rescue_minus_damage_interval_status"].startswith("DEPRECATED")


def test_all_paired_cells_preserve_conditional_and_population_denominators() -> None:
    # n_cc=2, n_cw=1, n_wc=3, n_ww=4. Baseline errors=7; successes=3.
    cells = [(True, True)] * 2 + [(True, False)] + [(False, True)] * 3 + [
        (False, False)
    ] * 4
    predictions = []
    for index, (base_correct, treatment_correct) in enumerate(cells):
        item_id = str(index)
        predictions.extend(
            [
                _prediction(item_id, "baseline", base_correct),
                _prediction(item_id, "steered", treatment_correct),
            ]
        )
    metrics = compute_paired_metrics(predictions)
    assert metrics["rescue_rate"] == 3 / 7
    assert metrics["damage_rate"] == 1 / 3
    assert metrics["rescue_fraction"] == 0.3
    assert metrics["damage_fraction"] == 0.1
    assert metrics["net_flip_fraction"] == 0.2
    assert np.isclose(metrics["delta_accuracy"], 0.2)


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


def test_cluster_bootstrap_uses_problem_not_test_as_scientific_unit() -> None:
    # Problem A has 100 redundant passing tests; problem B has one failing test.
    # Equal-cluster mean is 0.5, not the row-weighted 1/101.
    values = [0.0] * 100 + [1.0]
    clusters = ["problem-a"] * 100 + ["problem-b"]
    first = cluster_bootstrap_mean(values, clusters, seed=4, n_resamples=100)
    second = cluster_bootstrap_mean(values, clusters, seed=4, n_resamples=100)
    assert first == second
    assert first["estimate"] == 0.5
    assert first["n_clusters"] == 2
    assert first["n_nested_observations"] == 101
