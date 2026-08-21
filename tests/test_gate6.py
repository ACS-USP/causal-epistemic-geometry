from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments.gate6 import (
    LAYERS,
    classify_gate6_movement,
    covariance_spectrum,
    item_cluster_bootstrap,
    orthogonal_random_bank,
    paired_mean_direction,
    source_readout_metrics,
    standardized_budget,
    symmetric_first_stage_contributions,
    two_rollout_estimands,
)
from epistemic_geometry.steering.gate6 import Gate6HookTrace


def test_gate6_frozen_layers_and_paired_direction_are_deterministic() -> None:
    assert LAYERS == (8, 12, 17, 22, 27, 32)
    direct = np.zeros((4, 3))
    careful = direct.copy()
    careful[:, 0] = [1.0, 2.0, 1.0, 2.0]
    direction, delta, raw = paired_mean_direction(careful, direct)
    assert np.allclose(direction, [1.0, 0.0, 0.0])
    assert np.isclose(delta, 1.5)
    assert np.allclose(raw, [1.5, 0.0, 0.0])


def test_gate6_random_bank_is_orthogonal_and_reproducible() -> None:
    meaningful = np.eye(16, dtype=np.float64)[0]
    first = orthogonal_random_bank(meaningful, seeds=(11, 12, 13, 14))
    second = orthogonal_random_bank(meaningful, seeds=(11, 12, 13, 14))
    assert all(np.array_equal(first[name], second[name]) for name in first)
    vectors = [meaningful, *first.values()]
    for index, left in enumerate(vectors):
        assert np.isclose(np.linalg.norm(left), 1.0)
        for right in vectors[index + 1 :]:
            assert abs(float(np.dot(left, right))) < 1e-12


def test_gate6_spectrum_and_standardized_budget() -> None:
    values = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    atlas = covariance_spectrum(values)
    assert atlas["eigenvalues_top16"][0] > 0
    delta = standardized_budget(np.asarray([1.0, 0.0]), values, eta=2.0, n_layers=2)
    assert np.isclose(np.linalg.norm(delta), 2.0 * np.std(values[:, 0], ddof=1) / np.sqrt(2))


def test_gate6_source_metrics_and_estimand_identities() -> None:
    careful = np.asarray([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    direct = np.zeros_like(careful)
    metrics = source_readout_metrics(np.asarray([1.0, 0.0]), careful, direct)
    assert np.isclose(metrics["auroc"], 1.0)
    baseline = np.asarray([[1, 1], [1, 0], [0, 0], [0, 1]])
    condition = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]])
    result = two_rollout_estimands(baseline, condition)
    assert np.isclose(
        result["rescue"] - result["damage"],
        result["accuracy_condition"] - result["accuracy_baseline"],
    )
    intervals = item_cluster_bootstrap(baseline, {"condition": condition}, resamples=25, seed=7)
    assert set(intervals["condition"]) == {"accuracy_change", "G", "C", "D", "rescue", "damage"}


def test_gate6_symmetric_first_stage_formula_is_sign_sensitive() -> None:
    values = symmetric_first_stage_contributions([3.0, 4.0], [1.0, 2.0], [1.0, 1.0], [2.0, 3.0])
    assert np.allclose(values, [1.5, 2.0])


def test_gate6_classification_uses_random_and_reference_controls() -> None:
    baseline = {"validity": 1.0, "accuracy": 0.60}
    meaningful = {"validity": 1.0, "accuracy": 0.60, "D": 0.10, "G": 0.04, "C": 0.06}
    random = [{"D": 0.01, "C": 0.0} for _ in range(4)]
    reference = {"D": 0.02}
    movement, useful = classify_gate6_movement(
        meaningful,
        baseline=baseline,
        random=random,
        best_single=reference,
        multilayer_mean=reference,
    )
    assert movement is True
    assert useful is True


def test_gate6_multilayer_hook_shifts_only_current_positions_and_cleans_up() -> None:
    torch = pytest.importorskip("torch")
    layer_a = torch.nn.Identity()
    layer_b = torch.nn.Identity()
    values = torch.zeros((2, 4, 3), dtype=torch.float32)
    deltas = {8: torch.tensor([1.0, 2.0, 3.0]), 12: torch.tensor([-1.0, 0.5, 2.0])}
    with Gate6HookTrace({8: layer_a, 12: layer_b}, deltas=deltas, target_positions=(3, 1)) as trace:
        output = layer_b(layer_a(values))
        assert torch.allclose(output[0, 3], deltas[8] + deltas[12])
        assert torch.allclose(output[1, 1], deltas[8] + deltas[12])
        assert torch.allclose(output[0, 0], torch.zeros(3))
        assert trace.forward_count == 1
        assert all(entry["shift_error"] == 0.0 for entry in trace.applications)
        assert all(entry["relative_shift_error"] == 0.0 for entry in trace.applications)
        assert all(entry["non_current_change"] == 0.0 for entry in trace.applications)
    assert len(layer_a._forward_hooks) == 0
    assert len(layer_b._forward_hooks) == 0
