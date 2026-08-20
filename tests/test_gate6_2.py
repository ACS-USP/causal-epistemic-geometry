from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments.gate6_2 import (
    config_product,
    orthogonal_random_bank,
    paired_stratified_kfold_indices,
    score_teacher_forced_window,
    select_source_cv_config,
    stratified_kfold_indices,
    teacher_forced_score_window,
)


def test_prompt_and_execution_scoring_windows_are_causal() -> None:
    prompt = teacher_forced_score_window(
        source_location="PROMPT_BOUNDARY", continuation_length=6
    )
    execution = teacher_forced_score_window(
        source_location="EXECUTION_BOUNDARY", continuation_length=6, marker_token_index=3
    )
    assert prompt.as_dict() == {
        "intervention_token_index": 0,
        "score_start_index": 0,
        "score_end_index": 6,
        "scored_token_count": 6,
    }
    assert execution.as_dict() == {
        "intervention_token_index": 3,
        "score_start_index": 3,
        "score_end_index": 6,
        "scored_token_count": 3,
    }


def test_execution_score_excludes_pre_boundary_logits() -> None:
    log_probs = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
        ]
    )
    targets = [0, 0, 0, 0, 0]
    window = teacher_forced_score_window(
        source_location="EXECUTION_BOUNDARY", continuation_length=5, marker_token_index=3
    )
    mean, selected = score_teacher_forced_window(log_probs, targets, window)
    assert np.array_equal(selected, [4.0, 5.0])
    assert mean == pytest.approx(4.5)


def test_scoring_window_rejects_missing_or_invalid_execution_boundary() -> None:
    with pytest.raises(ValueError, match="marker_token_index"):
        teacher_forced_score_window(source_location="EXECUTION_BOUNDARY", continuation_length=3)
    with pytest.raises(ValueError, match="outside"):
        teacher_forced_score_window(
            source_location="EXECUTION_BOUNDARY", continuation_length=3, marker_token_index=3
        )


def test_stratified_folds_are_deterministic_disjoint_and_label_balanced() -> None:
    labels = [0] * 8 + [1] * 8
    first = stratified_kfold_indices(labels, n_splits=4, seed=17)
    second = stratified_kfold_indices(labels, n_splits=4, seed=17)
    assert all(
        np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        for a, b in zip(first, second, strict=True)
    )
    validation = np.concatenate([fold[1] for fold in first])
    assert sorted(validation.tolist()) == list(range(len(labels)))
    for _train, val in first:
        assert np.bincount(np.asarray(labels)[val]).tolist() == [2, 2]


def test_source_cv_config_selection_uses_source_only_metrics_and_is_deterministic() -> None:
    configs = config_product(
        iters=(4, 8), bandwidth=(10.0,), exponent=(1.0,), regularization=(1e-3,)
    )
    results = [
        {"config": configs[0], "auroc": 0.80, "best_iter": 3},
        {"config": configs[0], "auroc": 0.82, "best_iter": 4},
        {"config": configs[1], "auroc": 0.91, "best_iter": 7},
        {"config": configs[1], "auroc": 0.89, "best_iter": 8},
    ]
    selected = select_source_cv_config(results)
    assert selected["iters"] == 8
    assert selected["selected_mean_inner_auroc"] == pytest.approx(0.90)
    assert selected["selected_best_iter_values"] == [7, 8]


def test_paired_source_folds_keep_careful_direct_items_together() -> None:
    first = paired_stratified_kfold_indices(12, n_splits=3, seed=21)
    second = paired_stratified_kfold_indices(12, n_splits=3, seed=21)
    for (train_a, val_a), (train_b, val_b) in zip(first, second, strict=True):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)
        assert len(val_a) == 4
        assert not set(train_a) & set(val_a)
    assert np.array_equal(np.sort(np.concatenate([fold[1] for fold in first])), np.arange(12))


def test_gate6_2_random_mean_bank_is_orthogonal_and_reproducible() -> None:
    meaningful = np.eye(32, dtype=np.float64)[0]
    basis = np.eye(32, dtype=np.float64)[1:3]
    first = orthogonal_random_bank(meaningful, seeds=(3, 4, 5, 6), additional_basis=basis)
    second = orthogonal_random_bank(meaningful, seeds=(3, 4, 5, 6), additional_basis=basis)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    vectors = [meaningful, *basis, *first.values()]
    for index, left in enumerate(vectors):
        assert np.isclose(np.linalg.norm(left), 1.0)
        for right in vectors[index + 1 :]:
            assert abs(float(np.dot(left, right))) <= 1e-12


def test_toy_causal_transformer_keeps_pre_boundary_logits_unchanged() -> None:
    torch = pytest.importorskip("torch")
    from epistemic_geometry.steering.gate6 import Gate6HookTrace

    class CausalReadout(torch.nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            # A cumulative readout is a minimal causal analogue: position t
            # can depend on positions <= t, never on future positions.
            return torch.cumsum(hidden, dim=1)

    layer = torch.nn.Identity()
    readout = CausalReadout()
    values = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    delta = torch.tensor([0.25, -0.5, 1.0, 2.0], dtype=torch.float32)
    baseline = readout(layer(values))
    with Gate6HookTrace({17: layer}, deltas={17: delta}, target_positions=[3]) as trace:
        shifted = readout(layer(values))

    assert trace.forward_count == 1
    assert torch.equal(shifted[:, :3], baseline[:, :3])
    expected_change = torch.stack((delta, delta), dim=0).view(1, 2, -1)
    assert torch.allclose(shifted[:, 3:] - baseline[:, 3:], expected_change)
    window = teacher_forced_score_window(
        source_location="EXECUTION_BOUNDARY", continuation_length=5, marker_token_index=3
    )
    assert window.score_start_index == 3
    assert window.scored_token_count == 2
