"""Tests for the independent, no-inference V1.2 aggregation audit."""

from __future__ import annotations

import numpy as np

from epistemic_geometry.analysis.v1_v2_audit import (
    PRIMARY_ROLES,
    _flip_robustness,
    _instability_overlap,
    paired_metrics,
    recompute_aggregates,
    validate_raw_rows,
)


def _raw_row(
    item_id: str,
    shift: int,
    role: str,
    semantic_ids: list[int],
    scores: list[float],
    target: int = 0,
) -> dict:
    labels = ["A", "B"]
    score_map = dict(zip(labels, scores, strict=True))
    predicted_label = max(labels, key=lambda label: score_map[label])
    predicted_semantic = semantic_ids[labels.index(predicted_label)]
    return {
        "item_id": item_id,
        "cyclic_shift": shift,
        "condition": f"cyclic_{shift:02d}_{role}",
        "role": role,
        "option_count": 2,
        "candidate_labels": labels,
        "semantic_option_ids": semantic_ids,
        "candidate_scores": score_map,
        "target_semantic_original_index": target,
        "target_displayed_label": labels[semantic_ids.index(target)],
        "predicted_displayed_label": predicted_label,
        "predicted_semantic_original_index": predicted_semantic,
        "correct": predicted_semantic == target,
        "rendered_prompt_hash": f"prompt-{item_id}-{shift}",
        "candidate_score_semantics": "candidate_logits_no_vocab_normalization",
        "condition_spec": {"role": role, "cyclic_shift": shift},
    }


def _fixture_rows() -> list[dict]:
    rows: list[dict] = []
    for role in ("baseline", "pc1_minus", "pc1_plus", "probe_minus", "probe_plus"):
        rows.append(_raw_row("item-1", 0, role, [0, 1], [2.0, 1.0]))
        rows.append(_raw_row("item-1", 1, role, [1, 0], [1.0, 2.0]))
    return rows


def test_raw_validation_and_recomputation_are_deterministic() -> None:
    rows = _fixture_rows()
    index = validate_raw_rows(rows)
    first, instability = recompute_aggregates(rows, index)
    second, instability_again = recompute_aggregates(rows, index)

    assert index["row_count"] == 10
    assert len(first) == len(PRIMARY_ROLES)
    assert first == second
    assert instability == instability_again
    assert first[0]["predicted_semantic_original_index"] == 0
    assert first[0]["probability_mean_prediction"] == 0
    assert np.isclose(first[0]["symmetrized_margin"], 1.0)


def test_paired_metrics_preserve_undefined_phi_status() -> None:
    baseline = {"a": True, "b": True, "c": True}
    treatment = {"a": True, "b": True, "c": True}
    metrics = paired_metrics(baseline, treatment, "pc1_plus_S")

    assert metrics["baseline_accuracy"] == 1.0
    assert metrics["treatment_accuracy"] == 1.0
    assert metrics["error_correlation_phi"] is None
    assert metrics["error_correlation_phi_status"] == "undefined_zero_variance"
    assert metrics["error_jaccard"] == 1.0
    assert metrics["rescue_rate"] is None
    assert metrics["damage_rate"] == 0.0


def test_instability_overlap_and_flip_sets_are_explicit() -> None:
    rows = []
    for role in PRIMARY_ROLES:
        for item_id, prediction in (("a", 0), ("b", 1)):
            rows.append(
                {
                    "item_id": item_id,
                    "role": role,
                    "predicted_semantic_original_index": prediction,
                    "probability_mean_prediction": prediction if item_id == "a" else 0,
                    "target_semantic_original_index": 0,
                    "correct": prediction == 0,
                    "symmetrized_margin": 1.0,
                    "probability_mean_margin": 0.5,
                }
            )
    instability = _instability_overlap(rows)
    assert instability["pc1_plus"]["both_count"] == 1
    assert instability["pc1_plus"]["baseline_only_count"] == 0
    flip, audit_rows = _flip_robustness(rows)
    assert len(audit_rows) == 2
    assert flip["pc1_plus"]["semantic_flip_sets"]["S_count"] == 0
