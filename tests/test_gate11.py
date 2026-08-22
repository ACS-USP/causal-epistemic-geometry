from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments import gate11


def test_selection_is_deterministic_and_nested() -> None:
    items = [f"item_{index:03d}" for index in range(100)]
    source, propagation = gate11.select_items("CRUXEval", items)
    assert source == gate11.select_items("CRUXEval", list(reversed(items)))[0]
    assert len(source) == 40
    assert propagation == source[:24]


def test_baseline_sequence_fallback_is_mechanical() -> None:
    rows = {
        ("x", 0): {"generated_token_ids": []},
        ("x", 1): {"generated_token_ids": list(range(300)), "status": "VALID_WRONG"},
    }
    selected = gate11.choose_baseline_sequence(rows, "x")
    assert selected["selected_rollout_index"] == 1
    assert selected["continuation_length"] == 256
    assert selected["truncated_at_cap"] is True


def test_source_axis_and_relative_dose() -> None:
    direct = np.zeros((4, 3))
    careful = np.asarray([[1, 0, 0], [2, 0, 0], [1, 1, 0], [2, -1, 0]], dtype=float)
    result = gate11.source_axis_metrics(careful, direct)
    assert result["mean_gap"] > 0
    assert result["positive_gap_fraction"] == 1
    geometry = gate11.relative_dose_geometry(direct, careful, direct, result["direction"], 0.5)
    assert geometry["delta_over_gap"] > 0


def test_logit_metrics_identity_and_shift() -> None:
    baseline = np.asarray([0.0, 1.0, 2.0])
    identity = gate11.logit_metrics(baseline, baseline.copy(), 2)
    assert abs(identity["next_token_kl"]) < 1e-12
    assert abs(identity["symmetric_js"]) < 1e-12
    assert identity["top1_flip"] is False
    shifted = gate11.logit_metrics(baseline, np.asarray([3.0, 1.0, 2.0]), 2)
    assert shifted["next_token_kl"] > 0
    assert shifted["top1_flip"] is True


def test_primary_synthesis_is_exhaustive() -> None:
    assert (
        gate11.classify_components(
            source_transfer=False,
            control_gain_shift=False,
            policy_realization_shift=False,
            policy_utility_shift=False,
        )
        == "GATE11_SOURCE_AXIS_DOMAIN_MISMATCH"
    )
    assert (
        gate11.classify_components(
            source_transfer=False,
            control_gain_shift=True,
            policy_realization_shift=False,
            policy_utility_shift=False,
        )
        == "GATE11_MULTIPLE_DOMAIN_CONDITIONING_FACTORS"
    )
