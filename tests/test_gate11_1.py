from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments import gate11_1


def test_snapshot_order_has_prefill_and_only_available_checkpoints() -> None:
    labels = gate11_1.snapshot_labels({"PREFILL": {}, "0": {}, "3": {}, "15": {}})
    assert labels == ["PREFILL", "0", "3", "15"]
    assert gate11_1.snapshot_token_indices(labels).tolist() == [-1, 0, 3, 15]


def test_logit_metrics_use_full_vocabulary_and_exact_targets() -> None:
    baseline = np.asarray([[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    condition = baseline.copy()
    condition[0, 1] += 2.0
    condition[1, 0] += 4.0
    metrics = gate11_1.logit_metrics_from_arrays(baseline, condition, np.asarray([1, 0]))
    assert metrics["symmetric_js"].shape == (2,)
    assert np.all(metrics["symmetric_js"] >= 0)
    assert metrics["top1_flip"].tolist() == [0, 1]
    assert np.isfinite(metrics["target_logprob_shift"]).all()


def test_hidden_difference_metrics_preserve_every_captured_layer() -> None:
    differences = np.zeros((2, len(gate11_1.PROPAGATION_LAYERS), 4), dtype=np.float32)
    differences[1, -1, :] = 3.0
    metrics = gate11_1.hidden_metrics_from_differences(differences)
    assert set(metrics) == {"A27", "A28", "A30", "A32", "A35"}
    assert metrics["A35"].tolist() == [0.0, 6.0]


def test_raw_schema_constants_freeze_artifact_contract() -> None:
    assert gate11_1.RAW_SCHEMA_VERSION == 1
    assert gate11_1.RAW_DTYPE == "float32"
    assert gate11_1.CONDITIONS == (
        "TF_BASELINE",
        "TF_TEXTUAL_CAREFUL",
        "TF_MEANINGFUL_L27_D75",
        "TF_RANDOM_R0",
        "TF_RANDOM_R1",
        "TF_RANDOM_R2",
        "TF_RANDOM_R3",
    )
