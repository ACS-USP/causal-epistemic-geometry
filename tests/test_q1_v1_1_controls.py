"""Local tests for the frozen V1.1 control table and workload estimate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from epistemic_geometry.config import load_config
from epistemic_geometry.experiments.q1_v1_1 import (
    ORIGINAL_VECTOR_HASHES,
    PERMUTATION_IDS,
    _frozen_conditions,
    estimate_v1_v1,
)
from epistemic_geometry.types import SteeringVector


def _fake_reference() -> dict[str, object]:
    vectors = {
        name: SteeringVector(
            values=np.eye(4, dtype=np.float64)[index % 4],
            layer=17,
            constructor="test_fixture",
            normalization="unit",
            hash=vector_hash,
        )
        for index, (name, vector_hash) in enumerate(ORIGINAL_VECTOR_HASHES.items())
    }
    condition_specs = {
        f"{name}_plus": {"direction_sd_calibration": 0.1 + index}
        for index, name in enumerate(ORIGINAL_VECTOR_HASHES)
    }
    condition_specs["pca_pc1_minus"] = {"direction_sd_calibration": 0.5}
    condition_specs["pca_pc1_plus"] = {"direction_sd_calibration": 0.5}
    return {
        "vectors": vectors,
        "calibration_median_hidden_norm": 10.0,
        "alpha_pc1_minus": -2.5,
        "alpha_pc1_plus": 2.5,
        "condition_specs": condition_specs,
    }


def test_v1_1_condition_table_has_frozen_counts_and_norm_matching() -> None:
    specs, _vectors = _frozen_conditions(_fake_reference())

    assert len(specs) == 19
    assert sum(spec["family"] == "random_native_scale" for spec in specs) == 8
    assert sum(spec["family"] == "random_pc1_normmatched" for spec in specs) == 8
    normmatched = [spec for spec in specs if spec["family"] == "random_pc1_normmatched"]
    assert {abs(spec["alpha"]) for spec in normmatched} == {2.5}
    assert {spec["intervention_norm"] for spec in normmatched} == {2.5}
    assert PERMUTATION_IDS == (
        "permutation_0",
        "permutation_1",
        "permutation_2",
        "permutation_3",
    )


def test_v1_1_workload_estimate_is_below_frozen_cost_gate() -> None:
    config = load_config(Path("configs/q1_v1_1_qwen3_8b.yaml"))
    estimate = estimate_v1_v1(config)

    assert estimate["total_conditions"] == 31
    assert estimate["item_condition_evaluations"] == 512 * 31
    assert estimate["candidate_forward_passes"] == 512 * 31 * 10
    assert estimate["cost_gate_pass"] is True
    assert estimate["estimated_a40_cost_usd"] < 2.0
