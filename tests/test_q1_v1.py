"""Pure checks for the fixed Q1 V1 condition table."""

from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q1_v1 import (
    BETA_MAGNITUDE,
    _condition_specs,
)
from epistemic_geometry.types import SteeringVector


def test_q1_v1_has_exactly_fifteen_predeclared_conditions() -> None:
    vectors = {
        name: SteeringVector(
            values=np.eye(4)[index % 4],
            layer=17,
            constructor="test",
            normalization="unit",
            hash=name,
        )
        for index, name in enumerate(
            ["pca_pc1", "pca_pc2", "pca_pc3", "random_0", "random_1", "random_2", "random_3"]
        )
    }
    directions = {
        name: {"scale": float(index + 1), "vector_hash": name, "kind": "test"}
        for index, name in enumerate(vectors)
    }
    conditions = _condition_specs(vectors, directions, mean_activation_norm=10.0)
    assert len(conditions) == 15
    assert conditions[0]["condition"] == "baseline"
    assert sorted(condition["beta"] for condition in conditions[1:]) == [-BETA_MAGNITUDE] * 7 + [
        BETA_MAGNITUDE
    ] * 7
    assert all(condition["relative_shift_norm"] is not None for condition in conditions[1:])


def test_q1_v1_condition_alpha_uses_calibration_scale() -> None:
    vector = SteeringVector(
        values=np.array([1.0, 0.0]),
        layer=17,
        constructor="test",
        normalization="unit",
        hash="v",
    )
    names = ["pca_pc1", "pca_pc2", "pca_pc3", "random_0", "random_1", "random_2", "random_3"]
    vectors = {
        name: vector
        if name == "pca_pc1"
        else SteeringVector(
            values=np.array([0.0, 1.0]),
            layer=17,
            constructor="test",
            normalization="unit",
            hash=name,
        )
        for name in names
    }
    directions = {
        name: {
            "scale": 3.0 if name == "pca_pc1" else 1.0,
            "vector_hash": vector.hash if name == "pca_pc1" else name,
            "kind": "test",
        }
        for name in names
    }
    conditions = _condition_specs(
        vectors, directions, mean_activation_norm=2.0
    )
    assert conditions[1]["alpha"] == -1.5
    assert conditions[2]["alpha"] == 1.5
    assert conditions[1]["relative_shift_norm"] == 0.75
