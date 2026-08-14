"""Pure vector geometry helpers reserved for future Q2 scaffolding."""

from __future__ import annotations

import numpy as np

from epistemic_geometry.steering.vector import with_computed_hash
from epistemic_geometry.types import SteeringVector


def cosine_similarity(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return cosine similarity; zero vectors are explicitly undefined."""

    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Vectors must have equal dimensions")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def normalized_euclidean_distance(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Return Euclidean distance after unit normalization."""

    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Vectors must have equal dimensions")
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot normalize a zero vector")
    return float(np.linalg.norm(a / norm_a - b / norm_b))


def match_norm(reference: SteeringVector, direction: SteeringVector) -> SteeringVector:
    """Scale a direction to a reference norm for a future null control."""

    if reference.dimension != direction.dimension:
        raise ValueError("Reference and direction dimensions must match")
    direction_norm = float(np.linalg.norm(direction.values))
    if direction_norm == 0:
        raise ValueError("Cannot norm-match a zero direction")
    values = direction.values * (np.linalg.norm(reference.values) / direction_norm)
    return with_computed_hash(
        SteeringVector(
            values=values,
            layer=direction.layer,
            constructor="norm_matched:" + direction.constructor,
            normalization="matched_norm",
            metadata={
                **direction.metadata,
                "reference_vector_hash": reference.hash,
                "reference_norm": float(np.linalg.norm(reference.values)),
            },
        )
    )

