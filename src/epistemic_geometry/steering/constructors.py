"""Small development-stage steering-vector constructors."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epistemic_geometry.backends.base import ModelBackend
from epistemic_geometry.reproducibility import stable_seed
from epistemic_geometry.steering.vector import with_computed_hash
from epistemic_geometry.types import BenchmarkItem, SteeringVector


def _normalize(values: np.ndarray, normalize: bool) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if normalize:
        if norm == 0:
            raise ValueError("Cannot normalize a zero steering vector")
        return values / norm, "unit"
    return values, "none"


def random_unit_vector(
    dimension: int,
    seed: int,
    layer: int = 0,
    metadata: dict[str, object] | None = None,
) -> SteeringVector:
    """Construct a deterministic unit vector for null-control development runs."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    rng = np.random.default_rng(stable_seed("random_unit_vector", seed, dimension))
    values, normalization = _normalize(rng.normal(size=dimension), normalize=True)
    return with_computed_hash(
        SteeringVector(
            values=values,
            layer=layer,
            constructor="random_unit",
            normalization=normalization,
            metadata={"creation_seed": seed, **(metadata or {})},
        )
    )


def orthogonal_random_directions(
    dimension: int,
    count: int,
    seed: int,
    layer: int = 0,
) -> list[SteeringVector]:
    """Create a small deterministic orthonormal control basis."""

    if count <= 0 or count > dimension:
        raise ValueError("count must be in [1, dimension]")
    rng = np.random.default_rng(stable_seed("orthogonal", seed, dimension, count))
    matrix = rng.normal(size=(dimension, count))
    q, _ = np.linalg.qr(matrix)
    return [
        with_computed_hash(
            SteeringVector(
                values=q[:, index],
                layer=layer,
                constructor="orthogonal_random",
                normalization="unit",
                metadata={"creation_seed": seed, "direction_index": index},
            )
        )
        for index in range(count)
    ]


def difference_of_means(
    backend: ModelBackend,
    positive_items: Sequence[BenchmarkItem],
    negative_items: Sequence[BenchmarkItem],
    layer: int,
    normalize: bool = True,
    metadata: dict[str, object] | None = None,
    seed: int | None = None,
) -> SteeringVector:
    """Use ``mean(h_positive) - mean(h_negative)`` as a transparent constructor.

    The extraction policy belongs to the backend and is recorded in metadata by
    callers. This is an initial development constructor, not a final method.
    """

    if not positive_items or not negative_items:
        raise ValueError("Both positive_items and negative_items are required")
    positive = np.stack([np.asarray(backend.extract_activation(item)) for item in positive_items])
    negative = np.stack([np.asarray(backend.extract_activation(item)) for item in negative_items])
    if positive.shape[1] != backend.hidden_size or negative.shape[1] != backend.hidden_size:
        raise ValueError("Extracted activations do not match backend hidden size")
    values, normalization = _normalize(positive.mean(axis=0) - negative.mean(axis=0), normalize)
    backend_provenance = (
        backend.provenance() if hasattr(backend, "provenance") else {"backend_type": "unknown"}
    )
    vector = SteeringVector(
        values=values,
        layer=layer,
        constructor="difference_of_means",
        normalization=normalization,
        metadata={
            "positive_items": [item.id for item in positive_items],
            "negative_items": [item.id for item in negative_items],
            "creation_seed": seed,
            "model_provenance": backend_provenance,
            "extraction_policy": (
                "backend-defined; mock uses latent representation, HF uses "
                "last non-padding token"
            ),
            **(metadata or {}),
        },
    )
    return with_computed_hash(vector)
