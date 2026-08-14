import numpy as np

from epistemic_geometry.steering.constructors import (
    orthogonal_random_directions,
    random_unit_vector,
)
from epistemic_geometry.steering.vector import load_vector, save_vector, vector_hash


def test_random_vector_normalization_and_hash_stability() -> None:
    vector = random_unit_vector(8, seed=4, layer=2)
    assert np.isclose(np.linalg.norm(vector.values), 1.0)
    assert vector.hash == vector_hash(vector.values)
    assert vector.hash == random_unit_vector(8, seed=4, layer=2).hash


def test_orthogonal_directions_are_orthonormal() -> None:
    vectors = orthogonal_random_directions(6, 3, seed=4)
    matrix = np.stack([vector.values for vector in vectors])
    assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-10)


def test_vector_serialization_roundtrip(tmp_path) -> None:
    vector = random_unit_vector(5, seed=17, metadata={"purpose": "test"})
    vector_path, metadata_path = save_vector(vector, tmp_path / "vector.npz")
    restored = load_vector(vector_path, metadata_path)
    assert np.array_equal(restored.values, vector.values)
    assert restored.hash == vector.hash
    assert restored.metadata["purpose"] == "test"

