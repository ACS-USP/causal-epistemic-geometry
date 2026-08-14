import math

import numpy as np

from epistemic_geometry.benchmarks.base import AnswerParser
from epistemic_geometry.steering.geometry import (
    cosine_similarity,
    match_norm,
    normalized_euclidean_distance,
)
from epistemic_geometry.steering.vector import with_computed_hash
from epistemic_geometry.types import SteeringVector


def test_parser_distinguishes_exact_ambiguous_and_invalid_outputs() -> None:
    parser = AnswerParser({"A", "B", "C", "D"})
    assert parser.parse("").status == "EMPTY"
    assert parser.parse("A").status == "OK"
    assert parser.parse("A.").status == "OK"
    assert parser.parse("A because").status == "AMBIGUOUS"
    assert parser.parse("E").status == "INVALID"


def test_q2_geometry_helpers_are_pure_and_zero_safe() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert np.isclose(
        normalized_euclidean_distance(np.array([2.0, 0.0]), np.array([0.0, 3.0])),
        2**0.5,
    )
    assert math.isnan(cosine_similarity(np.zeros(2), np.ones(2)))
    reference = with_computed_hash(SteeringVector(np.array([3.0, 4.0]), 0, "ref", "none"))
    direction = with_computed_hash(SteeringVector(np.array([1.0, 0.0]), 0, "direction", "unit"))
    matched = match_norm(reference, direction)
    assert np.isclose(np.linalg.norm(matched.values), 5.0)
    assert matched.metadata["reference_vector_hash"] == reference.hash
