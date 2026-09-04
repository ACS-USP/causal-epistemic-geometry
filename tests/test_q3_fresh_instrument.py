from __future__ import annotations

import ast

import pytest

from epistemic_geometry.benchmarks.q3_fresh.instrument import (
    ARCHETYPES,
    OUTPUT_TYPES,
    build_family,
    canonical_skeleton,
    custom_reference,
    sandboxed_cpython_reference,
    validate_family,
    validate_restricted_source,
)


@pytest.mark.parametrize("candidate_index", range(48))
def test_dual_references_agree_across_archetypes_and_types(candidate_index: int) -> None:
    family = build_family("fixture", candidate_index, 12345)
    report = validate_family(family)
    assert report["dual_evaluator_agreement"] is True
    assert report["reference_repeat_determinism"] is True
    assert report["parser_reference_roundtrip"] is True


def test_candidate_stream_is_deterministic_and_namespaced() -> None:
    first = build_family("fixture", 7, 91)
    repeated = build_family("fixture", 7, 91)
    other = build_family("other", 7, 91)
    assert first == repeated
    assert first.canonical_skeleton != other.canonical_skeleton
    assert first.family_id != other.family_id


def test_family_identity_ignores_literal_values_but_not_structure() -> None:
    family = build_family("fixture", 2, 222)
    operations = tuple({**op, "a": int(op["a"]) + 100} for op in family.operations)
    assert canonical_skeleton(operations, family.output_type) == family.canonical_skeleton
    changed = list(operations)
    changed[0] = {**changed[0], "variant": (int(changed[0]["variant"]) + 1) % 4}
    assert canonical_skeleton(tuple(changed), family.output_type) != family.canonical_skeleton


def test_output_types_and_archetypes_are_covered() -> None:
    families = [build_family("fixture", index, 17) for index in range(192)]
    assert {family.archetype for family in families} == set(ARCHETYPES)
    assert {family.output_type for family in families} == set(OUTPUT_TYPES)


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef solve(data):\n return 1\n",
        "def solve(data):\n return open('/tmp/x')\n",
        "def solve(data):\n return data.__class__\n",
        "def solve(data):\n return eval('1')\n",
    ],
)
def test_restricted_ast_fails_closed(source: str) -> None:
    with pytest.raises(ValueError):
        validate_restricted_source(source)


def test_bool_is_not_integer_in_typed_contract() -> None:
    family = build_family("fixture", 32, 83)
    assert family.output_type == "bool"
    assert family.reference_type == "bool"
    assert type(ast.literal_eval(family.reference_repr)) is bool


def test_mutation_aliasing_is_exercised() -> None:
    family = next(
        build_family("fixture", index, 44)
        for index in range(64)
        if build_family("fixture", index, 44).archetype == "SEQUENCE_ALIASING"
    )
    assert any(op["kind"] == "MUTATE" for op in family.operations)
    expected = custom_reference(family.operations, family.output_type, family.input_value)
    observed_type, observed_repr = sandboxed_cpython_reference(family.source, family.input_value)
    assert observed_type == type(expected).__name__
    assert ast.literal_eval(observed_repr) == expected


def test_worker_cannot_execute_unvalidated_payload() -> None:
    with pytest.raises(ValueError):
        sandboxed_cpython_reference("def solve(data):\n return __import__('os').listdir('/')\n", {})
