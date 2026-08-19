from __future__ import annotations

from epistemic_geometry.benchmarks.dense_code import (
    TestCaseOutcome as Outcome,
)
from epistemic_geometry.benchmarks.dense_code import (
    TestCaseStatus as Status,
)
from epistemic_geometry.benchmarks.dense_code import (
    make_program_outcome,
)
from epistemic_geometry.benchmarks.v4.character_count import (
    STRATA,
    generate_character_count_manifest,
)
from epistemic_geometry.benchmarks.v4.character_parser import parse_final_integer
from epistemic_geometry.benchmarks.v4.geometry import (
    conceptual_distance,
    generate_geometry_manifest,
)
from epistemic_geometry.benchmarks.v4.postmortem import classify_postmortem, type_aware_equal


def test_character_manifest_is_deterministic_and_exactly_balanced() -> None:
    first = generate_character_count_manifest(seed=123)
    second = generate_character_count_manifest(seed=123)
    assert first == second
    assert len(first["items"]) == 30
    assert {item["stratum"] for item in first["items"]} == set(STRATA)
    assert all(
        item["text"].count(item["target_character"]) == item["answer"]
        for item in first["items"]
    )
    assert len({item["item_hash"] for item in first["items"]}) == 30


def test_geometry_manifest_and_distances() -> None:
    manifest = generate_geometry_manifest()
    assert len(manifest["items"]) == 94
    assert sum(item["domain"] == "WEEKDAYS" for item in manifest["items"]) == 49
    assert sum(item["domain"] == "LETTERS" for item in manifest["items"]) == 45
    assert conceptual_distance("WEEKDAYS", 0, 6) == 1
    assert conceptual_distance("LETTERS", 2, 25) == 23


def test_type_aware_postmortem_distinguishes_string_format() -> None:
    assert type_aware_equal("hello", "'hello'") == (True, "string_content")
    assert type_aware_equal("[1, 2]", "[1,2]") == (True, "literal_list")
    assert type_aware_equal("[1, 3]", "[1,2]")[0] is False
    diagnostic = classify_postmortem(
        original_status="INVALID_FORMAT",
        parsed_answer="hello",
        reference_answer="'hello'",
    )
    assert diagnostic.diagnostic_status == "SEMANTIC_CORRECT_FORMAT_ERROR"


def test_character_parser_accepts_only_explicit_final_variants() -> None:
    assert parse_final_integer("**FINAL: 5**") == ("PARSED", 5, None)
    assert parse_final_integer("`FINAL: -2`") == ("PARSED", -2, None)
    assert parse_final_integer("### Final Answer: **7**") == ("PARSED", 7, None)
    assert parse_final_integer("### Final Answer:\n**7**") == ("PARSED", 7, None)
    assert parse_final_integer("### ✅ FINAL: 2") == ("PARSED", 2, None)
    assert parse_final_integer("FINAL: 3\n") == ("PARSED", 3, None)
    assert parse_final_integer("The answer is 3") [0] == "INVALID_FORMAT"
    assert parse_final_integer("FINAL: 3\nFINAL: 4")[0] == "INVALID_FORMAT"
    assert parse_final_integer("FINAL: five")[0] == "INVALID_FORMAT"
    assert parse_final_integer("<think>unfinished", truncated=False)[0] == "TRUNCATED_THINKING"


def test_dense_code_vector_keeps_nested_test_identity() -> None:
    outcome = make_program_outcome(
        "problem-1",
        (
            Outcome("problem-1", "case-1", Status.PASS),
            Outcome("problem-1", "case-2", Status.FAIL),
            Outcome("problem-1", "case-3", Status.RUNTIME_ERROR),
        ),
    )
    assert outcome.failure_vector() == (0, 1, 1)
    assert outcome.summary()["n_tests"] == 3
