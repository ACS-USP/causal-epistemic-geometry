from __future__ import annotations

import pytest

from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (
    classify_output,
    conservative_repair_candidate,
    mechanical_repetition,
)


def classify(
    raw: str,
    *,
    valid: bool = False,
    evaluable: bool = False,
    value_type: str | None = None,
    expected: str = "int",
    truncated: bool = False,
):
    return classify_output(
        raw,
        list(range(40)),
        truncated=truncated,
        frozen_commitment_valid=valid,
        frozen_evaluable=evaluable,
        frozen_value_type=value_type,
        expected_type=expected,
    )


@pytest.mark.parametrize(
    ("raw", "expected_type"),
    [
        ("7", "int"),
        ("[1, 2]", "list"),
        ("True", "bool"),
        ("'answer'", "str"),
    ],
)
def test_unique_terminal_literals_are_mechanically_recoverable(
    raw: str, expected_type: str
) -> None:
    result = classify(raw, expected=expected_type)
    assert result.category == "MISSING_FINAL_MARKER_UNIQUE_LITERAL_PRESENT"
    assert conservative_repair_candidate(result) is not None


def test_case_and_whitespace_final_variant_is_recoverable() -> None:
    result = classify("final :   [1, 2]", expected="list")
    assert result.category == "FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT"
    assert result.recoverable


def test_hidden_thinking_markers_are_not_counted_as_visible_commitments() -> None:
    result = classify("<think>\nFINAL: 7\n</think>\nfinal: 8")
    assert result.category == "FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT"
    assert result.candidate is not None
    assert result.candidate.canonical_json == '[\"int\",8]'


def test_unclosed_thinking_is_unfinished_not_recoverable() -> None:
    result = classify("<think>\nFINAL: 7")
    assert result.category == "UNFINISHED_REASONING"
    assert conservative_repair_candidate(result) is None


def test_unique_code_block_literal_is_recoverable() -> None:
    result = classify("The answer is below.\n```python\n[1, 2]\n```", expected="list")
    assert result.category == "UNIQUE_LITERAL_IN_CODE_BLOCK"
    assert result.recoverable


def test_trailing_prose_is_diagnostic_but_excluded_from_conservative_repair() -> None:
    for raw in ("FINAL: 7 because that is the output", "FINAL: 7\nThat is the output."):
        result = classify(raw)
        assert result.category == "UNIQUE_LITERAL_WITH_TRAILING_PROSE"
        assert conservative_repair_candidate(result) is None


def test_identical_multiple_finals_are_recoverable() -> None:
    result = classify("FINAL: 7\nFINAL: 7")
    assert result.category == "MULTIPLE_IDENTICAL_COMMITMENTS"
    assert result.recoverable


@pytest.mark.parametrize(
    "raw",
    [
        "FINAL: 7\nFINAL: 8",
        "The reasoning mentions\n7\nthen concludes\n8",
        "```\n7\n```\n```\n8\n```",
    ],
)
def test_contradictory_or_multiple_literals_fail_closed(raw: str) -> None:
    result = classify(raw)
    assert result.category in {"MULTIPLE_CONTRADICTORY_COMMITMENTS", "AMBIGUOUS_OUTPUT"}
    assert conservative_repair_candidate(result) is None


@pytest.mark.parametrize(
    "raw",
    [
        "FINAL: [1,",
        "FINAL: __import__('os').system('id')",
        "FINAL: {unclosed",
    ],
)
def test_malformed_and_malicious_text_fail_closed(raw: str) -> None:
    result = classify(raw, valid=True)
    assert result.category == "MALFORMED_PYTHON_OR_JSON_LITERAL"
    assert conservative_repair_candidate(result) is None


def test_quoted_and_unquoted_strings_remain_distinct() -> None:
    quoted = classify("'seven'", expected="str")
    unquoted = classify("seven", expected="str")
    assert quoted.recoverable
    assert not unquoted.recoverable


def test_nested_container_is_supported_but_bool_int_type_mismatch_is_visible() -> None:
    nested = classify("[1, [2, True]]", expected="list")
    mismatch = classify("FINAL: True", valid=True, evaluable=True, value_type="bool")
    assert nested.recoverable
    assert mismatch.category == "TYPE_MISMATCH"


def test_recovered_candidate_with_wrong_reference_type_is_not_parser_eligible() -> None:
    result = classify("final: True", expected="int")
    assert result.category == "TYPE_MISMATCH"
    assert result.candidate is not None
    assert conservative_repair_candidate(result) is None


def test_truncation_always_remains_nonrecoverable() -> None:
    result = classify("FINAL: 7", truncated=True)
    assert result.category == "TRUNCATED_GENERATION"
    assert not result.recoverable


def test_no_answer_is_unrecoverable() -> None:
    result = classify("I cannot determine the result.")
    assert result.category == "NO_RECOVERABLE_LITERAL"


def test_structural_repetition_criterion() -> None:
    assert mechanical_repetition([1, 2, 3, 4] * 40)
    assert not mechanical_repetition(list(range(200)))
