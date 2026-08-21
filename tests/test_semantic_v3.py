from __future__ import annotations

import json

import pytest

from epistemic_geometry.benchmarks.external.semantic_v3 import (
    canonicalize_semantic_value,
    evaluate_external_answer_v3,
    extract_final_commitment,
)


@pytest.mark.parametrize(
    ("raw", "payload"),
    [
        ("FINAL: 2", "2"),
        ("**FINAL: 2**", "2"),
        ("### FINAL: 2", "2"),
        ("### Final Answer:\n2", "2"),
        ("### Final Answer:\n```\n2\n```", "2"),
        ("### Final Answer:\n```\nFINAL: 2\n```", "2"),
        ("### Final:\n`2`", "2"),
        ("```python\nFINAL: 2\n```", "2"),
        ("FINAL: `2`", "2"),
    ],
)
def test_supported_final_commitment_forms(raw: str, payload: str) -> None:
    result = extract_final_commitment(raw)
    assert result.valid
    assert result.payload == payload


def test_multiline_payload_preserves_newlines_and_leading_spaces() -> None:
    result = extract_final_commitment("### Final Answer:\n```\nfirst\n  second\nthird\n```")
    assert result.valid
    assert result.payload == "first\n  second\nthird"


def test_nested_and_indented_multiline_final_payloads() -> None:
    fenced = extract_final_commitment(
        "### Final Answer:\n```\nFINAL:   a  \n  bc \n     \n  d  \n```"
    )
    indented = extract_final_commitment("FINAL:  a  \n  bc \n     \n  d  ")
    assert fenced.valid and fenced.payload == "  a  \n  bc \n     \n  d  "
    assert indented.valid and indented.payload == " a  \n  bc \n     \n  d  "


@pytest.mark.parametrize(
    ("payload", "tag"),
    [
        ("None", "none"),
        ("True", "bool"),
        ("1", "int"),
        ("1.5", "float"),
        ("'text'", "str"),
        ("b'abc'", "bytes"),
        ("bytearray(b'abc')", "bytearray"),
        ("[1, 2]", "list"),
        ("(1, 2)", "tuple"),
        ("{'a': 1}", "dict"),
        ("{1, 2}", "set"),
        ("frozenset({1, 2})", "frozenset"),
        ("bare string", "str"),
    ],
)
def test_tagged_canonical_values(payload: str, tag: str) -> None:
    assert canonicalize_semantic_value(payload)[0] == tag


def test_wrong_type_commitment_is_evaluable_and_wrong() -> None:
    result = evaluate_external_answer_v3("FINAL: IndexError", "[1, 2, 3]")
    assert result.commitment_valid
    assert result.semantic_evaluable
    assert result.value_type == "str"
    assert not result.correct


def test_bytes_literal_and_bare_expected_string_are_correct() -> None:
    assert evaluate_external_answer_v3("FINAL: b'abc'", "b'abc'").correct
    assert evaluate_external_answer_v3("FINAL: Name unknown", "'Name unknown'").correct


def test_multiline_expected_string_compares_exactly() -> None:
    raw = "### Final Answer:\n```\nfirst\n  second\n```"
    assert evaluate_external_answer_v3(raw, repr("first\n  second")).correct


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ("reason only", "no final commitment"),
        ("FINAL:", "empty final commitment"),
        ("FINAL: 1\nprose", "substantive content follows final commitment"),
        ("FINAL: 1\nFINAL: 2", "multiple final commitments"),
        ("### Final Answer:\n```\nvalue", "unmatched final fence"),
        ("```\nFINAL: 1", "unmatched final fence or emphasis wrapper"),
    ],
)
def test_invalid_or_ambiguous_commitments(raw: str, reason: str) -> None:
    result = extract_final_commitment(raw)
    assert not result.valid
    assert result.failure_reason == reason


def test_truncated_and_unclosed_thinking_are_not_commitments() -> None:
    assert not extract_final_commitment("FINAL: 1", truncated=True).valid
    assert not extract_final_commitment("<think>unfinished\nFINAL: 1").valid


def test_malicious_expression_is_never_executed_and_becomes_raw_string() -> None:
    payload = "__import__('os').system('false')"
    canonical = canonicalize_semantic_value(payload)
    assert canonical == ["str", payload]
    result = evaluate_external_answer_v3(f"FINAL: {payload}", "None")
    assert result.semantic_evaluable
    assert not result.correct


def test_parser_is_invariant_to_external_condition_labels() -> None:
    rows = [
        {"condition": "BASELINE", "raw_output": "FINAL: [1, 2]"},
        {"condition": "BEST_SINGLE_MEAN_PLUS", "raw_output": "FINAL: [1, 2]"},
    ]
    results = [evaluate_external_answer_v3(row["raw_output"], "[1, 2]") for row in rows]
    assert results[0] == results[1]
    assert json.loads(results[0].canonical_value or "null")[0] == "list"


@pytest.mark.parametrize(
    ("pattern", "raw", "reference", "truncated", "commitment", "correct"),
    [
        (
            "v2_invalid_to_correct_wrapper",
            "### FINAL: Name unknown",
            "'Name unknown'",
            False,
            True,
            True,
        ),
        (
            "v2_invalid_to_wrong_fence",
            "```python\nFINAL: b'bad'\n```",
            "b'good'",
            False,
            True,
            False,
        ),
        ("v2_truncated_status", "FINAL: 2", "2", True, False, False),
        ("v2_wrong_to_correct_bare_string", "FINAL: yes", "'yes'", False, True, True),
        ("v2_wrong_payload_changed", "FINAL: **wrong**", "'right'", False, True, False),
    ],
)
def test_original_v2_disagreement_pattern_regressions(
    pattern: str,
    raw: str,
    reference: str,
    truncated: bool,
    commitment: bool,
    correct: bool,
) -> None:
    del pattern
    result = evaluate_external_answer_v3(raw, reference, truncated=truncated)
    assert result.commitment_valid is commitment
    assert result.correct is correct
