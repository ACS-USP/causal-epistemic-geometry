from __future__ import annotations

from epistemic_geometry.benchmarks.external.base import ExternalStatus
from epistemic_geometry.benchmarks.external.semantic_v2 import (
    PARSER_VERSION,
    parse_external_answer_v2,
)


def _answer(raw: str, *, truncated: bool = False) -> str | None:
    parsed = parse_external_answer_v2(raw, truncated=truncated)
    assert parsed.status is None, parsed
    return parsed.answer_text


def test_parser_version_is_explicit() -> None:
    assert PARSER_VERSION == "external-semantic-v2"


def test_naked_heading_checklist_and_wrapped_final_commitments() -> None:
    assert _answer("FINAL: 2") == "2"
    assert _answer("### FINAL: 2") == "2"
    assert _answer("- FINAL: 2") == "2"
    assert _answer("**FINAL: 2**") == "2"
    assert _answer("FINAL: `2`") == "2"


def test_fenced_and_python_fenced_final_commitments() -> None:
    assert _answer("### ✅ Final Answer:\n```\nFINAL: 2\n```") == "2"
    assert _answer("```python\nFINAL: [1, 2]\n```") == "[1, 2]"


def test_closing_fence_is_allowed_but_prose_after_it_is_not() -> None:
    assert _answer("```\nFINAL: 2\n```") == "2"
    parsed = parse_external_answer_v2("```\nFINAL: 2\n```\nDone.")
    assert parsed.status is ExternalStatus.INVALID_FORMAT
    assert parsed.parse_reason == "substantive content follows FINAL commitment"


def test_matching_standalone_emphasis_delimiter_is_allowed() -> None:
    assert _answer("FINAL: 2\n**") == "2"


def test_multiple_conflicting_or_missing_commitments_are_invalid() -> None:
    for raw in (
        "FINAL: 1\nFINAL: 2",
        "FINAL: 1\nSome prose",
        "The answer is 2.",
        "FINAL:",
        "**FINAL: 2",
        "FINAL: 2**",
        "FINAL: 2`",
    ):
        parsed = parse_external_answer_v2(raw)
        assert parsed.status is ExternalStatus.INVALID_FORMAT, raw


def test_structural_and_string_semantics_are_left_to_typed_evaluator() -> None:
    assert _answer("FINAL: [1, 2, 3]") == "[1, 2, 3]"
    assert _answer("FINAL: yes") == "yes"
    assert _answer("FINAL: [1, 2") == "[1, 2"


def test_truncated_output_is_never_repaired_offline() -> None:
    parsed = parse_external_answer_v2("FINAL: 2", truncated=True)
    assert parsed.status is ExternalStatus.TRUNCATED_THINKING


def test_unclosed_thinking_is_truncated() -> None:
    parsed = parse_external_answer_v2("<think>work\nFINAL: 2")
    assert parsed.status is ExternalStatus.TRUNCATED_THINKING
