"""Deterministic type-aware diagnostic for the frozen CRUXEval smoke."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Any


def _literal(value: str) -> Any:
    return ast.literal_eval(value.strip())


def _string_content(value: str) -> str:
    stripped = value.strip()
    try:
        parsed = _literal(stripped)
    except (SyntaxError, ValueError):
        return " ".join(stripped.split())
    return parsed if isinstance(parsed, str) else " ".join(stripped.split())


def type_aware_equal(actual: str, reference: str) -> tuple[bool, str]:
    """Compare a predicted Python output with a reference without executing code."""

    reference_value = _literal(reference)
    if isinstance(reference_value, str):
        return _string_content(actual) == reference_value, "string_content"
    try:
        actual_value = _literal(actual)
    except (SyntaxError, ValueError, TypeError):
        return False, f"unparseable_{type(reference_value).__name__}"
    return actual_value == reference_value, f"literal_{type(reference_value).__name__}"


@dataclass(frozen=True)
class SemanticPostmortem:
    original_status: str
    diagnostic_status: str
    semantic_equal: bool | None
    comparison_mode: str
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "original_status": self.original_status,
            "diagnostic_status": self.diagnostic_status,
            "semantic_equal": self.semantic_equal,
            "comparison_mode": self.comparison_mode,
            "reason": self.reason,
        }


def classify_postmortem(
    *, original_status: str, parsed_answer: str | None, reference_answer: str
) -> SemanticPostmortem:
    """Classify format failures without changing the original result."""

    if original_status == "VALID_CORRECT":
        return SemanticPostmortem(original_status, "ORIGINAL_VALID_CORRECT", True, "original", "")
    if original_status == "VALID_WRONG":
        return SemanticPostmortem(original_status, "ORIGINAL_VALID_WRONG", False, "original", "")
    if parsed_answer is None:
        return SemanticPostmortem(
            original_status, "UNASSESSABLE", None, "none", "no parsed answer retained"
        )
    try:
        equal, mode = type_aware_equal(parsed_answer, reference_answer)
    except (SyntaxError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return SemanticPostmortem(
            original_status, "UNASSESSABLE", None, "parse_error", type(exc).__name__
        )
    return SemanticPostmortem(
        original_status,
        "SEMANTIC_CORRECT_FORMAT_ERROR" if equal else "SEMANTIC_WRONG",
        equal,
        mode,
        "deterministic type-aware comparison",
    )
