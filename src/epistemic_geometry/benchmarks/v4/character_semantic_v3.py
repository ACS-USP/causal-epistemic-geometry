"""Typed character-count evaluator built on frozen semantic-v3 commitments."""

from __future__ import annotations

from epistemic_geometry.benchmarks.external.semantic_v3 import (
    SemanticV3Result,
    evaluate_external_answer_v3,
)

PARSER_VERSION = "character-count-semantic-v3"


def evaluate_character_count_answer_v3(
    raw_text: str,
    reference_answer: str,
    *,
    truncated: bool = False,
    runtime_error: bool = False,
) -> SemanticV3Result:
    """Evaluate one explicit final commitment against an integer oracle."""

    try:
        int(reference_answer)
    except ValueError as exc:
        raise ValueError("character-count reference must be an integer") from exc
    return evaluate_external_answer_v3(
        raw_text,
        reference_answer,
        truncated=truncated,
        runtime_error=runtime_error,
    )


__all__ = ["PARSER_VERSION", "evaluate_character_count_answer_v3"]
