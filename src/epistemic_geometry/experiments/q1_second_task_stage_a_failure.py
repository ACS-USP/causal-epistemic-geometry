"""Outcome-blind rules for the post-closeout Stage-A instrument audit.

The rules in this module were frozen before the sealed Stage-A journal was
opened.  They are condition-symmetric and never use a reference value to pick
an answer candidate.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from epistemic_geometry.benchmarks.external.base import _visible_text
from epistemic_geometry.benchmarks.external.semantic_v3 import _marker
from epistemic_geometry.experiments.q1_second_task import (
    evaluate_livecodebench_output,
    parse_safe_literal,
)

TAXONOMY = (
    "VALID_AS_FROZEN",
    "MISSING_FINAL_MARKER_UNIQUE_LITERAL_PRESENT",
    "FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT",
    "UNIQUE_LITERAL_IN_CODE_BLOCK",
    "UNIQUE_LITERAL_WITH_TRAILING_PROSE",
    "MULTIPLE_IDENTICAL_COMMITMENTS",
    "MULTIPLE_CONTRADICTORY_COMMITMENTS",
    "MALFORMED_PYTHON_OR_JSON_LITERAL",
    "TYPE_MISMATCH",
    "NO_RECOVERABLE_LITERAL",
    "TRUNCATED_GENERATION",
    "MECHANICAL_REPETITION",
    "UNFINISHED_REASONING",
    "AMBIGUOUS_OUTPUT",
    "REFERENCE_OR_PROMPT_AMBIGUITY",
    "OTHER",
)

_FINAL_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?final(?:\s+answer)?\s*:\s*(.*?)\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)```", re.DOTALL)
_SUPPORTED_TAGS = {"bool", "int", "list", "str"}


@dataclass(frozen=True)
class LiteralCandidate:
    source: str
    payload: str
    canonical_json: str
    value_type: str


@dataclass(frozen=True)
class AuditClassification:
    category: str
    recoverable: bool
    candidate: LiteralCandidate | None
    requires_semantic_judgment: bool
    mechanically_repetitive: bool
    unfinished: bool


def _candidate(payload: str, source: str) -> LiteralCandidate | None:
    try:
        canonical = parse_safe_literal(payload.strip())
    except ValueError:
        return None
    if canonical[0] not in _SUPPORTED_TAGS:
        return None
    return LiteralCandidate(
        source=source,
        payload=payload.strip(),
        canonical_json=json.dumps(canonical, ensure_ascii=False, separators=(",", ":")),
        value_type=str(canonical[0]),
    )


def _final_candidates(raw: str) -> tuple[list[LiteralCandidate], int, bool, bool]:
    candidates: list[LiteralCandidate] = []
    marker_count = 0
    malformed = False
    marker_line_index: int | None = None
    lines = raw.splitlines()
    for line_index, line in enumerate(lines):
        match = _FINAL_LINE.fullmatch(line)
        if match is None:
            continue
        marker_count += 1
        marker_line_index = line_index
        value = _candidate(match.group(1), "FINAL_MARKER")
        if value is None:
            malformed = True
        else:
            candidates.append(value)
    has_trailing_lines = (
        marker_count == 1
        and marker_line_index is not None
        and any(line.strip() for line in lines[marker_line_index + 1 :])
    )
    return candidates, marker_count, malformed, has_trailing_lines


def _fenced_candidates(raw: str) -> list[LiteralCandidate]:
    values: list[LiteralCandidate] = []
    for match in _FENCE.finditer(raw):
        body = match.group("body").strip()
        marker = _FINAL_LINE.fullmatch(body)
        if marker is not None:
            body = marker.group(1)
        value = _candidate(body, "CODE_BLOCK")
        if value is not None:
            values.append(value)
    return values


def _standalone_candidates(raw: str) -> list[LiteralCandidate]:
    values: list[LiteralCandidate] = []
    for line in raw.splitlines():
        payload = line.strip()
        if not payload or _FINAL_LINE.fullmatch(line) or payload.startswith("```"):
            continue
        if payload.startswith("`") and payload.endswith("`") and len(payload) >= 2:
            payload = payload[1:-1].strip()
        value = _candidate(payload, "STANDALONE_LINE")
        if value is not None:
            values.append(value)
    return values


def _trailing_prose_candidate(raw: str) -> LiteralCandidate | None:
    values: list[LiteralCandidate] = []
    for line in raw.splitlines():
        match = _FINAL_LINE.fullmatch(line)
        if match is None:
            continue
        payload = match.group(1).strip()
        for boundary in range(1, len(payload)):
            if not payload[boundary].isspace():
                continue
            value = _candidate(payload[:boundary], "FINAL_LITERAL_PREFIX_WITH_TRAILING_PROSE")
            if value is not None and payload[boundary:].strip():
                values.append(value)
    unique = {value.canonical_json: value for value in values}
    return next(iter(unique.values())) if len(unique) == 1 else None


def mechanical_repetition(token_ids: Sequence[int]) -> bool:
    """Frozen structural repetition criterion independent of decoded text."""

    tokens = [int(value) for value in token_ids]
    if len(tokens) < 128:
        return False
    tail = tokens[-128:]
    if len(set(tail)) / len(tail) <= 0.20:
        return True
    for width in (1, 2, 4, 8):
        for end in range(width * 8, len(tokens) + 1):
            block = tokens[end - width : end]
            repeated_blocks = (
                tokens[end - width * repeat : end - width * (repeat - 1)]
                for repeat in range(1, 9)
            )
            if all(candidate == block for candidate in repeated_blocks):
                return True
    return False


def unfinished_reasoning(raw: str) -> bool:
    lowered = raw.lower()
    if "<think>" in lowered and "</think>" not in lowered:
        return True
    if raw.count("```") % 2:
        return True
    tail = raw.rstrip()
    return bool(tail) and tail.endswith((":", "...", "…"))


def _unique(values: Sequence[LiteralCandidate]) -> LiteralCandidate | None:
    unique = {value.canonical_json: value for value in values}
    return next(iter(unique.values())) if len(unique) == 1 else None


def _typed_category(
    candidate: LiteralCandidate,
    expected_type: str,
    otherwise: str,
) -> str:
    return otherwise if candidate.value_type == expected_type else "TYPE_MISMATCH"


def terminal_candidate_after_empty_final_headings(
    raw_output: str,
) -> LiteralCandidate | None:
    """Select one terminal literal without consulting reference type or value.

    This is the post-diagnosis candidate repair.  It fails closed if another
    distinct standalone/fenced typed literal or another inline commitment is
    visible anywhere in the response. Reference typing and exact comparison
    happen only after this purely mechanical selection step.
    """

    visible, unclosed_thinking = _visible_text(raw_output)
    if unclosed_thinking:
        return None
    lines = visible.splitlines()
    markers = [
        (index, marker)
        for index, line in enumerate(lines)
        if (marker := _marker(line)) is not None
    ]
    if len(markers) < 2:
        return None
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty_indices:
        return None
    terminal_index, terminal_marker = markers[-1]
    terminal_kind, terminal_payload, terminal_closer = terminal_marker
    if (
        terminal_index != nonempty_indices[-1]
        or terminal_kind != "FINAL"
        or terminal_closer is not None
        or not terminal_payload.strip()
    ):
        return None
    terminal = _candidate(
        terminal_payload, "TERMINAL_FINAL_AFTER_NONLITERAL_FINAL_HEADINGS"
    )
    if terminal is None:
        return None
    for _index, marker in markers[:-1]:
        kind, payload, _closer = marker
        if kind not in {"FINAL_ANSWER", "FINAL_SECTION"} or payload.strip():
            return None
    outside = [*_fenced_candidates(visible), *_standalone_candidates(visible)]
    if any(value.canonical_json != terminal.canonical_json for value in outside):
        return None
    return terminal


def classify_output(
    raw: str,
    token_ids: Sequence[int],
    *,
    truncated: bool,
    frozen_commitment_valid: bool,
    frozen_evaluable: bool,
    frozen_value_type: str | None,
    expected_type: str,
) -> AuditClassification:
    repeated = mechanical_repetition(token_ids)
    unfinished = unfinished_reasoning(raw)
    if truncated:
        return AuditClassification("TRUNCATED_GENERATION", False, None, False, repeated, unfinished)
    if repeated and not frozen_evaluable:
        return AuditClassification("MECHANICAL_REPETITION", False, None, False, True, unfinished)
    if frozen_evaluable:
        category = "VALID_AS_FROZEN" if frozen_value_type == expected_type else "TYPE_MISMATCH"
        return AuditClassification(category, True, None, False, repeated, unfinished)
    visible, unclosed_thinking = _visible_text(raw)
    if unclosed_thinking:
        return AuditClassification("UNFINISHED_REASONING", False, None, False, repeated, True)
    final_values, marker_count, malformed, has_trailing_lines = _final_candidates(visible)
    if marker_count > 1:
        terminal = terminal_candidate_after_empty_final_headings(raw)
        if terminal is not None:
            return AuditClassification("OTHER", True, terminal, False, repeated, unfinished)
        unique = _unique(final_values)
        if not malformed and unique is not None:
            return AuditClassification(
                _typed_category(unique, expected_type, "MULTIPLE_IDENTICAL_COMMITMENTS"),
                True,
                unique,
                False,
                repeated,
                unfinished,
            )
        return AuditClassification(
            "MULTIPLE_CONTRADICTORY_COMMITMENTS", False, None, True, repeated, unfinished
        )
    if marker_count == 1:
        unique = _unique(final_values)
        if unique is not None and has_trailing_lines:
            return AuditClassification(
                _typed_category(unique, expected_type, "UNIQUE_LITERAL_WITH_TRAILING_PROSE"),
                True,
                unique,
                False,
                repeated,
                unfinished,
            )
        if unique is not None:
            return AuditClassification(
                _typed_category(unique, expected_type, "FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT"),
                True,
                unique,
                False,
                repeated,
                unfinished,
            )
        trailing = _trailing_prose_candidate(visible)
        if trailing is not None:
            return AuditClassification(
                _typed_category(trailing, expected_type, "UNIQUE_LITERAL_WITH_TRAILING_PROSE"),
                True,
                trailing,
                False,
                repeated,
                unfinished,
            )
        if malformed or frozen_commitment_valid:
            return AuditClassification(
                "MALFORMED_PYTHON_OR_JSON_LITERAL", False, None, False, repeated, unfinished
            )
    fenced = _fenced_candidates(visible)
    standalone = _standalone_candidates(visible)
    all_values = [*fenced, *standalone]
    unique_all = _unique(all_values)
    if unique_all is not None:
        source = (
            "UNIQUE_LITERAL_IN_CODE_BLOCK"
            if any(value.canonical_json == unique_all.canonical_json for value in fenced)
            else "MISSING_FINAL_MARKER_UNIQUE_LITERAL_PRESENT"
        )
        return AuditClassification(
            _typed_category(unique_all, expected_type, source),
            True,
            unique_all,
            False,
            repeated,
            unfinished,
        )
    if len({value.canonical_json for value in all_values}) > 1:
        return AuditClassification("AMBIGUOUS_OUTPUT", False, None, True, repeated, unfinished)
    if unfinished:
        return AuditClassification("UNFINISHED_REASONING", False, None, False, repeated, True)
    return AuditClassification("NO_RECOVERABLE_LITERAL", False, None, False, repeated, unfinished)


def conservative_repair_candidate(classification: AuditClassification) -> LiteralCandidate | None:
    """Candidate parser A: fail closed and exclude trailing-prose recovery."""

    allowed = {
        "FINAL_MARKER_CASE_OR_WHITESPACE_VARIANT",
        "UNIQUE_LITERAL_IN_CODE_BLOCK",
        "MISSING_FINAL_MARKER_UNIQUE_LITERAL_PRESENT",
        "MULTIPLE_IDENTICAL_COMMITMENTS",
    }
    return classification.candidate if classification.category in allowed else None


def terminal_contract_repair_candidate(
    classification: AuditClassification,
) -> LiteralCandidate | None:
    """Post-diagnosis Repair A2, additive to the original conservative repair."""

    candidate = classification.candidate
    if (
        classification.category == "OTHER"
        and candidate is not None
        and candidate.source == "TERMINAL_FINAL_AFTER_NONLITERAL_FINAL_HEADINGS"
    ):
        return candidate
    return None


def evaluate_livecodebench_output_stage_a2(
    raw_output: str,
    reference_json: str,
    token_ids: Sequence[int],
    *,
    truncated: bool = False,
    runtime_error: bool = False,
) -> dict[str, Any]:
    """Prospective Stage-A2 evaluator with the locked parser-only extension."""

    frozen = evaluate_livecodebench_output(
        raw_output,
        reference_json,
        truncated=truncated,
        runtime_error=runtime_error,
    )
    base = {
        **frozen,
        "parser_repair_applied": False,
        "parser_repair": None,
        "frozen_status_before_repair": frozen["status"],
    }
    if runtime_error or frozen["semantic_evaluable"]:
        return base
    expected = parse_safe_literal(reference_json)
    expected_type = str(expected[0])
    classification = classify_output(
        raw_output,
        token_ids,
        truncated=truncated,
        frozen_commitment_valid=bool(frozen["commitment_valid"]),
        frozen_evaluable=bool(frozen["semantic_evaluable"]),
        frozen_value_type=None,
        expected_type=expected_type,
    )
    candidate = terminal_contract_repair_candidate(classification)
    if candidate is None:
        return base
    if candidate.value_type != expected_type:
        return base
    correct = candidate.canonical_json == json.dumps(
        expected, ensure_ascii=False, separators=(",", ":")
    )
    return {
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": correct,
        "canonical_value": candidate.canonical_json,
        "failure_reason": None,
        "parser_repair_applied": True,
        "parser_repair": "TERMINAL_TYPED_FINAL_AFTER_EMPTY_NONLITERAL_FINAL_HEADINGS_V1",
        "frozen_status_before_repair": frozen["status"],
    }


__all__ = [
    "AuditClassification",
    "LiteralCandidate",
    "TAXONOMY",
    "classify_output",
    "conservative_repair_candidate",
    "evaluate_livecodebench_output_stage_a2",
    "mechanical_repetition",
    "terminal_candidate_after_empty_final_headings",
    "terminal_contract_repair_candidate",
    "unfinished_reasoning",
]
