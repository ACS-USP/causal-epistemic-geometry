"""Condition-blind semantic commitment parsing for the Gate 6.3 audit.

``external-semantic-v3`` is additive: it does not modify the historical V1/V2
parsers.  It separates the existence of one final commitment, deterministic
semantic evaluability, and correctness.  The implementation never executes a
model payload and never receives item IDs or condition labels.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from .base import _visible_text

PARSER_VERSION = "external-semantic-v3"

_DECORATION_PREFIXES = (
    re.compile(r"^#{1,6}[ \t]+"),
    re.compile(r"^[-*][ \t]+"),
    re.compile(r"^[✅☑✔][ \t]*"),
)
_FINAL = re.compile(r"^FINAL[ \t]*:[ \t]?(?P<payload>.*)$", re.IGNORECASE)
_FINAL_ANSWER = re.compile(
    r"^FINAL[ \t]+ANSWER[ \t]*:[ \t]?(?P<payload>.*)$", re.IGNORECASE
)
_FENCE = re.compile(r"^(?P<marker>`{3,}|~{3,})(?P<label>[^`]*)$")
_BYTEARRAY = re.compile(r"^bytearray\((?P<inner>.*)\)$", re.DOTALL)
_FROZENSET = re.compile(r"^frozenset\((?P<inner>.*)\)$", re.DOTALL)


@dataclass(frozen=True)
class FinalCommitment:
    """The condition-independent result of final-section extraction."""

    valid: bool
    payload: str | None
    failure_reason: str | None


@dataclass(frozen=True)
class SemanticV3Result:
    """Three-axis semantic result for one preserved output."""

    commitment_valid: bool
    semantic_evaluable: bool
    value_type: str | None
    canonical_value: str | None
    correct: bool
    failure_reason: str | None
    payload: str | None


def _remove_prefixes(value: str) -> str:
    value = value.lstrip()
    changed = True
    while changed and value:
        changed = False
        for pattern in _DECORATION_PREFIXES:
            match = pattern.match(value)
            if match is not None:
                value = value[match.end() :].lstrip()
                changed = True
                break
    return value


def _unwrap_line_wrapper(value: str) -> tuple[str, str | None, str | None]:
    """Remove a matched whole-line wrapper or record a required later closer."""

    value = value.lstrip()
    for wrapper in ("**", "__", "`"):
        if not value.startswith(wrapper):
            continue
        value = value[len(wrapper) :]
        without_outer_space = value.rstrip()
        if without_outer_space.endswith(wrapper) and len(without_outer_space) >= len(wrapper):
            return without_outer_space[: -len(wrapper)], None, None
        return value.lstrip(), wrapper, None
    return value, None, None


def _unwrap_payload(
    value: str, *, preserve_whitespace: bool = False
) -> tuple[str | None, str | None]:
    """Remove only one matched Markdown wrapper around a payload."""

    original = value
    value = value.strip()
    for wrapper in ("**", "__", "`"):
        starts = value.startswith(wrapper)
        ends = value.endswith(wrapper)
        if starts and ends:
            if len(value) <= 2 * len(wrapper):
                return None, "empty final commitment"
            return value[len(wrapper) : -len(wrapper)], None
        # A Python dunder name is not an unmatched Markdown wrapper.
        if wrapper != "__" and (starts or ends):
            if not (starts and ends):
                return None, "unmatched payload wrapper"
    return (original if preserve_whitespace else value), None


def _marker(line: str) -> tuple[str, str, str | None] | None:
    """Return marker kind, inline payload, and optional standalone closer."""

    is_heading = bool(re.match(r"^\s*#{1,6}[ \t]+", line))
    normalized = _remove_prefixes(line)
    normalized, closer, _reason = _unwrap_line_wrapper(normalized)
    match = _FINAL.fullmatch(normalized)
    if match is not None:
        payload = match.group("payload")
        return ("FINAL_SECTION" if is_heading and not payload else "FINAL"), payload, closer
    match = _FINAL_ANSWER.fullmatch(normalized)
    if match is not None:
        return "FINAL_ANSWER", match.group("payload"), closer
    return None


def _fence_open(line: str) -> tuple[str, int] | None:
    match = _FENCE.fullmatch(line.strip())
    if match is None:
        return None
    marker = match.group("marker")
    return marker[0], len(marker)


def _is_fence_close(line: str, fence: tuple[str, int]) -> bool:
    marker, length = fence
    return bool(re.fullmatch(re.escape(marker) + rf"{{{length},}}", line.strip()))


def _active_fence_before(lines: list[str], index: int) -> tuple[str, int] | None:
    active: tuple[str, int] | None = None
    for line in lines[:index]:
        if active is None:
            active = _fence_open(line)
        elif _is_fence_close(line, active):
            active = None
    return active


def _extract_inline(
    lines: list[str], index: int, payload: str, closer: str | None
) -> FinalCommitment:
    tail = list(enumerate(lines[index + 1 :], index + 1))
    while tail and not tail[-1][1].strip():
        tail.pop()
    fence = _active_fence_before(lines, index)
    permitted: list[str] = []
    if fence is not None:
        permitted.append("fence")
    if closer is not None:
        permitted.append(closer)
    continuation: list[str] = []
    for tail_position, (_tail_index, line) in enumerate(tail):
        if fence is not None and _is_fence_close(line, fence) and "fence" in permitted:
            permitted.remove("fence")
            fence = None
            continue
        if closer is not None and line.strip() == closer and closer in permitted:
            permitted.remove(closer)
            continue
        if not line.strip() and any(value.strip() for _, value in tail[tail_position + 1 :]):
            continuation.append(line)
            continue
        # An unfenced multiline string continuation is syntactically distinct
        # from prose only when every non-empty continuation line is indented.
        if fence is None and closer is None and line[:1].isspace():
            continuation.append(line)
            continue
        return FinalCommitment(False, None, "substantive content follows final commitment")
    if fence is not None or permitted:
        return FinalCommitment(False, None, "unmatched final fence or emphasis wrapper")
    payload, reason = _unwrap_payload(payload, preserve_whitespace=bool(continuation))
    if payload is None or not payload:
        return FinalCommitment(False, None, reason or "empty final commitment")
    if continuation:
        payload = "\n".join([payload, *continuation])
    return FinalCommitment(True, payload, None)


def _extract_section(lines: list[str], index: int, closer: str | None) -> FinalCommitment:
    tail = lines[index + 1 :]
    while tail and not tail[0].strip():
        tail.pop(0)
    while tail and not tail[-1].strip():
        tail.pop()
    if not tail:
        return FinalCommitment(False, None, "empty final commitment")

    fence = _fence_open(tail[0])
    if fence is not None:
        if len(tail) < 3 or not _is_fence_close(tail[-1], fence):
            return FinalCommitment(False, None, "unmatched final fence")
        payload_lines = tail[1:-1]
    else:
        payload_lines = tail
        if closer is not None:
            if not payload_lines or payload_lines[-1].strip() != closer:
                return FinalCommitment(False, None, "unmatched final emphasis wrapper")
            payload_lines = payload_lines[:-1]

    while payload_lines and not payload_lines[0].strip():
        payload_lines.pop(0)
    while payload_lines and not payload_lines[-1].strip():
        payload_lines.pop()
    if not payload_lines:
        return FinalCommitment(False, None, "empty final commitment")

    # A nested FINAL inside a Final Answer section is one commitment, not two.
    nested = _marker(payload_lines[0])
    if nested is not None:
        kind, inline, nested_closer = nested
        if kind != "FINAL" or not inline or nested_closer is not None:
            return FinalCommitment(False, None, "ambiguous nested final section")
        payload, reason = _unwrap_payload(
            inline, preserve_whitespace=len(payload_lines) > 1
        )
        if payload is None or not payload:
            return FinalCommitment(False, None, reason or "empty final commitment")
        if len(payload_lines) > 1:
            payload = "\n".join([payload, *payload_lines[1:]])
        return FinalCommitment(True, payload, None)

    # Preserve all internal newlines and per-line spaces. Only outer empty lines
    # and matched syntactic delimiters have been removed.
    payload = "\n".join(payload_lines)
    payload, reason = _unwrap_payload(payload, preserve_whitespace=len(payload_lines) > 1)
    if payload is None or not payload:
        return FinalCommitment(False, None, reason or "empty final commitment")
    return FinalCommitment(True, payload, None)


def extract_final_commitment(raw_text: str, *, truncated: bool = False) -> FinalCommitment:
    """Extract one globally defined final commitment without condition metadata."""

    visible, unclosed_thinking = _visible_text(raw_text)
    if truncated or unclosed_thinking:
        return FinalCommitment(False, None, "truncated or unclosed response")
    lines = visible.splitlines()
    markers = [(index, _marker(line)) for index, line in enumerate(lines) if _marker(line)]
    if not markers:
        return FinalCommitment(False, None, "no final commitment")

    # A Final Answer section may contain exactly one subordinate FINAL line.
    if len(markers) == 2:
        (outer_index, outer), (inner_index, inner) = markers
        if (
            outer is not None
            and inner is not None
            and outer[0] in {"FINAL_ANSWER", "FINAL_SECTION"}
            and not outer[1].strip()
            and inner[0] == "FINAL"
            and inner_index > outer_index
        ):
            markers = [(outer_index, outer)]
        else:
            return FinalCommitment(False, None, "multiple final commitments")
    elif len(markers) != 1:
        return FinalCommitment(False, None, "multiple final commitments")

    index, marker = markers[0]
    assert marker is not None
    kind, inline, closer = marker
    if inline.strip():
        return _extract_inline(lines, index, inline, closer)
    if kind == "FINAL":
        return FinalCommitment(False, None, "empty final commitment")
    return _extract_section(lines, index, closer)


def _canonical_literal(value: Any) -> list[Any]:
    if value is None:
        return ["none", None]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, bytearray):
        return ["bytearray", bytes(value).hex()]
    if isinstance(value, list):
        return ["list", [_canonical_literal(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [_canonical_literal(item) for item in value]]
    if isinstance(value, dict):
        entries = [
            [_canonical_literal(key), _canonical_literal(item)] for key, item in value.items()
        ]
        entries.sort(key=lambda entry: json.dumps(entry[0], sort_keys=True))
        return ["dict", entries]
    if isinstance(value, set):
        entries = [_canonical_literal(item) for item in value]
        entries.sort(key=lambda entry: json.dumps(entry, sort_keys=True))
        return ["set", entries]
    if isinstance(value, frozenset):
        entries = [_canonical_literal(item) for item in value]
        entries.sort(key=lambda entry: json.dumps(entry, sort_keys=True))
        return ["frozenset", entries]
    raise ValueError(f"unsupported Python literal type: {type(value).__name__}")


def canonicalize_semantic_value(payload: str) -> list[Any]:
    """Return a tagged value using literal_eval or an exact raw string fallback."""

    bytearray_match = _BYTEARRAY.fullmatch(payload.strip())
    if bytearray_match is not None:
        try:
            inner = ast.literal_eval(bytearray_match.group("inner"))
        except (TypeError, ValueError, SyntaxError):
            inner = None
        if isinstance(inner, bytes):
            return _canonical_literal(bytearray(inner))
    frozenset_match = _FROZENSET.fullmatch(payload.strip())
    if frozenset_match is not None:
        try:
            inner = ast.literal_eval(frozenset_match.group("inner"))
        except (TypeError, ValueError, SyntaxError):
            inner = None
        if isinstance(inner, (set, list, tuple)):
            return _canonical_literal(frozenset(inner))
    try:
        value = ast.literal_eval(payload)
    except (TypeError, ValueError, SyntaxError):
        return ["str", payload]
    try:
        return _canonical_literal(value)
    except ValueError:
        return ["str", payload]


def evaluate_external_answer_v3(
    raw_text: str,
    reference_answer: str,
    *,
    truncated: bool = False,
    runtime_error: bool = False,
) -> SemanticV3Result:
    """Evaluate one preserved output along commitment/evaluability/correctness axes."""

    if runtime_error:
        return SemanticV3Result(False, False, None, None, False, "runtime error", None)
    commitment = extract_final_commitment(raw_text, truncated=truncated)
    if not commitment.valid or commitment.payload is None:
        return SemanticV3Result(
            False,
            False,
            None,
            None,
            False,
            commitment.failure_reason,
            commitment.payload,
        )
    actual = canonicalize_semantic_value(commitment.payload)
    expected = canonicalize_semantic_value(reference_answer)
    canonical = json.dumps(actual, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return SemanticV3Result(
        True,
        True,
        str(actual[0]),
        canonical,
        actual == expected,
        None,
        commitment.payload,
    )


__all__ = [
    "PARSER_VERSION",
    "FinalCommitment",
    "SemanticV3Result",
    "canonicalize_semantic_value",
    "evaluate_external_answer_v3",
    "extract_final_commitment",
]
