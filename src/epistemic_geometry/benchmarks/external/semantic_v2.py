"""Deterministic semantic parser used by the Gate 6.3 reanalysis.

The historical ``external-semantic-v1`` parser required the FINAL marker to be
the last non-empty line.  That rule confounds harmless Markdown fences with
semantic failure when a model emits one unambiguous FINAL commitment inside a
closed code block.  This module is deliberately separate from the historical
parser: it accepts only the narrowly specified wrappers while retaining the
same type-aware evaluator downstream.
"""

from __future__ import annotations

import re

from .base import ExternalStatus, ParsedExternalAnswer, _visible_text

PARSER_VERSION = "external-semantic-v2"

_FINAL_COMMITMENT = re.compile(
    r"^FINAL\s*:\s*(?P<payload>.*?)\s*$",
    re.IGNORECASE,
)
_DECORATION_PREFIXES = (
    re.compile(r"^#{1,6}\s+"),
    re.compile(r"^[-*]\s+"),
    re.compile(r"^[✅☑✔]\s*"),
)
_FENCE = re.compile(r"^(?P<marker>`{3,}|~{3,})(?P<label>.*)$")


def _remove_decoration_prefix(line: str) -> str:
    """Remove only Markdown heading/list/checkmark decoration."""

    value = line.strip()
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


def _unwrap_matching(value: str, wrapper: str) -> str | None:
    if value.startswith(wrapper):
        if not value.endswith(wrapper) or len(value) <= 2 * len(wrapper):
            return None
        return value[len(wrapper) : -len(wrapper)].strip()
    return value


def _parse_final_line(line: str) -> tuple[bool, str | None, str | None]:
    """Return ``(is_commitment, payload, reason)`` for one complete line."""

    value = _remove_decoration_prefix(line)
    opening: str | None = None
    for wrapper in ("**", "__", "`"):
        if value.startswith(wrapper):
            opening = wrapper
            value = value[len(wrapper) :].strip()
            break

    match = _FINAL_COMMITMENT.fullmatch(value)
    if match is None:
        return False, None, None

    payload = match.group("payload").strip()
    if opening is not None:
        if not payload.endswith(opening) or len(payload) <= len(opening):
            return True, None, "unclosed FINAL wrapper"
        payload = payload[: -len(opening)].strip()

    if not payload:
        return True, None, "empty FINAL commitment"

    # Also accept a harmless wrapper around the payload itself, e.g.
    # ``FINAL: `2` ``.  Unmatched delimiters remain part of the payload and are
    # rejected by the typed evaluator rather than silently repaired.
    for wrapper in ("**", "__", "`"):
        unwrapped = _unwrap_matching(payload, wrapper)
        if unwrapped is None:
            return True, None, "malformed FINAL wrapper"
        if unwrapped != payload:
            payload = unwrapped
            break
        if payload.endswith(wrapper):
            return True, None, "unmatched FINAL wrapper"
    return True, payload, None


def _fence_transition(
    line: str, active: tuple[str, int] | None
) -> tuple[tuple[str, int] | None, bool]:
    """Track only standalone Markdown fences; return new state and close flag."""

    stripped = line.strip()
    if active is not None:
        marker, length = active
        if (
            stripped == marker * length
            or (marker == "`" and re.fullmatch(rf"`{{{length},}}", stripped))
            or (marker == "~" and re.fullmatch(rf"~{{{length},}}", stripped))
        ):
            return None, True
        return active, False
    match = _FENCE.fullmatch(stripped)
    if match is None:
        return None, False
    marker = match.group("marker")
    return (marker[0], len(marker)), False


def parse_external_answer_v2(raw_text: str, *, truncated: bool = False) -> ParsedExternalAnswer:
    """Parse one unique FINAL commitment with narrowly allowed Markdown wrappers.

    The parser never searches arbitrary prose for an answer.  A commitment is
    recognized only when an entire non-empty line has the ``FINAL:`` contract.
    After that line, only one matching fence closer or a matching standalone
    emphasis delimiter may occur.
    """

    visible, unclosed_thinking = _visible_text(raw_text)
    if unclosed_thinking or truncated:
        return ParsedExternalAnswer(
            raw_text, None, ExternalStatus.TRUNCATED_THINKING, "truncated response"
        )

    lines = visible.splitlines()
    nonempty = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    active_fence: tuple[str, int] | None = None
    candidates: list[tuple[int, str | None, str | None, tuple[str, int] | None]] = []
    for index, line in nonempty:
        is_commitment, payload, reason = _parse_final_line(line)
        if is_commitment:
            candidates.append((index, payload, reason, active_fence))
        active_fence, _closed = _fence_transition(line, active_fence)

    if len(candidates) != 1:
        return ParsedExternalAnswer(
            raw_text,
            None,
            ExternalStatus.INVALID_FORMAT,
            "expected exactly one FINAL commitment",
        )

    final_index, payload, reason, fence_before = candidates[0]
    if payload is None:
        return ParsedExternalAnswer(raw_text, None, ExternalStatus.INVALID_FORMAT, reason)

    after = [line for index, line in nonempty if index > final_index]
    allowed_emphasis = {"**", "__", "`"}
    if after:
        allowed_fence = False
        if fence_before is not None and len(after) == 1:
            marker, length = fence_before
            close = after[0]
            allowed_fence = close == marker * length or bool(
                re.fullmatch(re.escape(marker) + rf"{{{length},}}", close)
            )
        if not allowed_fence and not (len(after) == 1 and after[0] in allowed_emphasis):
            return ParsedExternalAnswer(
                raw_text,
                payload,
                ExternalStatus.INVALID_FORMAT,
                "substantive content follows FINAL commitment",
            )

    return ParsedExternalAnswer(raw_text, payload, None)


__all__ = ["PARSER_VERSION", "parse_external_answer_v2"]
