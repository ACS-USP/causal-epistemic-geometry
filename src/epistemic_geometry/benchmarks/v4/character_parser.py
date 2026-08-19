"""Deterministic semantic parser for the V4 character-count answer contract."""

from __future__ import annotations

import re

_FINAL_LINE = re.compile(
    r"^\s*(?:(?:[-*]|#{1,6})\s+)*(?:[✅☑✔]\s*)?"
    r"(?:\*\*|__|`)?FINAL(?:\s+ANSWER)?\s*:\s*"
    r"(?:\*\*|__|`)?([+-]?\d+)(?:\*\*|__|`)?\s*$",
    re.IGNORECASE,
)
_FINAL_LABEL = re.compile(
    r"^\s*(?:(?:[-*]|#{1,6})\s+)*(?:[✅☑✔]\s*)?"
    r"(?:\*\*|__|`)?FINAL(?:\s+ANSWER)?\s*:",
    re.IGNORECASE,
)
_INTEGER_LINE = re.compile(
    r"^\s*(?:\*\*|__|`)?([+-]?\d+)(?:\*\*|__|`)?\s*$"
)


def parse_final_integer(
    raw: str, *, truncated: bool = False
) -> tuple[str, int | None, str | None]:
    """Parse one explicit final integer without reading arbitrary reasoning text.

    The parser accepts harmless Markdown wrappers around the explicit ``FINAL``
    commitment, but requires that commitment to be the last non-empty line. It
    never searches the reasoning trace for a convenient number.
    """

    if truncated or re.search(r"<think>(?!.*?</think>)", raw, re.IGNORECASE | re.DOTALL):
        return "TRUNCATED_THINKING", None, "token cap or unclosed thinking block"
    visible = re.sub(r"<think>.*?</think>", "\n", raw, flags=re.IGNORECASE | re.DOTALL)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    matches: list[tuple[int, int, int]] = []
    label_lines: list[int] = []
    for index, line in enumerate(lines):
        match = _FINAL_LINE.fullmatch(line)
        if match:
            matches.append((index, index, int(match.group(1))))
        elif _FINAL_LABEL.match(line):
            label_lines.append(index)
            if index + 1 < len(lines):
                next_line = _INTEGER_LINE.fullmatch(lines[index + 1])
                if next_line:
                    matches.append((index, index + 1, int(next_line.group(1))))
    if not matches:
        if label_lines:
            return "INVALID_FORMAT", None, "FINAL commitment did not contain one integer"
        return "INVALID_FORMAT", None, "no explicit FINAL integer commitment"
    values = {value for _, _, value in matches}
    if len(values) > 1:
        return "INVALID_FORMAT", None, "conflicting FINAL integer commitments"
    if len(matches) != 1:
        return "INVALID_FORMAT", None, "multiple FINAL integer commitments"
    _, end_index, value = matches[0]
    if end_index != len(lines) - 1:
        return "INVALID_FORMAT", None, "FINAL commitment was not the last non-empty line"
    return "PARSED", value, None
