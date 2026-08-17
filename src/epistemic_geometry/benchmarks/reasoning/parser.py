"""Deterministic FINAL-field parsing for reasoning-agent outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ParseStatus = Literal[
    "OK",
    "MISSING_FINAL",
    "INVALID_FINAL",
    "THINKING_UNCLOSED",
    "TRUNCATED_NO_FINAL",
]
_FINAL_LINE = re.compile(r"^FINAL:\s*(.*?)\s*$", re.MULTILINE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_OPEN_THINK = re.compile(r"<think>(?!.*?</think>)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedFinal:
    raw_text: str
    answer_text: str | None
    answer: int | None
    status: ParseStatus

    @property
    def valid(self) -> bool:
        return self.status == "OK"


def _visible_text(raw_text: str) -> tuple[str, bool]:
    if _OPEN_THINK.search(raw_text):
        return "", True
    return _THINK_BLOCK.sub("\n", raw_text), False


def parse_exact_integer_final(raw_text: str, *, truncated: bool = False) -> ParsedFinal:
    """Parse the last exact non-negative integer FINAL line outside thinking."""

    visible, unclosed = _visible_text(raw_text)
    if unclosed:
        return ParsedFinal(raw_text, None, None, "THINKING_UNCLOSED")
    matches = _FINAL_LINE.findall(visible)
    if not matches:
        status: ParseStatus = "TRUNCATED_NO_FINAL" if truncated else "MISSING_FINAL"
        return ParsedFinal(raw_text, None, None, status)
    answer_text = matches[-1].strip()
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", answer_text):
        return ParsedFinal(raw_text, answer_text, None, "INVALID_FINAL")
    return ParsedFinal(raw_text, answer_text, int(answer_text), "OK")


def parse_family_final(raw_text: str, family: str, *, truncated: bool = False) -> ParsedFinal:
    if family not in {"MODREG-R", "FSM-R", "SATCOUNT-R"}:
        raise ValueError(f"unknown reasoning family {family!r}")
    return parse_exact_integer_final(raw_text, truncated=truncated)
