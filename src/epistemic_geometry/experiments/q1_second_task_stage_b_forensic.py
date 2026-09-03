"""Independent specification-equivalent parser for the Stage-B forensic resolution.

This module is additive post-closeout audit code.  It deliberately does not
import or call the hash-pinned Stage-A2 primary parser.  The shared frozen
``extract_final_commitment`` primitive is retained for the pre-existing direct
commitment path; the Stage-A2 repair and typed decisions are independently
implemented here.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from epistemic_geometry.benchmarks.external.semantic_v3 import extract_final_commitment

SUPPORTED_REPAIR_TYPES = {"bool", "int", "list", "str"}

_DECORATIONS = (
    re.compile(r"^#{1,6}[ \t]+"),
    re.compile(r"^[-*][ \t]+"),
    re.compile(r"^[✅☑✔][ \t]*"),
)
_FINAL = re.compile(r"^FINAL[ \t]*:[ \t]?(?P<payload>.*)$", re.IGNORECASE)
_FINAL_ANSWER = re.compile(
    r"^FINAL[ \t]+ANSWER[ \t]*:[ \t]?(?P<payload>.*)$", re.IGNORECASE
)
_LEGACY_FINAL_LINE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?final(?:\s+answer)?\s*:\s*(.*?)\s*(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)```", re.DOTALL)


@dataclass(frozen=True)
class IndependentCandidate:
    payload: str
    canonical: Any
    canonical_json: str
    value_type: str


def canonical(value: Any) -> Any:
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
    if isinstance(value, list):
        return ["list", [canonical(item) for item in value]]
    if isinstance(value, tuple):
        return ["tuple", [canonical(item) for item in value]]
    if isinstance(value, dict):
        records = [[canonical(key), canonical(item)] for key, item in value.items()]
        return ["dict", sorted(records, key=lambda record: json.dumps(record[0], sort_keys=True))]
    if isinstance(value, set):
        records = [canonical(item) for item in value]
        return ["set", sorted(records, key=lambda record: json.dumps(record, sort_keys=True))]
    raise ValueError(type(value).__name__)


def parse_payload(payload: str) -> Any:
    try:
        value = ast.literal_eval(payload)
    except (SyntaxError, ValueError, TypeError):
        try:
            value = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("payload is not a Python or JSON literal") from exc
    return canonical(value)


def visible_text(raw: str) -> tuple[str, bool]:
    lowered = raw.lower()
    if "<think>" not in lowered:
        return raw, False
    pieces: list[str] = []
    cursor = 0
    while True:
        start = lowered.find("<think>", cursor)
        if start < 0:
            pieces.append(raw[cursor:])
            return "".join(pieces), False
        pieces.append(raw[cursor:start])
        end = lowered.find("</think>", start + len("<think>"))
        if end < 0:
            return "", True
        pieces.append("\n")
        cursor = end + len("</think>")


def _remove_decorations(line: str) -> tuple[str, bool]:
    is_heading = bool(re.match(r"^\s*#{1,6}[ \t]+", line))
    value = line.lstrip()
    changed = True
    while changed and value:
        changed = False
        for pattern in _DECORATIONS:
            match = pattern.match(value)
            if match is not None:
                value = value[match.end() :].lstrip()
                changed = True
                break
    return value, is_heading


def _unwrap_line(value: str) -> tuple[str, str | None]:
    value = value.lstrip()
    for wrapper in ("**", "__", "`"):
        if not value.startswith(wrapper):
            continue
        rest = value[len(wrapper) :]
        trimmed = rest.rstrip()
        if trimmed.endswith(wrapper):
            return trimmed[: -len(wrapper)], None
        return rest.lstrip(), wrapper
    return value, None


def marker(line: str) -> tuple[str, str, str | None] | None:
    normalized, is_heading = _remove_decorations(line)
    normalized, closer = _unwrap_line(normalized)
    match = _FINAL.fullmatch(normalized)
    if match is not None:
        payload = match.group("payload")
        return ("FINAL_SECTION" if is_heading and not payload else "FINAL", payload, closer)
    match = _FINAL_ANSWER.fullmatch(normalized)
    if match is not None:
        return "FINAL_ANSWER", match.group("payload"), closer
    return None


def _candidate(payload: str) -> IndependentCandidate | None:
    try:
        value = parse_payload(payload.strip())
    except ValueError:
        return None
    if value[0] not in SUPPORTED_REPAIR_TYPES:
        return None
    return IndependentCandidate(
        payload=payload.strip(),
        canonical=value,
        canonical_json=json.dumps(value, ensure_ascii=False, separators=(",", ":")),
        value_type=str(value[0]),
    )


def _fenced_candidates(visible: str) -> list[IndependentCandidate]:
    candidates: list[IndependentCandidate] = []
    for match in _FENCE.finditer(visible):
        body = match.group("body").strip()
        parsed_marker = marker(body)
        if parsed_marker is not None:
            body = parsed_marker[1]
        candidate = _candidate(body)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _standalone_candidates(visible: str) -> list[IndependentCandidate]:
    candidates: list[IndependentCandidate] = []
    for line in visible.splitlines():
        payload = line.strip()
        if not payload or marker(line) is not None or payload.startswith("```"):
            continue
        if payload.startswith("`") and payload.endswith("`") and len(payload) >= 2:
            payload = payload[1:-1].strip()
        candidate = _candidate(payload)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def mechanical_repetition(token_ids: Sequence[int]) -> bool:
    tokens = [int(value) for value in token_ids]
    if len(tokens) < 128:
        return False
    tail = tokens[-128:]
    if len(set(tail)) / len(tail) <= 0.20:
        return True
    for width in (1, 2, 4, 8):
        for end in range(width * 8, len(tokens) + 1):
            block = tokens[end - width : end]
            if all(
                tokens[end - width * repeat : end - width * (repeat - 1)] == block
                for repeat in range(1, 9)
            ):
                return True
    return False


def repaired_terminal_candidate(raw: str) -> IndependentCandidate | None:
    # The frozen Stage-A2 repair is additive to the diagnostic classifier.  It
    # is eligible only from the classifier's multi-marker branch; recognizing
    # a terminal candidate by the broader semantic-v3 marker grammar alone is
    # not sufficient to enter the repair path.
    legacy_marker_count = sum(
        _LEGACY_FINAL_LINE.fullmatch(line) is not None for line in raw.splitlines()
    )
    if legacy_marker_count < 2:
        return None
    visible, unclosed = visible_text(raw)
    if unclosed:
        return None
    lines = visible.splitlines()
    markers = [
        (index, parsed)
        for index, line in enumerate(lines)
        if (parsed := marker(line)) is not None
    ]
    if len(markers) < 2:
        return None
    nonempty = [index for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return None
    terminal_index, (terminal_kind, terminal_payload, terminal_closer) = markers[-1]
    if (
        terminal_index != nonempty[-1]
        or terminal_kind != "FINAL"
        or terminal_closer is not None
        or not terminal_payload.strip()
    ):
        return None
    terminal = _candidate(terminal_payload)
    if terminal is None:
        return None
    for _index, (kind, payload, _closer) in markers[:-1]:
        if kind not in {"FINAL_ANSWER", "FINAL_SECTION"} or payload.strip():
            return None
    outside = [*_fenced_candidates(visible), *_standalone_candidates(visible)]
    if any(value.canonical_json != terminal.canonical_json for value in outside):
        return None
    return terminal


def independent_score(row: dict[str, Any]) -> dict[str, bool]:
    raw = str(row["raw_output"])
    truncated = bool(row["truncated"])
    direct = extract_final_commitment(raw, truncated=truncated)
    base = {"commitment_valid": False, "semantic_evaluable": False, "correct": False}
    if direct.valid and direct.payload is not None:
        base["commitment_valid"] = True
        try:
            actual = parse_payload(direct.payload)
        except ValueError:
            actual = None
        if actual is not None:
            expected = canonical(json.loads(row["reference_answer"]))
            return {
                "commitment_valid": True,
                "semantic_evaluable": True,
                "correct": actual == expected,
            }
    if mechanical_repetition(row["generated_token_ids"]):
        return base
    candidate = None if truncated else repaired_terminal_candidate(raw)
    if candidate is None:
        return base
    expected = canonical(json.loads(row["reference_answer"]))
    if candidate.value_type != expected[0]:
        return base
    return {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": candidate.canonical == expected,
    }


__all__ = [
    "SUPPORTED_REPAIR_TYPES",
    "canonical",
    "independent_score",
    "marker",
    "mechanical_repetition",
    "parse_payload",
    "repaired_terminal_candidate",
    "visible_text",
]
