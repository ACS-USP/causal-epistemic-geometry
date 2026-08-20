"""Pure helpers for the Gate-6.1 source attrition repair.

The original Gate-6 source runner used a tokenized ``FINAL:`` substring and
fell back to the prompt boundary when it could not find one.  That fallback is
unsafe: a missing marker is precisely the mechanical condition that the source
screen must record.  This module keeps marker localization and candidate
selection CPU-only and independent of model correctness.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

_FINAL_MARKER = re.compile(r"FINAL\s*:", re.IGNORECASE)
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)


@dataclass(frozen=True)
class FinalCommitmentBoundary:
    """The generated-token boundary immediately before a final commitment."""

    marker_text_span: tuple[int, int]
    marker_token_index: int
    marker_text: str
    line_text: str
    reason: str


@dataclass(frozen=True)
class SourceCandidateDecision:
    """Outcome-independent mechanical decision for one source candidate."""

    split: str
    candidate_item_id: str
    candidate_order: int
    allocation: str
    eligible: bool
    reason: str
    condition_status: dict[str, str]


def _has_unclosed_think(raw_output: str) -> bool:
    """Return whether the generated output contains an unclosed think block."""

    return len(_THINK_OPEN.findall(raw_output)) != len(_THINK_CLOSE.findall(raw_output))


def _visible_nonempty_lines(raw_output: str) -> list[tuple[int, str]]:
    """Return non-empty visible lines with their offsets in ``raw_output``."""

    hidden_ranges = [(match.start(), match.end()) for match in _THINK_BLOCK.finditer(raw_output)]
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in raw_output.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip() and not any(
            start <= offset < end or start < offset + len(content) <= end
            for start, end in hidden_ranges
        ):
            lines.append((offset, content))
        offset += len(line)
    return lines


def _parse_final_line(line: str) -> tuple[int, int, str] | None:
    """Parse one permitted final line and return marker offsets/payload.

    Markdown list and heading prefixes, one leading emphasis/code wrapper, and
    the matching trailing wrapper are presentation only.  The marker itself is
    retained so its offset can be mapped to generated tokens.
    """

    text = line.strip()
    while True:
        match = re.match(r"^(?:[-*]|#{1,6})\s+", text)
        if not match:
            break
        text = text[match.end() :].lstrip()
    if text[:1] in {"✅", "☑", "✔"}:
        text = text[1:].lstrip()
    wrapper = ""
    for candidate in ("**", "__", "`"):
        if text.startswith(candidate):
            wrapper = candidate
            text = text[len(candidate) :].lstrip()
            break
    marker = _FINAL_MARKER.match(text)
    if marker is None:
        return None
    payload = text[marker.end() :].strip()
    if wrapper:
        if not payload.endswith(wrapper):
            return None
        payload = payload[: -len(wrapper)].rstrip()
    if not payload:
        return None
    marker_start = line.find("FINAL", line.find(text))
    if marker_start < 0:
        return None
    return marker_start, marker_start + (marker.end() - marker.start()), payload


def _marker_span(raw_output: str) -> tuple[tuple[int, int], str, str] | None:
    """Find exactly one final commitment line, without evaluating its answer."""

    if _has_unclosed_think(raw_output):
        return None
    lines = _visible_nonempty_lines(raw_output)
    matches: list[tuple[tuple[int, int], str, str]] = []
    for offset, line in lines:
        parsed = _parse_final_line(line)
        if parsed is None:
            continue
        start, end, payload = parsed
        matches.append(((offset + start, offset + end), line, payload))
    if len(matches) != 1:
        return None
    match = matches[0]
    last_offset, last_line = lines[-1]
    if match[1] != last_line or match[0][0] != last_offset + _parse_final_line(last_line)[0]:
        return None
    return match


def _special_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "all_special_ids", ()) or ()
    return {int(value) for value in values}


def _tokenized_offset_index(
    raw_output: str,
    generated_token_ids: Sequence[int],
    marker_start: int,
    tokenizer: Any,
) -> int | None:
    """Use fast-tokenizer offsets when their IDs exactly match the output."""

    try:
        encoded = tokenizer(
            raw_output,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        if (
            offsets
            and isinstance(offsets[0], list)
            and offsets
            and offsets[0]
            and isinstance(offsets[0][0], (list, tuple))
        ):
            offsets = offsets[0]
        special = _special_ids(tokenizer)
        filtered = [
            (index, int(token_id))
            for index, token_id in enumerate(generated_token_ids)
            if int(token_id) not in special
        ]
        if [token_id for _index, token_id in filtered] != [int(token_id) for token_id in ids]:
            return None
        for offset_index, (start, end) in enumerate(offsets):
            if int(start) <= marker_start < int(end):
                return filtered[offset_index][0]
            if int(start) == int(end) and int(start) == marker_start:
                return filtered[offset_index][0]
    except (AttributeError, KeyError, TypeError, ValueError, IndexError):
        return None
    return None


def _decoded_prefix_index(
    raw_output: str,
    generated_token_ids: Sequence[int],
    marker_start: int,
    tokenizer: Any,
) -> int | None:
    """Fallback mapping based only on generated-token prefix decoding.

    It deliberately has no prompt-position fallback.  The first generated
    prefix containing the marker is the token that introduces the marker.
    """

    marker_text = raw_output[marker_start:]
    if not re.match(r"FINAL", marker_text, re.IGNORECASE):
        return None
    for index in range(len(generated_token_ids)):
        try:
            decoded = tokenizer.decode(
                list(map(int, generated_token_ids[: index + 1])),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            decoded = tokenizer.decode(
                list(map(int, generated_token_ids[: index + 1])), skip_special_tokens=True
            )
        last_line_start = max(decoded.rfind("\n"), decoded.rfind("\r")) + 1
        final_prefix = decoded[last_line_start:]
        for position, character in enumerate(final_prefix):
            if character.casefold() != "f":
                continue
            available = final_prefix[position : position + len("FINAL")]
            if "FINAL".casefold().startswith(available.casefold()) and available:
                return index
    return None


def locate_final_commitment_boundary(
    raw_output: str,
    generated_token_ids: Sequence[int],
    tokenizer: Any,
) -> FinalCommitmentBoundary | None:
    """Locate an unambiguous final marker in generated-token coordinates.

    The returned index is the first generated token belonging to ``FINAL:``.
    An index of zero is valid.  ``None`` means the output is mechanically
    ineligible: missing/conflicting marker, unclosed thinking, or ambiguous
    token mapping.
    """

    located = _marker_span(raw_output)
    if located is None:
        return None
    span, line, _payload = located
    index = _tokenized_offset_index(raw_output, generated_token_ids, span[0], tokenizer)
    if index is None:
        index = _decoded_prefix_index(raw_output, generated_token_ids, span[0], tokenizer)
    if index is None:
        return None
    return FinalCommitmentBoundary(
        marker_text_span=span,
        marker_token_index=int(index),
        marker_text=raw_output[span[0] : span[1]],
        line_text=line,
        reason="unique_final_marker_exact_token_boundary",
    )


def select_common_eligible(
    candidates: Iterable[dict[str, Any]],
    *,
    target: int,
    max_ineligible: int,
    split: str,
) -> tuple[list[dict[str, Any]], list[SourceCandidateDecision]]:
    """Select the first mechanically eligible candidates in frozen order."""

    selected: list[dict[str, Any]] = []
    decisions: list[SourceCandidateDecision] = []
    ineligible = 0
    for candidate in candidates:
        if len(selected) >= target:
            break
        conditions = dict(candidate.get("condition_status", {}))
        eligible = bool(candidate.get("eligible", False))
        reason = str(candidate.get("reason", "eligible" if eligible else "ineligible"))
        if not eligible:
            ineligible += 1
            if ineligible > max_ineligible:
                raise RuntimeError(f"{split}: GATE6_SOURCE_ATTRITION_EXCEEDS_LIMIT")
        else:
            selected.append(candidate)
        decisions.append(
            SourceCandidateDecision(
                split=split,
                candidate_item_id=str(candidate["item_id"]),
                candidate_order=int(candidate["candidate_order"]),
                allocation=str(candidate.get("allocation", "")),
                eligible=eligible,
                reason=reason,
                condition_status=conditions,
            )
        )
    if len(selected) < target:
        raise RuntimeError(f"{split}: GATE6_SOURCE_RESERVE_EXHAUSTED")
    return selected, decisions
