"""Typed records for deterministic external-benchmark evaluation."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest


class ExternalStatus(StrEnum):
    """Outcome categories kept separate during instrument qualification."""

    VALID_CORRECT = "VALID_CORRECT"
    VALID_WRONG = "VALID_WRONG"
    INVALID_FORMAT = "INVALID_FORMAT"
    TRUNCATED_THINKING = "TRUNCATED_THINKING"
    RUNTIME_ERROR = "RUNTIME_ERROR"


@dataclass(frozen=True)
class ExternalItem:
    """One objective benchmark item in the normalized internal schema."""

    item_id: str
    benchmark: str
    subtask: str
    prompt: str
    reference_answer: str
    evaluator: str
    source_revision: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_hash(self) -> str:
        return stable_digest("EXTERNAL-PROMPT", self.prompt)

    @property
    def item_hash(self) -> str:
        return stable_digest(
            "EXTERNAL-ITEM",
            self.benchmark,
            self.subtask,
            self.item_id,
            self.prompt_hash,
            self.reference_answer,
            self.evaluator,
            self.source_revision,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "subtask": self.subtask,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "reference_answer": self.reference_answer,
            "evaluator": self.evaluator,
            "source_revision": self.source_revision,
            "item_hash": self.item_hash,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ParsedExternalAnswer:
    """Strictly parsed final response before objective scoring."""

    raw_text: str
    answer_text: str | None
    status: ExternalStatus | None
    parse_reason: str | None = None


@dataclass(frozen=True)
class ExternalResult:
    """One model outcome retaining parsing and correctness separately."""

    item_id: str
    benchmark: str
    subtask: str
    rollout_seed: int
    raw_output: str
    parsed_answer: str | None
    status: ExternalStatus
    correct: bool
    reference_answer: str
    evaluator: str
    token_count: int | None = None
    prompt_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "benchmark": self.benchmark,
            "subtask": self.subtask,
            "rollout_seed": self.rollout_seed,
            "raw_output": self.raw_output,
            "parsed_answer": self.parsed_answer,
            "status": self.status.value,
            "correct": self.correct,
            "reference_answer": self.reference_answer,
            "evaluator": self.evaluator,
            "token_count": self.token_count,
            "prompt_hash": self.prompt_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ExternalResult:
        return cls(
            item_id=str(record["item_id"]),
            benchmark=str(record["benchmark"]),
            subtask=str(record["subtask"]),
            rollout_seed=int(record["rollout_seed"]),
            raw_output=str(record.get("raw_output", "")),
            parsed_answer=(
                str(record["parsed_answer"]) if record.get("parsed_answer") is not None else None
            ),
            status=ExternalStatus(str(record["status"])),
            correct=bool(record["correct"]),
            reference_answer=str(record["reference_answer"]),
            evaluator=str(record["evaluator"]),
            token_count=(
                int(record["token_count"]) if record.get("token_count") is not None else None
            ),
            prompt_hash=str(record.get("prompt_hash", "")),
            metadata=dict(record.get("metadata", {})),
        )


def _visible_text(raw_text: str) -> tuple[str, bool]:
    """Remove closed thinking blocks and detect an unclosed thinking block."""

    if re.search(r"<think>(?!.*?</think>)", raw_text, re.IGNORECASE | re.DOTALL):
        return "", True
    visible = re.sub(r"<think>.*?</think>", "\n", raw_text, flags=re.IGNORECASE | re.DOTALL)
    return visible, False


def parse_external_answer(raw_text: str, *, truncated: bool = False) -> ParsedExternalAnswer:
    """Parse exactly one final line without fuzzy extraction.

    The response contract is ``FINAL: <answer>``.  A model may reason before
    that line, but prose after the line is invalid.  This prevents a sentence
    containing multiple candidate answers from being silently relabeled.
    """

    visible, unclosed = _visible_text(raw_text)
    if unclosed or truncated:
        return ParsedExternalAnswer(raw_text, None, ExternalStatus.TRUNCATED_THINKING)
    lines = [line.strip() for line in visible.splitlines() if line.strip()]
    finals = [line[6:].strip() for line in lines if line.startswith("FINAL:")]
    if len(finals) != 1:
        return ParsedExternalAnswer(
            raw_text,
            None,
            ExternalStatus.INVALID_FORMAT,
            "expected exactly one FINAL line",
        )
    final_line_index = next(index for index, line in enumerate(lines) if line.startswith("FINAL:"))
    if final_line_index != len(lines) - 1 or not finals[0]:
        return ParsedExternalAnswer(
            raw_text,
            finals[0] or None,
            ExternalStatus.INVALID_FORMAT,
            "FINAL must be the last non-empty line",
        )
    return ParsedExternalAnswer(raw_text, finals[0], None)


def _normalize_exact(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_python_literal(value: str) -> str:
    """Compare Python output semantically, without executing arbitrary code."""

    parsed = ast.literal_eval(value.strip())
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str)


def _normalize_json(value: str) -> str:
    return json.dumps(json.loads(value), sort_keys=True, separators=(",", ":"), default=str)


EVALUATORS: dict[str, Callable[[str, str], bool]] = {
    "exact": lambda actual, expected: _normalize_exact(actual) == _normalize_exact(expected),
    "python_literal": lambda actual, expected: _normalize_python_literal(actual)
    == _normalize_python_literal(expected),
    "json": lambda actual, expected: _normalize_json(actual) == _normalize_json(expected),
}


def score_external_response(
    item: ExternalItem,
    raw_output: str,
    *,
    rollout_seed: int,
    truncated: bool = False,
    token_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExternalResult:
    """Parse and score one response using only the item's registered evaluator."""

    parsed = parse_external_answer(raw_output, truncated=truncated)
    if parsed.status is not None:
        return ExternalResult(
            item_id=item.item_id,
            benchmark=item.benchmark,
            subtask=item.subtask,
            rollout_seed=rollout_seed,
            raw_output=raw_output,
            parsed_answer=parsed.answer_text,
            status=parsed.status,
            correct=False,
            reference_answer=item.reference_answer,
            evaluator=item.evaluator,
            token_count=token_count,
            prompt_hash=item.prompt_hash,
            metadata={"parse_reason": parsed.parse_reason, **(metadata or {})},
        )
    evaluator = EVALUATORS.get(item.evaluator)
    if evaluator is None:
        raise ValueError(f"unsupported deterministic evaluator: {item.evaluator!r}")
    try:
        correct = evaluator(parsed.answer_text or "", item.reference_answer)
    except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
        return ExternalResult(
            item_id=item.item_id,
            benchmark=item.benchmark,
            subtask=item.subtask,
            rollout_seed=rollout_seed,
            raw_output=raw_output,
            parsed_answer=parsed.answer_text,
            status=ExternalStatus.INVALID_FORMAT,
            correct=False,
            reference_answer=item.reference_answer,
            evaluator=item.evaluator,
            token_count=token_count,
            prompt_hash=item.prompt_hash,
            metadata={"parse_reason": "evaluator could not parse answer", **(metadata or {})},
        )
    return ExternalResult(
        item_id=item.item_id,
        benchmark=item.benchmark,
        subtask=item.subtask,
        rollout_seed=rollout_seed,
        raw_output=raw_output,
        parsed_answer=parsed.answer_text,
        status=ExternalStatus.VALID_CORRECT if correct else ExternalStatus.VALID_WRONG,
        correct=correct,
        reference_answer=item.reference_answer,
        evaluator=item.evaluator,
        token_count=token_count,
        prompt_hash=item.prompt_hash,
        metadata=metadata or {},
    )


def validate_item(item: ExternalItem) -> None:
    """Reject ambiguous or incomplete normalized benchmark rows."""

    if not item.item_id or not item.prompt.strip():
        raise ValueError("external item requires non-empty item_id and prompt")
    if item.evaluator not in EVALUATORS:
        raise ValueError(f"item {item.item_id} uses unknown evaluator {item.evaluator!r}")
    if not item.reference_answer.strip():
        raise ValueError(f"item {item.item_id} has an empty reference answer")
    if item.prompt_hash != stable_digest("EXTERNAL-PROMPT", item.prompt):
        raise ValueError(f"item {item.item_id} has an invalid prompt hash")
    if item.item_hash != stable_digest(
        "EXTERNAL-ITEM",
        item.benchmark,
        item.subtask,
        item.item_id,
        item.prompt_hash,
        item.reference_answer,
        item.evaluator,
        item.source_revision,
    ):
        raise ValueError(f"item {item.item_id} has an invalid item hash")


def record_digest(items: list[ExternalItem]) -> str:
    return stable_digest("EXTERNAL-ITEMS", canonical_json([item.to_record() for item in items]))
