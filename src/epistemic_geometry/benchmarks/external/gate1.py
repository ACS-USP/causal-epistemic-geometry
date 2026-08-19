"""Prospective Gate 1 rules for the full non-thinking smoke.

The functions here are deliberately outcome-agnostic: Stage 1 only decides
whether an instrument is technically evaluable, while Stage 2 is the first
place where the frozen n=20 classification is allowed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from .base import ExternalResult, ExternalStatus

VALID_STATUSES = frozenset({ExternalStatus.VALID_CORRECT, ExternalStatus.VALID_WRONG})
MECHANICAL_STATUSES = frozenset(
    {ExternalStatus.INVALID_FORMAT, ExternalStatus.TRUNCATED_THINKING, ExternalStatus.RUNTIME_ERROR}
)


@dataclass(frozen=True)
class Gate1Summary:
    """Counts and frozen classification for one instrument."""

    instrument: str
    n: int
    valid_count: int
    correct_count: int
    wrong_count: int
    mechanical_failure_count: int
    valid_completion: float
    stage1_technical_pass: bool | None
    classification: str | None

    def to_record(self) -> dict[str, object]:
        return {
            "instrument": self.instrument,
            "n": self.n,
            "valid_count": self.valid_count,
            "correct_count": self.correct_count,
            "wrong_count": self.wrong_count,
            "mechanical_failure_count": self.mechanical_failure_count,
            "valid_completion": self.valid_completion,
            "stage1_technical_pass": self.stage1_technical_pass,
            "classification": self.classification,
        }


def stage1_technical_pass(results: Iterable[ExternalResult]) -> bool:
    """Apply the pre-registered 5-item technical-only continuation gate."""

    rows = list(results)
    if len(rows) != 5:
        raise ValueError("Gate 1 Stage 1 requires exactly five rows")
    valid = sum(row.status in VALID_STATUSES for row in rows)
    failures = len(rows) - valid
    # With five rows, four semantically evaluable outcomes means at most one
    # mechanical failure; this is the frozen meaning of “failures do not
    # dominate”.  Correctness is deliberately not consulted here.
    return valid >= 4 and failures < valid


def classify_full_n20(results: Iterable[ExternalResult]) -> str:
    """Classify a completed n=20 instrument without adaptive thresholds."""

    rows = list(results)
    if len(rows) != 20:
        raise ValueError("Gate 1 Stage 2 classification requires exactly 20 rows")
    valid = sum(row.status in VALID_STATUSES for row in rows)
    correct = sum(row.status == ExternalStatus.VALID_CORRECT for row in rows)
    wrong = sum(row.status == ExternalStatus.VALID_WRONG for row in rows)
    mechanical = len(rows) - valid
    if valid / len(rows) < 0.90:
        return "MECHANICAL_OR_COMPLETION_FAILURE"
    if correct >= 2 and wrong >= 2 and mechanical <= wrong:
        return "PROMISING"
    if wrong < 2:
        return "SATURATED"
    if correct < 2:
        return "FLOOR"
    # This residual category is conservative: a mixed outcome with too many
    # mechanical failures is not treated as semantic evidence.
    return "MECHANICAL_OR_COMPLETION_FAILURE"


def summarize_gate1(
    instrument: str,
    results: Iterable[ExternalResult],
    *,
    stage1: bool | None = None,
    classification: str | None = None,
) -> Gate1Summary:
    """Create a compact auditable summary for a Stage 1 or Stage 2 result."""

    rows = list(results)
    if not rows:
        raise ValueError("cannot summarize empty Gate 1 results")
    counts = Counter(row.status for row in rows)
    valid = counts[ExternalStatus.VALID_CORRECT] + counts[ExternalStatus.VALID_WRONG]
    return Gate1Summary(
        instrument=instrument,
        n=len(rows),
        valid_count=valid,
        correct_count=counts[ExternalStatus.VALID_CORRECT],
        wrong_count=counts[ExternalStatus.VALID_WRONG],
        mechanical_failure_count=len(rows) - valid,
        valid_completion=valid / len(rows),
        stage1_technical_pass=stage1,
        classification=classification,
    )
