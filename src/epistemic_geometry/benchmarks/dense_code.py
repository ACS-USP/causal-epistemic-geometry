"""Nested deterministic test-case outcomes for a future code pilot."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class TestCaseStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    COMPILE_ERROR = "COMPILE_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"


@dataclass(frozen=True)
class TestCaseOutcome:
    problem_id: str
    test_case_id: str
    status: TestCaseStatus
    duration_ms: float | None = None
    message: str | None = None

    @property
    def failed(self) -> bool:
        return self.status is not TestCaseStatus.PASS


@dataclass(frozen=True)
class ProgramOutcome:
    problem_id: str
    test_cases: tuple[TestCaseOutcome, ...]

    def failure_vector(self) -> tuple[int, ...]:
        return tuple(int(case.failed) for case in self.test_cases)

    def summary(self) -> dict[str, int | float]:
        counts = {status.value: 0 for status in TestCaseStatus}
        for case in self.test_cases:
            counts[case.status.value] += 1
        counts["n_tests"] = len(self.test_cases)
        counts["failure_count"] = sum(self.failure_vector())
        counts["pass_fraction"] = (
            1 - counts["failure_count"] / len(self.test_cases) if self.test_cases else 0.0
        )
        return counts


def make_program_outcome(problem_id: str, outcomes: Iterable[TestCaseOutcome]) -> ProgramOutcome:
    rows = tuple(outcomes)
    if not rows:
        raise ValueError("program outcome needs at least one test case")
    ids = [row.test_case_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("test case IDs must be unique within a program outcome")
    if any(row.problem_id != problem_id for row in rows):
        raise ValueError("nested test case problem IDs must match the program outcome")
    return ProgramOutcome(problem_id=problem_id, test_cases=rows)
