"""Pure analysis helpers for the Q1 V3 stochastic reasoning agent."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def plurality_answer(answers: Iterable[T], *, baseline_answer: T | None = None) -> T:
    """Return exact-answer plurality with the frozen deterministic tie rule."""

    values = list(answers)
    if not values:
        raise ValueError("plurality requires at least one valid exact answer")
    counts = Counter(values)
    maximum = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == maximum]
    if baseline_answer in tied:
        return baseline_answer  # type: ignore[return-value]
    return min(tied, key=lambda answer: (type(answer).__name__, str(answer)))


def plurality_ensemble(agents: list[list[T]], *, baseline_index: int = 0) -> list[T]:
    """Aggregate one exact answer from each agent for every item."""

    if not agents or any(len(agent) != len(agents[0]) for agent in agents):
        raise ValueError("ensemble agents must be non-empty and aligned by item")
    if not 0 <= baseline_index < len(agents):
        raise ValueError("baseline_index is outside the ensemble")
    return [
        plurality_answer(
            [agent[index] for agent in agents], baseline_answer=agents[baseline_index][index]
        )
        for index in range(len(agents[0]))
    ]
