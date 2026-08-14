"""Small benchmark interface with mechanical answer normalization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from epistemic_geometry.types import BenchmarkItem


class AnswerParser:
    """Normalize short exact-label outputs without fuzzy judging.

    The first non-empty line and first whitespace-delimited token are used. This
    keeps parsing deterministic for prompts that request labels such as ``A``.
    """

    def __init__(self, allowed_targets: set[str] | None = None) -> None:
        self.allowed_targets = (
            {target.upper() for target in allowed_targets} if allowed_targets else None
        )

    def normalize(self, raw_output: str) -> str:
        first_line = next((line.strip() for line in raw_output.splitlines() if line.strip()), "")
        token = first_line.split()[0] if first_line else ""
        return token.strip("[](){}.,:;\"'").upper()

    def validate(self, item: BenchmarkItem) -> None:
        if self.allowed_targets is not None and item.target.upper() not in self.allowed_targets:
            raise ValueError(
                f"Target {item.target!r} for {item.id!r} is not in allowed targets "
                f"{sorted(self.allowed_targets)}"
            )


class Benchmark(ABC):
    """Iterable collection of immutable, ground-truth benchmark items."""

    parser: AnswerParser

    @abstractmethod
    def items(self) -> list[BenchmarkItem]:
        """Return items in a stable order."""

    def __iter__(self) -> Iterator[BenchmarkItem]:
        return iter(self.items())

    def __len__(self) -> int:
        return len(self.items())
