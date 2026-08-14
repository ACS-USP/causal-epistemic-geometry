"""Deterministic synthetic benchmark used for software validation only."""

from __future__ import annotations

from epistemic_geometry.benchmarks.base import AnswerParser, Benchmark
from epistemic_geometry.reproducibility import stable_seed
from epistemic_geometry.types import BenchmarkItem


class MockBenchmark(Benchmark):
    """Tiny exact-label benchmark with deterministic class metadata."""

    def __init__(self, n_items: int, seed: int, allowed_targets: list[str] | None = None) -> None:
        if n_items <= 0:
            raise ValueError("n_items must be positive")
        self.seed = seed
        self.allowed_targets = allowed_targets or ["A", "B", "C", "D"]
        if len(self.allowed_targets) < 2:
            raise ValueError("MockBenchmark needs at least two target labels")
        self.parser = AnswerParser(set(self.allowed_targets))
        self._items = [self._make_item(index) for index in range(n_items)]

    def _make_item(self, index: int) -> BenchmarkItem:
        label_index = stable_seed(self.seed, "target", index) % len(self.allowed_targets)
        target = self.allowed_targets[label_index]
        prompt = (
            f"Synthetic item {index}. Select the exact label for latent class {label_index}. "
            "Answer with one label only."
        )
        return BenchmarkItem(
            id=f"mock-{index:04d}",
            prompt=prompt,
            target=target,
            metadata={"class_index": label_index, "source": "deterministic_mock"},
        )

    def items(self) -> list[BenchmarkItem]:
        return list(self._items)

