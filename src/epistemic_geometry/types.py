"""Typed scientific records shared by benchmarks, backends, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkItem:
    """One ground-truth item; paired comparisons must preserve its ``id``."""

    id: str
    prompt: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Benchmark item id must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"Benchmark item {self.id!r} has an empty prompt")
        if not self.target.strip():
            raise ValueError(f"Benchmark item {self.id!r} has an empty target")


@dataclass(frozen=True)
class BackendOutput:
    """Raw backend response before benchmark-specific normalization."""

    raw_output: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """A normalized, mechanically scored prediction for one benchmark item."""

    item_id: str
    condition: str
    raw_output: str
    normalized_output: str
    target: str
    correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SteeringVector:
    """A reproducible intervention vector and its scientific provenance."""

    values: np.ndarray
    layer: int
    constructor: str
    normalization: str
    metadata: dict[str, Any] = field(default_factory=dict)
    hash: str = ""

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Steering vector values must be a non-empty 1-D array")
        if not np.isfinite(values).all():
            raise ValueError("Steering vector values must be finite")
        object.__setattr__(self, "values", values)

    @property
    def dimension(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class Intervention:
    """One temporary activation intervention applied during treatment inference."""

    layer: int
    alpha: float
    vector_id: str
    token_scope: str
    vector: SteeringVector = field(repr=False, compare=False, default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.layer < 0:
            raise ValueError("Intervention layer must be non-negative")
        if self.token_scope not in {"all_tokens", "last_token"}:
            raise ValueError("token_scope must be 'all_tokens' or 'last_token'")
        if self.vector is None:
            raise ValueError("Intervention requires a SteeringVector")


@dataclass
class ExperimentResult:
    """Paired predictions, descriptive metrics, and provenance for one run."""

    predictions: list[Prediction]
    metrics: dict[str, Any]
    provenance: dict[str, Any]

