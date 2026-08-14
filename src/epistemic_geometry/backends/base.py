"""Backend abstraction and configuration-driven backend construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from epistemic_geometry.config import BackendConfig, RunConfig
from epistemic_geometry.types import BackendOutput, BenchmarkItem, Intervention

if TYPE_CHECKING:
    from epistemic_geometry.types import SteeringVector


class OptionalDependencyError(RuntimeError):
    """Actionable error for an intentionally optional backend dependency."""


class ModelBackend(ABC):
    """Minimal interface shared by the synthetic and Transformers backends."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the intervention dimension."""

    @abstractmethod
    def predict(self, item: BenchmarkItem) -> BackendOutput:
        """Generate one raw answer under the currently active condition."""

    @abstractmethod
    def extract_activation(self, item: BenchmarkItem) -> object:
        """Extract one explicit representation using the backend policy."""

    @abstractmethod
    @contextmanager
    def steer(self, intervention: Intervention) -> Iterator[None]:
        """Temporarily activate an intervention and always clean it up."""
        yield


def build_backend(config: RunConfig) -> ModelBackend:
    """Build a backend without making optional imports mandatory for mock mode."""

    backend_config: BackendConfig = config.backend
    if backend_config.type == "mock":
        from .mock import MockBackend

        return MockBackend(
            seed=config.experiment.seed,
            hidden_size=backend_config.hidden_size,
            n_classes=backend_config.n_classes,
        )
    if backend_config.type == "huggingface":
        from .huggingface import HuggingFaceBackend

        return HuggingFaceBackend(backend_config)
    raise ValueError(f"Unsupported backend type: {backend_config.type}")


def validate_vector_dimension(vector: SteeringVector, backend: ModelBackend) -> None:
    """Fail before inference if a vector cannot be applied to this backend."""

    if vector.dimension != backend.hidden_size:
        raise ValueError(
            f"Steering vector dimension {vector.dimension} does not match "
            f"backend hidden size {backend.hidden_size}"
        )

