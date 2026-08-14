"""Mathematically meaningful deterministic representation-space mock backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

from epistemic_geometry.backends.base import ModelBackend, validate_vector_dimension
from epistemic_geometry.reproducibility import stable_seed
from epistemic_geometry.steering.vector import with_computed_hash
from epistemic_geometry.types import BackendOutput, BenchmarkItem, Intervention, SteeringVector


class MockBackend(ModelBackend):
    """A miniature frozen classifier with literal ``h' = h + alpha * v``.

    This backend is a software fixture, not evidence. Each item representation
    is a class prototype plus deterministic item-specific noise, then a fixed
    linear readout produces a label. Steering changes the representation before
    the readout, so tests exercise the causal intervention path rather than a
    table of pre-written answers.
    """

    labels = ("A", "B", "C", "D")

    def __init__(self, seed: int, hidden_size: int = 8, n_classes: int = 4) -> None:
        if n_classes != len(self.labels):
            raise ValueError("MockBackend currently uses exactly four labels A-D")
        if hidden_size < n_classes:
            raise ValueError("MockBackend hidden_size must be at least n_classes")
        self.seed = seed
        self._hidden_size = hidden_size
        self.n_classes = n_classes
        self._active_intervention: Intervention | None = None
        self._readout = np.zeros((n_classes, hidden_size), dtype=np.float64)
        self._readout[:, :n_classes] = np.eye(n_classes) * 1.35
        self._readout[:, n_classes:] = 0.08

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def _representation(self, item: BenchmarkItem) -> np.ndarray:
        class_index = item.metadata.get("class_index")
        if not isinstance(class_index, int):
            class_index = self.labels.index(item.target) if item.target in self.labels else 0
        prototype = np.zeros(self._hidden_size, dtype=np.float64)
        prototype[class_index % self.n_classes] = 2.0
        rng = np.random.default_rng(stable_seed(self.seed, "representation", item.id))
        noise = rng.normal(loc=0.0, scale=0.82, size=self._hidden_size)
        return prototype + noise

    def activation_for(
        self, item: BenchmarkItem, intervention: Intervention | None = None
    ) -> np.ndarray:
        """Return the exact representation before the linear readout."""

        activation = self._representation(item)
        active = intervention if intervention is not None else self._active_intervention
        if active is not None:
            validate_vector_dimension(active.vector, self)
            activation = activation + active.alpha * active.vector.values
        return activation

    def predict(self, item: BenchmarkItem) -> BackendOutput:
        activation = self.activation_for(item)
        logits = self._readout @ activation
        index = int(np.argmax(logits))
        return BackendOutput(
            raw_output=self.labels[index],
            metadata={
                "logits": logits.tolist(),
                "activation_norm": float(np.linalg.norm(activation)),
                "intervened": self._active_intervention is not None,
            },
        )

    def extract_activation(self, item: BenchmarkItem) -> np.ndarray:
        """Extract the base representation, never the active treatment state."""

        return self._representation(item).copy()

    @contextmanager
    def steer(self, intervention: Intervention) -> Iterator[None]:
        validate_vector_dimension(intervention.vector, self)
        previous = self._active_intervention
        self._active_intervention = intervention
        try:
            yield
        finally:
            self._active_intervention = previous

    def fixture_vector(self, kind: str, seed: int | None = None) -> SteeringVector:
        """Return named mock controls for pipeline tests; never scientific evidence."""

        if kind == "random":
            rng = np.random.default_rng(stable_seed(self.seed, "fixture", seed or 0))
            values = rng.normal(size=self.hidden_size)
        elif kind == "useful":
            values = np.zeros(self.hidden_size)
            values[:4] = np.array([0.22, -0.08, 0.15, -0.12])
        elif kind == "destructive":
            values = np.zeros(self.hidden_size)
            values[:4] = np.array([3.5, -3.5, 3.5, -3.5])
        else:
            raise ValueError("Mock fixture kind must be random, useful, or destructive")
        norm = np.linalg.norm(values)
        values = values / norm if norm else values
        return with_computed_hash(
            SteeringVector(
                values=values,
                layer=0,
                constructor=f"mock_fixture:{kind}",
                normalization="unit",
                metadata={"seed": seed, "software_fixture": True},
            )
        )
