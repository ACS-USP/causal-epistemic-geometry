"""Stable identity and rendering records for Q1 V3 reasoning tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from epistemic_geometry.reproducibility import canonical_json, stable_digest

SUITE_VERSION = "Q1-V3-REASONING-v1"
GENERATOR_VERSION = "q1-v3-reasoning-generators-v1"
FINAL_ANSWER_INSTRUCTION = (
    "Reason through the problem. End with exactly one machine-readable line "
    "in the form FINAL: <answer>."
)
Surface = Literal["canonical", "surface_twin"]


def latent_hash_for(
    family: str,
    cell: str,
    seed: int,
    spec: dict[str, Any],
    answer: int,
    difficulty: dict[str, Any],
) -> str:
    return stable_digest(
        SUITE_VERSION,
        GENERATOR_VERSION,
        family,
        cell,
        seed,
        canonical_json(spec),
        answer,
        canonical_json(difficulty),
    )


@dataclass(frozen=True)
class ReasoningItem:
    """One procedurally generated problem with an exact executable answer."""

    latent_id: str
    family: str
    cell: str
    latent_seed: int
    spec: dict[str, Any]
    answer: int
    difficulty: dict[str, Any]
    latent_hash: str = ""
    suite_version: str = SUITE_VERSION
    generator_version: str = GENERATOR_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.answer, int) or self.answer < 0:
            raise ValueError("reasoning answers must be non-negative exact integers")
        if self.suite_version != SUITE_VERSION:
            raise ValueError(f"unsupported reasoning suite version: {self.suite_version}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(f"unsupported reasoning generator version: {self.generator_version}")
        expected = latent_hash_for(
            self.family,
            self.cell,
            self.latent_seed,
            self.spec,
            self.answer,
            self.difficulty,
        )
        if self.latent_hash and self.latent_hash != expected:
            raise ValueError("latent_hash does not match the reasoning item identity")
        if not self.latent_hash:
            object.__setattr__(self, "latent_hash", expected)
        expected_id = f"{self.family}:{self.cell}:{self.latent_hash[:16]}"
        if self.latent_id != expected_id:
            raise ValueError(f"latent_id must be {expected_id!r}, got {self.latent_id!r}")

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "family": self.family,
            "cell": self.cell,
            "latent_seed": self.latent_seed,
            "spec": self.spec,
            "answer": self.answer,
            "difficulty": self.difficulty,
            "latent_hash": self.latent_hash,
            "suite_version": self.suite_version,
            "generator_version": self.generator_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ReasoningItem:
        return cls(
            latent_id=str(record["latent_id"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            latent_seed=int(record["latent_seed"]),
            spec=dict(record["spec"]),
            answer=int(record["answer"]),
            difficulty=dict(record["difficulty"]),
            latent_hash=str(record.get("latent_hash", "")),
            suite_version=str(record.get("suite_version", SUITE_VERSION)),
            generator_version=str(record.get("generator_version", GENERATOR_VERSION)),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class ReasoningView:
    """Canonical or deterministic surface-twin prompt for one latent item."""

    latent_id: str
    view_id: str
    family: str
    cell: str
    surface: Surface
    answer: int
    prompt: str
    prompt_hash: str
    template_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_view = f"{self.latent_id}:{self.surface}"
        if self.view_id != expected_view:
            raise ValueError(f"view_id must be {expected_view!r}")
        if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != self.prompt_hash:
            raise ValueError("prompt_hash does not match the reasoning prompt")

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "view_id": self.view_id,
            "family": self.family,
            "cell": self.cell,
            "surface": self.surface,
            "answer": self.answer,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "template_hash": self.template_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ReasoningView:
        return cls(
            latent_id=str(record["latent_id"]),
            view_id=str(record["view_id"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            surface=record["surface"],
            answer=int(record["answer"]),
            prompt=str(record["prompt"]),
            prompt_hash=str(record["prompt_hash"]),
            template_hash=str(record["template_hash"]),
            metadata=dict(record.get("metadata", {})),
        )
