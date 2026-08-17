"""Typed identity and rendering types for the E3-10 instrument."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal

from epistemic_geometry.reproducibility import canonical_json, stable_digest

DIGITS = tuple(range(10))
NUMBER_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
SUITE_VERSION = "E3-10-v1"
GENERATOR_VERSION = "e3-generators-v2"
DECIMAL_ANSWER_INSTRUCTION = (
    "Return the final value as exactly one decimal digit from 0 through 9. Do not explain."
)
NUMBER_WORD_ANSWER_INSTRUCTION = (
    "Return the final value as exactly one number word from zero through nine. Do not explain."
)
ResponseChannel = Literal["decimal", "number_word"]
Surface = Literal["canonical", "surface_twin"]


@dataclass(frozen=True)
class LatentItem:
    """A procedurally generated problem and its exact executable target."""

    latent_id: str
    family: str
    cell: str
    latent_seed: int
    spec: dict[str, Any]
    target: int
    difficulty: dict[str, Any]
    latent_hash: str = ""
    suite_version: str = SUITE_VERSION
    generator_version: str = GENERATOR_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target not in DIGITS:
            raise ValueError(f"E3-10 target must be a digit 0..9, got {self.target!r}")
        expected = latent_hash_for(
            self.family,
            self.cell,
            self.latent_seed,
            self.spec,
            self.target,
            self.difficulty,
            suite_version=self.suite_version,
            generator_version=self.generator_version,
        )
        if self.latent_hash and self.latent_hash != expected:
            raise ValueError("latent_hash does not match the serialized latent identity")
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
            "target": self.target,
            "difficulty": self.difficulty,
            "latent_hash": self.latent_hash,
            "suite_version": self.suite_version,
            "generator_version": self.generator_version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> LatentItem:
        return cls(
            latent_id=str(record["latent_id"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            latent_seed=int(record["latent_seed"]),
            spec=dict(record["spec"]),
            target=int(record["target"]),
            difficulty=dict(record["difficulty"]),
            latent_hash=str(record.get("latent_hash", "")),
            suite_version=str(record.get("suite_version", SUITE_VERSION)),
            generator_version=str(record.get("generator_version", GENERATOR_VERSION)),
            metadata=dict(record.get("metadata", {})),
        )


@dataclass(frozen=True)
class RenderedView:
    """A deterministic surface/channel view of one latent item."""

    latent_id: str
    view_id: str
    family: str
    cell: str
    surface: Surface
    response_channel: ResponseChannel
    target: int
    target_text: str
    prompt: str
    prompt_hash: str
    template_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target not in DIGITS:
            raise ValueError("rendered target must be in 0..9")
        if self.response_channel == "decimal" and self.target_text != str(self.target):
            raise ValueError("decimal target_text must be the target digit")
        if self.response_channel == "number_word" and self.target_text != NUMBER_WORDS[self.target]:
            raise ValueError("number-word target_text must match the target digit")
        expected_view = f"{self.latent_id}:{self.surface}:{self.response_channel}"
        if self.view_id != expected_view:
            raise ValueError(f"view_id must be {expected_view!r}")
        if self.prompt_hash != hashlib.sha256(self.prompt.encode("utf-8")).hexdigest():
            raise ValueError("prompt_hash does not match prompt")

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "view_id": self.view_id,
            "family": self.family,
            "cell": self.cell,
            "surface": self.surface,
            "response_channel": self.response_channel,
            "target": self.target,
            "target_text": self.target_text,
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "template_hash": self.template_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> RenderedView:
        return cls(
            latent_id=str(record["latent_id"]),
            view_id=str(record["view_id"]),
            family=str(record["family"]),
            cell=str(record["cell"]),
            surface=record["surface"],
            response_channel=record["response_channel"],
            target=int(record["target"]),
            target_text=str(record["target_text"]),
            prompt=str(record["prompt"]),
            prompt_hash=str(record["prompt_hash"]),
            template_hash=str(record["template_hash"]),
            metadata=dict(record.get("metadata", {})),
        )


def latent_id_for(family: str, cell: str, latent_hash: str) -> str:
    """Construct the stable human-readable identity for a latent instance."""

    return f"{family}:{cell}:{latent_hash[:16]}"


def latent_hash_for(
    family: str,
    cell: str,
    latent_seed: int,
    spec: dict[str, Any],
    target: int,
    difficulty: dict[str, Any],
    *,
    suite_version: str = SUITE_VERSION,
    generator_version: str = GENERATOR_VERSION,
) -> str:
    """Hash all scientific latent fields, excluding presentation views."""

    return stable_digest(
        suite_version,
        generator_version,
        family,
        cell,
        latent_seed,
        canonical_json(spec),
        target,
        canonical_json(difficulty),
    )
