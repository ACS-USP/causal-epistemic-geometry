"""Typed raw rollout records and deterministic seed provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed
from epistemic_geometry.types import BackendOutput

from .base import ReasoningView
from .parser import ParsedFinal, parse_family_final


def generation_config_hash(config: dict[str, Any]) -> str:
    return stable_digest("Q1-V3-GENERATION-CONFIG", canonical_json(config))


def rollout_seed(
    base_seed: int,
    latent_id: str,
    intervention_id: str,
    rollout_index: int,
    *,
    regime: str,
) -> int:
    """Derive a seed without Python's randomized hash()."""

    if regime not in {"matched", "independent"}:
        raise ValueError("seed regime must be matched or independent")
    condition_key = "matched-common-random-number" if regime == "matched" else intervention_id
    return stable_seed("Q1-V3-ROLLOUT", regime, base_seed, latent_id, condition_key, rollout_index)


def seed_schedule(
    latent_ids: list[str],
    intervention_ids: list[str],
    *,
    base_seed: int,
    n_rollouts: int,
    regime: str,
) -> dict[tuple[str, str, int], int]:
    return {
        (latent_id, intervention_id, rollout): rollout_seed(
            base_seed,
            latent_id,
            intervention_id,
            rollout,
            regime=regime,
        )
        for latent_id in latent_ids
        for intervention_id in intervention_ids
        for rollout in range(n_rollouts)
    }


@dataclass(frozen=True)
class RolloutRecord:
    latent_id: str
    view_id: str
    family: str
    cell: str
    target: int
    intervention_id: str
    rollout_index: int
    sampling_seed: int
    raw_text: str
    parsed_answer: int | None
    parse_status: str
    correct: bool
    token_ids: tuple[int, ...] = ()
    stop_reason: str | None = None
    think_token_count: int | None = None
    final_answer_token_count: int | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parsed(
        cls,
        *,
        latent_id: str,
        view_id: str,
        family: str,
        cell: str,
        target: int,
        intervention_id: str,
        rollout_index: int,
        sampling_seed: int,
        raw_text: str,
        parsed: ParsedFinal,
        **kwargs: Any,
    ) -> RolloutRecord:
        return cls(
            latent_id=latent_id,
            view_id=view_id,
            family=family,
            cell=cell,
            target=target,
            intervention_id=intervention_id,
            rollout_index=rollout_index,
            sampling_seed=sampling_seed,
            raw_text=raw_text,
            parsed_answer=parsed.answer,
            parse_status=parsed.status,
            correct=parsed.valid and parsed.answer == target,
            **kwargs,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "latent_id": self.latent_id,
            "view_id": self.view_id,
            "family": self.family,
            "cell": self.cell,
            "target": self.target,
            "intervention_id": self.intervention_id,
            "rollout_index": self.rollout_index,
            "sampling_seed": self.sampling_seed,
            "raw_text": self.raw_text,
            "parsed_answer": self.parsed_answer,
            "parse_status": self.parse_status,
            "correct": self.correct,
            "token_ids": list(self.token_ids),
            "stop_reason": self.stop_reason,
            "think_token_count": self.think_token_count,
            "final_answer_token_count": self.final_answer_token_count,
            "generation_config": self.generation_config,
            "generation_config_hash": generation_config_hash(self.generation_config),
            "metadata": self.metadata,
        }


def rollout_record_from_output(
    view: ReasoningView,
    output: BackendOutput,
    *,
    intervention_id: str,
    rollout_index: int,
    sampling_seed: int,
    generation_config: dict[str, Any],
    truncated: bool = False,
) -> RolloutRecord:
    """Convert one backend response into the canonical scientific row.

    Parsing is deliberately performed after generation and before any
    aggregation.  A missing or malformed ``FINAL:`` field is retained as a
    parse failure and therefore counts as incorrect, while the raw reasoning
    text and backend provenance remain available for audit.
    """

    parsed = parse_family_final(output.raw_output, view.family, truncated=truncated)
    metadata = dict(output.metadata)
    token_ids = tuple(int(token) for token in metadata.pop("generated_token_ids", ()))
    think_count = metadata.pop("think_token_count", None)
    final_count = metadata.pop("final_answer_token_count", None)
    stop_reason = metadata.pop("stop_reason", None)
    return RolloutRecord.from_parsed(
        latent_id=view.latent_id,
        view_id=view.view_id,
        family=view.family,
        cell=view.cell,
        target=view.answer,
        intervention_id=intervention_id,
        rollout_index=rollout_index,
        sampling_seed=sampling_seed,
        raw_text=output.raw_output,
        parsed=parsed,
        token_ids=token_ids,
        stop_reason=stop_reason,
        think_token_count=int(think_count) if think_count is not None else None,
        final_answer_token_count=int(final_count) if final_count is not None else None,
        generation_config=dict(generation_config),
        metadata=metadata,
    )
