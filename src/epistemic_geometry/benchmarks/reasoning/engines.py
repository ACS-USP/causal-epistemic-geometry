"""Execution-engine primitives for Q1 V3 reasoning rollouts.

The scientific unit remains one latent item, rollout identity, and reasoning
budget.  This module only describes how a physical trajectory can be reused;
it does not alter parsing or qualification rules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from epistemic_geometry.reproducibility import stable_digest
from epistemic_geometry.types import BackendOutput

SERIAL_REASONING_REFERENCE = "serial_reasoning_reference"
MAX_BUDGET_PREFIX_REUSE = "max_budget_prefix_reuse"
BATCHED_REASONING = "batched_reasoning"
REASONING_ENGINE_VERSION = "q1-v3-reasoning-engine-v1"
SUPPORTED_REASONING_ENGINES = frozenset(
    {SERIAL_REASONING_REFERENCE, MAX_BUDGET_PREFIX_REUSE, BATCHED_REASONING}
)


@dataclass(frozen=True)
class PrefixProvenance:
    """Identity of one scientific row derived from a physical generation."""

    physical_generation_id: str
    source_max_budget: int
    prefix_length: int
    derived_from_prefix: bool
    natural_completion_length: int


def physical_generation_id(
    *, view_id: str, sampling_seed: int, source_max_budget: int
) -> str:
    """Return a stable identity shared by all budget views of one trajectory."""

    return stable_digest(
        "Q1-V3-PHYSICAL-GENERATION",
        view_id,
        sampling_seed,
        source_max_budget,
    )


def prefix_provenance(
    *,
    view_id: str,
    sampling_seed: int,
    source_max_budget: int,
    requested_budget: int,
    natural_completion_length: int,
) -> PrefixProvenance:
    if source_max_budget <= 0 or requested_budget <= 0:
        raise ValueError("reasoning budgets must be positive")
    if requested_budget > source_max_budget:
        raise ValueError("requested budget cannot exceed source max budget")
    prefix_length = min(requested_budget, natural_completion_length)
    return PrefixProvenance(
        physical_generation_id=physical_generation_id(
            view_id=view_id,
            sampling_seed=sampling_seed,
            source_max_budget=source_max_budget,
        ),
        source_max_budget=source_max_budget,
        prefix_length=prefix_length,
        derived_from_prefix=requested_budget < source_max_budget,
        natural_completion_length=natural_completion_length,
    )


def derive_budget_outputs(
    output: BackendOutput,
    *,
    view_id: str,
    sampling_seed: int,
    source_max_budget: int,
    budgets: Iterable[int],
    decode_tokens: Callable[[tuple[int, ...]], str],
) -> dict[int, BackendOutput]:
    """Create independently parseable outputs from one max-budget trajectory.

    The parser is deliberately not called here.  Each returned output contains
    only the exact token prefix for its budget and is parsed independently by
    the canonical rollout conversion path.
    """

    raw_ids = tuple(int(token) for token in output.metadata.get("generated_token_ids", ()))
    if not raw_ids:
        raise ValueError("prefix reuse requires generated_token_ids in backend metadata")
    natural_length = len(raw_ids)
    requested = sorted({int(budget) for budget in budgets})
    if not requested or any(budget <= 0 for budget in requested):
        raise ValueError("prefix reuse requires positive budgets")
    if source_max_budget < max(requested):
        raise ValueError("source max budget is smaller than requested budget")

    result: dict[int, BackendOutput] = {}
    for budget in requested:
        provenance = prefix_provenance(
            view_id=view_id,
            sampling_seed=sampling_seed,
            source_max_budget=source_max_budget,
            requested_budget=budget,
            natural_completion_length=natural_length,
        )
        prefix_ids = raw_ids[: provenance.prefix_length]
        metadata: dict[str, Any] = dict(output.metadata)
        metadata.update(
            {
                "generated_token_ids": list(prefix_ids),
                "generated_token_count": len(prefix_ids),
                "physical_generation_id": provenance.physical_generation_id,
                "source_max_budget": provenance.source_max_budget,
                "prefix_length": provenance.prefix_length,
                "derived_from_prefix": provenance.derived_from_prefix,
                "natural_completion_length": provenance.natural_completion_length,
                "reasoning_budget": budget,
            }
        )
        result[budget] = BackendOutput(
            raw_output=decode_tokens(prefix_ids),
            metadata=metadata,
        )
    return result


def deterministic_length_batches(
    lengths: Iterable[tuple[str, int]],
    *,
    batch_size: int,
    max_padded_tokens: int,
) -> list[tuple[str, ...]]:
    """Plan stable length-aware batches without changing scientific identity."""

    if batch_size <= 0 or max_padded_tokens <= 0:
        raise ValueError("batch_size and max_padded_tokens must be positive")
    ordered = sorted(lengths, key=lambda row: (row[1], row[0]))
    batches: list[tuple[str, ...]] = []
    current: list[tuple[str, int]] = []
    for item_id, length in ordered:
        if length <= 0:
            raise ValueError("prompt lengths must be positive")
        if length > max_padded_tokens:
            raise ValueError(
                f"item {item_id} prompt length {length} exceeds max_padded_tokens"
            )
        candidate = current + [(item_id, length)]
        padded_cost = len(candidate) * max(row[1] for row in candidate)
        if current and (len(candidate) > batch_size or padded_cost > max_padded_tokens):
            batches.append(tuple(row[0] for row in current))
            current = [(item_id, length)]
        else:
            current = candidate
    if current:
        batches.append(tuple(row[0] for row in current))
    return batches
