"""Deterministic length-bucket planning for prepared choice prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from epistemic_geometry.reproducibility import canonical_json, stable_digest
from epistemic_geometry.types import PreparedChoiceItem

PLANNER_VERSION = "prepared-choice-length-bucket-v1"


@dataclass(frozen=True)
class BatchPlan:
    """One deterministic item batch and its padding-cost accounting."""

    batch_index: int
    item_ids: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    max_prompt_length: int
    padded_token_cost: int


def _cost(items: list[PreparedChoiceItem]) -> int:
    return len(items) * max(item.prompt_length for item in items)


def plan_prepared_items(
    items: list[PreparedChoiceItem],
    *,
    max_items: int,
    max_prefill_tokens: int,
) -> tuple[list[BatchPlan], dict[str, Any]]:
    """Sort by length and create reproducible batches under two budgets.

    The plan only changes execution grouping.  Callers reconstruct canonical
    scientific output by item ID, so batch boundaries never enter the protocol.
    """

    if max_items <= 0 or max_prefill_tokens <= 0:
        raise ValueError("max_items and max_prefill_tokens must be positive")
    ordered = sorted(items, key=lambda item: (item.prompt_length, item.item_id))
    plans: list[BatchPlan] = []
    current: list[PreparedChoiceItem] = []
    for item in ordered:
        proposed = [*current, item]
        if current and (len(proposed) > max_items or _cost(proposed) > max_prefill_tokens):
            plans.append(
                BatchPlan(
                    batch_index=len(plans),
                    item_ids=tuple(row.item_id for row in current),
                    prompt_lengths=tuple(row.prompt_length for row in current),
                    max_prompt_length=max(row.prompt_length for row in current),
                    padded_token_cost=_cost(current),
                )
            )
            current = [item]
        else:
            current = proposed
        if _cost(current) > max_prefill_tokens:
            raise ValueError(
                f"Item {item.item_id} prompt length {item.prompt_length} exceeds the "
                "configured max_prefill_tokens budget"
            )
    if current:
        plans.append(
            BatchPlan(
                batch_index=len(plans),
                item_ids=tuple(row.item_id for row in current),
                prompt_lengths=tuple(row.prompt_length for row in current),
                max_prompt_length=max(row.prompt_length for row in current),
                padded_token_cost=_cost(current),
            )
        )
    payload = {
        "planner_version": PLANNER_VERSION,
        "max_items": max_items,
        "max_prefill_tokens": max_prefill_tokens,
        "input_item_count": len(items),
        "batches": [plan.__dict__ for plan in plans],
    }
    payload["plan_hash"] = stable_digest(canonical_json(payload))
    return plans, payload
