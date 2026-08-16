"""Deterministic batching planner tests."""

from __future__ import annotations

import pytest

from epistemic_geometry.inference.planner import group_conditions_by_layer, plan_prepared_items
from epistemic_geometry.types import PreparedChoiceItem


def _item(item_id: str, length: int) -> PreparedChoiceItem:
    return PreparedChoiceItem(
        item_id=item_id,
        target="A",
        metadata={},
        rendered_prompt=item_id,
        rendered_prompt_hash=item_id,
        prompt_ids=tuple(range(length)),
        candidate_labels=("A", "B"),
        candidate_token_ids={"A": (1,), "B": (2,)},
        context_compatible_candidate_ids={"A": (1,), "B": (2,)},
        semantic_option_ids=(0, 1),
    )


def test_length_planner_is_deterministic_and_respects_both_budgets() -> None:
    items = [_item("z", 7), _item("a", 2), _item("b", 3), _item("c", 4)]
    plans_a, payload_a = plan_prepared_items(
        items, max_items=3, max_prefill_tokens=8
    )
    plans_b, payload_b = plan_prepared_items(
        list(reversed(items)), max_items=3, max_prefill_tokens=8
    )
    assert plans_a == plans_b
    assert payload_a == payload_b
    assert [plan.item_ids for plan in plans_a] == [("a", "b"), ("c",), ("z",)]
    assert all(plan.padded_token_cost <= 8 for plan in plans_a)


def test_length_planner_rejects_item_larger_than_token_budget() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        plan_prepared_items([_item("too-long", 9)], max_items=2, max_prefill_tokens=8)


def test_conditions_group_deterministically_by_layer() -> None:
    conditions = [
        ({"condition": "l2", "layer": 2}, None),
        ({"condition": "default"}, None),
        ({"condition": "l2b", "layer": 2}, None),
    ]
    grouped = group_conditions_by_layer(conditions, default_layer=1)
    assert list(grouped) == [2, 1]
    assert [spec["condition"] for spec, _ in grouped[2]] == ["l2", "l2b"]
