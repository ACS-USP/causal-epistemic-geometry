"""Deterministic option-order controls for Q1 V1.1 and V1.2."""

from __future__ import annotations

import hashlib
from typing import Any

from epistemic_geometry.benchmarks.mmlu_pro import LABELS, render_mmlu_pro_question
from epistemic_geometry.types import BenchmarkItem


def _permutation_key(seed: int, item_id: str, permutation_id: str, index: int) -> str:
    payload = f"{seed}:{item_id}:{permutation_id}:{index}".encode()
    return hashlib.sha256(payload).hexdigest()


def deterministic_option_order(
    item_id: str,
    option_count: int,
    protocol_seed: int,
    permutation_id: str,
) -> list[int]:
    """Return a stable non-identity permutation of original option indices."""

    if option_count < 2 or option_count > len(LABELS):
        raise ValueError("Option permutations require between 2 and 10 options")
    order = sorted(
        range(option_count),
        key=lambda index: _permutation_key(protocol_seed, item_id, permutation_id, index),
    )
    if order == list(range(option_count)):
        order[0], order[1] = order[1], order[0]
    return order


def _question_from_rendered_prompt(prompt: str) -> str:
    marker = "Question:\n"
    if marker not in prompt:
        raise ValueError("MMLU-Pro prompt lacks the fixed Question marker")
    question = prompt.split(marker, 1)[1]
    option_marker = "\n\nA."
    if option_marker not in question:
        raise ValueError("MMLU-Pro prompt lacks the fixed option marker")
    return question.split(option_marker, 1)[0].strip()


def permute_mmlu_item(
    item: BenchmarkItem,
    protocol_seed: int,
    permutation_id: str,
) -> tuple[BenchmarkItem, dict[str, Any]]:
    """Permute option contents while preserving semantic target identity."""

    options = item.metadata.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"Item {item.id} lacks MMLU-Pro option metadata")
    options = [str(option) for option in options]
    order = deterministic_option_order(item.id, len(options), protocol_seed, permutation_id)
    original_target_index = int(item.metadata.get("answer_index", LABELS.index(item.target)))
    if original_target_index not in order:
        raise ValueError(f"Target index {original_target_index} is not in option order")
    permuted_target_index = order.index(original_target_index)
    question = _question_from_rendered_prompt(item.prompt)
    permuted_options = [options[index] for index in order]
    metadata = dict(item.metadata)
    metadata.update(
        {
            "permutation_id": permutation_id,
            "permutation_seed": protocol_seed,
            "original_item_id": item.id,
            "option_order_original_indices": order,
            "semantic_option_ids": order,
            "original_options": options,
            "options": permuted_options,
            "original_target_index": original_target_index,
            "permuted_target_index": permuted_target_index,
        }
    )
    permuted = BenchmarkItem(
        id=item.id,
        prompt=render_mmlu_pro_question(question, permuted_options),
        target=LABELS[permuted_target_index],
        metadata=metadata,
    )
    manifest = {
        "permutation_id": permutation_id,
        "protocol_seed": protocol_seed,
        "item_id": item.id,
        "option_count": len(options),
        "option_order_original_indices": order,
        "original_target_index": original_target_index,
        "permuted_target_index": permuted_target_index,
        "target_label": permuted.target,
    }
    return permuted, manifest


def permute_mmlu_items(
    items: list[BenchmarkItem],
    protocol_seed: int,
    permutation_id: str,
) -> tuple[list[BenchmarkItem], list[dict[str, Any]]]:
    """Apply one deterministic permutation family to a stable item list."""

    permuted: list[BenchmarkItem] = []
    manifest: list[dict[str, Any]] = []
    for item in items:
        changed, record = permute_mmlu_item(item, protocol_seed, permutation_id)
        permuted.append(changed)
        manifest.append(record)
    return permuted, manifest


def cyclic_option_order(option_count: int, shift: int) -> list[int]:
    """Return the V1.2 balanced order for one cyclic displayed-slot shift.

    Semantic option ``j`` is assigned to displayed slot ``(j + shift) mod K``.
    Consequently the semantic ID at displayed slot ``s`` is ``(s-shift) mod
    K``. Shift zero is exactly the original ordering.
    """

    if option_count < 2 or option_count > len(LABELS):
        raise ValueError("Cyclic option order requires between 2 and 10 options")
    if shift < 0 or shift >= option_count:
        raise ValueError("Cyclic shift must satisfy 0 <= shift < option_count")
    return [(slot - shift) % option_count for slot in range(option_count)]


def cyclic_mmlu_item(item: BenchmarkItem, shift: int) -> tuple[BenchmarkItem, dict[str, Any]]:
    """Apply one deterministic V1.2 cyclic ordering without randomization."""

    options = item.metadata.get("options")
    if not isinstance(options, list) or not options:
        raise ValueError(f"Item {item.id} lacks MMLU-Pro option metadata")
    options = [str(option) for option in options]
    order = cyclic_option_order(len(options), shift)
    original_target_index = int(item.metadata.get("answer_index", LABELS.index(item.target)))
    permuted_target_index = order.index(original_target_index)
    question = _question_from_rendered_prompt(item.prompt)
    permuted_options = [options[index] for index in order]
    metadata = dict(item.metadata)
    metadata.update(
        {
            "cyclic_shift": shift,
            "cyclic_ordering_convention": "semantic_j_to_displayed_(j_plus_r)_mod_K",
            "original_item_id": item.id,
            "option_order_original_indices": order,
            "semantic_option_ids": order,
            "original_options": options,
            "options": permuted_options,
            "original_target_index": original_target_index,
            "permuted_target_index": permuted_target_index,
        }
    )
    cycled = BenchmarkItem(
        id=item.id,
        prompt=render_mmlu_pro_question(question, permuted_options),
        target=LABELS[permuted_target_index],
        metadata=metadata,
    )
    manifest = {
        "cyclic_shift": shift,
        "item_id": item.id,
        "option_count": len(options),
        "option_order_original_indices": order,
        "original_target_index": original_target_index,
        "permuted_target_index": permuted_target_index,
        "target_label": cycled.target,
    }
    return cycled, manifest


def cyclic_mmlu_items(
    items: list[BenchmarkItem], shift: int
) -> tuple[list[BenchmarkItem], list[dict[str, Any]]]:
    """Apply one V1.2 cyclic shift to a stable item list."""

    cycled: list[BenchmarkItem] = []
    manifest: list[dict[str, Any]] = []
    for item in items:
        changed, record = cyclic_mmlu_item(item, shift)
        cycled.append(changed)
        manifest.append(record)
    return cycled, manifest


def validate_cyclic_balance(items: list[BenchmarkItem]) -> dict[str, Any]:
    """Assert exact per-item Latin-cycle balance and return an audit report."""

    if not items:
        raise ValueError("Cannot validate cyclic balance for an empty item list")
    records: list[dict[str, Any]] = []
    for item in items:
        options = item.metadata.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError(f"Item {item.id} lacks options for cyclic balance")
        option_count = len(options)
        orders = [cyclic_option_order(option_count, shift) for shift in range(option_count)]
        visits = {
            semantic: [order.index(semantic) for order in orders]
            for semantic in range(option_count)
        }
        if any(sorted(slots) != list(range(option_count)) for slots in visits.values()):
            raise ValueError(f"Cyclic balance failed for item {item.id}")
        original_target = int(item.metadata.get("answer_index", LABELS.index(item.target)))
        target_slots = [order.index(original_target) for order in orders]
        if sorted(target_slots) != list(range(option_count)):
            raise ValueError(f"Target balance failed for item {item.id}")
        records.append(
            {
                "item_id": item.id,
                "option_count": option_count,
                "cyclic_orderings": option_count,
                "every_semantic_option_visits_every_slot": True,
                "target_semantic_identity_invariant": True,
            }
        )
    return {
        "status": "PASS",
        "item_count": len(items),
        "records": records,
        "convention": "semantic_j_to_displayed_(j_plus_r)_mod_K",
    }
