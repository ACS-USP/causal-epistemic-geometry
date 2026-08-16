"""Deterministic option-order controls for Q1 V1.1."""

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
