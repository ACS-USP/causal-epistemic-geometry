from __future__ import annotations

from epistemic_geometry.benchmarks.mmlu_pro import row_to_item
from epistemic_geometry.benchmarks.permutations import (
    deterministic_option_order,
    permute_mmlu_item,
)


def _item():
    return row_to_item(
        {
            "question_id": "q-1",
            "question": "Which option is correct?",
            "options": ["zero", "one", "two", "three"],
            "answer_index": 2,
            "category": "fixture",
        },
        "test",
    )


def test_option_order_is_sha256_derived_and_not_identity() -> None:
    first = deterministic_option_order("test:q-1", 4, 20260816, "permutation_0")
    second = deterministic_option_order("test:q-1", 4, 20260816, "permutation_0")
    assert first == second
    assert sorted(first) == [0, 1, 2, 3]
    assert first != [0, 1, 2, 3]


def test_permutation_remaps_target_and_preserves_semantic_identity() -> None:
    original = _item()
    permuted, manifest = permute_mmlu_item(original, 20260816, "permutation_2")
    order = manifest["option_order_original_indices"]
    assert permuted.id == original.id
    assert permuted.metadata["semantic_option_ids"] == order
    assert permuted.metadata["original_target_index"] == 2
    assert permuted.metadata["permuted_target_index"] == order.index(2)
    assert permuted.target == "ABCD"[order.index(2)]
    assert sorted(permuted.metadata["options"]) == sorted(original.metadata["options"])
