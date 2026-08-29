from __future__ import annotations

import json

from epistemic_geometry.experiments import q1_second_task as base
from epistemic_geometry.experiments import q1_second_task_hierarchical as h


def _row(question: str, test_id: int) -> dict[str, object]:
    return {
        "question_content": f"Return the integer for {question}",
        "question_id": question,
        "test_id": test_id,
        "starter_code": "def solve(x):\n    pass",
        "function_name": "solve",
        "difficulty": "easy",
        "test": json.dumps(
            [{"input": str(test_id), "output": str(test_id), "testtype": "functional"}]
        ),
    }


def _pool() -> list[base.LiveCodeBenchItem]:
    items = []
    source_index = 0
    for family in range(182):
        size = 2 if family < 105 else 3
        if family == 181:
            size = 4
        for test_id in range(size):
            items.append(base.normalize_livecodebench_row(_row(str(family), test_id), source_index))
            source_index += 1
    return items


def test_family_split_is_deterministic_and_one_row_per_family() -> None:
    first = h.split_families(_pool())
    second = h.split_families(_pool())
    assert first == second
    stage_a, stage_b, reserve = first
    assert (len(stage_a), len(stage_b), len(reserve)) == (32, 130, 20)
    assert len({item.question_id for item in stage_a + stage_b}) == 162
    assert not ({item.question_id for item in stage_a} & {item.question_id for item in stage_b})


def test_representative_selection_is_order_invariant() -> None:
    rows = [base.normalize_livecodebench_row(_row("x", value), value) for value in range(4)]
    assert h.representative_row("x", rows) == h.representative_row("x", list(reversed(rows)))


def test_amended_schedules_have_unique_family_keys_and_seeds() -> None:
    stage_a, stage_b, _reserve = h.split_families(_pool())
    schedule_a = h.build_schedule(
        stage_a,
        stage="STAGE_A",
        conditions=h.STAGE_A_CONDITIONS,
        rollouts=h.STAGE_A_ROLLOUTS,
    )
    schedule_b = h.build_schedule(
        stage_b,
        stage="STAGE_B",
        conditions=h.STAGE_B_CONDITIONS,
        rollouts=h.STAGE_B_ROLLOUTS,
    )
    assert len(schedule_a) == 128
    assert len(schedule_b) == 5720
    assert len({row["seed"] for row in schedule_a + schedule_b}) == 5848
