from __future__ import annotations

import json

import numpy as np

from epistemic_geometry.experiments import q1_second_task as q1s
from epistemic_geometry.experiments.gate6_3_v3 import audit_two_rollout_estimands


def _row(question: str, test_id: int, count: int = 1) -> dict[str, object]:
    tests = [
        {"input": str(index), "output": json.dumps(index), "testtype": "functional"}
        for index in range(count)
    ]
    return {
        "question_content": f"Return the integer for {question}",
        "question_id": question,
        "test_id": test_id,
        "starter_code": "def solve(x):\n    pass",
        "function_name": "solve",
        "difficulty": "easy",
        "test": json.dumps(tests),
    }


def test_livecodebench_normalization_and_safe_evaluator() -> None:
    item = q1s.normalize_livecodebench_row(_row("q0", 0), 0)
    assert item.item_id == "q0:0"
    assert "FINAL:" in item.prompt
    assert q1s.evaluate_livecodebench_output("FINAL: 0", "0")["correct"]
    assert q1s.evaluate_livecodebench_output("FINAL: false", "false")["correct"]
    assert q1s.evaluate_livecodebench_output('FINAL: "hello"', '"hello"')["correct"]
    assert q1s.evaluate_livecodebench_output("FINAL: [1, 2]", "[1, 2]")["correct"]
    unsafe = q1s.evaluate_livecodebench_output("FINAL: __import__('os').system('id')", "0")
    assert unsafe["commitment_valid"]
    assert not unsafe["semantic_evaluable"]
    assert not unsafe["correct"]


def test_normalizer_requires_exactly_one_test() -> None:
    row = _row("q0", 0, count=2)
    with np.testing.assert_raises(ValueError):
        q1s.normalize_livecodebench_row(row, 0)


def test_question_group_split_and_schedules_are_deterministic() -> None:
    items = []
    source_index = 0
    for question in range(300):
        item = q1s.normalize_livecodebench_row(_row(f"q{question}", 0), source_index)
        items.append(item)
        source_index += 1
    stage_a, stage_b, reserve = q1s.split_items(items)
    assert (len(stage_a), len(stage_b), len(reserve)) == (50, 150, 100)
    assert not ({item.question_id for item in stage_a} & {item.question_id for item in stage_b})
    schedule_a = q1s.build_schedule(
        stage_a,
        stage="STAGE_A",
        conditions=q1s.STAGE_A_CONDITIONS,
        rollouts=q1s.STAGE_A_ROLLOUTS,
    )
    schedule_b = q1s.build_schedule(
        stage_b,
        stage="STAGE_B",
        conditions=q1s.STAGE_B_CONDITIONS,
        rollouts=q1s.STAGE_B_ROLLOUTS,
    )
    assert len(schedule_a) == 200
    assert len(schedule_b) == 6600
    assert len({row["seed"] for row in schedule_a + schedule_b}) == 6800
    assert q1s.split_items(items) == (stage_a, stage_b, reserve)


def test_r2_reduces_exactly_to_canonical_estimands() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.integers(0, 2, size=(41, 2), dtype=np.int8)
    condition = rng.integers(0, 2, size=(41, 2), dtype=np.int8)
    expected = audit_two_rollout_estimands(baseline, condition)
    observed = q1s.r_rollout_estimands(baseline, condition)
    for metric in expected:
        assert np.isclose(observed[metric], expected[metric], atol=1e-12)


def test_r4_identities_and_split_halves() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.integers(0, 2, size=(100, 4), dtype=np.int8)
    condition = rng.integers(0, 2, size=(100, 4), dtype=np.int8)
    result = q1s.r_rollout_estimands(baseline, condition)
    assert np.isclose(
        result["rescue"] - result["damage"],
        result["accuracy_condition"] - result["accuracy_baseline"],
    )
    halves = q1s.split_half_estimands(baseline, condition)
    assert set(halves) == {"A", "B"}


def test_extended_null_bank_preserves_existing_and_is_deterministic() -> None:
    rng = np.random.default_rng(17)
    dimension = 64
    meaningful = rng.normal(size=dimension)
    meaningful /= np.linalg.norm(meaningful)
    bases = [meaningful]
    existing = {}
    for index in range(4):
        value = rng.normal(size=dimension)
        for basis in bases:
            value -= np.dot(value, basis) * basis
        value /= np.linalg.norm(value)
        existing[f"RANDOM_R{index}"] = value
        bases.append(value)
    pairs = rng.normal(size=(104, dimension))
    bank_a, metadata_a = q1s.build_extended_null_bank(meaningful, existing, pairs)
    bank_b, metadata_b = q1s.build_extended_null_bank(meaningful, existing, pairs)
    assert metadata_a == metadata_b
    assert tuple(bank_a) == q1s.RANDOM_NAMES
    for name in existing:
        assert np.array_equal(bank_a[name], existing[name])
    for name in q1s.NEW_RANDOM_NAMES:
        assert np.array_equal(bank_a[name], bank_b[name])
