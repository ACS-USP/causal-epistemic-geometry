from __future__ import annotations

import json

import pytest

from epistemic_geometry.experiments.q1_second_task_stage_a_failure import (
    evaluate_livecodebench_output_stage_a2,
)
from epistemic_geometry.experiments.q1_second_task_stage_b_forensic import independent_score


def row(
    raw: str,
    reference: object,
    *,
    truncated: bool = False,
    token_ids: list[int] | None = None,
) -> dict:
    return {
        "raw_output": raw,
        "reference_answer": json.dumps(reference),
        "generated_token_ids": token_ids if token_ids is not None else list(range(256)),
        "truncated": truncated,
    }


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        (row("FINAL: 1", 1), (True, True, True)),
        (row("FINAL ANSWER: 1", 1), (True, True, True)),
        (row("FINAL ANSWER:\nFINAL: 1", 1), (True, True, True)),
        (row("FINAL ANSWER:\n# FINAL:\nFINAL: 1", 1), (True, True, True)),
        (row("**FINAL: 1**", 1), (True, True, True)),
        (row("FINAL ANSWER:\n```python\n1\n```", 1), (True, True, True)),
        (row("FINAL: `1`", 1), (True, True, True)),
        (row("FINAL ANSWER:\n1\nFINAL: 1", 1), (True, True, True)),
        (row("FINAL ANSWER:\n2\nFINAL: 1", 1), (True, False, False)),
        (row("<think>2</think>\nFINAL: 1", 1), (True, True, True)),
        (row("<think>unfinished\nFINAL: 1", 1), (False, False, False)),
        (row("FINAL ANSWER:\n[1, [2, 3]]\nFINAL: [1, [2, 3]]", [1, [2, 3]]), (True, True, True)),
        (row("FINAL: True", 1), (True, True, False)),
        (row("FINAL: 'answer'", "answer"), (True, True, True)),
        (row("FINAL: [1", [1]), (True, False, False)),
        (row("FINAL: true", True), (True, True, True)),
        (row("FINAL: 'python-string'", "python-string"), (True, True, True)),
        (row("FINAL: 1", 1, truncated=True), (False, False, False)),
        (row("# FINAL:\nFINAL: 1", 1), (True, True, True)),
        (row("FINAL ANSWER:\nFINAL: 1\nFINAL: 1", 1), (False, False, False)),
        (row("FINAL ANSWER:\nprose\n☑ FINAL: 1", 1), (True, False, False)),
    ],
)
def test_independent_parser_matches_primary_on_normative_fixtures(
    fixture: dict, expected: tuple[bool, bool, bool]
) -> None:
    primary = evaluate_livecodebench_output_stage_a2(
        fixture["raw_output"],
        fixture["reference_answer"],
        fixture["generated_token_ids"],
        truncated=fixture["truncated"],
    )
    audit = independent_score(fixture)
    observed = (
        bool(primary["commitment_valid"]),
        bool(primary["semantic_evaluable"]),
        bool(primary["correct"]),
    )
    assert observed == expected
    assert audit == {
        "commitment_valid": expected[0],
        "semantic_evaluable": expected[1],
        "correct": expected[2],
    }


def test_repair_runs_after_direct_commitment_is_not_parseable() -> None:
    fixture = row("FINAL ANSWER:\nintermediate prose\nFINAL: 7", 7)
    primary = evaluate_livecodebench_output_stage_a2(
        fixture["raw_output"], fixture["reference_answer"], fixture["generated_token_ids"]
    )
    assert primary["parser_repair_applied"] is True
    assert independent_score(fixture) == {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": True,
    }


def test_unsupported_tuple_is_not_a_competing_repair_literal() -> None:
    fixture = row("FINAL ANSWER:\n(1, 2)\nFINAL: 7", 7)
    assert independent_score(fixture) == {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": True,
    }


def test_mechanical_repetition_blocks_repair_but_not_a_valid_direct_final() -> None:
    repetitive = [1] * 256
    repaired = row(
        "FINAL ANSWER:\nintermediate prose\nFINAL: 7", 7, token_ids=repetitive
    )
    direct = row("FINAL: 7", 7, token_ids=repetitive)
    assert independent_score(repaired) == {
        "commitment_valid": True,
        "semantic_evaluable": False,
        "correct": False,
    }
    assert independent_score(direct) == {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": True,
    }


def test_direct_type_mismatch_is_evaluable_and_wrong() -> None:
    assert independent_score(row("FINAL: [1]", 1)) == {
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": False,
    }
