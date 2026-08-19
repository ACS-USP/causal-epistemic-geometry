"""Offline tests for the prospective Gate 1 smoke contract."""

from __future__ import annotations

from epistemic_geometry.benchmarks.external.base import (
    ExternalItem,
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.benchmarks.external.gate1 import (
    classify_full_n20,
    stage1_technical_pass,
)
from epistemic_geometry.benchmarks.v4.character_count import (
    generate_full_nonthinking_smoke_manifest,
)


def _item(index: int) -> ExternalItem:
    return ExternalItem(
        item_id=f"gate-{index}",
        benchmark="test",
        subtask="test",
        prompt="Return a value",
        reference_answer="ok",
        evaluator="exact",
        source_revision="fixture",
    )


def _results(correct: int, wrong: int, invalid: int = 0):
    rows = []
    for index in range(correct):
        item = _item(index)
        rows.append(score_external_response(item, "FINAL: ok", rollout_seed=index))
    for index in range(correct, correct + wrong):
        item = _item(index)
        rows.append(score_external_response(item, "FINAL: no", rollout_seed=index))
    for index in range(correct + wrong, correct + wrong + invalid):
        item = _item(index)
        rows.append(score_external_response(item, "not final", rollout_seed=index))
    return rows


def test_gate1_character_manifest_is_fresh_and_deterministic() -> None:
    first = generate_full_nonthinking_smoke_manifest(seed=91)
    second = generate_full_nonthinking_smoke_manifest(seed=91)
    assert first == second
    assert len(first["items"]) == 20
    assert len({row["item_id"] for row in first["items"]}) == 20
    assert all(
        row["item_id"].startswith("gate1_full_nonthinking_charcount_")
        for row in first["items"]
    )
    assert all("Think as needed" not in row["prompt"] for row in first["items"])
    assert all(
        row["text"].count(row["target_character"]) == row["answer"]
        for row in first["items"]
    )


def test_stage1_is_technical_only() -> None:
    assert stage1_technical_pass(_results(0, 5)) is True
    assert stage1_technical_pass(_results(4, 0, 1)) is True
    assert stage1_technical_pass(_results(3, 0, 2)) is False


def test_full_n20_classification_is_frozen() -> None:
    assert classify_full_n20(_results(5, 5, 10)) == "MECHANICAL_OR_COMPLETION_FAILURE"
    assert classify_full_n20(_results(10, 8, 2)) == "PROMISING"
    assert classify_full_n20(_results(18, 1, 1)) == "SATURATED"
    assert classify_full_n20(_results(1, 17, 2)) == "FLOOR"


def test_full_n20_mechanical_failures_can_not_be_hidden_as_wrong() -> None:
    rows = _results(2, 2, 16)
    assert sum(row.status == ExternalStatus.VALID_WRONG for row in rows) == 2
    assert classify_full_n20(rows) == "MECHANICAL_OR_COMPLETION_FAILURE"
