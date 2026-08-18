"""Offline tests for external benchmark adapters and qualification metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from epistemic_geometry.benchmarks.external.adapters import adapter_for, candidate_specs
from epistemic_geometry.benchmarks.external.base import (
    ExternalStatus,
    score_external_response,
)
from epistemic_geometry.benchmarks.external.metrics import summarize_qualification

ROOT = Path(__file__).parents[1]
FIXTURES = {
    "RE2-Bench": ROOT / "examples/external_benchmark_fixtures/re2bench_output.jsonl",
    "LiveCodeBench": ROOT / "examples/external_benchmark_fixtures/livecodebench_test_output.jsonl",
    "CRUXEval": ROOT / "examples/external_benchmark_fixtures/cruxeval_output.jsonl",
    "LiveBench": ROOT / "examples/external_benchmark_fixtures/livebench_objective.jsonl",
}


@pytest.mark.parametrize("spec", candidate_specs())
def test_candidate_fixture_adapters_are_deterministic(spec) -> None:
    adapter = adapter_for(spec.name)
    first = adapter.load_items(FIXTURES[spec.name])
    second = adapter.load_items(FIXTURES[spec.name])
    assert [item.to_record() for item in first] == [item.to_record() for item in second]
    assert adapter.validate(first)["item_digest"] == adapter.validate(second)["item_digest"]


def test_external_parser_keeps_failure_categories_separate() -> None:
    item = adapter_for("LiveBench").load_items(FIXTURES["LiveBench"])[0]
    correct = score_external_response(
        item, f"analysis\nFINAL: {item.reference_answer}", rollout_seed=0
    )
    assert correct.status == ExternalStatus.VALID_CORRECT
    assert (
        score_external_response(item, "FINAL: wrong", rollout_seed=0).status
        == ExternalStatus.VALID_WRONG
    )
    assert (
        score_external_response(item, "The answer is probably 42", rollout_seed=0).status
        == ExternalStatus.INVALID_FORMAT
    )
    assert (
        score_external_response(item, "<think>unfinished", rollout_seed=0).status
        == ExternalStatus.TRUNCATED_THINKING
    )


def test_python_literal_evaluator_is_semantic_and_does_not_execute_code() -> None:
    item = adapter_for("CRUXEval").load_items(FIXTURES["CRUXEval"])[0]
    result = score_external_response(item, "FINAL: 5", rollout_seed=0)
    assert result.status == ExternalStatus.VALID_CORRECT
    malicious = score_external_response(
        item, "FINAL: __import__('os').system('false')", rollout_seed=0
    )
    assert malicious.status == ExternalStatus.INVALID_FORMAT


def test_two_seed_metrics_report_pair_counts_and_headroom() -> None:
    item = adapter_for("LiveBench").load_items(FIXTURES["LiveBench"])[0]
    results = [
        score_external_response(item, "FINAL: 42", rollout_seed=0),
        score_external_response(item, "FINAL: wrong", rollout_seed=1),
    ]
    summary = summarize_qualification(results)
    assert summary.valid_completion == 1.0
    assert summary.conditional_accuracy == 0.5
    assert summary.paired_counts == {"cw": 1}
    assert summary.pair_oracle_accuracy == 1.0
    assert summary.resampling_gain == 0.5


def test_unknown_evaluator_rejected() -> None:
    item = adapter_for("LiveBench").load_items(FIXTURES["LiveBench"])[0]
    object.__setattr__(item, "evaluator", "llm_judge")
    with pytest.raises(ValueError, match="unsupported deterministic evaluator"):
        score_external_response(item, "FINAL: anything", rollout_seed=0)


def test_completion_diagnostic_caps_are_fixed_and_not_a_scientific_tuning_grid() -> None:
    for spec in candidate_specs():
        assert spec.completion_diagnostic_caps == (8192, 16384, 32768)
