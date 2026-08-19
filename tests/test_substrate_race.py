from __future__ import annotations

from scripts.run_substrate_race import _resampling_metrics, _select_cells

from epistemic_geometry.benchmarks.external.base import ExternalResult, ExternalStatus


def _result(item_id: str, correct: bool, seed: int) -> ExternalResult:
    return ExternalResult(
        item_id=item_id,
        benchmark="test",
        subtask="test",
        rollout_seed=seed,
        raw_output="FINAL: x",
        parsed_answer="x" if correct else "y",
        status=ExternalStatus.VALID_CORRECT if correct else ExternalStatus.VALID_WRONG,
        correct=correct,
        reference_answer="x",
        evaluator="exact",
    )


def test_substrate_selection_uses_frozen_lexicographic_ranking() -> None:
    summaries = {
        "QWEN_NONTHINKING/FRESH_PSEUDOWORD_LONG": {
            "eligible_for_resampling": True,
            "valid_completion": 1.0,
            "correct": 8,
            "wrong": 4,
            "mechanical_failures": 0,
            "mean_tokens": 100.0,
        },
        "LLAMA_INSTRUCT/CRUXEVAL_SEMANTIC": {
            "eligible_for_resampling": True,
            "valid_completion": 1.0,
            "correct": 6,
            "wrong": 6,
            "mechanical_failures": 0,
            "mean_tokens": 200.0,
        },
        "QWEN_NONTHINKING/CRUXEVAL_SEMANTIC": {
            "eligible_for_resampling": True,
            "valid_completion": 0.95,
            "correct": 7,
            "wrong": 5,
            "mechanical_failures": 1,
            "mean_tokens": 50.0,
        },
    }
    assert _select_cells(summaries) == [
        "LLAMA_INSTRUCT/CRUXEVAL_SEMANTIC",
        "QWEN_NONTHINKING/FRESH_PSEUDOWORD_LONG",
    ]


def test_substrate_resampling_metrics_keep_undefined_phi_explicit() -> None:
    seed_a = [_result(str(index), True, 0) for index in range(4)]
    seed_b = [_result(str(index), True, 1) for index in range(4)]
    metrics = _resampling_metrics(seed_a, seed_b)
    assert metrics["pair_oracle_accuracy"] == 1.0
    assert metrics["error_jaccard"] == 1.0
    assert metrics["double_fault"] == 0.0
    assert metrics["error_phi"] is None
    assert metrics["error_phi_status"] == "undefined_zero_variance"


def test_substrate_resampling_metrics_expose_the_paired_2x2_table() -> None:
    seed_a = [
        _result("cc", True, 0),
        _result("cw", True, 0),
        _result("wc", False, 0),
        _result("ww", False, 0),
    ]
    seed_b = [
        _result("cc", True, 1),
        _result("cw", False, 1),
        _result("wc", True, 1),
        _result("ww", False, 1),
    ]
    metrics = _resampling_metrics(seed_a, seed_b)
    assert (metrics["n_cc"], metrics["n_cw"], metrics["n_wc"], metrics["n_ww"]) == (1, 1, 1, 1)
    assert metrics["pair_oracle_accuracy"] == 0.75
    assert metrics["resampling_gain_vs_mean_accuracy"] == 0.25
