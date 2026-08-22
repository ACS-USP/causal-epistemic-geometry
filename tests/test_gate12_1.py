from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epistemic_geometry.experiments import gate12_1


def test_engineering_fixtures_are_frozen_nonsemantic_and_cover_lengths() -> None:
    fixtures = gate12_1.engineering_fixtures()
    assert len(fixtures) == 12
    assert len({row["fixture_sha256"] for row in fixtures}) == 12
    assert {len(row["continuation_token_ids"]) for row in fixtures} >= {1, 2, 8, 32, 64}
    assert all(row["source"] == "synthetic_non_benchmark_token_sequence" for row in fixtures)


def test_engineering_directions_are_unit_and_orthogonal() -> None:
    directions = gate12_1.engineering_directions()
    assert directions.shape == (2, 4096)
    assert np.linalg.norm(directions, axis=1) == pytest.approx([1.0, 1.0])
    assert abs(float(directions[0] @ directions[1])) <= 1e-12


def test_js_and_target_logp_identity() -> None:
    logits = np.array([[0.1, 0.2, -0.4], [0.3, -0.2, 0.7]])
    assert gate12_1.js_divergence(logits, logits) == pytest.approx([0.0, 0.0], abs=1e-15)
    values = gate12_1.target_logp(logits, np.array([1, 2]))
    assert values.shape == (2,)


def test_fisher_and_utility_match_explicit_forms() -> None:
    logits = np.array([0.3, -0.2, 1.1])
    derivative = np.array([1.0, -2.0, 0.5])
    p = np.exp(gate12_1.log_softmax64(logits))
    expected_q = np.sum(p * derivative**2) - np.sum(p * derivative) ** 2
    assert gate12_1.fisher_energy(logits, derivative) == pytest.approx(expected_q)
    assert gate12_1.utility_slope(logits, derivative, 1) == pytest.approx(
        derivative[1] - np.sum(p * derivative)
    )


def test_stable_window_requires_three_consecutive_scales() -> None:
    rows = []
    for index, epsilon in enumerate(gate12_1.EPSILONS):
        good = 2 <= index <= 4
        rows.append(
            {
                "epsilon": epsilon,
                "jvp_cosine": 0.9999 if good else 0.9,
                "fisher_relative_error": 0.01 if good else 0.2,
                "utility_relative_error": 0.01 if good else 0.2,
                "local_kl_relative_error": 0.01 if good else 0.2,
            }
        )
    result = gate12_1.stable_window(rows)
    assert result["pass"] is True
    assert result["three_consecutive_window"] == list(gate12_1.EPSILONS[2:5])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                semantic_bug_found=False,
                semantic_bug_repaired=False,
                fp32_sequence_pass=True,
                bf16_bridge_pass=True,
                derivative_pass=True,
            ),
            "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE_QUALIFIED",
        ),
        (
            dict(
                semantic_bug_found=True,
                semantic_bug_repaired=True,
                fp32_sequence_pass=True,
                bf16_bridge_pass=True,
                derivative_pass=True,
            ),
            "GATE12_1_SEQUENCE_SEMANTICS_BUG_REPAIRED_AND_ENGINE_QUALIFIED",
        ),
        (
            dict(
                semantic_bug_found=True,
                semantic_bug_repaired=False,
                fp32_sequence_pass=False,
                bf16_bridge_pass=False,
                derivative_pass=False,
            ),
            "GATE12_1_SEQUENCE_SEMANTICS_BUG_FOUND_NOT_REPAIRED",
        ),
        (
            dict(
                semantic_bug_found=False,
                semantic_bug_repaired=False,
                fp32_sequence_pass=True,
                bf16_bridge_pass=False,
                derivative_pass=True,
            ),
            "GATE12_1_FP32_ENGINE_QUALIFIED_BF16_BRIDGE_FAILED",
        ),
    ],
)
def test_classification(kwargs: dict[str, bool], expected: str) -> None:
    assert gate12_1.classify_qualification(**kwargs) == expected


def test_gate12_1_module_has_no_historical_outcome_imports() -> None:
    source = gate12_1.__file__
    assert source is not None
    text = Path(source).read_text(encoding="utf-8")
    forbidden = ("gate9", "gate10", "gate11", "journal.jsonl", "accuracy", "rescue")
    assert all(token not in text.lower() for token in forbidden)


def test_engineering_runner_has_no_scientific_item_or_outcome_path() -> None:
    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_gate12_1_continuous_geometry_engine.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        "gate9_selected",
        "gate10_cross",
        "gate11_domain",
        "control_validation_items",
        "utility_prediction_items",
        "journal.jsonl",
    )
    assert all(value not in runner.lower() for value in forbidden)
