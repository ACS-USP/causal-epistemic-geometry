from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from epistemic_geometry.analysis.cruxeval_provenance import (
    classify_item,
    deterministic_panel,
)
from epistemic_geometry.analysis.m3_qualification import (
    DIRECTION_COUNT,
    classify_m3,
    engineering_directions,
    engineering_fixtures,
    gram_geometry,
    stable_local_window,
    weighted_fisher_gram,
)


def test_m3_fixtures_are_nonsemantic_and_cover_contexts() -> None:
    fixtures = engineering_fixtures()
    assert len(fixtures) == 16
    assert len({row["fixture_sha256"] for row in fixtures}) == 16
    assert {len(row["continuation_token_ids"]) for row in fixtures} >= {1, 2, 8, 32, 64}
    assert all(row["no_task_oracle"] for row in fixtures)
    assert all(not row["semantic_correctness_available"] for row in fixtures)


def test_m3_directions_are_orthonormal() -> None:
    directions = engineering_directions()
    assert directions.shape == (DIRECTION_COUNT, 4096)
    assert directions @ directions.T == pytest.approx(np.eye(DIRECTION_COUNT), abs=1e-12)


def test_weighted_gram_matches_explicit_and_is_shift_invariant() -> None:
    p = np.asarray([0.2, 0.3, 0.5])
    rows = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.8, -0.4]])
    expected = rows @ (np.diag(p) - np.outer(p, p)) @ rows.T
    assert weighted_fisher_gram(rows, p) == pytest.approx(expected)
    shifted = rows + np.asarray([[7.0], [-3.0]])
    assert weighted_fisher_gram(shifted, p) == pytest.approx(expected)


def test_psd_geometry_and_polarization() -> None:
    rng = np.random.default_rng(11)
    tangents = rng.normal(size=(5, 13))
    p = rng.dirichlet(np.ones(13))
    gram = weighted_fisher_gram(tangents, p)
    geometry = gram_geometry(gram)
    assert geometry["minimum_eigenvalue"] >= -1e-12
    qsum = weighted_fisher_gram((tangents[0] + tangents[1])[None, :], p)[0, 0]
    assert (qsum - gram[0, 0] - gram[1, 1]) / 2 == pytest.approx(gram[0, 1])


def test_local_window_requires_three_consecutive_local_scales() -> None:
    rows = []
    for epsilon in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        good = epsilon in {0.03, 0.1, 0.3}
        rows.append(
            {
                "epsilon": epsilon,
                "jvp_cosine": 0.9999 if good else 0.9,
                "fisher_relative_error": 0.01 if good else 0.2,
                "kl_relative_error": 0.01 if good else 0.2,
                "hellinger_relative_error": 0.01 if good else 0.2,
                "js_relative_error": 0.01 if good else 0.2,
                "gram_relative_error": 0.01 if good else 0.2,
                "radius_relative_error": 0.01 if good else 0.2,
                "angle_max_abs_error": 0.01 if good else 0.2,
                "rms_logit_movement": 0.001,
            }
        )
    assert stable_local_window(rows) == [0.03, 0.1, 0.3]


def test_m3_classification_is_mechanical() -> None:
    assert (
        classify_m3(
            sequence_pass=True,
            derivative_pass=True,
            finite_window_pass=True,
            bf16_bridge_pass=True,
        )
        == "M3_DIRECTIONAL_ENGINE_QUALIFIED"
    )
    assert (
        classify_m3(
            sequence_pass=True,
            derivative_pass=True,
            finite_window_pass=True,
            bf16_bridge_pass=False,
        )
        == "M3_FP32_COHERENT_BF16_SURROGATE_NOT_QUALIFIED"
    )


def test_provenance_precedence_and_outcome_independent_panel() -> None:
    assert classify_item([])[0] == "A"
    assert classify_item([{"activation_only": True}])[0] == "B"
    assert classify_item([{"semantic_outcome_inspected": True}])[0] == "C"
    assert (
        classify_item([{"semantic_outcome_inspected": True}, {"q2_geometry_discovery": True}])[0]
        == "D"
    )
    rows = [
        {"item_id": "sample_2", "provenance_class": "C", "selection_rank": "b"},
        {"item_id": "sample_1", "provenance_class": "C", "selection_rank": "a"},
        {"item_id": "sample_3", "provenance_class": "D", "selection_rank": "0"},
    ]
    assert [row["item_id"] for row in deterministic_panel(rows, allowed_classes={"C"})] == [
        "sample_1",
        "sample_2",
    ]


def test_real_runner_has_no_scientific_outcome_or_generation_path() -> None:
    runner = (Path(__file__).resolve().parents[1] / "scripts/run_q2_m3_qualification.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "semantic_v3",
        "journal_qwen",
        "v2_common_panel_journal",
        "accuracy",
        "rescue",
        ".generate(",
    )
    assert all(value not in runner.lower() for value in forbidden)


def test_provenance_audit_does_not_read_outcome_values() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/audit_cruxeval_provenance.py"
    ).read_text(encoding="utf-8")
    forbidden = ("row['correct']", 'row["correct"]', "row['accuracy']", 'row["accuracy"]')
    assert all(value not in source.lower() for value in forbidden)
