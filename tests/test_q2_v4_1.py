from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q2_v4_1 import (
    EXPECTED_SAFE_IDS,
    V4_CLASSIFICATION,
    bank_geometry,
    binomial_safe_probability,
    load_frozen_candidates,
    reserve_fragility,
    selected_bank_coverage_checks,
)


def test_frozen_v4_bank_reconstructs_all_31_safe_rows() -> None:
    candidates, safe = load_frozen_candidates(
        "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json",
        "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json",
    )
    assert len(candidates) == 40
    assert len(safe) == 31
    assert [row["candidate_id"] for row in safe] == list(EXPECTED_SAFE_IDS)
    assert all(row["joint_safe"] for row in safe)
    assert V4_CLASSIFICATION == "Q2_V4_SAFE_BANK_INSUFFICIENT"


def test_geometry_reports_full_rank_and_known_frozen_spectrum() -> None:
    coefficients = np.asarray(
        [
            row["coefficients"]
            for row in load_frozen_candidates(
                "review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json",
                "review/q2_v4_spark1_presemantic/CANDIDATE_SAFETY_REPORT.json",
            )[1]
        ]
    )
    summary = bank_geometry(coefficients)
    assert summary["rank"] == 8
    assert summary["effective_rank"] >= 6.0
    assert summary["condition_number"] < 3.0
    assert summary["row_norm_max_error"] < 1e-12


def test_v4_selected_bank_gate_is_dimension_generalized_only_in_cardinality() -> None:
    coefficients = np.eye(8, dtype=np.float64)
    coefficients = np.vstack([coefficients, np.ones((23, 8), dtype=np.float64)])
    coefficients[8:] /= np.linalg.norm(coefficients[8:], axis=1, keepdims=True)
    amplitudes = np.ones((31, 2), dtype=np.float64)
    checks = selected_bank_coverage_checks(coefficients, amplitudes)
    assert checks["checks"]["selected_count_31"] is True
    assert checks["checks"]["full_subspace_rank"] is True
    assert checks["lineage"].endswith("32 to 31")


def test_reserve_probability_and_95_percent_recommendation_are_deterministic() -> None:
    assert 0.0 < binomial_safe_probability(40, 0.775) < 1.0
    rows = reserve_fragility()["rows"]
    assert [row["p_safe"] for row in rows] == [0.70, 0.75, 0.775, 0.80, 0.85, 0.90]
    assert all(row["minimum_candidates_for_95pct"] >= 32 for row in rows)
