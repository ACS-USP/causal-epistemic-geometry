from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.experiments import q1_confirmatory as q1


def test_schedule_is_complete_distinct_and_deterministic() -> None:
    items = [f"item_{index}" for index in range(57)]
    first = q1.build_schedule(items, model_role="Qwen")
    second = q1.build_schedule(items, model_role="Qwen")
    other = q1.build_schedule(items, model_role="Ministral")
    assert first == second
    assert len(first) == 798
    assert len({row["seed"] for row in first + other}) == 1596
    assert {row["condition"] for row in first} == set(q1.CONDITIONS)


def test_null_bank_is_stable_orthogonal_and_source_matched() -> None:
    rng = np.random.default_rng(42)
    meaningful = rng.normal(size=64)
    meaningful /= np.linalg.norm(meaningful)
    paired = rng.normal(size=(20, 64))
    first, metadata = q1.build_null_bank(meaningful, paired, model_role="fixture")
    second, second_metadata = q1.build_null_bank(meaningful, paired, model_role="fixture")
    assert metadata == second_metadata
    assert all(np.array_equal(first[name], second[name]) for name in q1.RANDOM_NAMES)
    matrix = np.stack([meaningful, *first.values()])
    assert np.allclose(matrix @ matrix.T, np.eye(5), atol=1e-6, rtol=0)
    assert metadata["records"]["RANDOM_R0"]["kind"] == "ISOTROPIC"
    assert metadata["records"]["RANDOM_R2"]["kind"] == (
        "CONSTRUCTION_MATCHED_SIGN_SHUFFLED"
    )


def test_resume_rejects_duplicate_or_mixed_source_rows() -> None:
    row = {
        "model_role": "Qwen",
        "item_id": "item",
        "condition": "BASELINE",
        "rollout_index": 0,
        "confirmatory_source_commit": "abc",
    }
    assert len(q1.completed_keys([row], source_commit="abc")) == 1
    with pytest.raises(RuntimeError, match="duplicate"):
        q1.completed_keys([row, row], source_commit="abc")
    with pytest.raises(RuntimeError, match="source commits"):
        q1.completed_keys([row], source_commit="def")


def test_cross_model_classification_is_conjunctive() -> None:
    assert q1.cross_model_classification(True, True) == "Q1_CONFIRMATORY_CROSS_MODEL_PASS"
    assert q1.cross_model_classification(True, False) == (
        "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL"
    )
    assert q1.cross_model_classification(False, True) == (
        "Q1_CONFIRMATORY_QWEN_FAIL_MINISTRAL_PASS"
    )
    assert q1.cross_model_classification(False, False) == "Q1_CONFIRMATORY_BOTH_FAIL"
