from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pca_is_fit_on_training_rows_only() -> None:
    module = load_script("analyze_q3_prompt_representation.py")
    rng = np.random.default_rng(4)
    train = rng.normal(size=(20, 40))
    test = rng.normal(size=(5, 40))
    first_train, first_test = module.fit_pca(train, test, 8)
    altered = test + 1000
    second_train, second_test = module.fit_pca(train, altered, 8)
    np.testing.assert_allclose(first_train, second_train, rtol=0, atol=0)
    assert not np.allclose(first_test, second_test)


def test_low_rank_geometry_and_blind_models_are_capacity_matched() -> None:
    dimension, rank, policies, coordinate_width = 16, 2, 8, 8
    geometry_parameters = dimension * rank + coordinate_width * rank + policies
    blind_parameters = dimension * rank + policies * rank + policies
    assert geometry_parameters == blind_parameters


def test_low_rank_fit_is_deterministic() -> None:
    module = load_script("analyze_q3_prompt_representation.py")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(25, 8))
    c = rng.normal(size=(8, 8))
    y = rng.integers(0, 2, size=(25, 8)).astype(float)
    first = module.fit_low_rank_logistic(x, c, y, 2, 1.0, "GEOMETRY", 11, 20, 0.03)
    second = module.fit_low_rank_logistic(x, c, y, 2, 1.0, "GEOMETRY", 11, 20, 0.03)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_single_forward_hook_order_if_torch_available() -> None:
    pytest.importorskip("torch")
    module = load_script("run_q3_prompt_representation_capture.py")
    result = module.verify_single_forward_hook_mechanics()
    assert result["passed"] is True
    assert result["only_current_final_position_changed"] is True


def test_capture_runner_has_no_generation_or_scoring_path() -> None:
    source = (ROOT / "scripts/run_q3_prompt_representation_capture.py").read_text()
    assert ".generate(" not in source
    assert "external-semantic-v3" not in source
    assert "reference_answer" in source  # forbidden-field validation only
    assert "correctness" in source  # provenance and forbidden-field validation only
    assert "backend.model(**encoded" in source


def test_release_summary_omits_itemwise_outcomes() -> None:
    path = (
        ROOT
        / "review/q3_route_a_prompt_representation"
        / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
    )
    if not path.exists():
        pytest.skip("closeout summary not built yet")
    value = json.loads(path.read_text())
    assert value["status"] == (
        "Q3_ROUTE_A_REPRESENTATION_SELECTABLE_BUT_GEOMETRY_NOT_INCREMENTAL"
    )
    serialized = json.dumps(value)
    assert "itemwise_routed_correctness" not in serialized
    assert "itemwise_champion_correctness" not in serialized
    assert value["firewall"]["new_semantic_trajectories"] == 0
    assert value["firewall"]["fresh_evaluation_outcomes_inspected"] is False


def test_release_summary_preserves_frozen_primary_gate_logic() -> None:
    path = (
        ROOT
        / "review/q3_route_a_prompt_representation"
        / "Q3_ROUTE_A_PROMPT_REPRESENTATION_RELEASE_SUMMARY.json"
    )
    if not path.exists():
        pytest.skip("closeout summary not built yet")
    value = json.loads(path.read_text())
    true = value["primary_bank"]["models"]["TRUE_GEOMETRY"]
    incremental = value["gate_results"]["incremental_geometry"]
    assert true["absolute_gain"] >= 0.03
    assert true["positive_outer_folds"] >= 4
    assert incremental["true_minus_blind_gain"] < 0.01
    assert incremental["fold_consistency"] is False
