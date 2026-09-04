from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_analysis():
    path = ROOT / "scripts/analyze_q3_geometry_role_decomposition.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_controller_banks_are_unique_and_deterministic() -> None:
    module = load_analysis()
    first = module.candidate_controller_banks(47, 100, 2, 991)
    second = module.candidate_controller_banks(47, 100, 2, 991)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert all(len(bank) == 8 and len(set(bank)) == 8 for bank in first)
    assert first != sorted(first)


def test_agnostic_routing_averages_controller_ties() -> None:
    module = load_analysis()
    policies = ["A_MEDIUM", "A_STRONG", "B_MEDIUM", "B_STRONG"]
    probability = np.asarray([[0.7, 0.4, 0.7, 0.4], [0.2, 0.8, 0.2, 0.8]])
    target = np.asarray([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
    routed, selected = module.route_fresh_policies("AGNOSTIC", probability, target, policies)
    assert routed.tolist() == [0.5, 0.5]
    assert selected == ["UNIFORM_TIE_MEDIUM", "UNIFORM_TIE_STRONG"]


def test_transfer_model_accepts_unseen_controller_descriptors() -> None:
    module = load_analysis()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 8))
    z_train = rng.normal(size=(12, 9))
    y = rng.integers(0, 2, size=(16, 12)).astype(float)
    fitted = module.fit_transfer_logistic(x, z_train, y, 2, 1.0, 7, 10, 0.03)
    z_fresh = rng.normal(size=(5, 9))
    probability = module.transfer_probabilities(x[:4], z_fresh, fitted)
    assert probability.shape == (4, 5)
    assert np.all((probability > 0) & (probability < 1))
    assert "policy_bias" not in fitted


def test_high_level_ruling_is_mechanical() -> None:
    module = load_analysis()
    assert (
        module.high_level_ruling(
            "GEOMETRY_BANK_SELECTION_SUPPORTED", "CONTROLLER_OOS_GEOMETRY_TRANSFER_SUPPORTED"
        )[0]
        == "Q3_GEOMETRY_BRIDGE_SUPPORTED_READY_FOR_FRESH_INSTRUMENT_DESIGN"
    )
    assert (
        module.high_level_ruling(
            "GEOMETRY_BANK_SELECTION_SUPPORTED", "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED"
        )[0]
        == "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING"
    )
    assert (
        module.high_level_ruling(
            "GEOMETRY_BANK_SELECTION_NOT_SUPPORTED", "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED"
        )[0]
        == "Q3_CAUSAL_POLICY_SELECTABILITY_WITHOUT_GEOMETRY_ATTRIBUTION"
    )


def test_analysis_has_no_model_or_gpu_execution_path() -> None:
    source = (ROOT / "scripts/analyze_q3_geometry_role_decomposition.py").read_text()
    assert "transformers" not in source
    assert "torch" not in source
    assert ".generate(" not in source
    assert "cuda" not in source.lower()


def test_execution_amendment_binds_precheck_and_analysis() -> None:
    review = ROOT / "review/q3_geometry_role_decomposition"
    amendment = json.loads((review / "Q3_GEOMETRY_ROLE_EXECUTION_AMENDMENT.json").read_text())
    assert amendment["scientific_results_opened_before_amendment"] == 0
    assert amendment["scientific_changes"] == 0
    assert amendment["precheck_sha256"] == sha256_file(
        review / "Q3_GEOMETRY_ROLE_DECOMPOSITION_PRECHECK.json"
    )
    assert amendment["analysis_sha256"] == sha256_file(
        ROOT / "scripts/analyze_q3_geometry_role_decomposition.py"
    )


def test_release_summary_matches_frozen_rulings() -> None:
    review = ROOT / "review/q3_geometry_role_decomposition"
    summary = json.loads(
        (review / "Q3_GEOMETRY_ROLE_DECOMPOSITION_RELEASE_SUMMARY.json").read_text()
    )
    assert summary["status"] == "Q3_GEOMETRY_SUPPORTS_PORTFOLIO_NOT_ROUTING"
    assert summary["scientific_state"] == "Q3_NOT_RUN_DEVELOPMENT_ONLY"
    assert summary["part_a"]["ruling"] == "GEOMETRY_BANK_SELECTION_SUPPORTED"
    assert summary["part_b"]["ruling"] == "CONTROLLER_OOS_TRANSFER_NOT_SUPPORTED"
    assert summary["part_a"]["attribution"]["a0_gain_percentile_matched_random"] == 0.986328125
    assert summary["part_b"]["models"]["TRUE"]["routing_gain"] == 0.011875
    assert summary["future_instrument"]["minimum_family_count"] == 800
    assert summary["future_instrument"]["future_outcomes_inspected"] is False


def test_release_hashes_and_safety_are_fail_closed() -> None:
    review = ROOT / "review/q3_geometry_role_decomposition"
    manifest = json.loads((review / "Q3_GEOMETRY_ROLE_ARTIFACT_HASHES.json").read_text())
    for relative, expected in manifest["artifacts"].items():
        assert sha256_file(ROOT / relative) == expected
    assert manifest["raw_text_included"] is False
    assert manifest["q3"] == "NOT_RUN"
    safety = json.loads((review / "Q3_GEOMETRY_ROLE_RELEASE_SAFETY.json").read_text())
    assert safety["status"] == "Q3_GEOMETRY_ROLE_RELEASE_SAFETY_PASS"
    assert safety["raw_benchmark_text_included"] is False
    assert safety["raw_model_outputs_included"] is False
    assert safety["prompt_representation_values_included"] is False
    assert safety["private_itemwise_outcomes_included"] is False
