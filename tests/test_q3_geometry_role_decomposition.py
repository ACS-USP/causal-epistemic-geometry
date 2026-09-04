from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


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
