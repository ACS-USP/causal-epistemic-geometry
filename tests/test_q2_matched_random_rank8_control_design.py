import hashlib
import json
from pathlib import Path

import pytest
from scripts.simulate_q2_matched_random_rank8_control import coefficient_audit

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "q2_matched_random_rank8_control_design"


def test_planning_precheck_is_model_free_and_subspace_level() -> None:
    precheck = json.loads((REVIEW / "PLANNING_PRECHECK.json").read_text())
    assert precheck["status"] == "FROZEN_MODEL_FREE_PLANNING_PRECHECK"
    assert precheck["independent_unit"] == "RANDOM_SUBSPACE_ORIENTATION"
    assert precheck["fixed_inputs"]["total_coefficient_identities"] == 47
    assert precheck["prohibitions"]["generate_final_random_basis"] is True
    assert precheck["prohibitions"]["derive_final_experimental_seed"] is True
    assert precheck["prohibitions"]["run_semantic_trajectories"] is True
    assert precheck["prohibitions"]["gpu_or_model_inference"] is True
    assert precheck["prohibitions"]["treat_controller_or_dyad_as_independent_subspace_unit"]


def test_exact_47_coefficient_identities_are_reconstructed_without_a_basis() -> None:
    audit = coefficient_audit()
    assert audit["status"] == "EXACT_47_COEFFICIENT_IDENTITIES_RECONSTRUCTED"
    assert audit["historical_count"] == 31
    assert audit["fresh_count"] == 16
    assert audit["total_count"] == 47
    assert audit["coefficient_dimension"] == 8
    assert audit["maximum_gram_asymmetry"] < 1e-12
    assert audit["maximum_gram_diagonal_error"] < 1e-12
    assert audit["final_random_basis_generated"] == 0
    assert audit["semantic_outcomes_used"] == 0


@pytest.mark.parametrize("subspaces", [20, 39, 79])
def test_subspace_tail_test_has_expected_resolution(subspaces: int) -> None:
    assert 1.0 / (subspaces + 1.0) <= 0.05
    if subspaces == 20:
        assert 1.0 / subspaces == 0.05


def test_planning_grid_does_not_change_required_controller_identity_design() -> None:
    precheck = json.loads((REVIEW / "PLANNING_PRECHECK.json").read_text())
    required = [
        split
        for split in precheck["planning_grid"]["controller_splits"]
        if split["role"] == "REQUIRED_FULL_47_IDENTITY_DESIGN"
    ]
    assert required == [{"K_reference": 31, "K_fresh": 16, "role": required[0]["role"]}]
    sensitivity = [
        split
        for split in precheck["planning_grid"]["controller_splits"]
        if split["role"].startswith("PLANNING_SENSITIVITY_ONLY")
    ]
    assert len(sensitivity) == 1


def test_design_ruling_preserves_scientific_firewall() -> None:
    ruling = json.loads((REVIEW / "DESIGN_RULING.json").read_text())
    assert ruling["status"] == "Q2_MATCHED_RANDOM_RANK8_CONTROL_REQUIRES_FURTHER_THEORY"
    assert ruling["scientific_unit"] == "RANDOM_SUBSPACE_ORIENTATION"
    assert ruling["closed_results_changed"] is False
    assert ruling["final_random_bases_generated"] == 0
    assert ruling["experimental_seeds_generated"] == 0
    assert ruling["semantic_trajectories"] == 0
    assert ruling["qwen_loaded"] is False
    assert ruling["gpu_used"] is False
    assert ruling["q3_run"] is False


def test_release_manifest_hashes() -> None:
    manifest = json.loads((REVIEW / "ARTIFACT_MANIFEST.json").read_text())
    assert manifest["classification"] == "MODEL_FREE_DESIGN_ARTIFACTS_RELEASE_SAFE"
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative if "/" in relative else REVIEW / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
