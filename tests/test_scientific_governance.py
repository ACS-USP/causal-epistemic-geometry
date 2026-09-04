from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python", str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_project_state_and_generated_status_are_current() -> None:
    result = _run("render_project_state.py", "--check")
    assert result.returncode == 0, result.stderr
    state = yaml.safe_load((ROOT / "project_state.yaml").read_text(encoding="utf-8"))
    assert state["project"]["claim_status"] == "Q2_V4_1_G2__Q2_OOS_V2_A0_PASS"
    assert state["scientific_firewall"]["confirmatory_holdout"] in {
        "UNTOUCHED",
        "SEALED_ASSIGNED_UNACCESSED",
        "CONSUMED_CONFIRMATORY_CLOSED",
    }
    workstream = state["current"]["workstream"]
    assert workstream in {
        "SUBSTRATE_RACE",
        "SUBSTRATE_RACE_COMPLETE_PRINCIPAL_REVIEW",
        "FIRST_MICRO_Q1_LOCKED",
        "FIRST_MICRO_Q1_COMPLETE",
        "FIRST_MICRO_Q1_AUDITED",
        "GATE5_SOURCE_DURATION_LOCKED",
        "GATE5_SOURCE_DURATION_COMPLETE",
        "GATE6_LAYER_SOURCE_RFM_ATLAS_LOCKED",
        "GATE6_LAYER_SOURCE_RFM_ATLAS",
        "GATE6_SOURCE_ATTRITION_REPAIR",
        "GATE6_2_FIRST_STAGE_REPAIR_MEAN_BRIDGE",
        "GATE6_2_FIRST_STAGE_REPAIR_MEAN_BRIDGE_COMPLETE",
        "GATE6_3_SINGLE_MEAN_SEMANTIC_EVALUATION_LOCKED",
        "GATE6_3_SINGLE_MEAN_SEMANTIC_EVALUATION_COMPLETE",
        "GATE6_3_SEMANTIC_VALIDITY_AUDIT_COMPLETE",
        "GATE7_FRESH_SINGLE_L27_REPLICATION",
        "GATE7_FRESH_SINGLE_L27_REPLICATION_COMPLETE",
        "GATE8_L27_DOSE_CALIBRATION",
        "GATE8_L27_DOSE_CALIBRATION_COMPLETE",
        "GATE9_SELECTED_D75_EVALUATION",
        "GATE9_SELECTED_D75_EVALUATION_COMPLETE",
        "GATE10_CROSS_DOMAIN_CHARCOUNT",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_COMPLETE",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_BLOCKED_COST",
        "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM",
        "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM_COMPLETE",
        "GATE11_1_ARTIFACT_COMPLETE_FORENSIC_REPLICATION",
        "GATE11_1_ARTIFACT_COMPLETE_FORENSIC_REPLICATION_COMPLETE",
        "GATE12_UTILITY_ALIGNED_PULLBACK",
        "GATE12_UTILITY_ALIGNED_PULLBACK_COMPLETE",
        "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE",
        "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE_COMPLETE",
        "GATE13_CROSS_MODEL_MINISTRAL3",
        "GATE13_CROSS_MODEL_MINISTRAL3_COMPLETE",
        "GATE13_1_ALL_LAYER_CAUSAL_ATLAS",
        "GATE13_1_ALL_LAYER_CAUSAL_ATLAS_COMPLETE",
        "Q1_CONFIRMATORY_OFFLINE_POWER_QUALIFICATION",
        "Q1_CONFIRMATORY_FIXED_CONTROLLERS",
        "Q2_CONTROLLER_HELDOUT_GEOMETRY_PILOT",
        "Q2_CONTROLLER_HELDOUT_GEOMETRY_PILOT_COMPLETE",
        "Q2_CONTROLLER_HELDOUT_GEOMETRY_V2",
        "Q2_CONTROLLER_HELDOUT_GEOMETRY_V2_COMPLETE",
        "Q2_V2_PRINCIPAL_REVIEW_Q2_V3_DESIGN",
        "Q2_GEOMETRY_FOUNDATIONS_Q2_V3_RADIAL_ANGULAR_DESIGN",
        "Q2_M3_QUALIFICATION_CRUXEVAL_PROVENANCE",
        "Q2_V3_RADIAL_ANGULAR_PROSPECTIVE_FREEZE",
        "Q2_V3_RADIAL_ANGULAR_EXECUTION",
        "Q2_V3_PANEL_PROVENANCE_MISMATCH",
        "Q2_V3_PROMPT_PROVENANCE_RECONCILIATION",
        "Q2_V3_AMENDMENT1_FREEZE",
        "Q2_V3_AMENDMENT1_EXECUTION",
        "Q2_V3_AMENDMENT1_EXECUTION_COMPLETE",
        "Q2_V3_REPLACEMENT_FAMILY_DESIGN_COMPLETE",
        "Q2_V3_FOUR_FAMILY_STATISTICAL_REDESIGN_COMPLETE",
        "Q2_V4_INTERVENTION_SUBSPACE_DESIGN_COMPLETE",
        "Q2_V4_1_31_SAFE_BANK_REVIEW_COMPLETE",
        "Q2_V4_1_PRESEMANTIC_PREDICTION_LOCK_COMPLETE",
        "Q2_V4_1_SEMANTIC_EXECUTION",
        "Q2_V4_1_SEMANTIC_EXECUTION_COMPLETE",
        "Q1_SECOND_TASK_LIVECODEBENCH_STAGE_B",
        "CLOSED_RESULT_INTEGRATION_AND_SPECIFICITY_CONTROL_DESIGN",
        "Q3_REALIZABLE_UTILITY_DESIGN",
        "Q2_V4_SPARK1_PRESEMANTIC_QUALIFICATION",
        "Q2_V4_SPARK1_PRESEMANTIC_QUALIFICATION_COMPLETE",
    }
    assert state["current"]["gpu_work_authorized"] is (
        workstream
        in {
            "SUBSTRATE_RACE",
            "FIRST_MICRO_Q1_LOCKED",
            "GATE5_SOURCE_DURATION_LOCKED",
            "GATE6_LAYER_SOURCE_RFM_ATLAS_LOCKED",
            "GATE6_SOURCE_ATTRITION_REPAIR",
            "GATE6_2_FIRST_STAGE_REPAIR_MEAN_BRIDGE",
            "GATE6_3_SINGLE_MEAN_SEMANTIC_EVALUATION_LOCKED",
            "GATE7_FRESH_SINGLE_L27_REPLICATION",
            "GATE8_L27_DOSE_CALIBRATION",
            "GATE9_SELECTED_D75_EVALUATION",
            "GATE10_CROSS_DOMAIN_CHARCOUNT",
            "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM",
            "GATE12_1_CONTINUOUS_GEOMETRY_ENGINE",
            "GATE13_1_ALL_LAYER_CAUSAL_ATLAS",
            "Q2_CONTROLLER_HELDOUT_GEOMETRY_PILOT",
            "Q2_CONTROLLER_HELDOUT_GEOMETRY_V2",
            "Q2_V3_RADIAL_ANGULAR_EXECUTION",
            "Q2_V3_AMENDMENT1_EXECUTION",
            "Q2_V4_SPARK1_PRESEMANTIC_QUALIFICATION",
            "Q2_V4_1_SEMANTIC_EXECUTION",
            "Q1_SECOND_TASK_LIVECODEBENCH_STAGE_B",
        }
    )
    if workstream == "Q1_CONFIRMATORY_FIXED_CONTROLLERS":
        assert state["current"]["lifecycle"] == "CLOSED"
        assert (
            state["scientific_firewall"]["confirmatory_holdout"] == "CONSUMED_CONFIRMATORY_CLOSED"
        )
        assert (
            state["scientific_firewall"]["steering"] == "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL"
        )
    assert state["scientific_firewall"]["steering"] in {
        "ORIGINAL_Q1_NOT_RUN",
        "ORIGINAL_Q1_MICRO_Q1_COMPLETE_DEVELOPMENT_NO_SIGNAL",
        "ORIGINAL_Q1_GATE4_AUDITED_BOUNDED_NULL_GATE5_LOCKED",
        "ORIGINAL_Q1_GATE4_AUDITED_BOUNDED_NULL_GATE5_DURATION_BELOW_MOVEMENT",
        "GATE7_DEVELOPMENT_DESTRUCTIVE_SPECIFIC_MOVEMENT_WITH_VALIDITY_LOSS",
        "GATE8_CALIBRATION_LOCKED_GATE7_REPLICATED_MOVEMENT_WITH_VALIDITY_LOSS",
        "GATE8_CALIBRATION_SELECTED_D75_GATE9_NOT_RUN",
        "GATE9_SELECTED_D75_LOCKED_NOT_YET_RUN",
        "GATE9_SELECTED_D75_EVALUATION_COMPLETE",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_LOCKED_NOT_YET_RUN",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_COMPLETE",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_INCOMPLETE_COST_STOP",
        "GATE10_CROSS_DOMAIN_CHARCOUNT_RESUME_AUTHORIZED",
        "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM_LOCKED",
        "GATE11_DOMAIN_CONDITIONED_CONTROL_POSTMORTEM_COMPLETE",
        "GATE11_1_FORENSIC_REPLICATION_CLOSED",
        "GATE12_JVP_ENGINE_FAILURE_CLOSED",
        "GATE12_1_ENGINEERING_ONLY_LOCKED",
        "GATE12_1_ENGINE_NOT_QUALIFIED_CLOSED",
        "GATE13_BOUNDED_CROSS_MODEL_NULL",
        "GATE13_1_ALL_LAYER_CAUSAL_ATLAS_LOCKED",
        "GATE13_1_CLOSED_STRONG_CROSS_MODEL_REPLICATION",
        "Q1_CONFIRMATORY_HOLDOUT_ASSIGNED_POWER_PENDING",
        "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL",
        "Q2_V2_PRE_SOURCE_LOCK_NO_COMMON_OUTCOMES",
        "Q2_V4_1_SEMANTIC_EXECUTION_OPEN_NO_OUTCOME_INSPECTION",
    }
    assert state["scientific_firewall"]["published_positive_control"] == "PASS"


def test_registry_and_document_audits_pass() -> None:
    registry = _run("check_experiment_registry.py")
    documents = _run("check_docs.py")
    external = _run("check_external_readiness.py")
    assert registry.returncode == 0, registry.stderr
    assert documents.returncode == 0, documents.stderr
    assert external.returncode == 0, external.stderr


def test_prospective_specs_are_explicitly_unexecuted() -> None:
    for path in sorted((ROOT / "experiments" / "specs").glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert (
            "PROSPECTIVE" in spec["status"]
            or "FROZEN" in spec["status"]
            or spec["status"].startswith("CLOSED_")
        )


def test_dense_code_wrapper_rejects_unpinned_image_before_docker(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    workspace.mkdir()
    results.mkdir()
    process = subprocess.run(
        [
            "bash",
            str(ROOT / "infra" / "evalplus_sandbox" / "run_eval.sh"),
            "unreviewed:latest",
            str(workspace),
            str(results),
            "humaneval",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert "immutable sha256 digest" in process.stderr
