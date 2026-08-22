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
    assert state["project"]["claim_status"] == "NONE_FROZEN"
    assert state["scientific_firewall"]["confirmatory_holdout"] == "UNTOUCHED"
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
        }
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
    }
    assert state["scientific_firewall"]["published_positive_control"] == "PASS"


def test_registry_and_document_audits_pass() -> None:
    registry = _run("check_experiment_registry.py")
    documents = _run("check_docs.py")
    assert registry.returncode == 0, registry.stderr
    assert documents.returncode == 0, documents.stderr


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
