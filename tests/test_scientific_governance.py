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
    }
    assert state["current"]["gpu_work_authorized"] is (
        workstream in {"SUBSTRATE_RACE", "FIRST_MICRO_Q1_LOCKED"}
    )
    assert state["scientific_firewall"]["steering"] in {
        "ORIGINAL_Q1_NOT_RUN",
        "ORIGINAL_Q1_MICRO_Q1_COMPLETE_DEVELOPMENT_NO_SIGNAL",
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
