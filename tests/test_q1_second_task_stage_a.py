from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from epistemic_geometry.experiments import q1_second_task_stage_a as stage_a


def _rows(*, textual_accuracy: float = 0.75, textual_tokens: int = 200) -> list[dict]:
    rows = []
    textual_correct = round(64 * textual_accuracy)
    textual_index = 0
    for family in range(32):
        for rollout in range(2):
            baseline_correct = family >= 8
            for condition in stage_a.STAGE_A_CONDITIONS:
                correct = baseline_correct
                tokens = 100
                if condition == "TEXTUAL_CAREFUL":
                    correct = textual_index < textual_correct
                    textual_index += 1
                    tokens = textual_tokens
                rows.append(
                    {
                        "stage": "STAGE_A",
                        "item_id": f"item-{family}",
                        "family_id": f"family-{family}",
                        "condition": condition,
                        "rollout_index": rollout,
                        "seed": family * 10 + rollout * 2 + (condition == "TEXTUAL_CAREFUL"),
                        "commitment_valid": True,
                        "semantic_evaluable": True,
                        "correct": correct,
                        "generated_token_count": tokens,
                    }
                )
    return rows


def test_stage_a_gate_qualifies_nonharmful_compute_manifestation() -> None:
    result = stage_a.stage_a_gate(_rows())
    assert result["classification"] == "Q1_SECOND_TASK_STAGE_A_QUALIFIED"
    assert result["baseline"]["B00"] == 0.25
    assert result["baseline"]["families_wrong_both_rollouts"] == 8
    assert result["textual_careful"]["descriptive_manifestation_classification"] == (
        "TEXTUAL_CAREFUL_NONHARMFUL_COMPUTE_MANIFESTATION"
    )
    assert result["textual_careful"]["manifestation_booleans"] == {
        "TEXTUAL_ACCURACY_GAIN_GE_0_03": False,
        "TEXTUAL_MEAN_TOKEN_RATIO_GE_1_5": True,
        "TEXTUAL_MEDIAN_TOKEN_GAIN_GE_10": True,
    }


def test_stage_a_gate_rejects_textual_competence_harm() -> None:
    result = stage_a.stage_a_gate(_rows(textual_accuracy=0.70, textual_tokens=200))
    assert result["classification"] == "Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED"
    assert not result["textual_careful"]["gates"]["accuracy_nonharm"]


def test_stage_a_schedule_rejects_activation_condition() -> None:
    rows = _rows()
    rows[0]["condition"] = "MEANINGFUL_FIXED_QWEN_L27_D75"
    with pytest.raises(ValueError, match="unauthorized condition"):
        stage_a.validate_schedule(rows)


def test_primary_and_independent_audit_agree_on_synthetic_campaign(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    schedule = json.loads(
        (
            root
            / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit/"
            "STAGE_A_SCHEDULE.json"
        ).read_text()
    )
    families = list(dict.fromkeys(row["family_id"] for row in schedule))
    wrong = set(families[:8])
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with (raw_dir / "journal.jsonl").open("w") as handle:
        for locked in schedule:
            correct = locked["family_id"] not in wrong
            tokens = 200 if locked["condition"] == "TEXTUAL_CAREFUL" else 100
            row = {
                **locked,
                "reference_answer": "1",
                "raw_output": "FINAL: 1" if correct else "FINAL: 0",
                "generated_token_count": tokens,
                "truncated": False,
                "intervention": "NONE",
                "activation_hook_active": False,
            }
            handle.write(json.dumps(row) + "\n")
    (raw_dir / "EXECUTION_COMPLETE.json").write_text(
        json.dumps({"classification": "STAGE_A_COLLECTION_COMPLETE_UNANALYZED"}) + "\n"
    )
    analysis_dir = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_q1_second_task_stage_a.py"),
            "--raw-dir",
            str(raw_dir),
            "--analysis-dir",
            str(analysis_dir),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/audit_q1_second_task_stage_a.py"),
            "--raw-dir",
            str(raw_dir),
            "--analysis-dir",
            str(analysis_dir),
        ],
        check=True,
    )
    primary = json.loads((analysis_dir / "PRIMARY_STAGE_A_RESULTS.json").read_text())
    audit = json.loads((analysis_dir / "FORENSIC_AUDIT.json").read_text())
    assert primary["classification"] == "Q1_SECOND_TASK_STAGE_A_QUALIFIED"
    assert audit["classification"] == "Q1_SECOND_TASK_STAGE_A_FORENSIC_CLEAN"
    assert audit["maximum_metric_difference"] == 0.0
