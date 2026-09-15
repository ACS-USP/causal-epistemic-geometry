from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.analyze_budget_interaction_sealed import analyze_sealed_journal
from scripts.audit_budget_interaction_sealed_analysis import (
    SealedAnalysisAuditError,
    audit_sealed_analysis,
    main,
)
from tests.test_analyze_budget_interaction_sealed import _sealed_fixture
from tests.test_seal_budget_interaction_raw import LOCK


def _analysis_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    journal, seal, raw_audit = _sealed_fixture(tmp_path)
    analysis = tmp_path / "ANALYSIS.json"
    analysis.write_text(
        json.dumps(analyze_sealed_journal(LOCK, journal, seal, raw_audit)), encoding="utf-8"
    )
    return journal, seal, raw_audit, analysis


def test_independent_analysis_audit_recomputes_primary(tmp_path: Path) -> None:
    journal, seal, raw_audit, analysis = _analysis_fixture(tmp_path)
    payload = audit_sealed_analysis(LOCK, journal, analysis, seal, raw_audit)

    assert payload["classification"] == "Q1_BUDGET_INTERACTION_ANALYSIS_AUDIT_PASS"
    assert payload["observation_count"] == 768
    assert payload["maximum_numeric_difference"] <= 1e-12


def test_independent_analysis_audit_rejects_changed_primary(tmp_path: Path) -> None:
    journal, seal, raw_audit, analysis = _analysis_fixture(tmp_path)
    payload = json.loads(analysis.read_text())
    payload["analysis"]["interaction"] = 0.25
    analysis.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SealedAnalysisAuditError, match="differs"):
        audit_sealed_analysis(LOCK, journal, analysis, seal, raw_audit)


def test_independent_analysis_audit_cli_writes_immutable_json(tmp_path: Path, capsys) -> None:
    journal, seal, raw_audit, analysis = _analysis_fixture(tmp_path)
    output = tmp_path / "ANALYSIS_AUDIT.json"
    args = [
        "--lock",
        str(LOCK),
        "--journal",
        str(journal),
        "--analysis",
        str(analysis),
        "--raw-seal",
        str(seal),
        "--raw-audit",
        str(raw_audit),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert (
        json.loads(output.read_text())["classification"]
        == "Q1_BUDGET_INTERACTION_ANALYSIS_AUDIT_PASS"
    )
    assert "Q1_BUDGET_INTERACTION_ANALYSIS_AUDIT_PASS" in capsys.readouterr().out
    with pytest.raises(SealedAnalysisAuditError, match="overwrite"):
        main(args)
