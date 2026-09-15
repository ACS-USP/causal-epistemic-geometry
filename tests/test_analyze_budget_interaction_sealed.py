from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.analyze_budget_interaction_sealed import (
    SealedAnalysisError,
    analyze_sealed_journal,
    main,
)
from scripts.audit_budget_interaction_raw_seal import audit_raw_seal
from scripts.seal_budget_interaction_raw import validate_completed_raw_journal
from tests.test_seal_budget_interaction_raw import LOCK, _complete_fixture_journal


def _sealed_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    journal = tmp_path / "JOURNAL.jsonl"
    seal_path = tmp_path / "RAW_SEAL.json"
    audit_path = tmp_path / "RAW_SEAL_INDEPENDENT_AUDIT.json"
    _complete_fixture_journal(journal)
    rows = []
    for line in journal.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["record"]["parse_status"] = "OK"
        rows.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
    journal.write_text("\n".join(rows) + "\n", encoding="utf-8")
    seal_path.write_text(
        json.dumps(validate_completed_raw_journal(LOCK, journal)), encoding="utf-8"
    )
    audit_path.write_text(json.dumps(audit_raw_seal(LOCK, journal, seal_path)), encoding="utf-8")
    return journal, seal_path, audit_path


def test_sealed_analysis_runs_frozen_implementation(tmp_path: Path) -> None:
    journal, seal, audit = _sealed_fixture(tmp_path)
    result = analyze_sealed_journal(LOCK, journal, seal, audit)

    assert result["status"] == "Q1_BUDGET_INTERACTION_ANALYSIS_COMPLETE"
    assert result["observation_count"] == 768
    assert result["latent_count"] == 96
    assert result["analysis"]["descriptive_bootstrap"]["n_resamples"] == 2_000
    assert (
        result["analysis_implementation_sha256"]
        == hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "src/epistemic_geometry/analysis/budget_interaction.py"
            ).read_bytes()
        ).hexdigest()
    )


def test_sealed_analysis_rejects_unverified_audit(tmp_path: Path) -> None:
    journal, seal, audit = _sealed_fixture(tmp_path)
    payload = json.loads(audit.read_text())
    payload["classification"] = "FAIL"
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SealedAnalysisError, match="independent"):
        analyze_sealed_journal(LOCK, journal, seal, audit)


def test_sealed_analysis_cli_writes_immutable_json(tmp_path: Path, capsys) -> None:
    journal, seal, audit = _sealed_fixture(tmp_path)
    output = tmp_path / "ANALYSIS.json"
    args = [
        "--lock",
        str(LOCK),
        "--journal",
        str(journal),
        "--raw-seal",
        str(seal),
        "--audit",
        str(audit),
        "--output",
        str(output),
    ]
    assert main(args) == 0
    assert json.loads(output.read_text())["observation_count"] == 768
    assert "Q1_BUDGET_INTERACTION_ANALYSIS_COMPLETE" in capsys.readouterr().out
    with pytest.raises(SealedAnalysisError, match="overwrite"):
        main(args)
