from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.audit_budget_interaction_raw_seal import RawSealAuditError, audit_raw_seal, main
from scripts.seal_budget_interaction_raw import validate_completed_raw_journal
from tests.test_seal_budget_interaction_raw import LOCK, _complete_fixture_journal


def _sealed_fixture(tmp_path: Path) -> tuple[Path, Path]:
    journal = tmp_path / "JOURNAL.jsonl"
    seal = tmp_path / "RAW_SEAL.json"
    _complete_fixture_journal(journal)
    seal.write_text(json.dumps(validate_completed_raw_journal(LOCK, journal)), encoding="utf-8")
    return journal, seal


def test_independent_audit_passes_without_outcome_analysis(tmp_path: Path) -> None:
    journal, seal = _sealed_fixture(tmp_path)
    payload = audit_raw_seal(LOCK, journal, seal)

    assert payload["classification"] == "RAW_SEAL_INDEPENDENT_AUDIT_PASS"
    assert payload["logical_key_count"] == 768
    assert payload["response_text_exposed"] is False
    assert payload["parse_or_correctness_evaluated"] is False
    assert payload["scientific_model_outcomes_computed"] is False
    assert payload["aggregates_computed"] is False


def test_independent_audit_rejects_journal_changed_after_seal(tmp_path: Path) -> None:
    journal, seal = _sealed_fixture(tmp_path)
    journal.write_bytes(journal.read_bytes() + b" ")

    with pytest.raises(RawSealAuditError, match="journal hash"):
        audit_raw_seal(LOCK, journal, seal)


def test_independent_audit_cli_writes_immutable_json(tmp_path: Path, capsys) -> None:
    journal, seal = _sealed_fixture(tmp_path)
    output = tmp_path / "RAW_SEAL_INDEPENDENT_AUDIT.json"

    assert (
        main(
            [
                "--lock",
                str(LOCK),
                "--journal",
                str(journal),
                "--raw-seal",
                str(seal),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["classification"] == "RAW_SEAL_INDEPENDENT_AUDIT_PASS"
    assert "RAW_SEAL_INDEPENDENT_AUDIT_PASS" in capsys.readouterr().out
    with pytest.raises(RawSealAuditError, match="overwrite"):
        main(
            [
                "--lock",
                str(LOCK),
                "--journal",
                str(journal),
                "--raw-seal",
                str(seal),
                "--output",
                str(output),
            ]
        )
