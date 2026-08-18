"""Tests for the low-cap correction and prospective cap selection."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _script_main(name: str):
    namespace = runpy.run_path(str(ROOT / "scripts" / name), run_name="cap_protocol_test")
    return namespace["main"]


def test_completion_cap_selection_ignores_accuracy(tmp_path, monkeypatch) -> None:
    run = tmp_path / "diagnostic"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"identity": {"candidate": "CRUXEval", "item_ids": ["a", "b"]}}),
        encoding="utf-8",
    )
    rows = [
        {
            "item_id": "a",
            "status": "VALID_WRONG",
            "token_count": 1200,
            "metadata": {"diagnostic_cap": 8192},
        },
        {
            "item_id": "b",
            "status": "TRUNCATED_THINKING",
            "token_count": 8192,
            "metadata": {"diagnostic_cap": 8192},
        },
        {
            "item_id": "a",
            "status": "VALID_WRONG",
            "token_count": 1200,
            "metadata": {"diagnostic_cap": 16384},
        },
        {
            "item_id": "b",
            "status": "VALID_CORRECT",
            "token_count": 9000,
            "metadata": {"diagnostic_cap": 16384},
        },
    ]
    (run / "journal.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = run / "recommendation.json"
    monkeypatch.setattr(
        sys, "argv", ["analyze_completion_diagnostics.py", str(run), "--output", str(output)]
    )
    assert _script_main("analyze_completion_diagnostics.py")() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["proposed_cap"] == 16384
    assert report["accuracy_used_for_selection"] is False


def test_low_cap_reclassification_changes_manifest_only(tmp_path, monkeypatch) -> None:
    run = tmp_path / "low_cap"
    run.mkdir()
    original_journal = '{"item_id":"a","status":"TRUNCATED_THINKING"}\n'
    (run / "journal.jsonl").write_text(original_journal, encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "status": "KILLED_Q1_SMOKE_EARLY",
                "identity": {"generation_config": {"max_new_tokens": 2048}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["reclassify_low_cap_runs.py", str(run)])
    assert _script_main("reclassify_low_cap_runs.py")() == 0
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "LOW_CAP_DIAGNOSTIC"
    assert manifest["scientific_qualification_eligible"] is False
    assert (run / "journal.jsonl").read_text(encoding="utf-8") == original_journal
