from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "scripts")

import analyze_q3_fresh_qualification as analysis  # noqa: E402
import audit_q3_fresh_qualification as forensic  # noqa: E402


def test_terminal_rows_fail_closed_before_parser() -> None:
    row = {
        "family_id": "f",
        "condition": "p",
        "rollout_index": 0,
        "terminal_reason": "EXTREME_MECHANICAL_REPETITION_V1",
        "runtime_error": None,
        "generated_token_count": 256,
        "raw_output": "FINAL: 7",
    }
    scored = analysis.classify_row(row, "7")
    assert not scored["commitment_valid"]
    assert not scored["semantic_evaluable"]
    assert not scored["correct"]
    assert scored["status"] == "REPETITION_STOP"


def test_external_semantic_v3_exact_typed_scoring() -> None:
    base = {
        "family_id": "f",
        "condition": "p",
        "rollout_index": 0,
        "terminal_reason": "eos",
        "runtime_error": None,
        "truncated": False,
        "generated_token_count": 4,
    }
    assert analysis.classify_row({**base, "raw_output": "FINAL: [1, 2]"}, "[1, 2]")["correct"]
    wrong_type = analysis.classify_row({**base, "raw_output": "FINAL: True"}, "1")
    assert wrong_type["semantic_evaluable"] and not wrong_type["correct"]


def _score(family: str, condition: str, rollout: int, correct: bool) -> dict:
    return {
        "family_id": family,
        "condition": condition,
        "rollout_index": rollout,
        "commitment_valid": True,
        "semantic_evaluable": True,
        "correct": correct,
        "status": "VALID_CORRECT" if correct else "VALID_WRONG",
        "generated_token_count": 8,
    }


def test_oracle_headroom_uses_family_then_two_rollout_bank_max() -> None:
    families = ["f0", "f1"]
    bank = [f"b{i}" for i in range(8)]
    scores = {}
    for family in families:
        for condition in [*bank, analysis.CHAMPION, analysis.ROUTER]:
            for rollout in (0, 1):
                correct = family == "f1"
                if family == "f0" and condition == "b0":
                    correct = True
                scores[(family, condition, rollout)] = _score(family, condition, rollout, correct)
    values = analysis.qualification_quantities(families, scores, bank)
    assert values["champion_accuracy"] == 0.5
    assert values["frozen_bank_oracle_accuracy"] == 1.0
    assert values["frozen_bank_oracle_headroom_over_champion"] == 0.5
    assert values["routed_gain_is_qualification_gate"] is False


def test_gate_conjunction_is_exact_and_routed_gain_is_not_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summaries = {
        condition: {
            "commitment_validity": 0.95,
            "semantic_evaluability": 0.95,
            "repetition_rate": 0.10,
        }
        for condition in [analysis.ROUTER, analysis.CHAMPION, *[f"b{i}" for i in range(8)]]
    }
    quantities = {
        "condition_summaries": summaries,
        "champion_accuracy": 0.25,
        "frozen_bank_oracle_headroom_over_champion": 0.05,
        "routed_minus_champion_accuracy": -1.0,
    }
    dataset = {
        "global_checks": {
            "dual_evaluator_agreement": 1.0,
            "reference_repeat_determinism": 1.0,
            "parser_reference_roundtrip": 1.0,
            "cross_split_collisions": 0,
        }
    }
    manifest = {"structural_near_duplicate_rate": 0.01}
    monkeypatch.setattr(
        analysis,
        "read_json",
        lambda path: dataset if path == analysis.DATASET_SEAL else manifest,
    )
    gates, status = analysis.classify_gates(quantities)
    assert all(row["pass"] for row in gates.values())
    assert status == "Q3_FRESH_INSTRUMENT_QUALIFIED_CONFIRMATION_NOT_AUTHORIZED"
    assert "routed_minus_champion_accuracy" not in gates


def test_scoring_inputs_are_exactly_hash_pinned() -> None:
    assert analysis.EXPECTED_JOURNAL_SHA256 == (
        "2194646bcf25ff9512c5e3aaf35d4c2d0ed922f1f86ba6480709a1958dc89431"
    )
    assert analysis.EXPECTED_RECOVERY_SEAL_SHA256 == (
        "e7eaf43da51690bd388c191283287374f3b45b7cb8f8015e33b790ff5a6e79ba"
    )
    assert analysis.sha256_file(analysis.PARSER_SOURCE) == analysis.EXPECTED_PARSER_SHA256


def test_forensic_script_does_not_import_primary_implementation() -> None:
    source = inspect.getsource(forensic)
    imports = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "analyze_q3_fresh_qualification" not in imports


def test_private_reference_file_is_not_tracked() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "qualification.jsonl").exists()
