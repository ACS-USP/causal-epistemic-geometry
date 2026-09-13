from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.run_budget_interaction_collection import load_and_validate_lock
from scripts.seal_budget_interaction_raw import (
    RawSealError,
    main,
    validate_completed_raw_journal,
)

from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (
    BudgetInteractionJournal,
)
from epistemic_geometry.benchmarks.reasoning.rollouts import generation_config_hash

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "review/q1_budget_causal_interaction/PRELOCK_ARTIFACTS/LOCK.json"


def _complete_fixture_journal(path: Path) -> None:
    loaded = load_and_validate_lock(LOCK)
    identity = loaded["journal_identity"]
    candidate_hash = loaded["lock"]["candidate_identity_hash"]
    controller_provenance = loaded["lock"]["controller_provenance"]
    journal = BudgetInteractionJournal(path, identity=identity)
    for row in loaded["schedule"]:
        controller = (
            controller_provenance
            if row["condition"] == "D75"
            else None
        )
        journal.append(
            row,
            {
                "latent_id": row["latent_id"],
                "view_id": "fixture-view",
                "family": row["family"],
                "cell": row["cell"],
                "rollout_index": row["rollout_index"],
                "sampling_seed": row["sampling_seed"],
                "metadata": {
                    "cap": row["cap"],
                    "reasoning_budget": row["cap"],
                    "condition": row["condition"],
                    "schedule_identity_hash": row["schedule_identity_hash"],
                    "seed_regime": row["seed_regime"],
                    "candidate_identity": loaded["candidate_identity"],
                    "candidate_identity_hash": candidate_hash,
                    "controller_provenance": controller,
                },
                "generation_config": {
                    "surface": row["surface"],
                    "condition": row["condition"],
                    "max_new_tokens": row["cap"],
                    "sampling_seed": row["sampling_seed"],
                },
                "generation_config_hash": generation_config_hash(
                    {
                        "surface": row["surface"],
                        "condition": row["condition"],
                        "max_new_tokens": row["cap"],
                        "sampling_seed": row["sampling_seed"],
                    }
                ),
                "intervention_id": (
                    "baseline"
                    if row["condition"] == "BASELINE"
                    else loaded["candidate_identity"]["vector_canonical_sha256"]
                ),
                "raw_text": "fixture-raw-text",
                "parsed_answer": None,
                "parse_status": "FIXTURE_UNPARSED",
                "correct": False,
            },
        )


def test_raw_seal_validates_all_768_keys_without_outcome_analysis(tmp_path: Path) -> None:
    journal = tmp_path / "JOURNAL.jsonl"
    _complete_fixture_journal(journal)

    payload = validate_completed_raw_journal(LOCK, journal)

    assert payload["status"] == "RAW_JOURNAL_SEALED"
    assert payload["logical_key_count"] == 768
    assert payload["expected_logical_key_count"] == 768
    assert payload["scientific_model_outcomes_computed"] is False
    assert payload["response_text_exposed"] is False
    assert payload["parse_or_correctness_evaluated"] is False
    assert payload["aggregates_computed"] is False
    assert "fixture-view" not in json.dumps(payload)


def test_raw_seal_rejects_missing_key(tmp_path: Path) -> None:
    journal = tmp_path / "JOURNAL.jsonl"
    _complete_fixture_journal(journal)
    raw = journal.read_bytes()
    journal.write_bytes(raw[: raw.rfind(b"\n", 0, -1) + 1])

    with pytest.raises(RawSealError, match="768 unique logical keys"):
        validate_completed_raw_journal(LOCK, journal)


def test_raw_seal_cli_writes_immutable_json(tmp_path: Path, capsys) -> None:
    journal = tmp_path / "JOURNAL.jsonl"
    output = tmp_path / "RAW_SEAL.json"
    _complete_fixture_journal(journal)

    assert main(["--lock", str(LOCK), "--journal", str(journal), "--output", str(output)]) == 0
    first = output.read_bytes()
    stdout = capsys.readouterr().out
    assert "RAW_JOURNAL_SEALED" in stdout
    assert "fixture-view" not in stdout
    with pytest.raises(RawSealError, match="overwrite"):
        main(["--lock", str(LOCK), "--journal", str(journal), "--output", str(output)])
    assert output.read_bytes() == first


def test_raw_seal_rejects_unterminated_tail_without_rewriting(tmp_path: Path) -> None:
    journal = tmp_path / "JOURNAL.jsonl"
    _complete_fixture_journal(journal)
    raw = journal.read_bytes()
    journal.write_bytes(raw[:-1])
    before = journal.read_bytes()

    with pytest.raises(RawSealError, match="unterminated"):
        validate_completed_raw_journal(LOCK, journal)
    assert journal.read_bytes() == before


def test_raw_seal_rejects_incomplete_runner_record(tmp_path: Path) -> None:
    journal = tmp_path / "JOURNAL.jsonl"
    _complete_fixture_journal(journal)
    rows = journal.read_bytes().splitlines(keepends=True)
    first = json.loads(rows[0])
    del first["record"]["raw_text"]
    rows[0] = (json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n").encode()
    journal.write_bytes(b"".join(rows))

    with pytest.raises(RawSealError, match="missing runner fields"):
        validate_completed_raw_journal(LOCK, journal)
