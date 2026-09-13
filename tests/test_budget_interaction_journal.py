from __future__ import annotations

import pytest

from epistemic_geometry.benchmarks.reasoning.budget_interaction import (
    build_manifest,
    build_schedule,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_journal import (
    BudgetInteractionJournal,
    physical_key,
    schedule_identity_hash,
)
from epistemic_geometry.reproducibility import stable_digest


def _identity() -> dict[str, object]:
    return {"experiment": "budget-d75", "manifest_hash": "manifest", "controller": "fixed"}


def _record(schedule: dict[str, object], *, target: int = 1) -> dict[str, object]:
    key = physical_key(schedule)
    schedule_hash = schedule["schedule_identity_hash"]
    return {
        "latent_id": key[0],
        "view_id": f"{key[0]}:canonical",
        "family": schedule["family"],
        "cell": schedule["cell"],
        "target": target,
        "intervention_id": "baseline" if key[2] == "BASELINE" else "fake-vector",
        "rollout_index": key[3],
        "sampling_seed": schedule["sampling_seed"],
        "raw_text": "partial",
        "parsed_answer": None,
        "parse_status": "TRUNCATED_NO_FINAL",
        "correct": False,
        "token_ids": [],
        "stop_reason": "max_new_tokens",
        "generation_config": {
            "surface": "canonical",
            "condition": key[2],
            "max_new_tokens": key[1],
            "sampling_seed": schedule["sampling_seed"],
        },
        "metadata": {
            "cap": key[1],
            "reasoning_budget": key[1],
            "condition": key[2],
            "schedule_identity_hash": schedule_hash,
        },
        "physical_generation_id": stable_digest(
            "Q1-V3-BUDGET-CAUSAL-INTERACTION-V1", "PHYSICAL-GENERATION", schedule_hash
        ),
    }


def test_budget_journal_append_reopen_and_exact_duplicate(tmp_path) -> None:
    schedule = build_schedule(build_manifest(n_per_cell=1))[0]
    path = tmp_path / "budget.jsonl"
    journal = BudgetInteractionJournal(path, identity=_identity())
    record = _record(schedule)
    journal.append(schedule, record)
    journal.append(schedule, record)
    assert list(journal.rows) == [physical_key(schedule)]
    reopened = BudgetInteractionJournal(path, identity=_identity())
    assert reopened.get_for_schedule(schedule) == record


def test_budget_journal_rejects_conflict_identity_and_schedule_mismatch(tmp_path) -> None:
    schedule = build_schedule(build_manifest(n_per_cell=1))[0]
    path = tmp_path / "budget.jsonl"
    journal = BudgetInteractionJournal(path, identity=_identity())
    journal.append(schedule, _record(schedule))
    with pytest.raises(ValueError, match="conflicting duplicate"):
        journal.append(schedule, _record(schedule, target=2))
    with pytest.raises(ValueError, match="provenance|identity"):
        BudgetInteractionJournal(path, identity={"experiment": "different"})
    tampered = {**schedule, "schedule_identity_hash": "bad"}
    with pytest.raises(ValueError, match="schedule identity hash"):
        journal.append(tampered, _record(schedule))


def test_budget_journal_recovers_only_malformed_final_partial_tail(tmp_path) -> None:
    schedule = build_schedule(build_manifest(n_per_cell=1))[0]
    path = tmp_path / "budget.jsonl"
    journal = BudgetInteractionJournal(path, identity=_identity())
    journal.append(schedule, _record(schedule))
    path.write_bytes(path.read_bytes() + b'{"journal_version":"partial"')
    reopened = BudgetInteractionJournal(path, identity=_identity())
    assert len(reopened.rows) == 1
    assert reopened.quarantined_tail is not None
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(lines[0] + b"not-json\n" + lines[0])
    with pytest.raises(ValueError, match="non-final"):
        BudgetInteractionJournal(path, identity=_identity())


def test_budget_journal_physical_keys_include_cap_and_condition(tmp_path) -> None:
    schedule = build_schedule(build_manifest(n_per_cell=1))
    baseline_2048 = schedule[0]
    d75_2048 = schedule[1]
    baseline_4096 = next(
        row for row in schedule if row["condition"] == "BASELINE" and row["cap"] == 4096
    )
    journal = BudgetInteractionJournal(tmp_path / "budget.jsonl", identity=_identity())
    for row in (baseline_2048, d75_2048, baseline_4096):
        journal.append(row, _record(row))
    assert len(journal.rows) == 3
    assert len({physical_key(row) for row in (baseline_2048, d75_2048, baseline_4096)}) == 3
    assert schedule_identity_hash(d75_2048) != schedule_identity_hash(baseline_2048)

