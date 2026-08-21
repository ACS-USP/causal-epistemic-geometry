from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from epistemic_geometry.experiments.gate6_source import (
    locate_final_commitment_boundary,
    select_common_eligible,
)
from epistemic_geometry.research.reliability import (
    CrashSafeJournal,
    OutputContractStatus,
    condition_formatting_differs,
    inspect_output_contract,
    managed_registrations,
    validate_logical_rows,
    validate_unit_vector,
)


@pytest.mark.parametrize(
    ("case", "raw", "expected"),
    [
        (
            "FINAL inside Markdown fence",
            "```text\nFINAL: 7\n```",
            OutputContractStatus.FINAL_INSIDE_FENCE,
        ),
        (
            "fence closure after FINAL",
            "~~~\nFINAL: 7\n~~~",
            OutputContractStatus.FINAL_INSIDE_FENCE,
        ),
        ("multiple FINAL values", "FINAL: 1\nFINAL: 2", OutputContractStatus.MULTIPLE_FINAL),
        ("no FINAL", "answer is seven", OutputContractStatus.MISSING_FINAL),
    ],
)
def test_output_contract_faults(case: str, raw: str, expected: OutputContractStatus) -> None:
    assert case
    assert inspect_output_contract(raw) is expected


def test_truncation_is_never_silently_accepted() -> None:
    assert (
        inspect_output_contract("reasoning\nFINAL: 7", truncated=True)
        is OutputContractStatus.TRUNCATED
    )


def _candidate(index: int, eligible: bool) -> dict[str, object]:
    return {
        "item_id": f"item-{index}",
        "candidate_order": index,
        "allocation": "ORIGINAL" if index < 2 else "RESERVE",
        "eligible": eligible,
        "reason": "eligible" if eligible else "missing_marker",
        "condition_status": {},
    }


def test_mechanical_attrition_uses_frozen_reserve_without_aborting_phase() -> None:
    selected, decisions = select_common_eligible(
        [_candidate(0, True), _candidate(1, False), _candidate(2, True)],
        target=2,
        max_ineligible=1,
        split="source",
    )
    assert [row["item_id"] for row in selected] == ["item-0", "item-2"]
    assert [decision.candidate_order for decision in decisions] == [0, 1, 2]


def test_crash_after_first_source_condition_resumes_without_duplication(tmp_path) -> None:
    path = tmp_path / "source.jsonl"
    identity = {"gate": "fixture", "commit": "abc"}
    interrupted = CrashSafeJournal(
        path, identity=identity, key_fields=("item_id", "condition")
    )
    first = {"item_id": "item-1", "condition": "careful", "mechanical_status": "eligible"}
    interrupted.append(first)

    resumed = CrashSafeJournal(path, identity=identity, key_fields=("item_id", "condition"))
    assert resumed.pending_conditions(["item-1"], ["careful", "direct"]) == [
        ("item-1", "direct")
    ]
    resumed.append(first)
    resumed.append({"item_id": "item-1", "condition": "direct", "mechanical_status": "eligible"})
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_partial_journal_line_before_fsync_is_quarantined(tmp_path) -> None:
    path = tmp_path / "journal.jsonl"
    identity = {"gate": "fixture"}
    journal = CrashSafeJournal(path, identity=identity, key_fields=("item_id", "condition"))
    journal.append({"item_id": "item-1", "condition": "baseline"})
    path.write_bytes(path.read_bytes() + b'{"version":"partial"')
    recovered = CrashSafeJournal(path, identity=identity, key_fields=("item_id", "condition"))
    assert recovered.quarantined_tail is not None
    assert list(recovered.rows) == [("item-1", "baseline")]


def test_duplicate_and_missing_logical_rows_are_both_reported() -> None:
    rows = [
        {"item_id": "a", "condition": "baseline"},
        {"item_id": "a", "condition": "baseline"},
    ]
    report = validate_logical_rows(
        rows,
        key_fields=("item_id", "condition"),
        expected_keys=(("a", "baseline"), ("a", "treatment")),
    )
    assert report.duplicate_keys == (("a", "baseline"),)
    assert report.missing_keys == (("a", "treatment"),)
    assert not report.valid


def test_hook_leakage_cleanup_runs_after_exception() -> None:
    active: list[str] = []

    @dataclass
    class Handle:
        name: str

        def remove(self) -> None:
            active.remove(self.name)

    def register(name: str):
        def callback() -> Handle:
            active.append(name)
            return Handle(name)

        return callback

    with pytest.raises(RuntimeError, match="injected"):
        with managed_registrations([register("L8"), register("L12")]):
            assert active == ["L8", "L12"]
            raise RuntimeError("injected failure")
    assert active == []


def test_incorrect_random_vector_norm_is_rejected() -> None:
    assert validate_unit_vector([1.0, 0.0]) == 1.0
    with pytest.raises(ValueError, match="norm must be 1"):
        validate_unit_vector(np.asarray([2.0, 0.0]))


class CharacterTokenizer:
    all_special_ids: list[int] = []

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        del add_special_tokens, return_offsets_mapping
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool = True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in ids)


def test_execution_boundary_marker_absent_fails_closed() -> None:
    raw = "no execution marker"
    assert (
        locate_final_commitment_boundary(
            raw, [ord(char) for char in raw], CharacterTokenizer()
        )
        is None
    )


def test_treatment_induced_formatting_differences_are_detected_symmetrically() -> None:
    assert condition_formatting_differs(
        {
            "baseline": ["FINAL: 1", "FINAL: 2"],
            "treatment": ["```\nFINAL: 1\n```", "```\nFINAL: 2\n```"],
        }
    )
