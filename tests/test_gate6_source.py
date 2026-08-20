from __future__ import annotations

import pytest

from epistemic_geometry.experiments.gate6_source import (
    locate_final_commitment_boundary,
    select_common_eligible,
)


class CharacterTokenizer:
    all_special_ids: list[int] = []

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
        assert not add_special_tokens
        assert return_offsets_mapping
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def decode(self, ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool = True):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(int(value)) for value in ids)


TOKENIZER = CharacterTokenizer()


@pytest.mark.parametrize(
    ("raw", "expected_index"),
    [
        ("FINAL: 2", 0),
        ("reason\nFINAL: 2", 7),
        ("**FINAL: 2**", 2),
        ("### FINAL: 2", 4),
        ("- FINAL: 2", 2),
        ("`FINAL: 2`", 1),
    ],
)
def test_final_boundary_maps_markdown_and_first_token(raw: str, expected_index: int) -> None:
    ids = [ord(char) for char in raw]
    boundary = locate_final_commitment_boundary(raw, ids, TOKENIZER)
    assert boundary is not None
    assert boundary.marker_token_index == expected_index
    assert raw[boundary.marker_text_span[0] : boundary.marker_text_span[1]].lower() == "final:"


def test_final_boundary_handles_marker_near_generation_cap() -> None:
    raw = "thinking\n" + ("x" * 128) + "\nFINAL: 2"
    ids = [ord(char) for char in raw]
    boundary = locate_final_commitment_boundary(raw, ids, TOKENIZER)
    assert boundary is not None
    assert boundary.marker_token_index == raw.index("FINAL")


@pytest.mark.parametrize(
    "raw",
    [
        "thinking without final",
        "FINAL: 1\nFINAL: 2",
        "FINAL: 1\nextra text",
        "<think>unfinished\nFINAL: 1",
        "FINAL:",
    ],
)
def test_final_boundary_rejects_missing_ambiguous_or_nonterminal_markers(raw: str) -> None:
    ids = [ord(char) for char in raw]
    assert locate_final_commitment_boundary(raw, ids, TOKENIZER) is None


def test_final_boundary_rejects_ambiguous_token_mapping() -> None:
    class MismatchTokenizer(CharacterTokenizer):
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool):
            del text, add_special_tokens, return_offsets_mapping
            return {"input_ids": [999], "offset_mapping": [(0, 1)]}

        def decode(
            self, ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool = True
        ):
            del ids, skip_special_tokens, clean_up_tokenization_spaces
            return "not the generated text"

    raw = "FINAL: 2"
    assert (
        locate_final_commitment_boundary(raw, [ord(char) for char in raw], MismatchTokenizer())
        is None
    )


def test_final_boundary_fallback_uses_first_marker_token_not_colon_token() -> None:
    class DecodeOnlyTokenizer:
        all_special_ids: list[int] = []

        def decode(
            self, ids, *, skip_special_tokens: bool, clean_up_tokenization_spaces: bool = True
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(chr(int(value)) for value in ids)

    raw = "reason\nFINAL: 2"
    boundary = locate_final_commitment_boundary(
        raw, [ord(char) for char in raw], DecodeOnlyTokenizer()
    )
    assert boundary is not None
    assert boundary.marker_token_index == raw.index("FINAL")


def _candidate(index: int, eligible: bool, reason: str = "eligible") -> dict[str, object]:
    return {
        "item_id": f"sample_{index}",
        "candidate_order": index,
        "allocation": "ORIGINAL" if index == 0 else "RESERVE",
        "eligible": eligible,
        "reason": reason,
        "condition_status": {},
    }


def test_common_selection_uses_frozen_order_and_skips_mechanical_attrition() -> None:
    selected, decisions = select_common_eligible(
        [_candidate(0, False, "missing_marker"), _candidate(1, True), _candidate(2, True)],
        target=2,
        max_ineligible=1,
        split="train",
    )
    assert [row["item_id"] for row in selected] == ["sample_1", "sample_2"]
    assert [row.candidate_item_id for row in decisions] == ["sample_0", "sample_1", "sample_2"]


def test_common_selection_stops_when_attrition_limit_is_exceeded() -> None:
    with pytest.raises(RuntimeError, match="ATTRITION_EXCEEDS_LIMIT"):
        select_common_eligible(
            [_candidate(0, False), _candidate(1, False)],
            target=1,
            max_ineligible=1,
            split="validation",
        )


def test_common_selection_reports_reserve_exhaustion() -> None:
    with pytest.raises(RuntimeError, match="RESERVE_EXHAUSTED"):
        select_common_eligible(
            [_candidate(0, True)],
            target=2,
            max_ineligible=1,
            split="validation",
        )
