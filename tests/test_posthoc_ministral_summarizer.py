import json
from pathlib import Path

from scripts.summarize_posthoc_ministral_invalidity import summarize


def _row(item: str, condition: str, rollout: int, *, correct: bool, valid: bool) -> dict:
    return {
        "item_id": item,
        "condition": condition,
        "rollout_index": rollout,
        "commitment_valid": valid,
        "semantic_evaluable": valid,
        "correct": correct,
        "generated_token_count": 10 if valid else 100,
    }


def test_remote_safe_summary_contains_no_row_identity_or_text(tmp_path: Path) -> None:
    rows = []
    for item in ("private-a", "private-b"):
        for rollout in (0, 1):
            rows.append(_row(item, "BASELINE", rollout, correct=item == "private-a", valid=True))
            rows.append(
                _row(
                    item,
                    "MEANINGFUL_FIXED",
                    rollout,
                    correct=False,
                    valid=not (item == "private-a" and rollout == 0),
                )
            )
    journal = tmp_path / "journal.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))

    result = summarize(journal)
    rendered = json.dumps(result)

    assert result["meaningful_invalidity"]["rows"] == 1
    assert result["meaningful_invalidity"]["unique_affected_items"] == 1
    assert result["pair_context"]["damage_pairs_from_invalid_meaningful"] == 2
    assert result["contains_raw_text"] is False
    assert "private-a" not in rendered
    assert "private-b" not in rendered
