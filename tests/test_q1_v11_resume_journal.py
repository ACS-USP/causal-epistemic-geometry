"""Crash-safe V1.1 prediction journal tests without loading a real model."""

from __future__ import annotations

import json

import pytest

from epistemic_geometry.experiments.q1_v1_1 import (
    _append_q1_prediction,
    _BufferedQ1Journal,
    _load_q1_prediction_journal,
)


def _row(item_id: str = "item-1", condition: str = "baseline") -> dict[str, object]:
    return {
        "item_id": item_id,
        "condition": condition,
        "raw_output": "A",
        "normalized_output": "A",
        "target": "A",
        "correct": True,
        "parse_status": "OK",
        "metadata": {"rendered_prompt_hash": "prompt-hash"},
        "provenance": {},
    }


def test_q1_journal_recovers_truncated_tail_and_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    _append_q1_prediction(path, _row())
    with path.open("ab") as handle:
        handle.write(b'{"item_id":"broken"')
    predictions, records = _load_q1_prediction_journal(path)
    assert list(predictions) == [("item-1", "baseline")]
    assert len(records) == 1
    assert list(tmp_path.glob("predictions.quarantine*.jsonl"))

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_row()) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        _load_q1_prediction_journal(path)


def test_q1_journal_flushes_complete_chunks(tmp_path) -> None:
    path = tmp_path / "buffered.jsonl"
    journal = _BufferedQ1Journal(path, chunk_size=2)
    journal.append(_row("item-1"))
    assert not path.exists()
    journal.append(_row("item-2"))
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2
    journal.append(_row("item-3"))
    journal.flush()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
