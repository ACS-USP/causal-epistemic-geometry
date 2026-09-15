from __future__ import annotations

import json

import pytest
from scripts.plan_finalization_timing import _write_new_json, plan


def test_grid_is_fixed_and_outcome_free() -> None:
    payload = plan(replications=10)
    assert payload["status"] == "OUTCOME_FREE_SENSITIVITY_COMPLETE"
    assert len(payload["rows"]) == 60
    assert payload["simulation"]["seed"] == 20260915


def test_write_is_immutable(tmp_path) -> None:
    output = tmp_path / "planning.json"
    payload = plan(replications=10)
    _write_new_json(output, payload)
    assert json.loads(output.read_text())["rows"] == payload["rows"]
    with pytest.raises(FileExistsError):
        _write_new_json(output, payload)
