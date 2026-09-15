from __future__ import annotations

import json

import pytest
from scripts.plan_late_exposure_ablation import _write_new_json, plan


def test_plan_is_fixed_outcome_free_grid() -> None:
    payload = plan(replications=5)
    assert payload["status"] == "OUTCOME_FREE_SENSITIVITY_COMPLETE"
    assert len(payload["rows"]) == 225
    assert all("Qwen" not in row for row in payload["rows"])


def test_output_write_is_immutable(tmp_path) -> None:
    output = tmp_path / "planning.json"
    _write_new_json(output, plan(replications=5))
    payload = json.loads(output.read_text())
    assert payload["simulation"]["seed"] == 20260915
    with pytest.raises(FileExistsError):
        _write_new_json(output, payload)
