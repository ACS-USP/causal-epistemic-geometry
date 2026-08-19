"""The dense-code pilot must stop before model execution without an approved sandbox."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).parents[1] / "scripts/prepare_q1_dense_code_pilot.py"
SPEC = importlib.util.spec_from_file_location("prepare_q1_dense_code_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

STATUS = MODULE.STATUS
prepare = MODULE.prepare


def test_dense_code_gate_blocks_without_isolated_evaluator(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE.shutil, "which", lambda _: None)

    manifest = prepare(tmp_path / "pilot")

    assert manifest["status"] == STATUS
    assert manifest["model_inference"] is False
    assert manifest["pod_started"] is False
    assert manifest["candidate_selection"] is None
    assert json.loads((tmp_path / "pilot/MANIFEST.json").read_text())["item_selection"] is None
    assert (tmp_path / "pilot/TEST_VECTOR_RESULTS.jsonl").read_text() == ""
