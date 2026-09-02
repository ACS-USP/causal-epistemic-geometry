from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_q2_oos_v2_label_free", ROOT / "scripts/run_q2_oos_v2_label_free.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_bank_is_exactly_sixteen_in_frozen_order() -> None:
    names, vectors, selected = MODULE.load_selected()
    assert len(names) == 16
    assert names == selected["selected_ids"]
    assert list(vectors) == names
    assert all(value.shape == (4096,) for value in vectors.values())


def test_existing_probe_validation_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "probe.npz"
    np.savez_compressed(path, BASELINE=np.zeros((4, 7), dtype=np.float32))
    assert MODULE.existing_probe_valid(path, {"BASELINE"})
    assert not MODULE.existing_probe_valid(path, {"BASELINE", "MISSING"})
    np.savez_compressed(path, BASELINE=np.zeros((3, 7), dtype=np.float32))
    assert not MODULE.existing_probe_valid(path, {"BASELINE"})


def test_capture_source_has_no_semantic_panel_or_correctness_path() -> None:
    source = (ROOT / "scripts/run_q2_oos_v2_label_free.py").read_text(encoding="utf-8")
    assert "SEMANTIC_PANEL_MANIFEST" not in source
    assert "reference_answer" not in source
    assert "generate_reasoning" not in source
    assert '"correctness": "FORBIDDEN"' in source


def test_capture_lock_pins_runner_and_forbids_semantic_outcomes() -> None:
    lock = MODULE.read_json(MODULE.CAPTURE_LOCK)
    assert MODULE.sha256_file(ROOT / lock["capture_runner_path"]) == lock[
        "capture_runner_sha256"
    ]
    assert lock["correctness"] == "FORBIDDEN"
    assert lock["semantic_outcomes"] == 0
    assert lock["semantic_N300_trajectories"] == 0
