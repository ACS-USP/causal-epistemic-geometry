from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_q2_oos_v2_prediction_lock",
    ROOT / "scripts/finalize_q2_oos_v2_prediction_lock.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_stable_seed_is_deterministic_and_63_bit() -> None:
    first = MODULE.stable_seed64("namespace", "item", 1)
    second = MODULE.stable_seed64("namespace", "item", 1)
    assert first == second
    assert 0 <= first < 2**63


def test_panel_has_300_items_and_two_rollouts() -> None:
    rows = MODULE.semantic_item_rollouts()
    assert len(rows) == 600
    assert len({row["item_id"] for row in rows}) == 300
    assert {row["rollout_index"] for row in rows} == {0, 1}


def test_future_schedule_is_balanced_unique_and_unopened() -> None:
    schedule = MODULE.build_schedule()
    assert schedule["row_count"] == 19200
    assert schedule["unique_logical_keys"] == 19200
    assert schedule["unique_seeds"] == 19200
    assert schedule["condition_count"] == 32
    assert schedule["semantic_outcomes"] == 0
    assert schedule["status"] == "FROZEN_NOT_AUTHORIZED_NOT_RUN"
