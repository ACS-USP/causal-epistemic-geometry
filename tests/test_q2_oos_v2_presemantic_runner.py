from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_q2_oos_v2_presemantic.py"
SPEC = importlib.util.spec_from_file_location("run_q2_oos_v2_presemantic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v2_safety_schedule_is_exact_and_has_no_baseline() -> None:
    rows = MODULE.build_schedule()
    keys = {(row["item_id"], row["condition"], row["rollout_index"]) for row in rows}
    assert len(rows) == 1_632
    assert len(keys) == 1_632
    assert len({row["item_id"] for row in rows}) == 12
    assert len({row["condition"] for row in rows}) == 68
    assert {row["rollout_index"] for row in rows} == {0, 1}
    assert all(row["condition"] != "BASELINE" for row in rows)


def test_v2_reuses_exactly_24_immutable_presemantic_baselines() -> None:
    rows = MODULE.historical_baselines()
    assert len(rows) == 24
    assert all(row["commitment_valid"] for row in rows.values())
    assert all(row["semantic_evaluable"] for row in rows.values())


def test_v2_candidate_stream_loads_all_frozen_vectors() -> None:
    vectors, manifest = MODULE.load_candidates()
    assert len(vectors) == manifest["candidate_count"] == 34
    assert list(vectors) == [row["candidate_id"] for row in manifest["candidates"]]
