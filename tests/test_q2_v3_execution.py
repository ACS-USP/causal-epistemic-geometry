from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location("run_q2_v3", ROOT / "scripts/run_q2_v3.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_execution_schedules_match_runner_contract() -> None:
    source = json.loads(
        (ROOT / "review/q2_v3_radial_angular_freeze/SOURCE_QUALIFICATION_SCHEDULE.json").read_text()
    )
    shell = json.loads(
        (ROOT / "review/q2_v3_radial_angular_freeze/SHELL_CALIBRATION_SCHEDULE.json").read_text()
    )
    panel = json.loads(
        (ROOT / "review/q2_v3_radial_angular_freeze/EVALUATION_SCHEDULE.json").read_text()
    )
    assert len(source["rows"]) == 480
    assert {"item_id", "family", "polarity", "rollout_index", "seed"} <= set(
        source["rows"][0]
    )
    assert len(shell["rows"]) == 504
    assert {"item_id", "condition", "rollout_index", "matched_seed"} <= set(shell["rows"][0])
    assert len(panel["rows"]) == 10_000
    assert len(
        {(row["item_id"], row["condition"], row["rollout_index"]) for row in panel["rows"]}
    ) == 10_000


def test_q2_v3_null_constructor_projects_against_full_span() -> None:
    runner = _runner()
    vectors: dict[str, np.ndarray] = {}
    for index, family in enumerate(runner.SOURCE_FAMILIES):
        for location_index, location in enumerate(runner.LOCATIONS):
            vector = np.zeros(32, dtype=np.float64)
            vector[index * 2 + location_index] = 1.0
            vectors[runner.base_direction_id(family.family_id, location)] = vector
    nulls, geometry = runner._construct_nulls(vectors)
    meaningful = np.stack(list(vectors.values()))
    for value in nulls.values():
        assert np.isclose(np.linalg.norm(value), 1.0)
        assert np.max(np.abs(meaningful @ value)) <= 1e-6
    assert geometry["max_span_absolute_cosine"] <= 1e-6
    assert geometry["pairwise_absolute_cosine"] <= 1e-6


def test_q2_v3_cross_family_edge_count_is_frozen_40() -> None:
    runner = _runner()
    names = list(runner.meaningful_controller_ids())[:10]
    deployment = {
        name: {"family_id": family.family_id}
        for family in runner.SOURCE_FAMILIES
        for name in names
        if name.startswith(f"MEAN_{family.family_id}_")
    }
    assert len(runner._cross_edges(names, deployment)) == 40
