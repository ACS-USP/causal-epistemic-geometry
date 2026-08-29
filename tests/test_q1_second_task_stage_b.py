from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from epistemic_geometry.experiments import q1_second_task as q1s
from epistemic_geometry.experiments import q1_second_task_stage_b as stage_b

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = (
    ROOT
    / "review/q1_second_task_spark2_design/amendment1_hierarchical_unit"
    / "STAGE_B_SCHEDULE.json"
)


def test_frozen_stage_b_schedule_is_complete_and_unique() -> None:
    rows = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    stage_b.validate_schedule(rows)
    assert len(rows) == 5_720
    assert len({stage_b.logical_key(row) for row in rows}) == 5_720
    assert len({row["seed"] for row in rows}) == 5_720


def test_primary_bootstrap_is_deterministic_and_matches_direct_point_scale() -> None:
    rng = np.random.default_rng(92)
    baseline = rng.integers(0, 2, size=(130, 4)).astype(float)
    conditions = {
        name: rng.integers(0, 2, size=(130, 4)).astype(float)
        for name in ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    }
    first = stage_b.primary_bootstrap(baseline, conditions, resamples=200, seed=71)
    second = stage_b.primary_bootstrap(baseline, conditions, resamples=200, seed=71)
    assert first == second
    point = q1s.r_rollout_estimands(baseline, conditions["MEANINGFUL_FIXED_QWEN_L27_D75"])["C"]
    assert first["C_meaningful"]["q025"] <= point <= first["C_meaningful"]["q975"]


def test_split_half_checks_use_predesignated_rollout_pairs() -> None:
    rng = np.random.default_rng(11)
    baseline = rng.integers(0, 2, size=(130, 4)).astype(float)
    conditions = {
        name: baseline.copy() for name in ("MEANINGFUL_FIXED_QWEN_L27_D75", *stage_b.RANDOM_NAMES)
    }
    conditions["MEANINGFUL_FIXED_QWEN_L27_D75"][:, (0, 2)] = 0.0
    result = stage_b.split_half_checks(baseline, conditions)
    assert set(result) == {"A", "B"}
    assert (
        result["A"]["C_meaningful"]
        == q1s.r_rollout_estimands(
            baseline[:, (0, 1)],
            conditions["MEANINGFUL_FIXED_QWEN_L27_D75"][:, (0, 1)],
        )["C"]
    )
    assert (
        result["B"]["C_meaningful"]
        == q1s.r_rollout_estimands(
            baseline[:, (2, 3)],
            conditions["MEANINGFUL_FIXED_QWEN_L27_D75"][:, (2, 3)],
        )["C"]
    )


def test_classification_is_mechanical_and_safety_separate() -> None:
    summaries = {
        "BASELINE": {
            "commitment_validity": 1.0,
            "semantic_evaluability": 1.0,
            "accuracy": 0.7,
        },
        "MEANINGFUL_FIXED_QWEN_L27_D75": {
            "commitment_validity": 0.96,
            "semantic_evaluability": 0.96,
            "accuracy": 0.61,
        },
    }
    estimands = {
        "MEANINGFUL_FIXED_QWEN_L27_D75": {"C": 0.2},
        **{name: {"C": 0.05} for name in stage_b.RANDOM_NAMES},
    }
    intervals = {
        "C_meaningful": {"q025": 0.01, "q975": 0.3},
        "delta_C_nullmean": {"q025": 0.01, "q975": 0.2},
    }
    halves = {"A": {"passes": True}, "B": {"passes": True}}
    result = stage_b.classify(
        summaries=summaries,
        estimands=estimands,
        intervals=intervals,
        split_halves=halves,
    )
    assert result["classification"] == "Q1_SECOND_TASK_FIXED_CONTROLLER_PASS"
    summaries["MEANINGFUL_FIXED_QWEN_L27_D75"]["accuracy"] = 0.59
    result = stage_b.classify(
        summaries=summaries,
        estimands=estimands,
        intervals=intervals,
        split_halves=halves,
    )
    assert result["classification"] == "Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL"
