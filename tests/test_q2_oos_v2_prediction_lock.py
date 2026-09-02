from __future__ import annotations

import importlib.util
import json
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


def test_oos_normative_sections_do_not_retain_v41_design_metadata() -> None:
    schedule = MODULE.build_schedule()
    normative = json.loads(MODULE.V41_NORMATIVE.read_text(encoding="utf-8"))
    metadata = json.loads(MODULE.MATRIX_METADATA.read_text(encoding="utf-8"))
    efficiency = json.loads(MODULE.EFFICIENCY.read_text(encoding="utf-8"))
    inference = {
        "item_bootstrap": {
            "resamples": 50000,
            "seed": 123,
            "role": "UNCERTAINTY_AND_SENSITIVITY_NOT_PRIMARY_SIGN_TEST_REPLACEMENT",
        },
        "fresh_old_primary": {"test": "exact one-sided Binomial upper-tail"},
        "fresh_fresh_secondary": {"method": "NODE_JACKKNIFE_PSEUDOVALUE_T"},
    }
    generation = MODULE.oos_generation_specification(
        normative["generation_specification"], schedule, "prelock"
    )
    retry = MODULE.oos_retry_resume_specification(
        normative["retry_resume_specification"], efficiency
    )
    estimands = MODULE.oos_semantic_estimands(
        normative["semantic_estimands"], metadata, inference
    )
    assert generation["schedule_and_seed"]["seed_uniqueness"] == 19200
    assert "baseline" not in generation["intervention"]
    assert "randoms" not in generation["intervention"]
    assert "legacy_helper_boundary" not in generation
    assert retry["terminal_generation_policy"][
        "hard_cap_or_repetition_stop_is_terminal"
    ]
    assert estimands["panel"]["future_conditions"] == 32
    assert estimands["panel"]["fresh_controllers"] == 16
    assert estimands["panel"]["historical_reference_controllers"] == 31
    assert estimands["bootstrap"]["resamples"] == 50000
    assert "radial_secondary" not in estimands
    assert (
        estimands["geometry_matrices"]["A0"]["blocks"]["MEDIUM"][
            "FRESH_REFERENCE"
        ]
        == metadata["matrix_hashes"]["A0_MEDIUM_FRESH_REFERENCE"]
    )
