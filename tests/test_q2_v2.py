from __future__ import annotations

import numpy as np

from epistemic_geometry.experiments.q2_controller_heldout_v2 import (
    DOSE_NAMES,
    LOCATIONS,
    SIGNS,
    build_null_bank,
    calibration_schedule,
    common_schedule,
    dose_condition_id,
    meaningful_ids,
    source_schedule,
    validate_null_bank,
    validate_schedule,
)


def test_v2_nulls_project_against_orthonormal_meaningful_span() -> None:
    base = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    meaningful = {f"v{index}": row / np.linalg.norm(row) for index, row in enumerate(base)}
    nulls, metadata = build_null_bank(meaningful, (11, 12, 13, 14))
    checks = validate_null_bank(meaningful, nulls)
    assert metadata["basis_rank"] == 3
    assert checks["span_orthogonality_pass"]
    assert checks["pairwise_null_orthogonality_pass"]
    assert checks["unit_norm_pass"]


def test_v2_source_calibration_and_common_schedules_are_complete() -> None:
    axes = ("AXIS_A", "AXIS_B", "AXIS_C", "AXIS_D", "AXIS_E")
    controllers = meaningful_ids(axes)
    source = source_schedule(["source_a", "source_b"])
    assert len(source) == 2 * 2 * 6 * len(SIGNS)
    source_keys = {
        (row["item_id"], row["axis_id"], row["polarity"], row["rollout_index"])
        for row in source
    }
    assert len(source_keys) == len(source)
    assert len({row["seed"] for row in source}) == len(source)

    calibration = calibration_schedule(["cal_a"], controllers)
    calibration_conditions = [
        "BASELINE",
        *[
            dose_condition_id(controller, dose)
            for controller in controllers
            for dose in DOSE_NAMES
        ],
    ]
    expected_calibration = [
        ("cal_a", condition, 0) for condition in calibration_conditions
    ]
    assert len(calibration) == len(expected_calibration)
    validate_schedule(calibration, expected_calibration, require_unique_seeds=False)

    common = common_schedule(["item_a", "item_b"], controllers)
    expected_common = [
        (item, condition, rollout)
        for item in ("item_a", "item_b")
        for condition in ["BASELINE", *controllers]
        for rollout in (0, 1)
    ]
    assert len(common) == len(expected_common)
    validate_schedule(common, expected_common)


def test_v2_meaningful_ids_have_two_locations_and_signs() -> None:
    ids = meaningful_ids(("AXIS_A", "AXIS_B"))
    assert len(ids) == 8
    assert {name.split("_")[-1] for name in ids} == set(SIGNS)
    assert {"_".join(name.split("_")[-3:-1]) for name in ids} == set(LOCATIONS)
