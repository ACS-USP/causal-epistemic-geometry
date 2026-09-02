from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from execute_q2_oos_v2_semantic import (  # noqa: E402
    EXPECTED_CONTROLLERS,
    EXPECTED_SCHEDULE_ROWS,
    validate_frozen_objects,
)


def test_frozen_oos_schedule_and_objects_are_complete_without_model() -> None:
    result = validate_frozen_objects()
    assert result["schedule_count"] == EXPECTED_SCHEDULE_ROWS == 19_200
    assert result["unique_logical_keys"] == EXPECTED_SCHEDULE_ROWS
    assert result["unique_seeds"] == EXPECTED_SCHEDULE_ROWS
    assert len(result["fresh_controller_order"]) == EXPECTED_CONTROLLERS == 16
    assert len(result["conditions"]) == 32
    assert result["semantic_outcomes_before_execution"] == 0
    assert result["correctness_inspected_before_execution"] is False
    assert result["spark1_only"] is True
    assert result["spark2_used"] is False
    assert result["runpod_used"] is False

