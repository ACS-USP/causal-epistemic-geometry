from __future__ import annotations

from scripts.run_q2_oos_v2_semantic import frozen_terminal_metadata

from epistemic_geometry.inference.terminal_policies import (
    EXTREME_REPETITION_NAME,
    extreme_mechanical_repetition_v1,
)
from epistemic_geometry.types import BackendOutput


def test_frozen_repetition_policy_matches_historical_examples() -> None:
    assert not extreme_mechanical_repetition_v1([1, 2, 3] * 80)
    assert extreme_mechanical_repetition_v1([1, 2, 3, 4] * 100)
    assert extreme_mechanical_repetition_v1(([1] * 200) + list(range(100)))
    assert not extreme_mechanical_repetition_v1(list(range(512)))


def test_terminal_metadata_fails_closed_and_retains_row() -> None:
    output = BackendOutput(
        raw_output="synthetic fixture",
        metadata={
            "generated_token_count": 256,
            "terminal_policy": {
                "name": EXTREME_REPETITION_NAME,
                "triggered": True,
                "trigger_token_count": 256,
            },
        },
    )
    metadata = frozen_terminal_metadata(output)
    assert metadata["truncated"] is True
    assert metadata["terminal_answer_channel_failure"] is True
    assert metadata["commitment_valid_if_terminal_failure"] is False
    assert metadata["semantic_evaluable_if_terminal_failure"] is False
    assert metadata["binary_error_e_if_terminal_failure"] == 1
