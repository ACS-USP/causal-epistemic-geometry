from __future__ import annotations

import numpy as np
from scripts.audit_q2_oos_v2_repetition_efficiency import earliest_repetition_stop
from scripts.audit_q2_oos_v2_semantic_efficiency import (
    controller_partition,
    exact,
    length_summary,
    replay_endpoint,
)
from scripts.posthoc_diagnose_q2_v4_1_generation import is_extreme_mechanical_repetition


def test_controller_partition_is_deterministic_and_disjoint() -> None:
    controllers = [f"V4_DIRECTION_{index:02d}" for index in range(31)]
    first_dev, first_validation, first_digests = controller_partition(controllers)
    second_dev, second_validation, second_digests = controller_partition(
        list(reversed(controllers))
    )
    assert first_dev == second_dev
    assert first_validation == second_validation
    assert first_digests == second_digests
    assert len(first_dev) == 16
    assert len(first_validation) == 15
    assert not first_dev & first_validation
    assert first_dev | first_validation == set(controllers)


def test_hard_cap_is_terminal_answer_channel_failure() -> None:
    row = {
        "generated_token_count": 3,
        "generated_token_ids": [1, 2, 3],
        "raw_output": "FINAL: 2",
        "truncated": False,
        "runtime_error": None,
    }
    decoded_lengths: list[int] = []

    def decode(values: list[int]) -> str:
        decoded_lengths.append(len(values))
        return "FINAL: 2"

    assert replay_endpoint(row, "2", 2, decode) == (False, False, False, 1)
    assert decoded_lengths == [2]
    assert replay_endpoint(row, "2", 3, decode) == (True, True, True, 0)
    assert decoded_lengths == [2]


def test_exact_requires_all_four_row_level_endpoints() -> None:
    clean = {
        "commitment_valid": 0,
        "semantic_evaluable": 0,
        "correct": 0,
        "binary_error_e": 0,
    }
    assert exact(clean)
    for field in clean:
        changed = dict(clean)
        changed[field] = 1
        assert not exact(changed)


def test_valid_length_margin_is_twice_observed_maximum() -> None:
    summary = length_summary(np.asarray([1, 2, 3, 4], dtype=np.int64))
    assert summary["maximum"] == 4
    assert summary["required_cap_at_2x_max"] == 8


def test_online_repetition_stop_uses_unchanged_historical_criterion() -> None:
    cases = [
        [1, 2, 3] * 80,
        [1, 2, 3, 4] * 100,
        ([1] * 200) + list(range(100)),
        list(range(512)),
    ]
    for tokens in cases:
        stop = earliest_repetition_stop(tokens)
        if stop is None:
            assert not is_extreme_mechanical_repetition(tokens)
        else:
            assert stop >= 256
            assert is_extreme_mechanical_repetition(tokens[:stop])
            assert not is_extreme_mechanical_repetition(tokens[: stop - 1])
