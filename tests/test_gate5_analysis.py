from __future__ import annotations

import numpy as np
from scripts.analyze_gate5_source_duration import (
    _evaluation_bootstrap,
    _evaluation_metrics,
    _source_bootstrap,
)

from epistemic_geometry.experiments.gate5 import CONDITIONS, SOURCE_CONDITIONS


def _row(item_id: str, condition: str, rollout: int, status: str) -> dict[str, object]:
    return {
        "item_id": item_id,
        "condition": condition,
        "rollout_index": rollout,
        "status": status,
        "parsed_answer": "answer" if status.startswith("VALID") else None,
        "generated_token_count": 3,
        "generated_token_ids": [1, 2, 3],
    }


def test_gate5_source_bootstrap_resamples_items_with_replacement() -> None:
    rows = [
        _row(item_id, condition, rollout, "VALID_CORRECT")
        for item_id in (f"item-{index}" for index in range(40))
        for condition in SOURCE_CONDITIONS
        for rollout in (0, 1)
    ]
    result = _source_bootstrap(rows, 25)
    assert result["n_resamples"] == 25
    assert set(result["intervals"]) == {"X", "W", "S", "accuracy_difference"}
    assert result["intervals"]["X"]["estimate"] == 0.0


def test_gate5_evaluation_bootstrap_uses_validity_not_error_matrix() -> None:
    rows: list[dict[str, object]] = []
    for item_index in range(60):
        item_id = f"item-{item_index}"
        for condition in CONDITIONS:
            for rollout in (0, 1):
                status = "VALID_CORRECT"
                if condition == "SUSTAINED_PLUS" and item_index == 0:
                    status = "INVALID_FORMAT"
                rows.append(_row(item_id, condition, rollout, status))
    estimates, matrices, valid_matrices, _ = _evaluation_metrics(rows)
    bootstrap = _evaluation_bootstrap(matrices, valid_matrices, 25)
    assert estimates["SUSTAINED_PLUS"]["validity"] < 1.0
    assert bootstrap["intervals"]["SUSTAINED_PLUS:validity_change"]["estimate"] < 0.0
    assert np.isfinite(bootstrap["intervals"]["SUSTAINED_PLUS:D"]["estimate"])
