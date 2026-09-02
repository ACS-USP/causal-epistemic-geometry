from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q2_oos_fresh_controller_design/v2_final_presemantic"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_amended_lock_binds_endpoint_equivalent_policy() -> None:
    amendment = json.loads((REVIEW / "Q2_OOS_V2_SEMANTIC_EFFICIENCY_AMENDMENT.json").read_text())
    lock = json.loads((REVIEW / "Q2_OOS_V2_AMENDED_SEMANTIC_EXECUTION_LOCK.json").read_text())
    hard = json.loads((REVIEW / "HISTORICAL_HARD_CAP_REPLAY.json").read_text())
    repetition = json.loads((REVIEW / "HISTORICAL_REPETITION_REPLAY.json").read_text())

    assert hard["status"] == "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY"
    assert hard["validation_opened"] is False
    assert repetition["status"] == "Q2_OOS_V2_ENDPOINT_EQUIVALENT_REPETITION_STOP_QUALIFIED"
    assert repetition["full_historical_certification"]["scientific"][
        "exact_scientific_equivalence"
    ]
    assert repetition["full_historical_certification"]["valid_evaluable_early_stops"] == 0
    assert amendment["selected_semantic_termination_policy"] == {
        "hard_cap_only": False,
        "max_new_tokens": 4096,
        "repetition_stop": "EXTREME_MECHANICAL_REPETITION_V1",
        "terminal_interpretation": {
            "binary_error_e": 1,
            "commitment_valid": False,
            "delete_retry_or_impute": False,
            "row_retained": True,
            "semantic_evaluable": False,
        },
    }
    assert lock["amendment_identity"]["amendment_sha256"] == sha256(
        REVIEW / "Q2_OOS_V2_SEMANTIC_EFFICIENCY_AMENDMENT.json"
    )
    assert lock["future_semantic_execution"]["authorization"] == "NOT_AUTHORIZED"
    assert lock["future_semantic_execution"]["trajectories_executed"] == 0
    assert lock["safety_execution"]["hard_max_new_tokens"] == 4096
    assert lock["safety_execution"]["repetition_stop_applied"] is False
    assert lock["generation_semantics"]["hard_max_new_tokens"] == 4096
    assert (
        lock["generation_semantics"]["repetition_policy"]["name"]
        == "EXTREME_MECHANICAL_REPETITION_V1"
    )
