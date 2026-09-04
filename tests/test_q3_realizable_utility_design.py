from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_realizable_utility_design"


def load(name: str) -> dict:
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_q3_precheck_is_frozen_and_nonexecuting() -> None:
    precheck = load("Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK.json")
    assert precheck["classification"] == "Q3_REALIZABLE_UTILITY_DESIGN_PRECHECK_FROZEN"
    assert precheck["automatic_execution"] == {
        "future_holdout_outcomes": 0,
        "gpu_or_model_inference": False,
        "prompt_activation_capture": 0,
        "q3_semantic_trajectories": 0,
    }
    assert (
        precheck["future_holdout_firewall"]["correctness_or_model_outcomes_may_be_opened"] is False
    )


def test_exposure_ledger_is_release_safe_and_complete() -> None:
    ledger = load("ITEM_EXPOSURE_LEDGER.json")
    assert ledger["summary"]["total_items"] == 800
    assert ledger["summary"]["q2_closed_development_items"] == 300
    assert ledger["summary"]["globally_untouched_items"] == 23
    assert ledger["summary"]["no_candidate_policy_outcome_items"] == 500
    assert ledger["raw_prompt_or_reference_content_included"] is False
    assert len(ledger["items"]) == 800
    forbidden = {"prompt", "reference_answer", "raw_output", "parsed_answer", "correct"}
    assert all(not (forbidden & set(item)) for item in ledger["items"])


def test_holdout_ruling_fails_closed() -> None:
    feasibility = load("FRESH_HOLDOUT_FEASIBILITY.json")
    assert feasibility["future_outcomes_inspected"] is False
    assert feasibility["future_holdout_permanently_allocated"] is False
    assert feasibility["power_requirement"]["minimum_N_from_grid"] == 800
    assert feasibility["tier_a_power_adequate"] is False
    assert feasibility["tier_b_power_adequate"] is False
    assert feasibility["provisional_ruling"] == "Q3_FRESH_HOLDOUT_INSUFFICIENT"


def test_routes_are_development_only_and_not_ready() -> None:
    routes = load("ROUTER_MECHANISM_COMPARISON.json")
    assert routes["labels"] == ["DEVELOPMENT_ONLY", "POST_CLOSED_RESULT_PLANNING"]
    assert routes["feasible_route_a_count"] == 0
    assert routes["best_feasible_route_a"] is None
    assert routes["route_b_feasible"] is False
    assert routes["route_c_feasible"] is False
    assert all(row["outer_folds_present"] == 5 for row in routes["route_a_results"][:3])


def test_recommended_protocol_does_not_authorize_q3() -> None:
    protocol = load("RECOMMENDED_PROTOCOL_DRAFT.json")
    assert protocol["status"] == "DRAFT_AWAITING_PRINCIPAL_PRELOCK"
    assert protocol["final_design_ruling"] == "Q3_FRESH_HOLDOUT_INSUFFICIENT"
    assert protocol["execution_authorized"] is False
    assert protocol["future_holdout"]["permanently_allocated"] is False


def test_artifact_hash_manifest_matches() -> None:
    manifest = load("ARTIFACT_HASHES.json")
    assert manifest["classification"] == "Q3_FRESH_HOLDOUT_INSUFFICIENT"
    assert manifest["q3_semantic_trajectories"] == 0
    assert manifest["raw_text_included"] is False
    assert manifest["future_holdout_outcomes_included"] is False
    immutable = 0
    for relative, expected in manifest["artifacts"].items():
        path = ROOT / relative
        assert path.is_file()
        assert len(expected) == 64
        if relative.startswith("review/q3_realizable_utility_design/"):
            assert sha256(path) == expected
            immutable += 1
    # Public state/docs are intentionally mutable derived projections. Their
    # hashes remain in the historical manifest as a closeout snapshot, while
    # only the frozen review namespace must continue to match byte-for-byte.
    assert immutable >= 10
