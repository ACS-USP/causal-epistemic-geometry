import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review/q3_final_system_and_evaluation_supply"


def read_json(name: str):
    return json.loads((REVIEW / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_precheck_and_additive_steer_are_bound() -> None:
    precheck = read_json("Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json")
    steer = read_json("Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_ADDITIVE_STEER.json")
    assert precheck["status"] == "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK_FROZEN"
    assert steer["base_precheck_sha256"] == sha256(
        REVIEW / "Q3_DEVELOPMENT_PHASE_CLOSURE_PRECHECK.json"
    )
    assert steer["development_phase"]["closed"] is True
    assert (
        steer["evaluation_designs"]["U_UTILITY_ONLY"]["cost_maximum"]
        == "N * R * 2 semantic trajectories"
    )


def test_final_system_is_exact_and_release_safe() -> None:
    system = read_json("FINAL_Q3_CANDIDATE_SYSTEM_DRAFT.json")
    assert system["status"] == "DEVELOPMENT_SELECTED_NOT_EVALUATED"
    assert system["development_phase_closed"] is True
    assert [row["policy_id"] for row in system["portfolio"]["policies"]] == [
        "V4_DIRECTION_31_MEDIUM",
        "V4_DIRECTION_10_MEDIUM",
        "V4_DIRECTION_32_MEDIUM",
        "Q2_OOS_V2_DIRECTION_13_MEDIUM",
        "V4_DIRECTION_19_MEDIUM",
        "Q2_OOS_V2_DIRECTION_03_MEDIUM",
        "Q2_OOS_V2_DIRECTION_16_MEDIUM",
        "V4_DIRECTION_35_STRONG",
    ]
    assert all(row["vector_sha256"] for row in system["portfolio"]["policies"])
    assert system["champion"]["policy_id"] == "V4_DIRECTION_02_MEDIUM"
    assert system["router"]["interaction_rank"] == 2
    assert system["router"]["l2"] == 1.0
    assert system["representation"]["pca_dimension"] == 8
    assert system["deployment"]["answer_generations"] == 1
    serialized = json.dumps(system)
    assert "/private/" not in serialized
    assert "/Users/" not in serialized


def test_tier_b_audit_is_fail_closed_and_release_safe() -> None:
    audit = read_json("Q3_TIER_B_EXPOSURE_SEVERITY_AUDIT.json")
    assert sum(audit["stratum_counts"].values()) == 500
    assert audit["stratum_counts"] == {"A": 0, "B": 11, "C": 0, "D": 177, "E": 0, "F": 312}
    assert audit["eligibility"]["confirmatory_families"] == 0
    assert audit["eligibility"]["bounded_internal_validation_families"] == 11
    assert audit["release_safety"]["correctness_values"] is False
    assert "candidate_ids" not in json.dumps(audit)


def test_power_grid_and_selected_regime() -> None:
    summary = read_json("Q3_UTILITY_POWER_PRECISION_SUMMARY.json")
    assert summary["selected_primary"] == "PAIRED_FAMILY_STUDENTIZED_T"
    assert summary["selected_regime"] == "INDEPENDENT_POLICY_SEEDS_AND_N_GE_800"
    assert summary["selected_regime_t_max_fpr"] <= 0.065
    assert summary["selected_regime_t_min_coverage"] >= 0.93
    with (REVIEW / "Q3_UTILITY_POWER_PRECISION.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 800
    selected = next(
        row
        for row in rows
        if row["scenario"] == "CONSERVATIVE_COMBINED"
        and row["seed_regime"] == "INDEPENDENT"
        and row["gain"] == "0.03"
        and row["N"] == "1000"
        and row["R"] == "2"
    )
    assert float(selected["t_power"]) >= 0.80
    assert int(selected["utility_trajectories_max"]) == 4000


def test_supply_decision_and_firewall() -> None:
    decision = read_json("Q3_EVALUATION_SUPPLY_DECISION.json")
    assert decision["status"] == "Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK"
    assert decision["routes"]["II_FULLY_FRESH"] == "SELECT"
    assert decision["fresh_supply"]["allocation"] == {
        "confirmation": 1000,
        "qualification": 300,
        "reserve": 300,
        "total_families": 1600,
    }
    assert decision["q3"] == "NOT_RUN"
    assert decision["firewall"] == {
        "fresh_evaluation_outcomes_inspected": False,
        "new_qwen_forwards": 0,
        "new_semantic_trajectories": 0,
        "spark1_gpu": False,
        "spark2": False,
    }


def test_runtime_designs_are_not_conflated() -> None:
    runtime = read_json("Q3_RUNTIME_ACCOUNTING.json")
    assert runtime["utility_only_selected_N1000_R2"]["trajectories"] == 4000
    assert runtime["full_bank_diagnostic_N1000_R2"]["trajectories"] == 16000
    assert runtime["portfolio_attribution_N1000_R2"]["20"]["trajectories"] == 44000
    assert runtime["interpretation"]["deployment_calls"] == 1
    assert runtime["interpretation"]["full_bank_required_for_primary_utility"] is False


def test_fresh_instrument_is_design_only() -> None:
    draft = read_json("Q3_FRESH_INSTRUMENT_DESIGN_DRAFT.json")
    assert draft["allocation_status"] == "DESIGN_ONLY_NO_IDS_OR_SEEDS_GENERATED"
    assert draft["terminal_firewall"] == {
        "final_items_generated": 0,
        "final_seeds_generated": 0,
        "holdout_allocated": 0,
        "semantic_trajectories": 0,
    }
    assert draft["freshness_and_diversity"]["same_program_different_input_is_new_family"] is False
    assert draft["instrument_qualification"]["confirmation_reuse"] is False


def test_review_ends_in_exact_terminal_state() -> None:
    text = (ROOT / "docs/Q3_FINAL_SYSTEM_AND_EVALUATION_SUPPLY_REVIEW.md").read_text()
    assert text.rstrip().endswith("`Q3_FRESH_INSTRUMENT_DESIGN_READY_FOR_PRELOCK`")
    assert "Q3 remains `NOT_RUN`" in text
