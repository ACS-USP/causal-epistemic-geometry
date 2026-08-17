import pytest

from epistemic_geometry.benchmarks.reasoning.base import ReasoningItem
from epistemic_geometry.benchmarks.reasoning.calibration import (
    generate_stage_a_manifests,
    select_stage_b_cells,
    stage_a_qualifies,
    stage_b_qualifies,
)
from epistemic_geometry.benchmarks.reasoning.families import FAMILY_CELLS, generate_item, oracle_for
from epistemic_geometry.benchmarks.reasoning.parser import parse_exact_integer_final
from epistemic_geometry.benchmarks.reasoning.rendering import render_reasoning
from epistemic_geometry.benchmarks.reasoning.rollouts import (
    RolloutRecord,
    rollout_seed,
    seed_schedule,
)
from epistemic_geometry.benchmarks.reasoning.runner import summarize_rollouts
from epistemic_geometry.benchmarks.reasoning.satcount_r import exact_oracle
from epistemic_geometry.benchmarks.reasoning.splits import (
    CONFIRMATORY_HOLDOUT,
    GEOMETRY_CALIBRATION,
    STEERING_DEVELOPMENT,
    generate_fresh_scientific_splits,
    generate_split,
)
from epistemic_geometry.benchmarks.reasoning.validation import validate_item
from epistemic_geometry.experiments.reasoning_agent import plurality_answer, plurality_ensemble


def test_all_reasoning_generators_have_exact_oracles_and_twins() -> None:
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            item = generate_item(family, cell, 123)
            assert oracle_for(item) == item.answer
            assert ReasoningItem.from_record(item.to_record()) == item
            report = validate_item(item)
            assert report["surface_twin_oracle_equal"] is True
            assert render_reasoning(item).answer == render_reasoning(
                item, surface="surface_twin"
            ).answer


def test_satcount_r_is_exact_not_modulo_ten() -> None:
    item = next(
        generate_item("SATCOUNT-R", "vars6_clauses10", seed)
        for seed in range(100)
        if generate_item("SATCOUNT-R", "vars6_clauses10", seed).answer > 9
    )
    assert item.answer == exact_oracle(item.spec)
    assert item.answer > 9


def test_reasoning_parser_uses_last_final_outside_thinking() -> None:
    parsed = parse_exact_integer_final("<think>FINAL: 1</think>\nFINAL: 7")
    assert parsed.valid
    assert parsed.answer == 7
    assert parse_exact_integer_final("FINAL: 3 extra").status == "INVALID_FINAL"
    assert parse_exact_integer_final("partial", truncated=True).status == "TRUNCATED_NO_FINAL"
    assert parse_exact_integer_final("<think>not closed").status == "THINKING_UNCLOSED"


def test_matched_and_independent_seed_regimes_are_distinct() -> None:
    matched = rollout_seed(9, "item", "baseline", 0, regime="matched")
    matched_treatment = rollout_seed(9, "item", "steered", 0, regime="matched")
    independent = rollout_seed(9, "item", "steered", 0, regime="independent")
    assert matched == matched_treatment
    assert matched != independent
    schedule = seed_schedule(
        ["item"], ["baseline", "steered"], base_seed=9, n_rollouts=2, regime="matched"
    )
    assert schedule[("item", "baseline", 1)] == schedule[("item", "steered", 1)]


def test_plurality_tie_rule_and_alignment() -> None:
    assert plurality_answer([2, 1], baseline_answer=1) == 1
    assert plurality_answer([2, 1]) == 1
    assert plurality_ensemble([[1, 2], [2, 2], [3, 1]]) == [1, 2]


def test_fresh_scientific_splits_are_unbalanced_and_firewalled() -> None:
    selected = {"FSM-R": {"cell": "length_4"}, "MODREG-R": {"cell": "depth_4"}}
    splits = generate_fresh_scientific_splits(selected, seed=17)
    assert len(splits) == 6
    assert {split.split_name for split in splits} == {
        GEOMETRY_CALIBRATION,
        "STEERING_DEVELOPMENT",
        CONFIRMATORY_HOLDOUT,
    }
    assert len({item.latent_id for split in splits for item in split.items}) == sum(
        len(split.items) for split in splits
    )
    holdout = generate_split("FSM-R", "length_4", CONFIRMATORY_HOLDOUT, seed=2, n_items=2)
    assert holdout.metadata["development_access"] is False
    assert generate_split("FSM-R", "length_4", GEOMETRY_CALIBRATION, seed=2, n_items=2)
    assert generate_split("FSM-R", "length_4", STEERING_DEVELOPMENT, seed=2, n_items=2)
    with pytest.raises(PermissionError):
        type(holdout).from_record(holdout.to_record())


def test_stage_a_b_rules_are_frozen_and_mechanical() -> None:
    outcomes = [
        {
            "family": "MODREG-R",
            "cell": "depth_4",
            "reasoning_budget": 512,
            "mean_accuracy": 0.55,
            "parse_success": 1.0,
            "seed_accuracy": [0.54, 0.56],
        },
        {
            "family": "FSM-R",
            "cell": "length_4",
            "reasoning_budget": 1024,
            "mean_accuracy": 0.55,
            "parse_success": 1.0,
            "seed_accuracy": [0.55, 0.55],
        },
    ]
    assert all(stage_a_qualifies(outcome) for outcome in outcomes)
    assert select_stage_b_cells(outcomes)["MODREG-R"]["reasoning_budget"] == 512
    assert stage_b_qualifies(
        {
            "mean_accuracy": 0.5,
            "parse_success": 1.0,
            "canonical_accuracy": 0.5,
            "twin_accuracy": 0.51,
            "twin_agreement": 0.8,
            "seed_accuracy_sd": 0.02,
        }
    )
    manifests = generate_stage_a_manifests(
        [("MODREG-R", "depth_4"), ("FSM-R", "length_4")], seed=3
    )
    assert len(manifests) == 6
    assert all(len(manifest.items) == 60 for manifest in manifests)


def test_calibration_summary_uses_canonical_as_primary_and_reports_twin() -> None:
    records = []
    for latent_id, canonical_answer, twin_answer in (
        ("item-a", 1, 1),
        ("item-b", 0, 1),
    ):
        for surface, answer in (("canonical", canonical_answer), ("surface_twin", twin_answer)):
            parsed = parse_exact_integer_final(f"FINAL: {answer}")
            records.append(
                RolloutRecord.from_parsed(
                    latent_id=latent_id,
                    view_id=f"MODREG-R:depth_4:{latent_id}:{surface}",
                    family="MODREG-R",
                    cell="depth_4",
                    target=1,
                    intervention_id="baseline",
                    rollout_index=0,
                    sampling_seed=7,
                    raw_text=f"FINAL: {answer}",
                    parsed=parsed,
                )
            )
    summary = summarize_rollouts(records)
    assert summary["mean_accuracy"] == 0.5
    assert summary["canonical_accuracy"] == 0.5
    assert summary["twin_accuracy"] == 1.0
    assert summary["twin_agreement"] == 0.5
