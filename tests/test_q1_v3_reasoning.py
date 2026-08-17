import json
from dataclasses import dataclass

import pytest

from epistemic_geometry.benchmarks.reasoning.base import ReasoningItem
from epistemic_geometry.benchmarks.reasoning.calibration import (
    generate_stage_a_manifests,
    select_stage_b_cells,
    stage_a_qualifies,
    stage_b_qualifies,
)
from epistemic_geometry.benchmarks.reasoning.engines import (
    BATCHED_REASONING,
    MAX_BUDGET_PREFIX_REUSE,
    SERIAL_REASONING_REFERENCE,
    derive_budget_outputs,
    deterministic_length_batches,
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
from epistemic_geometry.config import load_config
from epistemic_geometry.experiments.reasoning_agent import plurality_answer, plurality_ensemble
from epistemic_geometry.types import BackendOutput


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


def test_max_budget_prefix_reuse_derives_each_budget_independently() -> None:
    source = BackendOutput(
        raw_output="unused",
        metadata={"generated_token_ids": list(range(12)), "generation_seed": 17},
    )
    outputs = derive_budget_outputs(
        source,
        view_id="FSM-R:length_4:item:canonical",
        sampling_seed=17,
        source_max_budget=12,
        budgets=(4, 8, 12),
        decode_tokens=lambda ids: "TOKENS:" + ",".join(map(str, ids)),
    )
    assert [outputs[budget].metadata["generated_token_ids"] for budget in (4, 8, 12)] == [
        [0, 1, 2, 3],
        list(range(8)),
        list(range(12)),
    ]
    assert outputs[4].raw_output == "TOKENS:0,1,2,3"
    assert outputs[4].metadata["derived_from_prefix"] is True
    assert outputs[12].metadata["derived_from_prefix"] is False
    assert outputs[4].metadata["physical_generation_id"] == outputs[12].metadata[
        "physical_generation_id"
    ]


def test_prefix_reuse_requires_exact_generated_tokens() -> None:
    with pytest.raises(ValueError, match="generated_token_ids"):
        derive_budget_outputs(
            BackendOutput(raw_output="FINAL: 1"),
            view_id="item:canonical",
            sampling_seed=1,
            source_max_budget=4,
            budgets=(4,),
            decode_tokens=lambda ids: str(ids),
        )


def test_reasoning_length_batch_planner_is_deterministic_and_budget_independent() -> None:
    lengths = [("long", 100), ("short-b", 10), ("short-a", 10), ("mid", 50)]
    assert deterministic_length_batches(lengths, batch_size=2, max_padded_tokens=120) == [
        ("short-a", "short-b"),
        ("mid",),
        ("long",),
    ]
    assert SERIAL_REASONING_REFERENCE != MAX_BUDGET_PREFIX_REUSE
    with pytest.raises(ValueError, match="exceeds max_padded_tokens"):
        deterministic_length_batches([("too-long", 121)], batch_size=2, max_padded_tokens=120)


def test_prefix_runner_reduces_physical_generations_without_reducing_rows(
    tmp_path, monkeypatch
) -> None:
    from epistemic_geometry.benchmarks.reasoning import runner

    manifests = generate_stage_a_manifests([("FSM-R", "length_4")], seed=31)
    payload = {
        "manifests": {
            f"{split.family}/{split.cell}/{split.reasoning_budget}": split.to_record()
            for split in manifests
        }
    }
    manifest_path = tmp_path / "stage_a.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    config = load_config("configs/q1_v3_reasoning_instrument.example.yaml")

    @dataclass
    class FakeTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "<think>done</think>\nFINAL: 4" if ids else ""

    class FakeBackend:
        tokenizer = FakeTokenizer()

        def __init__(self):
            self.calls = []
            self.batch_calls = []

        def generate_reasoning_view(self, view, *, sampling_seed, max_new_tokens):
            self.calls.append((view.view_id, sampling_seed, max_new_tokens))
            return BackendOutput(
                raw_output="<think>done</think>\nFINAL: 4",
                metadata={
                    "generated_token_ids": list(range(max_new_tokens)),
                    "generation_seed": sampling_seed,
                },
            )

        def generate_reasoning_batch(
            self, rows, *, max_new_tokens, batch_size, max_prefill_tokens
        ):
            self.batch_calls.append(
                (len(rows), max_new_tokens, batch_size, max_prefill_tokens)
            )
            return [
                self.generate_reasoning_view(
                    view, sampling_seed=seed, max_new_tokens=max_new_tokens
                )
                for view, seed in rows
            ]

        def provenance(self):
            return {"model": "fake"}

    fake = FakeBackend()
    monkeypatch.setattr(runner, "build_backend", lambda _config: fake)
    output = runner.run_baseline_calibration(
        config,
        manifest_path,
        tmp_path / "run",
        inference_engine=MAX_BUDGET_PREFIX_REUSE,
        max_items=1,
    )
    rows = [json.loads(line) for line in (output / "rollouts.jsonl").read_text().splitlines()]
    assert len(fake.calls) == 2
    assert len(rows) == 6
    assert {row["generation_config"]["max_new_tokens"] for row in rows} == {512, 1024, 2048}
    assert len({row["physical_generation_id"] for row in rows}) == 2
    assert sum(row["derived_from_prefix"] for row in rows) == 4
    assert all(row["natural_completion_length"] == 2048 for row in rows)

    fake.calls.clear()
    batched_output = runner.run_baseline_calibration(
        config,
        manifest_path,
        tmp_path / "batched-run",
        inference_engine=BATCHED_REASONING,
        max_items=1,
    )
    batched_rows = [
        json.loads(line)
        for line in (batched_output / "rollouts.jsonl").read_text().splitlines()
    ]
    assert fake.batch_calls == [(2, 2048, 1, 8192)]
    assert len(fake.calls) == 2
    assert batched_rows == rows
    batched_manifest = json.loads((batched_output / "manifest.json").read_text())
    assert batched_manifest["inference_engine"] == BATCHED_REASONING
    assert batched_manifest["inference_engine_version"] == "q1-v3-reasoning-engine-v1"
    assert batched_manifest["inference_settings"]["batch_size"] == 1


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


def test_stage_a_budgets_share_exactly_one_latent_item_set_per_cell() -> None:
    eligible = [(family, cell) for family, cells in FAMILY_CELLS.items() for cell in cells]
    manifests = generate_stage_a_manifests(eligible, seed=20260817)
    grouped: dict[tuple[str, str], list] = {}
    for manifest in manifests:
        grouped.setdefault((manifest.family, manifest.cell), []).append(manifest)

    assert len(grouped) == len(eligible)
    all_cell_ids: set[str] = set()
    for (_family, _cell), cell_manifests in grouped.items():
        assert [manifest.reasoning_budget for manifest in cell_manifests] == [512, 1024, 2048]
        assert len({manifest.seed for manifest in cell_manifests}) == 1
        latent_sets = [
            tuple(item.latent_id for item in manifest.items) for manifest in cell_manifests
        ]
        assert latent_sets[0] == latent_sets[1] == latent_sets[2]
        assert len(latent_sets[0]) == len(set(latent_sets[0])) == 60
        assert all_cell_ids.isdisjoint(latent_sets[0])
        all_cell_ids.update(latent_sets[0])
        assert len({manifest.metadata["paired_item_set_hash"] for manifest in cell_manifests}) == 1


def test_stage_a_rollout_seed_schedule_is_budget_invariant() -> None:
    manifests = generate_stage_a_manifests([("FSM-R", "length_4")], seed=31)
    by_budget = {manifest.reasoning_budget: manifest for manifest in manifests}
    for index, _item in enumerate(by_budget[512].items):
        for rollout_index in (0, 1):
            seeds = {
                budget: rollout_seed(
                    99,
                    by_budget[budget].items[index].latent_id,
                    "baseline",
                    rollout_index,
                    regime="independent",
                )
                for budget in (512, 1024, 2048)
            }
            assert len(set(seeds.values())) == 1
            assert by_budget[512].items[index].latent_id == by_budget[1024].items[index].latent_id


def test_stage_a_budget_remains_execution_provenance() -> None:
    from epistemic_geometry.benchmarks.reasoning.runner import _generation_config
    from epistemic_geometry.config import load_config

    config = load_config("configs/q1_v3_reasoning_instrument.example.yaml")
    assert [
        _generation_config(config, budget)["max_new_tokens"] for budget in (512, 1024, 2048)
    ] == [512, 1024, 2048]


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
