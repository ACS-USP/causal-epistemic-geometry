from epistemic_geometry.benchmarks.e3.benchmark import (
    calibration_benchmark_items,
    view_to_benchmark_item,
)
from epistemic_geometry.benchmarks.e3.calibration import (
    baseline_calibration_conditions,
    run_baseline_calibration,
)
from epistemic_geometry.benchmarks.e3.rendering import render_latent
from epistemic_geometry.benchmarks.e3.splits import generate_calibration_manifest
from epistemic_geometry.types import BackendOutput, PreparedChoiceItem


def test_view_adapter_exposes_semantic_candidates_not_answer_slots() -> None:
    manifest = generate_calibration_manifest("MODREG10", "depth_4", seed=7, n_items=20)
    item = view_to_benchmark_item(render_latent(manifest.items[0]))
    assert item.metadata["candidate_labels"] == [str(digit) for digit in range(10)]
    assert item.metadata["semantic_option_ids"] == list(range(10))
    assert item.metadata["e3_10"] is True


def test_calibration_adapter_has_only_baseline_condition() -> None:
    conditions = baseline_calibration_conditions(17)
    assert len(conditions) == 1
    spec, vector = conditions[0]
    assert vector is None
    assert spec["condition"] == "baseline"
    assert spec["steering"] is False
    assert spec["alpha"] == 0.0


def test_calibration_views_are_three_per_latent() -> None:
    manifest = generate_calibration_manifest("FSM10", "length_4", seed=7, n_items=20)
    items = calibration_benchmark_items(manifest)
    assert len(items) == 60
    assert {item.metadata["response_channel"] for item in items} == {"decimal", "number_word"}
    assert {item.metadata["surface"] for item in items} == {"canonical", "surface_twin"}


def test_baseline_calibration_records_prepared_view_id() -> None:
    manifest = generate_calibration_manifest("FSM10", "length_4", seed=7, n_items=10)

    class FakeChoiceBackend:
        def prepare_choice_items(self, items):
            return [
                PreparedChoiceItem(
                    item_id=item.id,
                    target=item.target,
                    metadata=item.metadata,
                    rendered_prompt=item.prompt,
                    rendered_prompt_hash=item.metadata["rendered_prompt_hash"],
                    prompt_ids=(1, 2),
                    candidate_labels=tuple(item.metadata["candidate_labels"]),
                    candidate_token_ids={
                        label: (index,)
                        for index, label in enumerate(item.metadata["candidate_labels"])
                    },
                    context_compatible_candidate_ids={
                        label: (index,)
                        for index, label in enumerate(item.metadata["candidate_labels"])
                    },
                    semantic_option_ids=tuple(item.metadata["semantic_option_ids"]),
                )
                for item in items
            ]

        def predict_choice_batch(self, prepared, conditions, *, mode=None):
            spec, _vector = conditions[0]
            return [
                (
                    item,
                    spec,
                    BackendOutput(
                        raw_output="0",
                        metadata={
                            "candidate_score_semantics": "candidate_logits_no_vocab_normalization",
                            "candidate_scores": {
                                label: float(index)
                                for index, label in enumerate(item.candidate_labels)
                            },
                            "rendered_prompt_hash": item.rendered_prompt_hash,
                            "execution_engine": "test",
                        },
                    ),
                )
                for item in prepared
            ]

    rows = run_baseline_calibration(FakeChoiceBackend(), manifest)
    assert len(rows) == 30
    expected_view_ids = {
        item.id for item in calibration_benchmark_items(manifest)
    }
    assert {row.metadata["view_id"] for row in rows} == expected_view_ids
