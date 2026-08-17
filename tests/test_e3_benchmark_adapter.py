from epistemic_geometry.benchmarks.e3.benchmark import (
    calibration_benchmark_items,
    view_to_benchmark_item,
)
from epistemic_geometry.benchmarks.e3.calibration import baseline_calibration_conditions
from epistemic_geometry.benchmarks.e3.rendering import render_latent
from epistemic_geometry.benchmarks.e3.splits import generate_calibration_manifest


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
