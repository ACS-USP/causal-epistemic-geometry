from epistemic_geometry.benchmarks.e3.modreg10 import dependency_analysis
from epistemic_geometry.benchmarks.e3.rendering import render_latent
from epistemic_geometry.benchmarks.e3.splits import (
    CALIBRATION_SPLIT,
    FAMILY_CELLS,
    generate_balanced_items_with_stats,
    generate_latent,
)
from epistemic_geometry.benchmarks.e3.structural import (
    reachability_details,
    shallow_heuristic_prediction,
    structural_features,
    validate_structural_item,
)


def test_modreg_dependency_cone_tracks_register_expansion_and_swap() -> None:
    spec = {
        "initial": [0, 0, 0, 0],
        "operations": [
            {"op": "ADD_CONST", "r": "R1", "c": 1},
            {"op": "ADD_REG", "dst": "R0", "src": "R1"},
            {"op": "SWAP", "r1": "R0", "r2": "R2"},
            {"op": "MUL_UNIT", "r": "R2", "u": 3},
        ],
        "query": "R2",
    }
    analysis = dependency_analysis(spec)
    assert analysis["dependency_cone"] == [0, 1, 2, 3]
    assert analysis["effective_operation_count"] == 4


def test_generated_modreg_cells_realize_effective_depth() -> None:
    for depth in (4, 8, 12, 16):
        item = generate_latent("MODREG10", f"depth_{depth}", depth)
        validity = validate_structural_item(item)
        assert validity["valid"]
        assert validity["effective_operation_count"] >= depth - 2


def test_modreg_dependency_cone_matches_perturbation_effects() -> None:
    spec = {
        "initial": [2, 3, 5, 7],
        "operations": [
            {"op": "ADD_CONST", "r": "R3", "c": 1},
            {"op": "ADD_CONST", "r": "R0", "c": 4},
            {"op": "ADD_REG", "dst": "R2", "src": "R3"},
            {"op": "MUL_UNIT", "r": "R2", "u": 3},
        ],
        "query": "R2",
    }
    analysis = dependency_analysis(spec)
    assert analysis["dependency_cone"] == [0, 2, 3]

    from epistemic_geometry.benchmarks.e3.modreg10 import oracle

    for index in range(len(spec["operations"])):
        mutated = {**spec, "operations": [dict(op) for op in spec["operations"]]}
        operation = mutated["operations"][index]
        if operation["op"] == "ADD_CONST":
            operation["c"] = int(operation["c"]) + 1
        elif operation["op"] == "ADD_REG":
            operation["src"] = "R1" if operation["src"] != "R1" else "R0"
        else:
            operation["u"] = 7 if operation["u"] != 7 else 9
        changed = oracle(mutated) != oracle(spec)
        assert changed is (index in analysis["dependency_cone"])


def test_structural_features_exclude_target_and_heuristics_are_explicit() -> None:
    for family, cell in ((family, cells[0]) for family, cells in FAMILY_CELLS.items()):
        item = generate_latent(family, cell, 12)
        features = structural_features(item)
        assert "target" not in features
        assert shallow_heuristic_prediction(item) in range(10)


def test_reachability_details_report_bounded_frontiers() -> None:
    details = reachability_details(
        {"edges": [[0, 1], [1, 2], [2, 3], [0, 4]], "source": 0, "max_hops": 2}
    )
    assert details["reachable_subgraph_size"] == 3
    assert details["frontier_expansions"] == 2
    assert details["shortest_path_depth_counts"] == {"1": 2, "2": 1}


def test_balancing_stats_and_namespace_hashes_are_explicitly_disjoint() -> None:
    train, stats = generate_balanced_items_with_stats(
        "FSM10", "length_4", 100, 20260817, split_name="SHORTCUT_TRAIN"
    )
    test, _ = generate_balanced_items_with_stats(
        "FSM10", "length_4", 100, 20260817, split_name="SHORTCUT_TEST"
    )
    calibration, _ = generate_balanced_items_with_stats(
        "FSM10", "length_4", 100, 20260817, split_name=CALIBRATION_SPLIT
    )
    assert stats.accepted == 100
    assert stats.accepted_counts_by_target == {str(digit): 10 for digit in range(10)}
    assert all(
        0.0 < rate <= 1.0 for rate in stats.to_record()["acceptance_rate_by_target"].values()
    )
    for attribute in ("latent_id", "latent_seed"):
        assert not (
            {getattr(item, attribute) for item in train}
            & {getattr(item, attribute) for item in test}
        )
        assert not (
            {getattr(item, attribute) for item in train}
            & {getattr(item, attribute) for item in calibration}
        )
    prompt_hashes = [
        {render_latent(item).prompt_hash for item in namespace}
        for namespace in (train, test, calibration)
    ]
    assert not (prompt_hashes[0] & prompt_hashes[1])
    assert not (prompt_hashes[0] & prompt_hashes[2])
    assert not (prompt_hashes[1] & prompt_hashes[2])
