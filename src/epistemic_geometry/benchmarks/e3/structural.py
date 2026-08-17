"""Model-free structural summaries and validity checks for E3-10."""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import fsm10, modreg10
from .base import LatentItem


def modreg_structural_validity(item: LatentItem) -> dict[str, Any]:
    spec = item.spec
    operations = spec["operations"]
    dependency = modreg10.dependency_analysis(spec)
    explicit_no_ops = []
    for index, operation in enumerate(operations):
        kind = operation["op"]
        if kind == "ADD_CONST" and int(operation["c"]) % 10 == 0:
            explicit_no_ops.append(index)
        if kind in {"ADD_REG", "SUB_REG"} and operation["dst"] == operation["src"]:
            explicit_no_ops.append(index)
        if kind == "SWAP" and operation["r1"] == operation["r2"]:
            explicit_no_ops.append(index)
    nominal = len(operations)
    effective = int(dependency["effective_operation_count"])
    threshold_pass = effective / max(1, nominal) >= 0.75 and effective >= nominal - 2
    return {
        "valid": not explicit_no_ops and threshold_pass,
        "explicit_no_op_indices": explicit_no_ops,
        "nominal_depth": nominal,
        "effective_operation_count": effective,
        "effective_depth_fraction": float(dependency["effective_depth_fraction"]),
        "dependency_cone": dependency["dependency_cone"],
    }


def fsm_structural_validity(item: LatentItem) -> dict[str, Any]:
    transitions = item.spec["transitions"]
    identity_symbols = [
        symbol for symbol, mapping in transitions.items() if mapping == list(range(10))
    ]
    duplicate_maps = len({tuple(mapping) for mapping in transitions.values()}) != len(transitions)
    return {
        "valid": not identity_symbols and not duplicate_maps,
        "identity_transition_symbols": identity_symbols,
        "duplicate_transition_maps": duplicate_maps,
    }


def reach_structural_validity(item: LatentItem) -> dict[str, Any]:
    edges = [tuple(int(value) for value in edge) for edge in item.spec["edges"]]
    duplicate_edges = len(edges) != len(set(edges))
    self_loops = [edge for edge in edges if edge[0] == edge[1]]
    nodes = {value for edge in edges for value in edge} | {int(item.spec["source"])}
    return {
        "valid": not duplicate_edges and not self_loops and nodes <= set(range(10)),
        "duplicate_edges": duplicate_edges,
        "self_loops": self_loops,
        "node_count": 10,
        "observed_nodes": sorted(nodes),
    }


def _clause_subsumes(left: list[int], right: list[int]) -> bool:
    return set(left) <= set(right) and len(left) < len(right)


def _sat_raw_count(spec: dict[str, Any]) -> int:
    """Count satisfying assignments without applying modulo ten."""

    from itertools import product

    n_variables = int(spec["n_variables"])
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    return sum(
        all(
            any(assignment[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in clauses
        )
        for assignment in product((False, True), repeat=n_variables)
    )


def sat_structural_validity(item: LatentItem) -> dict[str, Any]:
    spec = item.spec
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    tautologies = [
        index for index, clause in enumerate(clauses) if set(clause) & {-value for value in clause}
    ]
    duplicate_literals = [
        index
        for index, clause in enumerate(clauses)
        if len({abs(value) for value in clause}) != len(clause)
    ]
    duplicate_clauses = len({tuple(sorted(clause)) for clause in clauses}) != len(clauses)
    declared = set(range(1, int(spec["n_variables"]) + 1))
    used = {abs(value) for clause in clauses for value in clause}
    return {
        "valid": not tautologies
        and not duplicate_literals
        and not duplicate_clauses
        and declared == used,
        "tautology_indices": tautologies,
        "duplicate_literal_indices": duplicate_literals,
        "duplicate_clause": duplicate_clauses,
        "unused_variables": sorted(declared - used),
    }


def reachability_details(spec: dict[str, Any]) -> dict[str, Any]:
    adjacency: dict[int, list[int]] = {node: [] for node in range(10)}
    for source, target in spec["edges"]:
        adjacency[int(source)].append(int(target))
    source = int(spec["source"])
    max_hops = int(spec["max_hops"])
    distances = {source: 0}
    frontier = {source}
    expansions = 0
    while frontier and expansions < max_hops:
        next_frontier = {
            target for node in frontier for target in adjacency[node] if target not in distances
        }
        distances.update({target: expansions + 1 for target in next_frontier})
        frontier = next_frontier
        expansions += 1
    shortest_depths = Counter(
        str(depth) for node, depth in distances.items() if node != source and depth <= max_hops
    )
    return {
        "reachable_subgraph_size": len(distances) - 1,
        "frontier_expansions": expansions,
        "shortest_path_depth_counts": dict(sorted(shortest_depths.items())),
    }


def structural_features(item: LatentItem) -> dict[str, float]:
    """Return shallow, non-oracle-trajectory features for shortcut audits."""

    spec = item.spec
    if item.family == "MODREG10":
        operations = spec["operations"]
        histogram = Counter(operation["op"] for operation in operations)
        dependency = modreg10.dependency_analysis(spec)
        return {
            "nominal_depth": float(len(operations)),
            "effective_operation_count": float(dependency["effective_operation_count"]),
            "effective_depth_fraction": float(dependency["effective_depth_fraction"]),
            "query_register": float(int(str(spec["query"])[1])),
            "initial_query_value": float(spec["initial"][int(str(spec["query"])[1])]),
            "swap_count": float(histogram["SWAP"]),
            **{
                f"op_{kind}": float(histogram[kind])
                for kind in ("ADD_CONST", "MUL_UNIT", "ADD_REG", "SUB_REG")
            },
        }
    if item.family == "FSM10":
        sequence = spec["sequence"]
        transitions = spec["transitions"]
        return {
            "sequence_length": float(len(sequence)),
            "start_state": float(spec["start"]),
            "symbol_A_count": float(sequence.count("A")),
            "symbol_B_count": float(sequence.count("B")),
            "symbol_C_count": float(sequence.count("C")),
            "fixed_points_A": float(sum(a == b for a, b in enumerate(transitions["A"]))),
            "fixed_points_B": float(sum(a == b for a, b in enumerate(transitions["B"]))),
            "fixed_points_C": float(sum(a == b for a, b in enumerate(transitions["C"]))),
        }
    if item.family == "REACHCOUNT10":
        details = reachability_details(spec)
        return {
            "edge_count": float(len(spec["edges"])),
            "source_outdegree": float(
                sum(int(edge[0]) == int(spec["source"]) for edge in spec["edges"])
            ),
            "max_hops": float(spec["max_hops"]),
            "graph_density": float(len(spec["edges"]) / 90),
            "reachable_subgraph_size": float(details["reachable_subgraph_size"]),
            "frontier_expansions": float(details["frontier_expansions"]),
            **{
                f"shortest_depth_{depth}": float(
                    details["shortest_path_depth_counts"].get(str(depth), 0)
                )
                for depth in range(1, int(spec["max_hops"]) + 1)
            },
        }
    if item.family == "SATCOUNT10":
        clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
        widths = Counter(len(clause) for clause in clauses)
        all_literals = [literal for clause in clauses for literal in clause]
        nonredundant = sum(
            not any(
                index != other_index and _clause_subsumes(other, clause)
                for other_index, other in enumerate(clauses)
            )
            for index, clause in enumerate(clauses)
        )
        raw_count = _sat_raw_count(spec)
        return {
            "variable_count": float(spec["n_variables"]),
            "clause_count": float(len(clauses)),
            "nonredundant_clause_count": float(nonredundant),
            "duplicate_clause_count": float(
                len(clauses) - len({tuple(sorted(c)) for c in clauses})
            ),
            "tautology_count": float(sum(bool(set(c) & {-value for value in c}) for c in clauses)),
            "unused_variable_count": float(
                len(
                    set(range(1, int(spec["n_variables"]) + 1))
                    - {abs(value) for value in all_literals}
                )
            ),
            "raw_satisfying_count": float(raw_count),
            "satisfying_fraction": float(raw_count / 2 ** int(spec["n_variables"])),
            "positive_literal_fraction": float(
                sum(value > 0 for value in all_literals) / len(all_literals)
            ),
            **{f"clause_width_{width}": float(widths[width]) for width in (2, 3, 4)},
        }
    raise ValueError(f"unknown E3-10 family: {item.family}")


def shallow_heuristic_prediction(item: LatentItem) -> int:
    """A deliberately incomplete family-specific semantic heuristic."""

    if item.family == "MODREG10":
        return int(item.spec["initial"][int(str(item.spec["query"])[1])])
    if item.family == "FSM10":
        return int(item.spec["start"])
    if item.family == "REACHCOUNT10":
        return min(
            9,
            sum(int(edge[0]) == int(item.spec["source"]) for edge in item.spec["edges"]),
        )
    if item.family == "SATCOUNT10":
        return 0
    raise ValueError(f"unknown E3-10 family: {item.family}")


def fsm_sensitivity(item: LatentItem) -> dict[str, float]:
    """Measure whether local sequence edits usually change the final state."""

    if item.family != "FSM10":
        raise ValueError("fsm_sensitivity requires an FSM10 item")
    spec = item.spec
    baseline = fsm10.oracle(spec)
    replacement_changes = 0
    removal_changes = 0
    symbols = fsm10.SYMBOLS
    for index, symbol in enumerate(spec["sequence"]):
        replacement = dict(spec)
        replacement["sequence"] = list(spec["sequence"])
        replacement["sequence"][index] = symbols[(symbols.index(symbol) + 1) % len(symbols)]
        replacement_changes += fsm10.oracle(replacement) != baseline
        removed = dict(spec)
        removed["sequence"] = list(spec["sequence"][:index] + spec["sequence"][index + 1 :])
        removal_changes += fsm10.oracle(removed) != baseline
    count = max(1, len(spec["sequence"]))
    return {
        "replacement_sensitivity_fraction": replacement_changes / count,
        "removal_sensitivity_fraction": removal_changes / count,
    }


def validate_structural_item(item: LatentItem) -> dict[str, Any]:
    validators = {
        "MODREG10": modreg_structural_validity,
        "FSM10": fsm_structural_validity,
        "REACHCOUNT10": reach_structural_validity,
        "SATCOUNT10": sat_structural_validity,
    }
    return validators[item.family](item)
