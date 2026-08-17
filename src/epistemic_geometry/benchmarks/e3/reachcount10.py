"""REACHCOUNT10: exact bounded graph reachability counts."""

from __future__ import annotations

from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

from .base import LatentItem, latent_hash_for, latent_id_for

NODES = tuple(range(10))


def _cell(cell: str) -> tuple[int, float]:
    cells = {"H1_p010": (1, 0.10), "H2_p010": (2, 0.10), "H2_p018": (2, 0.18), "H3_p015": (3, 0.15)}
    try:
        return cells[cell]
    except KeyError as exc:
        raise ValueError(f"unknown REACHCOUNT10 cell: {cell}") from exc


def generate(seed: int, cell: str) -> LatentItem:
    hops, probability = _cell(cell)
    rng = np.random.default_rng(stable_seed("E3-10", "REACHCOUNT10", cell, seed))
    source = int(rng.integers(0, 10))
    # The queried source row uses a uniform procedural out-degree.  This keeps
    # all ten semantic counts attainable without selecting on model behavior;
    # the other 81 possible edges retain the cell's stated density.  The
    # resulting whole-graph density remains approximately the requested p.
    source_degree = int(rng.integers(0, 10))
    source_targets = [
        int(x)
        for x in rng.choice(
            [node for node in NODES if node != source], size=source_degree, replace=False
        )
    ]
    edges = [[source, target] for target in source_targets]
    edges.extend(
        [other, target]
        for other in NODES
        if other != source
        for target in NODES
        if other != target and bool(rng.random() < probability)
    )
    spec = {"edges": edges, "source": source, "max_hops": hops}
    difficulty = {"max_hops": hops, "edge_probability": probability, "node_count": 10}
    target = oracle(spec)
    latent_hash = latent_hash_for("REACHCOUNT10", cell, seed, spec, target, difficulty)
    return LatentItem(
        latent_id=latent_id_for("REACHCOUNT10", cell, latent_hash),
        family="REACHCOUNT10",
        cell=cell,
        latent_seed=seed,
        spec=spec,
        target=target,
        difficulty=difficulty,
        latent_hash=latent_hash,
    )


def oracle(spec: dict[str, Any]) -> int:
    adjacency: dict[int, list[int]] = {node: [] for node in NODES}
    for source, target in spec["edges"]:
        source, target = int(source), int(target)
        if source == target:
            raise ValueError("REACHCOUNT10 forbids self-loops")
        adjacency[source].append(target)
    start = int(spec["source"])
    max_hops = int(spec["max_hops"])
    seen = {start}
    frontier = {start}
    for _ in range(max_hops):
        frontier = {target for node in frontier for target in adjacency[node]} - seen
        seen.update(frontier)
        if not frontier:
            break
    return len(seen) - 1
