"""SATCOUNT-R with exact raw satisfying-assignment counts, not modulo ten."""

from __future__ import annotations

from itertools import product
from typing import Any

from epistemic_geometry.benchmarks.e3 import satcount10
from epistemic_geometry.reproducibility import stable_seed

from .base import GENERATOR_VERSION, ReasoningItem, latent_hash_for

CELLS = ("vars4_clauses4", "vars4_clauses6", "vars5_clauses8", "vars6_clauses10")


def exact_oracle(spec: dict[str, Any]) -> int:
    n_variables = int(spec["n_variables"])
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    count = 0
    for assignment in product((False, True), repeat=n_variables):
        if all(
            any(assignment[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in clauses
        ):
            count += 1
    return count


def generate(seed: int, cell: str) -> ReasoningItem:
    if cell not in CELLS:
        raise ValueError(f"unknown SATCOUNT-R cell: {cell}")
    source_seed = stable_seed(GENERATOR_VERSION, "SATCOUNT-R", cell, seed)
    source = satcount10.generate(source_seed, cell)
    answer = exact_oracle(source.spec)
    difficulty = dict(source.difficulty)
    difficulty["answer_space_max"] = 2 ** int(source.spec["n_variables"])
    latent_hash = latent_hash_for("SATCOUNT-R", cell, seed, source.spec, answer, difficulty)
    return ReasoningItem(
        latent_id=f"SATCOUNT-R:{cell}:{latent_hash[:16]}",
        family="SATCOUNT-R",
        cell=cell,
        latent_seed=seed,
        spec=source.spec,
        answer=answer,
        difficulty=difficulty,
        latent_hash=latent_hash,
        metadata={"source_family": "SATCOUNT10", "modulo_removed": True},
    )


def oracle(spec: dict[str, Any]) -> int:
    return exact_oracle(spec)
