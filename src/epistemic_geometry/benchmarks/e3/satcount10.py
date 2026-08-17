"""SATCOUNT10: exact small-CNF model counts modulo ten."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

from .base import LatentItem, latent_hash_for, latent_id_for


def _cell(cell: str) -> tuple[int, int]:
    cells = {
        # Three Boolean variables can yield at most eight satisfying
        # assignments, making semantic target 9 impossible.  The original
        # draft D1 was therefore corrected to four variables before any model
        # calibration so every E3-10 digit remains attainable.
        "vars4_clauses4": (4, 4),
        "vars4_clauses6": (4, 6),
        "vars5_clauses8": (5, 8),
        "vars6_clauses10": (6, 10),
    }
    try:
        return cells[cell]
    except KeyError as exc:
        raise ValueError(f"unknown SATCOUNT10 cell: {cell}") from exc


def _valid_clause(clause: list[int], n_variables: int) -> bool:
    literals = {abs(int(literal)) for literal in clause}
    if len(literals) != len(clause):
        return False
    if any(abs(int(literal)) > n_variables or int(literal) == 0 for literal in clause):
        return False
    return not ({int(literal) for literal in clause} & {-int(literal) for literal in clause})


def generate(seed: int, cell: str) -> LatentItem:
    n_variables, n_clauses = _cell(cell)
    for attempt in range(10_000):
        rng = np.random.default_rng(stable_seed("E3-10", "SATCOUNT10", cell, seed, attempt))
        clauses: list[list[int]] = []
        while len(clauses) < n_clauses:
            width = int(rng.integers(2, min(4, n_variables) + 1))
            variables = [
                int(x) for x in rng.choice(np.arange(1, n_variables + 1), size=width, replace=False)
            ]
            signs = [1 if bool(rng.integers(0, 2)) else -1 for _ in variables]
            clause = [variable * sign for variable, sign in zip(variables, signs, strict=True)]
            existing = {tuple(sorted(existing_clause)) for existing_clause in clauses}
            if _valid_clause(clause, n_variables) and tuple(sorted(clause)) not in existing:
                clauses.append(clause)
        if {abs(literal) for clause in clauses for literal in clause} == set(
            range(1, n_variables + 1)
        ):
            break
    else:
        raise RuntimeError(f"could not generate a SATCOUNT10 item using all variables: {seed}")
    spec = {"n_variables": n_variables, "clauses": clauses}
    difficulty = {
        "variables": n_variables,
        "clauses": n_clauses,
        "clause_widths": [len(c) for c in clauses],
    }
    target = oracle(spec)
    latent_hash = latent_hash_for("SATCOUNT10", cell, seed, spec, target, difficulty)
    return LatentItem(
        latent_id=latent_id_for("SATCOUNT10", cell, latent_hash),
        family="SATCOUNT10",
        cell=cell,
        latent_seed=seed,
        spec=spec,
        target=target,
        difficulty=difficulty,
        latent_hash=latent_hash,
    )


def oracle(spec: dict[str, Any]) -> int:
    n_variables = int(spec["n_variables"])
    clauses = [[int(literal) for literal in clause] for clause in spec["clauses"]]
    satisfying = 0
    for assignment in product((False, True), repeat=n_variables):
        if all(
            any(assignment[abs(literal) - 1] == (literal > 0) for literal in clause)
            for clause in clauses
        ):
            satisfying += 1
    return satisfying % 10
