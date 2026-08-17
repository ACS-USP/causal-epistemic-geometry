"""FSM-R generation preserving bijective finite-state composition."""

from __future__ import annotations

from epistemic_geometry.benchmarks.e3 import fsm10
from epistemic_geometry.reproducibility import stable_seed

from .base import GENERATOR_VERSION, ReasoningItem, latent_hash_for

CELLS = ("length_4", "length_8", "length_12", "length_16")


def generate(seed: int, cell: str) -> ReasoningItem:
    if cell not in CELLS:
        raise ValueError(f"unknown FSM-R cell: {cell}")
    source_seed = stable_seed(GENERATOR_VERSION, "FSM-R", cell, seed)
    source = fsm10.generate(source_seed, cell)
    answer = fsm10.oracle(source.spec)
    difficulty = dict(source.difficulty)
    latent_hash = latent_hash_for("FSM-R", cell, seed, source.spec, answer, difficulty)
    return ReasoningItem(
        latent_id=f"FSM-R:{cell}:{latent_hash[:16]}",
        family="FSM-R",
        cell=cell,
        latent_seed=seed,
        spec=source.spec,
        answer=answer,
        difficulty=difficulty,
        latent_hash=latent_hash,
        metadata={"source_family": "FSM10", "bijection_required": True},
    )


def oracle(spec: dict[str, object]) -> int:
    return fsm10.oracle(spec)
