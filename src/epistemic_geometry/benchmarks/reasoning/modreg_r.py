"""MODREG-R generation preserving the validated E3 effective dependency cone."""

from __future__ import annotations

from epistemic_geometry.benchmarks.e3 import modreg10
from epistemic_geometry.reproducibility import stable_seed

from .base import GENERATOR_VERSION, ReasoningItem, latent_hash_for

CELLS = ("depth_4", "depth_8", "depth_12", "depth_16")


def generate(seed: int, cell: str) -> ReasoningItem:
    if cell not in CELLS:
        raise ValueError(f"unknown MODREG-R cell: {cell}")
    source_seed = stable_seed(GENERATOR_VERSION, "MODREG-R", cell, seed)
    source = modreg10.generate(source_seed, cell)
    answer = modreg10.oracle(source.spec)
    difficulty = dict(source.difficulty)
    latent_hash = latent_hash_for("MODREG-R", cell, seed, source.spec, answer, difficulty)
    return ReasoningItem(
        latent_id=f"MODREG-R:{cell}:{latent_hash[:16]}",
        family="MODREG-R",
        cell=cell,
        latent_seed=seed,
        spec=source.spec,
        answer=answer,
        difficulty=difficulty,
        latent_hash=latent_hash,
        metadata={"source_family": "MODREG10", "effective_dependency_preserved": True},
    )


def oracle(spec: dict[str, object]) -> int:
    return modreg10.oracle(spec)
