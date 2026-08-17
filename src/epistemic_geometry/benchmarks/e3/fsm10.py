"""FSM10: exact finite-state transition composition problems."""

from __future__ import annotations

from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

from .base import LatentItem, latent_hash_for, latent_id_for

STATES = tuple(range(10))
SYMBOLS = ("A", "B", "C")


def _length(cell: str) -> int:
    prefix, value = cell.split("_", 1)
    if prefix != "length" or int(value) not in (4, 8, 12, 16):
        raise ValueError(f"unknown FSM10 cell: {cell}")
    return int(value)


def generate(seed: int, cell: str) -> LatentItem:
    length = _length(cell)
    for attempt in range(10_000):
        rng = np.random.default_rng(stable_seed("E3-10", "FSM10", cell, seed, attempt))
        transitions = {symbol: [int(x) for x in rng.permutation(STATES)] for symbol in SYMBOLS}
        if any(mapping == list(STATES) for mapping in transitions.values()):
            continue
        if len({tuple(mapping) for mapping in transitions.values()}) != len(SYMBOLS):
            continue
        start = int(rng.integers(0, 10))
        sequence = [str(x) for x in rng.choice(SYMBOLS, size=length)]
        break
    else:
        raise RuntimeError(f"could not generate a non-degenerate FSM10 item for seed {seed}")
    spec = {"transitions": transitions, "start": start, "sequence": sequence}
    difficulty = {"sequence_length": length, "state_count": 10, "symbol_count": 3}
    target = oracle(spec)
    latent_hash = latent_hash_for("FSM10", cell, seed, spec, target, difficulty)
    return LatentItem(
        latent_id=latent_id_for("FSM10", cell, latent_hash),
        family="FSM10",
        cell=cell,
        latent_seed=seed,
        spec=spec,
        target=target,
        difficulty=difficulty,
        latent_hash=latent_hash,
    )


def oracle(spec: dict[str, Any]) -> int:
    state = int(spec["start"])
    transitions = spec["transitions"]
    for symbol in spec["sequence"]:
        state = int(transitions[str(symbol)][state])
    return state
