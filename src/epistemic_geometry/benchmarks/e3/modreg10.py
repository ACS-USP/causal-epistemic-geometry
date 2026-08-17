"""MODREG10: exact modular-register execution problems."""

from __future__ import annotations

from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

from .base import LatentItem, latent_hash_for, latent_id_for

REGISTERS = ("R0", "R1", "R2", "R3")
UNITS = (3, 7, 9)


def _depth(cell: str) -> int:
    prefix, value = cell.split("_", 1)
    if prefix != "depth" or int(value) not in (4, 8, 12, 16):
        raise ValueError(f"unknown MODREG10 cell: {cell}")
    return int(value)


def generate(seed: int, cell: str) -> LatentItem:
    """Generate one valid MODREG10 latent world from a stable seed."""

    depth = _depth(cell)
    rng = np.random.default_rng(stable_seed("E3-10", "MODREG10", cell, seed))
    initial = [int(x) for x in rng.integers(0, 10, size=4)]
    operations: list[dict[str, Any]] = []
    for _ in range(depth):
        kind = str(rng.choice(["ADD_CONST", "MUL_UNIT", "ADD_REG", "SUB_REG", "SWAP"]))
        if kind == "ADD_CONST":
            operations.append(
                {"op": kind, "r": str(rng.choice(REGISTERS)), "c": int(rng.integers(1, 10))}
            )
        elif kind == "MUL_UNIT":
            operations.append(
                {"op": kind, "r": str(rng.choice(REGISTERS)), "u": int(rng.choice(UNITS))}
            )
        elif kind in {"ADD_REG", "SUB_REG"}:
            dst, src = rng.choice(REGISTERS, size=2, replace=False).tolist()
            operations.append({"op": kind, "dst": str(dst), "src": str(src)})
        else:
            r1, r2 = rng.choice(REGISTERS, size=2, replace=False).tolist()
            operations.append({"op": kind, "r1": str(r1), "r2": str(r2)})
    query = str(rng.choice(REGISTERS))
    spec = {"initial": initial, "operations": operations, "query": query}
    difficulty = {"depth": depth, "operation_count": depth}
    target = oracle(spec)
    latent_hash = latent_hash_for("MODREG10", cell, seed, spec, target, difficulty)
    return LatentItem(
        latent_id=latent_id_for("MODREG10", cell, latent_hash),
        family="MODREG10",
        cell=cell,
        latent_seed=seed,
        spec=spec,
        target=target,
        difficulty=difficulty,
        latent_hash=latent_hash,
    )


def oracle(spec: dict[str, Any]) -> int:
    """Execute a MODREG10 program exactly."""

    registers = {
        name: int(value) % 10 for name, value in zip(REGISTERS, spec["initial"], strict=True)
    }
    for operation in spec["operations"]:
        kind = operation["op"]
        if kind == "ADD_CONST":
            registers[operation["r"]] = (registers[operation["r"]] + int(operation["c"])) % 10
        elif kind == "MUL_UNIT":
            registers[operation["r"]] = (int(operation["u"]) * registers[operation["r"]]) % 10
        elif kind == "ADD_REG":
            registers[operation["dst"]] = (
                registers[operation["dst"]] + registers[operation["src"]]
            ) % 10
        elif kind == "SUB_REG":
            registers[operation["dst"]] = (
                registers[operation["dst"]] - registers[operation["src"]]
            ) % 10
        elif kind == "SWAP":
            registers[operation["r1"]], registers[operation["r2"]] = (
                registers[operation["r2"]],
                registers[operation["r1"]],
            )
        else:
            raise ValueError(f"unknown MODREG10 operation: {kind}")
    return registers[str(spec["query"])]
