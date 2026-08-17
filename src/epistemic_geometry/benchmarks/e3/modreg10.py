"""MODREG10: exact modular-register execution problems."""

from __future__ import annotations

from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import stable_seed

from .base import LatentItem, latent_hash_for, latent_id_for

REGISTERS = ("R0", "R1", "R2", "R3")
UNITS = (3, 7, 9)


def dependency_analysis(spec: dict[str, Any]) -> dict[str, Any]:
    """Compute the exact backward dataflow cone for the queried register.

    The live set contains register values needed immediately before the
    operation currently being traversed.  Register-to-register operations
    expand it; swaps exchange live identities exactly.  Indices are zero-based
    and refer to the forward operation sequence in ``spec``.
    """

    live = {str(spec["query"])}
    cone: list[int] = []
    live_before: dict[int, tuple[str, ...]] = {}
    for index in range(len(spec["operations"]) - 1, -1, -1):
        operation = spec["operations"][index]
        kind = operation["op"]
        live_before[index] = tuple(sorted(live))
        if kind in {"ADD_CONST", "MUL_UNIT"}:
            register = str(operation["r"])
            if register in live:
                cone.append(index)
        elif kind in {"ADD_REG", "SUB_REG"}:
            destination = str(operation["dst"])
            source = str(operation["src"])
            if destination in live:
                cone.append(index)
                live.remove(destination)
                live.update({destination, source})
        elif kind == "SWAP":
            first, second = str(operation["r1"]), str(operation["r2"])
            if first in live or second in live:
                cone.append(index)
                first_live, second_live = first in live, second in live
                live.discard(first)
                live.discard(second)
                if first_live:
                    live.add(second)
                if second_live:
                    live.add(first)
        else:
            raise ValueError(f"unknown MODREG10 operation: {kind}")
    cone.sort()
    return {
        "dependency_cone": cone,
        "effective_operation_count": len(cone),
        "effective_depth_fraction": len(cone) / max(1, len(spec["operations"])),
        "root_query_register": str(spec["query"]),
        "live_registers_before_operation": {
            str(index): list(registers) for index, registers in sorted(live_before.items())
        },
    }


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
    query = str(rng.choice(REGISTERS))
    live = {query}
    reverse_operations: list[dict[str, Any]] = []
    for step in range(depth):
        expand = len(live) < len(REGISTERS) and (step < 2 or bool(rng.integers(0, 2)))
        if expand:
            destination = str(rng.choice(sorted(live)))
            source = str(rng.choice([register for register in REGISTERS if register not in live]))
            kind = str(rng.choice(["ADD_REG", "SUB_REG"]))
            reverse_operations.append({"op": kind, "dst": destination, "src": source})
            live.add(source)
        elif len(live) >= 2 and bool(rng.integers(0, 5) == 0):
            first, second = rng.choice(sorted(live), size=2, replace=False).tolist()
            reverse_operations.append({"op": "SWAP", "r1": str(first), "r2": str(second)})
            live.remove(str(first))
            live.remove(str(second))
            live.update({str(first), str(second)})
        else:
            register = str(rng.choice(sorted(live)))
            if bool(rng.integers(0, 2)):
                reverse_operations.append(
                    {"op": "ADD_CONST", "r": register, "c": int(rng.integers(1, 10))}
                )
            else:
                reverse_operations.append(
                    {"op": "MUL_UNIT", "r": register, "u": int(rng.choice(UNITS))}
                )
    operations = list(reversed(reverse_operations))
    spec = {"initial": initial, "operations": operations, "query": query}
    dependency = dependency_analysis(spec)
    difficulty = {
        "depth": depth,
        "nominal_depth": depth,
        "operation_count": depth,
        "effective_operation_count": dependency["effective_operation_count"],
        "effective_depth_fraction": dependency["effective_depth_fraction"],
        "dependency_cone": dependency["dependency_cone"],
    }
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
