"""Family-dispatched exact E3-10 oracles."""

from __future__ import annotations

from typing import Any

from . import fsm10, modreg10, reachcount10, satcount10
from .base import LatentItem


def compute_oracle(family: str, spec: dict[str, Any]) -> int:
    """Compute the semantic digit without rendering or model involvement."""

    functions = {
        "MODREG10": modreg10.oracle,
        "FSM10": fsm10.oracle,
        "REACHCOUNT10": reachcount10.oracle,
        "SATCOUNT10": satcount10.oracle,
    }
    try:
        target = int(functions[family](spec))
    except KeyError as exc:
        raise ValueError(f"unknown E3-10 family: {family}") from exc
    if target not in range(10):
        raise ValueError(f"oracle for {family} returned non-digit {target}")
    return target


def oracle_for(item: LatentItem) -> int:
    """Recompute and verify the exact target for a latent item."""

    target = compute_oracle(item.family, item.spec)
    if target != item.target:
        raise ValueError(f"stored target mismatch for {item.latent_id}: {item.target} != {target}")
    return target
