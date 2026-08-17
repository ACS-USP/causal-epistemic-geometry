"""Family registry for the Q1 V3 procedural reasoning suite."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import fsm_r, modreg_r, satcount_r
from .base import ReasoningItem

FAMILY_CELLS: dict[str, tuple[str, ...]] = {
    "MODREG-R": modreg_r.CELLS,
    "FSM-R": fsm_r.CELLS,
    "SATCOUNT-R": satcount_r.CELLS,
}
_GENERATORS: dict[str, Callable[[int, str], ReasoningItem]] = {
    "MODREG-R": modreg_r.generate,
    "FSM-R": fsm_r.generate,
    "SATCOUNT-R": satcount_r.generate,
}


def generate_item(family: str, cell: str, seed: int) -> ReasoningItem:
    try:
        return _GENERATORS[family](seed, cell)
    except KeyError as exc:
        raise ValueError(f"unknown Q1 V3 reasoning family: {family}") from exc


def oracle_for(item: ReasoningItem | str, spec: dict[str, Any] | None = None) -> int:
    if isinstance(item, ReasoningItem):
        family, spec = item.family, item.spec
    else:
        family = item
        if spec is None:
            raise ValueError("spec is required when oracle_for receives a family name")
    if family == "MODREG-R":
        return modreg_r.oracle(spec)
    if family == "FSM-R":
        return fsm_r.oracle(spec)
    if family == "SATCOUNT-R":
        return satcount_r.oracle(spec)
    raise ValueError(f"unknown Q1 V3 reasoning family: {family}")
