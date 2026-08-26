#!/usr/bin/env python3
"""Deterministically assign non-scientific logical work keys to Spark nodes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_NODES = ("ciaam-spark1", "ciaam-spark2")


def canonical_key(key: Sequence[str]) -> str:
    if len(key) != 3 or any(not isinstance(part, str) or not part for part in key):
        raise ValueError(
            "logical keys must contain three non-empty strings: condition,item,rollout"
        )
    return json.dumps(list(key), ensure_ascii=True, separators=(",", ":"))


def assign_node(key: Sequence[str], nodes: Sequence[str] = DEFAULT_NODES) -> str:
    if not nodes or len(set(nodes)) != len(nodes):
        raise ValueError("nodes must be non-empty and unique")
    digest = hashlib.sha256(canonical_key(key).encode("utf-8")).digest()
    return nodes[int.from_bytes(digest[:8], "big") % len(nodes)]


def partition(
    keys: Iterable[Sequence[str]], nodes: Sequence[str] = DEFAULT_NODES
) -> dict[str, list[tuple[str, str, str]]]:
    assignments: dict[str, list[tuple[str, str, str]]] = {node: [] for node in nodes}
    seen: set[str] = set()
    for raw_key in keys:
        normalized = tuple(raw_key)
        encoded = canonical_key(normalized)
        if encoded in seen:
            raise ValueError(f"duplicate logical key: {encoded}")
        seen.add(encoded)
        assignments[assign_node(normalized, nodes)].append(normalized)  # type: ignore[arg-type]
    return assignments


def recombine(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate unique keys and return a node/order-independent canonical ordering."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        encoded = canonical_key(row["logical_key"])
        if encoded in indexed:
            raise ValueError(f"duplicate result logical key: {encoded}")
        indexed[encoded] = row
    return [indexed[key] for key in sorted(indexed)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON array of [condition,item,rollout] keys")
    parser.add_argument("--nodes", nargs="+", default=list(DEFAULT_NODES))
    args = parser.parse_args()
    keys = json.loads(args.input.read_text(encoding="utf-8"))
    assignments = partition(keys, args.nodes)
    print(json.dumps(assignments, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
