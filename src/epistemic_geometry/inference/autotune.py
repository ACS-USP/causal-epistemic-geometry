"""Small deterministic item/condition autotuning utility.

The utility measures execution settings; it never changes scientific
conditions or chooses vectors, layers, or alphas.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any


def benchmark_batch_grid(
    backend: Any,
    prepared_items: list[Any],
    conditions: list[Any],
    *,
    item_batch_sizes: tuple[int, ...],
    condition_chunk_sizes: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Measure a small grid and record failures instead of hiding OOMs."""

    if not item_batch_sizes or not condition_chunk_sizes:
        raise ValueError("autotune grids must not be empty")
    original_config = backend.config
    rows: list[dict[str, Any]] = []
    try:
        for item_batch_size in item_batch_sizes:
            for condition_chunk_size in condition_chunk_sizes:
                if item_batch_size <= 0 or condition_chunk_size <= 0:
                    raise ValueError("autotune batch sizes must be positive")
                backend.config = replace(
                    original_config,
                    item_batch_size=item_batch_size,
                    condition_chunk_size=condition_chunk_size,
                )
                if hasattr(backend, "reset_execution_stats"):
                    backend.reset_execution_stats()
                started = time.perf_counter()
                try:
                    backend.predict_choice_batch(prepared_items, conditions)
                    elapsed = time.perf_counter() - started
                    stats = (
                        backend.execution_stats()
                        if hasattr(backend, "execution_stats")
                        else {}
                    )
                    rows.append(
                        {
                            "status": "PASS",
                            "item_batch_size": item_batch_size,
                            "condition_chunk_size": condition_chunk_size,
                            "seconds": elapsed,
                            "execution_stats": stats,
                        }
                    )
                except RuntimeError as exc:
                    rows.append(
                        {
                            "status": "FAIL",
                            "item_batch_size": item_batch_size,
                            "condition_chunk_size": condition_chunk_size,
                            "error": str(exc),
                        }
                    )
    finally:
        backend.config = original_config
    return rows


def choose_fastest_safe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the fastest measured PASS; no scientific parameter is involved."""

    successful = [row for row in rows if row.get("status") == "PASS"]
    if not successful:
        raise RuntimeError("No safe autotune configuration completed")
    return min(successful, key=lambda row: float(row["seconds"]))
