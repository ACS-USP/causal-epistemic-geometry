"""Small, transparent engineering cost model for inference benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModeMeasurement:
    mode: str
    seconds: float
    item_conditions: int
    peak_vram_bytes: int | None = None
    prompt_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.seconds < 0 or self.item_conditions <= 0:
            raise ValueError("seconds must be non-negative and item_conditions positive")


def cost_summary(
    measurement: ModeMeasurement, *, hourly_rate_usd: float = 0.44
) -> dict[str, Any]:
    """Convert a measured mode into comparable throughput and A40 cost units."""

    if hourly_rate_usd < 0:
        raise ValueError("hourly_rate_usd must be non-negative")
    seconds_per_thousand = measurement.seconds / measurement.item_conditions * 1000
    hours_per_ten_thousand = measurement.seconds / measurement.item_conditions * 10000 / 3600
    return {
        "mode": measurement.mode,
        "seconds": measurement.seconds,
        "item_conditions": measurement.item_conditions,
        "item_conditions_per_second": (
            measurement.item_conditions / measurement.seconds
            if measurement.seconds
            else None
        ),
        "seconds_per_1k_item_condition": seconds_per_thousand,
        "a40_hours_per_10k_item_condition": hours_per_ten_thousand,
        "cost_usd_per_10k_item_condition": hours_per_ten_thousand * hourly_rate_usd,
        "hourly_rate_usd": hourly_rate_usd,
        "peak_vram_bytes": measurement.peak_vram_bytes,
        "prompt_tokens": measurement.prompt_tokens,
    }


def compare_modes(
    measurements: list[ModeMeasurement], *, hourly_rate_usd: float = 0.44
) -> list[dict[str, Any]]:
    """Return summaries with speedups relative to the first measurement."""

    if not measurements:
        return []
    reference = measurements[0]
    rows: list[dict[str, Any]] = []
    for measurement in measurements:
        row = cost_summary(measurement, hourly_rate_usd=hourly_rate_usd)
        row["speedup_vs_reference"] = (
            reference.seconds / measurement.seconds if measurement.seconds else None
        )
        rows.append(row)
    return rows
