"""Structural timing estimands for a prospective Qwen3 finalization study.

This module treats the documented Qwen3 ``</think>`` token as an observable
transition event.  It uses token IDs and the predeclared horizon only; it never
reads rendered text, answer keys, parsed answers, or correctness.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np

QWEN3_THINK_CLOSE_TOKEN_ID = 151668
CONDITIONS = ("BASELINE", "D75")
ROLLOUTS = (0, 1)


@dataclass(frozen=True, slots=True)
class TimingObservation:
    """One raw structural trajectory, without semantic outcome fields."""

    family: str
    cell: str
    latent: str
    condition: str
    rollout: int
    cap: int
    generated_token_ids: tuple[int, ...]

    def key(self) -> tuple[str, str, str, str, int]:
        return (self.family, self.cell, self.latent, self.condition, self.rollout)


def _observation(value: TimingObservation | Mapping[str, Any]) -> TimingObservation:
    if isinstance(value, TimingObservation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("timing observations must be mappings")
    required = (
        "family",
        "cell",
        "latent",
        "condition",
        "rollout",
        "cap",
        "generated_token_ids",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"timing observation is missing fields: {missing}")
    token_ids = value["generated_token_ids"]
    if not isinstance(token_ids, Sequence) or isinstance(token_ids, (str, bytes)):
        raise ValueError("generated_token_ids must be a sequence")
    return TimingObservation(
        family=value["family"],
        cell=value["cell"],
        latent=value["latent"],
        condition=value["condition"],
        rollout=value["rollout"],
        cap=value["cap"],
        generated_token_ids=tuple(token_ids),
    )


def validate_timing_schedule(
    observations: Iterable[TimingObservation | Mapping[str, Any]],
    *,
    expected_latents: Mapping[tuple[str, str], Iterable[str]] | None = None,
    cap: int | None = None,
) -> tuple[TimingObservation, ...]:
    """Validate a complete paired two-rollout timing schedule."""

    rows = tuple(_observation(row) for row in observations)
    if not rows:
        raise ValueError("at least one timing observation is required")
    keys = [row.key() for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate timing observation key")
    actual_by_stratum: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        identifiers = (row.family, row.cell, row.latent)
        if not all(isinstance(value, str) and value for value in identifiers):
            raise ValueError("family, cell, and latent must be non-empty strings")
        if row.condition not in CONDITIONS:
            raise ValueError(f"unsupported condition: {row.condition!r}")
        if type(row.rollout) is not int or row.rollout not in ROLLOUTS:
            raise ValueError(f"unsupported rollout: {row.rollout!r}")
        if type(row.cap) is not int or row.cap <= 0:
            raise ValueError("cap must be a positive integer")
        if cap is not None and row.cap != cap:
            raise ValueError("timing observation cap does not match locked horizon")
        if len(row.generated_token_ids) > row.cap:
            raise ValueError("generated token sequence exceeds cap")
        if any(type(token) is not int or token < 0 for token in row.generated_token_ids):
            raise ValueError("generated_token_ids must contain non-negative integers")
        actual_by_stratum[(row.family, row.cell)].add(row.latent)
    if expected_latents is not None:
        expected = {key: set(values) for key, values in expected_latents.items()}
        if {key: set(value) for key, value in actual_by_stratum.items()} != expected:
            raise ValueError("observed latent population differs from locked manifest")
    expected_keys = {
        (family, cell, latent, condition, rollout)
        for (family, cell), latents in actual_by_stratum.items()
        for latent in latents
        for condition in CONDITIONS
        for rollout in ROLLOUTS
    }
    if set(keys) != expected_keys:
        raise ValueError("timing schedule is incomplete")
    return rows


def think_close_time(
    row: TimingObservation, *, close_token_id: int = QWEN3_THINK_CLOSE_TOKEN_ID
) -> int:
    """Return one-based close position; absent close is administratively censored at cap."""

    if type(close_token_id) is not int or close_token_id < 0:
        raise ValueError("close_token_id must be a non-negative integer")
    try:
        return row.generated_token_ids.index(close_token_id) + 1
    except ValueError:
        return row.cap


def normalized_paired_shortening(
    observations: Iterable[TimingObservation | Mapping[str, Any]],
    *,
    cap: int,
    close_token_id: int = QWEN3_THINK_CLOSE_TOKEN_ID,
    expected_latents: Mapping[tuple[str, str], Iterable[str]] | None = None,
) -> dict[tuple[str, str, str], float]:
    """Return baseline-minus-D75 restricted-time contrasts per latent.

    Positive values mean D75 moved the structural close event earlier.  An
    absent close remains a cap-censored observation in every condition.
    """

    rows = validate_timing_schedule(observations, expected_latents=expected_latents, cap=cap)
    grouped: dict[tuple[str, str, str, str], list[TimingObservation]] = defaultdict(list)
    for row in rows:
        grouped[(row.family, row.cell, row.latent, row.condition)].append(row)
    contrasts: dict[tuple[str, str, str], float] = {}
    for (family, cell, latent, condition), values in grouped.items():
        if len(values) != 2 or {value.rollout for value in values} != set(ROLLOUTS):
            raise ValueError("each latent-condition must contain both rollouts")
        grouped[(family, cell, latent, condition)] = sorted(values, key=lambda value: value.rollout)
    units = {(family, cell, latent) for family, cell, latent, _ in grouped}
    for family, cell, latent in units:
        baseline = grouped[(family, cell, latent, "BASELINE")]
        d75 = grouped[(family, cell, latent, "D75")]
        contrasts[(family, cell, latent)] = float(
            np.mean(
                [
                    (
                        think_close_time(base, close_token_id=close_token_id)
                        - think_close_time(treated, close_token_id=close_token_id)
                    )
                    / cap
                    for base, treated in zip(baseline, d75, strict=True)
                ]
            )
        )
    return contrasts


def equal_family_cell_mean(contrasts: Mapping[tuple[str, str, str], float]) -> float:
    """Average latent contrasts equally within cells then cells within families."""

    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (family, cell, _latent), value in contrasts.items():
        if not np.isfinite(value) or value < -1.0 or value > 1.0:
            raise ValueError("timing contrast must be finite in [-1, 1]")
        cells[(family, cell)].append(float(value))
    if not cells:
        raise ValueError("at least one contrast is required")
    families: dict[str, list[float]] = defaultdict(list)
    for (family, _cell), values in cells.items():
        families[family].append(float(np.mean(values)))
    return float(np.mean([np.mean(values) for values in families.values()]))


def directional_evalues(contrasts: Iterable[float], *, delta: float) -> dict[str, float]:
    """Return fixed-sample e-values for a relevant earlier/later close event."""

    values = np.asarray(tuple(contrasts), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("contrasts must be a non-empty finite vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("contrasts must lie in [-1, 1]")
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not 0 < delta <= 1:
        raise ValueError("delta must lie in (0, 1]")
    return {
        "d75_earlier_by_delta": float(prod(1.0 + (values - delta) / 2.0)),
        "d75_later_by_delta": float(prod(1.0 + (-values - delta) / 2.0)),
        "exclude_d75_earlier_by_delta": float(prod(1.0 + (delta - values) / 2.0)),
        "exclude_d75_later_by_delta": float(prod(1.0 + (delta + values) / 2.0)),
    }
