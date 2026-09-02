"""Prospectively frozen token-only terminal policies.

These policies inspect generated token IDs only.  They never decode text,
consult a reference answer, or use scientific condition identity.
"""

from __future__ import annotations

from collections import Counter

EXTREME_REPETITION_NAME = "EXTREME_MECHANICAL_REPETITION_V1"
MIN_REPETITION_TOKENS = 256
TAIL_WINDOW_TOKENS = 1024
MAX_PERIOD_TOKENS = 64
PERIODIC_MATCH_THRESHOLD = 0.90
DOMINANT_TOKEN_THRESHOLD = 0.50
HISTORICAL_DEFINITION_SOURCE_SHA256 = (
    "e340f9d622c4f874a868c0d9a4e203005b8bcfe778ab688f2202f35a39bda24e"
)


def extreme_mechanical_repetition_v1(token_ids: list[int]) -> bool:
    """Return the unchanged historical V1 structural-repetition decision."""

    if len(token_ids) < MIN_REPETITION_TOKENS:
        return False
    tail = token_ids[-TAIL_WINDOW_TOKENS:]
    dominant_share = max(Counter(tail).values()) / len(tail)
    if dominant_share >= DOMINANT_TOKEN_THRESHOLD:
        return True
    for period in range(1, min(MAX_PERIOD_TOKENS, len(tail) - 1) + 1):
        comparisons = len(tail) - period
        matches = sum(
            tail[index] == tail[index - period] for index in range(period, len(tail))
        )
        if comparisons and matches / comparisons >= PERIODIC_MATCH_THRESHOLD:
            return True
    return False


def extreme_repetition_policy_identity() -> dict[str, str | int | float]:
    """Machine-readable identity without scientific data."""

    return {
        "name": EXTREME_REPETITION_NAME,
        "minimum_generated_tokens": MIN_REPETITION_TOKENS,
        "tail_window_tokens": TAIL_WINDOW_TOKENS,
        "maximum_period_tokens": MAX_PERIOD_TOKENS,
        "periodic_match_threshold": PERIODIC_MATCH_THRESHOLD,
        "dominant_token_share_threshold": DOMINANT_TOKEN_THRESHOLD,
        "historical_definition_source_sha256": HISTORICAL_DEFINITION_SOURCE_SHA256,
    }
