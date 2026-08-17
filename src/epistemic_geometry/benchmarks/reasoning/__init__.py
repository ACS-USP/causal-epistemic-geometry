"""Q1 V3 reasoning-agent procedural suite.

This package is deliberately separate from the closed E3 direct-readout
instrument.  Its outputs are exact final answers parsed from generated
reasoning trajectories rather than candidate logits.
"""

from .base import (
    FINAL_ANSWER_INSTRUCTION,
    GENERATOR_VERSION,
    SUITE_VERSION,
    ReasoningItem,
    ReasoningView,
)
from .families import FAMILY_CELLS, generate_item, oracle_for

__all__ = [
    "FAMILY_CELLS",
    "FINAL_ANSWER_INSTRUCTION",
    "GENERATOR_VERSION",
    "ReasoningItem",
    "ReasoningView",
    "SUITE_VERSION",
    "generate_item",
    "oracle_for",
]
