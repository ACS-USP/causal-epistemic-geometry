"""E3-10 exact semantic instrument.

E3-10 is a development-only procedural benchmark family.  Its answer space is
the semantic digit set ``0..9``; no displayed answer positions are involved.
"""

from .base import (
    DECIMAL_ANSWER_INSTRUCTION,
    DIGITS,
    NUMBER_WORD_ANSWER_INSTRUCTION,
    NUMBER_WORDS,
    LatentItem,
    RenderedView,
)
from .splits import FAMILY_CELLS, generate_balanced_items

__all__ = [
    "DECIMAL_ANSWER_INSTRUCTION",
    "DIGITS",
    "FAMILY_CELLS",
    "LatentItem",
    "NUMBER_WORD_ANSWER_INSTRUCTION",
    "NUMBER_WORDS",
    "RenderedView",
    "generate_balanced_items",
]
