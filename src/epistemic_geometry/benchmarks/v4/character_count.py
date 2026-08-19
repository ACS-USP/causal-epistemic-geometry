"""Procedural, exact character-count tasks for the V4 Bench E screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from epistemic_geometry.reproducibility import canonical_json, stable_digest, stable_seed

STRATA: dict[str, tuple[int, int]] = {
    "WORDLIKE_SHORT": (8, 15),
    "FRESH_PSEUDOWORD_MEDIUM": (16, 30),
    "FRESH_PSEUDOWORD_LONG": (31, 60),
}
_VOWELS = "aeiou"
_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class CharacterCountItem:
    item_id: str
    stratum: str
    index: int
    seed: int
    text: str
    target_character: str
    answer: int
    prompt: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["prompt_hash"] = stable_digest("V4-CHARCOUNT-PROMPT", self.prompt)
        record["item_hash"] = stable_digest("V4-CHARCOUNT-ITEM", canonical_json(record))
        return record


def _filler(rng: np.random.Generator, length: int, target: str) -> list[str]:
    """Create deterministic lowercase filler while excluding the target."""

    allowed_vowels = "".join(char for char in _VOWELS if char != target) or "aeiou"
    allowed_consonants = "".join(char for char in _CONSONANTS if char != target) or "bcdf"
    chars: list[str] = []
    for position in range(length):
        alphabet = allowed_consonants if position % 2 == 0 else allowed_vowels
        chars.append(str(rng.choice(list(alphabet))))
    return chars


def _make_item(stratum: str, index: int, *, seed: int) -> CharacterCountItem:
    minimum, maximum = STRATA[stratum]
    rng = np.random.default_rng(seed)
    length = int(rng.integers(minimum, maximum + 1))
    target = _ALPHABET[(seed + index * 7) % len(_ALPHABET)]
    # The deterministic schedule deliberately creates counts 2..6 rather than
    # selecting difficulty from model outcomes.
    count = 2 + (index % 5)
    count = min(count, length - 1)
    chars = _filler(rng, length - count, target)
    positions = rng.choice(length, size=count, replace=False)
    output = chars[:]
    for position in sorted(int(value) for value in positions):
        output.insert(position, target)
    text = "".join(output)
    if len(text) != length or text.count(target) != count:
        raise AssertionError("character-count generator invariant failed")
    prompt = (
        f"How many times does the letter '{target}' appear in '{text}'?\n"
        "Think as needed, then answer exactly:\nFINAL: <integer>"
    )
    return CharacterCountItem(
        item_id=f"charcount_{stratum.lower()}_{index:02d}",
        stratum=stratum,
        index=index,
        seed=seed,
        text=text,
        target_character=target,
        answer=count,
        prompt=prompt,
    )


def generate_character_count_manifest(
    *, seed: int = 20260819, per_stratum: int = 10
) -> dict[str, Any]:
    """Generate the complete frozen 30-item development manifest."""

    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive")
    items: list[dict[str, Any]] = []
    for stratum in STRATA:
        for index in range(per_stratum):
            item_seed = stable_seed("V4-CHARCOUNT", seed, stratum, index)
            item = _make_item(stratum, index, seed=item_seed)
            items.append(item.to_record())
    digest = stable_digest("V4-CHARCOUNT-MANIFEST", canonical_json(items))
    return {
        "suite": "Q1_V4_MICROBENCH",
        "instrument": "CHARCOUNT",
        "generator_version": "v4-charcount-1",
        "seed": seed,
        "per_stratum": per_stratum,
        "strata": STRATA,
        "items": items,
        "manifest_hash": digest,
    }
