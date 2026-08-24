#!/usr/bin/env python3
"""Produce remote-safe aggregates from a local confirmatory journal.

The output never includes item IDs, seeds, prompts, references, parsed values,
or raw model text. An optional local taxonomy CSV may add category and human-
recoverability counts without exporting row-level labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def token_summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": max(values),
    }


def load_taxonomy(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    categories = Counter(row["adjudicated_category"] for row in rows)
    recoverability = Counter(
        row.get("candidate_reference_match", "")
        for row in rows
        if row.get("candidate_reference_match")
    )
    return {
        "rows": len(rows),
        "category_counts": dict(sorted(categories.items())),
        "candidate_reference_match_counts": dict(sorted(recoverability.items())),
    }


def summarize(journal: Path, taxonomy: Path | None = None) -> dict[str, Any]:
    rows = [json.loads(line) for line in journal.open()]
    condition_counts = Counter(row["condition"] for row in rows)
    invalid = [row for row in rows if not row["commitment_valid"] or not row["semantic_evaluable"]]
    meaningful = [row for row in rows if row["condition"] == "MEANINGFUL_FIXED"]
    meaningful_invalid = [
        row for row in meaningful if not row["commitment_valid"] or not row["semantic_evaluable"]
    ]
    meaningful_valid = [row for row in meaningful if row not in meaningful_invalid]

    invalid_rollouts_by_item: dict[str, int] = defaultdict(int)
    for row in meaningful_invalid:
        invalid_rollouts_by_item[row["item_id"]] += 1

    by_key = {(row["item_id"], row["condition"], row["rollout_index"]): row for row in rows}
    item_ids = {row["item_id"] for row in rows}
    damage = 0
    invalid_damage = 0
    rescue_from_invalid = 0
    for item_id in item_ids:
        baseline = [by_key[(item_id, "BASELINE", rollout)] for rollout in (0, 1)]
        treatment = [by_key[(item_id, "MEANINGFUL_FIXED", rollout)] for rollout in (0, 1)]
        for base in baseline:
            for current in treatment:
                current_invalid = (
                    not current["commitment_valid"] or not current["semantic_evaluable"]
                )
                if base["correct"] and not current["correct"]:
                    damage += 1
                    invalid_damage += int(current_invalid)
                if not base["correct"] and current["correct"] and current_invalid:
                    rescue_from_invalid += 1

    return {
        "scope": "POST_HOC_DESCRIPTIVE_ONLY",
        "contains_raw_text": False,
        "journal": {
            "bytes": journal.stat().st_size,
            "sha256": digest(journal),
            "rows": len(rows),
        },
        "rows_per_condition": dict(sorted(condition_counts.items())),
        "invalid_rows_per_condition": {
            condition: sum(row["condition"] == condition for row in invalid)
            for condition in sorted(condition_counts)
        },
        "meaningful_invalidity": {
            "rows": len(meaningful_invalid),
            "denominator": len(meaningful),
            "unique_affected_items": len(invalid_rollouts_by_item),
            "one_of_two_invalid_items": sum(
                value == 1 for value in invalid_rollouts_by_item.values()
            ),
            "two_of_two_invalid_items": sum(
                value == 2 for value in invalid_rollouts_by_item.values()
            ),
        },
        "token_counts": {
            "meaningful_valid": token_summary(
                [row["generated_token_count"] for row in meaningful_valid]
            ),
            "meaningful_invalid": token_summary(
                [row["generated_token_count"] for row in meaningful_invalid]
            ),
        },
        "pair_context": {
            "damage_pairs": damage,
            "damage_pairs_from_invalid_meaningful": invalid_damage,
            "rescues_from_invalid_meaningful": rescue_from_invalid,
        },
        "optional_manual_taxonomy": load_taxonomy(taxonomy),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-bytes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_sha256 and digest(args.journal) != args.expected_sha256:
        raise SystemExit("journal SHA-256 mismatch")
    if args.expected_bytes is not None and args.journal.stat().st_size != args.expected_bytes:
        raise SystemExit("journal byte-size mismatch")
    payload = summarize(args.journal, args.taxonomy)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
