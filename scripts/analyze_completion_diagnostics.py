#!/usr/bin/env python3
"""Select a prospective cap from a completed completion diagnostic.

The rule is mechanical: each item contributes the smallest ladder cap at which
that item reached a non-truncated outcome, and the prospective candidate cap is
the largest of those per-item requirements. Accuracy is never consulted.
If any item remains truncated at the final ladder cap, the candidate is
operationally unresolved and must not advance automatically.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

CAP_LADDER = (8192, 16384, 32768)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    with (args.run / "journal.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    by_cap: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cap = int(row["metadata"]["diagnostic_cap"])
        by_cap[cap].append(row)
    item_ids = manifest["identity"]["item_ids"]
    required_caps: dict[str, int | None] = {}
    for item_id in item_ids:
        item_attempts = [
            row
            for cap in CAP_LADDER
            for row in by_cap.get(cap, [])
            if row["item_id"] == item_id
        ]
        item_rows = {
            int(row["metadata"]["diagnostic_cap"]): row
            for row in item_attempts
        }
        required_caps[item_id] = next(
            (
                cap
                for cap in CAP_LADDER
                if cap in item_rows
                and item_rows[cap]["status"] not in {"TRUNCATED_THINKING", "RUNTIME_ERROR"}
            ),
            None,
        )
    maximum_required = (
        max(cap for cap in required_caps.values() if cap is not None)
        if all(cap is not None for cap in required_caps.values())
        else None
    )
    proposed_cap = maximum_required
    completion_rows = {
        item_id: next(
            row
            for row in (
                item
                for cap in CAP_LADDER
                for item in by_cap.get(cap, [])
                if item["item_id"] == item_id
            )
            if int(row["metadata"]["diagnostic_cap"]) == required_caps[item_id]
        )
        for item_id in item_ids
        if required_caps[item_id] is not None
    }
    cap_rows: list[dict[str, Any]] = []
    for cap in CAP_LADDER:
        cap_results = {row["item_id"]: row for row in by_cap.get(cap, [])}
        completed = [
            item_id
            for item_id, required in required_caps.items()
            if required is not None and required <= cap
        ]
        completed_rows = [completion_rows[item_id] for item_id in completed]
        token_counts = [int(row["token_count"]) for row in completed_rows]
        all_complete = len(completed) == len(item_ids)
        cap_rows.append(
            {
                "cap": cap,
                "items_observed": len(cap_results),
                "items_completed": len(completed),
                "all_items_completed": all_complete,
                "item_required_caps": required_caps,
                "token_count_min": min(token_counts) if token_counts else None,
                "token_count_median": statistics.median(token_counts) if token_counts else None,
                "token_count_max": max(token_counts) if token_counts else None,
            }
        )
    report = {
        "candidate": manifest["identity"]["candidate"],
        "classification": "DEVELOPMENT_ONLY_NOT_SCIENTIFIC_OUTCOMES",
        "rule": "largest per-item cap required for a non-truncated diagnostic outcome",
        "accuracy_used_for_selection": False,
        "proposed_cap": proposed_cap,
        "operationally_expensive": proposed_cap is None or proposed_cap >= 32768,
        "high_cap_warning": proposed_cap is not None and proposed_cap >= 32768,
        "caps": cap_rows,
    }
    output = args.output or args.run / "cap_recommendation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if proposed_cap is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
