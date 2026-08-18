#!/usr/bin/env python3
"""Select a prospective cap from a completed completion diagnostic.

The rule is mechanical: choose the smallest ladder cap at which every sampled
diagnostic item reached a non-truncated outcome.  Accuracy is never consulted.
If no ladder cap completes every item, the candidate is operationally expensive
and must not advance automatically.
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
    cap_rows: list[dict[str, Any]] = []
    proposed_cap: int | None = None
    for cap in CAP_LADDER:
        cap_results = {row["item_id"]: row for row in by_cap.get(cap, [])}
        completed = [
            row for item_id, row in cap_results.items()
            if row["status"] not in {"TRUNCATED_THINKING", "RUNTIME_ERROR"}
        ]
        token_counts = [int(row["token_count"]) for row in completed]
        all_complete = len(completed) == len(item_ids)
        cap_rows.append(
            {
                "cap": cap,
                "items_observed": len(cap_results),
                "items_completed": len(completed),
                "all_items_completed": all_complete,
                "token_count_min": min(token_counts) if token_counts else None,
                "token_count_median": statistics.median(token_counts) if token_counts else None,
                "token_count_max": max(token_counts) if token_counts else None,
            }
        )
        if proposed_cap is None and all_complete:
            proposed_cap = cap
    report = {
        "candidate": manifest["identity"]["candidate"],
        "classification": "DEVELOPMENT_ONLY_NOT_SCIENTIFIC_OUTCOMES",
        "rule": "smallest cap at which every diagnostic item is non-truncated",
        "accuracy_used_for_selection": False,
        "proposed_cap": proposed_cap,
        "operationally_expensive": proposed_cap is None,
        "caps": cap_rows,
    }
    output = args.output or args.run / "cap_recommendation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if proposed_cap is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
