#!/usr/bin/env python3
"""Recompute E3-10 calibration summaries from stored score vectors only.

This script intentionally never imports Torch, Transformers, or datasets.  It
is an independent CPU audit of baseline calibration artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from epistemic_geometry.benchmarks.e3.qualification import (
    CalibrationScoreRow,
    select_cells,
    summarize_cell,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path, help="JSONL baseline score-vector artifact")
    parser.add_argument("--output", type=Path, required=True, help="JSON summary output")
    args = parser.parse_args()
    grouped: dict[tuple[str, str], list[CalibrationScoreRow]] = defaultdict(list)
    with args.scores.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = CalibrationScoreRow.from_record(json.loads(line))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid score row at line {line_number}: {exc}") from exc
            grouped[(row.family, row.cell)].append(row)
    summaries = [summarize_cell(rows) for rows in grouped.values()]
    selected = select_cells(summaries)
    result = {"summaries": summaries, "selected": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(summaries), "families_selected": sorted(selected)}, indent=2))


if __name__ == "__main__":
    main()
