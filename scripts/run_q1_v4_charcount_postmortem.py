#!/usr/bin/env python3
"""Run the offline, diagnostic parser postmortem on the original 20 rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.v4.character_parser import parse_final_integer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--journal",
        type=Path,
        default=ROOT / "review" / "q1_v4_microbench" / "charcount_qwen" / "journal.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "review" / "q1_v4_microbench" / "CHARCOUNT_SEMANTIC_POSTMORTEM.md",
    )
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.journal.read_text(encoding="utf-8").splitlines()
        if line
    ]
    diagnostics: list[dict[str, object]] = []
    for row in rows:
        status, parsed, reason = parse_final_integer(row.get("raw_output", ""))
        diagnostic_status = status
        if status == "PARSED":
            diagnostic_status = (
                "VALID_CORRECT" if parsed == int(row["reference_answer"]) else "VALID_WRONG"
            )
        diagnostics.append(
            {
                "item_id": row["item_id"],
                "stratum": row["stratum"],
                "original_status": row["status"],
                "corrected_status": diagnostic_status,
                "original_parsed_answer": row.get("parsed_answer"),
                "corrected_parsed_answer": parsed,
                "reference_answer": row["reference_answer"],
                "reason": reason or "deterministic semantic FINAL parse",
            }
        )
    original = Counter(row["original_status"] for row in diagnostics)
    corrected = Counter(row["corrected_status"] for row in diagnostics)
    disagreement = [
        row for row in diagnostics if row["original_status"] != row["corrected_status"]
    ]
    valid = corrected["VALID_CORRECT"] + corrected["VALID_WRONG"]
    accuracy = corrected["VALID_CORRECT"] / valid if valid else None
    accuracy_line = (
        f"- Corrected valid semantic accuracy: `{corrected['VALID_CORRECT']}/{valid}` "
        f"({accuracy:.1%})"
        if accuracy is not None
        else "- Corrected valid semantic accuracy: not defined"
    )
    lines = [
        "# Character-count semantic parser postmortem",
        "",
        "This is a DEVELOPMENT-ONLY diagnostic over the original 20 journaled rows.",
        "The original V4 classifications and reports are preserved unchanged.",
        "The corrected parser is prospective for the previously unrun long stratum.",
        "",
        "## Original versus corrected totals",
        "",
        f"- Original statuses: `{dict(sorted(original.items()))}`",
        f"- Corrected diagnostic statuses: `{dict(sorted(corrected.items()))}`",
        accuracy_line,
        "",
        "All six original `INVALID_FORMAT` rows contain one deterministic final integer",
        "commitment under the corrected parser, including a two-line Markdown heading",
        "variant. Their disagreement cases are listed",
        "below; none was manually relabeled and no reasoning text was searched for an answer.",
        "",
        "## Exact disagreement cases",
        "",
        "| item | stratum | original | corrected | corrected answer | reference | reason |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in disagreement:
        lines.append(
            f"| {row['item_id']} | {row['stratum']} | {row['original_status']} | "
            f"{row['corrected_status']} | {row['corrected_parsed_answer']} | "
            f"{row['reference_answer']} | {row['reason']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The short and medium strata are semantically saturated in this diagnostic",
        "sample: all 20 model commitments are correct once harmless Markdown wrappers",
        "are accepted. This does not qualify the instrument; it removes a parser",
        "confound. The frozen long stratum must still be run before deciding whether",
        "increasing string length creates genuine semantic errors.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_path = args.output.with_name("CHARCOUNT_POSTMORTEM_ROWS.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    print(f"WROTE {args.output}")
    print(f"WROTE {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
