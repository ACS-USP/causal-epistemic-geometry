#!/usr/bin/env python3
"""Prepare the Gate-4 fresh CRUXEval allocation.

The exclusion digest can be built locally.  Dataset resolution/materialization
is deliberately a remote-only command and requires ``HF_HOME=/workspace/hf-cache``.
The output is a small protocol manifest; it contains no model outcomes.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

CRUX_REPO = "cruxeval-org/cruxeval"
CRUX_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
SELECTION_NAMESPACE = "GATE4-MICRO-Q1-CRUX-FRESH-ALLOCATION"
SELECTION_SEED = 20260819
SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")


def historical_cruxeval_ids(root: Path = ROOT) -> tuple[str, ...]:
    """Scan preserved local manifests/journals without loading CRUXEval data."""

    found: set[str] = set()
    review = root / "review"
    for path in review.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found.update(SAMPLE_ID.findall(text))
    if not found:
        raise RuntimeError("no historical CRUXEval IDs found in preserved review artifacts")
    return tuple(sorted(found, key=lambda value: int(value.split("_")[1])))


def write_exclusion(path: Path) -> dict[str, Any]:
    ids = historical_cruxeval_ids()
    payload = {
        "benchmark": "CRUXEval",
        "dataset_repo": CRUX_REPO,
        "dataset_revision": CRUX_REVISION,
        "historical_ids": list(ids),
        "historical_exclusion_digest": stable_digest(
            "GATE4-HISTORICAL-CRUX-IDS", canonical_json(ids)
        ),
        "source": "local preserved manifests/journals only; no dataset loaded",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _task_prompt(code: str, value: str) -> str:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def prepare_remote(exclusion_path: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize 130 new rows from the pinned dataset on RunPod only."""

    require_remote_hf_execution("Gate 4 CRUXEval allocation")
    from datasets import load_dataset

    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded = set(str(value) for value in exclusion["historical_ids"])
    dataset = load_dataset(CRUX_REPO, split="test", revision=CRUX_REVISION)
    candidates = [row for row in dataset if str(row["id"]) not in excluded]
    candidates.sort(
        key=lambda row: (
            stable_digest(SELECTION_NAMESPACE, SELECTION_SEED, str(row["id"])),
            str(row["id"]),
        )
    )
    if len(candidates) < 130:
        raise RuntimeError(f"MICRO_Q1_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(candidates)}")
    selected = candidates[:130]
    groups = (
        ("DIRECTION_CONSTRUCTION", 64),
        ("DIRECTION_VALIDATION", 16),
        ("MICRO_Q1_EVALUATION", 50),
    )
    records: list[dict[str, Any]] = []
    offset = 0
    for group, count in groups:
        for row in selected[offset : offset + count]:
            item_id = str(row["id"])
            prompt = _task_prompt(str(row["code"]), str(row["input"]))
            records.append(
                {
                    "allocation": group,
                    "item_id": item_id,
                    "benchmark": "CRUXEval",
                    "subtask": "output_prediction",
                    "prompt": prompt,
                    "reference_answer": str(row["output"]),
                    "evaluator": "python_literal",
                    "source_revision": CRUX_REVISION,
                    "prompt_hash": stable_digest("MICRO-Q1-TASK-PROMPT", prompt),
                    "metadata": {
                        "dataset_repo": CRUX_REPO,
                        "dataset_revision": CRUX_REVISION,
                        "selection_namespace": SELECTION_NAMESPACE,
                        "selection_seed": SELECTION_SEED,
                        "official_id": item_id,
                    },
                }
            )
        offset += count
    ids = [record["item_id"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & excluded:
        raise RuntimeError("Gate 4 allocation is not fresh and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, _count in groups:
        rows = [record for record in records if record["allocation"] == group]
        (output_dir / f"{group}.json").write_text(
            json.dumps(
                {
                    "allocation": group,
                    "items": rows,
                    "n_items": len(rows),
                    "dataset_repo": CRUX_REPO,
                    "dataset_revision": CRUX_REVISION,
                    "selection_namespace": SELECTION_NAMESPACE,
                    "selection_seed": SELECTION_SEED,
                    "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
                    "manifest_hash": stable_digest(
                        "MICRO-Q1-ALLOCATION", group, canonical_json(rows)
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    summary = {
        "dataset_repo": CRUX_REPO,
        "dataset_revision": CRUX_REVISION,
        "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
        "historical_excluded_count": len(excluded),
        "allocation_counts": {group: count for group, count in groups},
        "all_ids_digest": stable_digest("MICRO-Q1-ALL-ALLOCATED-IDS", canonical_json(ids)),
        "source": "RunPod HF_HOME=/workspace/hf-cache",
    }
    (output_dir / "ALLOCATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-exclusion", type=Path)
    parser.add_argument("--remote-exclusion", type=Path)
    parser.add_argument("--remote-output", type=Path)
    args = parser.parse_args()
    if args.write_exclusion:
        print(json.dumps(write_exclusion(args.write_exclusion), indent=2, sort_keys=True))
    if args.remote_exclusion and args.remote_output:
        print(
            json.dumps(
                prepare_remote(args.remote_exclusion, args.remote_output), indent=2, sort_keys=True
            )
        )
    if not args.write_exclusion and not (args.remote_exclusion and args.remote_output):
        parser.error("choose --write-exclusion or both remote arguments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
