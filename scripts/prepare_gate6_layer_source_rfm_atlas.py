#!/usr/bin/env python3
"""Prepare Gate-6 source and fresh CRUXEval manifests without model inference.

The ``--from-items`` mode is CPU-only and is used by local tests.  The
``--remote`` mode loads only the pinned dataset revision and writes the same
deterministic artifacts; it never constructs a model or calls a GPU backend.
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

from epistemic_geometry.experiments.gate6 import (  # noqa: E402
    DATASET_REPO,
    DATASET_REVISION,
    LAYERS,
    SYSTEM_CAREFUL,
    SYSTEM_DIRECT,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")
SELECTION_NAMESPACE = "GATE6-LAYER-SOURCE-RFM-ATLAS-CRUX-FRESH"
SELECTION_SEED = 20260820
FRESH_GROUPS = (
    ("SOURCE_VALIDATION", 32),
    ("CONTROLLER_MANIPULATION", 20),
    ("CONTROLLER_EVALUATION", 60),
)
SOURCE_TRAIN_GROUPS = ("DIRECTION_CONSTRUCTION", "SOURCE_CHECK")


def historical_ids(root: Path = ROOT) -> tuple[str, ...]:
    found: set[str] = set()
    for path in (root / "review").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        try:
            found.update(SAMPLE_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not found:
        raise RuntimeError("no preserved CRUXEval IDs found")
    return tuple(sorted(found, key=lambda value: int(value.split("_")[1])))


def write_exclusion(path: Path) -> dict[str, Any]:
    ids = historical_ids()
    payload = {
        "benchmark": "CRUXEval",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_ids": list(ids),
        "historical_count": len(ids),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL_EXCLUSION", canonical_json(ids)
        ),
        "source": "preserved local manifests/journals; no model or dataset weights loaded",
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


def source_train_items() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group in SOURCE_TRAIN_GROUPS:
        if group == "DIRECTION_CONSTRUCTION":
            path = ROOT / "review" / "micro_q1" / "CONSTRUCTION_MANIFEST.json"
        else:
            path = ROOT / "review" / "gate5_source_duration" / "SOURCE_CHECK.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["items"])
    if len(records) != 104:
        raise RuntimeError(f"expected 104 source-training records, got {len(records)}")
    ids = [str(row["item_id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("source-training records contain duplicate IDs")
    return records


def _normalize_row(row: dict[str, Any], allocation: str) -> dict[str, Any]:
    item_id = str(row.get("id", row.get("item_id")))
    if "prompt" in row:
        prompt = str(row["prompt"])
    else:
        prompt = _task_prompt(str(row["code"]), str(row["input"]))
    reference = str(row.get("output", row.get("reference_answer")))
    return {
        "allocation": allocation,
        "item_id": item_id,
        "benchmark": "CRUXEval",
        "subtask": "output_prediction",
        "prompt": prompt,
        "reference_answer": reference,
        "evaluator": "python_literal",
        "source_revision": DATASET_REVISION,
        "prompt_hash": stable_digest("GATE6-TASK-PROMPT", prompt),
        "metadata": {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "official_id": item_id,
            "source_system_prompts": {"careful": SYSTEM_CAREFUL, "direct": SYSTEM_DIRECT},
        },
    }


def allocate(
    candidates: list[dict[str, Any]], exclusion: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    excluded = set(map(str, exclusion["historical_ids"]))
    normalized = [_normalize_row(row, "UNASSIGNED") for row in candidates]
    eligible = [row for row in normalized if row["item_id"] not in excluded]
    eligible.sort(
        key=lambda row: (
            stable_digest(SELECTION_NAMESPACE, SELECTION_SEED, row["item_id"]),
            row["item_id"],
        )
    )
    requested = sum(count for _, count in FRESH_GROUPS)
    if len(eligible) < requested:
        raise RuntimeError(f"GATE6_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(eligible)} < {requested}")
    selected = eligible[:requested]
    if len({row["item_id"] for row in selected}) != requested:
        raise RuntimeError("Gate-6 allocation contains duplicate IDs")
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    groups: dict[str, list[str]] = {}
    manifest_hashes: dict[str, str] = {}
    for allocation_name, count in FRESH_GROUPS:
        records = [
            dict(row, allocation=allocation_name) for row in selected[offset : offset + count]
        ]
        payload = {
            "allocation": allocation_name,
            "items": records,
            "n_items": count,
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "selection_namespace": SELECTION_NAMESPACE,
            "selection_seed": SELECTION_SEED,
            "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
        }
        digest = stable_digest("GATE6-ALLOCATION", allocation_name, canonical_json(records))
        payload["manifest_hash"] = digest
        path = output_dir / f"{allocation_name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        groups[allocation_name] = [row["item_id"] for row in records]
        manifest_hashes[allocation_name] = digest
        offset += count
    all_ids = [item_id for values in groups.values() for item_id in values]
    summary = {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
        "historical_excluded_count": len(excluded),
        "allocation_counts": dict(FRESH_GROUPS),
        "groups": groups,
        "all_ids_digest": stable_digest("GATE6-ALL-ALLOCATED-IDS", canonical_json(all_ids)),
        "manifest_hashes": manifest_hashes,
        "layer_set": list(LAYERS),
        "source_train_count": len(source_train_items()),
        "source": "outcome-independent deterministic allocation; no model inference",
    }
    (output_dir / "ALLOCATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-exclusion", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "review" / "gate6_layer_source_rfm_atlas"
    )
    parser.add_argument(
        "--from-items", type=Path, help="JSON list or {items: [...]} for local deterministic tests"
    )
    parser.add_argument(
        "--dataset-jsonl", type=Path, help="Pinned CRUXEval JSONL downloaded without model weights"
    )
    parser.add_argument(
        "--remote", action="store_true", help="load only the pinned CRUXEval dataset"
    )
    args = parser.parse_args()
    exclusion_path = args.write_exclusion or args.output_dir / "HISTORICAL_EXCLUSION_DIGEST.json"
    exclusion = write_exclusion(exclusion_path)
    if args.from_items:
        payload = json.loads(args.from_items.read_text(encoding="utf-8"))
        candidates = payload["items"] if isinstance(payload, dict) else payload
    elif args.dataset_jsonl:
        with args.dataset_jsonl.open(encoding="utf-8") as handle:
            candidates = [json.loads(line) for line in handle if line.strip()]
    elif args.remote:
        require_remote_hf_execution("Gate 6 CRUXEval manifest preparation")
        from datasets import load_dataset

        dataset = load_dataset(DATASET_REPO, split="test", revision=DATASET_REVISION)
        candidates = list(dataset)
    else:
        raise SystemExit(
            "provide --from-items for local tests or --remote on the authorized environment"
        )
    print(json.dumps(allocate(candidates, exclusion, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
