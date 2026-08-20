#!/usr/bin/env python3
"""Prepare Gate-5 fresh manifests and the frozen random-controller bank.

Local mode scans preserved artifacts and constructs R1-R3 without model data.
Remote mode is limited to loading the pinned CRUXEval dataset and writing the
120 normalized fresh items; it never loads Qwen or runs inference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate5 import (  # noqa: E402
    DATASET_REVISION,
    RANDOM_SEEDS,
    SYSTEM_CAREFUL,
    SYSTEM_DIRECT,
    controller_metadata,
    random_controller_bank,
)
from epistemic_geometry.experiments.micro_q1 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

CRUX_REPO = "cruxeval-org/cruxeval"
NAMESPACE = "GATE5-SOURCE-DURATION-CRUX-FRESH-ALLOCATION"
SEED = 20260820
SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")
GROUPS = (("SOURCE_CHECK", 40), ("SUSTAINED_MANIPULATION", 20), ("SUSTAINED_EVALUATION", 60))


def historical_cruxeval_ids(root: Path = ROOT) -> tuple[str, ...]:
    """Find every preserved CRUXEval ID without loading the dataset."""

    found: set[str] = set()
    for path in (root / "review").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        try:
            found.update(SAMPLE_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not found:
        raise RuntimeError("no historical CRUXEval IDs found")
    return tuple(sorted(found, key=lambda value: int(value.split("_")[1])))


def write_exclusion(path: Path) -> dict[str, Any]:
    ids = historical_cruxeval_ids()
    payload = {
        "benchmark": "CRUXEval",
        "dataset_repo": CRUX_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_ids": list(ids),
        "historical_exclusion_digest": stable_digest(
            NAMESPACE, "HISTORICAL_EXCLUSION", canonical_json(ids)
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
    """Load only the pinned dataset and emit the outcome-independent split."""

    require_remote_hf_execution("Gate 5 CRUXEval manifest preparation")
    from datasets import load_dataset

    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    excluded = set(str(value) for value in exclusion["historical_ids"])
    dataset = load_dataset(CRUX_REPO, split="test", revision=DATASET_REVISION)
    candidates = [row for row in dataset if str(row["id"]) not in excluded]
    candidates.sort(
        key=lambda row: (stable_digest(NAMESPACE, SEED, str(row["id"])), str(row["id"]))
    )
    requested = sum(count for _, count in GROUPS)
    if len(candidates) < requested:
        raise RuntimeError(f"GATE5_BLOCKED_INSUFFICIENT_FRESH_ITEMS: {len(candidates)}")
    selected = candidates[:requested]
    output_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    all_ids: list[str] = []
    for group, count in GROUPS:
        records = []
        for row in selected[offset : offset + count]:
            item_id = str(row["id"])
            prompt = _task_prompt(str(row["code"]), str(row["input"]))
            all_ids.append(item_id)
            records.append(
                {
                    "allocation": group,
                    "item_id": item_id,
                    "benchmark": "CRUXEval",
                    "subtask": "output_prediction",
                    "prompt": prompt,
                    "reference_answer": str(row["output"]),
                    "evaluator": "python_literal",
                    "source_revision": DATASET_REVISION,
                    "prompt_hash": stable_digest("GATE5-TASK-PROMPT", prompt),
                    "metadata": {
                        "dataset_repo": CRUX_REPO,
                        "dataset_revision": DATASET_REVISION,
                        "selection_namespace": NAMESPACE,
                        "selection_seed": SEED,
                        "official_id": item_id,
                        "source_system_prompts": {
                            "careful": SYSTEM_CAREFUL,
                            "direct": SYSTEM_DIRECT,
                        },
                    },
                }
            )
        payload = {
            "allocation": group,
            "items": records,
            "n_items": len(records),
            "dataset_repo": CRUX_REPO,
            "dataset_revision": DATASET_REVISION,
            "selection_namespace": NAMESPACE,
            "selection_seed": SEED,
            "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
            "manifest_hash": stable_digest("GATE5-ALLOCATION", group, canonical_json(records)),
        }
        (output_dir / f"{group}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        offset += count
    if len(all_ids) != len(set(all_ids)) or set(all_ids) & excluded:
        raise RuntimeError("Gate 5 allocation is not fresh and unique")
    summary = {
        "dataset_repo": CRUX_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
        "historical_excluded_count": len(excluded),
        "allocation_counts": dict(GROUPS),
        "all_ids_digest": stable_digest("GATE5-ALL-ALLOCATED-IDS", canonical_json(all_ids)),
        "source": "RunPod HF_HOME=/workspace/hf-cache; model not loaded",
    }
    (output_dir / "ALLOCATION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def freeze_random_bank(gate4_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    meaningful = np.load(gate4_dir / "DIRECTION.npy", allow_pickle=False).astype(np.float64)
    r0 = np.load(gate4_dir / "RANDOM_DIRECTION.npy", allow_pickle=False).astype(np.float64)
    bank = random_controller_bank(meaningful, r0, seeds=RANDOM_SEEDS)
    for name, vector in bank.items():
        np.save(output_dir / f"{name}.npy", vector)
    metadata = controller_metadata(bank, seeds=RANDOM_SEEDS)
    metadata.update(
        {
            "meaningful_direction_sha256": vector_sha256(meaningful),
            "gate4_random_direction_sha256": vector_sha256(r0),
            "layer": 17,
            "alpha": 8.39900588973121,
            "random_seeds": list(RANDOM_SEEDS),
            "bank_sha256": {
                name: stable_digest(
                    "GATE5-RANDOM-BANK-FILE", name, (output_dir / f"{name}.npy").read_bytes()
                )
                for name in bank
            },
        }
    )
    (output_dir / "RANDOM_BANK_METADATA.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-exclusion", type=Path)
    parser.add_argument("--remote-exclusion", type=Path)
    parser.add_argument("--remote-output", type=Path)
    parser.add_argument("--freeze-bank", action="store_true")
    parser.add_argument("--gate4-dir", type=Path, default=ROOT / "review" / "micro_q1")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "review" / "gate5_source_duration"
    )
    args = parser.parse_args()
    if args.write_exclusion:
        print(json.dumps(write_exclusion(args.write_exclusion), indent=2, sort_keys=True))
    if args.remote_exclusion and args.remote_output:
        print(
            json.dumps(
                prepare_remote(args.remote_exclusion, args.remote_output), indent=2, sort_keys=True
            )
        )
    if args.freeze_bank:
        print(
            json.dumps(
                freeze_random_bank(args.gate4_dir, args.output_dir), indent=2, sort_keys=True
            )
        )
    if not any(
        (args.write_exclusion, args.remote_exclusion and args.remote_output, args.freeze_bank)
    ):
        parser.error(
            "choose --write-exclusion, --remote-exclusion/--remote-output, or --freeze-bank"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
