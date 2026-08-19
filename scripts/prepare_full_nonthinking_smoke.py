#!/usr/bin/env python3
"""Prepare Gate 1 manifests without model inference.

Character-count preparation is safe locally.  CRUXEval preparation is
explicitly remote-only because it loads the official dataset through the
RunPod HuggingFace cache.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.adapters import adapter_for  # noqa: E402
from epistemic_geometry.benchmarks.v4.character_count import (  # noqa: E402
    generate_full_nonthinking_smoke_manifest,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

CRUX_REPO = "cruxeval-org/cruxeval"
HISTORICAL_CRUX_IDS = frozenset(
    {
        "sample_74",
        "sample_375",
        "sample_554",
        "sample_281",
        "sample_476",
        "sample_149",
        "sample_700",
        "sample_125",
        "sample_777",
        "sample_496",
        "sample_698",
        "sample_145",
        "sample_251",
        "sample_21",
        "sample_377",
        "sample_728",
        "sample_383",
        "sample_791",
        "sample_376",
        "sample_45",
        "sample_659",
        "sample_300",
        "sample_745",
    }
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_charcount(output: Path, *, seed: int) -> dict[str, object]:
    manifest = generate_full_nonthinking_smoke_manifest(seed=seed, n_items=20)
    _write_json(output, manifest)
    return {
        "instrument": "FRESH_PSEUDOWORD_LONG",
        "path": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "n_items": manifest["n_items"],
    }


def _cruxeval_prompt(code: str, value: str) -> str:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def prepare_cruxeval(
    output: Path, *, seed: int, requested_revision: str | None
) -> dict[str, object]:
    require_remote_hf_execution("Gate 1 CRUXEval dataset preparation")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(CRUX_REPO, revision=requested_revision or "main")
    revision = str(getattr(info, "sha", "") or "")
    if not revision:
        raise RuntimeError("CRUXEval dataset revision could not be resolved immutably")
    dataset = load_dataset(CRUX_REPO, split="test", revision=revision)
    rows = []
    for row in dataset:
        item_id = str(row["id"])
        if item_id in HISTORICAL_CRUX_IDS:
            continue
        rows.append(row)
    rows.sort(
        key=lambda row: (
            stable_digest("FULL-NONTHINKING-CRUX-SELECTION", seed, str(row["id"])),
            str(row["id"]),
        )
    )
    selected = rows[:20]
    if len(selected) != 20:
        raise RuntimeError(f"CRUXEval has only {len(selected)} fresh rows after exclusions")
    records: list[dict[str, object]] = []
    for row in selected:
        records.append(
            {
                "item_id": str(row["id"]),
                "benchmark": "CRUXEval",
                "subtask": "output_prediction",
                "prompt": _cruxeval_prompt(str(row["code"]), str(row["input"])),
                "reference_answer": str(row["output"]),
                "evaluator": "python_literal",
                "source_revision": revision,
                "metadata": {
                    "dataset_repo": CRUX_REPO,
                    "dataset_revision": revision,
                    "official_evaluator": "facebookresearch/cruxeval/evaluation",
                    "selection_namespace": "FULL-NONTHINKING-CRUX-SELECTION",
                    "historical_ids_excluded": sorted(HISTORICAL_CRUX_IDS),
                },
            }
        )
    adapter = adapter_for("CRUXEval")
    items = adapter.load_items(_write_jsonl_and_return(output, records))
    manifest = {
        "suite": "Q1_GATE_1_FULL_NONTHINKING_SMOKE",
        "instrument": "CRUXEVAL_SEMANTIC",
        "dataset_repo": CRUX_REPO,
        "dataset_revision": revision,
        "selection_seed": seed,
        "excluded_historical_ids": sorted(HISTORICAL_CRUX_IDS),
        "items": [item.to_record() for item in items],
        "manifest_hash": stable_digest(
            "FULL-NONTHINKING-CRUX-MANIFEST",
            canonical_json([item.to_record() for item in items]),
        ),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _write_json(manifest_path, manifest)
    return {
        "instrument": "CRUXEVAL_SEMANTIC",
        "path": str(output),
        "manifest_path": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
        "dataset_revision": revision,
        "n_items": len(items),
    }


def _write_jsonl_and_return(path: Path, records: list[dict[str, object]]) -> Path:
    _write_jsonl(path, records)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charcount-output", type=Path)
    parser.add_argument("--cruxeval-output", type=Path)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--cruxeval-revision")
    args = parser.parse_args()
    if not args.charcount_output and not args.cruxeval_output:
        parser.error("request at least one output")
    summaries: list[dict[str, object]] = []
    if args.charcount_output:
        summaries.append(prepare_charcount(args.charcount_output, seed=args.seed))
    if args.cruxeval_output:
        summaries.append(
            prepare_cruxeval(
                args.cruxeval_output,
                seed=args.seed,
                requested_revision=args.cruxeval_revision,
            )
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
