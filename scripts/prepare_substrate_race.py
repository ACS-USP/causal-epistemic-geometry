#!/usr/bin/env python3
"""Prepare the frozen Gate 3 item sets without model inference.

Character-count preparation is safe locally. CRUXEval preparation is explicitly
remote-only because the official dataset must remain on RunPod/HF_HOME.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.external.adapters import adapter_for  # noqa: E402
from epistemic_geometry.benchmarks.v4.character_count import (  # noqa: E402
    generate_substrate_race_manifest,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

CRUX_REPO = "cruxeval-org/cruxeval"
CRUX_SELECTION_NAMESPACE = "GATE3-SUBSTRATE-RACE-CRUX-SELECTION"
SELECTION_SEED = 20260820
_SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")
_HISTORICAL_ROOTS = (
    ROOT / "review" / "external_benchmark_qualification",
    ROOT / "review" / "full_nonthinking_smoke",
    ROOT / "review" / "q1_v4_microbench",
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


def historical_cruxeval_ids() -> tuple[str, ...]:
    """Reconstruct all prior CRUX IDs from preserved local manifests/journals."""

    found: set[str] = set()
    for root in _HISTORICAL_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                continue
            found.update(_SAMPLE_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
    if not found:
        raise RuntimeError("could not reconstruct any historical CRUXEval item IDs")
    return tuple(sorted(found, key=lambda value: int(value.split("_")[1])))


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


def prepare_charcount(output: Path) -> dict[str, object]:
    manifest = generate_substrate_race_manifest(seed=SELECTION_SEED, n_items=20)
    _write_json(output, manifest)
    return {
        "instrument": "FRESH_PSEUDOWORD_LONG",
        "path": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "n_items": manifest["n_items"],
    }


def prepare_cruxeval(output: Path, *, requested_revision: str | None) -> dict[str, object]:
    require_remote_hf_execution("Gate 3 CRUXEval preparation")
    if Path("/workspace/hf-cache").resolve() != Path(os.environ.get("HF_HOME", "")).resolve():
        raise RuntimeError("Gate 3 CRUXEval preparation requires HF_HOME=/workspace/hf-cache")
    from datasets import load_dataset
    from huggingface_hub import HfApi

    excluded = historical_cruxeval_ids()
    info = HfApi().dataset_info(CRUX_REPO, revision=requested_revision or "main")
    revision = str(getattr(info, "sha", "") or "")
    if not revision:
        raise RuntimeError("CRUXEval dataset revision could not be resolved immutably")
    dataset = load_dataset(CRUX_REPO, split="test", revision=revision)
    rows = [row for row in dataset if str(row["id"]) not in set(excluded)]
    rows.sort(
        key=lambda row: (
            stable_digest(CRUX_SELECTION_NAMESPACE, SELECTION_SEED, str(row["id"])),
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
                    "selection_namespace": CRUX_SELECTION_NAMESPACE,
                    "selection_seed": SELECTION_SEED,
                    "historical_ids_excluded": list(excluded),
                },
            }
        )
    jsonl_path = output.with_suffix(".jsonl")
    _write_jsonl(jsonl_path, records)
    items = adapter_for("CRUXEval").load_items(jsonl_path)
    item_records = [item.to_record() for item in items]
    manifest: dict[str, Any] = {
        "suite": "Q1_GATE3_SUBSTRATE_RACE",
        "instrument": "CRUXEVAL_SEMANTIC",
        "dataset_repo": CRUX_REPO,
        "dataset_revision": revision,
        "selection_seed": SELECTION_SEED,
        "selection_namespace": CRUX_SELECTION_NAMESPACE,
        "historical_ids_excluded": list(excluded),
        "historical_exclusion_digest": stable_digest(
            "GATE3-HISTORICAL-CRUX-IDS", canonical_json(excluded)
        ),
        "items": item_records,
        "manifest_hash": stable_digest(
            "GATE3-SUBSTRATE-RACE-CRUX-MANIFEST", canonical_json(item_records)
        ),
    }
    _write_json(output, manifest)
    return {
        "instrument": "CRUXEVAL_SEMANTIC",
        "path": str(jsonl_path),
        "manifest_path": str(output),
        "manifest_hash": manifest["manifest_hash"],
        "dataset_revision": revision,
        "n_items": len(items),
        "historical_excluded": len(excluded),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--charcount-output", type=Path)
    parser.add_argument("--cruxeval-output", type=Path)
    parser.add_argument("--cruxeval-revision")
    args = parser.parse_args()
    if not args.charcount_output and not args.cruxeval_output:
        parser.error("request at least one output")
    summaries: list[dict[str, object]] = []
    if args.charcount_output:
        summaries.append(prepare_charcount(args.charcount_output))
    if args.cruxeval_output:
        summaries.append(
            prepare_cruxeval(args.cruxeval_output, requested_revision=args.cruxeval_revision)
        )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
