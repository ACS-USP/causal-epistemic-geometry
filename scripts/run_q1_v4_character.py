#!/usr/bin/env python3
"""Run the authorized Q1 V4 character-count development screen on RunPod."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.backends.huggingface import HuggingFaceBackend  # noqa: E402
from epistemic_geometry.benchmarks.v4.character_parser import (  # noqa: E402
    parse_final_integer,
)
from epistemic_geometry.config import BackendConfig  # noqa: E402
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    git_metadata,
    require_remote_hf_execution,
    stable_digest,
    stable_seed,
)
from epistemic_geometry.types import BenchmarkItem  # noqa: E402

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_rows(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    expected = stable_digest(
        "V4-CHARCOUNT-MANIFEST",
        canonical_json(items),
    )
    if expected != manifest.get("manifest_hash"):
        raise ValueError("character-count manifest hash mismatch")
    if len(items) != 30 or len({row["item_id"] for row in items}) != 30:
        raise ValueError("character-count manifest must contain 30 unique items")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--stratum", default=None)
    args = parser.parse_args()
    if args.max_new_tokens != 8192:
        parser.error("V4 character-count cap is prospectively fixed at 8192")
    manifest = _load_rows(args.manifest)
    selected_items = [
        item
        for item in manifest["items"]
        if args.stratum is None or item["stratum"] == args.stratum
    ]
    if not selected_items:
        parser.error(f"no manifest items found for stratum {args.stratum!r}")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "journal.jsonl"
    identity = {
        "instrument": (
            "Q1_V4_CHARCOUNT_LONG_FOLLOWUP"
            if args.stratum is not None
            else "Q1_V4_CHARCOUNT"
        ),
        "manifest_hash": manifest["manifest_hash"],
        "item_ids": [row["item_id"] for row in selected_items],
        "stratum": args.stratum,
        "rollout_index": 0,
        "max_new_tokens": args.max_new_tokens,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": "bf16",
        "enable_thinking": True,
        "generation": {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
        },
        "source_commit": (
            os.environ.get("CEG_SOURCE_COMMIT") or git_metadata(ROOT).get("git_commit")
        ),
        "steering": False,
        "geometry": False,
        "holdout": False,
    }
    identity_hash = stable_digest("V4-CHARCOUNT-RUN", canonical_json(identity))
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("identity_hash") != identity_hash:
            raise RuntimeError("refusing resume: V4 character identity changed")
    _atomic_json(
        manifest_path,
        {
            "status": "RUNNING",
            "identity": identity,
            "identity_hash": identity_hash,
            "started_utc": datetime.now(UTC).isoformat(),
            "model_outcomes": True,
        },
    )
    completed: dict[str, dict[str, Any]] = {}
    if journal.exists():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                key = str(row["item_id"])
                if key in completed and completed[key] != row:
                    raise RuntimeError(f"conflicting duplicate item: {key}")
                completed[key] = row
    require_remote_hf_execution("Q1 V4 character-count inference")
    config = BackendConfig(
        type="huggingface",
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        tokenizer_id=MODEL_ID,
        tokenizer_revision=MODEL_REVISION,
        device="auto",
        dtype="bf16",
        prompt_mode="chat",
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        enable_thinking=True,
        attention_implementation="sdpa",
        execution_mode="serial_reference",
        item_batch_size=1,
        batch_size=1,
    )
    backend = HuggingFaceBackend(config)
    for item in selected_items:
        item_id = str(item["item_id"])
        if item_id in completed:
            continue
        seed = stable_seed("V4-CHARCOUNT-ROLLOUT", manifest["manifest_hash"], item_id, 0)
        started = time.perf_counter()
        try:
            output_row = backend.generate_reasoning(
                BenchmarkItem(
                    id=item_id,
                    prompt=str(item["prompt"]),
                    target=str(item["answer"]),
                    metadata={"stratum": item["stratum"], "text": item["text"]},
                ),
                sampling_seed=seed,
                max_new_tokens=args.max_new_tokens,
            )
            metadata = dict(output_row.metadata)
            token_count = int(metadata.get("generated_token_count", 0))
            status, parsed, reason = parse_final_integer(
                output_row.raw_output,
                truncated=token_count >= args.max_new_tokens,
            )
            if status == "PARSED":
                status = "VALID_CORRECT" if parsed == int(item["answer"]) else "VALID_WRONG"
            row = {
                "item_id": item_id,
                "stratum": item["stratum"],
                "text": item["text"],
                "target_character": item["target_character"],
                "reference_answer": int(item["answer"]),
                "prompt": item["prompt"],
                "prompt_hash": item["prompt_hash"],
                "rollout_seed": seed,
                "raw_output": output_row.raw_output,
                "parsed_answer": parsed,
                "status": status,
                "correct": status == "VALID_CORRECT",
                "token_count": token_count,
                "timing_seconds_wall": time.perf_counter() - started,
                "metadata": {**metadata, "parse_reason": reason},
            }
        except Exception as exc:  # infrastructure failures are journaled, not retried
            row = {
                "item_id": item_id,
                "stratum": item["stratum"],
                "reference_answer": int(item["answer"]),
                "rollout_seed": seed,
                "raw_output": "",
                "parsed_answer": None,
                "status": "RUNTIME_ERROR",
                "correct": False,
                "token_count": None,
                "timing_seconds_wall": time.perf_counter() - started,
                "metadata": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        _append(journal, row)
        print(
            f"completed {item_id} status={row['status']} tokens={row.get('token_count')}",
            flush=True,
        )
    final_rows = [
        json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line
    ]
    expected_count = len(selected_items)
    expected_ids = {row["item_id"] for row in selected_items}
    actual_ids = {row["item_id"] for row in final_rows}
    if len(final_rows) != expected_count or actual_ids != expected_ids:
        raise RuntimeError(
            f"V4 character run did not complete exactly {expected_count} selected rows"
        )
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior.update(
        {
            "status": "COMPLETE",
            "completed_utc": datetime.now(UTC).isoformat(),
            "row_count": len(final_rows),
            "journal_sha256": stable_digest("V4-CHARCOUNT-JOURNAL", journal.read_text()),
        }
    )
    _atomic_json(manifest_path, prior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
