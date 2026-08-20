#!/usr/bin/env python3
"""Freeze the outcome-independent Gate-6.1 source attrition repair.

This script is CPU/model-free.  It audits the historical source attempt,
reconstructs the original candidate order, allocates deterministic reserves,
and writes the pre-outcome repair lock.  It never loads Qwen or evaluates a
model answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6 import source_seed  # noqa: E402

REVIEW = ROOT / "review" / "gate6_layer_source_rfm_atlas"
DATASET_REPO = "cruxeval-org/cruxeval"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_NEW_TOKENS = 4096
TRAIN_TARGET = 104
VALIDATION_TARGET = 32
TRAIN_MAX_INELIGIBLE = 20
VALIDATION_MAX_INELIGIBLE = 6
TRAIN_RESERVE_COUNT = 32
VALIDATION_RESERVE_COUNT = 16
TRAIN_RESERVE_NAMESPACE = "GATE6-SOURCE-TRAIN-RESERVE-V1"
VALIDATION_RESERVE_NAMESPACE = "GATE6-SOURCE-VALIDATION-RESERVE-V1"
SAMPLE_ID = re.compile(r"\bsample_[0-9]+\b")
REPAIR_OUTPUT_NAMES = {
    "SOURCE_ATTRITION_REPAIR_LOCK.json",
    "SOURCE_ATTRITION_REPAIR_LOCK.md",
    "SOURCE_ATTEMPTS.json",
    "SOURCE_ATTEMPT_PROVENANCE_AUDIT.md",
    "SOURCE_CANDIDATE_ORDER.json",
    "SOURCE_TRAIN_RESERVE.json",
    "SOURCE_VALIDATION_RESERVE.json",
    "SOURCE_RESERVE_ALLOCATION.json",
    "SOURCE_HISTORICAL_EXCLUSION.json",
    "SOURCE_CONDITION_JOURNAL.jsonl",
    "SOURCE_ATTRITION_LEDGER.csv",
    "SOURCE_ATTRITION_SUMMARY.json",
    "SOURCE_ATTRITION_BIAS_DIAGNOSTIC.json",
    "SOURCE_ATTRITION_REPORT.md",
    "SOURCE_SELECTED_TRAIN.json",
    "SOURCE_SELECTED_VALIDATION.json",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_digest(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            encoded = part
        else:
            encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit(revision: str = "HEAD") -> str:
    return subprocess.run(
        ["git", "rev-parse", revision], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def historical_ids() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "review").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        if path.parent == REVIEW and path.name in REPAIR_OUTPUT_NAMES:
            continue
        try:
            found.update(SAMPLE_ID.findall(path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    if not found:
        raise RuntimeError("no preserved CRUXEval IDs found")
    return found


def load_items(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["items"] if isinstance(payload, dict) else payload
    return [dict(row) for row in rows]


def task_prompt(code: str, value: str) -> str:
    return (
        "Solve this Python code-output prediction problem.\n\n"
        "Function:\n"
        f"```python\n{code}\n```\n\n"
        f"Input: {value}\n\n"
        "Return exactly one final line in this form:\n"
        "FINAL: <the exact Python output>\n"
        "Do not add any text after FINAL."
    )


def normalize_dataset_row(row: dict[str, Any], allocation: str) -> dict[str, Any]:
    item_id = str(row.get("id", row.get("item_id")))
    prompt = (
        str(row["prompt"]) if "prompt" in row else task_prompt(str(row["code"]), str(row["input"]))
    )
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
        "prompt_hash": stable_digest("EXTERNAL-PROMPT", prompt),
        "item_hash": stable_digest(
            "EXTERNAL-ITEM",
            "CRUXEval",
            "output_prediction",
            item_id,
            stable_digest("EXTERNAL-PROMPT", prompt),
            reference,
            "python_literal",
            DATASET_REVISION,
        ),
        "metadata": {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "selection_policy": "mechanical_reserve_allocation_only",
        },
    }


def original_order() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = load_items(REVIEW / "SOURCE_TRAIN_MANIFEST.json")
    validation = load_items(REVIEW / "SOURCE_VALIDATION.json")
    if len(train) != TRAIN_TARGET or len({row["item_id"] for row in train}) != TRAIN_TARGET:
        raise RuntimeError("historical SOURCE_TRAIN order is not exactly 104 unique items")
    if (
        len(validation) != VALIDATION_TARGET
        or len({row["item_id"] for row in validation}) != VALIDATION_TARGET
    ):
        raise RuntimeError("historical SOURCE_VALIDATION order is not exactly 32 unique items")
    return train, validation


def reserve_rows(
    dataset_rows: list[dict[str, Any]],
    *,
    excluded: set[str],
    namespace: str,
    count: int,
    forbidden: set[str],
) -> list[dict[str, Any]]:
    normalized = [normalize_dataset_row(row, "RESERVE") for row in dataset_rows]
    candidates = [row for row in normalized if row["item_id"] not in excluded | forbidden]
    candidates.sort(key=lambda row: (stable_digest(namespace, row["item_id"]), row["item_id"]))
    if len(candidates) < count:
        raise RuntimeError(f"{namespace}: insufficient deterministic reserve candidates")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(candidates[:count]):
        output.append(
            {
                **row,
                "reserve_namespace": namespace,
                "reserve_order": index,
                "reserve_key": stable_digest(namespace, row["item_id"]),
            }
        )
    return output


def candidate_entry(
    row: dict[str, Any], *, split: str, candidate_order: int, allocation: str
) -> dict[str, Any]:
    item = dict(row)
    item["allocation"] = allocation
    return {
        "split": split,
        "candidate_order": candidate_order,
        "allocation": allocation,
        "item_id": str(item["item_id"]),
        "item": item,
        "candidate_hash": stable_digest("GATE6-SOURCE-CANDIDATE", canonical_json(item)),
    }


def historical_provenance() -> dict[str, Any]:
    failure = json.loads((REVIEW / "SOURCE_PHASE_FAILURE.json").read_text(encoding="utf-8"))
    partial = REVIEW / "SOURCE_GENERATION_JOURNAL_REMOTE_PARTIAL.jsonl"
    log = REVIEW / "SOURCE_EXECUTION_REMOTE.log"
    lock = json.loads((REVIEW / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    return {
        "attempt_id": "GATE6_SOURCE_ATTEMPT_1",
        "classification": failure["classification"],
        "source_commit_recorded": failure["source_commit"],
        "effective_execution_commit": "a3233771332687acfd3a30ac86011cdfed5c23bf",
        "protocol_lock_source_commit": "e4223f6a8910464f9df479f6cb270c673ad84f20",
        "protocol_lock_sha256": file_sha256(REVIEW / "PROTOCOL_LOCK.json"),
        "partial_journal_sha256": file_sha256(partial),
        "partial_log_sha256": file_sha256(log),
        "source_rows_preserved": failure["source_rows_preserved"],
        "source_rows_expected": failure["source_rows_expected"],
        "failed_item_id": failure["failed_item_id"],
        "failed_condition": failure["failed_condition"],
        "failed_reason": failure["failed_reason"],
        "model": failure["model"],
        "model_revision": failure["model_revision"],
        "max_new_tokens": failure["max_new_tokens"],
        "pod_id": failure["pod_id"],
        "cost_usd": failure["approximate_gpu_cost_usd"],
        "scientific_outcomes_collected": False,
        "controller_construction_completed": False,
        "historical_lock_status": lock.get("status"),
    }


def reuse_decisions(train: list[dict[str, Any]]) -> list[dict[str, Any]]:
    partial_path = REVIEW / "SOURCE_GENERATION_JOURNAL_REMOTE_PARTIAL.jsonl"
    rows = [json.loads(line) for line in partial_path.read_text(encoding="utf-8").splitlines()]
    by_id = {str(row["item_id"]): row for row in rows}
    decisions: list[dict[str, Any]] = []
    for item in train:
        item_id = str(item["item_id"])
        row = by_id.get(item_id)
        if row is None:
            continue
        if item_id == "sample_169":
            decisions.append(
                {
                    "item_id": item_id,
                    "action": "IMPORT_PERMANENT_INELIGIBLE",
                    "reason": "historical frozen-cap no-FINAL outcome; never retry",
                    "source_commit": "a3233771332687acfd3a30ac86011cdfed5c23bf",
                }
            )
            continue
        checks: dict[str, bool] = {}
        for condition in ("ordinary", "careful", "direct"):
            metadata = row[f"{condition}_generation_metadata"]
            checks[f"{condition}_tokens"] = bool(row[f"{condition}_token_ids"])
            checks[f"{condition}_raw"] = isinstance(row[f"{condition}_raw_output"], str)
            checks[f"{condition}_model"] = metadata.get("model") == MODEL
            checks[f"{condition}_revision"] = metadata.get("model_revision") == MODEL_REVISION
            checks[f"{condition}_cap"] = (
                metadata.get("generation", {}).get("max_new_tokens") == MAX_NEW_TOKENS
            )
            checks[f"{condition}_prompt"] = (
                str(row[f"{condition}_prompt_meta"].get("rendered_prompt", "")).find(
                    str(item["prompt"])
                )
                >= 0
            )
            checks[f"{condition}_seed"] = int(metadata.get("generation_seed", -1)) == source_seed(
                item_id, "GENERATION", condition.upper()
            )
        checks["split"] = row.get("split") == "train"
        checks["item_id"] = item_id == str(item["item_id"])
        checks["token_counts"] = all(
            int(row[f"{condition}_generation_metadata"].get("generated_token_count", -1))
            == len(row[f"{condition}_token_ids"])
            for condition in ("ordinary", "careful", "direct")
        )
        decisions.append(
            {
                "item_id": item_id,
                "action": "REUSE_SOURCE_GENERATIONS" if all(checks.values()) else "DO_NOT_REUSE",
                "checks": checks,
                "reason": "all frozen generation invariants match"
                if all(checks.values())
                else "historical invariant mismatch",
                "source_commit": "a3233771332687acfd3a30ac86011cdfed5c23bf",
            }
        )
    return decisions


def write_lock(
    *,
    train_candidates: list[dict[str, Any]],
    validation_candidates: list[dict[str, Any]],
    reserves: dict[str, list[dict[str, Any]]],
    exclusion: set[str],
    provenance: dict[str, Any],
    reuse: list[dict[str, Any]],
) -> dict[str, Any]:
    train_reserve = reserves["train"]
    validation_reserve = reserves["validation"]
    candidate_payload = {
        "schema_version": 1,
        "selection_status": "FROZEN_PRE_OUTCOME",
        "source_commit": git_commit(),
        "historical_exclusion_digest": stable_digest(
            "GATE6-SOURCE-HISTORICAL-EXCLUSION", canonical_json(sorted(exclusion))
        ),
        "train": train_candidates,
        "validation": validation_candidates,
    }
    write_json(REVIEW / "SOURCE_CANDIDATE_ORDER.json", candidate_payload)
    write_json(
        REVIEW / "SOURCE_TRAIN_RESERVE.json",
        {
            "namespace": TRAIN_RESERVE_NAMESPACE,
            "target": TRAIN_RESERVE_COUNT,
            "items": train_reserve,
        },
    )
    write_json(
        REVIEW / "SOURCE_VALIDATION_RESERVE.json",
        {
            "namespace": VALIDATION_RESERVE_NAMESPACE,
            "target": VALIDATION_RESERVE_COUNT,
            "items": validation_reserve,
        },
    )
    allocation = {
        "schema_version": 1,
        "status": "FROZEN_PRE_OUTCOME",
        "train": {
            "target": TRAIN_TARGET,
            "max_ineligible": TRAIN_MAX_INELIGIBLE,
            "original_count": TRAIN_TARGET,
            "reserve_count": TRAIN_RESERVE_COUNT,
            "candidate_order_file": (
                "review/gate6_layer_source_rfm_atlas/SOURCE_CANDIDATE_ORDER.json"
            ),
            "reserve_file": "review/gate6_layer_source_rfm_atlas/SOURCE_TRAIN_RESERVE.json",
        },
        "validation": {
            "target": VALIDATION_TARGET,
            "max_ineligible": VALIDATION_MAX_INELIGIBLE,
            "original_count": VALIDATION_TARGET,
            "reserve_count": VALIDATION_RESERVE_COUNT,
            "candidate_order_file": (
                "review/gate6_layer_source_rfm_atlas/SOURCE_CANDIDATE_ORDER.json"
            ),
            "reserve_file": "review/gate6_layer_source_rfm_atlas/SOURCE_VALIDATION_RESERVE.json",
        },
        "selection_rule": "common_mechanical_eligibility_in_frozen_original_then_reserve_order",
        "unused_reserves_are_not_generated": True,
    }
    write_json(REVIEW / "SOURCE_RESERVE_ALLOCATION.json", allocation)
    attempts = {
        "schema_version": 1,
        "attempt_1": provenance,
        "attempt_2": {
            "attempt_id": "GATE6_SOURCE_ATTRITION_REPAIR",
            "status": "LOCKED_PENDING_EXECUTION",
            "source_commit": git_commit(),
            "reuses": reuse,
            "sample_169_retry": False,
            "new_outputs": False,
        },
    }
    write_json(REVIEW / "SOURCE_ATTEMPTS.json", attempts)
    correction_diff = subprocess.run(
        [
            "git",
            "diff",
            "--stat",
            provenance["protocol_lock_source_commit"],
            provenance["effective_execution_commit"],
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (REVIEW / "SOURCE_ATTEMPT_PROVENANCE_AUDIT.md").write_text(
        "# Gate 6 source-attempt provenance audit\n\n"
        "The first source attempt remains immutable and is classified as "
        "`GATE6_SOURCE_PHASE_INCOMPLETE`. It preserved two item-level rows and "
        "stopped at `sample_169` CAREFUL after the frozen 4096-token cap. No "
        "controller, manipulation, or evaluation outcome was collected.\n\n"
        f"Protocol-lock source commit: `{provenance['protocol_lock_source_commit']}`\n\n"
        f"Effective execution checkout: `{provenance['effective_execution_commit']}`\n\n"
        "The exact diff between those commits is recorded below. It contains "
        "only state/lock/provenance metadata; it does not change the runner, "
        "prompts, manifests, direction, alpha, seed schedule, evaluator, or "
        "intervention implementation.\n\n"
        "```text\n" + correction_diff + "\n```\n\n"
        "The repair is prospective: it adds deterministic mechanical marker "
        "localization, condition-level journaling, and preallocated reserves. "
        "It never regenerates the failed item and never uses correctness for "
        "attrition decisions.\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "status": "FROZEN_PRE_OUTCOME",
        "experiment": "GATE6_SOURCE_ATTRITION_REPAIR",
        "source_commit": git_commit(),
        "model": {"id": MODEL, "revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION},
        "policy": {
            "enable_thinking": False,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "engine": "serial_reference_batch_size_1",
        },
        "instrument": {
            "benchmark": "CRUXEval",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "evaluator": "corrected_deterministic_type_aware_semantics",
        },
        "historical_attempt": provenance,
        "historical_exclusion_digest": stable_digest(
            "GATE6-SOURCE-HISTORICAL-EXCLUSION", canonical_json(sorted(exclusion))
        ),
        "candidate_order_sha256": file_sha256(REVIEW / "SOURCE_CANDIDATE_ORDER.json"),
        "reserve_sha256": {
            "train": file_sha256(REVIEW / "SOURCE_TRAIN_RESERVE.json"),
            "validation": file_sha256(REVIEW / "SOURCE_VALIDATION_RESERVE.json"),
        },
        "allocation": allocation,
        "eligibility": {
            "conditions": ["ORDINARY", "CAREFUL", "DIRECT"],
            "requires_all_conditions": True,
            "requires_completed_generation": True,
            "requires_unambiguous_final_marker": True,
            "requires_exact_generated_token_boundary": True,
            "uses_correctness": False,
            "uses_semantic_outcome": False,
        },
        "reuse": reuse,
        "journal": {
            "path": "review/gate6_layer_source_rfm_atlas/SOURCE_CONDITION_JOURNAL.jsonl",
            "key": ["split", "candidate_item_id", "source_condition"],
            "flush_and_fsync_each_row": True,
            "completed_rows_are_not_regenerated": True,
        },
        "phase_6_1": {
            "screening": "mechanical_eligibility_only",
            "activation_extraction": "selected_common_eligible_items_only",
            "activation_or_rfm_before_screen": False,
        },
        "source_commit_is_code_provenance_before_lock_commit": True,
    }
    write_json(REVIEW / "SOURCE_ATTRITION_REPAIR_LOCK.json", lock)
    (REVIEW / "SOURCE_ATTRITION_REPAIR_LOCK.md").write_text(
        "# Gate 6.1 source attrition repair lock\n\n"
        "This is a prospective, outcome-independent repair of the incomplete "
        "Gate-6 source phase. The original candidate order is preserved, and "
        "reserves are allocated before any new model output. Mechanical "
        "eligibility requires all three source conditions to complete and have "
        "an unambiguous `FINAL` marker mapped to generated-token coordinates. "
        "No correctness, semantic answer, activation, or RFM outcome is used.\n\n"
        f"Code source commit at lock creation: `{lock['source_commit']}`\n\n"
        "`sample_169` is permanently ineligible from Attempt 1 and will never "
        "be retried. See `SOURCE_ATTRITION_REPAIR_LOCK.json`.\n",
        encoding="utf-8",
    )
    return lock


def main() -> int:
    global REVIEW
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    REVIEW = args.review_dir.resolve()
    dataset_rows = [
        json.loads(line)
        for line in args.dataset_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_original, validation_original = original_order()
    excluded = historical_ids()
    train_ids = {str(row["item_id"]) for row in train_original}
    validation_ids = {str(row["item_id"]) for row in validation_original}
    forbidden = train_ids | validation_ids
    train_reserve = reserve_rows(
        dataset_rows,
        excluded=excluded,
        namespace=TRAIN_RESERVE_NAMESPACE,
        count=TRAIN_RESERVE_COUNT,
        forbidden=forbidden,
    )
    validation_reserve = reserve_rows(
        dataset_rows,
        excluded=excluded,
        namespace=VALIDATION_RESERVE_NAMESPACE,
        count=VALIDATION_RESERVE_COUNT,
        forbidden=forbidden | {str(row["item_id"]) for row in train_reserve},
    )
    train_candidates = [
        candidate_entry(row, split="train", candidate_order=index, allocation="ORIGINAL")
        for index, row in enumerate(train_original)
    ] + [
        candidate_entry(
            row, split="train", candidate_order=TRAIN_TARGET + index, allocation="RESERVE"
        )
        for index, row in enumerate(train_reserve)
    ]
    validation_candidates = [
        candidate_entry(row, split="validation", candidate_order=index, allocation="ORIGINAL")
        for index, row in enumerate(validation_original)
    ] + [
        candidate_entry(
            row, split="validation", candidate_order=VALIDATION_TARGET + index, allocation="RESERVE"
        )
        for index, row in enumerate(validation_reserve)
    ]
    lock = write_lock(
        train_candidates=train_candidates,
        validation_candidates=validation_candidates,
        reserves={"train": train_reserve, "validation": validation_reserve},
        exclusion=excluded,
        provenance=historical_provenance(),
        reuse=reuse_decisions(train_original),
    )
    write_json(
        REVIEW / "SOURCE_HISTORICAL_EXCLUSION.json",
        {
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "historical_ids": sorted(excluded, key=lambda value: int(value.split("_")[1])),
            "historical_count": len(excluded),
            "digest": lock["historical_exclusion_digest"],
            "model_inference": False,
        },
    )
    print(
        json.dumps(
            {
                "lock": str(REVIEW / "SOURCE_ATTRITION_REPAIR_LOCK.json"),
                "historical_count": len(excluded),
                "train_candidates": len(train_candidates),
                "validation_candidates": len(validation_candidates),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
