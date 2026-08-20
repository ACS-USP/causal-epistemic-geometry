#!/usr/bin/env python3
"""Prepare and audit the prospective Gate 6.2 source-only continuation.

This script is deliberately CPU-only.  It reads the immutable Gate 6.1 source
artifacts, verifies their identity, copies only the already-frozen Gate-6
development manifests, and writes the prospective lock inputs.  It never
loads a model and never evaluates a benchmark answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "review" / "gate6_layer_source_rfm_atlas"
OUTPUT = ROOT / "review" / "gate6_2_first_stage_repair_mean_bridge"
LAYERS = (8, 12, 17, 22, 27, 32)
LOCATIONS = ("PROMPT_BOUNDARY", "EXECUTION_BOUNDARY")
MODEL = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"
EXPECTED_VECTOR_DIMENSION = 4096
EXPECTED_TRAIN = 104
EXPECTED_VALIDATION = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def vector_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).reshape(-1).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _item_ids(path: Path) -> list[str]:
    payload = read_json(path)
    return [str(row["item_id"]) for row in payload["items"]]


def _source_generation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (PARENT / "SOURCE_GENERATIONS.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _source_condition_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (PARENT / "SOURCE_CONDITION_JOURNAL.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _assert_unique(values: list[Any], label: str) -> None:
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate {label}")


def audit_source_artifacts() -> dict[str, Any]:
    required = [
        "SOURCE_SELECTED_TRAIN.json",
        "SOURCE_SELECTED_VALIDATION.json",
        "SOURCE_GENERATIONS.jsonl",
        "SOURCE_CONDITION_JOURNAL.jsonl",
        "SOURCE_ACTIVATIONS.npz",
        "SOURCE_METRICS.json",
        "CONTROLLERS_RAW.json",
        "MEAN_CONTROLLERS_RAW.json",
        "FIRST_STAGE_RESULTS.json",
        "RANDOM_CONTROLLER_RESULTS.json",
        "RANDOM_BANK_METADATA.json",
        "RUN_METADATA.json",
        "SOURCE_PHASE_DECISION.json",
    ]
    missing = [name for name in required if not (PARENT / name).exists()]
    if missing:
        raise RuntimeError(f"missing immutable Gate 6.1 artifacts: {missing}")

    train_ids = _item_ids(PARENT / "SOURCE_SELECTED_TRAIN.json")
    validation_ids = _item_ids(PARENT / "SOURCE_SELECTED_VALIDATION.json")
    if len(train_ids) != EXPECTED_TRAIN or len(validation_ids) != EXPECTED_VALIDATION:
        raise RuntimeError("Gate 6.1 source item counts are not 104/32")
    _assert_unique(train_ids, "train source IDs")
    _assert_unique(validation_ids, "validation source IDs")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("train/validation source IDs overlap")

    generations = _source_generation_rows()
    generation_keys = [(str(row["split"]), str(row["item_id"])) for row in generations]
    _assert_unique(generation_keys, "source generation keys")
    expected_keys = [("train", item_id) for item_id in train_ids] + [
        ("validation", item_id) for item_id in validation_ids
    ]
    if set(generation_keys) != set(expected_keys):
        raise RuntimeError("SOURCE_GENERATIONS does not match selected train/validation IDs")

    conditions = ("ORDINARY", "CAREFUL", "DIRECT")
    for row in generations:
        for condition in conditions:
            token_key = f"{condition.lower()}_token_ids"
            if not row.get(token_key):
                raise RuntimeError(f"missing source tokens for {row['item_id']}:{condition}")
            if row.get(f"{condition.lower()}_final_marker_found") is not True:
                raise RuntimeError(f"source marker missing for {row['item_id']}:{condition}")
    condition_rows = _source_condition_rows()
    condition_keys = [
        (str(row["split"]), str(row["candidate_item_id"]), str(row["source_condition"]))
        for row in condition_rows
    ]
    _assert_unique(condition_keys, "source condition keys")
    selected_condition_keys = {
        ("train", item_id, condition)
        for item_id in train_ids
        for condition in conditions
    } | {
        ("validation", item_id, condition)
        for item_id in validation_ids
        for condition in conditions
    }
    observed_condition_keys = set(condition_keys)
    if not selected_condition_keys.issubset(observed_condition_keys):
        raise RuntimeError("source condition journal is missing a selected source row")
    # The repair runner also preserved six mechanically ineligible train
    # candidates in the journal.  They are provenance rows, not source inputs.
    extra_condition_keys = observed_condition_keys - selected_condition_keys
    if any(key[0] != "train" for key in extra_condition_keys):
        raise RuntimeError("unexpected non-selected validation rows in source journal")
    if len(extra_condition_keys) != 6 * 3:
        raise RuntimeError("unexpected number of preserved ineligible source rows")
    if any("correct" in row or "parsed_answer" in row for row in generations + condition_rows):
        raise RuntimeError("source-only artifacts contain semantic outcome fields")

    activations = np.load(PARENT / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    activation_checks = 0
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        for item_id in ids:
            for location in LOCATIONS:
                for layer in LAYERS:
                    ordinary = f"{split}__ordinary__{location}__{layer}"
                    pair = f"{split}__{location}__careful__{layer}__{item_id}"
                    direct = f"{split}__{location}__direct__{layer}__{item_id}"
                    for key in (pair, direct):
                        if key not in activations or activations[key].shape != (
                            EXPECTED_VECTOR_DIMENSION,
                        ):
                            raise RuntimeError(f"missing or malformed activation key {key}")
                    if ordinary not in activations:
                        raise RuntimeError(f"missing ordinary activation matrix key {ordinary}")
                    activation_checks += 2

    controllers = read_json(PARENT / "CONTROLLERS_RAW.json")
    mean_controllers = read_json(PARENT / "MEAN_CONTROLLERS_RAW.json")
    first_stage = read_json(PARENT / "FIRST_STAGE_RESULTS.json")
    random_results = read_json(PARENT / "RANDOM_CONTROLLER_RESULTS.json")
    random_metadata = read_json(PARENT / "RANDOM_BANK_METADATA.json")
    expected_keys = {f"{location}:L{layer}" for location in LOCATIONS for layer in LAYERS}
    if set(controllers) != expected_keys or set(mean_controllers) != expected_keys:
        raise RuntimeError("Gate 6.1 does not contain exactly 24 RFM/mean records")
    if set(first_stage) != expected_keys | {f"MEAN:{key}" for key in expected_keys}:
        raise RuntimeError("Gate 6.1 first-stage atlas key set is incomplete")
    if set(random_results) != set(first_stage):
        raise RuntimeError("random first-stage key set does not match candidates")
    if set(random_metadata) != set(first_stage):
        raise RuntimeError("random metadata key set does not match candidates")

    vector_checks: list[dict[str, Any]] = []
    for constructor_name, records, _directory in (
        ("RFM_AGOP", controllers, "DIRECTIONS"),
        ("PAIRED_MEAN_DIFFERENCE", mean_controllers, "MEAN_DIRECTIONS"),
    ):
        for key, record in sorted(records.items()):
            path = ROOT / record["direction_path"]
            values = np.load(path, allow_pickle=False).astype(np.float64)
            if values.shape != (EXPECTED_VECTOR_DIMENSION,):
                raise RuntimeError(f"malformed {constructor_name} vector {key}")
            digest = vector_sha256(values)
            if digest != record["vector_hash"]:
                raise RuntimeError(f"vector hash mismatch for {constructor_name} {key}")
            vector_checks.append({"constructor": constructor_name, "key": key, "sha256": digest})

    historical_mean_first_stage: list[dict[str, Any]] = []
    for layer in (22, 27, 32):
        key = f"PROMPT_BOUNDARY:L{layer}"
        mean_key = f"MEAN:{key}"
        historical_mean_first_stage.append(
            {
                "key": mean_key,
                "readout": mean_controllers[key]["readout"],
                "F": first_stage[mean_key]["F"],
                "positive_count": first_stage[mean_key]["positive_count"],
                "random_mean_F": first_stage[mean_key]["random_mean_F"],
                "random_max_F": first_stage[mean_key]["random_max_F"],
                "pass": first_stage[mean_key]["pass"],
                "selection_parent_rule": "RFM_AGOP-only; paired-mean records were not eligible",
            }
        )

    digests = {
        name: sha256_file(PARENT / name)
        for name in (
            "SOURCE_SELECTED_TRAIN.json",
            "SOURCE_SELECTED_VALIDATION.json",
            "SOURCE_GENERATIONS.jsonl",
            "SOURCE_CONDITION_JOURNAL.jsonl",
            "SOURCE_ACTIVATIONS.npz",
            "CONTROLLERS_RAW.json",
            "MEAN_CONTROLLERS_RAW.json",
            "FIRST_STAGE_RESULTS.json",
        )
    }
    return {
        "classification": "GATE6_1_SOURCE_ONLY_AUDIT_CLEAN",
        "parent_artifact_dir": str(PARENT.relative_to(ROOT)),
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "dataset_revision": DATASET_REVISION,
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "source_generation_rows": len(generations),
        "source_condition_rows": len(condition_rows),
        "selected_source_condition_rows": len(selected_condition_keys),
        "preserved_ineligible_condition_rows": len(extra_condition_keys),
        "activation_arrays_checked": activation_checks,
        "vector_records_checked": len(vector_checks),
        "vector_hashes": vector_checks,
        "historical_paired_mean_prompt_first_stage": historical_mean_first_stage,
        "historical_source_only": True,
        "semantic_outcomes_used": False,
        "manipulation_outcomes_present": False,
        "evaluation_outcomes_present": False,
        "artifact_sha256": digests,
        "source_commit": git_commit(),
    }


def copy_frozen_manifests() -> dict[str, str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copies = {
        "CONTROLLER_MANIPULATION.json": "MANIPULATION_MANIFEST.json",
        "CONTROLLER_EVALUATION.json": "EVALUATION_MANIFEST.json",
    }
    hashes: dict[str, str] = {}
    for source_name, destination_name in copies.items():
        source = PARENT / source_name
        if not source.exists():
            raise RuntimeError(f"missing frozen Gate-6 manifest: {source}")
        destination = OUTPUT / destination_name
        shutil.copy2(source, destination)
        hashes[destination_name] = sha256_file(destination)
    return hashes


def copy_immutable_source_inputs() -> dict[str, str]:
    """Copy the audited Gate 6.1 inputs into the Gate 6.2 run directory.

    The source phase must run from one self-contained review directory, while
    the copied files remain byte-for-byte immutable inputs.  In particular,
    this includes the compact activation archive; no source trajectory or
    source item is regenerated here.
    """
    required = (
        "SOURCE_SELECTED_TRAIN.json",
        "SOURCE_SELECTED_VALIDATION.json",
        "SOURCE_GENERATIONS.jsonl",
        "SOURCE_CONDITION_JOURNAL.jsonl",
        "SOURCE_ACTIVATIONS.npz",
    )
    hashes: dict[str, str] = {}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in required:
        source = PARENT / name
        if not source.exists():
            raise RuntimeError(f"missing immutable Gate 6.1 source input: {source}")
        destination = OUTPUT / name
        shutil.copy2(source, destination)
        hashes[name] = sha256_file(destination)
    return hashes


def write_lock(audit: dict[str, Any], manifest_hashes: dict[str, str]) -> None:
    lock = {
        "schema_version": 1,
        "experiment": "GATE6_2_FIRST_STAGE_REPAIR_MEAN_BRIDGE",
        "status": "FROZEN_PRE_OUTCOME",
        "source_commit": git_commit(),
        "parent_gate6_1_head": "c689688047391ecdad8707fba8cb21222210ae8e",
        "model": {"id": MODEL, "revision": MODEL_REVISION, "tokenizer_revision": MODEL_REVISION},
        "instrument": {
            "benchmark": "CRUXEval",
            "dataset_revision": DATASET_REVISION,
            "evaluator": "corrected_deterministic_type_aware_semantics",
        },
        "source_items": {"train": 104, "validation": 32, "ids_reused": True},
        "locations": list(LOCATIONS),
        "layers": list(LAYERS),
        "teacher_forced_scoring": {
            "prompt_boundary": "score_all_continuation_tokens",
            "execution_boundary": "score_marker_token_through_continuation_end",
            "pre_intervention_tokens_excluded": True,
        },
        "rfm": {
            "family": "pinned_xRFM_AGOP",
            "upstream_commit": "773fae81097ab000e6e7292a295e1d24adacca55",
            "internal_validation": "deterministic_stratified_4fold_source_train_only",
            "base_config_grid": [
                {
                    "iters": 8,
                    "bandwidth": 10.0,
                    "exponent": 1.0,
                    "regularization": 1e-3,
                }
            ],
            "semantic_outcomes_used": False,
            "source_validation_consumption": "held_out_readout_first_stage_random_null_only",
        },
        "paired_mean": {
            "constructor": "careful_minus_direct_paired_difference_of_means",
            "historical_prompt_layers_independently_reproduced": [22, 27, 32],
            "best_single": "PROMPT_BOUNDARY:L27",
            "multilayer": ["PROMPT_BOUNDARY:L22", "PROMPT_BOUNDARY:L27", "PROMPT_BOUNDARY:L32"],
        },
        "random_mean": {
            "layers": [22, 27, 32],
            "bank_size": 4,
            "orthogonal_to": "multilayer_mean_subspace",
            "construction_seed_namespace": "GATE6-2-RANDOM-MEAN-BANK",
        },
        "frozen_manifests": manifest_hashes,
        "gate6_1_source_audit": audit,
        "scientific_firewall": {
            "correctness_used_for_controller_construction": False,
            "holdout": "UNTOUCHED",
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "new_source_generation": False,
        },
        "cost_gate_usd": {"target": 1.50, "hard_stop": 3.00},
    }
    write_json(OUTPUT / "PROTOCOL_LOCK.json", lock)
    lines = [
        "# Gate 6.2 — First-Stage Repair and Paired-Mean Controller Bridge",
        "",
        "Status: FROZEN_PRE_OUTCOME.",
        "",
        "This lock reuses the immutable Gate 6.1 source trajectories and source",
        "activations. It does not regenerate source items or use benchmark correctness",
        "during controller construction.",
        "",
        "## Causal scoring repair",
        "",
        "Prompt-boundary scoring includes every continuation token. Execution-boundary",
        "scoring begins at the first token of the final `FINAL:` marker and excludes",
        "all logits produced before the intervention state.",
        "",
        "## Controller policy",
        "",
        "RFM is refit with deterministic four-fold cross-validation entirely inside",
        "SOURCE_TRAIN. SOURCE_VALIDATION is consumed only once for held-out readout,",
        "corrected first-stage diagnostics, and random-null comparison. The paired-mean",
        "bridge independently tests the pre-registered prompt-boundary layers 22, 27,",
        "and 32; layer 27 is the single-controller candidate and all three form the",
        "multilayer candidate. Four orthogonal random mean controls use the same layers.",
        "",
        "## Frozen development data",
        "",
        f"Manipulation manifest SHA-256: `{manifest_hashes['MANIPULATION_MANIFEST.json']}`.",
        f"Evaluation manifest SHA-256: `{manifest_hashes['EVALUATION_MANIFEST.json']}`.",
        "These are copied byte-for-byte from the reviewed Gate 6 manifest files.",
        "",
        "No character count, confirmatory holdout, Q2, new layer, new alpha, or",
        "semantic-label controller is authorized in this protocol.",
        "",
        f"Experiment source commit at lock generation: `{git_commit()}`.",
    ]
    (OUTPUT / "PROTOCOL_LOCK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    audit = audit_source_artifacts()
    manifest_hashes = copy_frozen_manifests()
    source_input_hashes = copy_immutable_source_inputs()
    write_json(OUTPUT / "OFFLINE_SOURCE_AUDIT.json", audit)
    write_json(
        OUTPUT / "MANIFEST_HASHES.json",
        {"frozen_manifests": manifest_hashes, "immutable_source_inputs": source_input_hashes},
    )
    write_lock(audit, manifest_hashes)
    report = [
        "# Gate 6.2 offline source-only audit",
        "",
        f"Classification: `{audit['classification']}`.",
        "",
        (
            f"Train source items: {audit['train_count']}; "
            f"validation source items: {audit['validation_count']}."
        ),
        (
            f"Source generation rows: {audit['source_generation_rows']}; "
            f"condition rows: {audit['source_condition_rows']}."
        ),
        (
            f"Activation arrays checked: {audit['activation_arrays_checked']}; "
            f"vectors checked: {audit['vector_records_checked']}."
        ),
        "",
        "Historical paired-mean prompt-boundary candidates reproduced from the preserved Gate 6.1",
        "first-stage record (source-only; no semantic outcomes):",
        "",
        "| layer | AUROC | positive count | F | random mean F | random max F | historical pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in audit["historical_paired_mean_prompt_first_stage"]:
        layer = row["key"].split(":L", 1)[1]
        readout = row["readout"]
        report.append(
            f"| {layer} | {readout['auroc']:.8f} | {row['positive_count']}/32 | "
            f"{row['F']:.8g} | {row['random_mean_F']:.8g} | {row['random_max_F']:.8g} | "
            f"{row['pass']} |"
        )
    report += [
        "",
        "The parent Gate 6.1 selection rule considered only RFM candidates; these",
        "paired-mean candidates were therefore not promoted by that historical rule.",
        "Gate 6.2 treats them as a separately frozen bridge, without changing the",
        "historical result.",
    ]
    (OUTPUT / "OFFLINE_SOURCE_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), **audit, "manifests": manifest_hashes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
