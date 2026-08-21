#!/usr/bin/env python3
"""Freeze Gate 7 fresh items, random bank, schedule, precision, and lock.

The local exclusion-only mode never loads a benchmark. Dataset resolution is
restricted to an explicitly authorized remote environment or an injected test
fixture. No model is loaded by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate7 import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    CONDITIONS,
    DATASET_REPO,
    DATASET_REVISION,
    ETA,
    EXPERIMENT_ID,
    LAYER,
    MAX_NEW_TOKENS,
    MEANINGFUL,
    MODEL,
    MODEL_REVISION,
    PARSER_VERSION,
    RANDOM_NAMES,
    REFERENCE_SCALE,
    SELECTION_NAMESPACE,
    allocate_fresh_items,
    build_schedule,
    file_sha256,
    gate7_random_bank,
    historical_cruxeval_ids,
    pseudo_replication_projection,
    vector_sha256,
)
from epistemic_geometry.reproducibility import (  # noqa: E402
    canonical_json,
    require_remote_hf_execution,
    stable_digest,
)

OUTPUT = ROOT / "review" / "gate7_fresh_l27_replication"
CONTROLLER_PATH = (
    ROOT
    / "review"
    / "gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS"
    / "PROMPT_BOUNDARY"
    / "L27.npy"
)
GATE63_LOCK = ROOT / "review" / "gate6_3_single_mean_semantic_evaluation" / "PROTOCOL_LOCK.json"
V3_MODULE = ROOT / "src" / "epistemic_geometry" / "benchmarks" / "external" / "semantic_v3.py"
V3_SPEC = ROOT / "review" / "gate6_3_semantic_validity_audit" / "SEMANTIC_V3_SPEC.md"
V3_CORPUS = ROOT / "review" / "gate6_3_semantic_validity_audit" / "BLINDED_CORPUS.jsonl"
V3_ROWS = ROOT / "review" / "gate6_3_semantic_validity_audit" / "ROW_REANALYSIS_V3.csv"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_exclusion(output: Path) -> dict[str, Any]:
    ids = historical_cruxeval_ids(ROOT / "review", gate7_output=output)
    payload = {
        "benchmark": "CRUXEval",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "historical_ids": list(ids),
        "historical_count": len(ids),
        "historical_exclusion_digest": stable_digest(
            SELECTION_NAMESPACE, "HISTORICAL-EXCLUSION", canonical_json(ids)
        ),
        "source": (
            "all preserved local manifests, journals, reserves, diagnostics, and tracked "
            "allocations; Gate 7 output excluded"
        ),
        "future_or_confirmatory_allocations_excluded_if_present": True,
    }
    write_json(output / "HISTORICAL_EXCLUSION_DIGEST.json", payload)
    return payload


def _gate63_arrays() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    csv.field_size_limit(sys.maxsize)
    with V3_ROWS.open(encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["phase"] == "GATE6_3_PRIMARY_EVALUATION"
        ]
    if len(rows) != 840:
        raise RuntimeError(f"expected 840 preserved Gate-6.3 primary rows; found {len(rows)}")
    item_ids = sorted({row["item_id"] for row in rows})
    by_key = {
        (row["item_id"], row["condition"], int(row["rollout_index"])): int(
            row["v3_correct"] != "True"
        )
        for row in rows
    }
    expected_conditions = ("BASELINE", MEANINGFUL, *(f"SINGLE_L27_RANDOM_R{i}" for i in range(4)))
    arrays = {
        condition: np.asarray(
            [[by_key[(item, condition, rollout)] for rollout in (0, 1)] for item in item_ids],
            dtype=np.int8,
        )
        for condition in expected_conditions
    }
    return arrays.pop("BASELINE"), arrays


def _controller_identity() -> tuple[np.ndarray, dict[str, Any]]:
    gate63 = json.loads(GATE63_LOCK.read_text(encoding="utf-8"))
    vector = np.load(CONTROLLER_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    canonical_hash = vector_sha256(vector)
    expected = str(gate63["controller"]["vector_sha256"])
    if canonical_hash != expected:
        raise RuntimeError(f"meaningful vector mismatch: {canonical_hash} != {expected}")
    if not np.isclose(float(np.linalg.norm(vector)), 1.0, atol=1e-12):
        raise RuntimeError("frozen meaningful vector is not unit norm")
    if not np.isclose(float(gate63["controller"]["eta0"]), ETA, atol=0.0):
        raise RuntimeError("Gate 6.3 eta differs from Gate 7 constant")
    if not np.isclose(float(gate63["controller"]["reference_scale"]), REFERENCE_SCALE, atol=0.0):
        raise RuntimeError("Gate 6.3 reference scale differs from Gate 7 constant")
    delta = vector * ETA * REFERENCE_SCALE
    return vector, {
        "name": MEANINGFUL,
        "source": "PROMPT_BOUNDARY",
        "layer": LAYER,
        "constructor": "PAIRED_MEAN_DIFFERENCE",
        "vector_path": str(CONTROLLER_PATH.relative_to(ROOT)),
        "vector_file_sha256": file_sha256(CONTROLLER_PATH),
        "canonical_float64_vector_sha256": canonical_hash,
        "vector_norm": float(np.linalg.norm(vector)),
        "eta": ETA,
        "reference_scale": REFERENCE_SCALE,
        "per_forward_delta_norm": float(np.linalg.norm(delta)),
        "duration": "sustained_current_token",
        "historical_protocol": str(GATE63_LOCK.relative_to(ROOT)),
    }


def _premortem_pass(output: Path) -> None:
    path = output / "PREMORTEM.json"
    if not path.exists():
        raise RuntimeError("Gate 7 PREMORTEM.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("classification") != "PREMORTEM_PASS":
        raise RuntimeError(f"Gate 7 premortem is not passed: {payload.get('classification')}")


def freeze(candidates: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    _premortem_pass(output)
    exclusion = write_exclusion(output)
    selected, allocation = allocate_fresh_items(candidates, exclusion["historical_ids"])
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "allocation": "GATE7_EVALUATION",
        "items": selected,
        **allocation,
        "selection_outcome_independent": True,
    }
    write_json(output / "EVALUATION_MANIFEST.json", manifest)

    meaningful, controller = _controller_identity()
    bank, bank_metadata = gate7_random_bank(meaningful)
    random_records: dict[str, Any] = {}
    for name in RANDOM_NAMES:
        path = output / f"{name}.npy"
        np.save(path, bank[name].astype(np.float64))
        random_records[name] = {
            **bank_metadata["records"][name],
            "vector_path": str(path.relative_to(ROOT)),
            "vector_file_sha256": file_sha256(path),
            "canonical_float64_vector_sha256": vector_sha256(bank[name]),
            "layer": LAYER,
            "eta": ETA,
            "reference_scale": REFERENCE_SCALE,
            "duration": "sustained_current_token",
        }
    random_payload = {
        "schema_version": 1,
        "namespace": "GATE7-L27-RANDOM-BANK-V1",
        "construction": (
            "deterministic Gaussian; remove meaningful component; ordered Gram-Schmidt; "
            "unit normalize"
        ),
        "meaningful_controller": controller,
        "random_conditions": random_records,
        "geometry": bank_metadata["geometry"],
        "outcome_independent": True,
        "gate6_3_randoms_reused": False,
    }
    write_json(output / "RANDOM_BANK.json", random_payload)

    schedule = build_schedule([row["item_id"] for row in selected])
    write_json(output / "EVALUATION_SCHEDULE.json", schedule)

    baseline, old_conditions = _gate63_arrays()
    precision = pseudo_replication_projection(
        baseline, old_conditions, target_n=len(selected), resamples=2_000, seed=20260821
    )
    precision.update(
        {
            "source": "preserved Gate 6.3 external-semantic-v3 item-cluster outcomes",
            "source_rows_sha256": file_sha256(V3_ROWS),
            "sample_size_rule_unchanged": True,
        }
    )
    write_json(output / "PRECISION_PROJECTION.json", precision)

    parser = {
        "version": PARSER_VERSION,
        "module_sha256": file_sha256(V3_MODULE),
        "specification_sha256": file_sha256(V3_SPEC),
        "blinded_test_corpus_sha256": file_sha256(V3_CORPUS),
        "condition_invariance_test": "PASS",
        "frozen_before_gate7_outputs": True,
    }
    source_commit = git_commit()
    lock = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "FROZEN_PRE_OUTCOME",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_REPLICATION",
        "lock_preparation_source_commit": source_commit,
        "experiment_source_commit_binding": {
            "file": "EXPERIMENT_SOURCE_COMMIT.json",
            "timing": "after lock commit and before any Gate-7 model output",
            "semantics": (
                "Exact clean checkout used for engineering and collection; supplied to "
                "the runner and recorded in every trajectory."
            ),
        },
        "model": {
            "id": MODEL,
            "revision": MODEL_REVISION,
            "tokenizer_revision": MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "enable_thinking": False,
            "generation": "full_autoregressive",
            "max_new_tokens": MAX_NEW_TOKENS,
            "attention": "sdpa",
            "environment_profile": "CORE_QWEN",
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
        },
        "instrument": {
            "benchmark": "CRUXEval semantic output prediction",
            "dataset_repo": DATASET_REPO,
            "dataset_revision": DATASET_REVISION,
            "evaluator": parser,
        },
        "sample": {
            **allocation,
            "manifest_file_sha256": file_sha256(output / "EVALUATION_MANIFEST.json"),
        },
        "controller": controller,
        "random_bank": {
            "conditions": list(RANDOM_NAMES),
            "file_sha256": file_sha256(output / "RANDOM_BANK.json"),
            "records": random_records,
            "geometry": bank_metadata["geometry"],
        },
        "conditions": list(CONDITIONS),
        "rollouts_per_item_condition": 2,
        "seed_regime": "INDEPENDENT_PRIMARY",
        "schedule": {
            "logical_rows": len(schedule),
            "file_sha256": file_sha256(output / "EVALUATION_SCHEDULE.json"),
            "outcome_independent_interleaving": True,
        },
        "primary_outcome": "invalid_as_error; correctness only is e=0",
        "estimands": [
            "accuracy",
            "commitment_validity",
            "semantic_evaluability",
            "B0j",
            "O0j",
            "G",
            "C_unbiased_U_statistic",
            "D_unbiased_two_rollout",
            "rescue",
            "damage",
        ],
        "bootstrap": {
            "unit": "item_cluster_all_conditions_both_rollouts",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "guards": {
            "commitment_minimum": 0.90,
            "commitment_drop_max": 0.05,
            "evaluability_minimum": 0.90,
            "evaluability_drop_max": 0.05,
            "accuracy_drop_max": 0.10,
        },
        "minimum_replication": {
            "G": 0.10,
            "C": 0.05,
            "D": 0.08,
            "G_minus_random_mean": 0.08,
            "C_minus_random_mean": 0.05,
            "D_minus_random_mean": 0.05,
            "greater_than_random_max": ["G", "C", "D"],
            "rescue_gt_damage": True,
        },
        "strong_replication": {
            "bootstrap_lower_bound_positive": [
                "accuracy_change",
                "G",
                "C",
                "G_minus_random_mean",
                "C_minus_random_mean",
            ],
            "accuracy_gain_minimum": 0.08,
            "loo_sign_stable": ["accuracy_change", "G", "C"],
        },
        "source_policy": {
            "condition": "TEXTUAL_CAREFUL_REFERENCE",
            "validity_minimum": 0.90,
            "evaluability_minimum": 0.90,
            "mean_token_ratio_minimum": 1.5,
            "median_token_increase_minimum": 10,
        },
        "style_only_classification": {
            "purpose": (
                "Prospective operationalization of 'clearly reproduces the textual "
                "CAREFUL token/style regime' for the exhaustive classification."
            ),
            "requires_source_policy_replicated": True,
            "mean_textual_token_increase_fraction_recovered_minimum": 0.50,
            "median_textual_token_increase_fraction_recovered_minimum": 0.50,
            "does_not_modify_primary_replication_thresholds": True,
        },
        "classifications": [
            "GATE7_STRONG_SINGLE_L27_REPLICATION",
            "GATE7_MINIMUM_SINGLE_L27_REPLICATION",
            "GATE7_QUALITATIVE_PARTIAL_REPLICATION",
            "GATE7_CAREFUL_STYLE_CONTROL_WITHOUT_SPECIFIC_ERROR_CONTROL",
            "GATE7_NO_REPLICATION",
            "GATE7_DESTRUCTIVE",
            "GATE7_INSTRUMENT_FAILURE",
            "GATE7_ENGINE_FAILURE",
        ],
        "cost": {
            "target_usd": 2.0,
            "hard_stop_usd": 4.0,
            "projected_rows": len(schedule),
            "gate6_3_empirical_seconds_per_row": 6.306324623713432,
            "projected_generation_hours": len(schedule) * 6.306324623713432 / 3600,
            "projected_generation_cost_usd_at_0_44": (
                len(schedule) * 6.306324623713432 / 3600 * 0.44
            ),
        },
        "firewall": {
            "historical_gate6_3_result": "GATE6_3_SINGLE_MEAN_DESTRUCTIVE",
            "historical_result_immutable": True,
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
        },
    }
    write_json(output / "PROTOCOL_LOCK.json", lock)
    (output / "PROTOCOL_LOCK.md").write_text(
        "# Gate 7 prospective protocol lock\n\n"
        "Status: `FROZEN_PRE_OUTCOME`. Lifecycle: `PROSPECTIVE_LOCK`.\n\n"
        "Gate 7 reuses the exact frozen `BEST_SINGLE_MEAN_PLUS` L27 paired-mean "
        "controller, eta, reference scale, sustained current-token hook, Qwen revision, "
        "and external-semantic-v3 evaluator. It allocates a fresh deterministic CRUXEval "
        f"sample of {len(selected)} items and compares baseline, textual CAREFUL, the "
        "meaningful controller, and four new architecture-matched random controllers, "
        "with two independent rollouts per item-condition.\n\n"
        "Meaningful canonical vector SHA-256: "
        f"`{controller['canonical_float64_vector_sha256']}`.\n\n"
        f"Evaluation manifest hash: `{allocation['manifest_hash']}`.\n\n"
        f"Schedule file SHA-256: `{lock['schedule']['file_sha256']}`.\n\n"
        f"Semantic V3 module SHA-256: `{parser['module_sha256']}`.\n\n"
        "No controller, layer, dose, condition, threshold, parser, or sample choice may "
        "change after model outcomes. Q2 and character count are NOT RUN; confirmatory "
        "holdout is UNTOUCHED.\n",
        encoding="utf-8",
    )
    preoutcome_names = (
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "HISTORICAL_EXCLUSION_DIGEST.json",
        "EVALUATION_MANIFEST.json",
        "RANDOM_BANK.json",
        "EVALUATION_SCHEDULE.json",
        "PRECISION_PROJECTION.json",
        *(f"{name}.npy" for name in RANDOM_NAMES),
    )
    write_json(
        output / "artifact_hashes_preoutcome.json",
        {name: file_sha256(output / name) for name in preoutcome_names},
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--write-exclusion-only", action="store_true")
    parser.add_argument("--from-items", type=Path)
    parser.add_argument("--dataset-jsonl", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--bind-source-commit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.bind_source_commit:
        if git_commit() != args.bind_source_commit:
            raise RuntimeError("source binding must equal the current checkout commit")
        payload = {
            "experiment_source_commit": args.bind_source_commit,
            "bound_before_model_outputs": True,
            "protocol_lock_sha256": file_sha256(output / "PROTOCOL_LOCK.json"),
        }
        write_json(output / "EXPERIMENT_SOURCE_COMMIT.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.write_exclusion_only:
        payload = write_exclusion(output)
        print(json.dumps({"historical_count": payload["historical_count"]}, indent=2))
        return 0
    if args.from_items:
        payload = json.loads(args.from_items.read_text(encoding="utf-8"))
        candidates = list(payload["items"] if isinstance(payload, dict) else payload)
    elif args.dataset_jsonl:
        candidates = [
            json.loads(line)
            for line in args.dataset_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.remote:
        require_remote_hf_execution("Gate 7 pinned CRUXEval manifest preparation")
        from datasets import load_dataset

        candidates = list(load_dataset(DATASET_REPO, split="test", revision=DATASET_REVISION))
    else:
        raise SystemExit("provide --from-items/--dataset-jsonl or --remote on the authorized host")
    lock = freeze(candidates, output)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "actual_n": lock["sample"]["actual_n"],
                "logical_rows": lock["schedule"]["logical_rows"],
                "projected_cost_usd": lock["cost"]["projected_generation_cost_usd_at_0_44"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
