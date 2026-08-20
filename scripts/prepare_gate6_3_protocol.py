#!/usr/bin/env python3
"""Freeze the Gate 6.3 random bank and outcome-independent schedules offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments.gate6 import evaluation_seed  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import (  # noqa: E402
    bank_geometry,
    single_layer_random_bank,
    standardized_delta,
    vector_sha256,
)
from epistemic_geometry.reproducibility import git_metadata, stable_seed  # noqa: E402

GATE62 = ROOT / "review" / "gate6_2_first_stage_repair_mean_bridge"
PARSER_MODULE = ROOT / "src" / "epistemic_geometry" / "benchmarks" / "external" / "semantic_v2.py"
PARSER_TESTS = ROOT / "tests" / "test_external_semantic_v2.py"
RANDOM_NAMES = tuple(f"SINGLE_L27_RANDOM_R{i}" for i in range(4))
EVALUATION_CONDITIONS = (
    "BASELINE",
    "TEXTUAL_CAREFUL_REFERENCE",
    "BEST_SINGLE_MEAN_PLUS",
    *RANDOM_NAMES,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare(output: Path) -> dict[str, Any]:
    reanalysis = load_json(output / "SEMANTIC_V2_MANIPULATION_ESTIMANDS.json")
    expected = "GATE6_2A_PARSER_REANALYSIS_SINGLE_MEAN_PASS"
    if reanalysis["classification"] != expected:
        raise RuntimeError(
            "Gate 6.3 lock requires the offline parser-reanalysis single-mean pass; "
            f"observed {reanalysis['classification']}"
        )

    selection = load_json(GATE62 / "CONTROLLER_SELECTION_CORRECTED.json")
    mean_records = load_json(GATE62 / "MEAN_CONTROLLERS_RAW_CORRECTED.json")
    meaningful_path = GATE62 / "PAIRED_MEAN_DIRECTIONS" / "PROMPT_BOUNDARY" / "L27.npy"
    meaningful = np.load(meaningful_path, allow_pickle=False).astype(np.float64).reshape(-1)
    meaningful_hash = vector_sha256(meaningful)
    expected_hash = mean_records["PROMPT_BOUNDARY:L27"]["vector_hash"]
    if meaningful_hash != expected_hash:
        raise RuntimeError(f"frozen L27 vector hash mismatch: {meaningful_hash} != {expected_hash}")
    if float(np.linalg.norm(meaningful)) == 0.0:
        raise RuntimeError("frozen meaningful direction is zero")

    seeds = tuple(stable_seed("GATE6-3-SINGLE-L27-RANDOM-BANK", index) for index in range(4))
    bank = single_layer_random_bank(meaningful, seeds=seeds)
    geometry = bank_geometry(meaningful, bank)
    if not all(
        geometry[key]
        for key in (
            "unit_norm_pass",
            "meaningful_orthogonality_pass",
            "random_pairwise_orthogonality_pass",
        )
    ):
        raise RuntimeError(f"random bank geometry failed: {geometry}")

    eta0 = float(selection["eta0"])
    reference_scale = float(mean_records["PROMPT_BOUNDARY:L27"]["scale"])
    meaningful_delta = standardized_delta(meaningful, eta=eta0, reference_scale=reference_scale)
    random_metadata: dict[str, Any] = {}
    for name, seed in zip(RANDOM_NAMES, seeds, strict=True):
        short_name = name.removeprefix("SINGLE_L27_RANDOM_")
        vector = bank[short_name]
        vector_path = output / f"{name}.npy"
        np.save(vector_path, vector.astype(np.float64))
        delta = standardized_delta(vector, eta=eta0, reference_scale=reference_scale)
        random_metadata[name] = {
            "seed": int(seed),
            "layer": 27,
            "source": "deterministic_gaussian_gram_schmidt",
            "vector_path": str(vector_path.relative_to(ROOT)),
            "vector_sha256": vector_sha256(vector),
            "unit_norm": float(np.linalg.norm(vector)),
            "delta_norm": float(np.linalg.norm(delta)),
            "standardized_eta": eta0,
            "reference_scale": reference_scale,
        }
    random_bank_record = {
        "meaningful_controller": {
            "name": "BEST_SINGLE_MEAN_PLUS",
            "layer": 27,
            "location": "PROMPT_BOUNDARY",
            "vector_path": str(meaningful_path.relative_to(ROOT)),
            "vector_sha256": meaningful_hash,
            "eta0": eta0,
            "reference_scale": reference_scale,
            "delta_norm": float(np.linalg.norm(meaningful_delta)),
        },
        "random_conditions": random_metadata,
        "geometry": geometry,
        "seed_namespace": "GATE6-3-SINGLE-L27-RANDOM-BANK",
        "construction": "unit Gaussian, Gram-Schmidt against meaningful and prior random vectors",
        "outcome_independent": True,
    }
    write_json(output / "SINGLE_L27_RANDOM_BANK.json", random_bank_record)

    old_schedule = load_json(GATE62 / "CONTROLLER_MANIPULATION_SCHEDULE.json")
    baseline_seeds = {
        str(row["item_id"]): int(row["seed"])
        for row in old_schedule
        if str(row["condition"]) == "BASELINE"
    }
    if len(baseline_seeds) != 20:
        raise RuntimeError("Gate 6.2 baseline schedule is not exactly 20 items")
    manipulation_manifest = load_json(GATE62 / "MANIPULATION_MANIFEST.json")
    manipulation_items = [str(row["item_id"]) for row in manipulation_manifest["items"]]
    if set(manipulation_items) != set(baseline_seeds):
        raise RuntimeError("manipulation manifest and schedule item sets differ")
    matched_schedule = [
        {
            "phase": "GATE6_3_MATCHED_RANDOM_SUPPLEMENT",
            "item_id": item_id,
            "condition": condition,
            "rollout_index": 0,
            "seed": baseline_seeds[item_id],
            "seed_regime": "MATCHED_COUPLING_SECONDARY",
        }
        for item_id in manipulation_items
        for condition in RANDOM_NAMES
    ]
    write_json(output / "MATCHED_RANDOM_SCHEDULE.json", matched_schedule)

    evaluation_manifest = load_json(GATE62 / "EVALUATION_MANIFEST.json")
    evaluation_items = [str(row["item_id"]) for row in evaluation_manifest["items"]]
    if len(evaluation_items) != 60 or len(set(evaluation_items)) != 60:
        raise RuntimeError("frozen Gate 6.2 evaluation manifest is not 60 unique items")
    historical_items = {str(row["item_id"]) for row in old_schedule}
    if historical_items.intersection(evaluation_items):
        raise RuntimeError("evaluation manifest overlaps Gate 6.2 manipulation IDs")
    evaluation_schedule = [
        {
            "phase": "GATE6_3_PRIMARY_EVALUATION",
            "item_id": item_id,
            "condition": condition,
            "rollout_index": rollout,
            "seed": evaluation_seed(item_id, condition, rollout),
            "seed_regime": "INDEPENDENT_PRIMARY",
        }
        for item_id in evaluation_items
        for condition in EVALUATION_CONDITIONS
        for rollout in (0, 1)
    ]
    write_json(output / "EVALUATION_SCHEDULE.json", evaluation_schedule)

    historical = {
        "gate6_2_journal_sha256": digest(GATE62 / "journal.jsonl"),
        "gate6_2_manipulation_manifest_sha256": digest(GATE62 / "MANIPULATION_MANIFEST.json"),
        "gate6_2_evaluation_manifest_sha256": digest(GATE62 / "EVALUATION_MANIFEST.json"),
        "gate6_2_evaluation_manifest_hash_field": evaluation_manifest["manifest_hash"],
        "historical_exclusion_digest": evaluation_manifest["historical_exclusion_digest"],
        "gate6_2_classification": "GATE6_2_NO_BEHAVIORAL_FIRST_STAGE",
        "original_artifacts_immutable": True,
    }
    write_json(output / "HISTORICAL_INPUTS.json", historical)

    source_commit = git_metadata(ROOT).get("git_commit")
    lock = {
        "schema_version": 1,
        "experiment_id": "GATE6_3_SINGLE_MEAN_SEMANTIC_EVALUATION",
        "status": "FROZEN_PRE_OUTCOME",
        "stage": "DEVELOPMENT_LOCK",
        "source_commit": source_commit,
        "parent_experiment": "GATE6_2_FIRST_STAGE_REPAIR_MEAN_BRIDGE",
        "parent_classification": "GATE6_2_NO_BEHAVIORAL_FIRST_STAGE",
        "parser": {
            "version": "external-semantic-v2",
            "module_sha256": digest(PARSER_MODULE),
            "tests_sha256": digest(PARSER_TESTS),
            "contract": (
                "one unique FINAL commitment; harmless wrappers/fences only; no substantive suffix"
            ),
        },
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "dtype": "bf16",
            "quantization": "none",
            "enable_thinking": False,
            "max_new_tokens": 4096,
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
        },
        "controller": {
            "name": "BEST_SINGLE_MEAN_PLUS",
            "location": "PROMPT_BOUNDARY",
            "layer": 27,
            "constructor": "PAIRED_MEAN_DIFFERENCE",
            "vector_sha256": meaningful_hash,
            "eta0": eta0,
            "reference_scale": reference_scale,
            "delta_norm": float(np.linalg.norm(meaningful_delta)),
            "duration": "sustained_current_token_hook_already_used_by_gate6_2_engine",
        },
        "random_bank": random_bank_record,
        "manifests": {
            "manipulation_manifest_sha256": digest(GATE62 / "MANIPULATION_MANIFEST.json"),
            "evaluation_manifest_sha256": digest(GATE62 / "EVALUATION_MANIFEST.json"),
            "historical_exclusion_digest": evaluation_manifest["historical_exclusion_digest"],
            "matched_random_schedule_sha256": digest(output / "MATCHED_RANDOM_SCHEDULE.json"),
            "evaluation_schedule_sha256": digest(output / "EVALUATION_SCHEDULE.json"),
        },
        "phases": {
            "offline_reanalysis": {"rows": 200, "gpu": False},
            "matched_random_supplement": {
                "items": 20,
                "conditions": list(RANDOM_NAMES),
                "rollouts": 1,
                "trajectories": 80,
            },
            "conditional_evaluation": {
                "items": 60,
                "conditions": list(EVALUATION_CONDITIONS),
                "rollouts": 2,
                "trajectories": 840,
                "only_if_matched_null_passes": True,
            },
        },
        "gates": {
            "reanalysis": "BEST_SINGLE_MEAN_PLUS pass under frozen Gate6.2 logic",
            "matched_null": {
                "validity_minimum": 0.85,
                "semantic_change_minimum": 0.15,
                "controller_minus_random_mean": 0.05,
                "controller_gt_random_max": True,
                "token_count_toward_textual_careful": True,
            },
            "evaluation": {
                "validity_minimum": 0.90,
                "validity_drop_max": 0.05,
                "accuracy_drop_max": 0.10,
                "D_single": 0.05,
                "D_minus_random_mean": 0.05,
                "D_gt_random_max": True,
                "G_single": 0.03,
                "C_single": 0.03,
                "G_minus_random_mean": 0.05,
                "C_minus_random_mean": 0.05,
                "G_gt_random_max": True,
                "C_gt_random_max": True,
            },
        },
        "firewall": {
            "gate6_2_original_immutable": True,
            "character_count": "NOT_RUN",
            "q2": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
        },
        "cost": {"target_usd": 0.75, "hard_stop_usd": 1.50, "maximum_new_trajectories": 920},
    }
    write_json(output / "PROTOCOL_LOCK.json", lock)
    (output / "PARSER_V2_SPEC.md").write_text(
        "# external-semantic-v2\n\n"
        "The parser recognizes exactly one complete-line `FINAL:` commitment. "
        "It permits heading/list/checkmark decoration, matching emphasis or "
        "inline-code wrappers, and one matching closing Markdown fence. It "
        "rejects substantive suffix text, multiple commitments, missing or "
        "empty commitments, arbitrary reasoning extraction, and truncation. "
        "Reference-type-aware evaluation remains the canonical deterministic "
        "Python-literal evaluator.\n",
        encoding="utf-8",
    )
    write_json(
        output / "PARSER_V2_TESTS.json",
        {
            "test_file": str(PARSER_TESTS.relative_to(ROOT)),
            "tests_sha256": digest(PARSER_TESTS),
            "pytest_target": "tests/test_external_semantic_v2.py",
        },
    )
    lock_md = [
        "# Gate 6.3 Protocol Lock — Single-Mean Semantic Evaluation",
        "",
        "Status: `FROZEN_PRE_OUTCOME`.",
        "",
        "This lock preserves Gate 6.2 as `GATE6_2_NO_BEHAVIORAL_FIRST_STAGE` and",
        "authorizes only the deterministic parser-V2 diagnostic, the matched",
        "single-layer L27 random supplement, and a conditional evaluation of the",
        "promotable single controller. No Gate 6.2 row is rerun.",
        "",
        f"Parser module SHA-256: `{lock['parser']['module_sha256']}`.",
        f"Frozen meaningful L27 vector SHA-256: `{meaningful_hash}`.",
        f"Frozen Gate 6.2 eta0: `{eta0:.15g}`; reference scale: `{reference_scale:.15g}`.",
        "Gate 6.2 evaluation manifest SHA-256: "
        f"`{lock['manifests']['evaluation_manifest_sha256']}`.",
        "",
        "The 80-row random supplement is collected only if the offline V2",
        "reanalysis passes. The 840-row evaluation is collected only if the",
        "single-layer controller exceeds the matched random bank.",
        "",
        "Scientific firewall: character count NOT RUN; Q2 NOT RUN; confirmatory",
        "holdout UNTOUCHED.",
    ]
    (output / "PROTOCOL_LOCK.md").write_text("\n".join(lock_md) + "\n", encoding="utf-8")
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "review" / "gate6_3_single_mean_semantic_evaluation"
    )
    args = parser.parse_args()
    lock = prepare(args.output.resolve())
    print(
        json.dumps(
            {
                "status": lock["status"],
                "source_commit": lock["source_commit"],
                "random_conditions": list(RANDOM_NAMES),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
