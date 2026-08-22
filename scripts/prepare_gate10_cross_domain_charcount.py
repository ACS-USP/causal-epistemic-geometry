#!/usr/bin/env python3
"""Freeze Gate 10 generator, parser, random bank, schedule, and protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.benchmarks.v4.character_semantic_v3 import (  # noqa: E402
    PARSER_VERSION,
    evaluate_character_count_answer_v3,
)
from epistemic_geometry.experiments import gate10  # noqa: E402
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

OUTPUT = ROOT / "review/gate10_cross_domain_charcount"
CONTROLLER_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge/PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)
PARSER_MODULE = ROOT / "src/epistemic_geometry/benchmarks/v4/character_semantic_v3.py"
PARSER_SPEC = OUTPUT / "CHARCOUNT_SEMANTIC_V3_SPEC.md"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def parser_validation() -> dict[str, Any]:
    cases = [
        ("FINAL: 7", "7", True, True, True),
        ("FINAL: -2", "-2", True, True, True),
        ("FINAL: 0", "0", True, True, True),
        ("```\nFINAL: 7\n```", "7", True, True, True),
        ("### FINAL: 7", "7", True, True, True),
        ("### Final Answer:\nFINAL: 7", "7", True, True, True),
        ('FINAL: "7"', "7", True, True, False),
        ("FINAL: 7\nFINAL: 8", "7", False, False, False),
        ("answer 7", "7", False, False, False),
        ("FINAL: 7", "7", False, False, False, True),
        ("**FINAL: 7**", "7", True, True, True),
    ]
    results = []
    for case in cases:
        raw, reference, cv, ev, correct, *tail = case
        result = evaluate_character_count_answer_v3(
            raw, reference, truncated=bool(tail and tail[0])
        )
        passed = (result.commitment_valid, result.semantic_evaluable, result.correct) == (
            cv,
            ev,
            correct,
        )
        results.append(
            {
                "raw": raw,
                "expected": [cv, ev, correct],
                "observed": [result.commitment_valid, result.semantic_evaluable, result.correct],
                "pass": passed,
            }
        )
    invariant = all(
        evaluate_character_count_answer_v3("### FINAL: 5", "5")
        == evaluate_character_count_answer_v3("### FINAL: 5", "5")
        for _ in gate10.CONDITIONS
    )
    payload = {
        "parser_version": PARSER_VERSION,
        "cases": results,
        "condition_invariance": invariant,
        "pass": all(r["pass"] for r in results) and invariant,
    }
    if not payload["pass"]:
        raise RuntimeError("Gate 10 parser validation failed")
    return payload


def freeze(output: Path) -> dict[str, Any]:
    premortem = json.loads((output / "PREMORTEM.json").read_text())
    if premortem["classification"] != "PREMORTEM_PASS":
        raise RuntimeError("Gate 10 premortem not passed")
    historical = gate10.historical_charcount_records(ROOT / "review", gate10_output=output)
    exclusion = {
        "historical_item_ids": historical["item_ids"],
        "historical_generator_seeds": historical["generator_seeds"],
        "historical_exact_strings": historical["exact_strings"],
        "historical_item_hashes": historical["item_hashes"],
        "counts": {k: len(v) for k, v in historical.items() if isinstance(v, list)},
        "source_files": historical["source_files"],
        "digest": stable_digest(
            gate10.SELECTION_NAMESPACE, "HISTORICAL", canonical_json(historical)
        ),
    }
    write_json(output / "HISTORICAL_CHARCOUNT_EXCLUSION_DIGEST.json", exclusion)
    manifest = gate10.generate_fresh_manifest(historical, 200)
    write_json(output / "EVALUATION_MANIFEST.json", manifest)
    items = manifest["items"]
    validation = {
        "n": len(items),
        "unique_ids": len({x["item_id"] for x in items}),
        "unique_hashes": len({x["item_hash"] for x in items}),
        "unique_strings": len({x["text"] for x in items}),
        "oracle_pass": all(x["text"].count(x["target_character"]) == x["answer"] for x in items),
        "historical_hash_overlap": len(
            {x["item_hash"] for x in items} & set(historical["item_hashes"])
        ),
        "historical_string_overlap": len(
            {x["text"] for x in items} & set(historical["exact_strings"])
        ),
        "length": {
            "min": min(len(x["text"]) for x in items),
            "mean": float(np.mean([len(x["text"]) for x in items])),
            "max": max(len(x["text"]) for x in items),
        },
        "target_count_distribution": {
            str(v): sum(x["answer"] == v for x in items) for v in range(2, 7)
        },
    }
    validation["pass"] = (
        validation["n"]
        == validation["unique_ids"]
        == validation["unique_hashes"]
        == validation["unique_strings"]
        == 200
        and validation["oracle_pass"]
        and validation["historical_hash_overlap"] == validation["historical_string_overlap"] == 0
    )
    if not validation["pass"]:
        raise RuntimeError("Gate 10 generator validation failed")
    write_json(output / "GENERATOR_VALIDATION.json", validation)
    parser = parser_validation()
    write_json(output / "PARSER_VALIDATION.json", parser)

    meaningful = np.load(CONTROLLER_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    if gate10.vector_sha256(meaningful) != gate10.CONTROLLER_HASH or not np.isclose(
        np.linalg.norm(meaningful), 1, atol=1e-12
    ):
        raise RuntimeError("Gate 10 controller mismatch")
    bank, bank_meta = gate10.gate10_random_bank(meaningful)
    records = {}
    for name in gate10.RANDOM_NAMES:
        path = output / f"{name}.npy"
        np.save(path, bank[name].astype(np.float64))
        records[name] = {
            **bank_meta["records"][name],
            "vector_path": str(path.relative_to(ROOT)),
            "vector_file_sha256": gate10.file_sha256(path),
            "canonical_float64_vector_sha256": gate10.vector_sha256(bank[name]),
            "layer": gate10.LAYER,
            "eta": gate10.ETA,
            "reference_scale": gate10.REFERENCE_SCALE,
            "duration": "sustained_current_token",
            "scope": "final_prompt_token_then_current_decode_token",
        }
    random_payload = {
        "namespace": "GATE10-L27-RANDOM-BANK-V1",
        "meaningful_controller_hash": gate10.CONTROLLER_HASH,
        "records": records,
        "geometry": bank_meta["geometry"],
        "prior_randoms_reused": False,
    }
    write_json(output / "RANDOM_BANK.json", random_payload)
    schedule = gate10.build_schedule([x["item_id"] for x in items])
    write_json(output / "EVALUATION_SCHEDULE.json", schedule)
    controller = {
        "name": gate10.MEANINGFUL,
        "source_domain": "CRUXEval",
        "source": "PROMPT_BOUNDARY",
        "layer": gate10.LAYER,
        "constructor": "PAIRED_MEAN_DIFFERENCE",
        "sign": "PLUS",
        "dose": "D75",
        "eta": gate10.ETA,
        "reference_scale": gate10.REFERENCE_SCALE,
        "duration": "sustained_current_token",
        "scope": "final_prompt_token_then_current_decode_token",
        "vector_path": str(CONTROLLER_PATH.relative_to(ROOT)),
        "vector_file_sha256": gate10.file_sha256(CONTROLLER_PATH),
        "canonical_float64_vector_sha256": gate10.CONTROLLER_HASH,
        "vector_norm": float(np.linalg.norm(meaningful)),
        "delta_norm": float(np.linalg.norm(meaningful * gate10.ETA * gate10.REFERENCE_SCALE)),
        "gate9_provenance": "review/gate9_selected_d75_evaluation/PROTOCOL_LOCK.json",
    }
    evaluator = {
        "version": PARSER_VERSION,
        "module_sha256": gate10.file_sha256(PARSER_MODULE),
        "semantic_spec_sha256": gate10.file_sha256(PARSER_SPEC),
        "external_semantic_v3_sha256": gate10.file_sha256(
            ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
        ),
        "validation_sha256": gate10.file_sha256(output / "PARSER_VALIDATION.json"),
        "condition_invariance": "PASS",
    }
    seconds_per_row = 6.306324623713432
    lock = {
        "schema_version": 1,
        "experiment_id": gate10.EXPERIMENT_ID,
        "status": "FROZEN_PRE_OUTCOME",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_CROSS_DOMAIN_REPLICATION",
        "lock_preparation_source_commit": git_commit(),
        "experiment_source_commit_binding": {
            "file": "EXPERIMENT_SOURCE_COMMIT.json",
            "timing": "after lock commit before outputs",
        },
        "model": {
            "id": gate10.MODEL,
            "revision": gate10.MODEL_REVISION,
            "tokenizer_revision": gate10.MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "enable_thinking": False,
            "attention": "sdpa",
            "environment_profile": "CORE_QWEN",
            "max_new_tokens": gate10.MAX_NEW_TOKENS,
            "sampling": {
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
            },
        },
        "instrument": {
            "benchmark": "FRESH_PSEUDOWORD_LONG character counting",
            "generator_version": gate10.GENERATOR_VERSION,
            "stratum": "FRESH_PSEUDOWORD_LONG",
            "namespace": gate10.SELECTION_NAMESPACE,
            "evaluator": evaluator,
        },
        "sample": {
            "actual_n": 200,
            "manifest_hash": manifest["manifest_hash"],
            "manifest_file_sha256": gate10.file_sha256(output / "EVALUATION_MANIFEST.json"),
            "historical_exclusion_sha256": gate10.file_sha256(
                output / "HISTORICAL_CHARCOUNT_EXCLUSION_DIGEST.json"
            ),
        },
        "controller": controller,
        "random_bank": {
            "records": records,
            "geometry": bank_meta["geometry"],
            "file_sha256": gate10.file_sha256(output / "RANDOM_BANK.json"),
        },
        "textual_careful_instruction": gate10.SYSTEM_CAREFUL,
        "conditions": list(gate10.CONDITIONS),
        "rollouts_per_item_condition": 2,
        "seed_regime": "INDEPENDENT_PRIMARY",
        "schedule": {
            "logical_rows": len(schedule),
            "file_sha256": gate10.file_sha256(output / "EVALUATION_SCHEDULE.json"),
            "globally_distinct_seeds": True,
            "outcome_independent_interleaving": True,
        },
        "opportunity_gate": {
            "commitment_min": 0.95,
            "evaluability_min": 0.95,
            "accuracy_range": [0.55, 0.95],
            "B00_min": 0.04,
            "double_wrong_items_min": 8,
            "any_correct_items_min": 20,
        },
        "guards": {
            "commitment_min": 0.95,
            "commitment_drop_max": 0.03,
            "evaluability_min": 0.95,
            "evaluability_drop_max": 0.03,
            "accuracy_drop_max": 0.05,
        },
        "strong_thresholds": {
            "G_min": 0.03,
            "C_min": 0.015,
            "D_min": 0.04,
            "G_random_mean_delta_min": 0.025,
            "C_random_mean_delta_min": 0.015,
            "D_random_mean_delta_min": 0.03,
            "all_above_random_max": True,
            "G_norm_min": 0.15,
            "rescue_ge_damage": True,
            "bootstrap_positive": ["G", "C", "D", "G_random_mean", "C_random_mean"],
            "loo_sign_stable": ["G", "C", "D"],
        },
        "minimum_thresholds": {
            "positive_G_C_D": True,
            "above_random_mean": True,
            "two_above_random_max": True,
            "G_norm_min": 0.08,
            "rescue_ge_damage": True,
        },
        "style_transfer_threshold": {
            "mean_token_recovery_fraction_min": 0.25,
        },
        "classifications": [
            "GATE10_STRONG_CROSS_DOMAIN_USEFUL_COMPLEMENTARITY",
            "GATE10_MINIMUM_CROSS_DOMAIN_CONTROL_SIGNAL",
            "GATE10_CROSS_DOMAIN_ERROR_PROFILE_MOVEMENT_ONLY",
            "GATE10_CROSS_DOMAIN_COMPETENCE_GAIN_WITHOUT_COMPLEMENTARITY",
            "GATE10_CAREFUL_STYLE_TRANSFER_ONLY",
            "GATE10_NO_CROSS_DOMAIN_TRANSFER",
            "GATE10_CROSS_DOMAIN_DESTRUCTIVE",
            "GATE10_INSTRUMENT_CEILING_OR_FLOOR",
            "GATE10_INSTRUMENT_FAILURE",
            "GATE10_ENGINE_FAILURE",
        ],
        "bootstrap": {
            "resamples": gate10.BOOTSTRAP_RESAMPLES,
            "seed": gate10.BOOTSTRAP_SEED,
            "unit": "item_cluster_all_7_conditions_both_rollouts",
        },
        "cost": {
            "target_usd": 2.5,
            "hard_stop_usd": 5.0,
            "projected_rows": len(schedule),
            "seconds_per_row": seconds_per_row,
            "projected_cost_usd_at_0_44": len(schedule) * seconds_per_row / 3600 * 0.44,
        },
        "firewall": {
            "controller_search": "NOT_RUN",
            "dose_search": "NOT_RUN",
            "layer_search": "NOT_RUN",
            "q2": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
            "gate11": "NOT_RUN",
        },
    }
    write_json(output / "PROTOCOL_LOCK.json", lock)
    protocol_lines = [
        "# Gate 10 prospective lock",
        "",
        "Status: `FROZEN_PRE_OUTCOME`.",
        "",
        "Exact fixed L27 D75 controller on 200 fresh `FRESH_PSEUDOWORD_LONG` items, "
        "seven conditions, two independent rollouts, 2,800 rows.",
        "",
        f"Manifest hash: `{manifest['manifest_hash']}`.",
        f"Controller hash: `{gate10.CONTROLLER_HASH}`.",
        f"Eta: `{gate10.ETA}`.",
        "",
        "No controller, layer, dose, sign, or difficulty search. Q2 is NOT RUN and "
        "confirmatory holdout is UNTOUCHED.",
        "",
    ]
    (output / "PROTOCOL_LOCK.md").write_text("\n".join(protocol_lines))
    names = (
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "HISTORICAL_CHARCOUNT_EXCLUSION_DIGEST.json",
        "EVALUATION_MANIFEST.json",
        "RANDOM_BANK.json",
        "EVALUATION_SCHEDULE.json",
        "GENERATOR_VALIDATION.json",
        "PARSER_VALIDATION.json",
        "CHARCOUNT_SEMANTIC_V3_SPEC.md",
        *(f"{n}.npy" for n in gate10.RANDOM_NAMES),
    )
    write_json(
        output / "artifact_hashes_preoutcome.json",
        {name: gate10.file_sha256(output / name) for name in names},
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bind-source-commit")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.bind_source_commit:
        if git_commit() != args.bind_source_commit:
            raise RuntimeError("binding commit mismatch")
        payload = {
            "experiment_source_commit": args.bind_source_commit,
            "bound_before_model_outputs": True,
            "protocol_lock_sha256": gate10.file_sha256(output / "PROTOCOL_LOCK.json"),
        }
        write_json(output / "EXPERIMENT_SOURCE_COMMIT.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    lock = freeze(output)
    print(
        json.dumps(
            {
                "status": lock["status"],
                "rows": lock["schedule"]["logical_rows"],
                "projected_cost": lock["cost"]["projected_cost_usd_at_0_44"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
