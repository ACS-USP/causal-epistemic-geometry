#!/usr/bin/env python3
"""Freeze Gate-5 choices after model-free manifest preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from epistemic_geometry.experiments.gate5 import SYSTEM_CAREFUL, SYSTEM_DIRECT

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review" / "gate5_source_duration"
GATE4 = ROOT / "review" / "micro_q1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str) -> dict[str, Any]:
    path = REVIEW / name
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REVIEW)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = read("SOURCE_CHECK.json")
    manipulation = read("SUSTAINED_MANIPULATION.json")
    evaluation = read("SUSTAINED_EVALUATION.json")
    exclusion = read("HISTORICAL_EXCLUSION_DIGEST.json")
    random_bank = read("RANDOM_BANK_METADATA.json")
    allocations = {
        "SOURCE_CHECK": source,
        "SUSTAINED_MANIPULATION": manipulation,
        "SUSTAINED_EVALUATION": evaluation,
    }
    counts = {name: len(payload["items"]) for name, payload in allocations.items()}
    ids = {
        name: [str(item["item_id"]) for item in payload["items"]]
        for name, payload in allocations.items()
    }
    all_ids = [item_id for group in ids.values() for item_id in group]
    if counts != {"SOURCE_CHECK": 40, "SUSTAINED_MANIPULATION": 20, "SUSTAINED_EVALUATION": 60}:
        raise RuntimeError(f"Gate 5 allocation counts are not frozen 40/20/60: {counts}")
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Gate 5 allocations overlap")
    if set(all_ids) & set(exclusion["historical_ids"]):
        raise RuntimeError("Gate 5 allocation overlaps historical IDs")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    lock = {
        "schema_version": 1,
        "experiment": "GATE5_SOURCE_DURATION_BRIDGE",
        "status": "FROZEN_PRE_OUTCOME",
        "source_commit": commit,
        "model": {
            "id": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "dtype": "bf16",
            "quantization": "none",
        },
        "policy": {
            "enable_thinking": False,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 4096,
            "attention": "sdpa",
            "engine": "hf_generate_serial_prefill_hook",
        },
        "instrument": {
            "benchmark": "CRUXEval",
            "dataset_repo": "cruxeval-org/cruxeval",
            "dataset_revision": "b96af0450242eb4da433032b90998f25588a5d0f",
            "evaluator": "corrected_deterministic_type_aware_python_literal",
            "parser_version": "external-semantic-v1",
        },
        "gate4_controllers": {
            "direction_sha256": digest(GATE4 / "DIRECTION.npy"),
            "direction_vector_hash": random_bank["meaningful_direction_sha256"],
            "random_r0_vector_hash": random_bank["gate4_random_direction_sha256"],
            "alpha": 8.39900588973121,
            "layer": 17,
            "duration_one_shot": "prefill_final_prompt_token_only",
        },
        "random_bank": random_bank,
        "fresh_splits": {
            "historical_exclusion_digest": exclusion["historical_exclusion_digest"],
            "historical_excluded_count": len(exclusion["historical_ids"]),
            "groups": counts,
            "ids": ids,
            "manifest_sha256": {name: digest(output / f"{name}.json") for name in allocations},
            "all_ids_digest": digest(output / "ALLOCATION_SUMMARY.json"),
        },
        "source_check": {
            "conditions": ["ORDINARY", "CAREFUL", "DIRECT"],
            "rollouts_per_item_condition": 2,
            "seed_regime": "INDEPENDENT_PRIMARY",
            "careful_system_prompt": SYSTEM_CAREFUL,
            "direct_system_prompt": SYSTEM_DIRECT,
            "thresholds": {
                "validity": 0.90,
                "cross_disagreement_X": 0.10,
                "excess_disagreement_S": 0.05,
                "style_token_ratio": 1.25,
                "style_median_additional_tokens": 2,
            },
        },
        "sustained_semantics": {
            "prefill": "final_non_padding_prompt_token_only",
            "decode": "current_token_only_each_forward",
            "past_kv_modified": False,
            "historical_positions_modified": False,
            "one_application_per_forward": True,
            "one_shot_conditions": ["ONE_SHOT_PLUS", "ONE_SHOT_MINUS"],
            "sustained_conditions": [
                "SUSTAINED_PLUS",
                "SUSTAINED_MINUS",
                "SUSTAINED_RANDOM_R0",
                "SUSTAINED_RANDOM_R1",
                "SUSTAINED_RANDOM_R2",
                "SUSTAINED_RANDOM_R3",
            ],
        },
        "manipulation_gate": {
            "regime": "MATCHED_COUPLING_SECONDARY",
            "conditions": [
                "BASELINE",
                "ONE_SHOT_PLUS",
                "ONE_SHOT_MINUS",
                "SUSTAINED_PLUS",
                "SUSTAINED_MINUS",
                "SUSTAINED_RANDOM_R0",
                "SUSTAINED_RANDOM_R1",
                "SUSTAINED_RANDOM_R2",
                "SUSTAINED_RANDOM_R3",
            ],
            "rollouts_per_item_condition": 1,
            "thresholds": {
                "validity": 0.85,
                "semantic_change_rate": 0.15,
                "duration_contrast": 0.05,
                "versus_random_mean": 0.05,
            },
        },
        "primary_evaluation": {
            "regime": "INDEPENDENT_PRIMARY",
            "rollouts_per_item_condition": 2,
            "conditions": [
                "BASELINE",
                "ONE_SHOT_PLUS",
                "ONE_SHOT_MINUS",
                "SUSTAINED_PLUS",
                "SUSTAINED_MINUS",
                "SUSTAINED_RANDOM_R0",
                "SUSTAINED_RANDOM_R1",
                "SUSTAINED_RANDOM_R2",
                "SUSTAINED_RANDOM_R3",
            ],
            "guards": {
                "validity_min": 0.90,
                "validity_drop_max": 0.05,
                "competence_drop_max": 0.10,
            },
            "bootstrap": {"unit": "item_cluster", "resamples": 5000, "seed": 20260820},
            "thresholds": {
                "movement_D": 0.05,
                "movement_D_delta_random_mean": 0.05,
                "movement_duration_D": 0.03,
                "useful_G": 0.03,
                "useful_C": 0.03,
                "useful_C_delta_random_mean": 0.05,
            },
        },
        "cost_gate": {"max_trajectories": 1500, "target_usd": 0.75, "hard_stop_usd": 1.50},
        "firewall": {
            "gate4": "ACCEPTED_BOUNDED_NULL",
            "q2": "NOT_RUN",
            "character_count": "NOT_RUN",
            "confirmatory_holdout": "UNTOUCHED",
        },
    }
    (output / "PROTOCOL_LOCK.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "PROTOCOL_LOCK.md").write_text(
        "# Gate 5 — Source Validity and Temporal Persistence\n\n"
        "This is a pre-outcome development lock. It reuses the Gate-4 Qwen3-8B "
        "direction, alpha, layer, and CRUXEval substrate; it adds only the frozen "
        "source-check, sustained-current-token semantics, and R0–R3 random bank.\n\n"
        f"- Source commit before model outcomes: `{commit}`\n"
        f"- Fresh items: 40 source-check + 20 manipulation + 60 evaluation\n"
        f"- Historical exclusion digest: `{exclusion['historical_exclusion_digest']}`\n"
        "- Model: Qwen/Qwen3-8B, BF16, full non-thinking, sampled\n"
        "- No Q2, RFM/AGOP, multilayer steering, character count, or holdout access\n\n"
        "All machine-readable thresholds, conditions, IDs, prompt text, controller "
        "hashes, seed regimes, and estimands are frozen in `PROTOCOL_LOCK.json`.\n",
        encoding="utf-8",
    )
    spec = {
        "schema_version": 1,
        "id": "GATE5_SOURCE_DURATION_BRIDGE",
        "status": "PROSPECTIVE_FROZEN_DEVELOPMENT_LOCK",
        "stage": "DEVELOPMENT_LOCK",
        "source_commit": commit,
        "model": lock["model"],
        "policy": lock["policy"],
        "instrument": lock["instrument"],
        "layer": 17,
        "alpha": 8.39900588973121,
        "random_bank_metadata": "review/gate5_source_duration/RANDOM_BANK_METADATA.json",
        "protocol_lock": "review/gate5_source_duration/PROTOCOL_LOCK.json",
        "fresh_split_counts": counts,
        "cost_gate": lock["cost_gate"],
        "firewall": lock["firewall"],
    }
    (ROOT / "experiments" / "specs" / "gate5_source_duration.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    print(json.dumps({"source_commit": commit, "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
