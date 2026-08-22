#!/usr/bin/env python3
"""Freeze Gate 11 selections, random bank, schedules, and prospective lock."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate11  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

REVIEW = ROOT / "review/gate11_domain_conditioned_control"
GATE9 = ROOT / "review/gate9_selected_d75_evaluation"
GATE10 = ROOT / "review/gate10_cross_domain_charcount"
VECTOR_PATH = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def selection(domain: str, gate: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(gate / "EVALUATION_MANIFEST.json")
    items = {str(item["item_id"]): item for item in manifest["items"]}
    source_ids, propagation_ids = gate11.select_items(domain, list(items))
    payload = {
        "domain": domain,
        "selection_rule": f'SHA256("GATE11-SOURCE-AXIS|{domain}|" + item_id)',
        "source_axis_count": len(source_ids),
        "propagation_count": len(propagation_ids),
        "source_axis_item_ids": source_ids,
        "propagation_item_ids": propagation_ids,
        "outcome_fields_used": [],
        "source_manifest_sha256": sha256(gate / "EVALUATION_MANIFEST.json"),
        "items": [items[item_id] for item_id in source_ids],
    }
    return payload, [items[item_id] for item_id in propagation_ids]


def baseline_sequences(
    gate: Path, propagation_items: list[dict[str, Any]], domain: str
) -> list[dict[str, Any]]:
    rows = read_jsonl(gate / "journal.jsonl")
    baseline = {
        (str(row["item_id"]), int(row["rollout_index"])): row
        for row in rows
        if row["condition"] == "BASELINE"
    }
    result = []
    for item in propagation_items:
        selected = gate11.choose_baseline_sequence(baseline, str(item["item_id"]))
        result.append(
            {
                "domain": domain,
                "item_id": item["item_id"],
                "prompt": item["prompt"],
                "prompt_hash": item["prompt_hash"],
                "source_journal_sha256": sha256(gate / "journal.jsonl"),
                **selected,
            }
        )
    return result


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    required = [
        GATE9 / "journal.jsonl",
        GATE10 / "journal.jsonl",
        GATE9 / "EVALUATION_MANIFEST.json",
        GATE10 / "EVALUATION_MANIFEST.json",
        VECTOR_PATH,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Gate 11 required historical artifacts missing: {missing}")

    meaningful = np.load(VECTOR_PATH, allow_pickle=False).astype(np.float64).reshape(-1)
    if vector_sha256(meaningful) != gate11.CONTROLLER_HASH:
        raise RuntimeError("Gate 11 fixed controller hash mismatch")
    randoms, random_metadata = gate11.random_bank(meaningful)
    for name, vector in randoms.items():
        path = REVIEW / f"{name}.npy"
        np.save(path, vector.astype(np.float64), allow_pickle=False)
        random_metadata["records"][name]["vector_path"] = str(path.relative_to(ROOT))
        random_metadata["records"][name]["file_sha256"] = sha256(path)
    random_metadata.update(
        {
            "shared_across_domains": True,
            "layer": gate11.LAYER,
            "eta": gate11.ETA,
            "reference_scale": gate11.REFERENCE_SCALE,
            "duration": "sustained_current_token",
        }
    )
    write_json(REVIEW / "RANDOM_BANK.json", random_metadata)

    crux_selection, crux_propagation = selection("CRUXEval", GATE9)
    char_selection, char_propagation = selection("CHARCOUNT", GATE10)
    write_json(REVIEW / "CRUX_ITEM_SELECTION.json", crux_selection)
    write_json(REVIEW / "CHARCOUNT_ITEM_SELECTION.json", char_selection)

    prompts = {
        "P0_ORDINARY": {"system_prompt": None},
        "P1_SOURCE_CAREFUL": {"system_prompt": gate11.SYSTEM_CAREFUL},
        "P2_SOURCE_DIRECT": {"system_prompt": gate11.SYSTEM_DIRECT},
        "P3_DOMAIN_TEXTUAL_CAREFUL": {
            "CRUXEval": gate11.SYSTEM_CAREFUL,
            "CHARCOUNT": gate11.SYSTEM_CHARCOUNT_CAREFUL,
        },
        "physical_identity": {
            "CRUXEval:P1_SOURCE_CAREFUL=P3_DOMAIN_TEXTUAL_CAREFUL": True,
            "CHARCOUNT:P1_SOURCE_CAREFUL=P3_DOMAIN_TEXTUAL_CAREFUL": False,
        },
    }
    prompts["hash"] = stable_digest("GATE11-PROMPT-VARIANTS", canonical_json(prompts))
    write_json(REVIEW / "PROMPT_VARIANTS.json", prompts)

    sequences = baseline_sequences(GATE9, crux_propagation, "CRUXEval")
    sequences += baseline_sequences(GATE10, char_propagation, "CHARCOUNT")
    schedule = []
    for sequence in sequences:
        for condition in gate11.TF_CONDITIONS:
            schedule.append(
                {
                    **sequence,
                    "condition": condition,
                    "logical_key": f"{sequence['domain']}|{sequence['item_id']}|{condition}",
                    "sampling": False,
                    "same_sequence_group": f"{sequence['domain']}|{sequence['item_id']}",
                }
            )
    keys = [row["logical_key"] for row in schedule]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Gate 11 fixed-sequence schedule has duplicate keys")
    fixed_schedule = {
        "schema_version": 1,
        "conditions": list(gate11.TF_CONDITIONS),
        "sequence_cap": gate11.SEQUENCE_CAP,
        "checkpoints": list(gate11.CHECKPOINTS),
        "captured_layers": list(gate11.PROPAGATION_LAYERS),
        "logical_rows": len(schedule),
        "available_item_sequences": sum(row["available"] for row in sequences),
        "missing_item_sequences": sum(not row["available"] for row in sequences),
        "rows": schedule,
        "schedule_hash": stable_digest("GATE11-FIXED-SEQUENCE", canonical_json(schedule)),
    }
    write_json(REVIEW / "FIXED_SEQUENCE_SCHEDULE.json", fixed_schedule)

    source_schedule = []
    for domain, payload in (("CRUXEval", crux_selection), ("CHARCOUNT", char_selection)):
        for item_id in payload["source_axis_item_ids"]:
            for variant in gate11.PROMPT_VARIANTS:
                alias = domain == "CRUXEval" and variant == "P3_DOMAIN_TEXTUAL_CAREFUL"
                source_schedule.append(
                    {
                        "domain": domain,
                        "item_id": item_id,
                        "variant": variant,
                        "physical_variant": "P1_SOURCE_CAREFUL" if alias else variant,
                        "physical_forward_required": not alias,
                    }
                )
    write_json(
        REVIEW / "SOURCE_ACTIVATION_SCHEDULE.json",
        {
            "rows": source_schedule,
            "logical_rows": len(source_schedule),
            "physical_forwards": sum(row["physical_forward_required"] for row in source_schedule),
            "layers": list(gate11.SOURCE_LAYERS),
            "schedule_hash": stable_digest(
                "GATE11-SOURCE-SCHEDULE", canonical_json(source_schedule)
            ),
        },
    )

    source_hashes = {
        "gate9_protocol": sha256(GATE9 / "PROTOCOL_LOCK.json"),
        "gate9_manifest": sha256(GATE9 / "EVALUATION_MANIFEST.json"),
        "gate9_journal": sha256(GATE9 / "journal.jsonl"),
        "gate10_protocol": sha256(GATE10 / "PROTOCOL_LOCK.json"),
        "gate10_manifest": sha256(GATE10 / "EVALUATION_MANIFEST.json"),
        "gate10_journal": sha256(GATE10 / "journal.jsonl"),
        "controller_file": sha256(VECTOR_PATH),
    }
    projected_token_forwards = sum(
        int(row["continuation_length"]) + 1 for row in sequences if row["available"]
    ) * len(gate11.TF_CONDITIONS)
    projected_hours = (projected_token_forwards * 0.035 + 280 * 0.35 + 180) / 3600
    lock = {
        "schema_version": 1,
        "experiment_id": gate11.EXPERIMENT_ID,
        "status": "FROZEN_PRE_DIAGNOSTIC",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_MECHANISTIC_POSTMORTEM",
        "model": {
            "id": gate11.MODEL,
            "revision": gate11.MODEL_REVISION,
            "tokenizer_revision": gate11.MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "attention": "sdpa",
            "enable_thinking": False,
            "environment": "CORE_QWEN",
        },
        "historical_sources": source_hashes,
        "selection": {
            "rule": "SHA256(GATE11-SOURCE-AXIS|domain|item_id)",
            "source_items_per_domain": 40,
            "propagation_items_per_domain": 24,
            "outcome_fields_used": [],
        },
        "prompt_variants_sha256": sha256(REVIEW / "PROMPT_VARIANTS.json"),
        "source_layers": list(gate11.SOURCE_LAYERS),
        "propagation_layers": list(gate11.PROPAGATION_LAYERS),
        "controller": {
            "canonical_hash": gate11.CONTROLLER_HASH,
            "file_sha256": sha256(VECTOR_PATH),
            "vector_path": str(VECTOR_PATH.relative_to(ROOT)),
            "layer": gate11.LAYER,
            "eta": gate11.ETA,
            "reference_scale": gate11.REFERENCE_SCALE,
            "delta_norm": gate11.ETA * gate11.REFERENCE_SCALE,
            "duration": "sustained_current_token",
            "analysis_only_domain_directions": True,
        },
        "random_bank_sha256": sha256(REVIEW / "RANDOM_BANK.json"),
        "teacher_forcing": {
            "sampling": False,
            "sequence_cap": gate11.SEQUENCE_CAP,
            "rollout_fallback": [0, 1, "missing_no_replacement"],
            "checkpoints": list(gate11.CHECKPOINTS),
            "conditions": list(gate11.TF_CONDITIONS),
            "logical_rows": len(schedule),
            "same_sequence_within_item": True,
        },
        "metrics": [
            "next_token_kl",
            "symmetric_js",
            "logit_l2",
            "logit_cosine",
            "top1_flip",
            "target_logprob_shift",
            "downstream_hidden_amplification",
            "careful_logit_alignment",
            "random_specificity",
            "historical_policy_utility",
        ],
        "bootstrap": {
            "resamples": gate11.BOOTSTRAP_RESAMPLES,
            "seed": gate11.BOOTSTRAP_SEED,
            "unit": "item_with_all_conditions_and_checkpoints",
            "cross_domain": "independent_within_domain_resampling_per_replicate",
        },
        "component_rules": {
            "source_transfer": "gap>0; bootstrap lower>0; positive>=30/40; cosine>=0.20",
            "control_gain": "CRUX specific KL>0 and 2/3 positive domain-contrast lower bounds",
            "policy_realization": "CRUX alignment>0 and domain-contrast lower bound>0",
            "policy_utility": "both historical accuracy-effect contrast lower bounds>0",
        },
        "classifications": [
            "GATE11_SOURCE_AXIS_DOMAIN_MISMATCH",
            "GATE11_DOWNSTREAM_CONTROL_GAIN_DOMAIN_MISMATCH",
            "GATE11_POLICY_REALIZATION_DOMAIN_MISMATCH",
            "GATE11_POLICY_UTILITY_DOMAIN_MISMATCH",
            "GATE11_MULTIPLE_DOMAIN_CONDITIONING_FACTORS",
            "GATE11_POSTMORTEM_INCONCLUSIVE",
            "GATE11_ENGINE_FAILURE",
        ],
        "cost": {
            "projected_teacher_forced_forwards": projected_token_forwards,
            "projected_hours": projected_hours,
            "projected_usd_at_0_44": projected_hours * 0.44,
            "target_usd": 1.0,
            "hard_ceiling_usd": 2.5,
        },
        "firewall": {
            "free_generation": "NOT_AUTHORIZED",
            "new_semantic_evaluation": "NOT_RUN",
            "new_controller": "NO",
            "new_dose": "NO",
            "q2": "NOT_RUN",
            "holdout": "UNTOUCHED",
        },
        "experiment_source_commit_binding": "EXPERIMENT_SOURCE_COMMIT.json",
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Gate 11 prospective protocol lock\n\n"
        "Status: `FROZEN_PRE_DIAGNOSTIC`. No free generation is authorized.\n\n"
        "The lock freezes deterministic existing-item selection, exact historical prompts, "
        "the fixed L27-D75 controller, one shared random bank, sequential teacher forcing, "
        "checkpoints, metrics, bootstrap, component rules, and exhaustive synthesis before "
        "any activation or logit diagnostic exists.\n",
        encoding="utf-8",
    )
    spec = {
        "experiment_id": gate11.EXPERIMENT_ID,
        "status": "FROZEN_PROSPECTIVE",
        "stage": "DEVELOPMENT_MECHANISTIC_POSTMORTEM",
        "free_generation": False,
        "domains": ["CRUXEval", "FRESH_PSEUDOWORD_LONG"],
        "source_items_per_domain": 40,
        "propagation_items_per_domain": 24,
        "source_layers": list(gate11.SOURCE_LAYERS),
        "controller": "fixed paired-mean L27 plus D75",
        "teacher_forcing_conditions": list(gate11.TF_CONDITIONS),
        "sequence_cap": gate11.SEQUENCE_CAP,
        "bootstrap_resamples": gate11.BOOTSTRAP_RESAMPLES,
        "hard_cost_ceiling_usd": 2.5,
        "q2": "NOT_RUN",
        "holdout": "UNTOUCHED",
    }
    spec_path = ROOT / "experiments/specs/gate11_domain_conditioned_control.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")

    preoutcome_files = [
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "CRUX_ITEM_SELECTION.json",
        "CHARCOUNT_ITEM_SELECTION.json",
        "PROMPT_VARIANTS.json",
        "RANDOM_BANK.json",
        "FIXED_SEQUENCE_SCHEDULE.json",
        "SOURCE_ACTIVATION_SCHEDULE.json",
    ]
    write_json(
        REVIEW / "artifact_hashes_preoutcome.json",
        {name: sha256(REVIEW / name) for name in preoutcome_files},
    )
    print(
        json.dumps(
            {
                "classification": "PREMORTEM_PASS",
                "source_items": 80,
                "propagation_items": len(sequences),
                "logical_teacher_forcing_rows": len(schedule),
                "projected_cost_usd": lock["cost"]["projected_usd_at_0_44"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
