#!/usr/bin/env python3
"""Prepare the outcome-blind prospective lock for Gate 12."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate12  # noqa: E402
from epistemic_geometry.reproducibility import canonical_json, stable_digest  # noqa: E402

REVIEW = ROOT / "review/gate12_utility_aligned_pullback"
GATE9 = ROOT / "review/gate9_selected_d75_evaluation"
GATE10 = ROOT / "review/gate10_cross_domain_charcount"
GATE11 = ROOT / "review/gate11_domain_conditioned_control"
GATE11_1 = ROOT / "review/gate11_1_artifact_complete_replication"
MEANINGFUL = (
    ROOT
    / "review/gate6_2_first_stage_repair_mean_bridge"
    / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_record(path: Path, canonical_hash: str, label: str, historical_condition: str) -> dict:
    return {
        "label": label,
        "historical_condition": historical_condition,
        "vector_path": str(path.relative_to(ROOT)),
        "file_sha256": sha256(path),
        "canonical_float64_sha256": canonical_hash,
        "unit_direction": True,
        "layer": gate12.LAYER,
    }


def bank_records(path: Path) -> list[dict[str, Any]]:
    bank = read_json(path)
    records = bank.get("random_vectors", bank.get("records"))
    result = []
    for index in range(4):
        source_name = f"RANDOM_L27_D75_R{index}"
        record = records[source_name]
        vector_path = ROOT / record["vector_path"]
        result.append(
            file_record(
                vector_path,
                record.get("canonical_float64_vector_sha256", record["vector_sha256"]),
                f"RANDOM_R{index}",
                source_name,
            )
        )
    return result


def gate11_records() -> list[dict[str, Any]]:
    bank = read_json(GATE11_1 / "RANDOM_BANK.json")
    result = []
    for index in range(4):
        name = f"GATE11_RANDOM_R{index}"
        record = bank["records"][name]
        result.append(
            file_record(
                ROOT / record["vector_path"],
                record["canonical_float64_vector_sha256"],
                f"RANDOM_R{index}",
                f"TF_RANDOM_R{index}",
            )
        )
    return result


def meaningful_record() -> dict[str, Any]:
    return file_record(
        MEANINGFUL,
        gate12.CONTROLLER_HASH,
        "MEANINGFUL",
        "MEANINGFUL_L27_D75",
    )


def utility_selection(
    *, domain: str, manifest_path: Path, excluded_ids: set[str]
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    by_id = {str(row["item_id"]): row for row in manifest["items"]}
    eligible = [item_id for item_id in by_id if item_id not in excluded_ids]
    ranked = gate12.rank_utility_ids(domain, eligible)
    selected = ranked[: gate12.UTILITY_ITEMS_PER_DOMAIN]
    if len(selected) != gate12.UTILITY_ITEMS_PER_DOMAIN:
        raise RuntimeError(f"insufficient outcome-blind {domain} utility items")
    items = []
    for item_id in selected:
        source = by_id[item_id]
        reference = source["reference_answer"] if domain == "CRUXEval" else source["answer"]
        items.append(
            {
                **source,
                "canonical_correct_continuation": gate12.canonical_answer(domain, reference),
                "canonical_policy": "FINAL_COLON_CANONICAL_VALUE_NO_TRAILING_NEWLINE",
            }
        )
    return {
        "domain": domain,
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "source_manifest_sha256": sha256(manifest_path),
        "selection_namespace": (
            "GATE12-UTILITY-PREDICTION|" + ("CRUX" if domain == "CRUXEval" else "CHARCOUNT")
        ),
        "historical_ids_excluded": len(excluded_ids),
        "eligible_after_exclusion": len(eligible),
        "selected_count": len(items),
        "outcome_fields_used": [],
        "items": items,
    }


def control_selection() -> dict[str, Any]:
    schedule = read_json(GATE11_1 / "FIXED_SEQUENCE_SCHEDULE.json")
    selections = {
        "CRUXEval": read_json(GATE11_1 / "CRUX_ITEM_SELECTION.json"),
        "CHARCOUNT": read_json(GATE11_1 / "CHARCOUNT_ITEM_SELECTION.json"),
    }
    item_maps = {
        domain: {str(row["item_id"]): row for row in payload["items"]}
        for domain, payload in selections.items()
    }
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in schedule["rows"]:
        grouped.setdefault((row["domain"], str(row["item_id"])), row)
    items = []
    for (domain, item_id), row in sorted(grouped.items()):
        continuation = [int(value) for value in row["continuation_token_ids"]][
            : gate12.CONTROL_SEQUENCE_CAP
        ]
        checkpoints = [
            value
            for value in gate12.CONTROL_CHECKPOINTS
            if value == -1 or value < len(continuation)
        ]
        items.append(
            {
                "domain": domain,
                "item_id": item_id,
                "item": item_maps[domain][item_id],
                "selected_rollout_index": row["selected_rollout_index"],
                "continuation_token_ids": continuation,
                "continuation_sha256": hashlib.sha256(
                    json.dumps(continuation, separators=(",", ":")).encode()
                ).hexdigest(),
                "checkpoints": checkpoints,
                "outcome_fields_used": [],
            }
        )
    counts = {domain: sum(row["domain"] == domain for row in items) for domain in selections}
    if counts != {"CRUXEval": 24, "CHARCOUNT": 24}:
        raise RuntimeError(f"Gate-11.1 propagation item mismatch: {counts}")
    return {
        "source_schedule": str((GATE11_1 / "FIXED_SEQUENCE_SCHEDULE.json").relative_to(ROOT)),
        "source_schedule_sha256": sha256(GATE11_1 / "FIXED_SEQUENCE_SCHEDULE.json"),
        "sequence_cap": gate12.CONTROL_SEQUENCE_CAP,
        "checkpoint_policy": list(gate12.CONTROL_CHECKPOINTS),
        "outcome_fields_used": [],
        "items": items,
    }


def premortem() -> tuple[str, dict[str, Any]]:
    checks = {
        "local_vs_finite": "JVP is at alpha=0; Gate-11.1 D75 KL is a finite target only",
        "trajectory_semantics": "final prompt plus every continuation input token is shifted",
        "full_sequence_equivalence": "must pass remote KV-cache equivalence before collection",
        "jvp_exactness": "forward-mode autograd JVP primary; finite differences validation only",
        "output_geometry": "categorical Fisher q and q/4 Hellinger convention both reported",
        "utility_functional": "globally canonical minimal correct FINAL continuation",
        "outcome_blindness": "runner reads frozen manifests and vectors but no historical journals",
        "direction_matching": "all meaningful/random hashes imported from Gate 9/10/11",
        "item_selection": "manifest-only SHA256 selection with Gate-11 exclusions",
        "raw_persistence": "float32 complete logits/JVPs with masks, positions and hashes",
        "claim_boundary": "one-dimensional sustained-control pullback, not full matrix/Gramian",
        "q2_firewall": "no semantic-error geometry or direction-pair matrix claim",
        "storage": "lossless per-item shards; local verified archive required before Pod deletion",
    }
    payload = {
        "classification": "PREMORTEM_PASS",
        "checks": checks,
        "class_a_recoveries": [],
        "class_b_amendments": [
            "Pre-geometry: froze SDPA math kernel because torch 2.4 flash SDPA lacks "
            "forward-AD support; mathematical JVP, model, items, directions, and tests unchanged"
        ],
        "unresolved_scientific_ambiguities": [],
    }
    markdown = "# Gate 12 adversarial premortem\n\nClassification: `PREMORTEM_PASS`\n\n"
    markdown += "\n".join(
        f"- **{key.replace('_', ' ').title()}** — {value}." for key, value in checks.items()
    )
    markdown += "\n\nThe collection path is outcome-blind and performs no free generation.\n"
    return markdown, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)

    control = control_selection()
    crux_gate11 = set(read_json(GATE11_1 / "CRUX_ITEM_SELECTION.json")["source_axis_item_ids"])
    char_gate11 = set(read_json(GATE11_1 / "CHARCOUNT_ITEM_SELECTION.json")["source_axis_item_ids"])
    utility = {
        "CRUXEval": utility_selection(
            domain="CRUXEval",
            manifest_path=GATE9 / "EVALUATION_MANIFEST.json",
            excluded_ids=crux_gate11,
        ),
        "CHARCOUNT": utility_selection(
            domain="CHARCOUNT",
            manifest_path=GATE10 / "EVALUATION_MANIFEST.json",
            excluded_ids=char_gate11,
        ),
    }
    directions = {
        "control_validation": [meaningful_record(), *gate11_records()],
        "utility_prediction": {
            "CRUXEval": [meaningful_record(), *bank_records(GATE9 / "RANDOM_BANK.json")],
            "CHARCOUNT": [meaningful_record(), *bank_records(GATE10 / "RANDOM_BANK.json")],
        },
    }

    write_json(REVIEW / "CONTROL_VALIDATION_ITEMS.json", control)
    write_json(REVIEW / "UTILITY_PREDICTION_ITEMS.json", utility)
    write_json(REVIEW / "HISTORICAL_DIRECTION_MANIFEST.json", directions)
    write_json(
        REVIEW / "CANONICAL_ANSWER_POLICY.json",
        {
            "policy": "FINAL: <canonical exact answer>",
            "terminal_newline": False,
            "reasoning_text": False,
            "cruxeval": "exact semantic-V3 reference representation without item choice",
            "charcount": "exact base-10 integer oracle",
            "assistant_prefix_chat_template": "same accepted Qwen chat rendering as Gate 9/10",
        },
    )
    write_json(
        REVIEW / "JVP_ENGINE_SPEC.json",
        {
            "primary": "torch.autograd.forward_ad scalar-alpha JVP",
            "sdpa_kernel": "math (flash disabled; exact forward AD in torch 2.4)",
            "alpha_origin": 0.0,
            "unit_direction": True,
            "historical_d75_scalar": gate12.D75_SCALAR,
            "layer": gate12.LAYER,
            "full_sequence_causal_teacher_forcing": True,
            "intervention_positions": "final prompt and every continuation input token",
            "selected_outputs": "positions predicting frozen continuation targets/checkpoints",
            "finite_difference_primary": False,
        },
    )
    write_json(
        REVIEW / "NUMERICAL_VALIDATION_PLAN.json",
        {
            "finite_difference_scalars": [
                gate12.D75_SCALAR / divisor for divisor in gate12.FINITE_DIFFERENCE_DIVISORS
            ],
            "minimum_median_jvp_cosine": 0.995,
            "maximum_median_relative_q_difference": 0.10,
            "maximum_median_relative_u_difference": 0.10,
            "local_kl_identity": "2*KL(p0||p_epsilon)/epsilon^2 -> q",
            "bf16_equivalence_tolerances": {"logit_atol": 0.25, "kl_atol": 0.01},
        },
    )
    schedule = []
    for item in control["items"]:
        for direction in directions["control_validation"]:
            schedule.append(
                {
                    "component": "CONTROL_VALIDATION",
                    "domain": item["domain"],
                    "item_id": item["item_id"],
                    "direction": direction["label"],
                    "logical_key": (
                        f"CONTROL|{item['domain']}|{item['item_id']}|{direction['label']}"
                    ),
                }
            )
    for domain, payload in utility.items():
        for item in payload["items"]:
            for direction in directions["utility_prediction"][domain]:
                schedule.append(
                    {
                        "component": "UTILITY_PREDICTION",
                        "domain": domain,
                        "item_id": item["item_id"],
                        "direction": direction["label"],
                        "logical_key": f"UTILITY|{domain}|{item['item_id']}|{direction['label']}",
                    }
                )
    if len({row["logical_key"] for row in schedule}) != len(schedule):
        raise RuntimeError("duplicate Gate-12 geometry logical keys")
    write_json(
        REVIEW / "GEOMETRY_SCHEDULE.json",
        {
            "logical_rows": len(schedule),
            "control_rows": 48 * 5,
            "utility_rows": 64 * 5,
            "outcome_fields_used": [],
            "schedule_hash": stable_digest("GATE12-GEOMETRY", canonical_json(schedule)),
            "rows": schedule,
        },
    )
    cost_projection = {
        "basis": (
            "560 exact directional JVP rows plus baseline/careful reuse, scaled from "
            "the audited Gate-11.1 A40 fixed-sequence runtime"
        ),
        "projected_a40_hours": 3.50,
        "a40_hourly_rate_usd": 0.44,
        "projected_gpu_usd": 1.54,
        "target_usd": 1.75,
        "hard_ceiling_usd": 3.50,
        "cost_gate": "PASS",
    }
    write_json(REVIEW / "COST_PROJECTION.json", cost_projection)
    premortem_md, premortem_json = premortem()
    (REVIEW / "PREMORTEM.md").write_text(premortem_md, encoding="utf-8")
    write_json(REVIEW / "PREMORTEM.json", premortem_json)

    lock = {
        "schema_version": 1,
        "experiment_id": gate12.EXPERIMENT_ID,
        "status": "FROZEN_PRE_GEOMETRY",
        "lifecycle": "PROSPECTIVE_LOCK",
        "stage": "DEVELOPMENT_GEOMETRY",
        "experiment_source_commit": args.source_commit,
        "model": {
            "id": gate12.MODEL,
            "revision": gate12.MODEL_REVISION,
            "tokenizer_revision": gate12.MODEL_REVISION,
            "dtype": "bf16",
            "quantization": "none",
            "attention": "sdpa",
            "environment": "CORE_QWEN",
        },
        "controller": {
            "hash": gate12.CONTROLLER_HASH,
            "layer": gate12.LAYER,
            "eta_d75": gate12.ETA_D75,
            "reference_scale": gate12.REFERENCE_SCALE,
            "d75_scalar": gate12.D75_SCALAR,
            "semantics": "sustained current-token path; local derivative at alpha=0",
        },
        "sets": {"control": "24+24 historical", "utility": "32+32 held-out existing"},
        "jvp": read_json(REVIEW / "JVP_ENGINE_SPEC.json"),
        "fisher": {
            "q": "Var_{k~softmax(z0)} r_k",
            "hellinger": "q/4",
            "epsilon_q": gate12.EPSILON_Q,
            "complete_pullback_matrix": False,
        },
        "utility": {
            "primary": "U_mean correct-token log-likelihood directional derivative",
            "secondary": ["U_sum", "eta_utility", "Fisher careful alignment"],
        },
        "statistics": {
            "bootstrap_resamples": gate12.BOOTSTRAP_RESAMPLES,
            "utility_seed": gate12.BOOTSTRAP_SEED,
            "control_seed": gate12.CONTROL_BOOTSTRAP_SEED,
            "unit": "item with every direction",
            "primary_control": "domain-centered Spearman logQ vs logKL_D75",
            "primary_utility": "domain-centered Spearman U_mean vs historical Y",
        },
        "thresholds": {
            "control": {"rho": 0.50, "bootstrap_lower": 0.25},
            "item_utility": {"rho": 0.20, "bootstrap_lower": 0.0, "slope_lower": 0.0},
            "domain_utility": "five frozen sign/contrast rules",
        },
        "classifications": [
            "GATE12_UTILITY_ALIGNED_PULLBACK_SUPPORTED",
            "GATE12_PULLBACK_CONTROL_WITH_DOMAIN_LEVEL_UTILITY_ALIGNMENT",
            "GATE12_PULLBACK_CONTROL_WITHOUT_UTILITY_PREDICTION",
            "GATE12_UTILITY_ALIGNMENT_WITHOUT_PULLBACK_CONTROL_PREDICTION",
            "GATE12_LOCAL_GEOMETRY_NOT_PREDICTIVE",
            "GATE12_GEOMETRY_INCONCLUSIVE",
            "GATE12_JVP_ENGINE_FAILURE",
            "GATE12_SCIENTIFIC_INTEGRITY_CONCERN",
        ],
        "raw_persistence": (
            "complete float32 baseline/careful logits and JVP vectors, one compressed "
            "shard per item/path"
        ),
        "firewall": {
            "historical_outcomes_before_geometry_freeze": "FORBIDDEN",
            "free_generation": "NOT_AUTHORIZED",
            "new_semantic_evaluation": "NOT_AUTHORIZED",
            "q2": "NOT_RUN",
            "holdout": "UNTOUCHED",
        },
        "cost": cost_projection,
        "prospective_amendment": {
            "class": "B",
            "affected_geometry_outputs_existed": False,
            "change": "SDPA math kernel frozen for exact forward-mode AD",
            "scientific_definition_changed": False,
        },
    }
    write_json(REVIEW / "PROTOCOL_LOCK.json", lock)
    (REVIEW / "PROTOCOL_LOCK.md").write_text(
        "# Gate 12 prospective protocol lock\n\n"
        "Status: `FROZEN_PRE_GEOMETRY`. Exact forward-mode JVPs are measured at alpha=0. "
        "Historical semantic outcomes remain inaccessible until `GEOMETRY_FREEZE`. This "
        "is a one-dimensional sustained-control pullback, not a full pullback matrix.\n",
        encoding="utf-8",
    )
    spec = {
        "experiment_id": gate12.EXPERIMENT_ID,
        "status": "FROZEN_PROSPECTIVE",
        "stage": "DEVELOPMENT_GEOMETRY",
        "model_revision": gate12.MODEL_REVISION,
        "layer": gate12.LAYER,
        "control_items": 48,
        "utility_items": 64,
        "directions_per_domain": 5,
        "exact_jvp": True,
        "free_generation": False,
        "new_semantic_evaluation": False,
        "hard_cost_ceiling_usd": 3.5,
        "q2": "NOT_RUN",
        "holdout": "UNTOUCHED",
    }
    (ROOT / "experiments/specs/gate12_utility_aligned_pullback.yaml").write_text(
        yaml.safe_dump(spec, sort_keys=False), encoding="utf-8"
    )
    names = [
        "PREMORTEM.md",
        "PREMORTEM.json",
        "PROTOCOL_LOCK.md",
        "PROTOCOL_LOCK.json",
        "CONTROL_VALIDATION_ITEMS.json",
        "UTILITY_PREDICTION_ITEMS.json",
        "HISTORICAL_DIRECTION_MANIFEST.json",
        "CANONICAL_ANSWER_POLICY.json",
        "JVP_ENGINE_SPEC.json",
        "NUMERICAL_VALIDATION_PLAN.json",
        "GEOMETRY_SCHEDULE.json",
        "COST_PROJECTION.json",
    ]
    write_json(
        REVIEW / "artifact_hashes_preoutcome.json", {name: sha256(REVIEW / name) for name in names}
    )
    print(json.dumps({"premortem": "PREMORTEM_PASS", "geometry_rows": len(schedule)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
