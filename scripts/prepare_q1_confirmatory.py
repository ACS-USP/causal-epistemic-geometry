#!/usr/bin/env python3
"""Prepare sealed Q1 confirmatory locks without reading holdout content."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_confirmatory as q1  # noqa: E402
from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402
from epistemic_geometry.experiments.gate13 import SOURCE_CAREFUL  # noqa: E402

REVIEW = ROOT / "review/q1_confirmatory_fixed_controllers"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def qwen_pairs() -> np.ndarray:
    review = ROOT / "review/gate6_2_first_stage_repair_mean_bridge"
    archive = np.load(review / "SOURCE_ACTIVATIONS.npz", allow_pickle=False)
    items = read_json(review / "SOURCE_SELECTED_TRAIN.json")["items"]
    ids = [str(row["item_id"]) for row in items]
    careful = np.stack(
        [archive[f"train__PROMPT_BOUNDARY__careful__27__{item}"] for item in ids]
    ).astype(np.float64)
    direct = np.stack(
        [archive[f"train__PROMPT_BOUNDARY__direct__27__{item}"] for item in ids]
    ).astype(np.float64)
    return careful - direct


def ministral_pairs() -> np.ndarray:
    archive = np.load(
        ROOT / "review/gate13_cross_model_ministral3/SOURCE_ACTIVATIONS.npz",
        allow_pickle=False,
    )
    return archive["construction_careful"][:, 27].astype(np.float64) - archive[
        "construction_direct"
    ][:, 27].astype(np.float64)


def controller_records() -> dict[str, Any]:
    records = {
        "Qwen": {
            "model": "Qwen/Qwen3-8B",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "tokenizer_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "vector_path": (
                "review/gate6_2_first_stage_repair_mean_bridge/"
                "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy"
            ),
            "vector_hash": "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838",
            "vector_file_sha256": (
                "b1630039fcbb829028a0e8f9f521d7e87bb24e831bc81c74a1591a6c39f40772"
            ),
            "layer": 27,
            "dose": "D75",
            "eta": 9.637427952852196,
            "reference_scale": 10.153299177386142,
            "effective_delta_norm": 97.85168930581241,
            "development_source": "Gate 9",
            "generation": {
                "dtype": "BF16",
                "attention": "SDPA",
                "environment": "CORE_QWEN",
                "enable_thinking": False,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "max_new_tokens": 4096,
            },
            "textual_careful_sha256": hashlib.sha256(SYSTEM_CAREFUL.encode()).hexdigest(),
        },
        "Ministral": {
            "model": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
            "revision": "f6fae9795746f63c9be8344932f01275f3c63734",
            "tokenizer_revision": "f6fae9795746f63c9be8344932f01275f3c63734",
            "vector_path": "review/gate13_cross_model_ministral3/SOURCE_DIRECTIONS/L27.npy",
            "vector_hash": "0c467b7a452619d058afb07c96fd0cd8e20abb19a58d89674ab0a42e00ef2b94",
            "vector_file_sha256": (
                "c6a68967644ae51e60ffe879f6a2e126dc97630f8e6225eeef5ac8911803c9fc"
            ),
            "layer": 27,
            "dose": "D25",
            "eta": 4.469907677389362,
            "reference_scale": 1.0,
            "effective_delta_norm": 4.469907677389362,
            "development_source": "Gate 13.1",
            "generation": {
                "dtype": "BF16",
                "attention": "SDPA",
                "environment": "CORE_MINISTRAL3",
                "fix_mistral_regex": True,
                "text_only": True,
                "vision": False,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.95,
                "top_k": 0,
                "min_p": 0.0,
                "max_new_tokens": 4096,
            },
            "textual_careful_sha256": hashlib.sha256(SOURCE_CAREFUL.encode()).hexdigest(),
        },
    }
    for record in records.values():
        path = ROOT / record["vector_path"]
        vector = np.load(path, allow_pickle=False).astype(np.float64)
        if sha256(path) != record["vector_file_sha256"]:
            raise RuntimeError("fixed controller file hash mismatch")
        if vector_sha256(vector) != record["vector_hash"]:
            raise RuntimeError("fixed controller canonical hash mismatch")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-12):
            raise RuntimeError("fixed controller is not unit norm")
    return records


def build_banks(records: dict[str, Any]) -> dict[str, Any]:
    pairs = {"Qwen": qwen_pairs(), "Ministral": ministral_pairs()}
    output: dict[str, Any] = {}
    for model_role in ("Qwen", "Ministral"):
        meaningful = np.load(ROOT / records[model_role]["vector_path"], allow_pickle=False)
        reconstructed = pairs[model_role].mean(axis=0)
        reconstructed /= np.linalg.norm(reconstructed)
        if vector_sha256(reconstructed) != records[model_role]["vector_hash"]:
            raise RuntimeError(f"{model_role} source pairs do not reproduce fixed controller")
        bank, metadata = q1.build_null_bank(
            meaningful, pairs[model_role], model_role=model_role
        )
        directory = REVIEW / f"NULL_DIRECTIONS_{model_role.upper()}"
        directory.mkdir(parents=True, exist_ok=True)
        for condition, vector in bank.items():
            path = directory / f"{condition}.npy"
            np.save(path, vector.astype(np.float64), allow_pickle=False)
            metadata["records"][condition].update(
                {
                    "vector_path": str(path.relative_to(ROOT)),
                    "file_sha256": sha256(path),
                    "layer": 27,
                    "eta": records[model_role]["eta"],
                    "reference_scale": records[model_role]["reference_scale"],
                    "effective_delta_norm": records[model_role]["effective_delta_norm"],
                }
            )
        metadata["source_pair_archive"] = (
            "review/gate6_2_first_stage_repair_mean_bridge/SOURCE_ACTIVATIONS.npz"
            if model_role == "Qwen"
            else "review/gate13_cross_model_ministral3/SOURCE_ACTIVATIONS.npz"
        )
        metadata["source_pair_archive_sha256"] = sha256(ROOT / metadata["source_pair_archive"])
        write_json(REVIEW / f"NULL_BANK_LOCK_{model_role.upper()}.json", metadata)
        output[model_role] = metadata
    return output


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    identity = read_json(REVIEW / "HOLDOUT_IDENTITY_LOCK.json")
    audit = read_json(REVIEW / "HOLDOUT_PROVENANCE_AUDIT.json")
    power = read_json(REVIEW / "POWER_ANALYSIS.json")
    if identity["status"] != "SEALED_ASSIGNED_UNACCESSED":
        raise RuntimeError("confirmatory holdout identity is not sealed")
    if power["classification"] != "Q1_CONFIRMATORY_N57_POWER_QUALIFIED":
        raise RuntimeError("N=57 power did not qualify")
    item_ids = audit["reserved_cruxeval_57"]["ids"]
    records = controller_records()
    banks = build_banks(records)
    write_json(
        REVIEW / "CONTROLLER_IDENTITY_LOCK.json",
        {
            "status": "FIXED_DEVELOPMENT_CONTROLLERS_VERIFIED",
            "controllers": records,
            "constructor": "PAIRED_CAREFUL_MINUS_DIRECT_MEAN_DIFFERENCE",
            "timing": "SUSTAINED_CURRENT_TOKEN",
            "scope": "FINAL_PROMPT_TOKEN_PREFILL_AND_CURRENT_TOKEN_EACH_DECODE_FORWARD",
            "holdout_activations_used": False,
            "controller_recomputed": False,
        },
    )
    schedules = {}
    all_seeds = []
    for model_role in ("Qwen", "Ministral"):
        rows = q1.build_schedule(item_ids, model_role=model_role)
        schedules[model_role] = rows
        all_seeds.extend(int(row["seed"]) for row in rows)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("cross-model confirmatory schedule seed collision")
    write_json(
        REVIEW / "SEED_SCHEDULE_LOCK.json",
        {
            "seed_regime": "INDEPENDENT_PRIMARY",
            "global_seed_collisions": 0,
            "conditions": list(q1.CONDITIONS),
            "rollouts_per_item_condition": 2,
            "n_items": 57,
            "rows_per_model": 798,
            "total_rows": 1596,
            "schedules": schedules,
        },
    )
    parser_path = ROOT / "src/epistemic_geometry/benchmarks/external/semantic_v3.py"
    parser_parent = read_json(
        ROOT / "review/gate13_cross_model_ministral3/RESPONSE_PARSER_LOCK.json"
    )
    write_json(
        REVIEW / "RESPONSE_PARSER_LOCK.json",
        {
            "version": "external-semantic-v3",
            "module": str(parser_path.relative_to(ROOT)),
            "module_sha256": sha256(parser_path),
            "semantic_spec_sha256": parser_parent["spec_sha256"],
            "test_corpus_sha256": parser_parent["test_corpus_sha256"],
            "condition_invariance_required": True,
            "post_collection_amendment_permitted": False,
        },
    )
    write_json(
        REVIEW / "ANALYSIS_LOCK.json",
        {
            "primary_estimand": "C_MEANINGFUL",
            "C_definition": "B00-B0j-U00+U0j_CANONICAL_TWO_ROLLOUT_U_STATISTIC",
            "bootstrap": {
                "unit": "ITEM",
                "method": "TWO_SIDED_95_PERCENT_PERCENTILE",
                "resamples": q1.BOOTSTRAP_RESAMPLES,
                "seeds": q1.BOOTSTRAP_SEEDS,
                "move_all_conditions_and_both_rollouts_together": True,
            },
            "P1": "LOWER_95_CI_C_MEANINGFUL_GT_ZERO",
            "P2": [
                "LOWER_95_CI_C_MEANINGFUL_MINUS_MEAN_C_NULLS_GT_ZERO",
                "POINT_C_MEANINGFUL_GT_MAX_C_NULLS",
            ],
            "safety": {
                "commitment_relative_margin": -0.05,
                "evaluability_relative_margin": -0.05,
                "accuracy_relative_margin": -0.10,
            },
            "cross_model_rule": "BOTH_MODEL_SPECIFIC_HYPOTHESES_MUST_PASS",
            "textual_careful_role": "DESCRIPTIVE_ONLY",
            "G_D_role": "SECONDARY_DESCRIPTIVE_ONLY",
            "no_complete_case_filtering": True,
        },
    )
    write_json(
        REVIEW / "COST_LOCK.json",
        {
            "wallet_observation_usd": 12.09,
            "target_cumulative_runpod_cost_usd": 6.50,
            "hard_cumulative_runpod_ceiling_usd": 8.75,
            "desired_wallet_reserve_usd": 3.00,
            "mandatory_projection_safety_margin": 0.25,
            "preflight_must_use_non_holdout_or_consumed_development_fixtures": True,
            "projection_status": "PENDING_REMOTE_PREFLIGHT",
        },
    )
    (REVIEW / "HYPOTHESIS_LOCK.md").write_text(
        "# Q1 Confirmatory Hypothesis Lock\n\n"
        "For each fixed model-specific controller, the two-sided 95% item-bootstrap "
        "lower bound for competence-adjusted complementarity `C` must exceed zero, "
        "the corresponding lower bound versus the mean of four fresh nulls must "
        "exceed zero, and point `C` must exceed every null. Commitment validity and "
        "semantic evaluability may each fall by at most 0.05 from baseline; accuracy "
        "may fall by at most 0.10. The cross-model claim passes only if both models pass.\n",
        encoding="utf-8",
    )
    (REVIEW / "PREMORTEM.md").write_text(
        "# Q1 Confirmatory Premortem\n\n"
        "Classification: `PREMORTEM_PASS_FOR_DRESS_REHEARSAL`.\n\n"
        "The main risks are holdout leakage, controller drift, null selection, seed "
        "collision, parser drift, model-output attrition, cross-model peeking, resume "
        "duplication, and cost overrun. The locks prohibit content access before the "
        "cost gate, bind byte/canonical controller hashes, freeze two isotropic plus "
        "two source-pair sign-shuffled nulls per model, use globally distinct seeds, "
        "retain all model-level invalid outcomes, require both journals before analysis, "
        "resume by immutable logical key, and require the 25% cost margin.\n",
        encoding="utf-8",
    )
    write_json(
        REVIEW / "PREMORTEM.json",
        {
            "classification": "PREMORTEM_PASS_FOR_DRESS_REHEARSAL",
            "risks_checked": [
                "holdout_leakage",
                "controller_identity",
                "null_selection",
                "seed_independence",
                "parser_invariance",
                "attrition",
                "cross_model_peeking",
                "resume",
                "cost",
            ],
            "scientific_ambiguity_remaining": False,
        },
    )
    spec = {
        "experiment_id": q1.EXPERIMENT_ID,
        "status": "PROSPECTIVE_CONFIRMATORY_LOCK_PRE_HOLDOUT",
        "stage": "CONFIRMATORY_PRE_HOLDOUT_LOCK_PREPARATION",
        "holdout": {
            "role": identity["role"],
            "status": identity["status"],
            "n": 57,
            "ordered_id_list_sha256": identity["ordered_id_list_sha256"],
            "content_accessed": False,
        },
        "models": records,
        "conditions": list(q1.CONDITIONS),
        "rows": {"per_model": 798, "total": 1596},
        "analysis": "review/q1_confirmatory_fixed_controllers/ANALYSIS_LOCK.json",
        "cost": "review/q1_confirmatory_fixed_controllers/COST_LOCK.json",
    }
    spec_path = ROOT / "experiments/specs/q1_confirmatory_fixed_controllers.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    write_json(
        REVIEW / "PROTOCOL_LOCK.json",
        {
            **spec,
            "status": "PRE_HOLDOUT_DRESS_REHEARSAL_PENDING",
            "prepared_from_commit": git_commit(),
            "power_classification": power["classification"],
            "null_bank_hashes": {
                role: {
                    name: row["canonical_float64_vector_sha256"]
                    for name, row in bank["records"].items()
                }
                for role, bank in banks.items()
            },
            "exclusions": {
                "default": "NO_POST_COLLECTION_ITEM_EXCLUSIONS",
                "allowed_only": [
                    "CORRUPTED_SOURCE_RECORD_INDEPENDENT_OF_MODEL_OUTPUT",
                    "IMPOSSIBLE_MISSING_REFERENCE_DUE_TO_REPOSITORY_DAMAGE",
                ],
                "model_output_failures_retained_as_errors": True,
                "immutable_ledger_required": True,
            },
            "scientific_firewall": {"q2": "NOT_RUN", "q3": "NOT_RUN"},
        },
    )
    (REVIEW / "DESIGN_REVIEW.md").write_text(
        "# Q1 Confirmatory Pre-Holdout Design Review\n\n"
        "Phase Zero is resolved by a prospective principal assignment. Offline N=57 "
        "power qualifies for both models. Controller identities, null constructors, "
        "analysis, parser, schedule, exclusions, and cost rules are now frozen for the "
        "dress rehearsal. The holdout remains sealed and no model inference occurred.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PREPARED", "holdout_content_accessed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
