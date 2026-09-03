#!/usr/bin/env python3
"""Prepare the model-free Q1 second-task prospective lock and manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import q1_second_task as q1s  # noqa: E402
from epistemic_geometry.experiments.gate6 import SYSTEM_CAREFUL  # noqa: E402
from epistemic_geometry.experiments.gate6_3 import vector_sha256  # noqa: E402

REVIEW = ROOT / "review/q1_second_task_spark2_design"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_api_pages(directory: Path, prefix: str) -> list[dict[str, Any]]:
    paths = sorted(
        directory.glob(f"{prefix}_*.json"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if not paths:
        raise FileNotFoundError(f"no {prefix}_*.json pages in {directory}")
    rows = [entry for path in paths for entry in json.loads(path.read_text())["rows"]]
    rows.sort(key=lambda entry: int(entry["row_idx"]))
    return [dict(entry["row"]) for entry in rows]


_TOKEN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\w\s]")


def _shingles(text: str, width: int = 5) -> set[tuple[str, ...]]:
    tokens = _TOKEN.findall(text.lower())
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def _content_overlap(
    items: list[q1s.LiveCodeBenchItem], cruxeval: list[dict[str, Any]]
) -> dict[str, Any]:
    lcb_by_question: dict[str, q1s.LiveCodeBenchItem] = {}
    for item in items:
        lcb_by_question.setdefault(item.question_id, item)
    crux_codes = [(str(row["id"]), str(row["code"])) for row in cruxeval]
    crux_exact = {
        hashlib.sha256("".join(_TOKEN.findall(code.lower())).encode()).hexdigest()
        for _item_id, code in crux_codes
    }
    exact = 0
    near = 0
    maximum = 0.0
    for item in lcb_by_question.values():
        normalized_hash = hashlib.sha256(
            "".join(_TOKEN.findall(item.starter_code.lower())).encode()
        ).hexdigest()
        exact += int(normalized_hash in crux_exact)
        left = _shingles(item.starter_code)
        if len(left) < 20:
            continue
        for _crux_id, code in crux_codes:
            right = _shingles(code)
            if len(right) < 20:
                continue
            similarity = len(left & right) / len(left | right)
            maximum = max(maximum, similarity)
            near += int(similarity >= 0.80)
    return {
        "lcb_unique_questions": len(lcb_by_question),
        "cruxeval_rows_compared": len(crux_codes),
        "exact_normalized_code_collisions": exact,
        "near_duplicate_threshold": 0.80,
        "near_duplicate_minimum_shingles": 20,
        "near_duplicate_pairs": near,
        "maximum_five_token_shingle_jaccard": maximum,
        "coverage": (
            "entire official CRUXEval test revision; therefore covers all project "
            "development/source/calibration/confirmatory/Q2 CRUXEval subsets"
        ),
        "q2_semantic_outputs_read": False,
    }


def _qwen_pairs(source_activations: Path) -> np.ndarray:
    directory = source_activations.parent
    expected = "a0acf32382ce1a1c31a1af9c161868ab9aaf7d4aaabef58e99fe93261d608fd3"
    if _sha256(source_activations) != expected:
        raise RuntimeError("Qwen source-pair archive hash mismatch")
    archive = np.load(source_activations, allow_pickle=False)
    items = json.loads((directory / "SOURCE_SELECTED_TRAIN.json").read_text())["items"]
    ids = [str(row["item_id"]) for row in items]
    careful = np.stack(
        [archive[f"train__PROMPT_BOUNDARY__careful__27__{item_id}"] for item_id in ids]
    ).astype(np.float64)
    direct = np.stack(
        [archive[f"train__PROMPT_BOUNDARY__direct__27__{item_id}"] for item_id in ids]
    ).astype(np.float64)
    return careful - direct


def _controller_and_nulls(source_activations: Path) -> dict[str, Any]:
    vector_path = (
        ROOT
        / "review/gate6_2_first_stage_repair_mean_bridge/PAIRED_MEAN_DIRECTIONS/"
        "PROMPT_BOUNDARY/L27.npy"
    )
    meaningful = np.load(vector_path, allow_pickle=False).astype(np.float64)
    if _sha256(vector_path) != q1s.MEANINGFUL_VECTOR_FILE_SHA256:
        raise RuntimeError("fixed Qwen vector file hash mismatch")
    if vector_sha256(meaningful) != q1s.MEANINGFUL_VECTOR_HASH:
        raise RuntimeError("fixed Qwen canonical vector hash mismatch")
    old_lock = json.loads(
        (ROOT / "review/q1_confirmatory_fixed_controllers/NULL_BANK_LOCK_QWEN.json").read_text()
    )
    existing = {}
    for name in q1s.RANDOM_NAMES[:4]:
        record = old_lock["records"][name]
        path = ROOT / record["vector_path"]
        value = np.load(path, allow_pickle=False).astype(np.float64)
        if vector_sha256(value) != record["canonical_float64_vector_sha256"]:
            raise RuntimeError(f"historical null hash mismatch: {name}")
        existing[name] = value
    bank, metadata = q1s.build_extended_null_bank(
        meaningful, existing, _qwen_pairs(source_activations)
    )
    records = {}
    for name, value in bank.items():
        if name in existing:
            record = dict(old_lock["records"][name])
            record["provenance"] = "REUSED_EXACT_Q1_CONFIRMATORY_NULL"
        else:
            path = REVIEW / "NULL_DIRECTIONS" / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, value.astype(np.float64), allow_pickle=False)
            record = {
                **metadata["new_records"][name],
                "vector_path": str(path.relative_to(ROOT)),
                "file_sha256": _sha256(path),
                "provenance": "PROSPECTIVE_Q1_SECOND_TASK_NULL",
            }
        record.update(
            {
                "layer": q1s.LAYER,
                "eta": q1s.ETA,
                "reference_scale": q1s.REFERENCE_SCALE,
                "effective_delta_norm": q1s.EFFECTIVE_DELTA_NORM,
            }
        )
        records[name] = record
    metadata["records"] = records
    metadata["source_pair_archive"] = (
        "review/gate6_2_first_stage_repair_mean_bridge/SOURCE_ACTIVATIONS.npz"
    )
    metadata["source_pair_archive_sha256"] = _sha256(source_activations)
    metadata["source_selection_manifest"] = (
        "review/gate6_2_first_stage_repair_mean_bridge/SOURCE_SELECTED_TRAIN.json"
    )
    metadata["source_selection_manifest_sha256"] = _sha256(
        source_activations.parent / "SOURCE_SELECTED_TRAIN.json"
    )
    metadata["cosine_max_abs_off_diagonal"] = float(
        np.max(np.abs(np.asarray(metadata["cosine_matrix"]) - np.eye(9)))
    )
    return metadata


def _fixtures() -> list[dict[str, Any]]:
    prompts = [
        "Continue this neutral token sequence briefly: alpha beta gamma.",
        "Format one short line containing punctuation: [] {} ().",
        "Respond to this synthetic string without evaluating a benchmark: 0011223344.",
        "Echo a compact multilingual-neutral token pattern: uno deux drei.",
        "Write a short code-like placeholder using names foo and bar; no execution.",
        "Give a brief response to a repeated-token fixture: zeta zeta zeta zeta.",
        "Produce a short literal-like answer for parser plumbing: [1, 2].",
        "End a generic engineering response with the marker FINAL: 7.",
    ]
    return [
        {
            "fixture_id": f"SYNTHETIC_ENGINE_{index:02d}",
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "scientific_item": False,
            "correctness_oracle": None,
        }
        for index, prompt in enumerate(prompts)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--livecodebench-dir", type=Path, required=True)
    parser.add_argument("--livecodebench-parquet", type=Path, required=True)
    parser.add_argument("--cruxeval-dir", type=Path, required=True)
    parser.add_argument("--source-activations", type=Path, required=True)
    args = parser.parse_args()
    REVIEW.mkdir(parents=True, exist_ok=True)

    if _sha256(args.livecodebench_parquet) != q1s.LIVECODEBENCH_PARQUET_SHA256:
        raise RuntimeError("official LiveCodeBench parquet hash mismatch")
    lcb_rows = _load_api_pages(args.livecodebench_dir, "rows")
    crux_rows = _load_api_pages(args.cruxeval_dir, "cruxeval_rows")
    items = [q1s.normalize_livecodebench_row(row, index) for index, row in enumerate(lcb_rows)]
    if len(items) != 442 or len({item.item_id for item in items}) != 442:
        raise RuntimeError("official LiveCodeBench pool must contain 442 unique item IDs")
    stage_a, stage_b, reserve = q1s.split_items(items)
    if {item.question_id for item in stage_a} & {item.question_id for item in stage_b}:
        raise RuntimeError("question-family leakage across stages")

    overlap = _content_overlap(items, crux_rows)
    if overlap["exact_normalized_code_collisions"] or overlap["near_duplicate_pairs"]:
        raise RuntimeError("LiveCodeBench content overlaps the full CRUXEval source pool")
    reference_types = Counter(type(json.loads(item.reference_json)).__name__ for item in items)
    for item in items:
        expected = json.loads(item.reference_json)
        rendered = (
            json.dumps(expected) if isinstance(expected, (list, str, bool)) else str(expected)
        )
        result = q1s.evaluate_livecodebench_output(f"FINAL: {rendered}", item.reference_json)
        if not result["correct"]:
            raise RuntimeError(f"reference roundtrip failed for {item.item_id}")

    stage_a_schedule = q1s.build_schedule(
        stage_a,
        stage="STAGE_A",
        conditions=q1s.STAGE_A_CONDITIONS,
        rollouts=q1s.STAGE_A_ROLLOUTS,
    )
    stage_b_schedule = q1s.build_schedule(
        stage_b,
        stage="STAGE_B",
        conditions=q1s.STAGE_B_CONDITIONS,
        rollouts=q1s.STAGE_B_ROLLOUTS,
    )
    if len({row["seed"] for row in stage_a_schedule + stage_b_schedule}) != (
        len(stage_a_schedule) + len(stage_b_schedule)
    ):
        raise RuntimeError("cross-stage seed collision")

    pool_manifest = {
        "dataset": "livecodebench/test_generation",
        "revision": q1s.LIVECODEBENCH_DATASET_REVISION,
        "parquet_sha256": q1s.LIVECODEBENCH_PARQUET_SHA256,
        "n_items": len(items),
        "n_questions": len({item.question_id for item in items}),
        "content_redistributed": False,
        "records": [item.public_manifest_record() for item in items],
    }
    _write_json(REVIEW / "ITEM_POOL_HASH_MANIFEST.json", pool_manifest)
    for filename, role, values in (
        ("STAGE_A_MANIFEST.json", "DEVELOPMENT_OPPORTUNITY", stage_a),
        ("STAGE_B_HOLDOUT_MANIFEST.json", "SECOND_TASK_HOLDOUT", stage_b),
        ("RESERVE_MANIFEST.json", "UNALLOCATED_RESERVE", reserve),
    ):
        _write_json(
            REVIEW / filename,
            {
                "role": role,
                "dataset_revision": q1s.LIVECODEBENCH_DATASET_REVISION,
                "n_items": len(values),
                "n_questions": len({item.question_id for item in values}),
                "content_redistributed": False,
                "ordered_records": [item.public_manifest_record() for item in values],
            },
        )
    _write_json(REVIEW / "STAGE_A_SCHEDULE.json", stage_a_schedule)
    _write_json(REVIEW / "STAGE_B_SCHEDULE.json", stage_b_schedule)
    _write_json(REVIEW / "CONTENT_OVERLAP_AUDIT.json", overlap)

    instrument = {
        "classification": "LIVECODEBENCH_OUTPUT_INSTRUMENT_MODEL_FREE_PASS",
        "official_repository": {
            "url": "https://github.com/LiveCodeBench/LiveCodeBench",
            "commit": q1s.LIVECODEBENCH_REPOSITORY_COMMIT,
            "license": "MIT",
        },
        "official_dataset": {
            "repo": "livecodebench/test_generation",
            "revision": q1s.LIVECODEBENCH_DATASET_REVISION,
            "parquet_sha256": q1s.LIVECODEBENCH_PARQUET_SHA256,
            "rows": len(items),
            "unique_ids": len({item.item_id for item in items}),
            "unique_questions": len({item.question_id for item in items}),
            "dataset_card_license_field": "cc",
            "license_note": (
                "The dataset card does not specify a CC variant. Benchmark content is not "
                "redistributed in Git; only IDs and hashes are sealed. Publication redistribution "
                "requires separate license review."
            ),
        },
        "task": "test_output_prediction",
        "stable_item_id": "question_id:test_id",
        "one_test_per_row": True,
        "reference_type_counts": dict(sorted(reference_types.items())),
        "prompt_reconstruction": "deterministic CEG final-commitment adapter over official fields",
        "evaluator": {
            "name": "livecodebench-exact-literal-v1",
            "official_evaluator_audited": (
                "lcb_runner/evaluation/compute_test_output_prediction_metrics.py"
            ),
            "official_evaluator_note": (
                "The upstream evaluator calls Python eval on extracted model text. The CEG adapter "
                "freezes a narrower literal-only output grammar and uses "
                "ast.literal_eval/json.loads; "
                "there is no execution of generated code. It is an exact typed-value evaluator for "
                "the frozen grammar, not a claim of leaderboard-protocol identity."
            ),
            "llm_judge": False,
            "fuzzy_matching": False,
            "untrusted_generated_code_executed": False,
            "reference_roundtrips": len(items),
        },
        "failure_taxonomy": [
            "VALID_CORRECT",
            "VALID_WRONG",
            "INVALID_FORMAT",
            "UNEVALUABLE",
            "TRUNCATION",
            "RUNTIME_ERROR",
        ],
        "historical_low_cap_diagnostic": "LOW_CAP_DIAGNOSTIC_NOT_SCIENTIFIC_EVIDENCE",
        "content_overlap_audit": overlap,
        "model_inference": False,
        "correctness_outcomes_inspected": False,
    }
    _write_json(REVIEW / "MODEL_FREE_INSTRUMENT_AUDIT.json", instrument)

    null_bank = _controller_and_nulls(args.source_activations)
    _write_json(REVIEW / "RANDOM_BANK_LOCK.json", null_bank)
    vector_path = (
        ROOT
        / "review/gate6_2_first_stage_repair_mean_bridge/PAIRED_MEAN_DIRECTIONS/"
        "PROMPT_BOUNDARY/L27.npy"
    )
    controller = {
        "status": "EXACT_FIXED_QWEN_CONTROLLER_VERIFIED",
        "model": q1s.MODEL,
        "revision": q1s.MODEL_REVISION,
        "tokenizer_revision": q1s.TOKENIZER_REVISION,
        "layer": q1s.LAYER,
        "dose": "D75",
        "eta": q1s.ETA,
        "reference_scale": q1s.REFERENCE_SCALE,
        "effective_delta_norm": q1s.EFFECTIVE_DELTA_NORM,
        "vector_hash": q1s.MEANINGFUL_VECTOR_HASH,
        "vector_file_sha256": q1s.MEANINGFUL_VECTOR_FILE_SHA256,
        "vector_path": str(vector_path.relative_to(ROOT)),
        "timing": "SUSTAINED_CURRENT_TOKEN",
        "scope": "FINAL_PROMPT_TOKEN_PREFILL_AND_CURRENT_TOKEN_EACH_DECODE_FORWARD",
        "generation": {
            "dtype": "BF16",
            "attention": "SDPA",
            "enable_thinking": False,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_new_tokens": 4096,
        },
        "textual_careful": SYSTEM_CAREFUL,
        "textual_careful_sha256": hashlib.sha256(SYSTEM_CAREFUL.encode()).hexdigest(),
    }
    _write_json(REVIEW / "CONTROLLER_PROVENANCE_LOCK.json", controller)

    decision = {
        "primary_estimator": "POOLED_R4_UNBIASED_U_STATISTIC",
        "r2_reduction": "EXACT_CANONICAL_Q1_ESTIMATOR",
        "split_halves": {"A": [0, 1], "B": [2, 3]},
        "primary_endpoint": "C_MEANINGFUL",
        "secondary_endpoints": ["G", "D", "rescue", "damage", "accuracy"],
        "bootstrap": {
            "unit": "ITEM",
            "resamples": 50_000,
            "method": "TWO_SIDED_95_PERCENT_PERCENTILE",
            "move_all_conditions_and_four_rollouts_together": True,
            "seed": 2026082902,
            "negative_D_retained": True,
        },
        "stage_a_gate": {
            "baseline_commitment_validity_min": 0.95,
            "baseline_semantic_evaluability_min": 0.95,
            "baseline_pooled_accuracy_range_inclusive": [0.25, 0.90],
            "baseline_B00_min": 0.05,
            "items_wrong_both_rollouts_min": 5,
            "items_correct_at_least_once_min": 10,
            "textual_commitment_validity_min": 0.95,
            "textual_semantic_evaluability_min": 0.95,
            "textual_accuracy_relative_margin": -0.03,
            "textual_useful_any": [
                "accuracy_gain_ge_0.03",
                "mean_tokens_ge_1.5x_baseline",
                "median_tokens_ge_baseline_plus_10",
            ],
            "all_non_usefulness_conditions": "OR_WITHIN_USEFUL_ANY; AND_FOR_OTHER_GATES",
        },
        "stage_b_scientific": {
            "P1": "LOWER_95_CI_C_MEANINGFUL_GT_ZERO",
            "P2": [
                "LOWER_95_CI_C_MEANINGFUL_MINUS_MEAN_8_NULLS_GT_ZERO",
                "POINT_C_MEANINGFUL_GT_MAX_8_NULLS",
            ],
            "split_half_consistency": (
                "both predesignated R2 halves have C>0, delta_C_nullmean>0, and "
                "C_meaningful>mean_C_nulls"
            ),
            "safety": {
                "commitment_relative_margin": -0.05,
                "evaluability_relative_margin": -0.05,
                "accuracy_relative_margin": -0.10,
            },
            "multiplicity": (
                "single conjunctive primary hypothesis; eight-null mean interval plus point max; "
                "no extra post-hoc multiplicity family"
            ),
        },
        "terminal_states": {
            "Q1_SECOND_TASK_FIXED_CONTROLLER_PASS": (
                "Stage A and engine pass; P1, both P2 checks, split-half consistency, and all "
                "safety guards pass"
            ),
            "Q1_SECOND_TASK_COMPLEMENTARITY_WITH_SAFETY_FAIL": (
                "scientific P1/P2/split-half checks pass but any safety guard fails"
            ),
            "Q1_SECOND_TASK_NO_NULL_SPECIFIC_COMPLEMENTARITY": (
                "instrument and execution qualify but the scientific conjunction does not pass"
            ),
            "Q1_SECOND_TASK_INSTRUMENT_NOT_QUALIFIED": (
                "model-free, Spark-2 engine, or Stage-A opportunity/textual gate fails"
            ),
            "Q1_SECOND_TASK_EXECUTION_INCOMPLETE": "Stage B cannot complete frozen logical keys",
        },
    }
    _write_json(REVIEW / "ESTIMATOR_AND_DECISION_LOCK.json", decision)
    fixtures = _fixtures()
    _write_json(REVIEW / "ENGINEERING_FIXTURES.json", {"fixtures": fixtures})
    engine_protocol = {
        "classification_if_pass": "SPARK2_NATIVE_ENGINE_QUALIFIED",
        "backend_claim": "SPARK2_NATIVE_CROSS_BACKEND_REPLICATION",
        "cross_backend_numerical_equivalence_claimed": False,
        "expected": {
            "hostname": "spark2",
            "architecture": "aarch64",
            "gpu": "NVIDIA GB10",
            "gpu_count": 1,
            "python": "3.12.3",
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "transformers": "4.57.6",
            "dtype": "BF16",
            "attention": "SDPA",
            "torch_disable_native_jit": "1",
            "model_revision": q1s.MODEL_REVISION,
        },
        "required_checks": [
            "environment_exact",
            "model_and_tokenizer_identity",
            "vector_identity",
            "alpha_zero_token_identity",
            "seed_repeatability",
            "hook_cleanup",
            "per_forward_exact_shift_bf16_eps_le_2",
            "current_token_noncurrent_change_le_0.125",
            "one_application_per_forward",
            "cached_decode_observed",
            "random_delta_norm_range_le_1e-9",
            "parser_roundtrip",
            "journal_resume_synthetic",
        ],
        "scientific_benchmark_items_allowed": 0,
    }
    _write_json(REVIEW / "SPARK2_ENGINE_QUALIFICATION_PROTOCOL.json", engine_protocol)

    prelock = {
        "experiment_id": q1s.EXPERIMENT_ID,
        "status": "PROSPECTIVE_LOCK_PRE_ENGINE",
        "principal_execution_authorization": "DESIGN_AND_SYNTHETIC_ENGINE_ONLY",
        "stage_a_inference_authorized": False,
        "stage_b_inference_authorized": False,
        "instrument_classification": instrument["classification"],
        "controller": controller,
        "random_bank_file": "RANDOM_BANK_LOCK.json",
        "stage_a": {
            "n": q1s.STAGE_A_N,
            "rollouts": q1s.STAGE_A_ROLLOUTS,
            "conditions": list(q1s.STAGE_A_CONDITIONS),
            "logical_rows": len(stage_a_schedule),
        },
        "stage_b": {
            "n": q1s.STAGE_B_N,
            "rollouts": q1s.STAGE_B_ROLLOUTS,
            "conditions": list(q1s.STAGE_B_CONDITIONS),
            "logical_rows": len(stage_b_schedule),
        },
        "generation": controller["generation"],
        "evaluator": instrument["evaluator"],
        "decision_lock": "ESTIMATOR_AND_DECISION_LOCK.json",
        "resource_policy": {
            "spark1": "FORBIDDEN_OWNED_BY_OPEN_Q2",
            "spark2": "SYNTHETIC_ENGINE_ONLY_THIS_SPRINT",
            "runpod": "FORBIDDEN",
            "multi_node": False,
        },
        "q2_firewall": {
            "outputs_inspected": False,
            "process_modified": False,
            "paths_imported": False,
        },
        "hashes": {},
    }
    locked_files = [
        "ITEM_POOL_HASH_MANIFEST.json",
        "STAGE_A_MANIFEST.json",
        "STAGE_B_HOLDOUT_MANIFEST.json",
        "RESERVE_MANIFEST.json",
        "STAGE_A_SCHEDULE.json",
        "STAGE_B_SCHEDULE.json",
        "MODEL_FREE_INSTRUMENT_AUDIT.json",
        "CONTROLLER_PROVENANCE_LOCK.json",
        "RANDOM_BANK_LOCK.json",
        "ESTIMATOR_AND_DECISION_LOCK.json",
        "ENGINEERING_FIXTURES.json",
        "SPARK2_ENGINE_QUALIFICATION_PROTOCOL.json",
    ]
    prelock["hashes"] = {name: _sha256(REVIEW / name) for name in locked_files}
    _write_json(REVIEW / "PROTOCOL_LOCK.json", prelock)
    _write_json(
        REVIEW / "PREMORTEM.json",
        {
            "classification": "PREMORTEM_PASS_FOR_SYNTHETIC_ENGINE",
            "risks": [
                "q2_cross_contamination",
                "benchmark_shopping",
                "controller_drift",
                "prompt_search",
                "unsafe_upstream_eval",
                "question_family_leakage",
                "null_redraw",
                "backend_drift",
                "r4_estimator_mismatch",
                "split_half_selection",
                "stage_a_peeking",
                "spark1_misrouting",
            ],
            "scientific_benchmark_inference": False,
        },
    )
    (REVIEW / "PREMORTEM.md").write_text(
        "# Q1 Second-Task Premortem\n\n"
        "Classification: `PREMORTEM_PASS_FOR_SYNTHETIC_ENGINE`.\n\n"
        "The lock prohibits benchmark shopping, Q2 access, Spark 1 use, controller/dose/prompt "
        "search, unsafe execution of generated text, question-family leakage, null redraw, and "
        "post-outcome estimator changes. Stage A contains baseline and textual CAREFUL only; "
        "Stage B cannot open unless the frozen Stage-A gate passes.\n",
        encoding="utf-8",
    )
    _write_json(
        REVIEW / "Q2_FIREWALL_AUDIT.json",
        {
            "classification": "Q2_FIREWALL_CLEAN",
            "spark1_used": False,
            "q2_output_paths_read": False,
            "q2_generated_text_read": False,
            "q2_correctness_inspected": False,
            "q2_process_modified": False,
            "q2_artifacts_imported": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
