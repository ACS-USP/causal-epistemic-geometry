from __future__ import annotations

import hashlib
import json

import yaml

from epistemic_geometry.benchmarks.mmlu_pro import row_to_item
from epistemic_geometry.benchmarks.permutations import (
    cyclic_mmlu_item,
    cyclic_option_order,
    validate_cyclic_balance,
)
from epistemic_geometry.config import load_config
from epistemic_geometry.experiments import q1_v1_2 as v12
from epistemic_geometry.experiments.q1_v1_2 import _symmetrize
from epistemic_geometry.reproducibility import canonical_json, stable_digest
from epistemic_geometry.types import PreparedChoiceItem


def _item():
    return row_to_item(
        {
            "question_id": "q-v12",
            "question": "Which option is correct?",
            "options": ["zero", "one", "two", "three"],
            "answer_index": 2,
            "category": "fixture",
        },
        "test",
    )


def test_cyclic_ordering_visits_every_slot_once() -> None:
    orders = [cyclic_option_order(4, shift) for shift in range(4)]
    assert orders[0] == [0, 1, 2, 3]
    for semantic in range(4):
        assert sorted(order.index(semantic) for order in orders) == [0, 1, 2, 3]


def test_cyclic_item_preserves_semantic_target() -> None:
    original = _item()
    shifted, manifest = cyclic_mmlu_item(original, 1)
    assert manifest["original_target_index"] == 2
    assert manifest["permuted_target_index"] == 3
    assert shifted.metadata["semantic_option_ids"] == [3, 0, 1, 2]
    assert shifted.target == "D"


def test_cyclic_balance_report_is_exact() -> None:
    report = validate_cyclic_balance([_item()])
    assert report["status"] == "PASS"
    assert report["records"][0]["every_semantic_option_visits_every_slot"] is True


def test_v1_v2_centered_symmetrization_recovers_semantic_choice() -> None:
    item = _item()
    raw_rows = []
    for shift in range(4):
        shifted, _manifest = cyclic_mmlu_item(item, shift)
        semantic_ids = shifted.metadata["semantic_option_ids"]
        for role in ("baseline", "pc1_minus", "pc1_plus"):
            scores = {
                label: float(10.0 if semantic == 2 else 0.0) + float(index)
                for index, (label, semantic) in enumerate(zip("ABCD", semantic_ids, strict=True))
            }
            raw_rows.append(
                {
                    "item_id": item.id,
                    "cyclic_shift": shift,
                    "role": role,
                    "candidate_labels": list("ABCD"),
                    "semantic_option_ids": semantic_ids,
                    "candidate_scores": scores,
                }
            )
    sym_rows, predictions, agreement = _symmetrize(raw_rows, [item])
    assert len(sym_rows) == 3
    assert all(prediction.correct for prediction in predictions["baseline"])
    assert all(prediction.normalized_output == "2" for prediction in predictions["pc1_plus"])
    assert agreement == {"baseline": 1.0, "pc1_minus": 1.0, "pc1_plus": 1.0}


def test_v1_v2_config_is_frozen_and_approved_engine_is_explicit() -> None:
    config = load_config("configs/q1_v1_2_qwen3_8b.yaml")
    assert config.q1_v1_2["protocol"] == "Q1_DEVELOPMENT_PROTOCOL_V1_2"
    assert config.backend.execution_mode == "full_prompt_batched"
    assert config.backend.candidate_head_mode == "candidate_only"
    assert config.backend.serial_shape_reference is True
    assert config.q1_v1_2["beta_probe"] == 0.05


def test_v1_v2_validator_recomputes_synthetic_artifact(tmp_path, monkeypatch) -> None:
    """The local validator must audit derived files without model/data access."""

    monkeypatch.setattr(v12, "EVALUATION_SIZE", 1)
    item = row_to_item(
        {
            "question_id": "q-validator",
            "question": "Which option is correct?",
            "options": [f"option-{index}" for index in range(10)],
            "answer_index": 2,
        },
        "test",
    )
    split_payload = {
        "manifest_sha256": "synthetic-frozen-split-digest",
        "splits": {"dev_evaluation": [item.id], "confirmatory_holdout": []}
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split_payload), encoding="utf-8")
    split_file_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()
    monkeypatch.setattr(v12, "V1_SPLIT_HASH", "synthetic-frozen-split-digest")

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = load_config("configs/q1_v1_2_qwen3_8b.yaml")
    resolved = config.as_dict()
    resolved["q1_v1_2"]["bootstrap_resamples"] = 4
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    config_hash = stable_digest(
        canonical_json(
            {
                "config": resolved,
                "protocol": v12.PROTOCOL_ID,
                "split_manifest_sha256": split_file_hash,
            }
        )
    )[:10]

    raw_rows = []
    for shift in range(10):
        shifted, _manifest = cyclic_mmlu_item(item, shift)
        semantic_ids = shifted.metadata["semantic_option_ids"]
        prepared = PreparedChoiceItem(
            item_id=item.id,
            target=shifted.target,
            metadata=dict(shifted.metadata),
            rendered_prompt=shifted.prompt,
            rendered_prompt_hash=stable_digest(item.id, str(shift)),
            prompt_ids=(1, 2),
            candidate_labels=tuple("ABCDEFGHIJ"),
            candidate_token_ids={label: (index,) for index, label in enumerate("ABCDEFGHIJ")},
            context_compatible_candidate_ids={
                label: (index,) for index, label in enumerate("ABCDEFGHIJ")
            },
            semantic_option_ids=tuple(semantic_ids),
        )
        for role in v12.ALL_ROLES:
            spec = {
                "condition": f"cyclic_{shift:02d}_{role}",
                "role": role,
                "cyclic_shift": shift,
                "alpha": 0.0,
                "layer": 17,
                "token_scope": "last_token",
                "vector_hash": None if role == "baseline" else v12.PC1_HASH,
            }
            scores = {
                label: float(10.0 if semantic == 2 else 0.0) + index / 100.0
                for index, (label, semantic) in enumerate(
                    zip("ABCDEFGHIJ", semantic_ids, strict=True)
                )
            }
            raw_rows.append(v12._raw_record(item, prepared, spec, scores, "SYNTHETIC"))
    v12._write_jsonl(run_dir / "raw_permutation_scores.jsonl", raw_rows)
    sym_rows, predictions, _agreement = _symmetrize(raw_rows, [item])
    metrics = {
        "pc1_plus": v12._paired_metrics(predictions["baseline"], predictions["pc1_plus"], 7, 4),
        "pc1_minus": v12._paired_metrics(predictions["baseline"], predictions["pc1_minus"], 7, 4),
    }
    v12._write_jsonl(run_dir / "symmetrized_scores.jsonl", sym_rows)
    v12._write_json(run_dir / "paired_metrics.json", metrics)
    v12._write_json(run_dir / "balance_validation.json", {"status": "PASS"})
    v12._write_json(run_dir / "cyclic_permutation_manifests.json", {})
    v12._write_json(run_dir / "slot_response_summary.json", {})
    v12._write_json(run_dir / "margin_analysis.json", {})
    v12._write_json(run_dir / "category_analysis.json", {})
    v12._write_json(run_dir / "prediction_distributions.json", {})
    v12._write_jsonl(
        run_dir / "directional_responses.jsonl",
        [{"item_id": item.id, "cyclic_shift": shift} for shift in range(10)],
    )
    (run_dir / "summary.md").write_text("synthetic software validation", encoding="utf-8")
    manifest = {
        "protocol": v12.PROTOCOL_ID,
        "status": "COMPLETE",
        "item_count": 1,
        "cyclic_ordering_count": 10,
        "model_revision": v12.MODEL_REVISION,
        "dataset_revision": v12.V1_DATASET_REVISION,
        "confirmatory_accessed": "NO",
        "holdout_access": "forbidden",
        "config_hash": config_hash,
        "split_manifest_sha256": split_file_hash,
        "experiment_seed": 7,
        "pc1_hash": v12.PC1_HASH,
        "layer": 17,
        "token_scope": "last_token",
        "raw_row_count": len(raw_rows),
        "raw_scores_sha256": v12._sha256_bytes(
            (run_dir / "raw_permutation_scores.jsonl").read_bytes()
        ),
        "symmetrized_scores_sha256": v12._sha256_bytes(
            (run_dir / "symmetrized_scores.jsonl").read_bytes()
        ),
        "paired_metrics_sha256": v12._sha256_bytes((run_dir / "paired_metrics.json").read_bytes()),
    }
    v12._write_json(run_dir / "manifest.json", manifest)

    report = v12.validate_q1_v1_2_run(run_dir, split_path)
    assert report["valid"] is True
    assert report["derived_artifacts_recomputed"] is True
