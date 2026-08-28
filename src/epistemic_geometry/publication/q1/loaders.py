"""Read-only, hash-validating loaders for the frozen Q1 publication sources."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FIGURE_SPEC_PATH = ROOT / "manuscript/figures/paper1/FIGURE_SPEC.json"
FORBIDDEN_PARTS = {"q2_v4_1_semantic_execution", "q2-v4-1-semantic-execution"}
REQUIRED_CONDITIONS = {
    "BASELINE",
    "TEXTUAL_CAREFUL",
    "MEANINGFUL_FIXED",
    "RANDOM_R0",
    "RANDOM_R1",
    "RANDOM_R2",
    "RANDOM_R3",
}
JOURNAL_FIELDS = {
    "model_role",
    "item_id",
    "condition",
    "rollout_index",
    "correct",
    "commitment_valid",
    "semantic_evaluable",
    "generated_token_count",
    "retry_count",
    "seed",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    lowered = {part.lower() for part in relative.parts}
    if lowered & FORBIDDEN_PARTS or any("q2_v4_1_semantic" in part for part in lowered):
        raise RuntimeError("Q1 publication firewall rejected a Q2 semantic source")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise RuntimeError("publication source escaped repository root")
    return path


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads(_safe_path(relative_path).read_text())


def load_csv(relative_path: str) -> list[dict[str, str]]:
    with _safe_path(relative_path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def figure_spec() -> dict[str, Any]:
    return json.loads(FIGURE_SPEC_PATH.read_text())


def expected_source_hashes() -> dict[str, str]:
    return dict(figure_spec()["expected_source_sha256"])


def validate_frozen_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected in expected_source_hashes().items():
        path = _safe_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Q1 source: {relative_path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen Q1 source hash mismatch for {relative_path}: {actual} != {expected}"
            )
        observed[relative_path] = actual
    return observed


def holdout_item_ids() -> list[str]:
    manifest = load_json("review/q1_confirmatory_fixed_controllers/HOLDOUT_CONTENT_MANIFEST.json")
    if manifest.get("role") != "Q1_FIXED_CONTROLLER_CONFIRMATORY_HOLDOUT":
        raise RuntimeError("holdout manifest role mismatch")
    items = [str(item["item_id"]) for item in manifest["items"]]
    if len(items) != 57 or len(set(items)) != 57:
        raise RuntimeError("confirmatory holdout must contain exactly 57 unique items")
    return items


def load_confirmatory_journal(model_role: str) -> list[dict[str, Any]]:
    if model_role not in {"Qwen", "Ministral"}:
        raise ValueError(f"unknown confirmatory model role: {model_role}")
    relative_path = (
        "review/q1_confirmatory_fixed_controllers/journal_qwen.jsonl"
        if model_role == "Qwen"
        else "review/q1_confirmatory_fixed_controllers/journal_ministral.jsonl"
    )
    rows: list[dict[str, Any]] = []
    with _safe_path(relative_path).open() as handle:
        for line in handle:
            source = json.loads(line)
            missing = JOURNAL_FIELDS - source.keys()
            if missing:
                raise RuntimeError(f"confirmatory row missing fields: {sorted(missing)}")
            # Intentionally never carry raw text, parsed answers, or token IDs into
            # the publication data path.
            rows.append({field: source[field] for field in JOURNAL_FIELDS})
    _validate_confirmatory_rows(rows, model_role=model_role)
    return rows


def _validate_confirmatory_rows(rows: list[dict[str, Any]], *, model_role: str) -> None:
    if len(rows) != 798:
        raise RuntimeError(f"{model_role} confirmatory journal must contain 798 rows")
    keys = [(str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"{model_role} confirmatory journal contains duplicate keys")
    if {str(row["model_role"]) for row in rows} != {model_role}:
        raise RuntimeError(f"{model_role} journal mixes model roles")
    if {str(row["condition"]) for row in rows} != REQUIRED_CONDITIONS:
        raise RuntimeError(f"{model_role} confirmatory condition set mismatch")
    if {int(row["rollout_index"]) for row in rows} != {0, 1}:
        raise RuntimeError(f"{model_role} confirmatory rollout set mismatch")
    expected_items = set(holdout_item_ids())
    if {str(row["item_id"]) for row in rows} != expected_items:
        raise RuntimeError(f"{model_role} confirmatory item set mismatch")
    if any(int(row["retry_count"]) != 0 for row in rows):
        raise RuntimeError(f"{model_role} confirmatory journal contains a scientific retry")


def validate_controller_identities() -> dict[str, str]:
    lock = load_json("review/q1_confirmatory_fixed_controllers/CONTROLLER_IDENTITY_LOCK.json")
    controllers = lock["controllers"]
    expected = {
        "Qwen": "e7bf23a75e20aa02cf87587c8094a7b93b8d5d9eaeb820a3bf332a5e98931838",
        "Ministral": "0c467b7a452619d058afb07c96fd0cd8e20abb19a58d89674ab0a42e00ef2b94",
    }
    observed = {model: str(controllers[model]["vector_hash"]) for model in expected}
    if observed != expected:
        raise RuntimeError("confirmatory controller identity mismatch")
    if lock.get("holdout_activations_used") is not False:
        raise RuntimeError("confirmatory controller lock violates holdout firewall")
    return observed


def load_sources() -> dict[str, Any]:
    """Load all figure inputs after validating the complete frozen source set."""

    source_hashes = validate_frozen_sources()
    validate_controller_identities()
    return {
        "source_hashes": source_hashes,
        "spec": figure_spec(),
        "item_ids": holdout_item_ids(),
        "qwen_rows": load_confirmatory_journal("Qwen"),
        "ministral_rows": load_confirmatory_journal("Ministral"),
        "confirmatory": load_json(
            "review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json"
        ),
        "analysis_lock": load_json("review/q1_confirmatory_fixed_controllers/ANALYSIS_LOCK.json"),
        "gate4": load_json("review/micro_q1/ESTIMANDS.json"),
        "gate5": load_json("review/gate5_source_duration/ESTIMANDS.json"),
        "gate7": load_json("review/gate7_fresh_l27_replication/ESTIMANDS.json"),
        "gate8_doses": load_csv("review/gate8_l27_dose_calibration/DOSE_SUMMARY.csv"),
        "gate9": load_json("review/gate9_selected_d75_evaluation/ESTIMANDS.json"),
        "gate10": load_json("review/gate10_cross_domain_charcount/ESTIMANDS.json"),
        "gate13_1": load_json("review/gate13_1_all_layer_causal_atlas/ESTIMANDS.json"),
        "loo": load_csv("review/q1_confirmatory_fixed_controllers/LOO_SENSITIVITY.csv"),
        "invalidity": load_json("manuscript/data/posthoc_ministral_invalidity_aggregate.json"),
    }


__all__ = [
    "FIGURE_SPEC_PATH",
    "ROOT",
    "expected_source_hashes",
    "figure_spec",
    "holdout_item_ids",
    "load_confirmatory_journal",
    "load_sources",
    "sha256",
    "validate_controller_identities",
    "validate_frozen_sources",
]
