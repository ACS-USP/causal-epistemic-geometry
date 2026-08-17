"""Baseline-only Q1 V3 calibration runner.

The runner is intentionally separate from the ordinary Q1 steering runner. It
loads one frozen backend, evaluates only procedural baseline rollouts, and
never accepts an intervention. Real HuggingFace loading is still protected by
the repository's RunPod/HF_HOME guard.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.backends.base import build_backend
from epistemic_geometry.config import RunConfig
from epistemic_geometry.reproducibility import canonical_json, stable_digest

from .base import ReasoningView
from .rendering import render_reasoning
from .rollouts import (
    RolloutRecord,
    rollout_record_from_output,
    rollout_seed,
)
from .splits import ReasoningSplit


def _surface(record: RolloutRecord) -> str:
    return record.view_id.rsplit(":", 1)[-1]


def summarize_rollouts(records: list[RolloutRecord]) -> dict[str, Any]:
    """Compute calibration outcomes without hiding parse failures."""

    if not records:
        raise ValueError("cannot summarize an empty rollout collection")
    parse_success = sum(record.parse_status == "OK" for record in records) / len(records)
    by_view = {_surface(record) for record in records}
    primary_records = (
        [record for record in records if _surface(record) == "canonical"]
        if "canonical" in by_view and "surface_twin" in by_view
        else records
    )
    mean_accuracy = sum(record.correct for record in primary_records) / len(primary_records)
    seed_accuracy: dict[int, list[bool]] = defaultdict(list)
    for record in primary_records:
        seed_accuracy[record.rollout_index].append(record.correct)
    seed_values = [sum(values) / len(values) for _, values in sorted(seed_accuracy.items())]
    outcome: dict[str, Any] = {
        "n_rollouts": len(records),
        "n_latents": len({record.latent_id for record in records}),
        "parse_success": parse_success,
        "mean_accuracy": mean_accuracy,
        "seed_accuracy": seed_values,
        "seed_accuracy_sd": (
            math.sqrt(sum((value - mean_accuracy) ** 2 for value in seed_values) / len(seed_values))
            if seed_values
            else float("nan")
        ),
        "parse_status_counts": {
            status: sum(record.parse_status == status for record in records)
            for status in sorted({record.parse_status for record in records})
        },
    }
    if {"canonical", "surface_twin"}.issubset(by_view):
        canonical = {
            (record.latent_id, record.rollout_index): record
            for record in records
            if _surface(record) == "canonical"
        }
        twin = {
            (record.latent_id, record.rollout_index): record
            for record in records
            if _surface(record) == "surface_twin"
        }
        keys = sorted(set(canonical) & set(twin))
        outcome["canonical_accuracy"] = sum(canonical[key].correct for key in keys) / len(keys)
        outcome["twin_accuracy"] = sum(twin[key].correct for key in keys) / len(keys)
        valid_keys = [
            key
            for key in keys
            if canonical[key].parsed_answer is not None and twin[key].parsed_answer is not None
        ]
        outcome["twin_agreement_pairs"] = len(valid_keys)
        outcome["twin_agreement"] = (
            sum(canonical[key].parsed_answer == twin[key].parsed_answer for key in valid_keys)
            / len(valid_keys)
            if valid_keys
            else float("nan")
        )
    return outcome


def _generation_config(config: RunConfig, budget: int) -> dict[str, Any]:
    return {
        "enable_thinking": config.backend.enable_thinking,
        "do_sample": config.backend.do_sample,
        "temperature": config.backend.temperature,
        "top_p": config.backend.top_p,
        "top_k": config.backend.top_k,
        "min_p": config.backend.min_p,
        "max_new_tokens": budget,
        "prompt_mode": config.backend.prompt_mode,
        "model_id": config.backend.model_id,
        "model_revision": config.backend.model_revision,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifests = payload.get("manifests")
    if not isinstance(manifests, dict) or not manifests:
        raise ValueError("calibration manifest must contain a non-empty manifests mapping")
    return payload


def run_baseline_calibration(
    config: RunConfig,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    manifest_key: str | None = None,
    max_items: int | None = None,
) -> Path:
    """Run Stage A or B baseline rollouts and write auditable raw artifacts."""

    if config.backend.type != "huggingface":
        raise ValueError("Q1 V3 calibration requires backend.type=huggingface")
    if config.experiment.stage != "development":
        raise ValueError("Q1 V3 calibration is development-only")
    if config.steering.enabled:
        raise ValueError("Q1 V3 calibration is baseline-only; steering must be disabled")
    phase = str(config.q1_v3.get("phase", ""))
    if phase not in {"stage_a_screen", "stage_b_calibration"}:
        raise ValueError("q1_v3.phase must be stage_a_screen or stage_b_calibration")
    regime = str(config.q1_v3.get("seed_regime", "independent"))
    if regime != "independent":
        raise ValueError("calibration requires independent rollout seeds")
    rollout_count = int(config.q1_v3.get("rollout_count", 2 if phase == "stage_a_screen" else 4))
    if rollout_count <= 0:
        raise ValueError("q1_v3.rollout_count must be positive")

    payload = _load_manifest(Path(manifest_path))
    manifests = payload["manifests"]
    selected_keys = [manifest_key] if manifest_key else sorted(manifests)
    if any(key not in manifests for key in selected_keys):
        raise KeyError(f"manifest key not found: {manifest_key}")

    backend = build_backend(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_records: list[RolloutRecord] = []
    outcomes: list[dict[str, Any]] = []
    for key in selected_keys:
        split = ReasoningSplit.from_record(manifests[key], development=True)
        items = split.items[:max_items] if max_items is not None else split.items
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive")
        views: list[ReasoningView] = []
        surfaces = ("canonical",) if phase == "stage_a_screen" else (
            "canonical",
            "surface_twin",
        )
        for item in items:
            views.extend(render_reasoning(item, surface=surface) for surface in surfaces)
        records: list[RolloutRecord] = []
        budget = int(split.reasoning_budget or config.backend.max_new_tokens)
        generation_config = _generation_config(config, budget)
        for view in views:
            for rollout_index in range(rollout_count):
                seed = rollout_seed(
                    config.experiment.seed,
                    view.latent_id,
                    "baseline",
                    rollout_index,
                    regime=regime,
                )
                backend_output = backend.generate_reasoning_view(
                    view,
                    sampling_seed=seed,
                    max_new_tokens=budget,
                )
                record = rollout_record_from_output(
                    view,
                    backend_output,
                    intervention_id="baseline",
                    rollout_index=rollout_index,
                    sampling_seed=seed,
                    generation_config=generation_config,
                )
                records.append(record)
                all_records.append(record)
        summary = summarize_rollouts(records)
        summary.update(
            {
                "family": split.family,
                "cell": split.cell,
                "reasoning_budget": budget,
                "manifest_key": key,
                "phase": phase,
                "n_items_evaluated": len(items),
            }
        )
        outcomes.append(summary)

    rows_path = output / "rollouts.jsonl"
    rows_tmp = rows_path.with_suffix(".jsonl.tmp")
    rows_tmp.write_text(
        "".join(json.dumps(record.to_record(), sort_keys=True) + "\n" for record in all_records),
        encoding="utf-8",
    )
    rows_tmp.replace(rows_path)
    _atomic_json(output / "outcomes.json", {"phase": phase, "outcomes": outcomes})
    provenance = getattr(backend, "provenance", lambda: {})()
    _atomic_json(
        output / "manifest.json",
        {
            "status": "COMPLETE" if max_items is None else "PARTIAL_ENGINEERING_RUN",
            "phase": phase,
            "model_outcomes": True,
            "steering_outcomes": False,
            "confirmatory_accessed": False,
            "config_hash": stable_digest("q1-v3-config", canonical_json(config.as_dict())),
            "source_manifest": str(Path(manifest_path)),
            "manifest_keys": selected_keys,
            "rollout_count": rollout_count,
            "rollout_rows": len(all_records),
            "model_provenance": provenance,
        },
    )
    return output
