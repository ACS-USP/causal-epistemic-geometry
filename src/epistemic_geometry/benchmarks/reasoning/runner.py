"""Baseline-only Q1 V3 calibration runner.

The runner is intentionally separate from the ordinary Q1 steering runner. It
loads one frozen backend, evaluates only procedural baseline rollouts, and
never accepts an intervention. Real HuggingFace loading is still protected by
the repository's RunPod/HF_HOME guard.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from epistemic_geometry.backends.base import build_backend
from epistemic_geometry.config import RunConfig
from epistemic_geometry.reproducibility import canonical_json, git_metadata, stable_digest

from .engines import (
    BATCHED_REASONING,
    MAX_BUDGET_PREFIX_REUSE,
    REASONING_ENGINE_VERSION,
    SERIAL_REASONING_REFERENCE,
    SUPPORTED_REASONING_ENGINES,
    derive_budget_outputs,
    physical_generation_id,
)
from .journal import PhysicalGenerationJournal
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


def _source_commit() -> str | None:
    """Prefer the commit explicitly supplied by the canonical source host."""

    explicit = os.environ.get("CEG_SOURCE_COMMIT")
    if explicit:
        return explicit
    return git_metadata(Path.cwd()).get("git_commit")


def _journal_identity(
    *,
    config: RunConfig,
    payload: dict[str, Any],
    manifest_path: Path,
    selected_keys: list[str],
    phase: str,
    rollout_count: int,
    inference_engine: str,
) -> dict[str, Any]:
    """Build the immutable identity that makes resume incompatibilities loud."""

    return {
        "phase": phase,
        "selected_manifest_keys": list(selected_keys),
        "manifest_hash": stable_digest("Q1-V3-STAGE-A-MANIFEST", canonical_json(payload)),
        "manifest_path": str(manifest_path),
        "config_hash": stable_digest("q1-v3-config", canonical_json(config.as_dict())),
        "rollout_count": rollout_count,
        "inference_engine": inference_engine,
        "inference_engine_version": REASONING_ENGINE_VERSION,
        "source_commit": _source_commit(),
        "model_id": config.backend.model_id,
        "model_revision": config.backend.model_revision,
        "tokenizer_id": config.backend.tokenizer_id,
        "tokenizer_revision": config.backend.tokenizer_revision,
        "enable_thinking": config.backend.enable_thinking,
        "do_sample": config.backend.do_sample,
        "temperature": config.backend.temperature,
        "top_p": config.backend.top_p,
        "top_k": config.backend.top_k,
        "min_p": config.backend.min_p,
        "max_physical_budget": max(
            int(budget)
            for budget in config.q1_v3.get(
                "reasoning_budgets", [config.backend.max_new_tokens]
            )
        ),
    }


def _validate_calibration_config(config: RunConfig) -> tuple[str, int]:
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
        raise ValueError("rollout_count must be positive")
    return phase, rollout_count


def _selected_keys(manifests: dict[str, Any], manifest_key: str | None) -> list[str]:
    selected = [manifest_key] if manifest_key else sorted(manifests)
    if any(key not in manifests for key in selected):
        raise KeyError(f"manifest key not found: {manifest_key}")
    return selected


def _records_to_artifacts(
    *,
    output: Path,
    phase: str,
    config: RunConfig,
    manifest_path: Path,
    selected_keys: list[str],
    outcomes: list[dict[str, Any]],
    all_records: list[RolloutRecord],
    backend: Any,
    inference_engine: str,
    physical_generations: int,
    rollout_count: int,
    journal: PhysicalGenerationJournal | None = None,
) -> Path:
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
            "status": "COMPLETE",
            "phase": phase,
            "model_outcomes": True,
            "steering_outcomes": False,
            "confirmatory_accessed": False,
            "inference_engine": inference_engine,
            "inference_engine_version": REASONING_ENGINE_VERSION,
            "inference_settings": {
                "batch_size": int(
                    config.q1_v3.get("batch_size", config.backend.batch_size)
                ),
                "max_prefill_tokens": int(
                    config.q1_v3.get("max_prefill_tokens", config.backend.max_prefill_tokens)
                ),
                "padding_side": config.backend.padding_side,
                "enable_thinking": config.backend.enable_thinking,
                "do_sample": config.backend.do_sample,
                "temperature": config.backend.temperature,
                "top_p": config.backend.top_p,
                "top_k": config.backend.top_k,
                "min_p": config.backend.min_p,
            },
            "config_hash": stable_digest("q1-v3-config", canonical_json(config.as_dict())),
            "source_manifest": str(manifest_path),
            "source_manifest_hash": stable_digest(
                "Q1-V3-STAGE-A-MANIFEST", manifest_path.read_text(encoding="utf-8")
            ),
            "manifest_keys": selected_keys,
            "rollout_count": rollout_count,
            "rollout_rows": len(all_records),
            "physical_generation_count": physical_generations,
            "scientific_budget_outcomes": len(all_records),
            "model_provenance": provenance,
            "source_commit": _source_commit(),
            "physical_journal": str(journal.path) if journal is not None else None,
            "physical_journal_quarantined_tail": (
                journal.quarantined_tail if journal is not None else None
            ),
        },
    )
    return output


def _run_serial_reasoning(
    *,
    config: RunConfig,
    payload: dict[str, Any],
    manifest_path: Path,
    selected_keys: list[str],
    output: Path,
    phase: str,
    rollout_count: int,
    max_items: int | None,
    backend: Any,
) -> Path:
    """Known-correct one-request-at-a-time reference implementation."""

    all_records: list[RolloutRecord] = []
    outcomes: list[dict[str, Any]] = []
    for key in selected_keys:
        split = ReasoningSplit.from_record(payload["manifests"][key], development=True)
        items = split.items[:max_items] if max_items is not None else split.items
        surfaces = ("canonical",) if phase == "stage_a_screen" else ("canonical", "surface_twin")
        records: list[RolloutRecord] = []
        budget = int(split.reasoning_budget or config.backend.max_new_tokens)
        generation_config = _generation_config(config, budget)
        for item in items:
            for surface in surfaces:
                view = render_reasoning(item, surface=surface)
                for rollout_index in range(rollout_count):
                    seed = rollout_seed(
                        config.experiment.seed,
                        view.latent_id,
                        "baseline",
                        rollout_index,
                        regime="independent",
                    )
                    backend_output = backend.generate_reasoning_view(
                        view, sampling_seed=seed, max_new_tokens=budget
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
    result = _records_to_artifacts(
        output=output,
        phase=phase,
        config=config,
        manifest_path=manifest_path,
        selected_keys=selected_keys,
        outcomes=outcomes,
        all_records=all_records,
        backend=backend,
        inference_engine=SERIAL_REASONING_REFERENCE,
        physical_generations=len(all_records),
        rollout_count=rollout_count,
    )
    if max_items is not None:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "PARTIAL_ENGINEERING_RUN"
        _atomic_json(output / "manifest.json", manifest)
    return result


def _run_max_budget_prefix_reuse(
    *,
    config: RunConfig,
    payload: dict[str, Any],
    manifest_path: Path,
    selected_keys: list[str],
    output: Path,
    phase: str,
    rollout_count: int,
    max_items: int | None,
    backend: Any,
    batched: bool = False,
) -> Path:
    """Generate one max-budget trajectory and derive each paired budget row.

    ``batched=True`` changes only physical execution: the same ordered task
    list is passed through the backend's length-aware batched generator.  The
    budget prefixes and parser remain exactly the same as in the reference
    prefix-reuse engine.
    """

    if not hasattr(backend, "tokenizer"):
        raise TypeError("max-budget prefix reuse requires a tokenizer-backed backend")
    manifests = payload["manifests"]
    grouped: dict[tuple[str, str], dict[int, tuple[str, ReasoningSplit]]] = {}
    for key in selected_keys:
        split = ReasoningSplit.from_record(manifests[key], development=True)
        budget = int(split.reasoning_budget or config.backend.max_new_tokens)
        grouped.setdefault((split.family, split.cell), {})[budget] = (key, split)

    journal = (
        PhysicalGenerationJournal(
            output / "physical_journal.jsonl",
            identity=_journal_identity(
                config=config,
                payload=payload,
                manifest_path=manifest_path,
                selected_keys=selected_keys,
                phase=phase,
                rollout_count=rollout_count,
                inference_engine=MAX_BUDGET_PREFIX_REUSE,
            ),
        )
        if phase == "stage_a_screen" and not batched
        else None
    )
    if journal is not None:
        _atomic_json(
            output / "manifest.json",
            {
                "status": "RUNNING",
                "phase": phase,
                "model_outcomes": True,
                "steering_outcomes": False,
                "confirmatory_accessed": False,
                "inference_engine": MAX_BUDGET_PREFIX_REUSE,
                "inference_engine_version": REASONING_ENGINE_VERSION,
                "source_commit": _source_commit(),
                "physical_journal": str(journal.path),
                "physical_journal_identity_hash": journal.identity_digest,
                "manifest_keys": selected_keys,
            },
        )
    physical_generations = 0
    in_memory_rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for group_key in sorted(grouped):
        budget_rows = grouped[group_key]
        ordered_budgets = sorted(budget_rows)
        source_budget = max(ordered_budgets)
        _source_key, source_split = budget_rows[source_budget]
        source_items = (
            source_split.items[:max_items] if max_items is not None else source_split.items
        )
        source_ids = tuple(item.latent_id for item in source_items)
        for _budget, (_key, split) in budget_rows.items():
            if tuple(item.latent_id for item in split.items[: len(source_items)]) != source_ids:
                raise ValueError(
                    "max-budget prefix reuse requires paired latent IDs across budget manifests"
                )
        surfaces = ("canonical",) if phase == "stage_a_screen" else ("canonical", "surface_twin")
        tasks: list[tuple[Any, int]] = []
        task_rollouts: list[int] = []
        for item in source_items:
            for surface in surfaces:
                view = render_reasoning(item, surface=surface)
                for rollout_index in range(rollout_count):
                    seed = rollout_seed(
                        config.experiment.seed,
                        view.latent_id,
                        "baseline",
                        rollout_index,
                        regime="independent",
                    )
                    tasks.append((view, seed))
                    task_rollouts.append(rollout_index)

        if batched:
            if not hasattr(backend, "generate_reasoning_batch"):
                raise TypeError("batched reasoning requires a batch-capable backend")
            source_outputs = backend.generate_reasoning_batch(
                tasks,
                max_new_tokens=source_budget,
                batch_size=int(config.q1_v3.get("batch_size", config.backend.batch_size)),
                max_prefill_tokens=int(
                    config.q1_v3.get("max_prefill_tokens", config.backend.max_prefill_tokens)
                ),
            )
        else:
            if journal is None:
                source_outputs = [
                    backend.generate_reasoning_view(
                        view,
                        sampling_seed=seed,
                        max_new_tokens=source_budget,
                    )
                    for view, seed in tasks
                ]
            else:
                # Generate one row at a time so every completed trajectory is
                # journaled before the next model call begins.
                source_outputs = [None] * len(tasks)

        for task_position, ((view, seed), task_rollout_index) in enumerate(
            zip(tasks, task_rollouts, strict=True)
        ):
            if journal is not None and journal.has((view.latent_id, task_rollout_index)):
                continue
            source_output = source_outputs[task_position]
            if source_output is None:
                source_output = backend.generate_reasoning_view(
                    view,
                    sampling_seed=seed,
                    max_new_tokens=source_budget,
                )
            if source_output is None:
                raise RuntimeError("missing source output for an incomplete journal row")
            physical_generations += 1
            physical_id = physical_generation_id(
                view_id=view.view_id,
                sampling_seed=seed,
                source_max_budget=source_budget,
            )
            derived = derive_budget_outputs(
                source_output,
                view_id=view.view_id,
                sampling_seed=seed,
                source_max_budget=source_budget,
                budgets=ordered_budgets,
                decode_tokens=lambda ids: backend.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
            )
            derived_records: dict[str, dict[str, Any]] = {}
            for budget in ordered_budgets:
                key, _split = budget_rows[budget]
                generation_config = _generation_config(config, budget)
                record = rollout_record_from_output(
                    view,
                    derived[budget],
                    intervention_id="baseline",
                    rollout_index=task_rollout_index,
                    sampling_seed=seed,
                    generation_config=generation_config,
                )
                if record.physical_generation_id != physical_id:
                    raise RuntimeError("physical generation provenance mismatch")
                derived_records[key] = record.to_record()
            in_memory_rows[(view.latent_id, task_rollout_index)] = derived_records
            if journal is not None:
                    journal.append(
                    {
                        "latent_id": view.latent_id,
                        "view_id": view.view_id,
                        "family": view.family,
                        "cell": view.cell,
                        "target": view.answer,
                        "rollout_index": task_rollout_index,
                        "sampling_seed": seed,
                        "physical_generation_id": physical_id,
                        "source_max_budget": source_budget,
                        "source_raw_text": source_output.raw_output,
                        "source_token_ids": list(
                            source_output.metadata.get("generated_token_ids", ())
                        ),
                        "source_metadata": dict(source_output.metadata),
                        "derived_records": derived_records,
                        }
                    )

    # Reconstruct output ordering from the deterministic task plan.  This also
    # makes a resumed run byte-for-byte stable relative to an uninterrupted run.
    all_records: list[RolloutRecord] = []
    records_by_key: dict[str, list[RolloutRecord]] = {key: [] for key in selected_keys}
    for group_key in sorted(grouped):
        budget_rows = grouped[group_key]
        ordered_budgets = sorted(budget_rows)
        source_budget = max(ordered_budgets)
        source_split = budget_rows[source_budget][1]
        source_items = (
            source_split.items[:max_items] if max_items is not None else source_split.items
        )
        surfaces = (
            ("canonical",)
            if phase == "stage_a_screen"
            else ("canonical", "surface_twin")
        )
        for item in source_items:
            for surface in surfaces:
                view = render_reasoning(item, surface=surface)
                for rollout_index in range(rollout_count):
                    if journal is not None:
                        journal_row = journal.get((view.latent_id, rollout_index))
                        if journal_row is None:
                            raise RuntimeError(
                                "journal missing completed physical key "
                                f"{(view.latent_id, rollout_index)}"
                            )
                        derived_records = journal_row["derived_records"]
                    else:
                        derived_records = in_memory_rows.get((view.latent_id, rollout_index))
                        if derived_records is None:
                            raise RuntimeError(
                                "in-memory results missing physical key "
                                f"{(view.latent_id, rollout_index)}"
                            )
                    for budget in ordered_budgets:
                        key = budget_rows[budget][0]
                        record = RolloutRecord.from_record(derived_records[key])
                        records_by_key[key].append(record)
                        all_records.append(record)

    if journal is not None:
        physical_generations = len(journal.rows)

    outcomes: list[dict[str, Any]] = []
    for key in selected_keys:
        records = records_by_key[key]
        if not records:
            raise ValueError(f"no records generated for manifest key {key}")
        split = ReasoningSplit.from_record(manifests[key], development=True)
        budget = int(split.reasoning_budget or config.backend.max_new_tokens)
        summary = summarize_rollouts(records)
        summary.update(
            {
                "family": split.family,
                "cell": split.cell,
                "reasoning_budget": budget,
                "manifest_key": key,
                "phase": phase,
                "n_items_evaluated": len(split.items[:max_items])
                if max_items is not None
                else len(split.items),
            }
        )
        outcomes.append(summary)
    result = _records_to_artifacts(
        output=output,
        phase=phase,
        config=config,
        manifest_path=manifest_path,
        selected_keys=selected_keys,
        outcomes=outcomes,
        all_records=all_records,
        backend=backend,
        inference_engine=BATCHED_REASONING if batched else MAX_BUDGET_PREFIX_REUSE,
        physical_generations=physical_generations,
        rollout_count=rollout_count,
        journal=journal,
    )
    if max_items is not None:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        manifest["status"] = "PARTIAL_ENGINEERING_RUN"
        _atomic_json(output / "manifest.json", manifest)
    return result


def run_baseline_calibration(
    config: RunConfig,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    manifest_key: str | None = None,
    max_items: int | None = None,
    inference_engine: str | None = None,
) -> Path:
    """Run Stage A or B baseline rollouts and write auditable raw artifacts."""

    phase, rollout_count = _validate_calibration_config(config)
    selected_engine = str(
        inference_engine or config.q1_v3.get("inference_engine", SERIAL_REASONING_REFERENCE)
    )
    if selected_engine not in SUPPORTED_REASONING_ENGINES:
        raise ValueError(
            f"unsupported Q1 V3 inference_engine {selected_engine!r}; "
            f"choose one of {sorted(SUPPORTED_REASONING_ENGINES)}"
        )
    payload = _load_manifest(Path(manifest_path))
    manifests = payload["manifests"]
    selected_keys = _selected_keys(manifests, manifest_key)

    backend = build_backend(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive")
    if selected_engine in {MAX_BUDGET_PREFIX_REUSE, BATCHED_REASONING}:
        return _run_max_budget_prefix_reuse(
            config=config,
            payload=payload,
            manifest_path=Path(manifest_path),
            selected_keys=selected_keys,
            output=output,
            phase=phase,
            rollout_count=rollout_count,
            max_items=max_items,
            backend=backend,
            batched=selected_engine == BATCHED_REASONING,
        )
    return _run_serial_reasoning(
        config=config,
        payload=payload,
        manifest_path=Path(manifest_path),
        selected_keys=selected_keys,
        output=output,
        phase=phase,
        rollout_count=rollout_count,
        max_items=max_items,
        backend=backend,
    )
