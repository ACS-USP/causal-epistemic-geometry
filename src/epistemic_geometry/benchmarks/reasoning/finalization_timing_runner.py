"""Serial executor for the frozen Q1 budget-by-steering schedule.

This module is intentionally a small execution boundary.  It accepts an
already materialized manifest and schedule, performs no persistence, and
returns only parsed :class:`RolloutRecord` objects.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from epistemic_geometry.analysis.finalization_timing import QWEN3_THINK_CLOSE_TOKEN_ID
from epistemic_geometry.reproducibility import canonical_json, stable_digest
from epistemic_geometry.steering.vector import vector_hash
from epistemic_geometry.types import Intervention

from .base import ReasoningView
from .families import generate_item
from .finalization_timing import NAMESPACE, validate_schedule
from .finalization_timing_journal import FinalizationTimingJournal, physical_key
from .parser import parse_family_final
from .rendering import render_reasoning
from .rollouts import RolloutRecord, generation_config_hash, rollout_record_from_output

CANDIDATE_IDENTITY_FIELDS = (
    "model_repo",
    "model_revision",
    "tokenizer_repo",
    "tokenizer_revision",
    "dtype",
    "attention_backend",
    "vector_path",
    "vector_file_sha256",
    "vector_canonical_sha256",
    "layer",
    "eta",
    "hook_scope",
    "think_close_token_id",
    "decoding_config",
)
_CANDIDATE_IDENTITY_STRING_FIELDS = {
    "model_repo",
    "model_revision",
    "tokenizer_repo",
    "tokenizer_revision",
    "dtype",
    "attention_backend",
    "vector_path",
    "vector_file_sha256",
    "vector_canonical_sha256",
    "hook_scope",
}
DECODING_CONFIG_FIELDS = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "enable_thinking",
    "prompt_mode",
    "inference_mode",
    "execution_mode",
)
SUPPORTED_HOOK_SCOPE = "sustained_current_token"


def validate_candidate_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy the complete frozen candidate execution identity.

    The candidate identity is deliberately separate from controller and journal
    provenance.  Keeping an exact schema here prevents either of those
    narrower records from silently standing in for the model/vector setup.
    """

    if not isinstance(identity, Mapping):
        raise TypeError("candidate_identity must be a mapping")
    supplied = set(identity)
    expected = set(CANDIDATE_IDENTITY_FIELDS)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing:
        raise ValueError(f"candidate_identity is missing fields: {missing}")
    if extra:
        raise ValueError(f"candidate_identity has unexpected fields: {extra}")
    for field in _CANDIDATE_IDENTITY_STRING_FIELDS:
        value = identity[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate_identity {field} must be a non-empty string")
    layer = identity["layer"]
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ValueError("candidate_identity layer must be a non-negative integer")
    eta = identity["eta"]
    if isinstance(eta, bool) or not isinstance(eta, (int, float)) or not math.isfinite(float(eta)):
        raise ValueError("candidate_identity eta must be a finite number")
    close_token_id = identity["think_close_token_id"]
    if close_token_id != QWEN3_THINK_CLOSE_TOKEN_ID:
        raise ValueError(
            "candidate_identity think_close_token_id must match the Qwen3 structural token"
        )
    decoding_config = identity["decoding_config"]
    if not isinstance(decoding_config, Mapping):
        raise ValueError("candidate_identity decoding_config must be a mapping")
    decoding_fields = set(decoding_config)
    expected_decoding_fields = set(DECODING_CONFIG_FIELDS)
    missing_decoding = sorted(expected_decoding_fields - decoding_fields)
    extra_decoding = sorted(decoding_fields - expected_decoding_fields)
    if missing_decoding:
        raise ValueError(
            f"candidate_identity decoding_config is missing fields: {missing_decoding}"
        )
    if extra_decoding:
        raise ValueError(
            f"candidate_identity decoding_config has unexpected fields: {extra_decoding}"
        )
    if identity["hook_scope"] != SUPPORTED_HOOK_SCOPE:
        raise ValueError(
            "candidate_identity hook_scope must be "
            f"{SUPPORTED_HOOK_SCOPE!r}"
        )
    return copy.deepcopy(dict(identity))


def candidate_identity_hash(identity: Mapping[str, Any]) -> str:
    """Return the deterministic hash of a validated candidate identity."""

    validated = validate_candidate_identity(identity)
    return stable_digest(
        NAMESPACE,
        "CANDIDATE-IDENTITY",
        canonical_json(validated),
    )


def _physical_generation_id(schedule_identity_hash: str) -> str:
    return stable_digest(NAMESPACE, "PHYSICAL-GENERATION", schedule_identity_hash)


def _materialize_view(
    manifest_row: Mapping[str, Any],
) -> ReasoningView:
    """Regenerate one manifest item and render its canonical view."""

    family = manifest_row["family"]
    cell = manifest_row["cell"]
    generator_seed = manifest_row["generator_seed"]
    item = generate_item(family, cell, generator_seed)
    if (
        item.latent_id != manifest_row["latent_id"]
        or item.latent_hash != manifest_row["latent_hash"]
        or item.latent_seed != manifest_row["latent_seed"]
    ):
        raise ValueError(
            "manifest item identity changed when regenerated from family/cell/generator_seed"
        )
    return render_reasoning(item, surface="canonical")


def _validate_schedule_identity(
    row: Mapping[str, Any], manifest_row: Mapping[str, Any]
) -> None:
    """Check schedule fields that the design validator cannot infer by ID."""

    for field in ("family", "cell", "latent_seed", "latent_hash", "manifest_hash"):
        if row.get(field) != manifest_row.get(field):
            raise ValueError(f"schedule {field} does not match its manifest row")
    cap = row.get("cap")
    condition = row.get("condition")
    rollout_index = row.get("rollout_index")
    latent_id = row.get("latent_id")
    expected_identity = stable_digest(
        NAMESPACE, "SCHEDULE", latent_id, cap, condition, rollout_index
    )
    if row.get("schedule_identity_hash") != expected_identity:
        raise ValueError("schedule identity hash mismatch")
    if row.get("reasoning_budget") != cap:
        raise ValueError("schedule reasoning budget does not match cap")


class SerialFinalizationTimingAdapter:
    """Execute every frozen budget row as one independent serial generation."""

    def __init__(
        self,
        backend: Any,
        manifest: Sequence[Mapping[str, Any]],
        schedule: Sequence[Mapping[str, Any]],
        *,
        intervention: Intervention | None = None,
        d75_intervention: Intervention | None = None,
        candidate_identity: Mapping[str, Any] | None = None,
        controller_provenance: Mapping[str, Any] | None = None,
        journal: FinalizationTimingJournal | None = None,
        journal_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if intervention is not None and d75_intervention is not None:
            raise ValueError("provide only one D75 intervention")
        self.backend = backend
        self.manifest = tuple(manifest)
        self.schedule = tuple(schedule)
        self.intervention = d75_intervention if d75_intervention is not None else intervention
        if self.intervention is not None and not isinstance(self.intervention, Intervention):
            raise TypeError("D75 intervention must be an Intervention")
        if candidate_identity is None:
            raise ValueError("candidate_identity is required for serial finalization timing")
        self.candidate_identity = validate_candidate_identity(candidate_identity)
        self.candidate_identity_hash = candidate_identity_hash(self.candidate_identity)
        if controller_provenance is not None and not isinstance(controller_provenance, Mapping):
            raise TypeError("controller_provenance must be a mapping")
        self.controller_provenance = (
            dict(controller_provenance) if controller_provenance is not None else None
        )
        if journal is not None and not isinstance(journal, FinalizationTimingJournal):
            raise TypeError("journal must be a FinalizationTimingJournal")
        if journal is not None and journal_identity is None:
            raise ValueError(
                "journal_identity is required when a finalization timing journal is supplied"
            )
        if journal_identity is not None and not isinstance(journal_identity, Mapping):
            raise TypeError("journal_identity must be a mapping")
        if (
            journal is not None
            and journal_identity.get("candidate_identity_hash") != self.candidate_identity_hash
        ):
            raise ValueError(
                "journal_identity candidate_identity_hash does not match "
                "the frozen candidate identity"
            )
        self.journal = journal
        self.journal_identity = dict(journal_identity) if journal_identity is not None else None

    @staticmethod
    def _config_value(config: Any, field: str) -> Any:
        if isinstance(config, Mapping):
            if field not in config:
                raise ValueError(f"backend.config is missing {field}")
            return config[field]
        try:
            return getattr(config, field)
        except AttributeError as exc:
            raise ValueError(f"backend.config is missing {field}") from exc

    def _validate_vector_file(self) -> None:
        """Verify the caller-identified vector bytes without fetching anything."""

        candidate_path = Path(self.candidate_identity["vector_path"])
        try:
            resolved_path = candidate_path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(
                "candidate_identity vector_path must resolve to a regular file"
            ) from exc
        if not resolved_path.is_file():
            raise ValueError("candidate_identity vector_path must resolve to a regular file")
        digest = hashlib.sha256()
        try:
            with resolved_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ValueError("candidate_identity vector_path could not be read") from exc
        observed = digest.hexdigest()
        if observed != self.candidate_identity["vector_file_sha256"]:
            raise ValueError(
                "candidate vector file SHA-256 does not match candidate_identity"
            )

    def _validate_live_identity(self) -> None:
        """Cross-check frozen identity against the live backend and controller."""

        self._validate_vector_file()
        provenance_fn = getattr(self.backend, "provenance", None)
        if not callable(provenance_fn):
            raise ValueError("backend.provenance() is required for serial finalization timing")
        provenance = provenance_fn()
        if not isinstance(provenance, Mapping):
            raise ValueError("backend.provenance() must return a mapping")
        provenance_fields = {
            "model_repo": "model_identifier",
            "model_revision": "model_revision",
            "tokenizer_repo": "tokenizer_identifier",
            "tokenizer_revision": "tokenizer_revision",
            "dtype": "dtype",
        }
        for candidate_field, provenance_field in provenance_fields.items():
            if provenance.get(provenance_field) != self.candidate_identity[candidate_field]:
                raise ValueError(
                    f"backend provenance {provenance_field} does not match candidate_identity"
                )
        attention_backend = provenance.get("attention_backend")
        if attention_backend is None:
            attention_backend = provenance.get("attention_implementation")
        if attention_backend != self.candidate_identity["attention_backend"]:
            raise ValueError(
                "backend provenance attention backend does not match candidate_identity"
            )

        config = getattr(self.backend, "config", None)
        if config is None:
            raise ValueError("backend.config is required for serial finalization timing")
        decoding_config = self.candidate_identity["decoding_config"]
        for field in DECODING_CONFIG_FIELDS:
            if self._config_value(config, field) != decoding_config[field]:
                raise ValueError(f"backend.config {field} does not match candidate_identity")

        tokenizer = getattr(self.backend, "tokenizer", None)
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("backend.tokenizer is required for finalization timing")
        encoded = tokenizer("</think>", add_special_tokens=False)
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("backend tokenizer did not return input_ids for </think>")
        token_ids = encoded["input_ids"]
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        nested_ids = (
            isinstance(token_ids, Sequence)
            and len(token_ids) == 1
            and isinstance(token_ids[0], Sequence)
        )
        if nested_ids:
            token_ids = token_ids[0]
        if token_ids != [self.candidate_identity["think_close_token_id"]]:
            raise ValueError(
                "backend tokenizer </think> encoding does not match candidate_identity"
            )

        has_d75 = any(row["condition"] == "D75" for row in self.schedule)
        if has_d75:
            if self.intervention is None:
                raise ValueError("D75 schedule rows require an Intervention")
            if self.controller_provenance is None:
                raise ValueError("D75 schedule rows require controller_provenance")
            file_sha = self.controller_provenance.get("vector_file_sha256")
            if not isinstance(file_sha, str) or not file_sha.strip():
                raise ValueError("controller_provenance vector_file_sha256 is required")
            if file_sha != self.candidate_identity["vector_file_sha256"]:
                raise ValueError(
                    "controller_provenance vector_file_sha256 does not match candidate_identity"
                )
        if self.intervention is None:
            return
        candidate_layer = self.candidate_identity["layer"]
        if self.intervention.layer != candidate_layer:
            raise ValueError("intervention layer does not match candidate_identity")
        if self.intervention.vector.layer != candidate_layer:
            raise ValueError("intervention vector layer does not match candidate_identity")
        if self.intervention.alpha != self.candidate_identity["eta"]:
            raise ValueError("intervention alpha does not match candidate_identity eta")
        expected_vector_hash = self.candidate_identity["vector_canonical_sha256"]
        recomputed_vector_hash = vector_hash(self.intervention.vector.values)
        if recomputed_vector_hash != expected_vector_hash:
            raise ValueError(
                "recomputed intervention vector hash does not match candidate_identity"
            )
        if self.intervention.vector.hash != expected_vector_hash:
            raise ValueError("intervention vector hash does not match candidate_identity")
        if self.intervention.token_scope != "last_token":
            raise ValueError("intervention token_scope must be last_token")

    @staticmethod
    def _rehydrate_record(
        row: Mapping[str, Any], view: ReasoningView, stored: Mapping[str, Any]
    ) -> RolloutRecord:
        """Validate and rehydrate a journal row without regenerating it."""

        record = dict(stored)
        # The answer key is deliberately not persisted by this runner.  It is
        # recovered from the already validated manifest view at resume time.
        record.setdefault("target", view.answer)
        if record["target"] != view.answer:
            raise ValueError("journaled rollout target does not match manifest view")
        if record.get("view_id") != view.view_id:
            raise ValueError("journaled rollout view identity does not match manifest view")
        try:
            parsed = parse_family_final(
                record["raw_text"],
                view.family,
                truncated=record.get("stop_reason") == "max_new_tokens",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("journaled rollout raw response is corrupt") from exc
        if (
            record.get("parsed_answer") != parsed.answer
            or record.get("parse_status") != parsed.status
        ):
            raise ValueError("journaled rollout parse result does not match raw response")
        expected_correct = parsed.valid and parsed.answer == view.answer
        if record.get("correct") is not expected_correct:
            raise ValueError("journaled rollout correctness does not match parsed response")
        config = record.get("generation_config")
        if not isinstance(config, Mapping):
            raise ValueError("journaled rollout generation config is corrupt")
        if record.get("generation_config_hash") not in (None, generation_config_hash(dict(config))):
            raise ValueError("journaled rollout generation config hash mismatch")
        try:
            return RolloutRecord.from_record(record)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("journaled rollout record is corrupt") from exc

    def _validate_journal(
        self,
        schedule: Sequence[Mapping[str, Any]],
        views: Mapping[str, ReasoningView],
    ) -> dict[tuple[str, int, str, int], RolloutRecord]:
        if self.journal is None:
            return {}
        if self.journal_identity != self.journal.identity:
            raise ValueError(
                "finalization timing journal identity does not match the frozen run identity"
            )
        by_key = {physical_key(row): row for row in schedule}
        recovered: dict[tuple[str, int, str, int], RolloutRecord] = {}
        # Scan every persisted entry before the first backend call.  Extra rows
        # from another schedule are unsafe to silently ignore.
        for key, entry in self.journal.entries.items():
            stored_schedule = entry.get("schedule")
            stored_record = entry.get("record")
            if key not in by_key or not isinstance(stored_schedule, Mapping):
                raise ValueError(f"journal contains a row outside the frozen schedule: {key}")
            row = by_key[key]
            if dict(stored_schedule) != dict(row):
                raise ValueError(f"journal schedule mismatch for physical key: {key}")
            if not isinstance(stored_record, Mapping):
                raise ValueError("journaled rollout record is corrupt")
            recovered[key] = self._rehydrate_record(row, views[row["latent_id"]], stored_record)
        return recovered

    def run(self) -> list[RolloutRecord]:
        """Validate, execute, parse, and return the raw records in schedule order."""

        # This also validates the full frozen family/cell coverage and the
        # matched seed contract before any model call is made.
        validate_schedule(self.schedule, self.manifest)
        if any(row["condition"] == "D75" for row in self.schedule) and self.intervention is None:
            raise ValueError("D75 schedule rows require an Intervention")
        self._validate_live_identity()
        manifest_by_id = {row["latent_id"]: row for row in self.manifest}
        # Validate every schedule/manifest pairing before recovering or
        # generating any row.  validate_schedule intentionally focuses on the
        # combinatorial schedule contract and cannot infer these fields.
        for row in self.schedule:
            _validate_schedule_identity(row, manifest_by_id[row["latent_id"]])
        # Materialize and validate all views before touching the backend.  This
        # also gives journal resume a trusted answer for rehydrating records.
        views = {}
        for latent_id, manifest_row in manifest_by_id.items():
            views[latent_id] = _materialize_view(manifest_row)
        recovered = self._validate_journal(self.schedule, views)
        records: list[RolloutRecord] = []
        for row in self.schedule:
            manifest_row = manifest_by_id[row["latent_id"]]
            key = physical_key(row)
            if key in recovered:
                records.append(recovered[key])
            else:
                self._execute_row(row, views[row["latent_id"]], records)
                if self.journal is not None:
                    # Keep the raw response and provenance needed to audit and
                    # rehydrate; omit the manifest answer key from disk.
                    journal_record = records[-1].to_record()
                    journal_record.pop("target", None)
                    self.journal.append(row, journal_record)
        return records

    def _execute_row(
        self,
        row: Mapping[str, Any],
        view: ReasoningView,
        records: list[RolloutRecord],
    ) -> None:
        condition = row["condition"]
        cap = int(row["cap"])
        sampling_seed = int(row["sampling_seed"])
        rollout_index = int(row["rollout_index"])
        schedule_identity_hash = str(row["schedule_identity_hash"])
        physical_id = _physical_generation_id(schedule_identity_hash)
        if condition == "D75":
            if self.intervention is None:
                raise ValueError("D75 schedule rows require an Intervention")
            # The live identity check requires this mapping and its explicit
            # vector file hash before the first treatment generation.
            provenance = self.controller_provenance
            # The context is deliberately scoped to this one generation call.
            with self.backend.steer_sustained_current_token(self.intervention):
                output = self.backend.generate_reasoning_view(
                    view, sampling_seed=sampling_seed, max_new_tokens=cap
                )
            intervention_id = self.intervention.vector_id
        elif condition == "BASELINE":
            if self.intervention is not None and not isinstance(self.intervention, Intervention):
                raise TypeError("D75 intervention must be an Intervention")
            provenance = None
            output = self.backend.generate_reasoning_view(
                view, sampling_seed=sampling_seed, max_new_tokens=cap
            )
            intervention_id = "baseline"
        else:  # defensive; validate_schedule normally catches this first
            raise ValueError(f"unsupported schedule condition: {condition}")

        metadata = dict(output.metadata)
        truncated = metadata.get("stop_reason") == "max_new_tokens"
        generation_config = {
            "surface": "canonical",
            "condition": condition,
            "max_new_tokens": cap,
            "sampling_seed": sampling_seed,
        }
        record = rollout_record_from_output(
            view,
            output,
            intervention_id=intervention_id,
            rollout_index=rollout_index,
            sampling_seed=sampling_seed,
            generation_config=generation_config,
            truncated=truncated,
        )
        record_metadata = dict(record.metadata)
        record_metadata.update(
            {
                "cap": cap,
                "reasoning_budget": cap,
                "condition": condition,
                "schedule_identity_hash": schedule_identity_hash,
                "seed_regime": row["seed_regime"],
                "controller_provenance": provenance,
                "candidate_identity": copy.deepcopy(self.candidate_identity),
                "candidate_identity_hash": self.candidate_identity_hash,
                "physical_generation_id": physical_id,
            }
        )
        records.append(
            replace(
                record,
                metadata=record_metadata,
                physical_generation_id=physical_id,
            )
        )


def run_serial_finalization_timing(
    backend: Any,
    manifest: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    intervention: Intervention | None = None,
    d75_intervention: Intervention | None = None,
    candidate_identity: Mapping[str, Any] | None = None,
    controller_provenance: Mapping[str, Any] | None = None,
    journal: FinalizationTimingJournal | None = None,
    journal_identity: Mapping[str, Any] | None = None,
) -> list[RolloutRecord]:
    """Convenience wrapper around :class:`SerialFinalizationTimingAdapter`."""

    return SerialFinalizationTimingAdapter(
        backend,
        manifest,
        schedule,
        intervention=intervention,
        d75_intervention=d75_intervention,
        candidate_identity=candidate_identity,
        controller_provenance=controller_provenance,
        journal=journal,
        journal_identity=journal_identity,
    ).run()


__all__ = [
    "CANDIDATE_IDENTITY_FIELDS",
    "SerialFinalizationTimingAdapter",
    "candidate_identity_hash",
    "run_serial_finalization_timing",
    "validate_candidate_identity",
]
