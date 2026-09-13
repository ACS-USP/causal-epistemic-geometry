"""Serial executor for the frozen Q1 budget-by-steering schedule.

This module is intentionally a small execution boundary.  It accepts an
already materialized manifest and schedule, performs no persistence, and
returns only parsed :class:`RolloutRecord` objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from epistemic_geometry.reproducibility import stable_digest
from epistemic_geometry.types import Intervention

from .base import ReasoningView
from .budget_interaction import (
    NAMESPACE,
    validate_schedule,
)
from .families import generate_item
from .rendering import render_reasoning
from .rollouts import RolloutRecord, rollout_record_from_output


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


def _default_controller_provenance(intervention: Intervention) -> dict[str, Any]:
    return {
        "vector_id": intervention.vector_id,
        "layer": intervention.layer,
        "alpha": intervention.alpha,
        "token_scope": intervention.token_scope,
    }


class SerialBudgetInteractionAdapter:
    """Execute every frozen budget row as one independent serial generation."""

    def __init__(
        self,
        backend: Any,
        manifest: Sequence[Mapping[str, Any]],
        schedule: Sequence[Mapping[str, Any]],
        *,
        intervention: Intervention | None = None,
        d75_intervention: Intervention | None = None,
        controller_provenance: Mapping[str, Any] | None = None,
    ) -> None:
        if intervention is not None and d75_intervention is not None:
            raise ValueError("provide only one D75 intervention")
        self.backend = backend
        self.manifest = tuple(manifest)
        self.schedule = tuple(schedule)
        self.intervention = d75_intervention if d75_intervention is not None else intervention
        if self.intervention is not None and not isinstance(self.intervention, Intervention):
            raise TypeError("D75 intervention must be an Intervention")
        self.controller_provenance = (
            dict(controller_provenance) if controller_provenance is not None else None
        )

    def run(self) -> list[RolloutRecord]:
        """Validate, execute, parse, and return the raw records in schedule order."""

        # This also validates the full frozen family/cell coverage and the
        # matched seed contract before any model call is made.
        validate_schedule(self.schedule, self.manifest)
        if any(row["condition"] == "D75" for row in self.schedule) and self.intervention is None:
            raise ValueError("D75 schedule rows require an Intervention")
        manifest_by_id = {row["latent_id"]: row for row in self.manifest}
        records: list[RolloutRecord] = []
        for row in self.schedule:
            manifest_row = manifest_by_id[row["latent_id"]]
            _validate_schedule_identity(row, manifest_row)
            view = _materialize_view(manifest_row)
            self._execute_row(row, view, records)
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
            provenance = self.controller_provenance or _default_controller_provenance(
                self.intervention
            )
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


def run_serial_budget_interaction(
    backend: Any,
    manifest: Sequence[Mapping[str, Any]],
    schedule: Sequence[Mapping[str, Any]],
    *,
    intervention: Intervention | None = None,
    d75_intervention: Intervention | None = None,
    controller_provenance: Mapping[str, Any] | None = None,
) -> list[RolloutRecord]:
    """Convenience wrapper around :class:`SerialBudgetInteractionAdapter`."""

    return SerialBudgetInteractionAdapter(
        backend,
        manifest,
        schedule,
        intervention=intervention,
        d75_intervention=d75_intervention,
        controller_provenance=controller_provenance,
    ).run()


__all__ = ["SerialBudgetInteractionAdapter", "run_serial_budget_interaction"]
