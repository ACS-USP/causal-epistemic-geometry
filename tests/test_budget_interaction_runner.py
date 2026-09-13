from __future__ import annotations

import numpy as np
import pytest

from epistemic_geometry.benchmarks.reasoning.budget_interaction import (
    build_manifest,
    build_schedule,
)
from epistemic_geometry.benchmarks.reasoning.budget_interaction_runner import (
    run_serial_budget_interaction,
)
from epistemic_geometry.types import BackendOutput, Intervention, SteeringVector


def _intervention() -> Intervention:
    vector = SteeringVector(np.ones(3), 0, "fake", "none", hash="fake-vector")
    return Intervention(0, 0.75, "fake-vector", "last_token", vector)


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, bool]] = []
        self.contexts: list[tuple[str, bool]] = []
        self.active = False

    def steer_sustained_current_token(self, intervention):
        class Context:
            def __enter__(_self):
                assert not self.active
                self.active = True
                self.contexts.append((intervention.vector_id, True))

            def __exit__(_self, *_args):
                assert self.active
                self.active = False

        return Context()

    def generate_reasoning_view(self, view, *, sampling_seed, max_new_tokens):
        assert not self.active or view is not None
        self.calls.append((view.view_id, sampling_seed, max_new_tokens, self.active))
        return BackendOutput(
            raw_output="partial",
            metadata={"stop_reason": "max_new_tokens", "generation_seed": sampling_seed},
        )


def test_runner_is_serial_and_scopes_d75_contexts() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    backend = _FakeBackend()
    records = run_serial_budget_interaction(
        backend,
        manifest,
        schedule,
        intervention=_intervention(),
        controller_provenance={"controller": "frozen-d75"},
    )

    assert len(records) == len(schedule)
    assert [call[0] for call in backend.calls] == [
        f"{row['latent_id']}:canonical" for row in schedule
    ]
    assert [call[2] for call in backend.calls] == [row["cap"] for row in schedule]
    assert [call[1] for call in backend.calls] == [row["sampling_seed"] for row in schedule]
    assert [call[3] for call in backend.calls] == [row["condition"] == "D75" for row in schedule]
    assert len(backend.contexts) == sum(row["condition"] == "D75" for row in schedule)
    assert not backend.active
    assert all(record.parse_status == "TRUNCATED_NO_FINAL" for record in records)
    assert all(record.metadata["condition"] in {"BASELINE", "D75"} for record in records)
    assert all(
        record.metadata["cap"] == record.generation_config["max_new_tokens"]
        for record in records
    )
    assert all(record.metadata["seed_regime"] == "MATCHED" for record in records)
    assert all(record.metadata["physical_generation_id"] for record in records)
    assert len({record.metadata["physical_generation_id"] for record in records}) == len(records)
    assert all(
        record.metadata["controller_provenance"] == {"controller": "frozen-d75"}
        for record in records
        if record.metadata["condition"] == "D75"
    )


def test_runner_rejects_tampered_schedule_identity_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    schedule[0] = {**schedule[0], "family": "FSM-R"}
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="schedule family"):
        run_serial_budget_interaction(backend, manifest, schedule, intervention=_intervention())
    assert backend.calls == []


def test_runner_requires_d75_intervention_before_generation() -> None:
    manifest = build_manifest(n_per_cell=1)
    backend = _FakeBackend()
    with pytest.raises(ValueError, match="require an Intervention"):
        run_serial_budget_interaction(backend, manifest, build_schedule(manifest))
    assert backend.calls == []
