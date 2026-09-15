from __future__ import annotations

import pytest

from epistemic_geometry.benchmarks.reasoning.finalization_timing import (
    CAPS,
    CONDITIONS,
    NAMESPACE,
    build_manifest,
    build_schedule,
    validate_schedule,
)


def test_timing_manifest_and_schedule_are_complete_and_matched() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    assert CAPS == (4096,)
    assert len(schedule) == len(manifest) * len(CONDITIONS) * 2
    assert {row["namespace"] for row in schedule} == {NAMESPACE}
    pairs = {}
    for row in schedule:
        key = (row["latent_id"], row["cap"], row["rollout_index"])
        pairs.setdefault(key, set()).add((row["condition"], row["sampling_seed"]))
    assert all(len(values) == 2 for values in pairs.values())
    assert all(len({seed for _, seed in values}) == 1 for values in pairs.values())


def test_timing_schedule_refuses_treatment_specific_seed() -> None:
    manifest = build_manifest(n_per_cell=1)
    schedule = build_schedule(manifest)
    schedule[0] = {**schedule[0], "sampling_seed": schedule[0]["sampling_seed"] + 1}
    with pytest.raises(ValueError, match="seed"):
        validate_schedule(schedule, manifest)
