from __future__ import annotations

import pytest

from epistemic_geometry.benchmarks.reasoning.novelty import (
    DIVERSIFICATION_COVERAGE_NAMESPACE,
    DIVERSIFICATION_COVERAGE_SPLIT,
    DiversificationManifest,
    build_diversification_schedule,
    generate_diversification_manifest,
)
from epistemic_geometry.benchmarks.reasoning.splits import GEOMETRY_CALIBRATION


def test_new_manifest_is_deterministic_and_excludes_supplied_latents() -> None:
    first = generate_diversification_manifest(
        "FSM-R",
        "length_4",
        seed=23,
        n_items=3,
        namespace="TEST-DIVERSIFICATION-COVERAGE-V1",
    )
    excluded = {first.items[0].latent_id}
    second = generate_diversification_manifest(
        "FSM-R",
        "length_4",
        seed=23,
        n_items=3,
        namespace="TEST-DIVERSIFICATION-COVERAGE-V1",
        excluded_latent_ids=excluded,
    )
    repeat = generate_diversification_manifest(
        "FSM-R",
        "length_4",
        seed=23,
        n_items=3,
        namespace="TEST-DIVERSIFICATION-COVERAGE-V1",
        excluded_latent_ids=excluded,
    )
    assert second.to_record() == repeat.to_record()
    assert not ({item.latent_id for item in second.items} & excluded)
    assert second.to_record()["split_name"] == DIVERSIFICATION_COVERAGE_SPLIT


def test_manifest_roundtrip_and_schedule_are_namespaced_and_deterministic() -> None:
    manifest = generate_diversification_manifest(
        "MODREG-R",
        "depth_4",
        seed=9,
        n_items=2,
        namespace=DIVERSIFICATION_COVERAGE_NAMESPACE,
    )
    restored = DiversificationManifest.from_record(manifest.to_record())
    first = build_diversification_schedule(restored, base_seed=71)
    second = build_diversification_schedule(restored, base_seed=71)
    assert first == second
    assert len(first) == 2 * 3 * 2
    assert {row["namespace"] for row in first} == {DIVERSIFICATION_COVERAGE_NAMESPACE}
    assert len({row["seed"] for row in first}) == len(first)


def test_historical_split_names_are_rejected_by_new_facility() -> None:
    with pytest.raises(ValueError, match="historical"):
        generate_diversification_manifest(
            "FSM-R",
            "length_4",
            seed=1,
            n_items=1,
            namespace=GEOMETRY_CALIBRATION,
        )
    manifest = generate_diversification_manifest(
        "FSM-R", "length_4", seed=1, n_items=1, namespace="new-study"
    )
    record = manifest.to_record()
    record["split_name"] = GEOMETRY_CALIBRATION
    with pytest.raises(ValueError, match="diversification coverage manifest"):
        DiversificationManifest.from_record(record)


def test_manifest_rejects_excluded_collision_and_schedule_argument_errors() -> None:
    manifest = generate_diversification_manifest(
        "SATCOUNT-R", "vars4_clauses4", seed=4, n_items=1, namespace="new-study"
    )
    with pytest.raises(ValueError, match="collides"):
        DiversificationManifest(
            namespace="new-study",
            family=manifest.family,
            cell=manifest.cell,
            seed=manifest.seed,
            items=manifest.items,
            excluded_latent_ids={manifest.items[0].latent_id},
        )
    with pytest.raises(ValueError, match="unique"):
        build_diversification_schedule(manifest, conditions=("A", "A"))
