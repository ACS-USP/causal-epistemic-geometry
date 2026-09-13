from __future__ import annotations

import pytest

from epistemic_geometry.benchmarks.reasoning.novelty import (
    DIVERSIFICATION_COVERAGE_NAMESPACE,
    DIVERSIFICATION_COVERAGE_SPLIT,
    DiversificationManifest,
    audit_diversification_novelty,
    build_diversification_schedule,
    generate_diversification_manifest,
    plan_diversification_coverage,
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


def test_novelty_audit_reports_candidate_collision_and_historical_duplicates() -> None:
    candidate = generate_diversification_manifest(
        "FSM-R", "length_4", seed=31, n_items=2, namespace="candidate"
    )
    first = {"split_name": "z-history", "items": [candidate.items[0].to_record()]}
    second = {
        "split_name": "a-history",
        "items": [candidate.items[0].to_record(), candidate.items[0].to_record()],
    }
    report = audit_diversification_novelty([first, second], candidate)
    assert report.candidate_disjoint is False
    assert report.historical_duplicate_ids == (candidate.items[0].latent_id,)
    assert report.candidate_collisions[0]["latent_id"] == candidate.items[0].latent_id
    assert report.passed is False


def test_novelty_audit_detects_duplicate_across_historical_records() -> None:
    manifest = generate_diversification_manifest(
        "FSM-R", "length_4", seed=32, n_items=1, namespace="candidate"
    )
    item = manifest.items[0].to_record()
    report = audit_diversification_novelty(
        [{"split_name": "b", "items": [item]}, {"split_name": "a", "items": [item]}],
        manifest,
    )
    assert report.historical_collisions[0]["kind"] == "across_manifests"
    assert report.historical_collisions[0]["manifest_names"] == ["a", "b"]


def test_novelty_audit_rejects_malformed_or_tampered_inputs() -> None:
    manifest = generate_diversification_manifest(
        "FSM-R", "length_4", seed=33, n_items=1, namespace="candidate"
    )
    tampered = manifest.to_record()
    tampered["items"][0]["latent_id"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_diversification_novelty([], tampered)
    with pytest.raises(ValueError, match="malformed items"):
        audit_diversification_novelty([{"split_name": "bad", "items": "latent-id"}], manifest)


def test_novelty_audit_output_order_is_deterministic() -> None:
    candidate = generate_diversification_manifest(
        "FSM-R", "length_4", seed=34, n_items=2, namespace="candidate"
    )
    historical = [
        {"split_name": "z", "items": [candidate.items[1].to_record()]},
        {"split_name": "a", "items": [candidate.items[0].to_record()]},
    ]
    first = audit_diversification_novelty(historical, candidate).to_record()
    second = audit_diversification_novelty(list(reversed(historical)), candidate).to_record()
    assert first == second
    assert first["historical_manifest_names"] == ["a", "z"]
    assert first["candidate_collisions"] == sorted(
        first["candidate_collisions"], key=lambda row: row["latent_id"]
    )


def test_coverage_plan_is_disjoint_and_uses_at_only_for_qualification() -> None:
    counts = {
        "MODREG-R": {"depth_4": (2, 3)},
        "FSM-R": {"length_4": (2, 3)},
        "SATCOUNT-R": {"vars4_clauses4": (2, 3)},
    }
    first = plan_diversification_coverage(counts, seed=101)
    second = plan_diversification_coverage(counts, seed=101)
    assert first.to_record() == second.to_record()
    assert first.passed
    assert not (set(first.qualification_latent_ids) & set(first.evaluation_latent_ids))
    assert first.schedule_counts == {"qualification": 24, "evaluation": 54}
    assert {key[1] for key in first.schedule_keys["qualification"]} == {"A", "T"}
    assert {key[1] for key in first.schedule_keys["evaluation"]} == {"A", "B", "T"}


def test_standard_qualification_plan_has_exactly_384_at_generations() -> None:
    counts = {
        "MODREG-R": {cell: (8, 1) for cell in ("depth_4", "depth_8", "depth_12", "depth_16")},
        "FSM-R": {cell: (8, 1) for cell in ("length_4", "length_8", "length_12", "length_16")},
        "SATCOUNT-R": {
            cell: (8, 1)
            for cell in ("vars4_clauses4", "vars4_clauses6", "vars5_clauses8", "vars6_clauses10")
        },
    }
    plan = plan_diversification_coverage(counts, seed=102)
    assert len(plan.qualification_latent_ids) == 96
    assert plan.schedule_counts["qualification"] == 384


def test_coverage_plan_excludes_history_and_rejects_bad_cells() -> None:
    initial = plan_diversification_coverage(
        {"FSM-R": {"length_4": 1}}, seed=103
    )
    excluded = initial.qualification_latent_ids + initial.evaluation_latent_ids
    plan = plan_diversification_coverage(
        {"FSM-R": {"length_4": 1}},
        seed=103,
        historical_excluded_latent_ids=excluded,
    )
    assert not (set(plan.all_latent_ids) & set(excluded))
    with pytest.raises(ValueError, match="unknown reasoning family/cell"):
        plan_diversification_coverage({"FSM-R": {"not-a-cell": 1}}, seed=104)
