import pytest

from epistemic_geometry.benchmarks.reasoning.budget_interaction import (
    CAPS,
    CONDITIONS,
    NAMESPACE,
    build_manifest,
    build_schedule,
)


def test_default_manifest_and_schedule_contract() -> None:
    manifest = build_manifest()
    assert len(manifest) == 96
    assert len(build_schedule(manifest)) == 768
    assert {row["namespace"] for row in manifest} == {NAMESPACE}
    assert all("answer" not in row and "spec" not in row for row in manifest)


def test_history_excludes_latent_without_loading_it() -> None:
    original = build_manifest()
    excluded = original[0]["latent_id"]
    fresh = build_manifest([excluded])
    assert len(fresh) == 96
    assert excluded not in {row["latent_id"] for row in fresh}


def test_manifest_and_schedule_are_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()
    assert first == second
    assert build_schedule(first) == build_schedule(second)


def test_schedule_matches_conditions_and_separates_caps_and_rollouts() -> None:
    rows = build_schedule(build_manifest(n_per_cell=1))
    by_key = {
        (row["latent_id"], row["cap"], row["rollout_index"], row["condition"]): row
        for row in rows
    }
    for latent_id in {row["latent_id"] for row in rows}:
        for cap in CAPS:
            for rollout in (0, 1):
                baseline = by_key[(latent_id, cap, rollout, CONDITIONS[0])]
                d75 = by_key[(latent_id, cap, rollout, CONDITIONS[1])]
                assert baseline["seed"] == d75["seed"]
        seeds = {
            by_key[(latent_id, cap, rollout, CONDITIONS[0])]["seed"]
            for cap in CAPS
            for rollout in (0, 1)
        }
        assert len(seeds) == 4


@pytest.mark.parametrize("bad_count", [0, -1, True, 1.5])
def test_manifest_rejects_malformed_count(bad_count: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_manifest(n_per_cell=bad_count)  # type: ignore[arg-type]


def test_manifest_rejects_duplicate_or_malformed_history() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        build_manifest(["historical", "historical"])
    with pytest.raises(TypeError, match="iterable"):
        build_manifest("historical")
