"""CPU-only validation of E3-10 identity, oracles, views, and splits."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .base import DIGITS, LatentItem
from .oracle import oracle_for
from .rendering import render_latent
from .splits import HOLDOUT_SPLIT, SplitManifest, assert_split_disjoint, generate_latent


def validate_item(item: LatentItem) -> dict[str, Any]:
    """Validate one latent item and all deterministic views."""

    if oracle_for(item) != item.target:
        raise AssertionError("oracle mismatch")
    roundtrip = LatentItem.from_record(item.to_record())
    if roundtrip != item:
        raise AssertionError(f"latent serialization changed {item.latent_id}")
    views = [
        render_latent(item, surface=surface, response_channel=channel)
        for surface in ("canonical", "surface_twin")
        for channel in ("decimal", "number_word")
    ]
    for view in views:
        if view.target != item.target or view.metadata["latent_hash"] != item.latent_hash:
            raise AssertionError(f"view identity mismatch for {view.view_id}")
        if (
            render_latent(item, surface=view.surface, response_channel=view.response_channel)
            != view
        ):
            raise AssertionError(f"view is not deterministic: {view.view_id}")
        if (
            view.prompt_hash
            != render_latent(
                item, surface=view.surface, response_channel=view.response_channel
            ).prompt_hash
        ):
            raise AssertionError("prompt hash is not deterministic")
    return {
        "latent_id": item.latent_id,
        "target": item.target,
        "views": len(views),
        "surface_oracle_equal": True,
    }


def validate_family(
    family: str, cells: tuple[str, ...], *, n_per_cell: int = 500
) -> dict[str, Any]:
    """Run extensive deterministic generator checks without a model."""

    reports: dict[str, Any] = {}
    for cell in cells:
        items = [generate_latent(family, cell, seed) for seed in range(n_per_cell)]
        if len({item.latent_id for item in items}) != n_per_cell:
            raise AssertionError(f"duplicate latent IDs in {family}/{cell}")
        for item in items:
            validate_item(item)
        reports[cell] = {
            "count": n_per_cell,
            "targets": dict(sorted(Counter(item.target for item in items).items())),
            "unique_latents": n_per_cell,
        }
    return reports


def validate_balanced_manifest(manifest: SplitManifest) -> dict[str, Any]:
    if manifest.split_name == HOLDOUT_SPLIT and manifest.metadata.get("development_access", True):
        raise AssertionError("holdout is marked accessible")
    ids = [item.latent_id for item in manifest.items]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate latent IDs in split")
    counts = Counter(item.target for item in manifest.items)
    expected = len(manifest.items) // 10
    if len(manifest.items) % 10 or any(counts[digit] != expected for digit in DIGITS):
        raise AssertionError(f"split is not exactly balanced: {counts}")
    for item in manifest.items:
        oracle_for(item)
    return {
        "split": manifest.split_name,
        "n_items": len(ids),
        "counts": dict(sorted(counts.items())),
    }


def validate_split_manifests(manifests: tuple[SplitManifest, ...]) -> dict[str, Any]:
    assert_split_disjoint(manifests)
    return {manifest.split_name: validate_balanced_manifest(manifest) for manifest in manifests}


def validate_suite(*, n_per_cell: int = 500) -> dict[str, Any]:
    from .splits import FAMILY_CELLS

    reports = {
        family: validate_family(family, cells, n_per_cell=n_per_cell)
        for family, cells in FAMILY_CELLS.items()
    }
    return {"families": reports, "status": "PASS"}
