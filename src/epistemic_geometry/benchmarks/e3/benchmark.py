"""Adapters from E3 latent views to the existing typed benchmark interface."""

from __future__ import annotations

from collections.abc import Iterable

from epistemic_geometry.types import BenchmarkItem

from .base import RenderedView
from .rendering import render_latent
from .splits import SplitManifest, assert_development_access


def view_to_benchmark_item(view: RenderedView) -> BenchmarkItem:
    """Expose a view as a choice-scoring item with semantic candidates."""

    labels = (
        [str(digit) for digit in range(10)]
        if view.response_channel == "decimal"
        else [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
    )
    return BenchmarkItem(
        id=view.view_id,
        prompt=view.prompt,
        target=view.target_text,
        metadata={
            "e3_10": True,
            "latent_id": view.latent_id,
            "response_channel": view.response_channel,
            "surface": view.surface,
            "family": view.family,
            "cell": view.cell,
            "target_digit": view.target,
            "target_text": view.target_text,
            "candidate_labels": labels,
            "semantic_option_ids": list(range(10)),
            "rendered_prompt_hash": view.prompt_hash,
            "template_hash": view.template_hash,
        },
    )


def views_for_item(item: object) -> tuple[RenderedView, ...]:
    """Render the three pre-registered calibration views for a latent item."""

    from .base import LatentItem

    if not isinstance(item, LatentItem):
        raise TypeError("views_for_item requires a LatentItem")
    return (
        render_latent(item),
        render_latent(item, surface="surface_twin"),
        render_latent(item, response_channel="number_word"),
    )


def calibration_benchmark_items(manifest: SplitManifest) -> list[BenchmarkItem]:
    """Return baseline calibration views and reject the confirmatory holdout."""

    assert_development_access(manifest.split_name)
    return [
        view_to_benchmark_item(view) for item in manifest.items for view in views_for_item(item)
    ]


def assert_no_holdout_items(items: Iterable[BenchmarkItem]) -> None:
    """Fail closed if a runner is handed a confirmatory item."""

    for item in items:
        if item.metadata.get("split_name") == "CONFIRMATORY_HOLDOUT":
            raise PermissionError("E3-10 development runner cannot consume CONFIRMATORY_HOLDOUT")
