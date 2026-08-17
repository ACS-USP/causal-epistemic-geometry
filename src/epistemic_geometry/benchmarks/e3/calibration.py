"""Baseline-only calibration execution adapter for the future RunPod phase."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from epistemic_geometry.backends.base import ModelBackend

from .benchmark import calibration_benchmark_items
from .qualification import CalibrationScoreRow
from .splits import SplitManifest, assert_development_access


def baseline_calibration_conditions(layer: int) -> list[tuple[dict[str, Any], None]]:
    """Return the only permitted calibration condition: baseline/no steering."""

    return [
        (
            {
                "condition": "baseline",
                "layer": layer,
                "alpha": 0.0,
                "token_scope": "none",
                "steering": False,
            },
            None,
        )
    ]


def run_baseline_calibration(
    backend: ModelBackend,
    manifest: SplitManifest,
    *,
    execution_mode: str | None = None,
    layer: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> list[CalibrationScoreRow]:
    """Score only calibration views; no activations or directions are built.

    The backend must be configured for candidate-only semantic logits.  This
    function deliberately refuses steering vectors and uses one baseline
    condition for every view.
    """

    assert_development_access(manifest.split_name)
    items = calibration_benchmark_items(manifest)
    if not hasattr(backend, "prepare_choice_items") or not hasattr(backend, "predict_choice_batch"):
        raise TypeError("E3-10 calibration requires the prepared-choice HuggingFace backend")
    prepared = backend.prepare_choice_items(items)  # type: ignore[attr-defined]
    if any(not row.all_candidates_single_token for row in prepared):
        raise RuntimeError("E3-10 candidate tokenization is not single-token for every view")
    outputs = backend.predict_choice_batch(  # type: ignore[attr-defined]
        prepared,
        baseline_calibration_conditions(layer),
        mode=execution_mode,
    )
    results: list[CalibrationScoreRow] = []
    for index, (item, _condition, output) in enumerate(outputs, start=1):
        metadata = output.metadata
        if metadata.get("candidate_score_semantics") != "candidate_logits_no_vocab_normalization":
            raise RuntimeError(
                "E3-10 calibration requires raw candidate logits from candidate-only head; "
                f"got {metadata.get('candidate_score_semantics')!r}"
            )
        scores_by_label = metadata.get("candidate_scores")
        if not isinstance(scores_by_label, dict):
            raise RuntimeError("baseline calibration output is missing candidate_scores")
        labels = item.metadata["candidate_labels"]
        scores = tuple(float(scores_by_label[label]) for label in labels)
        results.append(
            CalibrationScoreRow(
                latent_id=str(item.metadata["latent_id"]),
                family=str(item.metadata["family"]),
                cell=str(item.metadata["cell"]),
                surface=str(item.metadata["surface"]),
                response_channel=str(item.metadata["response_channel"]),
                target=int(item.metadata["target_digit"]),
                scores=scores,
                prompt_hash=str(
                    metadata.get("rendered_prompt_hash", item.metadata["rendered_prompt_hash"])
                ),
                metadata={
                    "view_id": item.item_id,
                    "source_prompt_hash": item.metadata["rendered_prompt_hash"],
                    "condition": "baseline",
                    "candidate_score_semantics": metadata["candidate_score_semantics"],
                    "execution_engine": metadata.get("execution_engine"),
                },
            )
        )
        if progress is not None:
            progress(index, len(outputs))
    return results
