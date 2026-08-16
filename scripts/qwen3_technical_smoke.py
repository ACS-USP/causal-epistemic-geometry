"""Run a cache-only Qwen3 mechanics and score-equivalence smoke test.

This deliberately uses the checked-in technical MCQ fixture, not MMLU-Pro.
Its outputs validate implementation mechanics only and must not be interpreted
as a scientific result.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from epistemic_geometry.backends import build_backend
from epistemic_geometry.backends.huggingface import HuggingFaceBackend
from epistemic_geometry.config import load_config
from epistemic_geometry.experiments.baseline_vs_steering import build_benchmark, build_vector
from epistemic_geometry.types import Intervention


def _sync_if_cuda(backend: HuggingFaceBackend) -> None:
    if backend.torch.cuda.is_available():
        backend.torch.cuda.synchronize()


def _margin(scores: dict[str, float]) -> float:
    values = sorted(scores.values(), reverse=True)
    return values[0] - values[1]


def _serial_output(
    backend: HuggingFaceBackend,
    item: Any,
    condition: str,
    vector: Any | None,
    alpha: float,
    layer: int,
) -> Any:
    if vector is None or alpha == 0.0:
        return backend.predict(item)
    intervention = Intervention(
        layer=layer,
        alpha=alpha,
        vector_id=vector.hash,
        token_scope="last_token",
        vector=vector,
    )
    with backend.steer(intervention):
        return backend.predict(item)


def _index_outputs(outputs: list[tuple[Any, dict[str, Any], Any]]) -> dict[tuple[str, str], Any]:
    return {(item.item_id, str(spec["condition"])): output for item, spec, output in outputs}


def _compare_scores(reference: Any, candidate: Any) -> dict[str, Any]:
    reference_scores = reference.metadata["candidate_scores"]
    candidate_scores = candidate.metadata["candidate_scores"]
    if set(reference_scores) != set(candidate_scores):
        raise AssertionError("Candidate labels differ between execution engines")
    differences = [
        abs(float(reference_scores[label]) - float(candidate_scores[label]))
        for label in reference_scores
    ]
    reference_ranking = sorted(reference_scores, key=reference_scores.get, reverse=True)
    candidate_ranking = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
    return {
        "prediction_equal": reference.raw_output == candidate.raw_output,
        "ranking_equal": reference_ranking == candidate_ranking,
        "reference_ranking": reference_ranking,
        "candidate_ranking": candidate_ranking,
        "max_absolute_score_difference": max(differences, default=0.0),
        "margin_difference": abs(_margin(reference_scores) - _margin(candidate_scores)),
        "reference_prediction": reference.raw_output,
        "candidate_prediction": candidate.raw_output,
        "reference_scores": reference_scores,
        "candidate_scores": candidate_scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serial-items", type=int, default=2)
    args = parser.parse_args()

    config = load_config(args.config)
    benchmark = build_benchmark(config)
    backend = build_backend(config)
    if not isinstance(backend, HuggingFaceBackend):
        raise TypeError("This smoke requires the HuggingFace backend")
    items = benchmark.items()
    prepared = backend.prepare_choice_items(items)
    audits = [backend.candidate_token_audit(item) for item in prepared]
    context_mismatches = []
    for audit in audits:
        for label, candidate in audit["candidates"].items():
            if not candidate["context_compatible"]:
                context_mismatches.append(f"{audit['item_id']}:{label}")
    if context_mismatches:
        raise AssertionError(
            "Technical smoke found context-incompatible candidate continuations: "
            + ", ".join(context_mismatches)
        )

    vector = build_vector(config, backend, benchmark)
    conditions = [
        ({"condition": "baseline", "layer": config.steering.layer, "alpha": 0.0}, None),
        (
            {
                "condition": "steered",
                "layer": config.steering.layer,
                "alpha": config.steering.alpha,
            },
            vector,
        ),
    ]
    compare_items = items[: args.serial_items]

    repeated_serial_outputs = [
        _serial_output(
            backend,
            compare_items[0],
            "baseline",
            None,
            0.0,
            config.steering.layer,
        )
        for _ in range(2)
    ]
    repeated_serial_audit = _compare_scores(
        repeated_serial_outputs[0], repeated_serial_outputs[1]
    )

    backend.reset_execution_stats()
    _sync_if_cuda(backend)
    serial_start = time.perf_counter()
    serial = {
        (item.id, "baseline"): _serial_output(
            backend, item, "baseline", None, 0.0, config.steering.layer
        )
        for item in compare_items
    }
    serial.update(
        {
            (item.id, "steered"): _serial_output(
                backend,
                item,
                "steered",
                vector,
                float(config.steering.alpha),
                config.steering.layer,
            )
            for item in compare_items
        }
    )
    _sync_if_cuda(backend)
    serial_seconds = time.perf_counter() - serial_start
    serial_stats = backend.execution_stats()

    def run_batch(
        mode: str,
        candidate_head_mode: str = "full_vocab_reference",
        selected_prepared: list[Any] | None = None,
        condition_chunk_size: int | None = None,
        selected_conditions: list[tuple[dict[str, Any], Any | None]] | None = None,
    ) -> tuple[dict[tuple[str, str], Any], float, dict[str, int]]:
        mode_config = replace(
            backend.config,
            execution_mode=mode,
            candidate_head_mode=candidate_head_mode,
            condition_chunk_size=condition_chunk_size or backend.config.condition_chunk_size,
        )
        mode_backend = backend if mode_config == backend.config else HuggingFaceBackend(
            mode_config,
            model=backend.model,
            tokenizer=backend.tokenizer,
            model_identifier=backend.model_name,
            tokenizer_identifier=backend.tokenizer_name,
            model_revision=backend.model_revision,
        )
        mode_backend.reset_execution_stats()
        _sync_if_cuda(mode_backend)
        started = time.perf_counter()
        outputs = mode_backend.predict_choice_batch(
            selected_prepared or prepared,
            selected_conditions or conditions,
            mode=mode,
        )
        _sync_if_cuda(mode_backend)
        return (
            _index_outputs(outputs),
            time.perf_counter() - started,
            mode_backend.execution_stats(),
        )

    full_prompt, full_prompt_seconds, full_prompt_stats = run_batch("full_prompt_batched")
    cached, cached_seconds, cached_stats = run_batch("cached_decode")
    candidate_only, candidate_only_seconds, candidate_only_stats = run_batch(
        "cached_decode", "candidate_only"
    )
    single_prepared = prepared[:1]
    single_full_prompt, _, _ = run_batch(
        "full_prompt_batched", selected_prepared=single_prepared
    )
    single_cached, _, _ = run_batch("cached_decode", selected_prepared=single_prepared)
    single_full_prompt_chunk1, _, _ = run_batch(
        "full_prompt_batched", selected_prepared=single_prepared, condition_chunk_size=1
    )
    single_cached_chunk1, _, _ = run_batch(
        "cached_decode", selected_prepared=single_prepared, condition_chunk_size=1
    )
    baseline_only = conditions[:1]
    single_full_baseline, _, _ = run_batch(
        "full_prompt_batched", selected_prepared=single_prepared, selected_conditions=baseline_only
    )
    single_cached_baseline, _, _ = run_batch(
        "cached_decode", selected_prepared=single_prepared, selected_conditions=baseline_only
    )

    serial_comparison = {}
    full_prompt_comparison = {}
    candidate_head_comparison = {}
    for item in compare_items:
        for condition, _vector in conditions:
            key = (item.id, condition["condition"])
            serial_comparison[f"{item.id}/{key[1]}"] = _compare_scores(serial[key], cached[key])
            full_prompt_comparison[f"{item.id}/{key[1]}"] = _compare_scores(
                full_prompt[key], cached[key]
            )
            candidate_head_comparison[f"{item.id}/{key[1]}"] = _compare_scores(
                cached[key], candidate_only[key]
            )
            if not serial_comparison[f"{item.id}/{key[1]}"]["prediction_equal"]:
                raise AssertionError(f"Serial/cached prediction mismatch for {key}")
            if not full_prompt_comparison[f"{item.id}/{key[1]}"]["prediction_equal"]:
                raise AssertionError(f"Full-prompt/cached prediction mismatch for {key}")
            if not candidate_head_comparison[f"{item.id}/{key[1]}"]["prediction_equal"]:
                raise AssertionError(f"Full-vocab/candidate-only prediction mismatch for {key}")

    result = {
        "status": "PASS",
        "warning": "TINY/TECHNICAL RESULTS ARE SOFTWARE VALIDATION ONLY.",
        "model": backend.provenance(),
        "technical_items": len(items),
        "serial_comparison_items": len(compare_items),
        "repeated_serial_audit": repeated_serial_audit,
        "prompt_lengths": {item.item_id: item.prompt_length for item in prepared},
        "candidate_token_audit": audits,
        "vector": {"hash": vector.hash, "dimension": vector.dimension, "layer": vector.layer},
        "engines": {
            "serial_reference": {"seconds": serial_seconds, "stats": serial_stats},
            "full_prompt_batched": {"seconds": full_prompt_seconds, "stats": full_prompt_stats},
            "cached_decode": {"seconds": cached_seconds, "stats": cached_stats},
            "candidate_only_cached_decode": {
                "seconds": candidate_only_seconds,
                "stats": candidate_only_stats,
            },
        },
        "serial_vs_cached": serial_comparison,
        "full_prompt_vs_cached": full_prompt_comparison,
        "full_vocab_vs_candidate_only": candidate_head_comparison,
        "single_item_padding_isolation": {
            "full_prompt_vs_cached": {
                f"{item.id}/{condition['condition']}": _compare_scores(
                    single_full_prompt[(item.id, condition["condition"])],
                    single_cached[(item.id, condition["condition"])],
                )
                for item in compare_items[:1]
                for condition, _vector in conditions
            },
            "full_prompt_vs_cached_condition_chunk_1": {
                f"{item.id}/{condition['condition']}": _compare_scores(
                    single_full_prompt_chunk1[(item.id, condition["condition"])],
                    single_cached_chunk1[(item.id, condition["condition"])],
                )
                for item in compare_items[:1]
                for condition, _vector in conditions
            },
            "baseline_only": {
                f"{item.id}/{condition['condition']}": _compare_scores(
                    single_full_baseline[(item.id, condition["condition"])],
                    single_cached_baseline[(item.id, condition["condition"])],
                )
                for item in compare_items[:1]
                for condition, _vector in baseline_only
            },
        },
        "max_serial_cached_margin_difference": max(
            (row["margin_difference"] for row in serial_comparison.values()), default=0.0
        ),
        "max_full_prompt_cached_margin_difference": max(
            (row["margin_difference"] for row in full_prompt_comparison.values()), default=0.0
        ),
        "max_candidate_head_margin_difference": max(
            (row["margin_difference"] for row in candidate_head_comparison.values()), default=0.0
        ),
        "ranking_equivalence": {
            "serial_vs_cached": all(
                row["ranking_equal"] for row in serial_comparison.values()
            ),
            "full_prompt_vs_cached": all(
                row["ranking_equal"] for row in full_prompt_comparison.values()
            ),
            "full_vocab_vs_candidate_only": all(
                row["ranking_equal"] for row in candidate_head_comparison.values()
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
