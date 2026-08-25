#!/usr/bin/env python3
"""Run the outcome-free real-Qwen M3 engineering qualification.

The runner imports the already-audited Gate-12.1 model plumbing, never calls
``generate``, and has no benchmark or semantic-evaluator import.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.analysis import m3_qualification as m3  # noqa: E402
from epistemic_geometry.reproducibility import require_remote_hf_execution  # noqa: E402
from scripts.run_gate12_1_continuous_geometry_engine import (  # noqa: E402
    build_backend,
    full_logits,
    hash_state,
    numpy_logits,
    sequential_logits,
)

REVIEW = ROOT / "review/q2_m3_qualification_cruxeval_provenance"
RAW = REVIEW / "raw_m3_engineering"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_is_ancestor(source: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source, "HEAD"], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def selected(values: Any, fixture: dict[str, Any]) -> Any:
    return values[fixture["checkpoint_offsets"]]


def probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = probabilities(left)
    q = probabilities(right)
    midpoint = 0.5 * (p + q)
    return 0.5 * np.sum(p * (np.log(p) - np.log(midpoint)), axis=-1) + 0.5 * np.sum(
        q * (np.log(q) - np.log(midpoint)), axis=-1
    )


def kl_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = probabilities(left)
    q = probabilities(right)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def hellinger_squared(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    p = probabilities(left)
    q = probabilities(right)
    return 0.5 * np.sum(np.square(np.sqrt(p) - np.sqrt(q)), axis=-1)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    return float((x @ y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-30))


def relative(observed: float, expected: float, floor: float = 1e-12) -> float:
    return float(abs(observed - expected) / max(abs(expected), floor))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = rankdata(np.asarray(left))
    right_ranks = rankdata(np.asarray(right))
    return cosine(left_ranks - np.mean(left_ranks), right_ranks - np.mean(right_ranks))


def target_ids(fixture: dict[str, Any]) -> np.ndarray:
    continuation = fixture["continuation_token_ids"]
    fallback = fixture["prompt_token_ids"][0]
    return np.asarray(
        [
            continuation[offset] if offset < len(continuation) else fallback
            for offset in fixture["checkpoint_offsets"]
        ],
        dtype=np.int64,
    )


def comparison_rows(
    fixture: dict[str, Any], left: np.ndarray, right: np.ndarray
) -> list[dict[str, float | int | str]]:
    targets = target_ids(fixture)
    left_logp = np.log(probabilities(left))
    right_logp = np.log(probabilities(right))
    js = js_divergence(left, right)
    rows = []
    for checkpoint in range(len(left)):
        rows.append(
            {
                "fixture_id": fixture["fixture_id"],
                "checkpoint_offset": fixture["checkpoint_offsets"][checkpoint],
                "top1_agreement": int(np.argmax(left[checkpoint]) == np.argmax(right[checkpoint])),
                "vocabulary_js": float(js[checkpoint]),
                "target_logp_abs_difference": float(
                    abs(
                        left_logp[checkpoint, targets[checkpoint]]
                        - right_logp[checkpoint, targets[checkpoint]]
                    )
                ),
                "logit_cosine": cosine(left[checkpoint], right[checkpoint]),
            }
        )
    return rows


def summarize_comparisons(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return {
        "rows": len(rows),
        "top1_agreement": float(np.mean([row["top1_agreement"] for row in rows])),
        "median_vocabulary_js": float(np.median([row["vocabulary_js"] for row in rows])),
        "max_vocabulary_js": float(np.max([row["vocabulary_js"] for row in rows])),
        "p95_vocabulary_js": float(np.quantile([row["vocabulary_js"] for row in rows], 0.95)),
        "p99_vocabulary_js": float(np.quantile([row["vocabulary_js"] for row in rows], 0.99)),
        "median_target_logp_abs_difference": float(
            np.median([row["target_logp_abs_difference"] for row in rows])
        ),
        "max_target_logp_abs_difference": float(
            np.max([row["target_logp_abs_difference"] for row in rows])
        ),
        "median_logit_cosine": float(np.median([row["logit_cosine"] for row in rows])),
    }


def model_function(backend: Any, fixture: dict[str, Any], vector: Any, implementation: str):
    kernel = "eager" if implementation == "eager" else "math"

    def function(alpha: Any) -> Any:
        values, _ = full_logits(
            backend,
            fixture,
            alpha,
            vector,
            implementation=implementation,
            kernel=kernel,
            hook=True,
        )
        return selected(values, fixture)

    return function


def exact_jvp(backend: Any, function: Any) -> tuple[Any, Any]:
    torch = backend.torch
    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
        alpha = torch.tensor(0.0, dtype=torch.float32, device=backend.device)
        tangent = torch.tensor(1.0, dtype=torch.float32, device=backend.device)
        dual = torch.autograd.forward_ad.make_dual(alpha, tangent)
        output = function(dual)
        primal, jvp = torch.autograd.forward_ad.unpack_dual(output)
    return primal, jvp


def fixture_exact(
    backend: Any,
    fixture: dict[str, Any],
    directions: np.ndarray,
    implementation: str,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    torch = backend.torch
    baseline: np.ndarray | None = None
    jvps = []
    for direction in directions:
        vector = torch.tensor(direction, dtype=torch.float32, device=backend.device)
        primal, tangent = exact_jvp(
            backend, model_function(backend, fixture, vector, implementation)
        )
        primal_np = numpy_logits(primal)
        if baseline is None:
            baseline = primal_np
        elif not np.array_equal(baseline, primal_np):
            raise RuntimeError("exact JVP primals changed across direction order")
        jvps.append(numpy_logits(tangent))
    assert baseline is not None
    checkpoint_grams = []
    stacked = np.stack(jvps, axis=1)
    for checkpoint in range(len(baseline)):
        checkpoint_grams.append(
            m3.weighted_fisher_gram(stacked[checkpoint], probabilities(baseline[checkpoint]))
        )
    return baseline, stacked, checkpoint_grams


def aggregate_grams(grams: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(grams), axis=0)


def finite_geometry(
    baseline: np.ndarray,
    plus: list[np.ndarray],
    minus: list[np.ndarray],
    epsilon: float,
) -> tuple[np.ndarray, dict[str, float]]:
    finite = np.stack(
        [
            (right.astype(np.float64) - left.astype(np.float64)) / (2 * epsilon)
            for right, left in zip(plus, minus, strict=True)
        ],
        axis=1,
    )
    grams = [
        m3.weighted_fisher_gram(finite[index], probabilities(baseline[index]))
        for index in range(len(baseline))
    ]
    movement = np.concatenate([(right.astype(np.float64) - baseline) for right in plus], axis=None)
    return aggregate_grams(grams), {"rms_logit_movement": float(np.sqrt(np.mean(movement**2)))}


def run(args: argparse.Namespace) -> int:
    require_remote_hf_execution("Q2 M3 engineering qualification")
    if not source_is_ancestor(args.experiment_source_commit):
        raise RuntimeError("experiment source commit is not an ancestor of execution HEAD")
    lock = read_json(REVIEW / "PROTOCOL_LOCK.json")
    if lock["experiment_source_commit"] != args.experiment_source_commit:
        raise RuntimeError("M3 source commit mismatch")
    if sha256(REVIEW / "M3_QUALIFICATION_PROTOCOL.json") != lock["protocol_sha256"]:
        raise RuntimeError("M3 protocol hash mismatch")
    fixtures = read_json(REVIEW / "M3_ENGINEERING_FIXTURES.json")["fixtures"]
    directions = np.load(REVIEW / "M3_ENGINEERING_DIRECTIONS.npz", allow_pickle=False)["directions"]
    if len(fixtures) != m3.FIXTURE_COUNT or directions.shape != (
        m3.DIRECTION_COUNT,
        m3.HIDDEN_SIZE,
    ):
        raise RuntimeError("frozen M3 fixtures or directions have unexpected shape")

    backend = build_backend(args.model_path)
    torch = backend.torch
    started = time.time()
    bf16_hash = hash_state(backend.model, torch)
    zero = torch.tensor(0.0, dtype=torch.float32, device=backend.device)
    bridge_eps = torch.tensor(m3.BF16_BRIDGE_EPSILON, dtype=torch.float32, device=backend.device)
    bf16_baselines: dict[int, np.ndarray] = {}
    bf16_bridge_q: dict[int, dict[str, Any]] = {}

    for fixture_index, fixture in enumerate(fixtures):
        vector = torch.tensor(directions[0], dtype=torch.float32, device=backend.device)
        with torch.inference_mode():
            logits, _ = sequential_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation="sdpa",
                kernel="default",
                hook=False,
            )
        bf16_baselines[fixture_index] = numpy_logits(selected(logits, fixture))

    for fixture_index in m3.BRIDGE_FIXTURE_INDICES:
        fixture = fixtures[fixture_index]
        baseline = bf16_baselines[fixture_index]
        q_values: list[list[float]] = []
        q_sums: dict[str, list[float]] = {}
        for direction in directions:
            vector = torch.tensor(direction, dtype=torch.float32, device=backend.device)
            with torch.inference_mode():
                moved, _ = sequential_logits(
                    backend,
                    fixture,
                    bridge_eps,
                    vector,
                    implementation="sdpa",
                    kernel="default",
                    hook=True,
                )
            moved_np = numpy_logits(selected(moved, fixture))
            q_values.append(
                (
                    8 * js_divergence(baseline, moved_np) / bridge_eps.item() ** 2
                ).tolist()
            )
        for left in range(len(directions)):
            for right in range(left + 1, len(directions)):
                vector = torch.tensor(
                    directions[left] + directions[right], dtype=torch.float32, device=backend.device
                )
                with torch.inference_mode():
                    moved, _ = sequential_logits(
                        backend,
                        fixture,
                        bridge_eps,
                        vector,
                        implementation="sdpa",
                        kernel="default",
                        hook=True,
                    )
                moved_np = numpy_logits(selected(moved, fixture))
                q_sums[f"{left}:{right}"] = (
                    8 * js_divergence(baseline, moved_np) / bridge_eps.item() ** 2
                ).tolist()
        bf16_bridge_q[fixture_index] = {"diagonal": q_values, "sums": q_sums}
        write_json(
            REVIEW / "M3_REMOTE_PROGRESS.json", {"phase": "BF16_BRIDGE", "completed": fixture_index}
        )

    backend.model.float()
    backend.device = next(backend.model.parameters()).device
    fp32_hash = hash_state(backend.model, torch)
    roundtrip_hash = hash_state(backend.model, torch, cast_back_bf16=True)
    if roundtrip_hash != bf16_hash:
        raise RuntimeError("FP32 lift is not an exact cast of loaded BF16 values")
    eager_available = True
    eager_error = None
    try:
        fixture = fixtures[0]
        vector = torch.tensor(directions[0], dtype=torch.float32, device=backend.device)
        with torch.inference_mode():
            full_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation="eager",
                kernel="eager",
                hook=False,
            )
    except Exception as exc:  # pragma: no cover - real architecture dependent
        eager_available = False
        eager_error = f"{type(exc).__name__}: {exc}"
        torch.cuda.empty_cache()
    implementation = "eager" if eager_available else "sdpa"
    kernel = "eager" if eager_available else "math"

    fp32_sequence_rows: list[dict[str, Any]] = []
    alpha_zero_identity_rows: list[dict[str, Any]] = []
    bf16_bridge_rows: list[dict[str, Any]] = []
    for fixture_index, fixture in enumerate(fixtures):
        vector = torch.tensor(
            directions[fixture_index % len(directions)], dtype=torch.float32, device=backend.device
        )
        with torch.inference_mode():
            sequential, _ = sequential_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=False,
            )
            full, _ = full_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=False,
            )
            sequential_zero_hook, _ = sequential_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=True,
            )
            full_zero_hook, _ = full_logits(
                backend,
                fixture,
                zero,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=True,
            )
        sequential_np = numpy_logits(selected(sequential, fixture))
        full_np = numpy_logits(selected(full, fixture))
        sequential_zero_np = numpy_logits(selected(sequential_zero_hook, fixture))
        full_zero_np = numpy_logits(selected(full_zero_hook, fixture))
        alpha_zero_identity_rows.extend(comparison_rows(fixture, sequential_np, sequential_zero_np))
        alpha_zero_identity_rows.extend(comparison_rows(fixture, full_np, full_zero_np))
        fp32_sequence_rows.extend(comparison_rows(fixture, sequential_zero_np, full_zero_np))
        bf16_bridge_rows.extend(
            comparison_rows(fixture, bf16_baselines[fixture_index], sequential_np)
        )
        alpha = torch.tensor(0.1, dtype=torch.float32, device=backend.device)
        with torch.inference_mode():
            sequential, _ = sequential_logits(
                backend,
                fixture,
                alpha,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=True,
            )
            full, _ = full_logits(
                backend,
                fixture,
                alpha,
                vector,
                implementation=implementation,
                kernel=kernel,
                hook=True,
            )
        fp32_sequence_rows.extend(
            comparison_rows(
                fixture,
                numpy_logits(selected(sequential, fixture)),
                numpy_logits(selected(full, fixture)),
            )
        )

    all_checkpoint_grams: list[np.ndarray] = []
    fixture_summaries: dict[int, dict[str, Any]] = {}
    exact_cache: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = {}
    for fixture_index, fixture in enumerate(fixtures):
        baseline, jvps, grams = fixture_exact(backend, fixture, directions, implementation)
        exact_cache[fixture_index] = (baseline, jvps, grams)
        all_checkpoint_grams.extend(grams)
        weighted_means = []
        second_moments = []
        for checkpoint in range(len(baseline)):
            p = probabilities(baseline[checkpoint])
            rows = jvps[checkpoint].astype(np.float64)
            weighted_means.append(rows @ p)
            second_moments.append((rows * p[None, :]) @ rows.T)
        fixture_summaries[fixture_index] = {
            "fixture_id": fixture["fixture_id"],
            "checkpoint_offsets": fixture["checkpoint_offsets"],
            "weighted_means": np.stack(weighted_means).tolist(),
            "weighted_second_moments": np.stack(second_moments).tolist(),
            "checkpoint_grams": np.stack(grams).tolist(),
        }
        write_json(
            REVIEW / "M3_REMOTE_PROGRESS.json",
            {"phase": "EXACT_GRAM", "completed": fixture_index + 1, "total": len(fixtures)},
        )
        torch.cuda.empty_cache()
    exact_gram = aggregate_grams(all_checkpoint_grams)

    repeat_fixture = fixtures[0]
    repeat = fixture_exact(backend, repeat_fixture, directions, implementation)[2]
    reverse = fixture_exact(backend, repeat_fixture, directions[::-1], implementation)[2]
    reverse_gram = aggregate_grams(reverse)[::-1, ::-1]
    reproducibility = {
        "repeat_relative_frobenius": m3.relative_frobenius(
            aggregate_grams(repeat), aggregate_grams(exact_cache[0][2])
        ),
        "direction_order_relative_frobenius": m3.relative_frobenius(
            reverse_gram, aggregate_grams(exact_cache[0][2])
        ),
        "chunked_aggregation_relative_frobenius": m3.relative_frobenius(
            0.5
            * (
                aggregate_grams(all_checkpoint_grams[: len(all_checkpoint_grams) // 2])
                + aggregate_grams(all_checkpoint_grams[len(all_checkpoint_grams) // 2 :])
            ),
            exact_gram,
        ),
    }

    crosschecks = []
    for fixture_index, direction_index in m3.EXACT_CROSSCHECK_CASES:
        fixture = fixtures[fixture_index]
        vector = torch.tensor(
            directions[direction_index], dtype=torch.float32, device=backend.device
        )
        function = model_function(backend, fixture, vector, implementation)
        primal, forward = exact_jvp(backend, function)
        alpha = torch.tensor(0.0, dtype=torch.float32, device=backend.device)
        tangent = torch.tensor(1.0, dtype=torch.float32, device=backend.device)
        independent_primal, independent = torch.autograd.functional.jvp(
            function, alpha, tangent, create_graph=False, strict=True
        )
        rng = np.random.default_rng(2_026_082_500 + fixture_index * 10 + direction_index)
        cotangent_np = rng.standard_normal(forward.numel()).astype(np.float32)
        cotangent_np /= np.linalg.norm(cotangent_np)
        cotangent = torch.tensor(cotangent_np.reshape(forward.shape), device=backend.device)
        alpha_vjp = torch.tensor(
            0.0, dtype=torch.float32, device=backend.device, requires_grad=True
        )
        output = function(alpha_vjp)
        right = torch.autograd.grad(torch.sum(output.double() * cotangent.double()), alpha_vjp)[0]
        left = torch.sum(forward.double() * cotangent.double())
        forward_np = numpy_logits(forward)
        independent_np = numpy_logits(independent)
        raw_path = RAW / f"crosscheck_{fixture_index}_{direction_index}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raw_path,
            primal=numpy_logits(primal),
            independent_primal=numpy_logits(independent_primal),
            forward_jvp=forward_np,
            independent_jvp=independent_np,
            cotangent=cotangent_np,
        )
        crosschecks.append(
            {
                "fixture_index": fixture_index,
                "direction_index": direction_index,
                "jvp_cosine": cosine(forward_np, independent_np),
                "jvp_relative_norm": relative(
                    np.linalg.norm(independent_np), np.linalg.norm(forward_np)
                ),
                "jvp_vjp_relative_error": relative(float(left.item()), float(right.item())),
                "raw_path": str(raw_path.relative_to(ROOT)),
                "raw_sha256": sha256(raw_path),
            }
        )

    direct_subset_grams = []
    polarization_subset_grams = []
    for fixture_index in m3.POLARIZATION_FIXTURE_INDICES:
        fixture = fixtures[fixture_index]
        baseline, _jvps, grams = exact_cache[fixture_index]
        direct_subset_grams.extend(grams)
        q_diag = np.diag(aggregate_grams(grams))
        gram = np.diag(q_diag.copy())
        for left_index in range(len(directions)):
            for right_index in range(left_index + 1, len(directions)):
                vector = torch.tensor(
                    directions[left_index] + directions[right_index],
                    dtype=torch.float32,
                    device=backend.device,
                )
                _primal, tangent = exact_jvp(
                    backend, model_function(backend, fixture, vector, implementation)
                )
                tangent_np = numpy_logits(tangent)
                qsum = np.mean(
                    [
                        m3.weighted_fisher_gram(
                            tangent_np[index][None, :], probabilities(baseline[index])
                        )[0, 0]
                        for index in range(len(baseline))
                    ]
                )
                cross = 0.5 * (qsum - q_diag[left_index] - q_diag[right_index])
                gram[left_index, right_index] = cross
                gram[right_index, left_index] = cross
        polarization_subset_grams.append(gram)
    direct_subset = aggregate_grams(direct_subset_grams)
    polarization_subset = aggregate_grams(polarization_subset_grams)

    finite_rows = []
    exact_diff_grams = [
        gram
        for fixture_index in m3.DIFFERENTIAL_FIXTURE_INDICES
        for gram in exact_cache[fixture_index][2]
    ]
    exact_diff_gram = aggregate_grams(exact_diff_grams)
    exact_geometry = m3.gram_geometry(exact_diff_gram)
    for epsilon in m3.EPSILONS:
        finite_grams = []
        all_exact = []
        all_finite = []
        q_exact = []
        q_finite = []
        kl_q = []
        h_q = []
        js_q = []
        movements = []
        for fixture_index in m3.DIFFERENTIAL_FIXTURE_INDICES:
            fixture = fixtures[fixture_index]
            baseline, jvps, _grams = exact_cache[fixture_index]
            plus = []
            minus = []
            for direction_index, direction in enumerate(directions):
                vector = torch.tensor(direction, dtype=torch.float32, device=backend.device)
                with torch.inference_mode():
                    plus_values = model_function(backend, fixture, vector, implementation)(
                        torch.tensor(epsilon, device=backend.device)
                    )
                    minus_values = model_function(backend, fixture, vector, implementation)(
                        torch.tensor(-epsilon, device=backend.device)
                    )
                plus_np = numpy_logits(plus_values)
                minus_np = numpy_logits(minus_values)
                finite = (plus_np.astype(np.float64) - minus_np.astype(np.float64)) / (2 * epsilon)
                plus.append(plus_np)
                minus.append(minus_np)
                all_exact.append(jvps[:, direction_index])
                all_finite.append(finite)
                for checkpoint in range(len(baseline)):
                    p = probabilities(baseline[checkpoint])
                    q0 = m3.weighted_fisher_gram(jvps[checkpoint, direction_index][None, :], p)[
                        0, 0
                    ]
                    q1 = m3.weighted_fisher_gram(finite[checkpoint][None, :], p)[0, 0]
                    q_exact.append(q0)
                    q_finite.append(q1)
                    kl_q.append(
                        2 * kl_divergence(baseline[checkpoint], plus_np[checkpoint]) / epsilon**2
                    )
                    h_q.append(
                        8
                        * hellinger_squared(baseline[checkpoint], plus_np[checkpoint])
                        / epsilon**2
                    )
                    js_q.append(
                        8 * js_divergence(baseline[checkpoint], plus_np[checkpoint]) / epsilon**2
                    )
            finite_gram, movement = finite_geometry(baseline, plus, minus, epsilon)
            finite_grams.append(finite_gram)
            movements.append(movement["rms_logit_movement"])
        aggregate_finite = aggregate_grams(finite_grams)
        finite_geometry_values = m3.gram_geometry(aggregate_finite)
        exact_cosine = np.asarray(exact_geometry["cosine"])
        finite_cosine = np.asarray(finite_geometry_values["cosine"])
        mask = np.isfinite(exact_cosine) & ~np.eye(len(exact_cosine), dtype=bool)
        finite_rows.append(
            {
                "epsilon": epsilon,
                "jvp_cosine": cosine(np.concatenate(all_exact), np.concatenate(all_finite)),
                "fisher_relative_error": float(
                    np.median(
                        [
                            relative(observed, expected)
                            for observed, expected in zip(q_finite, q_exact, strict=True)
                        ]
                    )
                ),
                "kl_relative_error": float(
                    np.median(
                        [
                            relative(observed, expected)
                            for observed, expected in zip(kl_q, q_exact, strict=True)
                        ]
                    )
                ),
                "hellinger_relative_error": float(
                    np.median(
                        [
                            relative(observed, expected)
                            for observed, expected in zip(h_q, q_exact, strict=True)
                        ]
                    )
                ),
                "js_relative_error": float(
                    np.median(
                        [
                            relative(observed, expected)
                            for observed, expected in zip(js_q, q_exact, strict=True)
                        ]
                    )
                ),
                "gram_relative_error": m3.relative_frobenius(aggregate_finite, exact_diff_gram),
                "radius_relative_error": float(
                    np.median(
                        np.abs(
                            np.asarray(finite_geometry_values["radii"])
                            - np.asarray(exact_geometry["radii"])
                        )
                        / np.maximum(np.asarray(exact_geometry["radii"]), 1e-15)
                    )
                ),
                "angle_max_abs_error": float(
                    np.max(np.abs(finite_cosine[mask] - exact_cosine[mask]))
                ),
                "rms_logit_movement": float(np.median(movements)),
            }
        )
        write_json(REVIEW / "M3_REMOTE_PROGRESS.json", {"phase": "FINITE", "epsilon": epsilon})
        torch.cuda.empty_cache()

    bf16_grams = []
    fp32_bridge_grams = []
    bf16_q_all = []
    fp32_q_all = []
    for fixture_index in m3.BRIDGE_FIXTURE_INDICES:
        diagonal_by_direction = np.asarray(
            bf16_bridge_q[fixture_index]["diagonal"], dtype=np.float64
        )
        fp32_checkpoint_grams = exact_cache[fixture_index][2]
        for checkpoint, fp32_checkpoint_gram in enumerate(fp32_checkpoint_grams):
            diag = diagonal_by_direction[:, checkpoint]
            gram = np.diag(diag.copy())
            for left in range(len(directions)):
                for right in range(left + 1, len(directions)):
                    qsum = bf16_bridge_q[fixture_index]["sums"][f"{left}:{right}"][
                        checkpoint
                    ]
                    cross = 0.5 * (qsum - diag[left] - diag[right])
                    gram[left, right] = cross
                    gram[right, left] = cross
            bf16_grams.append(gram)
            fp32_bridge_grams.append(fp32_checkpoint_gram)
            bf16_q_all.extend(diag.tolist())
            fp32_q_all.extend(np.diag(fp32_checkpoint_gram).tolist())
    bf16_gram = aggregate_grams(bf16_grams)
    fp32_bridge_gram = aggregate_grams(fp32_bridge_grams)
    bf16_geometry = m3.gram_geometry(bf16_gram)
    fp32_bridge_geometry = m3.gram_geometry(fp32_bridge_gram)
    tri = np.triu_indices(len(directions), 1)
    bf16_radii = np.asarray(bf16_geometry["radii"])
    fp32_radii = np.asarray(fp32_bridge_geometry["radii"])
    bf16_distances = np.asarray(bf16_geometry["distances"])[tri]
    fp32_distances = np.asarray(fp32_bridge_geometry["distances"])[tri]
    top_n = max(1, len(directions) // 4)
    bf16_top = set(np.argsort(bf16_radii)[-top_n:])
    bf16_bottom = set(np.argsort(bf16_radii)[:top_n])
    fp32_top = set(np.argsort(fp32_radii)[-top_n:])
    fp32_bottom = set(np.argsort(fp32_radii)[:top_n])

    RAW.mkdir(parents=True, exist_ok=True)
    sufficient_path = RAW / "m3_sufficient_statistics.npz"
    np.savez_compressed(
        sufficient_path,
        exact_gram=exact_gram,
        direct_subset=direct_subset,
        polarization_subset=polarization_subset,
        bf16_gram=bf16_gram,
        fp32_bridge_gram=fp32_bridge_gram,
        finite_metrics=np.asarray(
            [
                [
                    row[key]
                    for key in (
                        "epsilon",
                        "jvp_cosine",
                        "fisher_relative_error",
                        "kl_relative_error",
                        "hellinger_relative_error",
                        "js_relative_error",
                        "gram_relative_error",
                        "radius_relative_error",
                        "angle_max_abs_error",
                        "rms_logit_movement",
                    )
                ]
                for row in finite_rows
            ],
            dtype=np.float64,
        ),
    )
    write_json(REVIEW / "M3_FIXTURE_SUFFICIENT_STATISTICS.json", fixture_summaries)
    write_json(
        REVIEW / "M3_REMOTE_RESULTS.json",
        {
            "source_commit": args.experiment_source_commit,
            "model": m3.MODEL,
            "revision": m3.MODEL_REVISION,
            "layer": m3.LAYER,
            "parameter_hash_bf16": bf16_hash,
            "parameter_hash_fp32_lift": fp32_hash,
            "parameter_roundtrip_verified": roundtrip_hash == bf16_hash,
            "selected_fp32_attention": implementation,
            "eager_available": eager_available,
            "eager_error": eager_error,
            "fp32_sequence": summarize_comparisons(fp32_sequence_rows),
            "alpha_zero_identity": summarize_comparisons(alpha_zero_identity_rows),
            "bf16_baseline_bridge": summarize_comparisons(bf16_bridge_rows),
            "exact_gram": exact_gram.tolist(),
            "exact_gram_geometry": {
                "eigenvalues": np.asarray(m3.gram_geometry(exact_gram)["eigenvalues"]).tolist(),
                "radii": np.asarray(m3.gram_geometry(exact_gram)["radii"]).tolist(),
                "cosine": np.asarray(m3.gram_geometry(exact_gram)["cosine"]).tolist(),
            },
            "reproducibility": reproducibility,
            "exact_crosschecks": crosschecks,
            "direct_polarization_relative_frobenius": m3.relative_frobenius(
                polarization_subset, direct_subset
            ),
            "finite_ladder": finite_rows,
            "bf16_geometry_bridge": {
                "epsilon": m3.BF16_BRIDGE_EPSILON,
                "radius_spearman": spearman(fp32_radii, bf16_radii),
                "distance_spearman": spearman(fp32_distances, bf16_distances),
                "median_curvature_relative_error": float(
                    np.median(
                        [
                            relative(observed, expected)
                            for observed, expected in zip(bf16_q_all, fp32_q_all, strict=True)
                        ]
                    )
                ),
                "upper_lower_quartile_crossing": bool(
                    (bf16_top & fp32_bottom) or (fp32_top & bf16_bottom)
                ),
                "bf16_gram": bf16_gram.tolist(),
                "fp32_gram": fp32_bridge_gram.tolist(),
            },
            "raw_sufficient_statistics": str(sufficient_path.relative_to(ROOT)),
            "raw_sufficient_statistics_sha256": sha256(sufficient_path),
            "scientific_items_processed": 0,
            "semantic_outcomes_read": False,
            "free_generation_outputs": 0,
            "q2_v3_behavioral_trajectories": 0,
            "elapsed_seconds": time.time() - started,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    )
    write_json(REVIEW / "M3_REMOTE_PROGRESS.json", {"phase": "COMPLETE"})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-source-commit", required=True)
    parser.add_argument("--model-path")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
