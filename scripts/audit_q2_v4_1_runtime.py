#!/usr/bin/env python3
# ruff: noqa: E501
"""Allowlist-only operational autopsy of the sealed Q2 V4.1 journal."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epistemic_geometry.experiments.q2_v4 import average_ranks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/q2_oos_fresh_controller_design/heterogeneity_robust_inference"
EXPECTED_SHA256 = "d726b473feca8c6922b545bdf8a217e8171c8267697ff2b9714b14e1a0363a99"
ANON_PREFIX = "Q2-OOS-V2-RUNTIME-ANON|74ba16ad03d63dd680bb85e8a4486e29eaa12c9c|"
SHELLS = ("MEDIUM", "STRONG")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anonymize(condition: str) -> tuple[str, str]:
    if condition == "BASELINE":
        return "BASELINE", "BASELINE"
    shell = next((value for value in SHELLS if condition.endswith(f"_{value}")), None)
    if shell is None:
        raise RuntimeError("unrecognized condition shell")
    controller = condition[: -(len(shell) + 1)]
    anonymous = hashlib.sha256(f"{ANON_PREFIX}{controller}".encode()).hexdigest()[:12]
    return anonymous, shell


def allowlist_read(path: Path) -> list[dict[str, Any]]:
    if sha256(path) != EXPECTED_SHA256:
        raise RuntimeError("sealed journal hash mismatch")
    metadata = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            wrapper = json.loads(line)
            row = wrapper["row"]
            controller, shell = anonymize(str(row["condition"]))
            metadata.append(
                {
                    "controller": controller,
                    "shell": shell,
                    "rollout": int(row["rollout_index"]),
                    "generated_token_count": int(row["generated_token_count"]),
                    "elapsed_seconds": float(row["elapsed_seconds"]),
                    "truncated": bool(row["truncated"]),
                    "retry_count": int(row.get("retry_count", 0)),
                    "runtime_error": row.get("runtime_error") is not None,
                    "schedule_index": int(row["schedule_index"]),
                    "source_line": line_number,
                }
            )
            del row, wrapper, line
    return metadata


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.quantile(values, q))
        for name, q in (("mean", None), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99), ("p995", 0.995), ("max", 1.0))
        if q is not None
    } | {"mean": float(np.mean(values))}


def concentration(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.sum(np.partition(values, len(values) - count)[-count:]) / np.sum(values))


def correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    pearson = float(np.corrcoef(left, right)[0, 1])
    ranks_left = average_ranks(left)
    ranks_right = average_ranks(right)
    spearman = float(np.corrcoef(ranks_left, ranks_right)[0, 1])
    return {"Pearson": pearson, "Spearman": spearman}


def distribution(values: np.ndarray) -> dict[str, float]:
    return {name: float(np.quantile(values, q)) for name, q in (("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p80", 0.80), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))} | {"mean": float(np.mean(values))}


def campaign_draws(profile_totals: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng_finite = np.random.Generator(np.random.PCG64DXSM(seed))
    random_keys = rng_finite.random((100000, len(profile_totals)))
    finite_indices = np.argpartition(random_keys, 16, axis=1)[:, :16]
    finite = np.sum(profile_totals[finite_indices], axis=1)
    rng_super = np.random.Generator(np.random.PCG64DXSM(seed ^ 0x9E3779B97F4A7C15))
    super_indices = rng_super.integers(0, len(profile_totals), size=(100000, 16))
    superpopulation = np.sum(profile_totals[super_indices], axis=1)
    return finite, superpopulation, super_indices


def prefix_backtest(
    elapsed_cube: np.ndarray,
    profile_totals: np.ndarray,
    prior: dict[str, float],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    draws = rng.integers(0, len(profile_totals), size=(10000, 16))
    checkpoints = (256, 512, 1024, 2048)
    rows = []
    for checkpoint in checkpoints:
        positions = []
        for position in range(checkpoint):
            q, slot = divmod(position, 64)
            combination = (slot + 17 * q) % 64
            controller = combination % 16
            shell = (combination // 16) % 2
            rollout = (combination // 32) % 2
            item = (q + 37 * combination) % 300
            positions.append((controller, shell, rollout, item))
        observed = np.zeros(len(draws), dtype=np.float64)
        historical_prefix_population = []
        for controller, shell, rollout, item in positions:
            observed += elapsed_cube[draws[:, controller], shell, rollout, item]
            historical_prefix_population.append(float(np.mean(elapsed_cube[:, shell, rollout, item])))
        actual = np.sum(profile_totals[draws], axis=1)
        naive = observed / checkpoint * 19200
        ratio = (observed / checkpoint) / np.mean(historical_prefix_population)
        weight = min(checkpoint / 2048.0, 1.0)
        scale = np.exp(weight * np.log(np.maximum(ratio, 1e-12)))
        forecasts = {name: scale * prior[name] for name in ("p50", "p80", "p95")}
        rows.append(
            {
                "checkpoint": checkpoint,
                "label": "EARLY_UNSTABLE_RUNTIME_ESTIMATE" if checkpoint < 1024 else "TAIL_AWARE_RUNTIME_ESTIMATE",
                "naive_median_absolute_percentage_error": float(np.median(np.abs(naive - actual) / actual)),
                "tail_p50_median_absolute_percentage_error": float(np.median(np.abs(forecasts["p50"] - actual) / actual)),
                "tail_p80_upper_coverage": float(np.mean(actual <= forecasts["p80"])),
                "tail_p95_upper_coverage": float(np.mean(actual <= forecasts["p95"])),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    args = parser.parse_args()
    journal = Path(args.journal)
    rows = allowlist_read(journal)
    if len(rows) != 37800:
        raise RuntimeError("unexpected historical row count")
    elapsed = np.asarray([row["elapsed_seconds"] for row in rows])
    tokens = np.asarray([row["generated_token_count"] for row in rows])
    truncated = np.asarray([row["truncated"] for row in rows], dtype=bool)
    controller_rows = [row for row in rows if row["controller"] != "BASELINE"]
    controller_ids = sorted({row["controller"] for row in controller_rows})
    if len(controller_ids) != 31:
        raise RuntimeError("historical controller count is not 31")
    index = {controller: position for position, controller in enumerate(controller_ids)}
    cube = np.full((31, 2, 2, 300), np.nan)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in controller_rows:
        grouped[(row["controller"], row["shell"], row["rollout"])].append(row)
    for (controller, shell, rollout), group in grouped.items():
        ordered = sorted(group, key=lambda row: row["schedule_index"])
        if len(ordered) != 300:
            raise RuntimeError("controller-shell-rollout group is not 300")
        cube[index[controller], SHELLS.index(shell), rollout] = [row["elapsed_seconds"] for row in ordered]
    if np.any(~np.isfinite(cube)):
        raise RuntimeError("runtime cube incomplete")
    profile_totals = np.sum(cube, axis=(1, 2, 3))
    finite, superpopulation, super_indices = campaign_draws(profile_totals, 181879861245714525386395017528639201702)
    finite_prior = distribution(finite)
    super_prior = distribution(superpopulation)
    controller_profile_rows = []
    for controller in controller_ids:
        values = [row for row in controller_rows if row["controller"] == controller]
        controller_profile_rows.append(
            {
                "anonymous_controller": controller,
                "total_hours": sum(row["elapsed_seconds"] for row in values) / 3600.0,
                "MEDIUM_hours": sum(row["elapsed_seconds"] for row in values if row["shell"] == "MEDIUM") / 3600.0,
                "STRONG_hours": sum(row["elapsed_seconds"] for row in values if row["shell"] == "STRONG") / 3600.0,
                "capped_fraction": float(np.mean([row["truncated"] for row in values])),
            }
        )
    stress = {}
    capped_means = np.zeros((31, 2))
    noncapped_means = np.zeros((31, 2))
    cap_rates = np.zeros((31, 2))
    for controller in controller_ids:
        for shell in SHELLS:
            values = [row for row in controller_rows if row["controller"] == controller and row["shell"] == shell]
            position = index[controller]
            shell_position = SHELLS.index(shell)
            caps = np.asarray([row["truncated"] for row in values], dtype=bool)
            times = np.asarray([row["elapsed_seconds"] for row in values])
            cap_rates[position, shell_position] = np.mean(caps)
            capped_means[position, shell_position] = np.mean(times[caps]) if np.any(caps) else np.mean(times)
            noncapped_means[position, shell_position] = np.mean(times[~caps]) if np.any(~caps) else np.mean(times)
    for multiplier in (1.0, 1.5, 2.0):
        expected_profiles = np.sum(600 * (np.minimum(multiplier * cap_rates, 1.0) * capped_means + (1.0 - np.minimum(multiplier * cap_rates, 1.0)) * noncapped_means), axis=1)
        stress[str(multiplier)] = distribution(np.sum(expected_profiles[super_indices], axis=1))
    population_summaries = {}
    for name, subset in {
        "all_rows": rows,
        "baseline": [row for row in rows if row["controller"] == "BASELINE"],
        "controller_rows": controller_rows,
        "MEDIUM": [row for row in controller_rows if row["shell"] == "MEDIUM"],
        "STRONG": [row for row in controller_rows if row["shell"] == "STRONG"],
    }.items():
        values = np.asarray([row["elapsed_seconds"] for row in subset])
        population_summaries[name] = quantiles(values)
    result = {
        "schema_version": "q2-v4-1-runtime-autopsy-v1",
        "journal_sha256": EXPECTED_SHA256,
        "rows": len(rows),
        "total_generation_hours": float(np.sum(elapsed) / 3600.0),
        "population_runtime_seconds": population_summaries,
        "token_quantiles": quantiles(tokens),
        "elapsed_token_correlation": correlation(elapsed, tokens),
        "time_concentration": {"top_1_percent": concentration(elapsed, 0.01), "top_5_percent": concentration(elapsed, 0.05), "top_10_percent": concentration(elapsed, 0.10)},
        "capped_rows": int(np.sum(truncated)),
        "capped_row_fraction": float(np.mean(truncated)),
        "capped_fraction_total_time": float(np.sum(elapsed[truncated]) / np.sum(elapsed)),
        "capped_mean_seconds": float(np.mean(elapsed[truncated])),
        "noncapped_mean_seconds": float(np.mean(elapsed[~truncated])),
        "retry_rows": int(np.sum([row["retry_count"] > 0 for row in rows])),
        "runtime_error_rows": int(np.sum([row["runtime_error"] for row in rows])),
        "finite_bank_16_of_31_seconds": finite_prior,
        "controller_superpopulation_16_seconds": super_prior,
        "exact_descriptive_worst_16_seconds": float(np.sum(np.sort(profile_totals)[-16:])),
        "cap_propensity_stress_superpopulation_seconds": stress,
        "early_prefix_backtest": prefix_backtest(cube, profile_totals, super_prior, 181879861245714525386395017528639201702 ^ 17),
        "raw_text_inspected": False,
        "correctness_inspected": False,
        "scientific_metrics_inspected": False,
    }
    write_json(OUT / "HISTORICAL_Q2_RUNTIME_AUTOPSY.json", result)
    write_csv(OUT / "HISTORICAL_Q2_RUNTIME_CONTROLLER_PROFILES.csv", controller_profile_rows)
    print(json.dumps({"total_generation_hours": result["total_generation_hours"], "finite": finite_prior, "superpopulation": super_prior}, indent=2))


if __name__ == "__main__":
    main()
