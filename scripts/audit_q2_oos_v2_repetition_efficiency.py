#!/usr/bin/env python3
"""Single permitted EXTREME_MECHANICAL_REPETITION_V1 efficiency replay.

This additive audit is reachable only after the frozen hard-cap family has no
qualifying member at or below 1,024 tokens.  It uses the unchanged historical
criterion and treats an online repetition stop as the prospectively defined
terminal answer-channel failure.  No text is decoded or emitted.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.audit_q2_oos_v2_semantic_efficiency import (  # noqa: E402
    RUNTIME_SEED,
    K,
    controller_from_condition,
    controller_partition,
    distribution,
    endpoint_tuple,
    exact,
    fit_runtime,
    load_references,
    load_scores,
    runtime_report,
    scientific_certification,
    sha256_file,
    unwrap_journal,
    write_json,
)
from scripts.posthoc_diagnose_q2_v4_1_generation import (  # noqa: E402
    DOMINANT_TOKEN_THRESHOLD,
    MAX_PERIOD_TOKENS,
    MIN_REPETITION_TOKENS,
    PERIODIC_MATCH_THRESHOLD,
    TAIL_WINDOW_TOKENS,
)

CRITERION_SOURCE = ROOT / "scripts/posthoc_diagnose_q2_v4_1_generation.py"
CRITERION_SOURCE_SHA256 = "e340f9d622c4f874a868c0d9a4e203005b8bcfe778ab688f2202f35a39bda24e"
TERMINAL_FAILURE = (False, False, False, 1)


def earliest_dominant_stop(tokens: list[int]) -> int | None:
    counts: Counter[int] = Counter()
    heap: list[tuple[int, int]] = []
    for end, token in enumerate(tokens, 1):
        counts[token] += 1
        heapq.heappush(heap, (-counts[token], token))
        start = max(0, end - TAIL_WINDOW_TOKENS)
        if start > 0:
            removed = tokens[start - 1]
            counts[removed] -= 1
            heapq.heappush(heap, (-counts[removed], removed))
        while heap and -heap[0][0] != counts[heap[0][1]]:
            heapq.heappop(heap)
        if end >= MIN_REPETITION_TOKENS:
            window = end - start
            if heap and (-heap[0][0]) / window >= DOMINANT_TOKEN_THRESHOLD:
                return end
    return None


def earliest_periodic_stop(tokens: list[int]) -> int | None:
    values = np.asarray(tokens, dtype=np.int64)
    length = len(values)
    earliest: int | None = None
    for period in range(1, min(MAX_PERIOD_TOKENS, length - 1) + 1):
        matches = np.zeros(length + 1, dtype=np.int64)
        indicators = values[period:] == values[:-period]
        matches[period + 1 :] = np.cumsum(indicators, dtype=np.int64)
        ends = np.arange(max(MIN_REPETITION_TOKENS, period + 1), length + 1)
        starts = np.maximum(0, ends - TAIL_WINDOW_TOKENS)
        comparisons = ends - starts - period
        totals = matches[ends] - matches[starts + period]
        passing = ends[totals / comparisons >= PERIODIC_MATCH_THRESHOLD]
        if len(passing):
            candidate = int(passing[0])
            earliest = candidate if earliest is None else min(earliest, candidate)
    return earliest


def earliest_repetition_stop(token_ids: list[int]) -> int | None:
    if len(token_ids) < MIN_REPETITION_TOKENS:
        return None
    dominant = earliest_dominant_stop(token_ids)
    periodic = earliest_periodic_stop(token_ids)
    values = [value for value in (dominant, periodic) if value is not None]
    return min(values) if values else None


def replay_repetition(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[
    dict[str, int],
    dict[tuple[str, str, int], tuple[bool, bool, bool, int]],
    dict[tuple[str, str, int], int],
    int,
]:
    differences = {
        "commitment_valid": 0,
        "semantic_evaluable": 0,
        "correct": 0,
        "binary_error_e": 0,
    }
    replayed: dict[tuple[str, str, int], tuple[bool, bool, bool, int]] = {}
    stops: dict[tuple[str, str, int], int] = {}
    valid_evaluable_stops = 0
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        old = endpoint_tuple(scores[key])
        stop = earliest_repetition_stop([int(value) for value in row["generated_token_ids"]])
        new = TERMINAL_FAILURE if stop is not None else old
        if stop is not None:
            stops[key] = stop
            valid_evaluable_stops += int(old[0] and old[1])
        replayed[key] = new
        for index, field in enumerate(differences):
            differences[field] += int(old[index] != new[index])
    return differences, replayed, stops, valid_evaluable_stops


def repetition_runtime_report(
    rows: list[dict[str, Any]],
    stops: dict[tuple[str, str, int], int],
    development: set[str],
    validation: set[str],
) -> dict[str, Any]:
    dev_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in development
    ]
    intercept, slope = fit_runtime(dev_rows)
    hard_caps = runtime_report(rows, development, validation, 4096)
    times = []
    stop_flags = []
    for row in rows:
        key = (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"]))
        stop = stops.get(key)
        times.append(
            float(row["elapsed_seconds"])
            if stop is None
            else max(0.0, intercept + slope * stop)
        )
        stop_flags.append(stop is not None)
    values = np.asarray(times, dtype=np.float64)
    old = float(sum(float(row["elapsed_seconds"]) for row in rows))
    total = float(np.sum(values))

    by_controller: dict[str, list[float]] = defaultdict(list)
    flags_by_controller: dict[str, list[bool]] = defaultdict(list)
    for row, elapsed, stopped in zip(rows, values, stop_flags, strict=True):
        controller = controller_from_condition(str(row["condition"]))
        if controller is not None:
            by_controller[controller].append(float(elapsed))
            flags_by_controller[controller].append(stopped)
    controller_ids = sorted(by_controller)
    if len(controller_ids) != K:
        raise RuntimeError("repetition runtime controller count mismatch")
    profiles = np.asarray([sum(by_controller[controller]) for controller in controller_ids])
    rng = np.random.Generator(np.random.PCG64DXSM(RUNTIME_SEED))
    draws = rng.integers(0, K, size=(100_000, 16))
    future = np.sum(profiles[draws], axis=1)
    stress: dict[str, Any] = {}
    for multiplier in (1.5, 2.0):
        expected = []
        for controller in controller_ids:
            controller_values = np.asarray(by_controller[controller], dtype=np.float64)
            flags = np.asarray(flags_by_controller[controller], dtype=bool)
            rate = float(np.mean(flags))
            stopped_mean = (
                float(np.mean(controller_values[flags]))
                if np.any(flags)
                else max(0.0, intercept + slope * MIN_REPETITION_TOKENS)
            )
            natural_mean = (
                float(np.mean(controller_values[~flags]))
                if np.any(~flags)
                else float(np.mean(controller_values))
            )
            stressed_rate = min(1.0, multiplier * rate)
            expected.append(
                1200.0
                * (stressed_rate * stopped_mean + (1.0 - stressed_rate) * natural_mean)
            )
        stress[str(multiplier)] = distribution(np.sum(np.asarray(expected)[draws], axis=1))
    return {
        "OLS": hard_caps["model"],
        "hard_cap_counterfactuals": hard_caps["candidate_counterfactuals"],
        "selected_repetition_policy": {
            "historical_counterfactual_hours": total / 3600.0,
            "hours_saved": (old - total) / 3600.0,
            "percent_saved": 100.0 * (old - total) / old,
            "stopped_rows": int(sum(stop_flags)),
            "future_19200_seconds": distribution(future),
            "stress_future_19200_seconds": stress,
        },
        "observed_historical_hours": old / 3600.0,
        "runtime_used_for_policy_selection": False,
        "interpretation": "MODEL_ESTIMATED_COUNTERFACTUAL_RUNTIME_NOT_OBSERVED_EXECUTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--hard-cap-development-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if sha256_file(CRITERION_SOURCE) != CRITERION_SOURCE_SHA256:
        raise RuntimeError("EXTREME_MECHANICAL_REPETITION_V1 source changed")
    hard_cap = json.loads(Path(args.hard_cap_development_result).read_text())
    if hard_cap.get("status") != "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY":
        raise RuntimeError("optional repetition path is not authorized by hard-cap result")
    if hard_cap.get("validation_opened") is not False:
        raise RuntimeError("hard-cap validation was unexpectedly opened")

    rows = unwrap_journal(Path(args.journal))
    scores = load_scores(Path(args.scores))
    item_ids, _references = load_references()
    controller_ids = sorted(
        {
            controller
            for row in rows
            if (controller := controller_from_condition(str(row["condition"]))) is not None
        }
    )
    development, validation, _digests = controller_partition(controller_ids)
    dev_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in development
    ]
    validation_rows = [
        row
        for row in rows
        if controller_from_condition(str(row["condition"])) in validation
    ]

    dev_differences, _dev_replayed, dev_stops, dev_valid_stops = replay_repetition(
        dev_rows, scores
    )
    dev_qualifies = exact(dev_differences) and dev_valid_stops == 0
    if not dev_qualifies:
        payload = {
            "schema_version": "q2-oos-v2-repetition-efficiency-replay-v1",
            "status": "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY",
            "criterion": "EXTREME_MECHANICAL_REPETITION_V1_UNCHANGED",
            "criterion_source_sha256": CRITERION_SOURCE_SHA256,
            "development": {
                "row_level_differences": dev_differences,
                "early_stops": len(dev_stops),
                "valid_evaluable_early_stops": dev_valid_stops,
                "exact_endpoint_equivalence": exact(dev_differences),
            },
            "validation_opened": False,
            "model_inference": 0,
            "raw_text_decoded_printed_or_persisted": False,
        }
        write_json(Path(args.output), payload)
        print(json.dumps({"status": payload["status"]}))
        return

    val_differences, _val_replayed, val_stops, val_valid_stops = replay_repetition(
        validation_rows, scores
    )
    validation_qualifies = exact(val_differences) and val_valid_stops == 0
    if not validation_qualifies:
        payload = {
            "schema_version": "q2-oos-v2-repetition-efficiency-replay-v1",
            "status": "Q2_OOS_V2_NO_ENDPOINT_EQUIVALENT_STOP_POLICY",
            "criterion": "EXTREME_MECHANICAL_REPETITION_V1_UNCHANGED",
            "criterion_source_sha256": CRITERION_SOURCE_SHA256,
            "development": {
                "row_level_differences": dev_differences,
                "early_stops": len(dev_stops),
                "valid_evaluable_early_stops": dev_valid_stops,
            },
            "validation": {
                "opened_once": True,
                "row_level_differences": val_differences,
                "early_stops": len(val_stops),
                "valid_evaluable_early_stops": val_valid_stops,
            },
            "no_fallback_after_validation_failure": True,
            "model_inference": 0,
            "raw_text_decoded_printed_or_persisted": False,
        }
        write_json(Path(args.output), payload)
        print(json.dumps({"status": payload["status"]}))
        return

    full_differences, replayed, full_stops, full_valid_stops = replay_repetition(rows, scores)
    scientific = scientific_certification(item_ids, controller_ids, scores, replayed)
    qualifies = bool(
        exact(full_differences)
        and full_valid_stops == 0
        and scientific["exact_scientific_equivalence"]
    )
    if not qualifies:
        raise RuntimeError("repetition policy failed complete historical certification")
    runtime = repetition_runtime_report(rows, full_stops, development, validation)
    payload = {
        "schema_version": "q2-oos-v2-repetition-efficiency-replay-v1",
        "status": "Q2_OOS_V2_ENDPOINT_EQUIVALENT_REPETITION_STOP_QUALIFIED",
        "criterion": {
            "name": "EXTREME_MECHANICAL_REPETITION_V1",
            "source_sha256": CRITERION_SOURCE_SHA256,
            "minimum_generated_tokens": MIN_REPETITION_TOKENS,
            "tail_window_tokens": TAIL_WINDOW_TOKENS,
            "maximum_period_tokens": MAX_PERIOD_TOKENS,
            "periodic_match_threshold": PERIODIC_MATCH_THRESHOLD,
            "dominant_token_share_threshold": DOMINANT_TOKEN_THRESHOLD,
        },
        "development": {
            "rows": len(dev_rows),
            "row_level_differences": dev_differences,
            "early_stops": len(dev_stops),
            "valid_evaluable_early_stops": dev_valid_stops,
        },
        "held_out_validation": {
            "opened_once": True,
            "rows": len(validation_rows),
            "row_level_differences": val_differences,
            "early_stops": len(val_stops),
            "valid_evaluable_early_stops": val_valid_stops,
        },
        "full_historical_certification": {
            "rows": len(rows),
            "row_level_differences": full_differences,
            "early_stops": len(full_stops),
            "valid_evaluable_early_stops": full_valid_stops,
            "scientific": scientific,
        },
        "selected_policy": {
            "semantic_max_new_tokens": 4096,
            "repetition_stop": "EXTREME_MECHANICAL_REPETITION_V1",
            "safety_max_new_tokens": 4096,
        },
        "runtime": runtime,
        "V2_safety_results_used": False,
        "V2_semantic_outcomes_observed": 0,
        "model_inference": 0,
        "raw_text_decoded_printed_or_persisted": False,
    }
    write_json(Path(args.output), payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "historical_early_stops": len(full_stops),
                "exact_full_endpoint_equivalence": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
