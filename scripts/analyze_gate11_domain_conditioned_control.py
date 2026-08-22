#!/usr/bin/env python3
"""Primary model-free analysis for Gate 11."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epistemic_geometry.experiments import gate11  # noqa: E402
from epistemic_geometry.experiments.gate6 import two_rollout_estimands  # noqa: E402

REVIEW = ROOT / "review/gate11_domain_conditioned_control"
GATE9 = ROOT / "review/gate9_selected_d75_evaluation"
GATE10 = ROOT / "review/gate10_cross_domain_charcount"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval(values: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(np.mean(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(gate11.BOOTSTRAP_RESAMPLES)
    for index in range(len(draws)):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    return interval(draws) | {"point": float(values.mean())}


def bootstrap_domain_contrast(
    crux: np.ndarray, char: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    crux = np.asarray(crux, dtype=np.float64)
    char = np.asarray(char, dtype=np.float64)
    draws = np.empty(gate11.BOOTSTRAP_RESAMPLES)
    for index in range(len(draws)):
        left = crux[rng.integers(0, len(crux), len(crux))].mean()
        right = char[rng.integers(0, len(char), len(char))].mean()
        draws[index] = left - right
    return interval(draws) | {"point": float(crux.mean() - char.mean())}


def load_source_arrays(domain: str) -> dict[str, dict[int, np.ndarray]]:
    journal = read_jsonl(REVIEW / "source_activation_journal.jsonl")
    result: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in journal:
        if row["domain"] != domain:
            continue
        values = np.load(ROOT / row["activation_path"], allow_pickle=False)
        for layer in gate11.SOURCE_LAYERS:
            result[row["variant"]][layer].append(values[f"L{layer}"].astype(np.float64))
    return {
        variant: {layer: np.stack(rows) for layer, rows in layers.items()}
        for variant, layers in result.items()
    }


def source_analysis(rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    arrays = {domain: load_source_arrays(domain) for domain in ("CRUXEval", "CHARCOUNT")}
    frozen = np.load(
        ROOT
        / "review/gate6_2_first_stage_repair_mean_bridge"
        / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy",
        allow_pickle=False,
    ).astype(np.float64)
    atlas: list[dict[str, Any]] = []
    directions: dict[tuple[str, int], np.ndarray] = {}
    bootstraps: dict[str, Any] = {}
    for domain, variants in arrays.items():
        for layer in gate11.SOURCE_LAYERS:
            careful = variants["P1_SOURCE_CAREFUL"][layer]
            direct = variants["P2_SOURCE_DIRECT"][layer]
            metrics = gate11.source_axis_metrics(careful, direct)
            directions[(domain, layer)] = metrics.pop("direction")
            metrics.pop("gaps")
            atlas.append({"domain": domain, "layer": layer, **metrics})
        careful = variants["P1_SOURCE_CAREFUL"][27]
        direct = variants["P2_SOURCE_DIRECT"][27]
        gaps = (careful - direct) @ frozen
        bootstraps[f"{domain}:frozen_L27_gap"] = bootstrap_mean(gaps, rng)
        atlas.append(
            {
                "domain": domain,
                "layer": 27,
                "axis": "FROZEN_L27",
                "mean_gap": float(gaps.mean()),
                "median_gap": float(np.median(gaps)),
                "positive_gap_fraction": float(np.mean(gaps > 0)),
                "paired_standardized_effect": float(gaps.mean() / gaps.std(ddof=1)),
                "auroc": gate11._auroc(careful @ frozen, direct @ frozen),  # noqa: SLF001
                "cosine_to_frozen": gate11.cosine(directions[(domain, 27)], frozen),
            }
        )
    for layer in gate11.SOURCE_LAYERS:
        cosine = gate11.cosine(directions[("CRUXEval", layer)], directions[("CHARCOUNT", layer)])
        for row in atlas:
            if row["layer"] == layer and row.get("axis") is None:
                row["cross_domain_direction_cosine"] = cosine
                if layer == 27:
                    row["cosine_to_frozen"] = gate11.cosine(
                        directions[(row["domain"], layer)], frozen
                    )
    char_frozen = next(
        row for row in atlas if row["domain"] == "CHARCOUNT" and row.get("axis") == "FROZEN_L27"
    )
    char_interval = bootstraps["CHARCOUNT:frozen_L27_gap"]
    source_transfer = bool(
        char_frozen["mean_gap"] > 0
        and char_interval["q025"] > 0
        and char_frozen["positive_gap_fraction"] >= 0.75
        and char_frozen["cosine_to_frozen"] >= 0.20
    )
    return atlas, {
        "arrays": arrays,
        "directions": directions,
        "bootstraps": bootstraps,
        "source_transfer_supported": source_transfer,
    }


def relative_geometry(source: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = np.load(
        ROOT
        / "review/gate6_2_first_stage_repair_mean_bridge"
        / "PAIRED_MEAN_DIRECTIONS/PROMPT_BOUNDARY/L27.npy",
        allow_pickle=False,
    ).astype(np.float64)
    result = []
    for domain, variants in source["arrays"].items():
        metrics = gate11.relative_dose_geometry(
            variants["P0_ORDINARY"][27],
            variants["P1_SOURCE_CAREFUL"][27],
            variants["P2_SOURCE_DIRECT"][27],
            frozen,
            gate11.ETA * gate11.REFERENCE_SCALE,
        )
        result.append({"domain": domain, **metrics})
    return result


def propagation_analysis(
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw = [
        row
        for row in read_jsonl(REVIEW / "fixed_sequence_journal.jsonl")
        if row["status"] == "COMPLETE"
    ]
    flat: list[dict[str, Any]] = []
    item_metrics: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    delta_norm = gate11.ETA * gate11.REFERENCE_SCALE
    for row in raw:
        for checkpoint in row["checkpoints"]:
            base = {
                "domain": row["domain"],
                "item_id": row["item_id"],
                "condition": row["condition"],
                "checkpoint": checkpoint["checkpoint"],
                "next_token_kl": checkpoint["next_token_kl"],
                "symmetric_js": checkpoint["symmetric_js"],
                "logit_l2": checkpoint["logit_l2"],
                "logit_cosine": checkpoint["logit_cosine"],
                "top1_flip": int(checkpoint["top1_flip"]),
                "target_logprob_shift": checkpoint["target_logprob_shift"],
                "careful_logit_alignment": checkpoint.get("careful_logit_alignment"),
            }
            for layer in gate11.PROPAGATION_LAYERS:
                base[f"A{layer}"] = (
                    checkpoint["hidden"][f"L{layer}"]["displacement_norm"] / delta_norm
                )
            flat.append(base)
            numeric = {
                key: float(value)
                for key, value in base.items()
                if key not in {"domain", "item_id", "condition", "checkpoint"}
                and value is not None
            }
            item_metrics[(row["domain"], row["item_id"], row["condition"])].append(numeric)
    item_rows: list[dict[str, Any]] = []
    for (domain, item_id, condition), values in item_metrics.items():
        keys = set.intersection(*(set(value) for value in values))
        item_rows.append(
            {
                "domain": domain,
                "item_id": item_id,
                "condition": condition,
                **{key: float(np.mean([value[key] for value in values])) for key in keys},
            }
        )
    by_key = {(row["domain"], row["item_id"], row["condition"]): row for row in item_rows}
    control_summary: list[dict[str, Any]] = []
    alignment_summary: list[dict[str, Any]] = []
    domain_values: dict[str, dict[str, np.ndarray]] = {}
    for domain in ("CRUXEval", "CHARCOUNT"):
        item_ids = sorted({row["item_id"] for row in item_rows if row["domain"] == domain})
        metrics = ("next_token_kl", "A35", "top1_flip")
        domain_values[domain] = {}
        for metric in metrics:
            meaningful = np.asarray(
                [by_key[(domain, item, gate11.TF_MEANINGFUL)][metric] for item in item_ids]
            )
            random_matrix = np.asarray(
                [
                    [by_key[(domain, item, condition)][metric] for condition in gate11.TF_RANDOMS]
                    for item in item_ids
                ]
            )
            specific = meaningful - random_matrix.mean(axis=1)
            domain_values[domain][metric] = specific
            control_summary.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "meaningful": float(meaningful.mean()),
                    "random_mean": float(random_matrix.mean()),
                    "random_max": float(random_matrix.mean(axis=0).max()),
                    "meaningful_minus_random_mean": float(specific.mean()),
                    "meaningful_minus_random_max": float(
                        meaningful.mean() - random_matrix.mean(axis=0).max()
                    ),
                }
            )
        alignment = np.asarray(
            [
                by_key[(domain, item, gate11.TF_MEANINGFUL)].get(
                    "careful_logit_alignment", np.nan
                )
                for item in item_ids
            ]
        )
        alignment = alignment[np.isfinite(alignment)]
        domain_values[domain]["alignment"] = alignment
        alignment_summary.append(
            {"domain": domain, "mean_careful_logit_alignment": float(alignment.mean())}
        )
    contrasts = {}
    for metric in ("next_token_kl", "A35", "top1_flip", "alignment"):
        contrasts[metric] = bootstrap_domain_contrast(
            domain_values["CRUXEval"][metric], domain_values["CHARCOUNT"][metric], rng
        )
    crux_kl = next(
        row
        for row in control_summary
        if row["domain"] == "CRUXEval" and row["metric"] == "next_token_kl"
    )
    control_gain_shift = bool(
        crux_kl["meaningful_minus_random_mean"] > 0
        and sum(contrasts[name]["q025"] > 0 for name in ("next_token_kl", "A35", "top1_flip"))
        >= 2
    )
    crux_alignment = alignment_summary[0]["mean_careful_logit_alignment"]
    policy_realization_shift = bool(
        crux_alignment > 0 and contrasts["alignment"]["q025"] > 0
    )
    return flat, control_summary, alignment_summary, {
        "item_rows": item_rows,
        "bootstrap_contrasts": contrasts,
        "control_gain_shift_supported": control_gain_shift,
        "policy_realization_shift_supported": policy_realization_shift,
    }


def error_bank(
    rows: list[dict[str, Any]], baseline: str, condition: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    item_ids = sorted({str(row["item_id"]) for row in rows})
    lookup = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): int(
            not row["correct"]
        )
        for row in rows
    }
    bank = lambda name: np.asarray(  # noqa: E731
        [[lookup[(item, name, rollout)] for rollout in (0, 1)] for item in item_ids],
        dtype=np.float64,
    )
    return bank(baseline), bank(condition), item_ids


def utility_analysis(rng: np.random.Generator) -> dict[str, Any]:
    configs = {
        "CRUXEval": (
            GATE9,
            "BASELINE",
            "TEXTUAL_CAREFUL_REFERENCE",
            "MEANINGFUL_L27_D75",
        ),
        "CHARCOUNT": (
            GATE10,
            "BASELINE",
            "TEXTUAL_CAREFUL_CHARCOUNT_REFERENCE",
            "MEANINGFUL_L27_D75",
        ),
    }
    item_effects: dict[str, dict[str, np.ndarray]] = {}
    points: dict[str, Any] = {}
    banks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for domain, (path, baseline_name, textual_name, meaningful_name) in configs.items():
        rows = read_jsonl(path / "journal.jsonl")
        baseline, meaningful, _ids = error_bank(rows, baseline_name, meaningful_name)
        _, textual, _ids = error_bank(rows, baseline_name, textual_name)
        meaningful_est = two_rollout_estimands(baseline, meaningful)
        textual_est = two_rollout_estimands(baseline, textual)
        item_effects[domain] = {
            "meaningful_accuracy": baseline.mean(axis=1) - meaningful.mean(axis=1),
            "textual_accuracy": baseline.mean(axis=1) - textual.mean(axis=1),
        }
        points[domain] = {
            "meaningful_accuracy_change": meaningful_est["accuracy_condition"]
            - meaningful_est["accuracy_baseline"],
            "textual_accuracy_change": textual_est["accuracy_condition"]
            - textual_est["accuracy_baseline"],
            "meaningful_estimands": meaningful_est,
            "textual_estimands": textual_est,
        }
        banks[domain] = (baseline, meaningful)
    contrasts = {
        name: bootstrap_domain_contrast(
            item_effects["CRUXEval"][name], item_effects["CHARCOUNT"][name], rng
        )
        for name in ("meaningful_accuracy", "textual_accuracy")
    }
    gcd_draws = {name: np.empty(gate11.BOOTSTRAP_RESAMPLES) for name in ("G", "C", "D")}
    for index in range(gate11.BOOTSTRAP_RESAMPLES):
        estimates = {}
        for domain, (baseline, meaningful) in banks.items():
            selected = rng.integers(0, len(baseline), len(baseline))
            estimates[domain] = two_rollout_estimands(baseline[selected], meaningful[selected])
        for name in gcd_draws:
            gcd_draws[name][index] = estimates["CRUXEval"][name] - estimates["CHARCOUNT"][name]
    contrasts.update({name: interval(values) for name, values in gcd_draws.items()})
    supported = bool(
        contrasts["meaningful_accuracy"]["q025"] > 0
        and contrasts["textual_accuracy"]["q025"] > 0
    )
    return {
        "domains": points,
        "domain_contrasts": contrasts,
        "policy_utility_shift_supported": supported,
    }


def report(
    classification: str,
    components: dict[str, Any],
    relative: list[dict[str, Any]],
    control: list[dict[str, Any]],
    alignment: list[dict[str, Any]],
    utility: dict[str, Any],
) -> None:
    text = f"""GATE 11 — DOMAIN-CONDITIONED CONTROL POSTMORTEM
======================================================================

No new free generation or semantic evaluation was performed. Prompt-only
activations and teacher-forced historical baseline sequences were used.

COMPONENT DIAGNOSTICS
----------------------------------------------------------------------

source-axis transfer: {components['source_axis']}
downstream control-gain shift: {components['control_gain']}
policy-realization shift: {components['policy_realization']}
policy-utility shift: {components['policy_utility']}

Relative-dose geometry:

{json.dumps(relative, indent=2)}

Control-gain summary:

{json.dumps(control, indent=2)}

Careful-alignment summary:

{json.dumps(alignment, indent=2)}

Historical utility reanalysis:

{json.dumps(utility, indent=2)}

PRIMARY SYNTHESIS
----------------------------------------------------------------------

{classification}

MEASUREMENT DISTINCTIONS
----------------------------------------------------------------------

1. Source-axis gaps, AUROC, and direction cosines measure representation
   transfer.
2. D75 next-token KL/JS and downstream hidden displacement are finite-
   displacement control-gain diagnostics.
3. Gate 11 did not measure an exact local pullback metric and did not establish
   Fisher geometry.
4. Historical accuracy and G/C/D measure task utility, not control energy.

RAW-PERSISTENCE BOUNDARY
----------------------------------------------------------------------

Prompt-boundary activations were preserved in float32. The fixed-sequence
journal preserved per-item/per-condition/per-checkpoint scalar logit metrics,
hidden displacement norms, token checkpoints, target-token indexing, D75
normalization, and provenance. It did not preserve complete per-checkpoint
vocabulary-logit arrays or hidden-state difference vectors. Consequently the
primitive KL/JS/vector calculations cannot be independently recomputed from
the recovered artifact alone; no replacement diagnostic collection was run.

INTERPRETATION BOUNDARY
----------------------------------------------------------------------

This DEVELOPMENT postmortem localizes candidate domain conditioning. It does
not establish Q2, optimize a controller, score new semantic responses, or touch
the confirmatory holdout.
"""
    (REVIEW / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    rng = np.random.default_rng(gate11.BOOTSTRAP_SEED)
    atlas, source = source_analysis(rng)
    relative = relative_geometry(source)
    flat, control, alignment, propagation = propagation_analysis(rng)
    utility = utility_analysis(rng)
    components = {
        "source_axis": (
            "SOURCE_AXIS_TRANSFER_SUPPORTED"
            if source["source_transfer_supported"]
            else "SOURCE_AXIS_MISMATCH"
        ),
        "control_gain": (
            "CONTROL_GAIN_DOMAIN_SHIFT_SUPPORTED"
            if propagation["control_gain_shift_supported"]
            else "CONTROL_GAIN_SHIFT_NOT_ESTABLISHED"
        ),
        "policy_realization": (
            "POLICY_REALIZATION_DOMAIN_SHIFT_SUPPORTED"
            if propagation["policy_realization_shift_supported"]
            else "POLICY_REALIZATION_SHIFT_NOT_ESTABLISHED"
        ),
        "policy_utility": (
            "POLICY_UTILITY_DOMAIN_SHIFT_SUPPORTED"
            if utility["policy_utility_shift_supported"]
            else "POLICY_UTILITY_SHIFT_NOT_ESTABLISHED"
        ),
    }
    classification = gate11.classify_components(
        source_transfer=source["source_transfer_supported"],
        control_gain_shift=propagation["control_gain_shift_supported"],
        policy_realization_shift=propagation["policy_realization_shift_supported"],
        policy_utility_shift=utility["policy_utility_shift_supported"],
    )
    write_csv(REVIEW / "SOURCE_AXIS_ATLAS.csv", atlas)
    write_json(REVIEW / "SOURCE_AXIS_ATLAS.json", {"rows": atlas})
    write_csv(REVIEW / "RELATIVE_DOSE_GEOMETRY.csv", relative)
    write_json(REVIEW / "RELATIVE_DOSE_GEOMETRY.json", {"rows": relative})
    write_csv(REVIEW / "FIXED_SEQUENCE_PROPAGATION.csv", flat)
    write_csv(REVIEW / "CONTROL_GAIN_SUMMARY.csv", control)
    write_csv(REVIEW / "CAREFUL_ALIGNMENT_SUMMARY.csv", alignment)
    write_json(REVIEW / "POLICY_UTILITY_REANALYSIS.json", utility)
    write_json(
        REVIEW / "BOOTSTRAP_INTERVALS.json",
        {
            "source": source["bootstraps"],
            "propagation_domain_contrasts": propagation["bootstrap_contrasts"],
            "utility_domain_contrasts": utility["domain_contrasts"],
        },
    )
    write_json(
        REVIEW / "COMPONENT_DIAGNOSTICS.json",
        {"classification": classification, "components": components},
    )
    report(classification, components, relative, control, alignment, utility)
    print(json.dumps({"classification": classification, "components": components}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
