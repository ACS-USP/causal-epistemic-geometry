"""Deterministic Q1 figure-table derivations from validated frozen artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from epistemic_geometry.experiments import q1_confirmatory

TABLE_DIR = Path("manuscript/data/paper1/derived_figure_tables")
MODEL_ORDER = {"Qwen": 0, "Ministral": 1}
CONDITION_ORDER = {
    "BASELINE": 0,
    "TEXTUAL_CAREFUL": 1,
    "MEANINGFUL_FIXED": 2,
    "RANDOM_R0": 3,
    "RANDOM_R1": 4,
    "RANDOM_R2": 5,
    "RANDOM_R3": 6,
}


def _error(row: Mapping[str, Any]) -> int:
    invalid = not bool(row["commitment_valid"]) or not bool(row["semantic_evaluable"])
    if invalid and bool(row["correct"]):
        raise RuntimeError("frozen Q1 row is invalid/unevaluable but marked correct")
    return int(not bool(row["correct"]))


def _arrays(rows: list[dict[str, Any]], item_ids: list[str]) -> dict[str, np.ndarray]:
    lookup = {
        (str(row["item_id"]), str(row["condition"]), int(row["rollout_index"])): _error(row)
        for row in rows
    }
    return {
        condition: np.asarray(
            [[lookup[(item_id, condition, rollout)] for rollout in (0, 1)] for item_id in item_ids],
            dtype=np.int8,
        )
        for condition in q1_confirmatory.CONDITIONS
    }


def confirmatory_item_profiles(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model, rows in (("Qwen", data["qwen_rows"]), ("Ministral", data["ministral_rows"])):
        lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows:
            lookup.setdefault((str(row["item_id"]), str(row["condition"])), []).append(row)
        for item_index, item_id in enumerate(data["item_ids"], start=1):
            for condition in ("BASELINE", "MEANINGFUL_FIXED"):
                pair = sorted(
                    lookup[(item_id, condition)], key=lambda row: int(row["rollout_index"])
                )
                errors = [_error(row) for row in pair]
                invalid = [
                    int(not bool(row["commitment_valid"]) or not bool(row["semantic_evaluable"]))
                    for row in pair
                ]
                records.append(
                    {
                        "model_role": model,
                        "item_index": item_index,
                        "item_id": item_id,
                        "condition": condition,
                        "error_rollout_0": errors[0],
                        "error_rollout_1": errors[1],
                        "q_hat_error": float(np.mean(errors)),
                        "invalid_rollouts": int(sum(invalid)),
                        "valid_error_rollouts": int(sum(errors) - sum(invalid)),
                    }
                )
    return pd.DataFrame.from_records(records)


def transition_decomposition(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    frozen = data["confirmatory"]["models"]
    for model, rows in (("Qwen", data["qwen_rows"]), ("Ministral", data["ministral_rows"])):
        arrays = _arrays(rows, data["item_ids"])
        baseline = arrays["BASELINE"].astype(float)
        meaningful = arrays["MEANINGFUL_FIXED"].astype(float)
        cross = {
            "shared_correct": (1 - baseline)[:, :, None] * (1 - meaningful)[:, None, :],
            "rescue": baseline[:, :, None] * (1 - meaningful)[:, None, :],
            "damage": (1 - baseline)[:, :, None] * meaningful[:, None, :],
            "shared_error": baseline[:, :, None] * meaningful[:, None, :],
        }
        values = {name: float(matrix.mean()) for name, matrix in cross.items()}
        if not np.isclose(sum(values.values()), 1.0, atol=1e-12, rtol=0):
            raise RuntimeError("cross-rollout decomposition does not sum to one")
        point = frozen[model]["estimands"]["MEANINGFUL_FIXED"]
        if not np.isclose(values["rescue"], point["rescue"], atol=1e-12, rtol=0):
            raise RuntimeError(f"{model} rescue decomposition does not reconcile")
        if not np.isclose(values["damage"], point["damage"], atol=1e-12, rtol=0):
            raise RuntimeError(f"{model} damage decomposition does not reconcile")
        for component in ("shared_correct", "rescue", "damage", "shared_error"):
            records.append(
                {
                    "model_role": model,
                    "component": component,
                    "fraction": values[component],
                    "rollout_convention": "all_four_cross_products",
                }
            )
    return pd.DataFrame.from_records(records)


def confirmatory_effects(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in ("Qwen", "Ministral"):
        result = data["confirmatory"]["models"][model]
        for condition in ("MEANINGFUL_FIXED", *q1_confirmatory.RANDOM_NAMES):
            point = result["estimands"][condition]
            interval = (
                result["intervals"].get("C_meaningful") if condition == "MEANINGFUL_FIXED" else None
            )
            records.append(
                {
                    "model_role": model,
                    "condition": condition,
                    "controller_kind": "meaningful"
                    if condition == "MEANINGFUL_FIXED"
                    else "random",
                    "C": float(point["C"]),
                    "C_ci_lower": float(interval["q025"]) if interval else np.nan,
                    "C_ci_upper": float(interval["q975"]) if interval else np.nan,
                    "accuracy_change": float(
                        point["accuracy_condition"] - point["accuracy_baseline"]
                    ),
                    "rescue": float(point["rescue"]),
                    "damage": float(point["damage"]),
                }
            )
        records.append(
            {
                "model_role": model,
                "condition": "RANDOM_MEAN",
                "controller_kind": "random_mean",
                "C": float(np.mean(list(result["null_C_values"].values()))),
                "C_ci_lower": np.nan,
                "C_ci_upper": np.nan,
                "accuracy_change": np.nan,
                "rescue": np.nan,
                "damage": np.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def confirmatory_safety(data: dict[str, Any]) -> pd.DataFrame:
    margin = float(data["analysis_lock"]["safety"]["commitment_relative_margin"])
    records: list[dict[str, Any]] = []
    for model in ("Qwen", "Ministral"):
        result = data["confirmatory"]["models"][model]
        summaries = result["summaries"]
        for metric in ("commitment_validity", "semantic_evaluability"):
            baseline = float(summaries["BASELINE"][metric])
            records.append(
                {
                    "model_role": model,
                    "metric": metric,
                    "baseline": baseline,
                    "meaningful": float(summaries["MEANINGFUL_FIXED"][metric]),
                    "relative_floor": baseline + margin,
                    "model_pass": bool(result["model_pass"]),
                }
            )
    return pd.DataFrame.from_records(records)


def genealogy(data: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame.from_records(data["spec"]["genealogy_rows"])


def cross_domain_effects(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    configs = (
        ("CRUXEval", data["gate9"], "MEANINGFUL_L27_D75", "RANDOM_L27_D75_R"),
        ("Long character count", data["gate10"], "MEANINGFUL_L27_D75", "RANDOM_L27_D75_R"),
    )
    for domain, payload, meaningful_name, random_prefix in configs:
        conditions = [meaningful_name, *(f"{random_prefix}{index}" for index in range(4))]
        for index, condition in enumerate(conditions):
            point = payload["estimands"][condition]
            records.append(
                {
                    "domain": domain,
                    "condition": "MEANINGFUL" if index == 0 else f"RANDOM_R{index - 1}",
                    "controller_kind": "meaningful" if index == 0 else "random",
                    "accuracy_change": float(
                        point["accuracy_condition"] - point["accuracy_baseline"]
                    ),
                    "C": float(point["C"]),
                    "D": float(point["D"]),
                }
            )
    return pd.DataFrame.from_records(records)


def duration_history(data: dict[str, Any]) -> pd.DataFrame:
    records = [
        {
            "stage": "Gate 4",
            "condition": name.upper(),
            "duration": "one-shot",
            "D": float(data["gate4"][name]["D"]),
            "validity": np.nan,
        }
        for name in ("plus", "minus", "random")
    ]
    for condition in (
        "ONE_SHOT_PLUS",
        "ONE_SHOT_MINUS",
        "SUSTAINED_PLUS",
        "SUSTAINED_MINUS",
        "SUSTAINED_RANDOM_R0",
        "SUSTAINED_RANDOM_R1",
        "SUSTAINED_RANDOM_R2",
        "SUSTAINED_RANDOM_R3",
    ):
        point = data["gate5"]["evaluation"][condition]
        records.append(
            {
                "stage": "Gate 5",
                "condition": condition,
                "duration": "sustained" if condition.startswith("SUSTAINED") else "one-shot",
                "D": float(point["D"]),
                "validity": float(point["validity"]),
            }
        )
    return pd.DataFrame.from_records(records)


def dose_calibration(data: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for source in data["gate8_doses"]:
        rows.append(
            {
                "stage": "Gate 8 calibration",
                "dose": source["dose"],
                "eta": float(source["eta"]),
                "Q": float(source["Q"]),
                "random_Q_mean": float(source["random_Q_mean"]),
                "commitment_validity": float(source["commitment_validity"]),
                "semantic_evaluability": float(source["semantic_evaluability"]),
                "eligible": source["eligible"] == "True",
            }
        )
    gate7 = data["gate7"]
    rows.append(
        {
            "stage": "Gate 7 fresh full dose",
            "dose": "D100",
            "eta": np.nan,
            "Q": np.nan,
            "random_Q_mean": np.nan,
            "commitment_validity": float(
                gate7["summaries"]["BEST_SINGLE_MEAN_PLUS"]["commitment_validity"]
            ),
            "semantic_evaluability": float(
                gate7["summaries"]["BEST_SINGLE_MEAN_PLUS"]["semantic_evaluability"]
            ),
            "eligible": False,
        }
    )
    return pd.DataFrame.from_records(rows)


def development_confirmation_controls(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    configs = (
        ("Qwen", "DEVELOPMENT", data["gate9"], "MEANINGFUL_L27_D75", "RANDOM_L27_D75_R"),
        ("Ministral", "DEVELOPMENT", data["gate13_1"], "MEANINGFUL_SELECTED", "RANDOM_R"),
    )
    for model, stage, payload, meaningful_name, random_prefix in configs:
        for index, condition in enumerate(
            [meaningful_name, *(f"{random_prefix}{number}" for number in range(4))]
        ):
            records.append(
                {
                    "model_role": model,
                    "stage": stage,
                    "condition": "MEANINGFUL" if index == 0 else f"RANDOM_R{index - 1}",
                    "controller_kind": "meaningful" if index == 0 else "random",
                    "C": float(payload["estimands"][condition]["C"]),
                }
            )
    for model in ("Qwen", "Ministral"):
        result = data["confirmatory"]["models"][model]
        for index, condition in enumerate(("MEANINGFUL_FIXED", *q1_confirmatory.RANDOM_NAMES)):
            records.append(
                {
                    "model_role": model,
                    "stage": "CONFIRMATORY",
                    "condition": "MEANINGFUL" if index == 0 else f"RANDOM_R{index - 1}",
                    "controller_kind": "meaningful" if index == 0 else "random",
                    "C": float(result["estimands"][condition]["C"]),
                }
            )
    return pd.DataFrame.from_records(records)


def invalidity_taxonomy(data: dict[str, Any]) -> pd.DataFrame:
    labels = {
        "token_cap_truncation": "Token-cap truncation",
        "malformed_final": "Malformed FINAL",
        "multiple_or_contradictory_finals": "Multiple/contradictory FINALs",
        "semantic_answer_present_outside_accepted_commitment": "Answer outside accepted commitment",
    }
    return pd.DataFrame.from_records(
        [
            {"category": labels[key], "count": int(value), "status": "POST_HOC_DESCRIPTIVE_ONLY"}
            for key, value in data["invalidity"]["taxonomy"].items()
        ]
    )


def loo_sensitivity(data: dict[str, Any]) -> pd.DataFrame:
    item_order = {item_id: index for index, item_id in enumerate(data["item_ids"])}
    frame = pd.DataFrame.from_records(data["loo"])
    for column in ("C", "delta_C_nullmean", "G", "D"):
        frame[column] = frame[column].astype(float)
    frame["item_index"] = frame["left_out_item_id"].map(item_order).astype(int) + 1
    frame["model_order"] = frame["model_role"].map(MODEL_ORDER)
    return frame.sort_values(["model_order", "item_index"]).drop(columns="model_order")


def token_regimes(data: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for model in ("Qwen", "Ministral"):
        summaries = data["confirmatory"]["models"][model]["summaries"]
        for condition in (
            "BASELINE",
            "TEXTUAL_CAREFUL",
            "MEANINGFUL_FIXED",
            *q1_confirmatory.RANDOM_NAMES,
        ):
            summary = summaries[condition]
            records.append(
                {
                    "model_role": model,
                    "condition": condition,
                    "mean_tokens": float(summary["mean_tokens"]),
                    "median_tokens": float(summary["median_tokens"]),
                    "max_tokens": float(summary["max_tokens"]),
                    "interpretation": "CORRELATE_POSSIBLE_MEDIATOR_NOT_ESTABLISHED_CAUSE",
                }
            )
    return pd.DataFrame.from_records(records)


def build_all_tables(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    tables = {
        "figure2_genealogy": genealogy(data),
        "confirmatory_item_profiles": confirmatory_item_profiles(data),
        "confirmatory_transition_decomposition": transition_decomposition(data),
        "confirmatory_effects": confirmatory_effects(data),
        "confirmatory_safety": confirmatory_safety(data),
        "cross_domain_effects": cross_domain_effects(data),
        "s1_duration_history": duration_history(data),
        "s2_dose_calibration": dose_calibration(data),
        "s3_development_confirmation_controls": development_confirmation_controls(data),
        "s5_ministral_invalidity": invalidity_taxonomy(data),
        "s7_loo_sensitivity": loo_sensitivity(data),
        "s8_token_regimes": token_regimes(data),
    }
    reconcile_confirmatory_tables(data, tables)
    return tables


def reconcile_confirmatory_tables(data: dict[str, Any], tables: Mapping[str, pd.DataFrame]) -> None:
    profiles = tables["confirmatory_item_profiles"]
    for model in ("Qwen", "Ministral"):
        rows = data["qwen_rows"] if model == "Qwen" else data["ministral_rows"]
        arrays = _arrays(rows, data["item_ids"])
        recomputed = q1_confirmatory.primary_estimands(arrays)["MEANINGFUL_FIXED"]
        frozen = data["confirmatory"]["models"][model]["estimands"]["MEANINGFUL_FIXED"]
        for metric in ("C", "G", "D", "rescue", "damage"):
            if not np.isclose(recomputed[metric], frozen[metric], atol=1e-12, rtol=0):
                raise RuntimeError(f"{model} derived {metric} does not match frozen result")
        model_profiles = profiles[profiles["model_role"] == model]
        for condition in ("BASELINE", "MEANINGFUL_FIXED"):
            observed_accuracy = 1.0 - float(
                model_profiles[model_profiles["condition"] == condition]["q_hat_error"].mean()
            )
            frozen_accuracy = float(
                data["confirmatory"]["models"][model]["summaries"][condition]["accuracy"]
            )
            if not np.isclose(observed_accuracy, frozen_accuracy, atol=1e-12, rtol=0):
                raise RuntimeError(f"{model} {condition} item profile does not reconcile")


def write_tables(root: Path, tables: Mapping[str, pd.DataFrame]) -> dict[str, Path]:
    output = root / TABLE_DIR
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        paths[name] = path
    return paths


__all__ = [
    "TABLE_DIR",
    "build_all_tables",
    "confirmatory_item_profiles",
    "reconcile_confirmatory_tables",
    "transition_decomposition",
    "write_tables",
]
