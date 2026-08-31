"""Deterministic public figure tables derived from frozen Q2 V4.1 aggregates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE_DIR = Path("manuscript/data/paper1_q2/derived_figure_tables")
METRICS = ("A0", "A1", "A2")
SHELLS = ("MEDIUM", "STRONG")


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def _upper_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(count, 1)


def relational_geometry(data: dict[str, Any]) -> pd.DataFrame:
    estimands = data["estimands"]
    matrices = data["matrices"]
    controllers = list(estimands["controller_order"])
    dshape = estimands["semantic_distance"]["D_shape_superpopulation"]
    upper = _upper_indices(len(controllers))
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        for shell in SHELLS:
            intervention = np.asarray(matrices[f"{metric}_{shell}"], dtype=np.float64)[upper]
            blind_spot = np.asarray(dshape[shell], dtype=np.float64)[upper]
            x_rank = _rank(intervention)
            y_rank = _rank(blind_spot)
            denominator = len(intervention) - 1
            for pair_index, (left, right) in enumerate(zip(*upper, strict=True)):
                rows.append(
                    {
                        "metric": metric,
                        "shell": shell,
                        "pair_index": pair_index,
                        "controller_i": controllers[int(left)],
                        "controller_j": controllers[int(right)],
                        "intervention_distance": intervention[pair_index],
                        "blind_spot_D_shape": blind_spot[pair_index],
                        "intervention_rank_fraction": (x_rank[pair_index] - 1.0) / denominator,
                        "blind_spot_rank_fraction": (y_rank[pair_index] - 1.0) / denominator,
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(METRICS) * len(SHELLS) * 465
    if len(frame) != expected:
        raise RuntimeError("Q2 pairwise table cardinality changed")
    return frame


def association_summary(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        record = data["estimands"]["metrics"][metric]
        bootstrap = data["bootstrap"][metric]
        loo = record["leave_one_controller_out"]["values"]
        rows.append(
            {
                "metric": metric,
                "medium_rho": record["shell_rho"]["MEDIUM"],
                "strong_rho": record["shell_rho"]["STRONG"],
                "aggregate_full_sample_rho": record["aggregate_rho"],
                "qap_raw_p": record["qap"]["raw_p"],
                "qap_maxT_p": record["qap"]["maxT_adjusted_p"],
                "bootstrap_median": bootstrap["estimate"],
                "bootstrap_q025": bootstrap["q025"],
                "bootstrap_q975": bootstrap["q975"],
                "bootstrap_resamples": bootstrap["resamples"],
                "loo_aggregate_min": min(value["aggregate"] for value in loo),
                "loo_aggregate_max": max(value["aggregate"] for value in loo),
                "loo_all_shell_signs_positive": record["leave_one_controller_out"][
                    "all_sign_stable"
                ],
                "qualifies": record["qualifies"],
                "classification": data["estimands"]["classification"],
            }
        )
    return pd.DataFrame(rows)


def g3_contrasts(data: dict[str, Any]) -> pd.DataFrame:
    observed = data["estimands"]["g3_superiority"]["observed"]
    p_values = data["estimands"]["g3_superiority"]["maxT_superiority_p"]
    labels = ("A2_minus_A0", "A2_minus_A1")
    return pd.DataFrame(
        [
            {
                "contrast": label,
                "observed_full_sample": observed[label],
                "bootstrap_median": data["bootstrap"][label]["estimate"],
                "bootstrap_q025": data["bootstrap"][label]["q025"],
                "bootstrap_q975": data["bootstrap"][label]["q975"],
                "superiority_maxT_p": p_values[label],
                "required_margin": 0.10,
                "g3_pass": False,
            }
            for label in labels
        ]
    )


def radial_by_direction(data: dict[str, Any]) -> pd.DataFrame:
    controllers = list(data["estimands"]["controller_order"])
    shape = data["radial"]["R_shape"]["values"]
    total = data["radial"]["R_total"]["values"]
    if len(shape) != len(controllers) or len(total) != len(controllers):
        raise RuntimeError("Q2 radial/controller cardinality mismatch")
    return pd.DataFrame(
        {
            "controller_order_index": np.arange(len(controllers)),
            "controller_id": controllers,
            "shape_strong_minus_medium": shape,
            "total_strong_minus_medium": total,
            "ordering_rule": "FROZEN_CONTROLLER_ORDER_NOT_OUTCOME_SORTED",
        }
    )


def radial_summary(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for endpoint, key in (("D_shape", "R_shape"), ("D_total", "R_total")):
        record = data["radial"][key]
        rows.append(
            {
                "endpoint": endpoint,
                "observed_median_strong_minus_medium": record["median"],
                "positive_directions": record["positive_directions"],
                "controller_count": 31,
                "permutation_p": record["permutation_p"],
                "bootstrap_median": record["bootstrap"]["median_estimate"],
                "bootstrap_q025": record["bootstrap"]["q025"],
                "bootstrap_q975": record["bootstrap"]["q975"],
                "classification": record["classification"],
            }
        )
    return pd.DataFrame(rows)


def behavioral_context(data: dict[str, Any]) -> pd.DataFrame:
    estimands = data["estimands"]
    baseline = float(estimands["estimands"]["BASELINE"]["accuracy"])
    rows: list[dict[str, Any]] = [
        {
            "condition": "BASELINE",
            "accuracy": baseline,
            "mean_C_vs_baseline": 0.0,
            "mean_D_total_vs_baseline": 0.0,
            "role": "REFERENCE",
        }
    ]
    for shell in SHELLS:
        names = [f"{controller}_{shell}" for controller in estimands["controller_order"]]
        records = [estimands["estimands"][name] for name in names]
        rows.append(
            {
                "condition": shell,
                "accuracy": float(np.mean([record["accuracy_condition"] for record in records])),
                "mean_C_vs_baseline": float(np.mean([record["C"] for record in records])),
                "mean_D_total_vs_baseline": float(np.mean([record["D"] for record in records])),
                "role": "SHELL_MEAN_ACROSS_31_CONTROLLERS",
            }
        )
    return pd.DataFrame(rows)


def loo_robustness(data: dict[str, Any]) -> pd.DataFrame:
    controllers = list(data["estimands"]["controller_order"])
    rows: list[dict[str, Any]] = []
    for metric in METRICS:
        values = data["estimands"]["metrics"][metric]["leave_one_controller_out"]["values"]
        for record in values:
            dropped = int(record["dropped_index"])
            rows.append(
                {
                    "metric": metric,
                    "dropped_index": dropped,
                    "dropped_controller": controllers[dropped],
                    "medium_rho": record["MEDIUM"],
                    "strong_rho": record["STRONG"],
                    "aggregate_rho": record["aggregate"],
                }
            )
    return pd.DataFrame(rows)


def build_all_tables(data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    return {
        "pairwise_geometry": relational_geometry(data),
        "association_summary": association_summary(data),
        "g3_contrasts": g3_contrasts(data),
        "radial_by_direction": radial_by_direction(data),
        "radial_summary": radial_summary(data),
        "behavioral_context": behavioral_context(data),
        "loo_robustness": loo_robustness(data),
    }


def write_tables(root: Path, tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    output = root / TABLE_DIR
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n", float_format="%.15g")
        paths[name] = path
    return paths


__all__ = [
    "METRICS",
    "SHELLS",
    "TABLE_DIR",
    "build_all_tables",
    "write_tables",
]
