#!/usr/bin/env python3
"""Generate the remote-safe Paper 1 figure package from canonical aggregates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript/figures/paper1"
SOURCES = {
    "confirmatory": ROOT / "review/q1_confirmatory_fixed_controllers/CONFIRMATORY_RESULTS.json",
    "analysis_lock": ROOT / "review/q1_confirmatory_fixed_controllers/ANALYSIS_LOCK.json",
    "qwen_development": ROOT / "review/gate9_selected_d75_evaluation/ESTIMANDS.json",
    "ministral_development": ROOT / "review/gate13_1_all_layer_causal_atlas/ESTIMANDS.json",
    "cross_domain": ROOT / "review/gate10_cross_domain_charcount/ESTIMANDS.json",
}

COLORS = {
    "meaningful": "#2369A0",
    "random": "#A9B4BF",
    "rescue": "#2A9D8F",
    "damage": "#D95F59",
    "baseline": "#5B6573",
    "fail": "#C43C39",
    "pass": "#25855A",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#26313B",
            "text.color": "#26313B",
            "xtick.color": "#26313B",
            "ytick.color": "#26313B",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str, outputs: dict[str, str]) -> None:
    fig.tight_layout()
    for suffix in ("svg", "png"):
        path = OUTPUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", metadata={"Creator": "CEG"})
        outputs[str(path.relative_to(ROOT))] = sha256(path)
    plt.close(fig)


def conceptual(outputs: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 2.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Behavioral\nsource contrast", "careful − direct"),
        ("Internal\ndirection", "layer, sign, dose"),
        ("Frozen\nintervention", "timing + scope"),
        ("Model\nvariant", "same weights"),
        ("Error profile", "item propensities"),
        ("Complementarity", "G, C, D\nrescue / damage"),
    ]
    xs = np.linspace(0.08, 0.92, len(labels))
    for index, ((title, subtitle), x) in enumerate(zip(labels, xs, strict=True)):
        box = FancyBboxPatch(
            (x - 0.068, 0.36),
            0.136,
            0.38,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#4A6275",
            facecolor="#EEF4F8" if index < 4 else "#E7F4EF",
        )
        ax.add_patch(box)
        ax.text(x, 0.59, title, ha="center", va="center", weight="bold")
        ax.text(x, 0.43, subtitle, ha="center", va="center", fontsize=8.5)
        if index < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.071, 0.55),
                    (xs[index + 1] - 0.071, 0.55),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.2,
                    color="#52697A",
                )
            )
    ax.text(
        0.5,
        0.12,
        "Controls: repeated baseline sampling + architecture-matched random directions",
        ha="center",
        va="center",
        color="#5B6573",
        style="italic",
    )
    save(fig, "figure1_conceptual_framework", outputs)


def c_and_rescue_damage(
    outputs: dict[str, str],
    *,
    stem: str,
    title: str,
    development: dict[str, Any],
    development_meaningful: str,
    development_random_prefix: str,
    confirmatory: dict[str, Any],
    model: str,
) -> None:
    dev = development["estimands"]
    conf = confirmatory["models"][model]["estimands"]
    dev_random = [
        values["C"] for name, values in dev.items() if name.startswith(development_random_prefix)
    ]
    conf_random = [conf[f"RANDOM_R{index}"]["C"] for index in range(4)]
    meaningful = [dev[development_meaningful]["C"], conf["MEANINGFUL_FIXED"]["C"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    x = np.arange(2)
    axes[0].bar(x, meaningful, width=0.48, color=COLORS["meaningful"], label="meaningful")
    for stage, values in enumerate((dev_random, conf_random)):
        offsets = np.linspace(-0.17, 0.17, len(values))
        axes[0].scatter(
            np.full(len(values), stage) + offsets,
            values,
            color=COLORS["random"],
            edgecolor="#58636E",
            zorder=3,
            label="random controls" if stage == 0 else None,
        )
    axes[0].axhline(0, color="#6B737B", linewidth=0.8)
    axes[0].set_xticks(x, ["DEVELOPMENT", "CONFIRMATORY"])
    axes[0].set_ylabel("Complementarity C")
    axes[0].set_title("Meaningful controller vs random bank")
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )

    width = 0.34
    rescue = [
        dev[development_meaningful]["rescue"],
        conf["MEANINGFUL_FIXED"]["rescue"],
    ]
    damage = [
        dev[development_meaningful]["damage"],
        conf["MEANINGFUL_FIXED"]["damage"],
    ]
    axes[1].bar(x - width / 2, rescue, width, color=COLORS["rescue"], label="rescue")
    axes[1].bar(x + width / 2, damage, width, color=COLORS["damage"], label="damage")
    axes[1].set_xticks(x, ["DEVELOPMENT", "CONFIRMATORY"])
    axes[1].set_ylabel("Cross-rollout fraction")
    axes[1].set_title("Rescue and damage")
    axes[1].legend(frameon=False)
    fig.suptitle(title, fontsize=13, weight="bold")
    save(fig, stem, outputs)


def dissociation(
    outputs: dict[str, str], confirmatory: dict[str, Any], analysis_lock: dict[str, Any]
) -> None:
    models = ["Qwen", "Ministral"]
    x = np.arange(2)
    c_values = []
    acc_changes = []
    baseline_validity = []
    meaningful_validity = []
    validity_floor = []
    decisions = []
    margin = analysis_lock["safety"]["commitment_relative_margin"]
    for model in models:
        result = confirmatory["models"][model]
        metric = result["estimands"]["MEANINGFUL_FIXED"]
        summaries = result["summaries"]
        c_values.append(metric["C"])
        acc_changes.append(metric["accuracy_condition"] - metric["accuracy_baseline"])
        baseline = summaries["BASELINE"]["commitment_validity"]
        baseline_validity.append(baseline)
        meaningful_validity.append(summaries["MEANINGFUL_FIXED"]["commitment_validity"])
        validity_floor.append(baseline + margin)
        decisions.append(result["model_pass"])

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    width = 0.34
    axes[0].bar(x - width / 2, c_values, width, color=COLORS["meaningful"], label="C")
    axes[0].bar(x + width / 2, acc_changes, width, color="#6E9F72", label="accuracy change")
    axes[0].axhline(0, color="#6B737B", linewidth=0.8)
    axes[0].set_xticks(x, models)
    axes[0].set_ylabel("Absolute scale")
    axes[0].set_title("Supported effect dimensions")
    axes[0].legend(frameon=False)

    axes[1].bar(
        x - width / 2,
        baseline_validity,
        width,
        color=COLORS["baseline"],
        label="baseline validity",
    )
    axes[1].bar(
        x + width / 2,
        meaningful_validity,
        width,
        color=COLORS["meaningful"],
        label="meaningful validity",
    )
    axes[1].scatter(
        x,
        validity_floor,
        marker="_",
        s=500,
        linewidth=2.5,
        color=COLORS["fail"],
        label="frozen relative floor",
    )
    for index, passed in enumerate(decisions):
        axes[1].text(
            index,
            0.997,
            "MODEL PASS" if passed else "MODEL FAIL",
            ha="center",
            va="top",
            weight="bold",
            color=COLORS["pass"] if passed else COLORS["fail"],
        )
    axes[1].set_ylim(0.82, 1.005)
    axes[1].set_xticks(x, models)
    axes[1].set_ylabel("Commitment validity")
    axes[1].set_title("Frozen safety adjudication")
    axes[1].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        fontsize=8,
    )
    fig.suptitle("Confirmatory evidence vector: f(E) is not E", fontsize=13, weight="bold")
    save(fig, "figure4_complementarity_safety_dissociation", outputs)


def cross_domain(outputs: dict[str, str], qwen: dict[str, Any], charcount: dict[str, Any]) -> None:
    tasks = ["CRUXEval", "Character count"]
    meaningful_names = ["MEANINGFUL_L27_D75", "MEANINGFUL_L27_D75"]
    payloads = [qwen, charcount]
    metrics = ["accuracy_change", "C", "D"]
    labels = ["Accuracy change", "C", "D"]
    meaningful_values: dict[str, list[float]] = {key: [] for key in metrics}
    random_values: dict[str, list[float]] = {key: [] for key in metrics}
    for payload, meaningful_name in zip(payloads, meaningful_names, strict=True):
        meaningful = payload["estimands"][meaningful_name]
        meaningful_values["accuracy_change"].append(
            meaningful["accuracy_condition"] - meaningful["accuracy_baseline"]
        )
        meaningful_values["C"].append(meaningful["C"])
        meaningful_values["D"].append(meaningful["D"])
        for key in metrics:
            random_values[key].append(payload["random_summary"][key]["mean"])

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True)
    x = np.arange(2)
    width = 0.34
    for axis, key, label in zip(axes, metrics, labels, strict=True):
        axis.bar(
            x - width / 2,
            meaningful_values[key],
            width,
            color=COLORS["meaningful"],
            label="meaningful",
        )
        axis.bar(
            x + width / 2,
            random_values[key],
            width,
            color=COLORS["random"],
            label="random mean",
        )
        axis.axhline(0, color="#6B737B", linewidth=0.8)
        axis.set_xticks(x, tasks, rotation=12)
        axis.set_title(label)
    axes[0].set_ylabel("Absolute scale")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Fixed Qwen L27-D75 controller: positive semantic control, negative task transfer",
        fontsize=12.5,
        weight="bold",
    )
    save(fig, "figure5_cross_domain_boundary", outputs)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    data = {name: load(path) for name, path in SOURCES.items()}
    outputs: dict[str, str] = {}
    conceptual(outputs)
    c_and_rescue_damage(
        outputs,
        stem="figure2_qwen_development_confirmation",
        title="Qwen: prospectively fixed control survives confirmation",
        development=data["qwen_development"],
        development_meaningful="MEANINGFUL_L27_D75",
        development_random_prefix="RANDOM_L27_D75_",
        confirmatory=data["confirmatory"],
        model="Qwen",
    )
    c_and_rescue_damage(
        outputs,
        stem="figure3_ministral_development_confirmation",
        title="Ministral: complementarity reproduces; safety guard fails",
        development=data["ministral_development"],
        development_meaningful="MEANINGFUL_SELECTED",
        development_random_prefix="RANDOM_R",
        confirmatory=data["confirmatory"],
        model="Ministral",
    )
    dissociation(outputs, data["confirmatory"], data["analysis_lock"])
    cross_domain(outputs, data["qwen_development"], data["cross_domain"])
    manifest = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "contains_raw_outputs": False,
        "source_artifacts": {
            str(path.relative_to(ROOT)): sha256(path) for path in SOURCES.values()
        },
        "output_artifacts": outputs,
    }
    (OUTPUT / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
