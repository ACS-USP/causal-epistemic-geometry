"""Pure publication plots for the frozen Q1 figure tables."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUTPUT_DIR = Path("manuscript/figures/paper1")
COLORS = {
    "ink": "#24313B",
    "muted": "#65717C",
    "grid": "#D9E0E5",
    "baseline": "#65717C",
    "meaningful": "#246B9E",
    "random": "#A9B4BF",
    "random_edge": "#5E6973",
    "rescue": "#2A8F78",
    "damage": "#C6534F",
    "shared_correct": "#CBD7CF",
    "shared_error": "#8C98A3",
    "pass": "#287A54",
    "fail": "#B74441",
    "calibration": "#A97316",
    "development": "#6C67A3",
}
FIXED_DATE = datetime(2026, 8, 27, tzinfo=UTC)


def _wrap_identifier(value: str, width: int = 20) -> str:
    """Wrap frozen classification identifiers only at semantic underscores."""
    lines: list[str] = []
    current = ""
    for token in value.split("_"):
        candidate = token if not current else f"{current}_{token}"
        if current and len(candidate) > width:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["muted"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "causal-epistemic-geometry-q1-v2",
            "pdf.compression": 9,
        }
    )


def save_figure(fig: plt.Figure, root: Path, stem: str) -> dict[str, Path]:
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout()
    paths: dict[str, Path] = {}
    metadata: dict[str, dict[str, Any]] = {
        "svg": {"Creator": "causal-epistemic-geometry", "Date": "2026-08-27"},
        "pdf": {
            "Creator": "causal-epistemic-geometry",
            "CreationDate": FIXED_DATE,
            "ModDate": FIXED_DATE,
        },
        "png": {"Software": "causal-epistemic-geometry"},
    }
    for suffix in ("svg", "pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            metadata=metadata[suffix],
        )
        paths[suffix] = path
    plt.close(fig)
    return paths


def figure1(root: Path) -> dict[str, Path]:
    fig = plt.figure(figsize=(11.5, 5.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.05, 1], hspace=0.35)
    ax = fig.add_subplot(grid[0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        ("Same frozen\nmodel", "same weights"),
        ("Repeated\nbaseline", "independent rollouts"),
        ("Meaningful\ncontroller", "fixed layer + dose"),
        ("Matched\nrandoms", "prospective null bank"),
        ("Same item\npanel", "itemwise errors"),
        ("Blind-spot\ncomplementarity", "beyond mean competence"),
    ]
    xs = np.linspace(0.075, 0.925, len(labels))
    for index, ((title, subtitle), x) in enumerate(zip(labels, xs, strict=True)):
        face = "#EDF3F7" if index < 4 else "#E9F3EE"
        box = FancyBboxPatch(
            (x - 0.067, 0.34),
            0.134,
            0.36,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=1.0,
            edgecolor="#536675",
            facecolor=face,
        )
        ax.add_patch(box)
        ax.text(x, 0.57, title, ha="center", va="center", weight="bold", fontsize=8.4)
        ax.text(x, 0.43, subtitle, ha="center", va="center", fontsize=7.8)
        if index < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.069, 0.52),
                    (xs[index + 1] - 0.069, 0.52),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    linewidth=1,
                    color="#5D6F7D",
                )
            )
    ax.text(
        0.5,
        0.12,
        "Causal comparison: meaningful movement versus repeated sampling "
        "and matched random directions",
        ha="center",
        color=COLORS["muted"],
        style="italic",
    )

    toy = fig.add_subplot(grid[1])
    profiles = np.asarray([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0]])
    toy.imshow(
        profiles,
        cmap=ListedColormap(["#DCE7DF", "#C85A55"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
        aspect="auto",
    )
    toy.set_yticks([0, 1], ["Policy A", "Policy B"])
    toy.set_xticks(range(8), [f"item {index}" for index in range(1, 9)])
    toy.tick_params(axis="x", rotation=35)
    toy.set_title("Illustrative only: equal accuracy, different error identities")
    toy.set_xlabel("Green = correct; red = error — schematic values, not empirical Q1 items")
    for spine in toy.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])
    fig.suptitle("Causal control of where a model fails", fontsize=14, weight="bold")
    return save_figure(fig, root, "figure1_causal_control_where_model_fails")


def figure2(root: Path, genealogy: pd.DataFrame) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(12, 7.2))
    x = np.arange(len(genealogy))
    colors = [
        COLORS["fail"]
        if category == "FAILURE"
        else COLORS["calibration"]
        if category == "CALIBRATION"
        else COLORS["pass"]
        if category == "REPLICATION"
        else COLORS["development"]
        for category in genealogy["category"]
    ]
    ax.axhline(0.5, color=COLORS["grid"], linewidth=1)
    ax.scatter(x, np.full(len(x), 0.5), s=180, color=colors, zorder=3)
    for index, row in genealogy.reset_index(drop=True).iterrows():
        upper = index % 2 == 0
        y_method = 0.88 if upper else 0.72
        y_result = 0.34 if upper else 0.18
        ax.plot([index, index], [0.53, y_method - 0.04], color=COLORS["grid"], linewidth=1)
        ax.plot([index, index], [y_result + 0.05, 0.47], color=COLORS["grid"], linewidth=1)
        ax.text(
            index,
            y_method,
            fill(str(row["methodological_decision"]), width=27),
            ha="center",
            va="bottom",
            fontsize=7.1,
            linespacing=1.22,
        )
        ax.text(
            index,
            y_result,
            f"{fill(str(row['observed_result']), width=28)}\n"
            f"{_wrap_identifier(str(row['classification']))}\n"
            f"N={int(row['n_items'])}",
            ha="center",
            va="top",
            fontsize=6.8,
            color=colors[index],
            linespacing=1.20,
        )
    ax.text(-0.55, 0.95, "METHOD DECISION", weight="bold", color=COLORS["muted"])
    ax.text(-0.55, 0.09, "OBSERVED RESULT", weight="bold", color=COLORS["muted"])
    ax.set_xlim(-0.65, len(x) - 0.35)
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks(x, genealogy["stage"])
    ax.set_yticks([])
    ax.spines.left.set_visible(False)
    ax.spines.bottom.set_visible(False)
    ax.set_title(
        "The qualified instrument emerged through falsifiable gates\n"
        "Failures, calibration, and replication are distinct—not an inevitable path",
        pad=14,
    )
    return save_figure(fig, root, "figure2_falsifiable_instrument_genealogy")


def _item_matrix(profiles: pd.DataFrame, model: str) -> tuple[np.ndarray, np.ndarray]:
    selected = profiles[profiles["model_role"] == model]
    matrix = []
    invalid = []
    for condition in ("BASELINE", "MEANINGFUL_FIXED"):
        rows = selected[selected["condition"] == condition].sort_values("item_index")
        matrix.append(rows["q_hat_error"].to_numpy(float))
        invalid.append(rows["invalid_rollouts"].to_numpy(int))
    return np.asarray(matrix), np.asarray(invalid)


def _draw_item_heatmap(ax: plt.Axes, profiles: pd.DataFrame, model: str) -> None:
    matrix, invalid = _item_matrix(profiles, model)
    cmap = ListedColormap(["#DDE9E1", "#E8C77B", "#C95D58"])
    norm = BoundaryNorm([-0.25, 0.25, 0.75, 1.25], cmap.N)
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for row, column in np.argwhere(invalid > 0):
        ax.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                hatch="////" if invalid[row, column] == 1 else "xxxx",
                edgecolor="#3B4147",
                linewidth=0,
            )
        )
    ax.set_yticks([0, 1], ["Repeated baseline", "Meaningful controller"])
    ticks = [0, 9, 19, 29, 39, 49, 56]
    ax.set_xticks(ticks, [str(index + 1) for index in ticks])
    ax.set_xlabel("Holdout item index (frozen manifest order; N=57)")
    ax.set_title(f"{model}: empirical error proportion across two rollouts")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])


def figure3(
    root: Path,
    profiles: pd.DataFrame,
    decomposition: pd.DataFrame,
    effects: pd.DataFrame,
    confirmatory: dict[str, Any],
) -> dict[str, Path]:
    fig = plt.figure(figsize=(12.2, 7.8))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.15, 1],
        width_ratios=[1.15, 1],
        hspace=0.68,
    )
    heat = fig.add_subplot(grid[0, :])
    _draw_item_heatmap(heat, profiles, "Qwen")
    heat.text(
        1.0,
        -0.20,
        "q̂ error: green=0, amber=0.5, red=1; hatch=invalid/unevaluable rollout retained as e=1",
        transform=heat.transAxes,
        ha="right",
        fontsize=8,
        color=COLORS["muted"],
    )

    transition = fig.add_subplot(grid[1, 0])
    components = decomposition[decomposition["model_role"] == "Qwen"].set_index("component")
    labels = ["shared_correct", "rescue", "damage", "shared_error"]
    names = ["Shared correct", "Rescue", "Damage", "Shared error"]
    colors = [COLORS[name] for name in labels]
    left = 0.0
    for label, name, color in zip(labels, names, colors, strict=True):
        value = float(components.loc[label, "fraction"])
        transition.barh([0], [value], left=left, color=color, height=0.46, label=name)
        if value >= 0.07:
            transition.text(
                left + value / 2, 0, f"{value:.3f}", ha="center", va="center", fontsize=8
            )
        left += value
    transition.set_xlim(0, 1)
    transition.set_yticks([])
    transition.set_xlabel("Mean mass over all four rollout cross-products")
    transition.set_title("Cross-rollout error-profile decomposition")
    transition.legend(
        frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.24)
    )

    null = fig.add_subplot(grid[1, 1])
    qwen = effects[effects["model_role"] == "Qwen"]
    meaningful = qwen[qwen["condition"] == "MEANINGFUL_FIXED"].iloc[0]
    randoms = qwen[qwen["controller_kind"] == "random"].copy()
    x_random = np.arange(1, 5)
    null.errorbar(
        [0],
        [meaningful["C"]],
        yerr=[
            [meaningful["C"] - meaningful["C_ci_lower"]],
            [meaningful["C_ci_upper"] - meaningful["C"]],
        ],
        fmt="o",
        color=COLORS["meaningful"],
        capsize=4,
        markersize=7,
        label="Meaningful + frozen 95% CI",
    )
    null.scatter(
        x_random,
        randoms["C"],
        color=COLORS["random"],
        edgecolor=COLORS["random_edge"],
        s=55,
        label="Prospective random controls",
    )
    random_mean = float(randoms["C"].mean())
    null.axhline(
        random_mean, color=COLORS["random_edge"], linestyle="--", linewidth=1, label="Random mean"
    )
    null.axhline(0, color=COLORS["muted"], linewidth=0.8)
    null.set_xticks(range(5), ["Meaningful", "R0", "R1", "R2", "R3"])
    null.set_ylabel("Competence-adjusted complementarity C")
    null.set_title("Meaningful controller versus frozen null bank")
    interval = confirmatory["models"]["Qwen"]["intervals"]["delta_C_nullmean"]
    null.text(
        0.02,
        0.98,
        f"C − random mean 95% interval\n[{interval['q025']:.3f}, {interval['q975']:.3f}]",
        transform=null.transAxes,
        va="top",
        fontsize=8,
    )
    null.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(
        "Qwen confirmatory itemwise blind-spot reorganization",
        fontsize=14,
        weight="bold",
    )
    return save_figure(fig, root, "figure3_qwen_confirmatory_itemwise_blindspots")


def figure4(
    root: Path,
    effects: pd.DataFrame,
    safety: pd.DataFrame,
    confirmatory: dict[str, Any],
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5))
    models = ["Qwen", "Ministral"]
    for model_index, model in enumerate(models):
        subset = effects[effects["model_role"] == model]
        meaningful = subset[subset["condition"] == "MEANINGFUL_FIXED"].iloc[0]
        randoms = subset[subset["controller_kind"] == "random"]
        axes[0].errorbar(
            [model_index],
            [meaningful["C"]],
            yerr=[
                [meaningful["C"] - meaningful["C_ci_lower"]],
                [meaningful["C_ci_upper"] - meaningful["C"]],
            ],
            fmt="o",
            color=COLORS["meaningful"],
            capsize=4,
            markersize=7,
            label="Meaningful + 95% CI" if model_index == 0 else None,
        )
        offsets = np.linspace(-0.12, 0.12, 4)
        axes[0].scatter(
            model_index + offsets,
            randoms["C"],
            color=COLORS["random"],
            edgecolor=COLORS["random_edge"],
            s=42,
            label="R0–R3" if model_index == 0 else None,
        )
    axes[0].axhline(0, color=COLORS["muted"], linewidth=0.8)
    axes[0].set_xticks([0, 1], models)
    axes[0].set_ylabel("Complementarity C")
    axes[0].set_title("Positive and null-specific C")
    axes[0].legend(frameon=False, fontsize=8)

    meaningful_effects = effects[effects["condition"] == "MEANINGFUL_FIXED"].set_index("model_role")
    axes[1].bar(
        models,
        [meaningful_effects.loc[model, "accuracy_change"] for model in models],
        color=COLORS["meaningful"],
        width=0.56,
    )
    axes[1].axhline(0, color=COLORS["muted"], linewidth=0.8)
    axes[1].set_ylabel("Accuracy change")
    axes[1].set_title("Aggregate competence")

    width = 0.16
    positions = np.arange(2)
    for metric_index, metric in enumerate(("commitment_validity", "semantic_evaluability")):
        metric_frame = safety[safety["metric"] == metric].set_index("model_role")
        base_positions = positions + (metric_index - 0.5) * 0.38
        axes[2].bar(
            base_positions - width / 2,
            [metric_frame.loc[model, "baseline"] for model in models],
            width,
            color=COLORS["baseline"],
            alpha=0.65 if metric_index else 1,
            label=f"Baseline {metric.replace('_', ' ')}",
        )
        axes[2].bar(
            base_positions + width / 2,
            [metric_frame.loc[model, "meaningful"] for model in models],
            width,
            color=COLORS["meaningful"],
            alpha=0.65 if metric_index else 1,
            label=f"Meaningful {metric.replace('_', ' ')}",
        )
        axes[2].scatter(
            base_positions,
            [metric_frame.loc[model, "relative_floor"] for model in models],
            marker="_",
            s=190,
            linewidth=2,
            color=COLORS["fail"],
            label="Frozen relative floor" if metric_index == 0 else None,
        )
    axes[2].set_ylim(0, 1.02)
    axes[2].set_xticks(positions, ["Qwen\nMODEL PASS", "Ministral\nMODEL FAIL"])
    axes[2].set_ylabel("Fraction")
    axes[2].set_title("Safe realization is conjunctive")
    axes[2].legend(frameon=False, fontsize=6.8, loc="lower left")
    fig.suptitle(
        "Complementarity replicates more robustly than safe realization\n"
        "Q1_CONFIRMATORY_QWEN_PASS_MINISTRAL_FAIL",
        fontsize=13,
        weight="bold",
    )
    return save_figure(fig, root, "figure4_cross_model_complementarity_safety")


def figure5(root: Path, cross_domain: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2))
    domains = ["CRUXEval", "Long character count"]
    for ax, metric, title in zip(
        axes,
        ("accuracy_change", "C", "D"),
        ("Accuracy change", "Complementarity C", "Profile distance D"),
        strict=True,
    ):
        for domain_index, domain in enumerate(domains):
            subset = cross_domain[cross_domain["domain"] == domain]
            meaningful = subset[subset["controller_kind"] == "meaningful"].iloc[0]
            randoms = subset[subset["controller_kind"] == "random"]
            ax.scatter(
                [domain_index],
                [meaningful[metric]],
                color=COLORS["meaningful"],
                marker="D",
                s=62,
                label="Fixed L27-D75" if domain_index == 0 else None,
                zorder=3,
            )
            offsets = np.linspace(-0.12, 0.12, 4)
            ax.scatter(
                domain_index + offsets,
                randoms[metric],
                color=COLORS["random"],
                edgecolor=COLORS["random_edge"],
                s=38,
                label="Prospective R0–R3" if domain_index == 0 else None,
            )
        ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
        ax.set_xticks([0, 1], ["CRUXEval\nN=100", "Long character count\nN=200"])
        ax.set_title(title)
    axes[0].set_ylabel("Absolute effect")
    axes[-1].legend(frameon=False, fontsize=7.5)
    fig.suptitle(
        "Fixed Qwen L27-D75 controller: positive same-domain evidence, negative transfer boundary\n"
        "Same model, vector, layer, dose, and timing; no character-count adaptation",
        fontsize=12.2,
        weight="bold",
    )
    return save_figure(fig, root, "figure5_fixed_qwen_cross_domain_boundary")


def supplement_s1(root: Path, duration: pd.DataFrame) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    gate5 = duration[duration["stage"] == "Gate 5"].copy()
    order = [
        "ONE_SHOT_PLUS",
        "SUSTAINED_PLUS",
        "ONE_SHOT_MINUS",
        "SUSTAINED_MINUS",
        "SUSTAINED_RANDOM_R0",
        "SUSTAINED_RANDOM_R1",
        "SUSTAINED_RANDOM_R2",
        "SUSTAINED_RANDOM_R3",
    ]
    gate5["order"] = gate5["condition"].map({name: index for index, name in enumerate(order)})
    gate5 = gate5.sort_values("order")
    colors = [
        COLORS["meaningful"] if "RANDOM" not in name else COLORS["random"]
        for name in gate5["condition"]
    ]
    ax.bar(np.arange(len(gate5)), gate5["D"], color=colors)
    ax.axhline(
        0.05, color=COLORS["fail"], linestyle="--", linewidth=1, label="Frozen movement threshold"
    )
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
    ax.set_xticks(
        np.arange(len(gate5)),
        [name.replace("_", "\n") for name in gate5["condition"]],
        fontsize=7.4,
    )
    ax.set_ylabel("Unbiased two-rollout profile distance D")
    ax.set_title(
        "Gate 5 isolated duration, but no meaningful condition reached the primary movement gate"
    )
    ax.legend(frameon=False)
    return save_figure(fig, root, "supplement_s1_duration_history")


def supplement_s2(root: Path, dose: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    calibration = dose[dose["stage"] == "Gate 8 calibration"].copy()
    x = np.arange(len(calibration))
    axes[0].plot(x, calibration["Q"], marker="o", color=COLORS["meaningful"], label="Meaningful Q")
    axes[0].plot(
        x,
        calibration["random_Q_mean"],
        marker="o",
        color=COLORS["random_edge"],
        label="Random mean Q",
    )
    axes[0].set_xticks(x, calibration["dose"])
    axes[0].set_ylabel("Label-free semantic-change Q")
    axes[0].set_title("Prospective calibration curve")
    axes[0].legend(frameon=False)
    axes[1].plot(
        x,
        calibration["commitment_validity"],
        marker="o",
        color=COLORS["meaningful"],
        label="Gate 8 validity",
    )
    gate7 = dose[dose["stage"] == "Gate 7 fresh full dose"].iloc[0]
    axes[1].scatter(
        [3],
        [gate7["commitment_validity"]],
        marker="X",
        s=75,
        color=COLORS["fail"],
        label="Gate 7 full-dose replication",
    )
    axes[1].set_xticks(x, calibration["dose"])
    axes[1].set_ylim(0.86, 1.01)
    axes[1].set_ylabel("Commitment validity")
    axes[1].set_title("Full-dose safety did not generalize across samples")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Full-dose overshoot motivated label-free D75 selection", fontsize=12.5, weight="bold"
    )
    return save_figure(fig, root, "supplement_s2_dose_calibration")


def supplement_s3(root: Path, controls: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=False)
    for ax, model in zip(axes, ("Qwen", "Ministral"), strict=True):
        subset = controls[controls["model_role"] == model]
        for stage_index, stage in enumerate(("DEVELOPMENT", "CONFIRMATORY")):
            stage_rows = subset[subset["stage"] == stage]
            meaningful = stage_rows[stage_rows["controller_kind"] == "meaningful"].iloc[0]
            randoms = stage_rows[stage_rows["controller_kind"] == "random"]
            ax.scatter(
                [stage_index],
                [meaningful["C"]],
                marker="D",
                s=58,
                color=COLORS["meaningful"],
                label="Meaningful" if stage_index == 0 else None,
            )
            ax.scatter(
                stage_index + np.linspace(-0.12, 0.12, 4),
                randoms["C"],
                s=38,
                color=COLORS["random"],
                edgecolor=COLORS["random_edge"],
                label="R0–R3" if stage_index == 0 else None,
            )
        ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
        ax.set_xticks([0, 1], ["DEVELOPMENT", "CONFIRMATORY"])
        ax.set_ylabel("Complementarity C")
        ax.set_title(model)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Development and confirmation remain separate; every null controller is shown",
        fontsize=12,
        weight="bold",
    )
    return save_figure(fig, root, "supplement_s3_development_confirmation_controls")


def supplement_s4(root: Path, profiles: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 1, figsize=(11.8, 4.8), sharex=True)
    for ax, model in zip(axes, ("Qwen", "Ministral"), strict=True):
        _draw_item_heatmap(ax, profiles, model)
        ax.set_title(model)
    axes[0].set_xlabel("")
    fig.suptitle(
        "Shared 57-item holdout: observed two-rollout profiles in frozen manifest order",
        fontsize=12,
        weight="bold",
    )
    return save_figure(fig, root, "supplement_s4_dual_model_item_profiles")


def supplement_s5(root: Path, taxonomy: pd.DataFrame) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(9.6, 4.3))
    ordered = taxonomy.sort_values(["count", "category"], ascending=[True, True])
    ax.barh(ordered["category"], ordered["count"], color=COLORS["fail"], alpha=0.82)
    for y, value in enumerate(ordered["count"]):
        ax.text(value + 0.08, y, str(int(value)), va="center")
    ax.set_xlabel("Invalid meaningful rows (13 total)")
    ax.set_title("Ministral invalidity taxonomy — POST_HOC_DESCRIPTIVE_ONLY")
    ax.text(
        0.99,
        0.02,
        "Recovered answers never enter frozen outcomes",
        transform=ax.transAxes,
        ha="right",
        color=COLORS["muted"],
        style="italic",
    )
    return save_figure(fig, root, "supplement_s5_ministral_invalidity_posthoc")


def supplement_s7(root: Path, loo: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), sharey=False)
    for ax, model in zip(axes, ("Qwen", "Ministral"), strict=True):
        subset = loo[loo["model_role"] == model].sort_values("item_index")
        ax.plot(
            subset["item_index"], subset["C"], color=COLORS["meaningful"], linewidth=1, label="C"
        )
        ax.plot(
            subset["item_index"],
            subset["delta_C_nullmean"],
            color=COLORS["random_edge"],
            linewidth=1,
            label="C − null mean",
        )
        ax.axhline(0, color=COLORS["muted"], linewidth=0.8)
        ax.set_xlabel("Left-out item index (manifest order)")
        ax.set_ylabel("Leave-one-item-out estimate")
        ax.set_title(model)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Leave-one-item-out sensitivity without item selection", fontsize=12, weight="bold"
    )
    return save_figure(fig, root, "supplement_s7_loo_robustness")


def supplement_s8(root: Path, tokens: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    order = [
        "BASELINE",
        "TEXTUAL_CAREFUL",
        "MEANINGFUL_FIXED",
        "RANDOM_R0",
        "RANDOM_R1",
        "RANDOM_R2",
        "RANDOM_R3",
    ]
    for ax, model in zip(axes, ("Qwen", "Ministral"), strict=True):
        subset = tokens[tokens["model_role"] == model].set_index("condition").loc[order]
        x = np.arange(len(order))
        ax.bar(x - 0.18, subset["mean_tokens"], 0.36, color=COLORS["meaningful"], label="Mean")
        ax.bar(x + 0.18, subset["median_tokens"], 0.36, color=COLORS["baseline"], label="Median")
        ax.set_xticks(
            x,
            [
                name.replace("TEXTUAL_", "TXT\n")
                .replace("MEANINGFUL_", "MEAN\n")
                .replace("RANDOM_", "")
                for name in order
            ],
            fontsize=7.2,
        )
        ax.set_ylabel("Generated tokens")
        ax.set_title(model)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Token regime is a correlate / possible mediator — not an established cause",
        fontsize=12,
        weight="bold",
    )
    return save_figure(fig, root, "supplement_s8_token_regimes")


def generate_all_figures(
    root: Path, tables: dict[str, pd.DataFrame], data: dict[str, Any]
) -> dict[str, dict[str, Path]]:
    publication_style()
    return {
        "figure1": figure1(root),
        "figure2": figure2(root, tables["figure2_genealogy"]),
        "figure3": figure3(
            root,
            tables["confirmatory_item_profiles"],
            tables["confirmatory_transition_decomposition"],
            tables["confirmatory_effects"],
            data["confirmatory"],
        ),
        "figure4": figure4(
            root,
            tables["confirmatory_effects"],
            tables["confirmatory_safety"],
            data["confirmatory"],
        ),
        "figure5": figure5(root, tables["cross_domain_effects"]),
        "s1": supplement_s1(root, tables["s1_duration_history"]),
        "s2": supplement_s2(root, tables["s2_dose_calibration"]),
        "s3": supplement_s3(root, tables["s3_development_confirmation_controls"]),
        "s4": supplement_s4(root, tables["confirmatory_item_profiles"]),
        "s5": supplement_s5(root, tables["s5_ministral_invalidity"]),
        "s7": supplement_s7(root, tables["s7_loo_sensitivity"]),
        "s8": supplement_s8(root, tables["s8_token_regimes"]),
    }


__all__ = ["OUTPUT_DIR", "generate_all_figures", "publication_style", "save_figure"]
