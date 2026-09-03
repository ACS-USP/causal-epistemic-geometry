"""Deterministic publication plots for Q2 OOS V2."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAVY = "#16324F"
BLUE = "#2878B5"
TEAL = "#2A9D8F"
ORANGE = "#E76F51"
GOLD = "#E9C46A"
GRAY = "#6B7280"
LIGHT = "#E8EEF3"
OUTDIR = Path("manuscript/figures/paper1_q2_oos")
FIXED_DATE = datetime(2026, 9, 3, tzinfo=UTC)


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "causal-epistemic-geometry-q2-oos-v1",
            "pdf.compression": 9,
        }
    )


def _panel(ax: plt.Axes, label: str, title: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold", pad=10)


def _save(fig: plt.Figure, root: Path, stem: str) -> dict[str, Path]:
    directory = root / OUTDIR
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    metadata: dict[str, dict[str, Any]] = {
        "svg": {"Creator": "causal-epistemic-geometry", "Date": "2026-09-03"},
        "pdf": {
            "Creator": "causal-epistemic-geometry",
            "CreationDate": FIXED_DATE,
            "ModDate": FIXED_DATE,
        },
        "png": {"Software": "causal-epistemic-geometry"},
    }
    for suffix in ("svg", "pdf", "png"):
        path = directory / f"{stem}.{suffix}"
        fig.savefig(path, facecolor="white", metadata=metadata[suffix])
        if suffix == "svg":
            normalized = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            path.write_text(normalized)
        outputs[suffix] = path
    plt.close(fig)
    return outputs


def main_figure(
    root: Path, tables: dict[str, pd.DataFrame], analysis: dict[str, Any]
) -> dict[str, Path]:
    controllers = tables["controller_associations"]
    global_table = tables["global_associations"]
    ff = tables["fresh_fresh_summary"].iloc[0]
    fig = plt.figure(figsize=(14.2, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 1.65), height_ratios=(1.0, 1.0))

    ax = fig.add_subplot(grid[0, 0])
    _panel(ax, "A", "Prospective controller-identity generalization")
    ax.axis("off")
    boxes = [
        (0.08, 0.62, "16 fresh\ncontrollers", TEAL),
        (0.58, 0.62, "31 fixed historical\nreferences", NAVY),
        (0.33, 0.18, "one $r_i$ per fresh controller\nSpearman over 31 references", BLUE),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.55", "facecolor": color, "edgecolor": "none"},
        )
    ax.annotate(
        "",
        xy=(0.28, 0.59),
        xytext=(0.35, 0.36),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.8},
    )
    ax.annotate(
        "",
        xy=(0.55, 0.59),
        xytext=(0.49, 0.36),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 1.8},
    )
    ax.text(
        0.5,
        0.02,
        "Inference unit: one prospectively sampled fresh controller",
        transform=ax.transAxes,
        ha="center",
        color=NAVY,
        fontweight="bold",
    )

    ax = fig.add_subplot(grid[0, 1])
    _panel(ax, "B", "All 16 controller-level associations — frozen order")
    x = controllers["controller_order"].to_numpy()
    ax.axhline(0.0, color="#9CA3AF", lw=1.2, ls="--")
    ax.plot(x, controllers["medium_rho"], "o-", color=BLUE, lw=1.2, ms=4, label="MEDIUM")
    ax.plot(x, controllers["strong_rho"], "s-", color=ORANGE, lw=1.2, ms=4, label="STRONG")
    ax.plot(
        x,
        controllers["equal_shell_r_i"],
        "D-",
        color=NAVY,
        lw=2.2,
        ms=5,
        label="equal-shell $r_i$ (primary)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{i:02d}" for i in x], rotation=0)
    ax.set_ylim(-0.08, 0.94)
    ax.set_ylabel("Spearman association over 31 references")
    ax.set_xlabel("fresh controller identity (prospectively frozen order; not outcome-sorted)")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.text(
        0.02,
        0.97,
        "16 / 16 $r_i > 0$\nexact sign-test $p=1.5259\\times10^{-5}$",
        transform=ax.transAxes,
        va="top",
        color=NAVY,
        fontweight="bold",
    )

    ax = fig.add_subplot(grid[1, 0])
    _panel(ax, "C", "Global fresh×old geometry (descriptive)")
    colors = [NAVY, GRAY, GRAY, GRAY]
    bars = ax.bar(global_table["metric"], global_table["equal_shell_rho"], color=colors, width=0.68)
    ax.axhline(0, color="#9CA3AF", lw=1)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("global equal-shell Spearman $\\rho$")
    for bar, value in zip(bars, global_table["equal_shell_rho"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.96,
        "A0 = primary geometry\nA1 / A2 / D2 = secondary",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
    )

    ax = fig.add_subplot(grid[1, 1])
    _panel(ax, "D", "Fresh×fresh geometry — secondary only")
    estimate = float(ff["association"])
    se = float(ff["jackknife_standard_error"])
    ax.axvline(0, color="#9CA3AF", lw=1.2, ls="--")
    ax.errorbar(estimate, 0.55, xerr=se, fmt="o", ms=11, color=TEAL, ecolor=TEAL, capsize=5, lw=2.2)
    ax.set_xlim(-0.05, 0.82)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("fresh×fresh equal-shell association (±1 node-jackknife SE)")
    ax.text(
        estimate,
        0.72,
        f"$\\rho={estimate:.3f}$\nSE = {se:.3f}",
        ha="center",
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.18,
        "SECONDARY_ONLY  ·  CANNOT_RESCUE_PRIMARY",
        transform=ax.transAxes,
        ha="center",
        color=ORANGE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF4ED", "edgecolor": ORANGE},
    )
    fig.suptitle(
        "Q2 OOS V2: A0 relational alignment generalized to fresh controller identities",
        fontsize=15,
        fontweight="bold",
        color=NAVY,
    )
    return _save(fig, root, "figure_q2_oos_fresh_controller_generalization")


def supplement_shells(root: Path, controllers: pd.DataFrame) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    x = controllers["medium_rho"]
    y = controllers["strong_rho"]
    ax.axhline(0, color="#9CA3AF", lw=1, ls="--")
    ax.axvline(0, color="#9CA3AF", lw=1, ls="--")
    ax.plot([-0.1, 1], [-0.1, 1], color=LIGHT, lw=2, zorder=0)
    ax.scatter(x, y, s=55, color=BLUE, edgecolor="white", linewidth=0.8)
    for order, xv, yv in zip(controllers["controller_order"], x, y, strict=True):
        ax.annotate(
            f"F{int(order):02d}", (xv, yv), xytext=(4, 4), textcoords="offset points", fontsize=7
        )
    ax.set(
        xlabel="MEDIUM row-Spearman",
        ylabel="STRONG row-Spearman",
        xlim=(-0.05, 0.95),
        ylim=(-0.05, 0.95),
    )
    ax.set_title("Shell consistency for all fresh controllers", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.98,
        "Frozen controller order labels; no outcome sorting",
        transform=ax.transAxes,
        va="top",
        color=GRAY,
    )
    return _save(fig, root, "supplement_q2_oos_medium_vs_strong")


def supplement_lofo(root: Path, controllers: pd.DataFrame, lofo: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.4), constrained_layout=True)
    axes[0].hist(
        controllers["equal_shell_r_i"],
        bins=np.linspace(0.4, 0.88, 13),
        color=BLUE,
        edgecolor="white",
    )
    axes[0].axvline(0, color="#9CA3AF", ls="--")
    axes[0].axvline(
        controllers["equal_shell_r_i"].median(),
        color=NAVY,
        lw=2,
        label=f"median = {controllers['equal_shell_r_i'].median():.3f}",
    )
    axes[0].set(
        xlabel="controller-level $r_i$",
        ylabel="fresh controllers",
        title="A. Primary controller-level distribution",
    )
    axes[0].legend(frameon=False)
    order = np.arange(1, len(lofo) + 1)
    axes[1].plot(order, lofo["mean"], "o-", color=TEAL, label="LOFO mean")
    axes[1].plot(order, lofo["median"], "s-", color=ORANGE, label="LOFO median")
    axes[1].set_xticks(order)
    axes[1].set_xticklabels([f"F{i:02d}" for i in order])
    axes[1].set(
        xlabel="omitted fresh controller (frozen order)",
        ylabel="association",
        ylim=(0.62, 0.76),
        title="B. Leave-one-fresh-controller-out sensitivity",
    )
    axes[1].legend(frameon=False, ncol=2)
    axes[1].text(
        0.02,
        0.05,
        "Every 15-controller subset: 15/15 positive; exact $p=3.0518\\times10^{-5}$",
        transform=axes[1].transAxes,
        color=NAVY,
        fontweight="bold",
    )
    return _save(fig, root, "supplement_q2_oos_distribution_and_lofo")


def supplement_fresh_fresh(root: Path, pairs: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    for ax, shell, color in zip(axes, ("medium", "strong"), (BLUE, ORANGE), strict=True):
        ax.scatter(
            pairs[f"A0_{shell}"],
            pairs[f"Dshape_{shell}"],
            s=18,
            alpha=0.68,
            color=color,
            edgecolor="none",
        )
        ax.set(xlabel=f"A0 distance — {shell.upper()}", ylabel=f"Dshape — {shell.upper()}")
        ax.set_title(f"{shell.upper()} · 120 fresh×fresh pairs", loc="left", fontweight="bold")
    fig.suptitle(
        "Fresh×fresh relational structure — SECONDARY_ONLY",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    return _save(fig, root, "supplement_q2_oos_fresh_fresh_pairs")


def supplement_runtime(
    root: Path, runtime: pd.DataFrame, analysis: dict[str, Any]
) -> dict[str, Path]:
    fig, ax = plt.subplots(figsize=(8.7, 5.2), constrained_layout=True)
    colors = [TEAL if m == "observed" else LIGHT for m in runtime["measure"]]
    bars = ax.bar(runtime["measure"], runtime["hours"], color=colors, edgecolor=NAVY, linewidth=0.8)
    for bar, value in zip(bars, runtime["hours"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f} h",
            ha="center",
            fontweight="bold",
        )
    ax.set_ylim(0, max(runtime["hours"]) + 1.4)
    ax.set_ylabel("campaign wall time")
    ax.set_title("Efficient-termination runtime audit", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.95,
        "359 repetition stops · 2 hard caps\nobserved 19,200 / 19,200 rows",
        transform=ax.transAxes,
        va="top",
        color=NAVY,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.08,
        "Operational diagnostic only; not evidence for relational geometry",
        transform=ax.transAxes,
        ha="right",
        color=GRAY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    return _save(fig, root, "supplement_q2_oos_runtime")


def supplement_bootstrap(root: Path, data: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    panel = data.iloc[:2].copy()
    labels = ["global A0", "median $r_i$"]
    for index, row in panel.iterrows():
        axes[0].plot(
            [row["q025"], row["q975"]], [index, index], color=GRAY, lw=5, solid_capstyle="round"
        )
        axes[0].plot(row["q50"], index, "o", color=ORANGE, ms=7)
        axes[0].plot(row["estimate"], index, "D", color=NAVY, ms=7)
    axes[0].set_yticks([0, 1], labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("association")
    axes[0].set_title("A. Archived item-resampling sensitivity", loc="left", fontweight="bold")
    axes[0].legend(
        handles=[
            mpl.lines.Line2D([], [], marker="D", color=NAVY, ls="", label="full panel"),
            mpl.lines.Line2D([], [], marker="o", color=ORANGE, ls="", label="resampling median"),
        ],
        frameon=False,
    )
    axes[0].text(
        0.02,
        0.06,
        "Not a conventional 95% CI",
        transform=axes[0].transAxes,
        color=ORANGE,
        fontweight="bold",
    )
    support = float(
        data.loc[data["object"] == "ordinary_bootstrap_effective_support", "q50"].iloc[0]
    )
    axes[1].bar(
        ["draws", "mean unique", "mean effective"],
        [300, 189.8224, support],
        color=[NAVY, BLUE, ORANGE],
    )
    axes[1].set_ylim(0, 330)
    axes[1].set_ylabel("item count / effective support")
    axes[1].set_title("B. Why the panel perturbation is severe", loc="left", fontweight="bold")
    axes[1].text(
        0.98,
        0.92,
        "300 draws contain ≈190 unique items\nand ≈150 multiplicity-effective items",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )
    fig.suptitle(
        "Post-hoc item-bootstrap diagnostic — method not calibrated",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    return _save(fig, root, "supplement_q2_oos_item_bootstrap_diagnostic")


def generate_all(root: Path, data: dict[str, Any]) -> dict[str, dict[str, Path]]:
    configure()
    tables = data["tables"]
    return {
        "main": main_figure(root, tables, data["analysis"]),
        "s1": supplement_shells(root, tables["controller_associations"]),
        "s2": supplement_lofo(root, tables["controller_associations"], tables["lofo"]),
        "s3": supplement_fresh_fresh(root, tables["fresh_fresh_pairs"]),
        "s4": supplement_runtime(root, tables["runtime_summary"], data["analysis"]),
        "s5": supplement_bootstrap(root, tables["bootstrap_diagnostic"]),
    }


__all__ = ["generate_all"]
