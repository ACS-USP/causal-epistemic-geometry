"""Publication plots for the frozen Q2 V4.1 visual evidence package."""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUTPUT_DIR = Path("manuscript/figures/paper1_q2")
FIXED_DATE = datetime(2026, 8, 30, tzinfo=timezone.utc)  # noqa: UP017 -- Python 3.10 kernel
FIGURE_SIZES = {
    "figure1": (7.2, 4.8),
    "figure2": (7.2, 5.4),
    "figure2_raw": (7.2, 5.4),
    "figure2_hexbin": (7.2, 5.4),
    "figure2_decile": (7.2, 5.4),
    "figure3": (7.2, 4.5),
    "figure4": (7.2, 4.4),
    "s1": (7.2, 3.4),
    "s2": (7.2, 4.0),
}
COLORS = {
    "ink": "#24313B",
    "muted": "#65717C",
    "grid": "#D9E0E5",
    "a0": "#246B9E",
    "a1": "#6C67A3",
    "a2": "#A97316",
    "medium": "#2A8F78",
    "strong": "#C6534F",
    "bootstrap": "#8C98A3",
    "null": "#CBD3DA",
    "pass": "#287A54",
    "fail": "#B74441",
    "light_blue": "#E8F1F7",
    "light_green": "#E9F3EE",
    "light_amber": "#F8F0E1",
}


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.2,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["muted"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "causal-epistemic-geometry-q2-v1",
            "pdf.compression": 9,
        }
    )


def save_figure(fig: plt.Figure, root: Path, stem: str) -> dict[str, Path]:
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message=".*tight_layout.*")
        fig.tight_layout()
    metadata: dict[str, dict[str, Any]] = {
        "svg": {"Creator": "causal-epistemic-geometry", "Date": "2026-08-30"},
        "pdf": {
            "Creator": "causal-epistemic-geometry",
            "CreationDate": FIXED_DATE,
            "ModDate": FIXED_DATE,
        },
        "png": {"Software": "causal-epistemic-geometry"},
    }
    paths: dict[str, Path] = {}
    for suffix in ("svg", "pdf", "png"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata[suffix])
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n",
                encoding="utf-8",
            )
        paths[suffix] = path
    plt.close(fig)
    return paths


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    face: str,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=0.9,
        edgecolor="#536675",
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x, y + 0.028, title, ha="center", va="center", weight="bold", fontsize=7.2)
    ax.text(x, y - 0.038, subtitle, ha="center", va="center", fontsize=6.5)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color="#5D6F7D",
        )
    )


def _coefficient_projection(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(
        [row["coefficients"] for row in data["safe_bank"]["directions"]],
        dtype=np.float64,
    )
    centered = coefficients - coefficients.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    loadings = vt[:2].copy()
    for index in range(2):
        pivot = int(np.argmax(np.abs(loadings[index])))
        if loadings[index, pivot] < 0:
            loadings[index] *= -1.0
    coordinates = centered @ loadings.T
    explained = np.square(singular[:2]) / np.sum(np.square(singular))
    return coordinates, explained


def figure1(root: Path, data: dict[str, Any]) -> dict[str, Path]:
    fig = plt.figure(figsize=FIGURE_SIZES["figure1"])
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.2], width_ratios=[1.05, 1.2])
    flow = fig.add_subplot(grid[0, :])
    flow.set_xlim(0, 1)
    flow.set_ylim(0, 1)
    flow.axis("off")
    boxes = [
        (0.10, "Q1", "blind spots can move", COLORS["light_green"]),
        (0.30, "Fixed 8-D lab", "8 sources define Q", COLORS["light_blue"]),
        (0.50, "31 directions", "MEDIUM + STRONG", COLORS["light_blue"]),
        (0.70, "Blind-spot profiles", "300 items × 2 rollouts", COLORS["light_amber"]),
        (0.90, "Relational test", "A0/A1/A2 ↔ Dshape", COLORS["light_green"]),
    ]
    for x, title, subtitle, face in boxes:
        _box(flow, x, 0.53, 0.17, 0.34, title, subtitle, face)
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        _arrow(flow, (left[0] + 0.09, 0.53), (right[0] - 0.09, 0.53))
    flow.text(
        0.5,
        0.08,
        "Candidate geometries fixed before semantic correctness  |  Q3 utility remains untested",
        ha="center",
        fontsize=7.2,
        color=COLORS["muted"],
        style="italic",
    )

    lab = fig.add_subplot(grid[1, 0])
    coordinates, explained = _coefficient_projection(data)
    medium = 0.5 * coordinates
    strong = coordinates
    for med, high in zip(medium, strong, strict=True):
        lab.plot([med[0], high[0]], [med[1], high[1]], color=COLORS["grid"], linewidth=0.7)
    lab.scatter(
        medium[:, 0],
        medium[:, 1],
        s=13,
        facecolors="white",
        edgecolors=COLORS["medium"],
        linewidths=0.8,
        label="MEDIUM",
    )
    lab.scatter(
        strong[:, 0],
        strong[:, 1],
        s=15,
        color=COLORS["strong"],
        alpha=0.78,
        marker="s",
        label="STRONG",
    )
    lab.axhline(0, color=COLORS["grid"], linewidth=0.6)
    lab.axvline(0, color=COLORS["grid"], linewidth=0.6)
    lab.set_xticks([])
    lab.set_yticks([])
    lab.set_title("Intervention laboratory (pre-outcome inset)")
    lab.legend(frameon=False, loc="lower right", fontsize=6.8)
    lab.text(
        0.02,
        0.98,
        (
            "2-D PCA of frozen 8-D coefficients\n"
            f"{100 * explained.sum():.1f}% variance shown; not primary geometry"
        ),
        transform=lab.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    relation = fig.add_subplot(grid[1, 1])
    relation.set_xlim(0, 1)
    relation.set_ylim(0, 1)
    relation.axis("off")
    _box(
        relation,
        0.24,
        0.70,
        0.40,
        0.24,
        "Intervention geometry",
        "A0  ·  A1  ·  A2",
        COLORS["light_blue"],
    )
    _box(
        relation,
        0.76,
        0.70,
        0.40,
        0.24,
        "Blind-spot geometry",
        "centered item profiles Dshape",
        COLORS["light_amber"],
    )
    _arrow(relation, (0.45, 0.70), (0.55, 0.70))
    relation.text(0.5, 0.86, "Spearman", ha="center", fontsize=6.7, weight="bold")
    relation.text(
        0.5,
        0.51,
        "31 identities → 465 dependent dyads per shell",
        ha="center",
        fontsize=6.4,
        color=COLORS["muted"],
    )
    _box(
        relation,
        0.50,
        0.25,
        0.68,
        0.21,
        "Dependence-aware inference",
        "controller-label QAP + item bootstrap + delete-one",
        COLORS["light_green"],
    )
    _arrow(relation, (0.50, 0.45), (0.50, 0.37))
    relation.text(
        0.5,
        0.06,
        "Angle and amplitude are separate tested questions — not a manifold theorem",
        ha="center",
        fontsize=6.7,
        color=COLORS["muted"],
        style="italic",
    )
    fig.suptitle(
        "Q2: from causal controllability to relational geometry", fontsize=12, weight="bold"
    )
    return save_figure(fig, root, "figure1_q2_from_controllability_to_geometry")


def _binned_rank_trend(frame: pd.DataFrame, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    groups = np.clip(np.digitize(frame["intervention_rank_fraction"], edges) - 1, 0, bins - 1)
    x_values: list[float] = []
    y_values: list[float] = []
    for index in range(bins):
        subset = frame[groups == index]
        if len(subset):
            x_values.append(float(subset["intervention_rank_fraction"].median()))
            y_values.append(float(subset["blind_spot_rank_fraction"].median()))
    return np.asarray(x_values), np.asarray(y_values)


def _decile_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Fixed equal-width rank deciles; the rule never depends on outcomes."""
    edges = np.linspace(0.0, 1.0, 11)
    groups = np.clip(
        np.digitize(frame["intervention_rank_fraction"], edges, right=False) - 1,
        0,
        9,
    )
    records: list[dict[str, float | int]] = []
    for index in range(10):
        values = frame.loc[groups == index, "blind_spot_rank_fraction"].to_numpy(dtype=float)
        records.append(
            {
                "decile": index + 1,
                "x": (edges[index] + edges[index + 1]) / 2,
                "count": len(values),
                "q10": float(np.quantile(values, 0.10)),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.quantile(values, 0.50)),
                "q75": float(np.quantile(values, 0.75)),
                "q90": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame.from_records(records)


def _apply_shared_count_normalization(collections: list[Any]) -> Normalize:
    """Apply one count scale to every hexbin collection."""
    shared_max = max(float(collection.get_array().max()) for collection in collections)
    shared_norm = Normalize(vmin=1, vmax=shared_max)
    for collection in collections:
        collection.set_norm(shared_norm)
    return shared_norm


def _format_relational_panel(
    ax: plt.Axes,
    *,
    row: int,
    column: int,
    metric: str,
    shell: str,
    rho: float,
    detailed_title: bool = True,
) -> None:
    ax.plot([0, 1], [0, 1], color=COLORS["grid"], linewidth=0.8, linestyle="--")
    ax.text(
        0.05,
        0.92,
        f"ρ = {rho:.3f}\nmaxT p = 0.00002",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.1,
        weight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    if row == 0:
        title = {
            "A0": "A0  coordinate",
            "A1": "A1  whitened",
            "A2": "A2  finite response",
        }[metric]
        ax.set_title(title if detailed_title else metric)
    if column == 0:
        ax.set_ylabel(f"{shell}\nBlind-spot Dshape rank")
    if row == 1:
        ax.set_xlabel("Intervention-distance rank")


def figure2_raw_scatter(
    root: Path,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
) -> dict[str, Path]:
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZES["figure2_raw"], sharex=True, sharey=True)
    colors = {"MEDIUM": COLORS["medium"], "STRONG": COLORS["strong"]}
    lookup = associations.set_index("metric")
    for row, shell in enumerate(("MEDIUM", "STRONG")):
        for column, metric in enumerate(("A0", "A1", "A2")):
            ax = axes[row, column]
            subset = pairwise[(pairwise["metric"] == metric) & (pairwise["shell"] == shell)]
            ax.scatter(
                subset["intervention_rank_fraction"],
                subset["blind_spot_rank_fraction"],
                s=8,
                alpha=0.20,
                color=colors[shell],
                edgecolors="none",
                rasterized=False,
            )
            trend_x, trend_y = _binned_rank_trend(subset)
            ax.plot(
                trend_x, trend_y, color=COLORS["ink"], linewidth=1.3, marker="o", markersize=2.5
            )
            field = "medium_rho" if shell == "MEDIUM" else "strong_rho"
            rho = float(lookup.loc[metric, field])
            _format_relational_panel(
                ax,
                row=row,
                column=column,
                metric=metric,
                shell=shell,
                rho=rho,
                detailed_title=False,
            )
    fig.suptitle(
        "Raw pairwise ranks show a moderate tendency with broad scatter",
        fontsize=11.2,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        (
            "All 465 dyads are shown per panel; they share 31 controllers. "
            "The black line is a binned rank median, not a fitted linear model."
        ),
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.18, hspace=0.16)
    return save_figure(fig, root, "supplement_q2_raw_pairwise_scatter")


def figure2_hexbin(
    root: Path,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
) -> dict[str, Path]:
    fig = plt.figure(figsize=FIGURE_SIZES["figure2_hexbin"])
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 0.06))
    axes = np.empty((2, 3), dtype=object)
    for row in range(2):
        for column in range(3):
            reference = axes[0, 0] if (row, column) != (0, 0) else None
            axes[row, column] = fig.add_subplot(
                grid[row, column],
                sharex=reference,
                sharey=reference,
            )
    colorbar_axis = fig.add_subplot(grid[:, 3])
    lookup = associations.set_index("metric")
    collections = []
    for row, shell in enumerate(("MEDIUM", "STRONG")):
        for column, metric in enumerate(("A0", "A1", "A2")):
            ax = axes[row, column]
            subset = pairwise[(pairwise["metric"] == metric) & (pairwise["shell"] == shell)]
            collection = ax.hexbin(
                subset["intervention_rank_fraction"],
                subset["blind_spot_rank_fraction"],
                gridsize=12,
                extent=(0, 1, 0, 1),
                mincnt=1,
                cmap="Blues",
                linewidths=0.25,
                edgecolors="white",
            )
            collections.append(collection)
            summary = _decile_summary(subset)
            ax.vlines(
                summary["x"],
                summary["q25"],
                summary["q75"],
                color=COLORS["ink"],
                linewidth=1.0,
                alpha=0.8,
                zorder=4,
            )
            ax.plot(
                summary["x"],
                summary["median"],
                color=COLORS["ink"],
                linewidth=1.1,
                marker="o",
                markerfacecolor="white",
                markeredgewidth=0.7,
                markersize=2.8,
                zorder=5,
            )
            field = "medium_rho" if shell == "MEDIUM" else "strong_rho"
            _format_relational_panel(
                ax,
                row=row,
                column=column,
                metric=metric,
                shell=shell,
                rho=float(lookup.loc[metric, field]),
            )
    _apply_shared_count_normalization(collections)
    colorbar = fig.colorbar(collections[0], cax=colorbar_axis)
    colorbar.set_label("Dyads / hex (shared scale)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.5)
    fig.suptitle(
        "Moderate rank association coexists with broad pairwise dispersion",
        fontsize=11.2,
        weight="bold",
    )
    fig.text(
        0.48,
        0.012,
        (
            "Shared 12-bin hex grid; black markers show fixed-decile "
            "medians and IQR, not a fitted model."
        ),
        ha="center",
        fontsize=6.7,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.18, hspace=0.16)
    return save_figure(fig, root, "alternate_q2_hexbin_pairwise_density")


def figure2_decile_distribution(
    root: Path,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
    *,
    filename: str = "alternate_q2_decile_distributions",
) -> dict[str, Path]:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=FIGURE_SIZES["figure2_decile"],
        sharex=True,
        sharey=True,
    )
    lookup = associations.set_index("metric")
    colors = {"MEDIUM": COLORS["medium"], "STRONG": COLORS["strong"]}
    for row, shell in enumerate(("MEDIUM", "STRONG")):
        for column, metric in enumerate(("A0", "A1", "A2")):
            ax = axes[row, column]
            subset = pairwise[(pairwise["metric"] == metric) & (pairwise["shell"] == shell)]
            summary = _decile_summary(subset)
            ax.vlines(
                summary["x"],
                summary["q10"],
                summary["q90"],
                color=colors[shell],
                linewidth=0.8,
                alpha=0.55,
                zorder=2,
            )
            ax.vlines(
                summary["x"],
                summary["q25"],
                summary["q75"],
                color=colors[shell],
                linewidth=3.2,
                alpha=0.9,
                zorder=3,
            )
            ax.plot(
                summary["x"],
                summary["median"],
                color=COLORS["ink"],
                linewidth=1.0,
                marker="o",
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.8,
                zorder=4,
            )
            field = "medium_rho" if shell == "MEDIUM" else "strong_rho"
            _format_relational_panel(
                ax,
                row=row,
                column=column,
                metric=metric,
                shell=shell,
                rho=float(lookup.loc[metric, field]),
            )
    fig.suptitle(
        "Pre-outcome geometry is moderately associated with blind-spot rank structure",
        fontsize=11.0,
        weight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Fixed equal-width rank deciles: line = 10–90%, thick bar = IQR, dot = median.",
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.12, top=0.88, wspace=0.18, hspace=0.16)
    return save_figure(fig, root, filename)


def figure2(
    root: Path,
    pairwise: pd.DataFrame,
    associations: pd.DataFrame,
) -> dict[str, Path]:
    """Selected main view: fixed deciles expose tendency and conditional spread."""
    return figure2_decile_distribution(
        root,
        pairwise,
        associations,
        filename="figure2_q2_primary_relational_geometry",
    )


def figure3(
    root: Path,
    associations: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> dict[str, Path]:
    fig, axes = plt.subplots(
        1, 3, figsize=FIGURE_SIZES["figure3"], gridspec_kw={"width_ratios": [1.05, 1.0, 1.2]}
    )
    x = np.arange(3)
    metric_colors = [COLORS["a0"], COLORS["a1"], COLORS["a2"]]

    ax = axes[0]
    for index, row in associations.iterrows():
        ax.plot(
            [index - 0.10, index + 0.10],
            [row["medium_rho"], row["strong_rho"]],
            color=metric_colors[index],
            linewidth=1.0,
            alpha=0.7,
        )
        ax.scatter(
            index - 0.10, row["medium_rho"], color=COLORS["medium"], s=25, marker="o", zorder=3
        )
        ax.scatter(
            index + 0.10, row["strong_rho"], color=COLORS["strong"], s=25, marker="s", zorder=3
        )
        ax.scatter(
            index,
            row["aggregate_full_sample_rho"],
            color=metric_colors[index],
            s=38,
            marker="D",
            zorder=4,
        )
    ax.axhline(0.20, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x, ["A0", "A1", "A2"])
    ax.set_ylim(0.15, 0.64)
    ax.set_ylabel("Full-sample Spearman ρ")
    ax.set_title("A  Frozen full sample")
    ax.legend(
        handles=[
            plt.Line2D([], [], marker="o", linestyle="", color=COLORS["medium"], label="MEDIUM"),
            plt.Line2D([], [], marker="s", linestyle="", color=COLORS["strong"], label="STRONG"),
            plt.Line2D([], [], marker="D", linestyle="", color=COLORS["ink"], label="Aggregate"),
        ],
        frameon=False,
        fontsize=6.5,
        loc="lower left",
    )

    ax = axes[1]
    for index, row in associations.iterrows():
        ax.vlines(
            index,
            row["bootstrap_q025"],
            row["bootstrap_q975"],
            color=metric_colors[index],
            linewidth=2.2,
        )
        ax.scatter(
            index, row["bootstrap_median"], color=metric_colors[index], s=34, marker="o", zorder=3
        )
    ax.axhline(0, color=COLORS["muted"], linewidth=0.7)
    ax.set_xticks(x, ["A0", "A1", "A2"])
    ax.set_ylim(0.27, 0.57)
    ax.set_ylabel("Bootstrap-resample aggregate ρ")
    ax.set_title("B  Frozen item bootstrap")
    ax.text(
        0.5,
        0.02,
        "Median + percentile interval\nnot centered on panel A",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=COLORS["muted"],
    )

    ax = axes[2]
    y = np.arange(2)
    for index, row in contrasts.iterrows():
        ax.hlines(
            y[index],
            row["bootstrap_q025"],
            row["bootstrap_q975"],
            color=COLORS["bootstrap"],
            linewidth=3,
        )
        ax.scatter(
            row["bootstrap_median"], y[index], color=COLORS["bootstrap"], marker="o", s=30, zorder=3
        )
        ax.scatter(
            row["observed_full_sample"], y[index], color=COLORS["a2"], marker="D", s=34, zorder=4
        )
    ax.axvline(0, color=COLORS["fail"], linewidth=0.9)
    ax.axvline(0.10, color=COLORS["muted"], linestyle="--", linewidth=0.8)
    ax.set_yticks(y, ["A2 − A0", "A2 − A1"])
    ax.set_xlim(-0.17, 0.13)
    ax.set_xlabel("Superiority contrast")
    ax.set_title("C  G3 contrasts")
    ax.text(0.06, 0.56, "◆  full sample", transform=ax.transAxes, fontsize=6.6, color=COLORS["a2"])
    ax.text(
        0.06,
        0.50,
        "—●—  bootstrap median + interval",
        transform=ax.transAxes,
        fontsize=6.6,
        color=COLORS["bootstrap"],
    )
    ax.text(
        0.06,
        0.42,
        "Both superiority maxT p = 1.0",
        transform=ax.transAxes,
        fontsize=6.6,
        color=COLORS["muted"],
    )
    for panel in axes:
        panel.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.6)
    fig.suptitle(
        "Q2 is G2, not G3: A2 qualifies but does not outperform A0/A1", fontsize=11.2, weight="bold"
    )
    fig.subplots_adjust(bottom=0.15, top=0.83, wspace=0.42)
    return save_figure(fig, root, "figure3_q2_g2_not_g3")


def figure4(
    root: Path,
    radial: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["figure4"], sharex=True)
    definitions = [
        ("shape_strong_minus_medium", "D_shape", COLORS["medium"]),
        ("total_strong_minus_medium", "D_total", COLORS["a0"]),
    ]
    summary_lookup = summary.set_index("endpoint")
    x = radial["controller_order_index"].to_numpy() + 1
    for ax, (field, endpoint, color) in zip(axes, definitions, strict=True):
        values = radial[field].to_numpy(dtype=float)
        ax.vlines(x, 0, values, color=color, alpha=0.55, linewidth=1.2)
        ax.scatter(x, values, color=color, s=18, zorder=3)
        ax.axhline(0, color=COLORS["ink"], linewidth=0.8)
        record = summary_lookup.loc[endpoint]
        ax.axhline(
            record["observed_median_strong_minus_medium"],
            color=COLORS["fail"],
            linestyle="--",
            linewidth=1,
        )
        ax.text(
            31.5,
            record["observed_median_strong_minus_medium"],
            "median Δ ",
            ha="right",
            va="bottom",
            fontsize=6.4,
            color=COLORS["fail"],
        )
        ax.text(
            0.03,
            0.96,
            (
                "31 / 31 positive\n"
                f"median Δ = {record['observed_median_strong_minus_medium']:.3f}\n"
                f"95% bootstrap: [{record['bootstrap_q025']:.3f}, "
                f"{record['bootstrap_q975']:.3f}]\n"
                f"{record['classification']}  ·  p = {record['permutation_p']:.5f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            weight="bold",
        )
        ax.set_title("Blind-spot shape" if endpoint == "D_shape" else "Total profile displacement")
        ax.set_xlabel("Controller (frozen order)")
        ax.set_xlim(0, 32)
        ax.set_xticks([1, 8, 16, 24, 31])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    axes[0].set_ylabel("STRONG − MEDIUM baseline displacement")
    fig.suptitle(
        "The stronger shell moves farther in every tested direction", fontsize=11.5, weight="bold"
    )
    fig.text(
        0.5,
        0.015,
        (
            "Controllers remain in frozen manifest order. Two shells establish this "
            "paired contrast, not a global dose-response law."
        ),
        ha="center",
        fontsize=6.9,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.17, top=0.84, wspace=0.22)
    return save_figure(fig, root, "figure4_q2_radial_31_of_31")


def supplement_s1(root: Path, context: pd.DataFrame) -> dict[str, Path]:
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["s1"])
    fields = [
        ("accuracy", "Accuracy", COLORS["a0"]),
        ("mean_C_vs_baseline", "Mean C vs baseline", COLORS["a1"]),
        ("mean_D_total_vs_baseline", "Mean D vs baseline", COLORS["strong"]),
    ]
    x = np.arange(3)
    labels = ["BASE", "MED", "STRONG"]
    for ax, (field, title, color) in zip(axes, fields, strict=True):
        values = context[field].to_numpy(dtype=float)
        ax.plot(x, values, color=color, linewidth=1.4)
        ax.scatter(x, values, color=color, s=28, zorder=3)
        span = max(float(np.ptp(values)), 0.01)
        lower = float(values.min() - 0.10 * span)
        upper = float(values.max() + 0.22 * span)
        ax.set_ylim(lower, upper)
        for index, value in enumerate(values):
            ax.text(index, value + 0.07 * span, f"{value:.3f}", ha="center", fontsize=6.8)
        ax.set_xticks(x, labels)
        ax.set_title(title)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
        if field != "accuracy":
            ax.axhline(0, color=COLORS["muted"], linewidth=0.7)
    fig.suptitle(
        "Stronger steering increases movement, not uniform mean accuracy",
        fontsize=11.2,
        weight="bold",
    )
    fig.text(
        0.5,
        0.015,
        (
            "Panels use separate y-axes and units. C and D are secondary "
            "baseline-relative context, not the primary Q2 relational endpoint."
        ),
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.19, top=0.79, wspace=0.34)
    return save_figure(fig, root, "supplement_q2_movement_not_accuracy")


def _rank_matrix(matrix: np.ndarray) -> np.ndarray:
    count = len(matrix)
    upper = np.triu_indices(count, 1)
    ranks = pd.Series(matrix[upper]).rank(method="average").to_numpy(dtype=np.float64)
    output = np.zeros_like(matrix, dtype=np.float64)
    output[upper] = ranks
    output[(upper[1], upper[0])] = ranks
    return output


def qap_null_values(data: dict[str, Any], batch_size: int = 1000) -> dict[str, np.ndarray]:
    permutations = np.asarray(data["qap_permutations"], dtype=np.int64)
    count = permutations.shape[1]
    upper = np.triu_indices(count, 1)
    metrics = ("A0", "A1", "A2")
    shells = ("MEDIUM", "STRONG")
    dshape = data["estimands"]["semantic_distance"]["D_shape_superpopulation"]
    y_rank = {shell: _rank_matrix(np.asarray(dshape[shell], dtype=np.float64)) for shell in shells}
    x_centered: dict[tuple[str, str], np.ndarray] = {}
    x_norm: dict[tuple[str, str], float] = {}
    for metric in metrics:
        for shell in shells:
            values = _rank_matrix(
                np.asarray(data["matrices"][f"{metric}_{shell}"], dtype=np.float64)
            )[upper]
            centered = values - values.mean()
            x_centered[(metric, shell)] = centered
            x_norm[(metric, shell)] = float(np.linalg.norm(centered))
    y_values = {shell: y_rank[shell][upper] for shell in shells}
    y_norm = {
        shell: float(np.linalg.norm(values - values.mean())) for shell, values in y_values.items()
    }
    output = {metric: np.empty(len(permutations), dtype=np.float64) for metric in metrics}
    for start in range(0, len(permutations), batch_size):
        stop = min(start + batch_size, len(permutations))
        batch = permutations[start:stop]
        shell_permuted: dict[str, np.ndarray] = {}
        for shell in shells:
            values = y_rank[shell][batch[:, upper[0]], batch[:, upper[1]]]
            shell_permuted[shell] = values - values.mean(axis=1, keepdims=True)
        for metric in metrics:
            correlations = []
            for shell in shells:
                numerator = shell_permuted[shell] @ x_centered[(metric, shell)]
                correlations.append(numerator / (y_norm[shell] * x_norm[(metric, shell)]))
            output[metric][start:stop] = 0.5 * (correlations[0] + correlations[1])
    return output


def supplement_s2(
    root: Path,
    data: dict[str, Any],
    associations: pd.DataFrame,
    loo: pd.DataFrame,
) -> dict[str, Path]:
    null = qap_null_values(data)
    lookup = associations.set_index("metric")
    fig, axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZES["s2"], gridspec_kw={"width_ratios": [1.25, 1]}
    )

    ax = axes[0]
    datasets = [null[metric] for metric in ("A0", "A1", "A2")]
    violin = ax.violinplot(
        datasets,
        positions=[0, 1, 2],
        orientation="horizontal",
        widths=0.68,
        showextrema=False,
    )
    for body, color in zip(
        violin["bodies"], (COLORS["a0"], COLORS["a1"], COLORS["a2"]), strict=True
    ):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
    for position, metric, color in zip(
        [0, 1, 2], ("A0", "A1", "A2"), (COLORS["a0"], COLORS["a1"], COLORS["a2"]), strict=True
    ):
        observed = float(lookup.loc[metric, "aggregate_full_sample_rho"])
        ax.scatter(observed, position, color=color, marker="D", s=38, zorder=4)
        computed_raw_p = float(np.mean(null[metric] >= observed))
        if computed_raw_p != float(lookup.loc[metric, "qap_raw_p"]):
            raise RuntimeError(f"QAP null reconstruction failed for {metric}")
        ax.text(observed + 0.015, position, "p = 0.00002", va="center", fontsize=6.7)
    ax.axvline(0, color=COLORS["muted"], linewidth=0.7)
    ax.set_yticks([0, 1, 2], ["A0", "A1", "A2"])
    ax.set_xlabel("Shell-aggregated Spearman ρ")
    ax.set_title("A  Frozen controller-label QAP")
    ax.text(
        0.02,
        0.03,
        "50,000 shared identity maps\nmaxT-adjusted p = 0.00002 for all",
        transform=ax.transAxes,
        fontsize=6.8,
        color=COLORS["muted"],
    )

    ax = axes[1]
    y = np.arange(3)
    for position, metric, color in zip(
        y, ("A0", "A1", "A2"), (COLORS["a0"], COLORS["a1"], COLORS["a2"]), strict=True
    ):
        subset = loo[loo["metric"] == metric]
        lower = float(subset["aggregate_rho"].min())
        upper = float(subset["aggregate_rho"].max())
        observed = float(lookup.loc[metric, "aggregate_full_sample_rho"])
        ax.hlines(position, lower, upper, color=color, linewidth=4, alpha=0.65)
        ax.scatter(observed, position, color=color, marker="D", s=38, zorder=3)
    ax.axvline(0, color=COLORS["muted"], linewidth=0.7)
    ax.set_yticks(y, ["A0", "A1", "A2"])
    ax.set_xlabel("Aggregate ρ")
    ax.set_title("B  Delete one controller")
    ax.text(
        0.5,
        0.04,
        "All MEDIUM and STRONG\nleave-one values remain positive",
        transform=ax.transAxes,
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    for panel in axes:
        panel.grid(axis="x", color=COLORS["grid"], linewidth=0.45, alpha=0.65)
    fig.suptitle(
        "Dependence-aware inference and controller stability", fontsize=11.2, weight="bold"
    )
    fig.text(
        0.5,
        0.012,
        (
            "QAP permutes whole controller identities with the same map in both shells; "
            "dyads are never shuffled independently."
        ),
        ha="center",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.18, top=0.80, wspace=0.36)
    return save_figure(fig, root, "supplement_q2_qap_and_loo_robustness")


def generate_all_figures(
    root: Path,
    tables: dict[str, pd.DataFrame],
    data: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    publication_style()
    return {
        "figure1": figure1(root, data),
        "figure2": figure2(root, tables["pairwise_geometry"], tables["association_summary"]),
        "figure2_raw": figure2_raw_scatter(
            root, tables["pairwise_geometry"], tables["association_summary"]
        ),
        "figure2_hexbin": figure2_hexbin(
            root, tables["pairwise_geometry"], tables["association_summary"]
        ),
        "figure3": figure3(root, tables["association_summary"], tables["g3_contrasts"]),
        "figure4": figure4(root, tables["radial_by_direction"], tables["radial_summary"]),
        "s1": supplement_s1(root, tables["behavioral_context"]),
        "s2": supplement_s2(root, data, tables["association_summary"], tables["loo_robustness"]),
    }


__all__ = [
    "COLORS",
    "FIGURE_SIZES",
    "OUTPUT_DIR",
    "generate_all_figures",
    "publication_style",
    "qap_null_values",
    "save_figure",
]
