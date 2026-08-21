import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from build_focus_broad_families_uniform_axis_scatter import (
    DEFAULT_BROAD_DATA,
    DEFAULT_SIGE_DATA,
    FIXED_LOWER,
    FIXED_UPPER,
    GROUP_SPECS,
    clean_rows,
    source_path,
)


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURES = (
    EXP_DIR / "figures" / "focus_broad_families_uniform_axes" / "overview"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Arrange the fixed-axis scatter plots for all 15 material "
            "classifications as separate panels in one figure."
        )
    )
    parser.add_argument("--broad-data", type=Path, default=DEFAULT_BROAD_DATA)
    parser.add_argument("--sige-data", type=Path, default=DEFAULT_SIGE_DATA)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--axis-lower", type=float, default=FIXED_LOWER)
    parser.add_argument("--axis-upper", type=float, default=FIXED_UPPER)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (args.axis_lower > 0 and args.axis_upper > args.axis_lower):
        raise ValueError("axis limits must satisfy 0 < lower < upper")


def title_font_size(title: str) -> float:
    if len(title) >= 13:
        return 9.0
    if len(title) >= 9:
        return 9.8
    return 10.8


def plot_panel(
    ax: plt.Axes,
    rows: pd.DataFrame,
    classification_ja: str,
    lower: float,
    upper: float,
) -> None:
    markers = {"p": "o", "n": "^"}
    colors = {"p": "#1f77b4", "n": "#d62728"}
    for carrier in ["p", "n"]:
        carrier_rows = rows[rows["carrier_type"].astype("string").eq(carrier)]
        if carrier_rows.empty:
            continue
        ax.scatter(
            carrier_rows["sigma_S_per_m"],
            carrier_rows["sigma_pred_S_per_m"],
            s=10,
            alpha=0.58,
            marker=markers[carrier],
            color=colors[carrier],
            edgecolors="none",
            rasterized=True,
        )

    line = np.geomspace(lower, upper, 300)
    ax.plot(line, line, color="black", linewidth=1.05)
    ax.plot(line, line * 2.0, color="gray", linewidth=0.65, linestyle="--", alpha=0.55)
    ax.plot(line, line / 2.0, color="gray", linewidth=0.65, linestyle="--", alpha=0.55)
    ax.plot(line, line * 10.0, color="gray", linewidth=0.55, linestyle=":", alpha=0.45)
    ax.plot(line, line / 10.0, color="gray", linewidth=0.55, linestyle=":", alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"{classification_ja}  (n={len(rows):,})",
        fontsize=title_font_size(classification_ja),
        fontweight="bold",
        pad=5,
    )
    ax.grid(True, which="both", linewidth=0.28, alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=8, length=3)
    ax.tick_params(axis="both", which="minor", length=1.8)


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.figures.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(15.2, 9.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    plotted = []
    for ax, (classification_ja, _, safe_name, source_kind) in zip(
        axes.flat, GROUP_SPECS
    ):
        path = source_path(args.broad_data, args.sige_data, safe_name, source_kind)
        if not path.exists():
            raise FileNotFoundError(f"missing source for {classification_ja}: {path}")
        rows = clean_rows(pd.read_csv(path, low_memory=False))
        if rows.empty:
            raise ValueError(f"no valid rows for {classification_ja}: {path}")
        plot_panel(ax, rows, classification_ja, args.axis_lower, args.axis_upper)
        plotted.append(classification_ja)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#1f77b4",
            markeredgecolor="none",
            markersize=7,
            label="p-type",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#d62728",
            markeredgecolor="none",
            markersize=7,
            label="n-type",
        ),
        Line2D([0], [0], color="black", linewidth=1.2, label="y=x"),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=0.8,
            linestyle="--",
            label="factor 2",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            linewidth=0.8,
            linestyle=":",
            label="factor 10",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.subplots_adjust(left=0.055, right=0.992, top=0.965, bottom=0.075, wspace=0.13, hspace=0.18)

    png_path = (
        args.figures
        / "focus_broad_families_all_material_systems_fixed_1e2_1e8.png"
    )
    pdf_path = (
        args.figures
        / "focus_broad_families_all_material_systems_fixed_1e2_1e8.pdf"
    )
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)

    if len(plotted) != 15 or set(plotted) != {spec[0] for spec in GROUP_SPECS}:
        raise ValueError("overview does not contain all 15 material classifications")
    for path in [png_path, pdf_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty overview: {path}")
    print(f"[uniform_axes_overview] panels: {len(plotted)}", flush=True)
    print(f"[uniform_axes_overview] png: {png_path}", flush=True)
    print(f"[uniform_axes_overview] pdf: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
