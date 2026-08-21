"""Create the all-material sigma scatter on 1e-6--1e10 axes.

The main plot reproduces the supplied image's 2471:1991 canvas aspect ratio.
The material legend is exported separately so the plot panel keeps that ratio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogFormatterMathtext
import numpy as np
import pandas as pd

from plot_sigma_ml_all_materials_reference_colors_step40 import MATERIAL_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/"
    "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/starrydata2_step41_sigma_ml_all_materials_wide_axes"
)
AXIS_MIN = 1e-6
AXIS_MAX = 1e10
REFERENCE_WIDTH_PX = 2471
REFERENCE_HEIGHT_PX = 1991
REFERENCE_DPI = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all material systems on 1e-6--1e10 axes at the reference aspect ratio."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists; pass --replace-output to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    family_column = "material_group_key_broad_family"
    exp_column = "sigma_obs_S_per_m_step11"
    pred_column = "sigma_pred_ML_S_per_m_step32"
    required = [family_column, exp_column, pred_column]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    for column in [exp_column, pred_column]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        np.isfinite(data[exp_column])
        & np.isfinite(data[pred_column])
        & data[exp_column].gt(0)
        & data[pred_column].gt(0)
    ].copy()
    plotted = data[
        data[exp_column].between(AXIS_MIN, AXIS_MAX)
        & data[pred_column].between(AXIS_MIN, AXIS_MAX)
    ].copy()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    palette = list(plt.get_cmap("tab10").colors)
    fig, ax = plt.subplots(
        figsize=(REFERENCE_WIDTH_PX / REFERENCE_DPI, REFERENCE_HEIGHT_PX / REFERENCE_DPI)
    )
    legend_rows: list[dict[str, object]] = []
    legend_handles: list[Line2D] = []
    for index, (family, material_name) in enumerate(MATERIAL_SPECS):
        subset = plotted[plotted[family_column].eq(family)]
        if subset.empty:
            continue
        color = palette[index % len(palette)]
        ax.scatter(
            subset[exp_column],
            subset[pred_column],
            s=17,
            alpha=0.50,
            color=color,
            edgecolors="none",
            rasterized=True,
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="none",
                alpha=0.50,
                label=f"{material_name} (n={len(subset)})",
            )
        )
        legend_rows.append(
            {
                "legend_order": len(legend_rows) + 1,
                "material_name_ja": material_name,
                "material_group_key_broad_family": family,
                "plotted_point_count_1e-6_1e10": len(subset),
                "tab10_color_index": index % len(palette),
                "color_hex": matplotlib.colors.to_hex(color),
            }
        )

    line = np.geomspace(AXIS_MIN, AXIS_MAX, 500)
    ax.plot(line, line, color="black", linewidth=1.55)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ticks = 10.0 ** np.arange(-6, 11, 2)
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.minorticks_off()
    ax.grid(True, which="major", color="#b8b8b8", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Experimental electrical conductivity (S/m)", fontsize=22)
    ax.set_ylabel("Predicted electrical conductivity (S/m)", fontsize=22)
    ax.tick_params(axis="both", which="major", labelsize=18, length=5, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")
    fig.subplots_adjust(left=0.115, right=0.975, bottom=0.105, top=0.975)
    figure_path = output / "sigma_ml_scatter_all_material_colors_1e-6_1e10_step41.png"
    fig.savefig(figure_path, dpi=REFERENCE_DPI, facecolor="white")
    plt.close(fig)

    legend_fig = plt.figure(figsize=(8.2, 7.6))
    legend = legend_fig.legend(
        handles=legend_handles,
        loc="center",
        frameon=True,
        fontsize=14,
        markerscale=1.35,
        handletextpad=1.1,
        borderpad=0.9,
        labelspacing=0.85,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_edgecolor("#bdbdbd")
    legend_fig.savefig(
        output / "material_color_legend_1e-6_1e10_step41.png",
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(legend_fig)

    legend_table = pd.DataFrame(legend_rows)
    legend_table.to_csv(
        output / "material_color_legend_1e-6_1e10_step41.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output / "step41_wide_axes_report.txt").write_text(
        "\n".join(
            [
                "Step41 all-material sigma ML scatter on wide axes",
                "=================================================",
                f"axis range: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on both axes",
                f"reference canvas: {REFERENCE_WIDTH_PX} x {REFERENCE_HEIGHT_PX} px",
                f"reference width/height ratio: {REFERENCE_WIDTH_PX / REFERENCE_HEIGHT_PX:.9f}",
                f"input positive finite points: {len(data):,}",
                f"plotted points: {len(plotted):,}",
                "material colors: identical to Step40 and the supplied legend",
                "legend exported separately to preserve the reference plot ratio",
                "Existing Step40 and earlier figures were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    from PIL import Image

    with Image.open(figure_path) as image:
        if image.size != (REFERENCE_WIDTH_PX, REFERENCE_HEIGHT_PX):
            raise AssertionError(f"Unexpected figure size: {image.size}")
    if len(plotted) != len(data):
        raise AssertionError(
            f"Expected all positive points inside the wide axes: {len(plotted)} / {len(data)}"
        )
    print(f"Output: {output}")
    print(f"Figure: {figure_path}")
    print(f"Canvas: {REFERENCE_WIDTH_PX} x {REFERENCE_HEIGHT_PX} px")
    print(f"Plotted points: {len(plotted):,}")


if __name__ == "__main__":
    main()
