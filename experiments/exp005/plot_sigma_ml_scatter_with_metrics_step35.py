"""Create fixed-axis Step35 scatter plots annotated with Step34 y=x metrics."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/"
    "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv"
)
DEFAULT_METRICS = (
    PROJECT_ROOT
    / "data/output/starrydata2_step34_sigma_ml_y_equals_x_metrics/"
    "y_equals_x_metrics_by_material_system_step34.csv"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/material_system_index_step32.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/output/starrydata2_step35_sigma_ml_scatter_annotated_metrics"
)
AXIS_MIN = 1e2
AXIS_MAX = 1e8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate fixed-axis exp005 sigma scatter plots with y=x agreement metrics."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def safe_slug(value: str) -> str:
    value = value.replace("::", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def family_order(data: pd.DataFrame, index_path: Path) -> list[str]:
    present = set(data["material_group_key_broad_family"].astype(str))
    ordered: list[str] = []
    if index_path.exists():
        index = pd.read_csv(index_path)
        if "material_group_key_broad_family" in index.columns:
            ordered = list(dict.fromkeys(index["material_group_key_broad_family"].astype(str)))
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def format_axes(ax: plt.Axes) -> None:
    ax.plot(
        [AXIS_MIN, AXIS_MAX],
        [AXIS_MIN, AXIS_MAX],
        color="#222222",
        linestyle="--",
        lw=1.2,
        label="ideal y=x",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("Measured sigma (S/m)")
    ax.set_ylabel("exp005 ML predicted sigma (S/m)")
    ax.grid(True, which="both", alpha=0.22)


def metric_text(row: pd.Series | None) -> str:
    if row is None or int(row.get("visible_1e2_1e8_point_count", 0)) == 0:
        return "n = 0\nnot evaluated"
    return "\n".join(
        [
            f"n = {int(row['visible_1e2_1e8_point_count']):,}",
            f"RMSE(log10) = {row['visible_1e2_1e8_rmse_log10']:.3f}",
            f"MAE(log10) = {row['visible_1e2_1e8_mae_log10']:.3f}",
            f"Factor-2 = {100.0 * row['visible_1e2_1e8_within_factor_2_rate']:.1f}%",
            f"CCC(log10) = {row['visible_1e2_1e8_concordance_correlation_log10']:.3f}",
        ]
    )


def add_metric_box(ax: plt.Axes, text: str, fontsize: float) -> None:
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        linespacing=1.25,
        bbox={
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#888888",
            "boxstyle": "round,pad=0.35",
        },
        zorder=10,
    )


def metric_lookup(metrics: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        str(row["material_group_key_broad_family"]): row
        for _, row in metrics.iterrows()
    }


def create_individuals(
    data: pd.DataFrame,
    families: list[str],
    lookup: dict[str, pd.Series],
    output: Path,
    dpi: int,
) -> None:
    individual = output / "individual_material_systems"
    individual.mkdir(parents=True, exist_ok=True)
    for family in families:
        subset = data[data["material_group_key_broad_family"].eq(family)]
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=13,
            alpha=0.38,
            color="#1b9e77",
            edgecolors="none",
        )
        format_axes(ax)
        add_metric_box(ax, metric_text(lookup.get(family)), fontsize=9.2)
        ax.set_title(f"{family} | DOI holdout")
        fig.tight_layout()
        fig.savefig(
            individual / f"sigma_ml_scatter_metrics_{safe_slug(family)}_step35.png",
            dpi=dpi,
        )
        plt.close(fig)


def create_combined(
    data: pd.DataFrame,
    families: list[str],
    lookup: dict[str, pd.Series],
    output: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(families)))
    for color, family in zip(colors, families):
        subset = data[data["material_group_key_broad_family"].eq(family)]
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=11,
            alpha=0.35,
            color=color,
            edgecolors="none",
            label=family,
        )
    format_axes(ax)
    add_metric_box(ax, metric_text(lookup.get("__all_material_systems__")), fontsize=9.5)
    ax.set_title("DOI-group holdout: all exp006 material systems | y=x agreement")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_all_with_metrics_step35.png", dpi=dpi)
    plt.close(fig)


def create_grid(
    data: pd.DataFrame,
    families: list[str],
    lookup: dict[str, pd.Series],
    output: Path,
    dpi: int,
) -> None:
    ncols = 3
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.8, 4.9 * nrows), squeeze=False)
    for ax, family in zip(axes.flat, families):
        subset = data[data["material_group_key_broad_family"].eq(family)]
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=11,
            alpha=0.36,
            color="#1b9e77",
            edgecolors="none",
        )
        format_axes(ax)
        add_metric_box(ax, metric_text(lookup.get(family)), fontsize=7.4)
        ax.set_title(family, fontsize=10)
    for ax in axes.flat[len(families) :]:
        ax.axis("off")
    fig.suptitle(
        "DOI-group holdout scatter by exp006 material system | x/y: 10^2--10^8 S/m",
        fontsize=15,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_grid_with_metrics_step35.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists; pass --replace-output to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)
    require_columns(
        data,
        [
            "material_group_key_broad_family",
            "sigma_obs_S_per_m_step11",
            "sigma_pred_ML_S_per_m_step32",
        ],
        "scatter input",
    )
    require_columns(
        metrics,
        [
            "material_group_key_broad_family",
            "visible_1e2_1e8_point_count",
            "visible_1e2_1e8_rmse_log10",
            "visible_1e2_1e8_mae_log10",
            "visible_1e2_1e8_within_factor_2_rate",
            "visible_1e2_1e8_concordance_correlation_log10",
        ],
        "Step34 metrics",
    )
    for column in ["sigma_obs_S_per_m_step11", "sigma_pred_ML_S_per_m_step32"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        np.isfinite(data["sigma_obs_S_per_m_step11"])
        & data["sigma_obs_S_per_m_step11"].gt(0)
        & np.isfinite(data["sigma_pred_ML_S_per_m_step32"])
        & data["sigma_pred_ML_S_per_m_step32"].gt(0)
    ].copy()

    families = family_order(data, args.material_index)
    lookup = metric_lookup(metrics)
    create_individuals(data, families, lookup, output, args.dpi)
    create_combined(data, families, lookup, output, args.dpi)
    create_grid(data, families, lookup, output, args.dpi)
    metrics.to_csv(output / "step34_metrics_used_for_annotations_step35.csv", index=False)
    (output / "step35_annotated_scatter_report.txt").write_text(
        "\n".join(
            [
                "Step35 annotated exp005 sigma ML scatter figures",
                "=================================================",
                f"scatter input: {args.input.resolve()}",
                f"metrics input: {args.metrics.resolve()}",
                f"axis range: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on x and y",
                f"material systems: {len(families)}",
                "annotations: n, RMSE(log10), MAE(log10), Factor-2 rate, CCC(log10)",
                "Existing Step33 figures were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    expected_png = len(families) + 2
    actual_png = len(list(output.rglob("*.png")))
    if actual_png != expected_png:
        raise AssertionError(f"Expected {expected_png} PNG files, found {actual_png}")
    print("Done.")
    print(f"Output: {output}")
    print(f"Material systems: {len(families)}")
    print(f"PNG files: {actual_png}")


if __name__ == "__main__":
    main()
