"""Annotate scatter plots using all-point metrics, then apply fixed display limits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_sigma_ml_scatter_with_metrics_step35 import (
    AXIS_MAX,
    AXIS_MIN,
    PROJECT_ROOT,
    add_metric_box,
    family_order,
    format_axes,
    metric_lookup,
    require_columns,
    safe_slug,
)


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
    PROJECT_ROOT / "data/output/starrydata2_step36_sigma_ml_scatter_all_point_metrics"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate/annotate y=x agreement from all positive points first, "
            "then clip only the displayed axes to 1e2--1e8 S/m."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def all_point_metric_text(row: pd.Series | None) -> str:
    if row is None or int(row.get("all_points_point_count", 0)) == 0:
        return "n(metric, all) = 0\nnot evaluated"
    return "\n".join(
        [
            f"n(metric, all) = {int(row['all_points_point_count']):,}",
            f"visible = {int(row['visible_1e2_1e8_point_count']):,}",
            f"RMSE(log10, all) = {row['all_points_rmse_log10']:.3f}",
            f"MAE(log10, all) = {row['all_points_mae_log10']:.3f}",
            f"Factor-2(all) = {100.0 * row['all_points_within_factor_2_rate']:.1f}%",
            f"CCC(log10, all) = {row['all_points_concordance_correlation_log10']:.3f}",
        ]
    )


def validate_metric_counts(
    data: pd.DataFrame, families: list[str], lookup: dict[str, pd.Series]
) -> None:
    for family in families:
        actual = int(data["material_group_key_broad_family"].eq(family).sum())
        row = lookup.get(family)
        expected = int(row["all_points_point_count"]) if row is not None else 0
        if actual != expected:
            raise AssertionError(
                f"All-point metric count mismatch for {family}: data={actual}, metrics={expected}"
            )
    overall = lookup.get("__all_material_systems__")
    if overall is None or int(overall["all_points_point_count"]) != len(data):
        raise AssertionError("Overall all-point metric count does not match scatter input")


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
        fig, ax = plt.subplots(figsize=(6.5, 6.1))
        # Add every positive point first. Axis limits are applied afterwards by format_axes().
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=13,
            alpha=0.38,
            color="#1b9e77",
            edgecolors="none",
        )
        add_metric_box(ax, all_point_metric_text(lookup.get(family)), fontsize=8.8)
        format_axes(ax)
        ax.set_title(f"{family} | metrics from all points")
        fig.tight_layout()
        fig.savefig(
            individual / f"sigma_ml_scatter_all_point_metrics_{safe_slug(family)}_step36.png",
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
    fig, ax = plt.subplots(figsize=(9.4, 7.6))
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
    add_metric_box(
        ax,
        all_point_metric_text(lookup.get("__all_material_systems__")),
        fontsize=9.2,
    )
    format_axes(ax)
    ax.set_title("DOI holdout | metrics from all points, display clipped afterwards")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_all_point_metrics_all_step36.png", dpi=dpi)
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.0, 5.0 * nrows), squeeze=False)
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
        add_metric_box(ax, all_point_metric_text(lookup.get(family)), fontsize=6.9)
        format_axes(ax)
        ax.set_title(family, fontsize=10)
    for ax in axes.flat[len(families) :]:
        ax.axis("off")
    fig.suptitle(
        "Metrics calculated from all points first | display x/y clipped to 10^2--10^8 S/m",
        fontsize=15,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_all_point_metrics_grid_step36.png", dpi=dpi)
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
            "all_points_point_count",
            "visible_1e2_1e8_point_count",
            "all_points_rmse_log10",
            "all_points_mae_log10",
            "all_points_within_factor_2_rate",
            "all_points_concordance_correlation_log10",
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
    # This check proves the annotated metrics include every point before display clipping.
    validate_metric_counts(data, families, lookup)
    create_individuals(data, families, lookup, output, args.dpi)
    create_combined(data, families, lookup, output, args.dpi)
    create_grid(data, families, lookup, output, args.dpi)
    metrics.to_csv(output / "all_point_metrics_used_for_annotations_step36.csv", index=False)
    (output / "step36_all_point_metrics_scatter_report.txt").write_text(
        "\n".join(
            [
                "Step36 scatter plots with metrics calculated before clipping",
                "============================================================",
                f"scatter input: {args.input.resolve()}",
                f"metrics input: {args.metrics.resolve()}",
                f"all positive points used for metrics: {len(data):,}",
                f"display range applied after metrics: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on x and y",
                f"material systems: {len(families)}",
                "annotations: all-point n, visible n, all-point RMSE/MAE/Factor-2/CCC",
                "Existing Step35 figures were not modified.",
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
    print(f"All points used for metrics: {len(data):,}")
    print(f"Material systems: {len(families)}")
    print(f"PNG files: {actual_png}")


if __name__ == "__main__":
    main()
