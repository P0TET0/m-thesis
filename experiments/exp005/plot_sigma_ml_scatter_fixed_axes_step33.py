"""Redraw every Step32 DOI-holdout scatter with fixed 10^2--10^8 axes."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

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
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/material_system_index_step32.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/output/starrydata2_step33_sigma_ml_scatter_fixed_1e2_1e8"
)
AXIS_MIN = 1e2
AXIS_MAX = 1e8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create new Step33 scatter figures with x/y axes fixed to 1e2--1e8 S/m."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def safe_slug(value: str) -> str:
    value = value.replace("::", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def family_order(data: pd.DataFrame, index_path: Path) -> list[str]:
    present = set(data["material_group_key_broad_family"].astype(str))
    ordered: list[str] = []
    if index_path.exists():
        index = pd.read_csv(index_path)
        if "material_group_key_broad_family" in index.columns:
            # Keep even zero-evaluation families so "all material systems"
            # includes an explicitly empty, labeled panel.
            ordered = list(dict.fromkeys(index["material_group_key_broad_family"].astype(str)))
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def visible_mask(data: pd.DataFrame) -> pd.Series:
    obs = data["sigma_obs_S_per_m_step11"]
    pred = data["sigma_pred_ML_S_per_m_step32"]
    return obs.between(AXIS_MIN, AXIS_MAX) & pred.between(AXIS_MIN, AXIS_MAX)


def format_axes(ax: plt.Axes) -> None:
    ax.plot(
        [AXIS_MIN, AXIS_MAX],
        [AXIS_MIN, AXIS_MAX],
        color="#222222",
        linestyle="--",
        lw=1.2,
        label="ideal",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("Measured sigma (S/m)")
    ax.set_ylabel("exp005 ML predicted sigma (S/m)")
    ax.grid(True, which="both", alpha=0.22)


def create_individual_figures(
    data: pd.DataFrame, families: list[str], output: Path, dpi: int
) -> list[dict[str, object]]:
    individual = output / "individual_material_systems"
    individual.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, object]] = []
    for family in families:
        subset = data[data["material_group_key_broad_family"].eq(family)].copy()
        visible = visible_mask(subset)
        fig, ax = plt.subplots(figsize=(6.2, 5.8))
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=13,
            alpha=0.38,
            color="#1b9e77",
            edgecolors="none",
        )
        format_axes(ax)
        ax.set_title(f"{family} | DOI holdout")
        ax.text(
            0.03,
            0.97,
            f"visible: {int(visible.sum()):,} / total: {len(subset):,}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )
        fig.tight_layout()
        filename = f"sigma_ml_scatter_fixed_1e2_1e8_{safe_slug(family)}.png"
        fig.savefig(individual / filename, dpi=dpi)
        plt.close(fig)
        audit.append(
            {
                "material_group_key_broad_family": family,
                "total_point_count": len(subset),
                "visible_point_count_1e2_1e8": int(visible.sum()),
                "clipped_point_count_1e2_1e8": int((~visible).sum()),
                "figure": f"individual_material_systems/{filename}",
            }
        )
    return audit


def create_combined(data: pd.DataFrame, families: list[str], output: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 7.2))
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
    ax.set_title("DOI-group holdout: all exp006 material systems | fixed 10^2--10^8")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_all_fixed_1e2_1e8_step33.png", dpi=dpi)
    plt.close(fig)


def create_grid(data: pd.DataFrame, families: list[str], output: Path, dpi: int) -> None:
    ncols = 3
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.8 * nrows), squeeze=False)
    for ax, family in zip(axes.flat, families):
        subset = data[data["material_group_key_broad_family"].eq(family)]
        visible = visible_mask(subset)
        ax.scatter(
            subset["sigma_obs_S_per_m_step11"],
            subset["sigma_pred_ML_S_per_m_step32"],
            s=11,
            alpha=0.36,
            color="#1b9e77",
            edgecolors="none",
        )
        format_axes(ax)
        ax.set_title(family, fontsize=10)
        ax.text(
            0.03,
            0.97,
            f"visible {int(visible.sum()):,}/{len(subset):,}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
    for ax in axes.flat[len(families) :]:
        ax.axis("off")
    fig.suptitle(
        "DOI-group holdout scatter by exp006 material system | x/y: 10^2--10^8 S/m",
        fontsize=15,
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(output / "sigma_ml_scatter_grid_fixed_1e2_1e8_step33.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output already exists; pass --replace-output to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    require_columns(
        data,
        [
            "material_group_key_broad_family",
            "sigma_obs_S_per_m_step11",
            "sigma_pred_ML_S_per_m_step32",
        ],
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

    audit = create_individual_figures(data, families, output, args.dpi)
    create_combined(data, families, output, args.dpi)
    create_grid(data, families, output, args.dpi)
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(output / "scatter_fixed_axes_visibility_audit_step33.csv", index=False)
    (output / "step33_scatter_fixed_axes_report.txt").write_text(
        "\n".join(
            [
                "Step33 fixed-axis sigma ML scatter figures",
                "==========================================",
                f"input: {args.input.resolve()}",
                f"axis range: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on both axes",
                f"material systems: {len(families)}",
                f"total points: {len(data):,}",
                f"visible points: {int(visible_mask(data).sum()):,}",
                f"clipped points: {int((~visible_mask(data)).sum()):,}",
                "Existing Step32 figures were not modified.",
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
    print(f"Visible points: {int(visible_mask(data).sum()):,} / {len(data):,}")


if __name__ == "__main__":
    main()
