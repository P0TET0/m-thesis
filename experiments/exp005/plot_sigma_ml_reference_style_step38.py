"""Create one exp006-style sigma scatter PNG per material system.

The carrier type is not separated: every observation uses the same green circle.
RMSE is kept out of the figures and is copied to a separate CSV.  As in Step37,
RMSE covers every positive point before the displayed 1e2--1e8 range is clipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
import numpy as np
import pandas as pd

from plot_sigma_ml_minimal_transparent_step37 import (
    AXIS_MAX,
    AXIS_MIN,
    DEFAULT_INDEX,
    DEFAULT_INPUT,
    DEFAULT_METRICS,
    MATERIAL_NAMES_JA,
    build_rmse_index,
    family_order,
    require_columns,
    validate_transparency,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data/output/starrydata2_step38_sigma_ml_exp006_style"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create exp006-reference-style material-system scatter plots with "
            "one green marker style and RMSE in a separate CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def configure_reference_axes(ax: plt.Axes) -> None:
    ax.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")

    major = LogLocator(base=10.0, numticks=7)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    ax.xaxis.set_major_locator(major)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", color="#9aa0a6", linewidth=0.35, alpha=0.27)
    ax.tick_params(axis="both", which="major", labelsize=10, length=4, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def save_material_figure(
    data: pd.DataFrame,
    material_name_ja: str,
    path: Path,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
    fig.patch.set_alpha(0.0)

    ax.scatter(
        data["sigma_obs_S_per_m_step11"],
        data["sigma_pred_ML_S_per_m_step32"],
        s=20,
        alpha=0.68,
        marker="o",
        color="#1b9e77",
        edgecolors="none",
        label="データ",
        rasterized=True,
        clip_on=True,
    )
    line = np.geomspace(AXIS_MIN, AXIS_MAX, 400)
    ax.plot(line, line, color="black", linewidth=1.25, label="y=x", clip_on=True)
    configure_reference_axes(ax)
    ax.text(
        0.035,
        0.965,
        material_name_ja,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    ax.legend(loc="lower right", frameon=False, fontsize=10, handlelength=2.2)
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.115, top=0.975)
    fig.savefig(
        path,
        dpi=dpi,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor="none",
        transparent=False,
    )
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
            "all_points_rmse_log10",
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
    rmse_index = build_rmse_index(data, metrics, families)
    rmse_index["figure_file"] = [
        f"{row.material_order:02d}_{row.material_group_key_broad_family.replace('broad::', 'broad_')}.png"
        for row in rmse_index.itertuples(index=False)
    ]

    paths: list[Path] = []
    for row in rmse_index.itertuples(index=False):
        family = row.material_group_key_broad_family
        path = output / row.figure_file
        save_material_figure(
            data[data["material_group_key_broad_family"].eq(family)],
            MATERIAL_NAMES_JA.get(family, family),
            path,
            args.dpi,
        )
        paths.append(path)

    rmse_path = output / "rmse_by_material_system_step38.csv"
    rmse_index.to_csv(rmse_path, index=False, encoding="utf-8-sig")
    audit = validate_transparency(paths)
    audit.to_csv(output / "png_transparency_audit_step38.csv", index=False)
    (output / "step38_sigma_ml_exp006_style_report.txt").write_text(
        "\n".join(
            [
                "Step38 exp006-style sigma ML scatter plots",
                "==========================================",
                f"material-system figures: {len(paths)}",
                "style: material name at upper left, dense log grid, legend at lower right",
                "markers: one green circle series; carrier type is not separated",
                f"display range: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on x and y",
                "RMSE scope: all positive points before display clipping",
                f"RMSE location: {rmse_path.name}",
                "background: opaque white inside axes; transparent outside axes",
                "Existing Step37 and earlier figures were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if len(paths) != len(families) or len(list(output.glob("*.png"))) != len(families):
        raise AssertionError("Expected exactly one PNG per material system")
    print(f"Output: {output}")
    print(f"Material-system PNG files: {len(paths)}")
    print(f"RMSE: {rmse_path}")


if __name__ == "__main__":
    main()
