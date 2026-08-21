"""Create one minimal transparent scatter PNG per material system.

RMSE is not drawn on the figure.  It is exported separately and is calculated
from all positive points before the displayed axes are clipped to 1e2--1e8.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter
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
    PROJECT_ROOT / "data/output/starrydata2_step37_sigma_ml_minimal_transparent"
)
AXIS_MIN = 1e2
AXIS_MAX = 1e8

MATERIAL_NAMES_JA = {
    "broad::other_formula_system": "その他の化学式系",
    "broad::oxide": "酸化物系",
    "broad::CoSb_skutterudite_like": "Co-Sb系・スクッテルダイト系",
    "broad::selenide": "セレン化物系",
    "broad::SbTe_like": "Sb-Te系",
    "broad::BiTe_like": "Bi-Te系",
    "broad::BiSbTe_tetradymite_like": "Bi-Sb-Te系",
    "broad::sulfide": "硫化物系",
    "broad::PbTe_like": "Pb-Te系",
    "broad::Mg2SiSn_like": "Mg-Si/Sn系",
    "broad::telluride": "テルル化物系",
    "broad::SnTe_like": "Sn-Te系",
    "broad::SiGe_like": "Si-Ge系",
    "broad::GeTe_like": "Ge-Te系",
    "unknown_material_group": "分類不能",
    "broad::half_heusler": "ハーフホイスラー系",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one unlabeled scatter per material system with a white axes "
            "field, transparent exterior, and RMSE in a separate CSV."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
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


def configure_minimal_axes(ax: plt.Axes) -> None:
    # Only the plotting field is opaque white.  The surrounding figure stays transparent.
    ax.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    major = LogLocator(base=10.0, numticks=7)
    minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    ax.xaxis.set_major_locator(major)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=7))
    ax.xaxis.set_minor_locator(minor)
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color="#bdbdbd", alpha=0.48, linewidth=0.65)
    ax.grid(True, which="minor", color="#d9d9d9", alpha=0.34, linewidth=0.42)
    ax.tick_params(axis="both", which="major", labelsize=10, length=5, width=0.9)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.55)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.0)


def save_material_figure(data: pd.DataFrame, path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    # Transparent figure/padding; the axes patch is reset to opaque white below.
    fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
    fig.patch.set_alpha(0.0)
    ax.scatter(
        data["sigma_obs_S_per_m_step11"],
        data["sigma_pred_ML_S_per_m_step32"],
        s=13,
        alpha=0.42,
        color="#1b9e77",
        edgecolors="none",
        clip_on=True,
    )
    ax.plot(
        [AXIS_MIN, AXIS_MAX],
        [AXIS_MIN, AXIS_MAX],
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        clip_on=True,
    )
    configure_minimal_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.14, top=0.97)
    fig.savefig(
        path,
        dpi=dpi,
        facecolor=(1.0, 1.0, 1.0, 0.0),
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)


def build_rmse_index(
    data: pd.DataFrame,
    metrics: pd.DataFrame,
    families: list[str],
) -> pd.DataFrame:
    metric_lookup = metrics.set_index("material_group_key_broad_family", drop=False)
    records: list[dict[str, Any]] = []
    for position, family in enumerate(families, start=1):
        subset = data[data["material_group_key_broad_family"].eq(family)]
        row = metric_lookup.loc[family] if family in metric_lookup.index else None
        rmse = row["all_points_rmse_log10"] if row is not None else math.nan
        expected_count = int(row["all_points_point_count"]) if row is not None else 0
        if len(subset) != expected_count:
            raise AssertionError(
                f"All-point RMSE count mismatch for {family}: data={len(subset)}, metrics={expected_count}"
            )
        filename = f"{position:02d}_{safe_slug(family)}.png"
        records.append(
            {
                "material_order": position,
                "material_name_ja": MATERIAL_NAMES_JA.get(family, family),
                "material_group_key_broad_family": family,
                "rmse_log10_all_points": rmse,
                "figure_file": filename,
            }
        )
    return pd.DataFrame(records)


def validate_transparency(paths: list[Path]) -> pd.DataFrame:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to validate PNG transparency") from exc
    records: list[dict[str, Any]] = []
    for path in paths:
        with Image.open(path).convert("RGBA") as image:
            width, height = image.size
            corner_alpha = image.getpixel((0, 0))[3]
            center_alpha = image.getpixel((width // 2, height // 2))[3]
            records.append(
                {
                    "figure_file": path.name,
                    "width_px": width,
                    "height_px": height,
                    "outside_corner_alpha": corner_alpha,
                    "plot_field_center_alpha": center_alpha,
                    "outside_is_transparent": corner_alpha == 0,
                    "plot_field_is_opaque": center_alpha == 255,
                }
            )
    audit = pd.DataFrame(records)
    if not audit["outside_is_transparent"].all():
        raise AssertionError("At least one PNG has a non-transparent exterior corner")
    if not audit["plot_field_is_opaque"].all():
        raise AssertionError("At least one PNG has a non-opaque plot field center")
    return audit


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
    paths: list[Path] = []
    for row in rmse_index.itertuples(index=False):
        family = row.material_group_key_broad_family
        path = output / row.figure_file
        save_material_figure(
            data[data["material_group_key_broad_family"].eq(family)],
            path,
            args.dpi,
        )
        paths.append(path)

    rmse_index.to_csv(output / "rmse_by_material_system_step37.csv", index=False)
    transparency = validate_transparency(paths)
    transparency.to_csv(output / "png_transparency_audit_step37.csv", index=False)
    (output / "step37_minimal_transparent_scatter_report.txt").write_text(
        "\n".join(
            [
                "Step37 minimal transparent sigma ML scatter plots",
                "=================================================",
                f"material-system figures: {len(paths)}",
                "one PNG per material system; no combined or grid figures",
                "figure text: tick values only; no title, axis labels, material name, or RMSE",
                f"display range: {AXIS_MIN:.0e}--{AXIS_MAX:.0e} S/m on x and y",
                "RMSE scope: all positive points before display clipping",
                "RMSE location: rmse_by_material_system_step37.csv",
                "background: opaque white inside axes; transparent outside axes",
                "identity line: black dashed y=x",
                "Existing Step36 and earlier figures were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if len(paths) != len(families) or len(list(output.glob("*.png"))) != len(families):
        raise AssertionError("Expected exactly one PNG per material system")
    print("Done.")
    print(f"Output: {output}")
    print(f"Material-system PNG files: {len(paths)}")
    print("RMSE is separate; figures contain tick values only.")
    print("Transparency validation: exterior alpha=0, plot-field alpha=255 for every PNG.")


if __name__ == "__main__":
    main()
