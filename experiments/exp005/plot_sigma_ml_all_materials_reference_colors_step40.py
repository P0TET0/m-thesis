"""Plot all exp005 ML sigma predictions with the exp006 reference color order."""

from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data/output/starrydata2_step40_sigma_ml_all_materials_reference_colors"
)
AXIS_MIN = 1e2
AXIS_MAX = 1e8

# This order reproduces the material-to-color mapping in the supplied exp006
# legend.  Colors repeat after the tenth entry, as in Matplotlib's tab10 cycle.
MATERIAL_SPECS = [
    ("broad::other_formula_system", "その他の化学式系"),
    ("broad::oxide", "酸化物系"),
    ("broad::selenide", "セレン化物系"),
    ("broad::BiTe_like", "Bi-Te系"),
    ("broad::CoSb_skutterudite_like", "Co-Sb系・スクッテルダイト系"),
    ("broad::SbTe_like", "Sb-Te系"),
    ("broad::sulfide", "硫化物系"),
    ("broad::BiSbTe_tetradymite_like", "Bi-Sb-Te系"),
    ("broad::PbTe_like", "Pb-Te系"),
    ("broad::telluride", "テルル化物系"),
    ("broad::SnTe_like", "Sn-Te系"),
    ("broad::Mg2SiSn_like", "Mg-Si/Sn系"),
    ("broad::SiGe_like", "Si-Ge系"),
    ("broad::GeTe_like", "Ge-Te系"),
    ("unknown_material_group", "分類不能"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an all-material sigma scatter with the supplied reference colors."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists; pass --replace-output to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    required = [
        "material_group_key_broad_family",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_ML_S_per_m_step32",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    for column in required[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    valid = (
        np.isfinite(data[required[1]])
        & np.isfinite(data[required[2]])
        & data[required[1]].gt(0)
        & data[required[2]].gt(0)
    )
    data = data.loc[valid].copy()
    visible = data[
        data[required[1]].between(AXIS_MIN, AXIS_MAX)
        & data[required[2]].between(AXIS_MIN, AXIS_MAX)
    ].copy()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    palette = list(plt.get_cmap("tab10").colors)
    fig, ax = plt.subplots(figsize=(11.0, 8.0))
    legend_rows: list[dict[str, object]] = []
    for index, (family, material_name) in enumerate(MATERIAL_SPECS):
        subset = visible[visible[required[0]].eq(family)]
        if subset.empty:
            continue
        color = palette[index % len(palette)]
        ax.scatter(
            subset[required[1]],
            subset[required[2]],
            s=13,
            alpha=0.45,
            color=color,
            edgecolors="none",
            label=f"{material_name} (n={len(subset)})",
            rasterized=True,
        )
        legend_rows.append(
            {
                "legend_order": len(legend_rows) + 1,
                "material_name_ja": material_name,
                "material_group_key_broad_family": family,
                "plotted_point_count_1e2_1e8": len(subset),
                "tab10_color_index": index % len(palette),
                "color_hex": matplotlib.colors.to_hex(color),
            }
        )

    line = np.geomspace(AXIS_MIN, AXIS_MAX, 400)
    ax.plot(line, line, color="black", linestyle="--", linewidth=1.35, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(AXIS_MIN, AXIS_MAX)
    ax.set_ylim(AXIS_MIN, AXIS_MAX)
    ax.set_xlabel("Measured sigma (S/m)")
    ax.set_ylabel("exp005 ML predicted sigma (S/m)")
    ax.set_title("DOI-group holdout: all exp006 material systems | fixed 10^2--10^8")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9.2,
        frameon=True,
        markerscale=1.5,
        handletextpad=0.8,
    )
    fig.tight_layout()
    figure_path = output / "sigma_ml_scatter_all_material_colors_step40.png"
    fig.savefig(figure_path, dpi=args.dpi, facecolor="white")
    plt.close(fig)

    legend_table = pd.DataFrame(legend_rows)
    legend_table.to_csv(output / "material_color_legend_step40.csv", index=False, encoding="utf-8-sig")
    (output / "step40_all_material_colors_report.txt").write_text(
        "\n".join(
            [
                "Step40 all-material sigma ML scatter with reference colors",
                "==========================================================",
                f"input positive finite points: {len(data):,}",
                f"plotted points inside 1e2--1e8 on both axes: {len(visible):,}",
                f"legend material systems with points: {len(legend_table)}",
                "color mapping: supplied exp006 legend order using Matplotlib tab10",
                "legend n: plotted point count after fixed-axis filtering",
                "half-Heusler is omitted because its plotted point count is zero",
                "Existing Step33 and later figures were not modified.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if len(legend_table) != 15 or len(visible) != 11930:
        raise AssertionError(
            f"Unexpected output counts: legend={len(legend_table)}, visible={len(visible)}"
        )
    print(f"Output: {output}")
    print(f"Figure: {figure_path}")
    print(f"Plotted points: {len(visible):,}")


if __name__ == "__main__":
    main()
