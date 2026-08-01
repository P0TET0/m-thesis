import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from plot_starrydata2_sige_vs_similar_count_material_step28 import (
    DEFAULT_INPUT,
    PROJECT_ROOT,
    SIGE_SYSTEM,
    element_system,
    load_normalized_curves,
    safe_system_filename,
)
from plot_starrydata2_sigma_temperature_comparison_step27 import (
    plot_curves_by_label,
    write_curve_map,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "output" / "starrydata2_step29_sige_vs_listed_materials"
)

LISTED_MATERIAL_SYSTEMS = [
    "Bi-Sb-Te",
    "Ca-Co-O",
    "Bi-Te",
    "Bi-Se-Te",
    "Sb-Zn",
    "Co-Sb",
    "La-O-Sr-Ti",
    "Cu-Se",
    "Al-O-Zn",
    "Mn-Si",
    "Mg-Si-Sn",
    "Ni-Sn-Ti",
    "La-Mn-O-Sr",
    "Bi-Sb",
    "Co-Na-O",
    "Mg-Sb-Si-Sn",
    "Ba-Cu-O-Y",
    "Sb-Te",
    "Co-Sb-Yb",
    "Pb-Te",
    "Pb-Se",
    "Pb-Sb-Te",
    "Sn-Te",
    "Ge-Te",
    "Ge-Sb-Te",
    "Ag-Sb-Te",
    "Ag-Pb-Sb-Te",
    "Mg-Si",
    "Mg-Sb-Si",
    "Co-Ni-Sb",
    "Hf-Ni-Sb-Sn-Zr",
    "Fe-Si",
    "Bi-Ca-Co-O",
    "Bi-Cu-O-Se",
    "O-Ti",
    "Nb-O-Sr-Ti",
    "Ba-Co-Fe-O-Sr",
    "Co-La-O-Sr",
    "O-Zn",
    "Ga-O-Zn",
    "O-Sr-Ti",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one SiGe-vs-material conductivity plot for each listed material system."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--no-log-y",
        action="store_true",
        help="Use a linear conductivity axis instead of the default log scale.",
    )
    parser.add_argument("--y-quantile-low", type=float, default=0.01)
    parser.add_argument("--y-quantile-high", type=float, default=0.995)
    parser.add_argument(
        "--also-without-trend-lines",
        action="store_true",
        help="Also save a version without the thick trend lines for every material.",
    )
    return parser.parse_args()


def unique_material_systems() -> list[str]:
    seen: set[str] = set()
    systems: list[str] = []
    for system in LISTED_MATERIAL_SYSTEMS:
        if system == SIGE_SYSTEM or system in seen:
            continue
        seen.add(system)
        systems.append(system)
    return systems


def select_pair(curves: pd.DataFrame, material_system: str) -> pd.DataFrame:
    selected = curves[curves["element_system"].isin([SIGE_SYSTEM, material_system])].copy()
    selected["family_label"] = selected["element_system"].map(
        {SIGE_SYSTEM: "SiGe", material_system: material_system}
    )
    return selected


def plot_pair(
    selected: pd.DataFrame,
    material_system: str,
    output_dir: Path,
    *,
    log_y: bool,
    dpi: int,
    y_quantile_low: float,
    y_quantile_high: float,
    show_median: bool,
) -> Path:
    slug = safe_system_filename(material_system)
    suffix = "" if show_median else "_without_trend_lines"
    output_path = output_dir / f"figure_sige_vs_{slug}_sigma{suffix}.png"
    labels = ["SiGe", material_system]
    title = f"Starrydata2 Si-Ge vs {material_system} electrical conductivity curves"
    plot_curves_by_label(
        selected,
        output_path,
        title,
        "family_label",
        labels,
        log_y=log_y,
        dpi=dpi,
        y_quantile_low=y_quantile_low,
        y_quantile_high=y_quantile_high,
        line_styles={"SiGe": "-", material_system: "--"},
        line_colors={"SiGe": "#6baed6", material_system: "#fdae6b"},
        median_colors={"SiGe": "#08519c", material_system: "#b44a00"},
        line_alphas={"SiGe": 0.26, material_system: 0.22},
        line_widths={"SiGe": 0.75, material_system: 0.75},
        median_widths={"SiGe": 3.6, material_system: 3.8},
        median_line_styles={"SiGe": "-", material_system: "-"},
        font_size=13,
        title_font_size=16,
        label_font_size=14,
        tick_label_size=12,
        legend_font_size=12,
        legend_line_width=3.0,
        show_median_in_legend=True,
        show_median=show_median,
    )
    return output_path


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "step29_sige_vs_listed_materials_summary.csv", index=False)
    report_lines = [
        f"materials_requested: {len(summary)}",
        f"figures_created: {int(summary['figure_created'].sum()) if not summary.empty else 0}",
        f"missing_or_empty_materials: {int((~summary['figure_created']).sum()) if not summary.empty else 0}",
    ]
    (output_dir / "step29_sige_vs_listed_materials_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_y = not args.no_log_y

    curves = load_normalized_curves(args.input)
    system_counts = curves["element_system"].value_counts()
    sige_count = int(system_counts.get(SIGE_SYSTEM, 0))
    if sige_count == 0:
        raise SystemExit("SiGe curves were not found")

    summary_rows: list[dict[str, Any]] = []
    curve_map_frames: list[pd.DataFrame] = []
    for material_system in unique_material_systems():
        material_count = int(system_counts.get(material_system, 0))
        selected = select_pair(curves, material_system)
        figure_created = material_count > 0 and not selected.empty
        figure_path = ""
        figure_without_trend_path = ""
        if figure_created:
            output_path = plot_pair(
                selected,
                material_system,
                args.output_dir,
                log_y=log_y,
                dpi=args.dpi,
                y_quantile_low=args.y_quantile_low,
                y_quantile_high=args.y_quantile_high,
                show_median=True,
            )
            figure_path = output_path.name
            if args.also_without_trend_lines:
                no_trend_path = plot_pair(
                    selected,
                    material_system,
                    args.output_dir,
                    log_y=log_y,
                    dpi=args.dpi,
                    y_quantile_low=args.y_quantile_low,
                    y_quantile_high=args.y_quantile_high,
                    show_median=False,
                )
                figure_without_trend_path = no_trend_path.name
            curve_map_frames.append(selected.assign(comparison_material_system=material_system))

        summary_rows.append(
            {
                "comparison_material_system": material_system,
                "sige_curves": sige_count,
                "comparison_curves": material_count,
                "figure_created": figure_created,
                "figure_file": figure_path,
                "figure_without_trend_lines_file": figure_without_trend_path,
            }
        )

    write_summary(summary_rows, args.output_dir)
    if curve_map_frames:
        write_curve_map(pd.concat(curve_map_frames, ignore_index=True), args.output_dir)

    print(f"output_dir: {args.output_dir}")
    print(f"materials_requested: {len(summary_rows)}")
    print(f"figures_created: {sum(1 for row in summary_rows if row['figure_created'])}")
    print("summary: step29_sige_vs_listed_materials_summary.csv")


if __name__ == "__main__":
    main()
