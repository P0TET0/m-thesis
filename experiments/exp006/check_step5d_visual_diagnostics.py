import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step5D-1 visual diagnostics outputs.")
    parser.add_argument("--figure-index", type=Path, default=PROCESSED_DIR / "step5d_figure_index.csv")
    parser.add_argument("--diagnostics-summary", type=Path, default=PROCESSED_DIR / "step5d_visual_diagnostics_summary.csv")
    parser.add_argument(
        "--diff-summary",
        type=Path,
        default=PROCESSED_DIR / "step5d_global_vs_material_family_prediction_diff_summary.csv",
    )
    parser.add_argument(
        "--largest-error-diagnostics",
        type=Path,
        default=PROCESSED_DIR / "step5d_largest_error_diagnostics_top100.csv",
    )
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step5d_visual_diagnostics_report.md")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [
        args.figure_index,
        args.diagnostics_summary,
        args.diff_summary,
        args.largest_error_diagnostics,
        args.report,
    ]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    figure_index = pd.read_csv(args.figure_index, low_memory=False)
    diagnostics = pd.read_csv(args.diagnostics_summary, low_memory=False)
    diff_summary = pd.read_csv(args.diff_summary, low_memory=False)
    largest = pd.read_csv(args.largest_error_diagnostics, low_memory=False)

    require_columns(
        figure_index,
        ["figure_id", "figure_path_png", "figure_path_pdf", "title", "source_file", "config_id", "n_points_plotted", "description"],
        "figure_index",
        failures,
    )
    require_columns(diagnostics, ["diagnostic_item", "status", "value", "comment"], "diagnostics_summary", failures)
    require_columns(
        diff_summary,
        [
            "comparison_label",
            "joined_row_count",
            "max_abs_delta_log10_sigma_pred",
            "different_prediction_count",
            "unique_material_group_key_count",
        ],
        "diff_summary",
        failures,
    )
    require_columns(
        largest,
        [
            "config_id",
            "row_id",
            "abs_log10_sigma_pred_over_exp",
            "outlier_direction",
            "outlier_severity",
            "likely_error_origin_hint",
        ],
        "largest_error_diagnostics",
        failures,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if figure_index.empty:
        failures.append("figure index is empty")
    png_missing = []
    pdf_missing = []
    for _, row in figure_index.iterrows():
        png = Path(row["figure_path_png"])
        pdf = Path(row["figure_path_pdf"])
        if not png.exists() or png.stat().st_size == 0:
            png_missing.append(str(png))
        if not pdf.exists() or pdf.stat().st_size == 0:
            pdf_missing.append(str(pdf))
    if png_missing:
        failures.append(f"missing or empty PNG files: {png_missing[:5]}")
    if pdf_missing:
        failures.append(f"missing or empty PDF files: {pdf_missing[:5]}")
    if len(figure_index["figure_id"]) != figure_index["figure_id"].nunique():
        failures.append("figure_id values are not unique")
    if diagnostics.empty:
        failures.append("diagnostics summary is empty")
    if diff_summary.empty:
        failures.append("diff summary is empty")
    if largest.empty:
        failures.append("largest error diagnostics is empty")
    if len(largest) > 100:
        failures.append("largest error diagnostics has more than 100 rows")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    print(f"figures: {len(figure_index)}")
    print(f"png files: {len(figure_index)}")
    print(f"pdf files: {len(figure_index)}")
    print(f"diagnostic items: {len(diagnostics)}")
    print(f"diff summary rows: {len(diff_summary)}")
    print(f"largest error diagnostic rows: {len(largest)}")
    if not diff_summary.empty:
        print(diff_summary[["comparison_label", "joined_row_count", "different_prediction_count", "unique_material_group_key_count"]].to_string(index=False))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step5d visual diagnostics checks passed")


if __name__ == "__main__":
    main()
