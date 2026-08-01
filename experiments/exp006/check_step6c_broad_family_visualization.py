import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed" / "step6c_broad_family"
REPORT_DIR = EXP_DIR / "reports" / "step6c_broad_family"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step6C broad_family visual diagnostics outputs.")
    parser.add_argument("--figure-index", type=Path, default=PROCESSED_DIR / "step6c_figure_index.csv")
    parser.add_argument("--diagnostics-summary", type=Path, default=PROCESSED_DIR / "step6c_visual_diagnostics_summary.csv")
    parser.add_argument("--original-vs-broad", type=Path, default=PROCESSED_DIR / "step6c_original_vs_broad_metrics_summary.csv")
    parser.add_argument("--group-performance", type=Path, default=PROCESSED_DIR / "step6c_broad_family_group_performance_summary.csv")
    parser.add_argument("--largest-error-diagnostics", type=Path, default=PROCESSED_DIR / "step6c_broad_largest_error_diagnostics_top100.csv")
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step6c_broad_family_visual_report.md")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    required_paths = [
        args.figure_index,
        args.diagnostics_summary,
        args.original_vs_broad,
        args.group_performance,
        args.largest_error_diagnostics,
        args.report,
    ]
    for path in required_paths:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    figure_index = pd.read_csv(args.figure_index, low_memory=False)
    diagnostics = pd.read_csv(args.diagnostics_summary, low_memory=False)
    original_vs_broad = pd.read_csv(args.original_vs_broad, low_memory=False)
    group_perf = pd.read_csv(args.group_performance, low_memory=False)
    largest = pd.read_csv(args.largest_error_diagnostics, low_memory=False)

    require_columns(
        figure_index,
        ["figure_id", "figure_path_png", "figure_path_pdf", "title", "source_file", "config_id", "n_points_plotted", "description"],
        "figure_index",
        failures,
    )
    require_columns(diagnostics, ["diagnostic_item", "status", "value", "comment"], "diagnostics_summary", failures)
    require_columns(
        original_vs_broad,
        [
            "default_label",
            "metric_weighting",
            "original_mae_log10",
            "broad_mae_log10",
            "delta_mae_broad_minus_original",
            "original_factor_2_accuracy",
            "broad_factor_2_accuracy",
            "delta_factor2_broad_minus_original",
            "original_factor_10_accuracy",
            "broad_factor_10_accuracy",
            "delta_factor10_broad_minus_original",
            "original_coverage_fraction",
            "broad_coverage_fraction",
            "delta_coverage_broad_minus_original",
        ],
        "original_vs_broad",
        failures,
    )
    require_columns(
        group_perf,
        [
            "material_group_key",
            "material_family_raw",
            "n_rows",
            "n_samples",
            "n_papers",
            "mae_log10",
            "rmse_log10",
            "factor_2_accuracy",
            "factor_10_accuracy",
            "coverage_fraction",
            "is_reliable_eval_group",
            "eval_group_reliability",
        ],
        "group_performance",
        failures,
    )
    require_columns(
        largest,
        [
            "config_id",
            "row_id",
            "sigma_S_per_m",
            "sigma_pred_S_per_m",
            "sigma_pred_over_exp",
            "log10_sigma_pred_over_exp",
            "abs_log10_sigma_pred_over_exp",
            "sigma0_ref_over_row_sigma0",
            "log10_sigma0_ref_over_row_sigma0",
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
    if len(figure_index["figure_id"]) != figure_index["figure_id"].nunique():
        failures.append("figure_id values are not unique")
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
    if len(figure_index) < 20:
        failures.append(f"expected at least 20 figures, found {len(figure_index)}")
    if diagnostics.empty:
        failures.append("diagnostics summary is empty")
    if original_vs_broad.empty:
        failures.append("original_vs_broad summary is empty")
    if group_perf.empty:
        failures.append("group performance summary is empty")
    if largest.empty:
        failures.append("largest error diagnostics is empty")
    if len(largest) > 100:
        failures.append("largest error diagnostics has more than 100 rows")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    diagnostic_items = set(diagnostics["diagnostic_item"])
    for item in [
        "broad_family_material_group_key_unique_count",
        "broad_family_default_mae_log10",
        "original_default_mae_log10",
        "delta_default_mae_log10",
        "material_family_vs_global_different_prediction_fraction",
        "broad_max_abs_log10_error",
        "recommended_next_action",
    ]:
        if item not in diagnostic_items:
            failures.append(f"diagnostics missing item: {item}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    png_count = sum(1 for value in figure_index["figure_path_png"] if Path(value).exists())
    pdf_count = sum(1 for value in figure_index["figure_path_pdf"] if Path(value).exists())
    print(f"figures: {len(figure_index)}")
    print(f"png files: {png_count}")
    print(f"pdf files: {pdf_count}")
    print(f"diagnostic items: {len(diagnostics)}")
    print(f"original_vs_broad rows: {len(original_vs_broad)}")
    print(f"group performance rows: {len(group_perf)}")
    print(f"largest error diagnostic rows: {len(largest)}")
    interesting = diagnostics[
        diagnostics["diagnostic_item"].isin(
            [
                "broad_family_material_group_key_unique_count",
                "broad_family_default_mae_log10",
                "delta_default_mae_log10",
                "material_family_vs_global_different_prediction_fraction",
                "broad_max_abs_log10_error",
            ]
        )
    ]
    print(interesting.to_string(index=False))
    print("step6c broad_family visualization checks passed")


if __name__ == "__main__":
    main()
