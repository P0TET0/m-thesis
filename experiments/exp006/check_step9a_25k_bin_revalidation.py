import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_ROOT = EXP_DIR / "reports"
DEFAULT_INPUT = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet"
DEFAULT_OUTPUT = PROCESSED_DIR / "step9a_25k_bin_broad_family"
DEFAULT_REPORT = REPORT_ROOT / "step9a_25k_bin_broad_family" / "step9a_25k_bin_revalidation_report.md"
HUNDRED_K_DIR = PROCESSED_DIR / "step6b_broad_family"

OLD_COLUMNS = [
    "old_T_bin_index",
    "old_T_bin_left_K",
    "old_T_bin_right_K",
    "old_T_bin_center_K",
    "old_T_bin_label",
]
NEW_COLUMNS = [
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "temperature_bin_version",
    "temperature_bin_width_K",
    "temperature_bin_start_K",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step9A 25 K broad-family revalidation outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validation-rows",
        type=Path,
        default=DEFAULT_OUTPUT / "step9a_25k_validation_rows_with_splits.csv",
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_OUTPUT / "step5c_metrics_by_config.csv")
    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_OUTPUT / "step9a_25k_vs_100k_default_metrics_comparison.csv",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns, low_memory=False)


def clean_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.str.casefold().isin({"", "nan", "none", "null", "na", "n/a"}))


def run_check(command: list[str], label: str, failures: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        failures.append(f"{label} failed with return code {result.returncode}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    required_output_files = [
        "step9a_25k_validation_rows_with_splits.csv",
        "step9a_25k_validation_rows_with_splits.parquet",
        "step9a_25k_target_dropped_rows.csv",
        "step5b_train_reference_curve_bins.csv",
        "step5b_train_reference_curve_bins.parquet",
        "step5b_test_predictions.csv",
        "step5b_test_predictions.parquet",
        "step5b_test_predictions_valid.csv",
        "step5b_test_predictions_valid.parquet",
        "step5b_test_predictions_unavailable.csv",
        "step5b_prediction_coverage_by_config.csv",
        "step5b_prediction_unavailable_summary.csv",
        "step5b_dropped_rows.csv",
        "step5c_metrics_by_config.csv",
        "step5c_metrics_by_config.parquet",
        "step5c_metrics_by_carrier_type.csv",
        "step5c_metrics_by_material_family.csv",
        "step5c_metrics_by_temperature_bin.csv",
        "step5c_metrics_by_eta_bin.csv",
        "step5c_default_comparison.csv",
        "step5c_config_ranking.csv",
        "step5c_largest_abs_error_rows.csv",
        "step5c_dropped_rows.csv",
        "step9a_25k_vs_100k_default_metrics_comparison.csv",
        "step9a_25k_vs_100k_material_family_metrics_comparison.csv",
        "step9a_25k_bin_coverage_summary.csv",
        "step9a_25k_default_metrics_summary.csv",
        "step9a_25k_material_family_metrics_summary.csv",
        "step9a_100k_directory_protection_manifest.csv",
    ]
    for path in [args.input, args.validation_rows, args.metrics, args.comparison, args.report]:
        if not path.exists():
            failures.append(f"missing required path: {path}")
    for filename in required_output_files:
        if not (args.output / filename).exists():
            failures.append(f"missing output: {args.output / filename}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if args.output.resolve() == HUNDRED_K_DIR.resolve():
        failures.append("Step9A output points to the protected 100 K directory")

    input_frame = read_table(args.input)
    validation = read_table(args.validation_rows)
    missing_columns = sorted(set([*OLD_COLUMNS, *NEW_COLUMNS]) - set(validation.columns))
    if missing_columns:
        failures.append(f"validation rows missing columns: {missing_columns}")
    if len(validation) != len(input_frame):
        failures.append(f"validation row count {len(validation)} != input row count {len(input_frame)}")

    if not missing_columns:
        width_values = pd.to_numeric(validation["temperature_bin_width_K"], errors="coerce")
        start_values = pd.to_numeric(validation["temperature_bin_start_K"], errors="coerce")
        if width_values.isna().any() or width_values.nunique() != 1:
            failures.append("temperature_bin_width_K must be one finite value")
        if start_values.isna().any() or start_values.nunique() != 1:
            failures.append("temperature_bin_start_K must be one finite value")
        if not validation["temperature_bin_version"].eq("25K").all():
            failures.append('temperature_bin_version is not uniformly "25K"')
        width = float(width_values.iloc[0])
        start = float(start_values.iloc[0])
        index = pd.to_numeric(validation["T_bin_index"], errors="coerce")
        center = pd.to_numeric(validation["T_bin_center_K"], errors="coerce")
        expected_center = start + (index + 0.5) * width
        finite_center = np.isfinite(center)
        if not np.allclose(center[finite_center], expected_center[finite_center], rtol=0.0, atol=1e-9):
            failures.append("T_bin_center_K is inconsistent with bin index/width/start")
        if np.isclose(width, 25.0) and np.isclose(start, 12.5):
            if not np.allclose(center[finite_center] / 25.0, np.round(center[finite_center] / 25.0), atol=1e-9):
                failures.append("T_bin_center_K is not on the 25 K grid")
        temperature = pd.to_numeric(validation["T_K"], errors="coerce")
        left = pd.to_numeric(validation["T_bin_left_K"], errors="coerce")
        right = pd.to_numeric(validation["T_bin_right_K"], errors="coerce")
        finite_temperature = np.isfinite(temperature)
        inside = (left[finite_temperature] <= temperature[finite_temperature]) & (
            temperature[finite_temperature] < right[finite_temperature]
        )
        if not inside.all():
            failures.append("T_bin_left_K <= T_K < T_bin_right_K is violated")

    if clean_text(validation["material_group_key"]).isna().any():
        failures.append("material_group_key contains missing/blank values")
    for column in ["sample_holdout_split", "paper_holdout_split"]:
        if column not in validation.columns:
            failures.append(f"validation rows missing {column}")
        elif not validation[column].fillna("<NA>").astype(str).equals(
            input_frame[column].fillna("<NA>").astype(str)
        ):
            failures.append(f"{column} was not preserved")

    metrics = pd.read_csv(args.metrics, low_memory=False)
    if metrics["config_id"].nunique() != 32:
        failures.append(f"expected 32 Step5C configs, got {metrics['config_id'].nunique()}")
    if set(metrics["metric_weighting"].dropna()) != {"row_equal", "sample_equal"}:
        failures.append("Step5C metrics do not contain both metric weightings")

    comparison = pd.read_csv(args.comparison, low_memory=False)
    expected_comparison_columns = [
        "config_label",
        "metric_weighting",
        "metric_name",
        "value_100k",
        "value_25k",
        "delta_25k_minus_100k",
        "interpretation_hint",
    ]
    if list(comparison.columns) != expected_comparison_columns:
        failures.append("comparison columns do not match the Step9A specification")
    if set(comparison["metric_weighting"]) != {"row_equal", "sample_equal"}:
        failures.append("comparison does not retain both metric weightings")
    expected_metrics = {
        "n_rows",
        "n_samples",
        "n_papers",
        "coverage_fraction",
        "mae_log10",
        "rmse_log10",
        "median_log10_error",
        "factor_2_accuracy",
        "factor_5_accuracy",
        "factor_10_accuracy",
        "max_abs_log10_error",
        "extreme_ge_10_count",
        "severe_ge_5_count",
    }
    if set(comparison["metric_name"]) != expected_metrics:
        failures.append("comparison metric set does not match the Step9A specification")

    default_summary = pd.read_csv(args.output / "step9a_25k_default_metrics_summary.csv", low_memory=False)
    if len(default_summary) != 8:
        failures.append(f"default metrics summary should have 8 rows, got {len(default_summary)}")
    if default_summary["config_id"].nunique() != 4:
        failures.append("default metrics summary does not contain four default configs")

    protection = pd.read_csv(args.output / "step9a_100k_directory_protection_manifest.csv", low_memory=False)
    if protection.empty or not protection["unchanged"].astype(str).str.casefold().isin({"true", "1"}).all():
        failures.append("100 K protection manifest reports a changed file")
    else:
        current = {
            path.relative_to(HUNDRED_K_DIR).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in HUNDRED_K_DIR.rglob("*")
            if path.is_file()
        }
        for _, row in protection.iterrows():
            path_key = str(row["relative_path"])
            expected = (int(row["size_after"]), int(row["mtime_ns_after"]))
            if current.get(path_key) != expected:
                failures.append(f"protected 100 K file changed after Step9A: {path_key}")
                break

    if args.report.stat().st_size == 0:
        failures.append("Step9A report is empty")
    report_text = args.report.read_text(encoding="utf-8")
    for phrase in [
        "Starrydata2 raw data was not read",
        "existing 100 K outputs were not overwritten",
        "not a new model",
        "No figures were created",
    ]:
        if phrase not in report_text:
            failures.append(f"report missing required statement: {phrase}")

    run_check(
        [
            sys.executable,
            str(EXP_DIR / "check_step5b_predictions.py"),
            "--predictions",
            str(args.output / "step5b_test_predictions.csv"),
            "--valid",
            str(args.output / "step5b_test_predictions_valid.csv"),
            "--coverage",
            str(args.output / "step5b_prediction_coverage_by_config.csv"),
            "--reference",
            str(args.output / "step5b_train_reference_curve_bins.csv"),
            "--dropped",
            str(args.output / "step5b_dropped_rows.csv"),
            "--unavailable",
            str(args.output / "step5b_test_predictions_unavailable.csv"),
            "--default",
            str(args.output / "step5b_test_predictions_default.csv"),
            "--global-default",
            str(args.output / "step5b_test_predictions_global_default.csv"),
            "--require-full-run",
        ],
        "Step5B full check",
        failures,
    )
    run_check(
        [
            sys.executable,
            str(EXP_DIR / "check_step5c_evaluation_metrics.py"),
            "--metrics-config",
            str(args.output / "step5c_metrics_by_config.csv"),
            "--default-comparison",
            str(args.output / "step5c_default_comparison.csv"),
            "--ranking",
            str(args.output / "step5c_config_ranking.csv"),
            "--largest-errors",
            str(args.output / "step5c_largest_abs_error_rows.csv"),
            "--dropped",
            str(args.output / "step5c_dropped_rows.csv"),
        ],
        "Step5C full check",
        failures,
    )

    coverage_summary = pd.read_csv(args.output / "step9a_25k_bin_coverage_summary.csv", low_memory=False)
    summary_map = dict(zip(coverage_summary["item"], coverage_summary["value"]))
    print(f"input rows: {len(input_frame)}")
    print(f"25 K validation rows: {len(validation)}")
    print(f"unique T bins: {summary_map.get('unique_T_bin_count')}")
    print(f"Step5B prediction ok rows: {summary_map.get('step5b_prediction_ok_rows')}")
    print(f"Step5B prediction unavailable rows: {summary_map.get('step5b_prediction_unavailable_rows')}")
    print(f"default coverage: {summary_map.get('default_coverage_fraction')}")
    print(f"default MAE: {summary_map.get('default_mae_log10')}")
    print(f"default RMSE: {summary_map.get('default_rmse_log10')}")
    print(f"default factor2: {summary_map.get('default_factor2')}")
    print(f"default factor10: {summary_map.get('default_factor10')}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step9a 25 K broad_family revalidation checks passed")


if __name__ == "__main__":
    main()
