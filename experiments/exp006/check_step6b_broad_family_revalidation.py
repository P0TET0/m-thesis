import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"
DEFAULT_INPUT = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step6B broad_family revalidation outputs.")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "step6b_broad_family")
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step6b_broad_family" / "step6b_broad_family_revalidation_report.md")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    step5b_files = [
        "step5b_test_predictions.csv",
        "step5b_test_predictions_valid.csv",
        "step5b_prediction_coverage_by_config.csv",
        "step5b_train_reference_curve_bins.csv",
        "step5b_dropped_rows.csv",
    ]
    step5c_files = [
        "step5c_metrics_by_config.csv",
        "step5c_default_comparison.csv",
        "step5c_config_ranking.csv",
        "step5c_largest_abs_error_rows.csv",
        "step5c_dropped_rows.csv",
    ]
    extra_files = [
        "step6b_material_family_vs_global_prediction_diff_summary.csv",
        "step6b_material_family_vs_global_prediction_diff_examples.csv",
        "step6b_reference_group_diagnostics.csv",
        "step6b_broad_family_vs_original_default_metrics_comparison.csv",
        "step6b_broad_family_default_metrics_summary.csv",
        "step6b_revalidation_summary.csv",
    ]
    for name in [*step5b_files, *step5c_files, *extra_files]:
        if not (args.output / name).exists():
            failures.append(f"missing output: {args.output / name}")
    if not args.report.exists():
        failures.append(f"missing report: {args.report}")
    if not args.input.exists():
        failures.append(f"missing input: {args.input}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    input_df = read_table(args.input)
    valid = pd.read_csv(args.output / "step5b_test_predictions_valid.csv", usecols=["config_id"], low_memory=False)
    metrics = pd.read_csv(args.output / "step5c_metrics_by_config.csv", low_memory=False)
    default = pd.read_csv(args.output / "step5c_default_comparison.csv", low_memory=False)
    diff = pd.read_csv(args.output / "step6b_material_family_vs_global_prediction_diff_summary.csv", low_memory=False)
    default_summary = pd.read_csv(args.output / "step6b_broad_family_default_metrics_summary.csv", low_memory=False)
    summary = pd.read_csv(args.output / "step6b_revalidation_summary.csv", low_memory=False)

    if input_df["material_group_key"].nunique() <= 1:
        failures.append("input material_group_key unique count is not > 1")
    if input_df["material_group_key"].isna().any():
        failures.append("input material_group_key contains missing values")
    if len(valid) == 0:
        failures.append("Step5B valid predictions are empty")
    if valid["config_id"].nunique() != 32:
        failures.append(f"Step5B config count should be 32, got {valid['config_id'].nunique()}")
    if len(metrics) == 0:
        failures.append("Step5C metrics_by_config is empty")
    if metrics["config_id"].nunique() != 32:
        failures.append(f"Step5C config count should be 32, got {metrics['config_id'].nunique()}")
    if len(default) != 8:
        failures.append(f"Step5C default_comparison should have 8 rows, got {len(default)}")
    if diff.empty:
        failures.append("prediction diff summary is empty")
    if default_summary.empty:
        failures.append("default metrics summary is empty")
    if summary.empty:
        failures.append("Step6B summary is empty")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    print(f"input rows: {len(input_df)}")
    print(f"input material groups: {input_df['material_group_key'].nunique()}")
    print(f"Step5B valid rows: {len(valid)}")
    print(f"Step5B configs: {valid['config_id'].nunique()}")
    print(f"Step5C metrics rows: {len(metrics)}")
    print(f"Step5C configs: {metrics['config_id'].nunique()}")
    print(f"default comparison rows: {len(default)}")
    print(diff.to_string(index=False))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step6b broad_family revalidation checks passed")


if __name__ == "__main__":
    main()
