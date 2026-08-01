import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step6d_broad_family_audit"
DEFAULT_REPORT = EXP_DIR / "reports" / "step6d_broad_family_audit" / "step6d_outlier_robustness_audit_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step6D outlier robustness audit outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--outliers", type=Path, default=DEFAULT_OUTPUT / "step6d_outlier_rows_topN.csv")
    parser.add_argument("--robust-filter", type=Path, default=DEFAULT_OUTPUT / "step6d_robust_metrics_by_filter.csv")
    parser.add_argument("--robust-config", type=Path, default=DEFAULT_OUTPUT / "step6d_robust_metrics_by_config.csv")
    parser.add_argument("--manual-review", type=Path, default=DEFAULT_OUTPUT / "step6d_manual_review_shortlist.csv")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_OUTPUT / "step6d_broad_family_main_result_readiness_summary.csv")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    paths = {
        "outliers": args.outliers,
        "row_summary": args.output / args.outliers.name.replace("outlier_rows_topN", "outlier_summary_by_row_id"),
        "sample_summary": args.output / args.outliers.name.replace("outlier_rows_topN", "outlier_summary_by_sample"),
        "paper_summary": args.output / args.outliers.name.replace("outlier_rows_topN", "outlier_summary_by_paper"),
        "context_rows": args.output / args.outliers.name.replace("outlier_rows_topN", "top_outlier_sample_context_rows"),
        "robust_filter": args.robust_filter,
        "robust_config": args.robust_config,
        "original_vs_broad": args.output / args.robust_filter.name.replace("robust_metrics_by_filter", "original_vs_broad_robust_metrics_comparison"),
        "contribution": args.output / args.outliers.name.replace("outlier_rows_topN", "error_contribution_concentration"),
        "contribution_summary": args.output / args.outliers.name.replace("outlier_rows_topN", "error_contribution_summary"),
        "manual_review": args.manual_review,
        "readiness": args.readiness,
        "report": args.report,
    }
    for label, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty {label}: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    outliers = pd.read_csv(args.outliers, low_memory=False)
    row_summary = pd.read_csv(paths["row_summary"], low_memory=False)
    sample_summary = pd.read_csv(paths["sample_summary"], low_memory=False)
    paper_summary = pd.read_csv(paths["paper_summary"], low_memory=False)
    robust_filter = pd.read_csv(args.robust_filter, low_memory=False)
    robust_config = pd.read_csv(args.robust_config, low_memory=False)
    original_vs_broad = pd.read_csv(paths["original_vs_broad"], low_memory=False)
    contribution = pd.read_csv(paths["contribution"], low_memory=False)
    contribution_summary = pd.read_csv(paths["contribution_summary"], low_memory=False)
    manual = pd.read_csv(args.manual_review, low_memory=False)
    readiness = pd.read_csv(args.readiness, low_memory=False)

    require_columns(
        outliers,
        [
            "config_id",
            "row_id",
            "paper_id",
            "doi",
            "sample_id",
            "sample_key",
            "sigma_S_per_m",
            "sigma_pred_S_per_m",
            "sigma_pred_over_exp",
            "log10_sigma_pred_over_exp",
            "abs_error_decades",
            "sigma0_ref_over_row_sigma0",
            "log10_sigma0_ref_over_row_sigma0",
            "error_direction",
            "error_severity",
            "likely_error_origin_hint",
        ],
        "outliers",
        failures,
    )
    require_columns(row_summary, ["row_id", "max_abs_error_decades", "config_count", "extreme_ge_10_config_count"], "row_summary", failures)
    require_columns(sample_summary, ["validation_sample_group_id", "max_abs_error_decades", "fraction_factor10_or_more"], "sample_summary", failures)
    require_columns(paper_summary, ["validation_paper_group_id", "max_abs_error_decades", "fraction_factor10_or_more"], "paper_summary", failures)
    require_columns(
        robust_filter,
        [
            "default_label",
            "config_id",
            "filter_label",
            "n_rows",
            "mae_log10",
            "rmse_log10",
            "factor_2_accuracy",
            "factor_10_accuracy",
            "retained_row_fraction",
        ],
        "robust_filter",
        failures,
    )
    require_columns(
        robust_config,
        [
            "config_id",
            "filter_label",
            "n_rows",
            "n_samples",
            "mae_log10",
            "factor_2_accuracy",
            "rank_by_mae_log10_within_filter",
        ],
        "robust_config",
        failures,
    )
    require_columns(original_vs_broad, ["default_label", "filter_label", "metric_name", "original_value", "broad_family_value", "delta_broad_minus_original"], "original_vs_broad", failures)
    require_columns(contribution, ["aggregation_level", "group_id", "fraction_of_total_abs_error", "fraction_of_total_squared_error"], "contribution", failures)
    require_columns(contribution_summary, ["item", "value", "comment"], "contribution_summary", failures)
    require_columns(manual, ["review_priority", "review_type", "row_id", "abs_error_decades", "note_for_manual_review"], "manual_review", failures)
    require_columns(readiness, ["criterion", "status", "value", "threshold_or_reason", "comment"], "readiness", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if outliers.empty:
        failures.append("outlier topN table is empty")
    if robust_filter.empty:
        failures.append("robust filter table is empty")
    if robust_config.empty:
        failures.append("robust config table is empty")
    if manual.empty:
        failures.append("manual review shortlist is empty")
    if readiness.empty:
        failures.append("readiness summary is empty")
    if len(manual) > 200:
        failures.append("manual review shortlist has more than 200 rows")
    if set(["no_filter", "exclude_abs_error_ge_5", "exclude_top_1p0_percent_abs_error"]) - set(robust_config["filter_label"]):
        failures.append("robust config table missing required filters")
    default_no_filter = robust_filter[
        robust_filter["default_label"].eq("broad_material_family_default")
        & robust_filter["filter_label"].eq("no_filter")
    ]
    if default_no_filter.empty:
        failures.append("robust filter missing broad material_family default no_filter row")
    readiness_items = set(readiness["criterion"])
    for item in [
        "coverage_is_high",
        "material_family_differs_from_global",
        "mae_improved_vs_original",
        "factor2_improved_vs_original",
        "manual_review_needed",
        "recommended_next_action",
    ]:
        if item not in readiness_items:
            failures.append(f"readiness summary missing criterion: {item}")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print(f"outlier rows: {len(outliers)}")
    print(f"row summary rows: {len(row_summary)}")
    print(f"sample summary rows: {len(sample_summary)}")
    print(f"paper summary rows: {len(paper_summary)}")
    print(f"robust filter rows: {len(robust_filter)}")
    print(f"robust config rows: {len(robust_config)}")
    print(f"original vs broad rows: {len(original_vs_broad)}")
    print(f"manual review rows: {len(manual)}")
    print(f"readiness rows: {len(readiness)}")
    if not default_no_filter.empty:
        row = default_no_filter.iloc[0]
        print(
            "broad default no_filter: "
            f"mae={row['mae_log10']}, rmse={row['rmse_log10']}, "
            f"factor2={row['factor_2_accuracy']}, factor10={row['factor_10_accuracy']}, "
            f"max_abs={row['max_abs_log10_error']}"
        )
    print(readiness[["criterion", "status", "value"]].to_string(index=False))
    print("step6d outlier robustness audit checks passed")


if __name__ == "__main__":
    main()
