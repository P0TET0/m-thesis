import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step7b_review_applied"
DEFAULT_REPORT = EXP_DIR / "reports" / "step7b_review_applied" / "step7b_review_application_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step7B review-applied outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=Path, default=DEFAULT_OUTPUT / "step7b_prediction_rows_with_review_flags.csv")
    parser.add_argument("--primary", type=Path, default=DEFAULT_OUTPUT / "step7b_primary_analysis_predictions.csv")
    parser.add_argument("--sensitivity", type=Path, default=DEFAULT_OUTPUT / "step7b_sensitivity_analysis_predictions.csv")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_OUTPUT / "step7b_metrics_by_review_scenario_config.csv")
    parser.add_argument("--default-metrics", type=Path, default=DEFAULT_OUTPUT / "step7b_default_metrics_by_review_scenario.csv")
    parser.add_argument("--readiness", type=Path, default=DEFAULT_OUTPUT / "step7b_review_readiness_summary.csv")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def sibling(path: Path, old: str, new: str) -> Path:
    return path.with_name(path.name.replace(old, new))


def require_columns(df: pd.DataFrame, cols: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    paths = {
        "normalized": sibling(args.rows, "prediction_rows_with_review_flags", "review_decisions_normalized"),
        "rows": args.rows,
        "primary": args.primary,
        "sensitivity": args.sensitivity,
        "excluded_primary": sibling(args.rows, "prediction_rows_with_review_flags", "excluded_from_primary"),
        "excluded_sensitivity": sibling(args.rows, "prediction_rows_with_review_flags", "excluded_from_sensitivity"),
        "pending": sibling(args.rows, "prediction_rows_with_review_flags", "pending_or_unresolved_rows"),
        "conflicts": sibling(args.rows, "prediction_rows_with_review_flags", "review_conflicts"),
        "summary": sibling(args.rows, "prediction_rows_with_review_flags", "review_application_summary"),
        "metrics": args.metrics,
        "default_metrics": args.default_metrics,
        "effect": sibling(args.rows, "prediction_rows_with_review_flags", "review_effect_summary"),
        "readiness": args.readiness,
        "unresolved": sibling(args.rows, "prediction_rows_with_review_flags", "manual_review_unresolved_checklist"),
        "report": args.report,
    }
    for label, path in paths.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
        elif label not in ["conflicts", "excluded_primary", "excluded_sensitivity"] and path.stat().st_size == 0:
            failures.append(f"empty {label}: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    normalized = pd.read_csv(paths["normalized"], low_memory=False)
    rows = pd.read_csv(args.rows, low_memory=False)
    primary = pd.read_csv(args.primary, low_memory=False)
    sensitivity = pd.read_csv(args.sensitivity, low_memory=False)
    excluded_primary = pd.read_csv(paths["excluded_primary"], low_memory=False)
    excluded_sensitivity = pd.read_csv(paths["excluded_sensitivity"], low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)
    default_metrics = pd.read_csv(args.default_metrics, low_memory=False)
    effect = pd.read_csv(paths["effect"], low_memory=False)
    readiness = pd.read_csv(args.readiness, low_memory=False)
    summary = pd.read_csv(paths["summary"], low_memory=False)

    require_columns(normalized, ["review_case_id", "review_status", "apply_to_scope", "decision_is_pending", "decision_validity_status"], "normalized", failures)
    require_columns(
        rows,
        [
            "row_id",
            "config_id",
            "review_case_ids_applied",
            "review_status_applied",
            "review_is_pending",
            "review_has_conflict",
            "keep_in_primary_analysis",
            "keep_in_sensitivity_analysis",
            "sigma_pred_S_per_m",
        ],
        "rows",
        failures,
    )
    require_columns(metrics, ["review_scenario", "config_id", "metric_weighting", "n_rows", "mae_log10", "factor_2_accuracy", "factor_10_accuracy"], "metrics", failures)
    require_columns(default_metrics, ["review_scenario", "config_id", "config_label", "metric_weighting", "mae_log10"], "default_metrics", failures)
    require_columns(effect, ["config_label", "metric_weighting", "metric_name", "baseline_value", "primary_review_applied_value", "delta_primary_minus_baseline"], "effect", failures)
    require_columns(readiness, ["criterion", "status", "value", "threshold_or_reason", "comment"], "readiness", failures)
    require_columns(summary, ["item", "value", "comment"], "summary", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if rows.empty:
        failures.append("review-applied rows are empty")
    if rows["row_id"].isna().any():
        failures.append("row_id has missing values")
    if rows["config_id"].isna().any():
        failures.append("config_id has missing values")
    if rows["keep_in_primary_analysis"].isna().any():
        failures.append("keep_in_primary_analysis has missing values")
    if rows["keep_in_sensitivity_analysis"].isna().any():
        failures.append("keep_in_sensitivity_analysis has missing values")
    if len(primary) and not primary["keep_in_primary_analysis"].astype(bool).all():
        failures.append("primary file contains rows not kept in primary")
    if len(sensitivity) and not sensitivity["keep_in_sensitivity_analysis"].astype(bool).all():
        failures.append("sensitivity file contains rows not kept in sensitivity")
    if len(excluded_primary) and excluded_primary["keep_in_primary_analysis"].astype(bool).any():
        failures.append("excluded_from_primary contains kept rows")
    if len(excluded_sensitivity) and excluded_sensitivity["keep_in_sensitivity_analysis"].astype(bool).any():
        failures.append("excluded_from_sensitivity contains kept rows")
    required_scenarios = {"all_predictions_no_review_filter", "primary_review_applied", "sensitivity_review_applied"}
    if required_scenarios - set(metrics["review_scenario"]):
        failures.append("metrics missing required review scenarios")
    if len(default_metrics) < 8:
        failures.append("default metrics should contain default configs and weightings")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print(f"review-applied rows: {len(rows)}")
    print(f"normalized decisions: {len(normalized)}")
    print(f"primary rows: {len(primary)}")
    print(f"sensitivity rows: {len(sensitivity)}")
    print(f"excluded primary rows: {len(excluded_primary)}")
    print(f"excluded sensitivity rows: {len(excluded_sensitivity)}")
    print(f"metrics rows: {len(metrics)}")
    print(f"default metrics rows: {len(default_metrics)}")
    print(f"review effect rows: {len(effect)}")
    print(f"readiness rows: {len(readiness)}")
    interesting = default_metrics[
        default_metrics["config_label"].eq("broad_material_family_default")
        & default_metrics["metric_weighting"].eq("row_equal")
    ][["review_scenario", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "extreme_ge_10_count", "n_rows"]]
    print(interesting.to_string(index=False))
    print(readiness[["criterion", "status", "value"]].to_string(index=False))
    print("step7b review-applied checks passed")


if __name__ == "__main__":
    main()
