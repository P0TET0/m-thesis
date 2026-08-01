import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step7c_reviewed_decisions_applied"
DEFAULT_REPORT_DIR = EXP_DIR / "reports" / "step7c_reviewed_decisions_applied"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step7C reviewed-decision outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--allow-no-human-review", action="store_true")
    return parser.parse_args()


def out_name(base: str, suffix: str, ext: str = "csv") -> str:
    return f"{base}{suffix}.{ext}"


def policy_dir(base: Path, policy: str, suffix: str) -> Path:
    return base / f"{policy}{suffix}" if suffix else base / policy


def require_columns(df: pd.DataFrame, cols: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def check_policy_outputs(policy: str, directory: Path, suffix: str, failures: list[str]) -> dict[str, Path]:
    paths = {
        "rows": directory / out_name("step7b_prediction_rows_with_review_flags", suffix),
        "primary": directory / out_name("step7b_primary_analysis_predictions", suffix),
        "sensitivity": directory / out_name("step7b_sensitivity_analysis_predictions", suffix),
        "excluded_primary": directory / out_name("step7b_excluded_from_primary", suffix),
        "excluded_sensitivity": directory / out_name("step7b_excluded_from_sensitivity", suffix),
        "pending": directory / out_name("step7b_pending_or_unresolved_rows", suffix),
        "metrics": directory / out_name("step7b_metrics_by_review_scenario_config", suffix),
        "default_metrics": directory / out_name("step7b_default_metrics_by_review_scenario", suffix),
        "readiness": directory / out_name("step7b_review_readiness_summary", suffix),
        "unresolved": directory / out_name("step7b_manual_review_unresolved_checklist", suffix),
    }
    for label, path in paths.items():
        if not path.exists():
            failures.append(f"missing {policy} {label}: {path}")
        elif label not in ["excluded_primary", "excluded_sensitivity"] and path.stat().st_size == 0:
            failures.append(f"empty {policy} {label}: {path}")
    return paths


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    suffix = args.output_suffix

    top_paths = {
        "validation": args.output / out_name("step7c_reviewed_decision_validation", suffix),
        "validation_summary": args.output / out_name("step7c_reviewed_decision_validation_summary", suffix),
        "extreme_summary": args.output / out_name("step7c_extreme_review_completion_summary", suffix),
        "policy_comparison": args.output / out_name("step7c_policy_comparison_metrics", suffix),
        "baseline_comparison": args.output / out_name("step7c_reviewed_vs_pending_baseline_comparison", suffix),
        "manifest": args.output / out_name("step7c_final_candidate_dataset_manifest", suffix),
        "unresolved": args.output / out_name("step7c_unresolved_after_review_checklist", suffix),
        "readiness": args.output / out_name("step7c_final_readiness_summary", suffix),
        "report": args.report_dir / out_name("step7c_reviewed_decisions_report", suffix, "md"),
    }
    for label, path in top_paths.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
        elif label != "baseline_comparison" and path.stat().st_size == 0:
            failures.append(f"empty {label}: {path}")

    keep_dir = policy_dir(args.output, "keep_pending", suffix)
    exclude_dir = policy_dir(args.output, "exclude_pending_primary", suffix)
    keep_paths = check_policy_outputs("keep_pending", keep_dir, suffix, failures)
    exclude_paths = check_policy_outputs("exclude_pending_primary", exclude_dir, suffix, failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    validation = read_csv(top_paths["validation"])
    summary = read_csv(top_paths["validation_summary"])
    extreme = read_csv(top_paths["extreme_summary"])
    policy_comparison = read_csv(top_paths["policy_comparison"])
    manifest = read_csv(top_paths["manifest"])
    readiness = read_csv(top_paths["readiness"])
    keep_rows = read_csv(keep_paths["rows"])
    keep_primary = read_csv(keep_paths["primary"])
    keep_sensitivity = read_csv(keep_paths["sensitivity"])
    exclude_rows = read_csv(exclude_paths["rows"])
    exclude_primary = read_csv(exclude_paths["primary"])
    exclude_sensitivity = read_csv(exclude_paths["sensitivity"])
    keep_default = read_csv(keep_paths["default_metrics"])
    exclude_default = read_csv(exclude_paths["default_metrics"])

    require_columns(validation, ["review_case_id", "review_status", "decision_is_human_reviewed", "decision_validation_status"], "validation", failures)
    require_columns(summary, ["item", "value", "comment"], "validation_summary", failures)
    require_columns(extreme, ["extreme_case_count", "extreme_human_reviewed_count", "extreme_pending_count"], "extreme_summary", failures)
    require_columns(policy_comparison, ["config_label", "metric_weighting", "metric_name", "keep_pending_value", "exclude_pending_primary_value"], "policy_comparison", failures)
    require_columns(manifest, ["dataset_role", "dataset_path", "policy", "description", "recommended_use"], "manifest", failures)
    require_columns(readiness, ["criterion", "status", "value", "threshold_or_reason", "comment"], "readiness", failures)
    require_columns(keep_rows, ["row_id", "config_id", "review_is_pending", "keep_in_primary_analysis", "sigma_pred_S_per_m"], "keep rows", failures)
    require_columns(exclude_rows, ["row_id", "config_id", "review_is_pending", "keep_in_primary_analysis", "sigma_pred_S_per_m"], "exclude rows", failures)
    require_columns(keep_default, ["review_scenario", "config_label", "metric_weighting", "mae_log10", "factor_10_accuracy"], "keep default metrics", failures)
    require_columns(exclude_default, ["review_scenario", "config_label", "metric_weighting", "mae_log10", "factor_10_accuracy"], "exclude default metrics", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    summary_values = {str(row["item"]): row["value"] for _, row in summary.iterrows()}
    invalid = int(summary_values.get("invalid_decisions", 0))
    human = int(summary_values.get("human_reviewed_decisions", 0))
    if invalid != 0:
        failures.append(f"invalid decisions present: {invalid}")
    if human == 0 and not args.allow_no_human_review:
        failures.append("human reviewed decisions are zero")
    if keep_rows.empty:
        failures.append("keep_pending rows are empty")
    if exclude_rows.empty:
        failures.append("exclude_pending_primary rows are empty")
    if len(keep_primary) > len(keep_rows):
        failures.append("keep_pending primary has more rows than source rows")
    if len(exclude_primary) > len(exclude_rows):
        failures.append("exclude_pending_primary primary has more rows than source rows")
    if len(keep_sensitivity) > len(keep_rows):
        failures.append("keep_pending sensitivity has more rows than source rows")
    if len(exclude_sensitivity) > len(exclude_rows):
        failures.append("exclude_pending_primary sensitivity has more rows than source rows")
    if {"keep_pending", "exclude_pending_primary"} - set(manifest["policy"].astype(str)):
        failures.append("manifest missing one or more Step7C policies")
    if len(policy_comparison) < 40:
        failures.append("policy comparison has too few rows")
    if "ready_for_step8" not in set(readiness["criterion"].astype(str)):
        failures.append("readiness summary missing ready_for_step8")
    if top_paths["report"].stat().st_size == 0:
        failures.append("report is empty")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print(f"validated decisions: {len(validation)}")
    print(f"human reviewed decisions: {human}")
    print(f"invalid decisions: {invalid}")
    print(f"keep_pending rows: {len(keep_rows)}")
    print(f"keep_pending primary rows: {len(keep_primary)}")
    print(f"keep_pending sensitivity rows: {len(keep_sensitivity)}")
    print(f"exclude_pending_primary rows: {len(exclude_rows)}")
    print(f"exclude_pending_primary primary rows: {len(exclude_primary)}")
    print(f"exclude_pending_primary sensitivity rows: {len(exclude_sensitivity)}")
    print(f"policy comparison rows: {len(policy_comparison)}")
    print(readiness[["criterion", "status", "value"]].to_string(index=False))
    print("step7c reviewed-decision checks passed")


if __name__ == "__main__":
    main()
