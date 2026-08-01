import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

CONFIG_KEYS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
]

REQUIRED_METRIC_COLUMNS = [
    *CONFIG_KEYS,
    "metric_weighting",
    "n_rows",
    "n_samples",
    "n_papers",
    "mae_log10",
    "rmse_log10",
    "median_log10_error",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "max_abs_log10_error",
    "is_reliable_eval_group",
    "eval_group_reliability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step5C evaluation metric outputs.")
    parser.add_argument("--metrics-config", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_config.csv")
    parser.add_argument("--default-comparison", type=Path, default=PROCESSED_DIR / "step5c_default_comparison.csv")
    parser.add_argument("--ranking", type=Path, default=PROCESSED_DIR / "step5c_config_ranking.csv")
    parser.add_argument("--largest-errors", type=Path, default=PROCESSED_DIR / "step5c_largest_abs_error_rows.csv")
    parser.add_argument("--dropped", type=Path, default=PROCESSED_DIR / "step5c_dropped_rows.csv")
    parser.add_argument("--min-eval-rows", type=int, default=30)
    parser.add_argument("--min-eval-samples", type=int, default=5)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def expected_reliability(row: pd.Series, min_rows: int, min_samples: int) -> str:
    reliable = int(row["n_rows"]) >= min_rows and int(row["n_samples"]) >= min_samples
    if not reliable:
        return "insufficient"
    if int(row["n_samples"]) >= 30 and int(row["n_papers"]) >= 5:
        return "high"
    if int(row["n_samples"]) >= 10 and int(row["n_papers"]) >= 2:
        return "medium"
    return "low"


def default_configs() -> list[str]:
    return [
        "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
        "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
        "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
        "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    ]


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.metrics_config, args.default_comparison, args.ranking, args.largest_errors, args.dropped]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    metrics = pd.read_csv(args.metrics_config, low_memory=False)
    default = pd.read_csv(args.default_comparison, low_memory=False)
    ranking = pd.read_csv(args.ranking, low_memory=False)
    largest = pd.read_csv(args.largest_errors, low_memory=False)
    dropped = pd.read_csv(args.dropped, low_memory=False)

    require_columns(metrics, REQUIRED_METRIC_COLUMNS, "metrics_by_config", failures)
    require_columns(default, REQUIRED_METRIC_COLUMNS, "default_comparison", failures)
    require_columns(ranking, REQUIRED_METRIC_COLUMNS, "ranking", failures)
    require_columns(largest, ["config_id", "row_id", "abs_log10_sigma_pred_over_exp"], "largest_errors", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if metrics.empty:
        failures.append("metrics_by_config is empty")
    if metrics["config_id"].nunique() != 32:
        failures.append(f"metrics_by_config should contain 32 configs, got {metrics['config_id'].nunique()}")
    if set(metrics["metric_weighting"].dropna()) != {"row_equal", "sample_equal"}:
        failures.append("metrics_by_config does not contain row_equal and sample_equal")
    if not (numeric(metrics, "n_rows") > 0).all():
        failures.append("metrics contain non-positive n_rows")
    for column in ["mae_log10", "rmse_log10", "max_abs_log10_error"]:
        if not (np.isfinite(numeric(metrics, column)).all() and (numeric(metrics, column) >= 0).all()):
            failures.append(f"invalid {column}")
    if not (numeric(metrics, "max_abs_log10_error") + 1e-12 >= numeric(metrics, "mae_log10")).all():
        failures.append("max_abs_log10_error is smaller than mae_log10")
    for column in ["factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"]:
        if not numeric(metrics, column).between(0, 1).all():
            failures.append(f"{column} outside 0..1")
    expected_reliable = (numeric(metrics, "n_rows") >= args.min_eval_rows) & (
        numeric(metrics, "n_samples") >= args.min_eval_samples
    )
    if not (as_bool(metrics["is_reliable_eval_group"]) == expected_reliable).all():
        failures.append("is_reliable_eval_group rule mismatch")
    expected_levels = metrics.apply(lambda row: expected_reliability(row, args.min_eval_rows, args.min_eval_samples), axis=1)
    if not (metrics["eval_group_reliability"] == expected_levels).all():
        failures.append("eval_group_reliability rule mismatch")

    expected_default = {(config, weighting) for config in default_configs() for weighting in ["row_equal", "sample_equal"]}
    found_default = set(zip(default["config_id"], default["metric_weighting"]))
    if not expected_default.issubset(found_default):
        failures.append("default_comparison does not contain all four defaults with both weightings")
    if ranking.empty:
        failures.append("ranking is empty")
    for column in ["rank_by_mae_log10", "rank_by_rmse_log10", "rank_by_factor_2_accuracy", "rank_by_factor_10_accuracy"]:
        if column not in ranking.columns:
            failures.append(f"ranking missing {column}")
    if len(largest) > 1000:
        failures.append("largest error rows exceed 1000")
    if not largest.empty and not numeric(largest, "abs_log10_sigma_pred_over_exp").is_monotonic_decreasing:
        failures.append("largest error rows are not sorted by descending abs error")
    if "coverage_fraction" in metrics.columns and not numeric(metrics, "coverage_fraction").dropna().between(0, 1).all():
        failures.append("coverage_fraction outside 0..1")

    print(f"metrics_by_config rows: {len(metrics)}")
    print(f"configs: {metrics['config_id'].nunique()}")
    print(f"default_comparison rows: {len(default)}")
    print(f"ranking rows: {len(ranking)}")
    print(f"largest error rows: {len(largest)}")
    print(f"dropped rows: {len(dropped)}")
    print(f"metric_weighting counts: {metrics['metric_weighting'].value_counts().to_dict()}")
    if not default.empty:
        preview_cols = ["config_id", "metric_weighting", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy"]
        print(default[preview_cols].to_string(index=False))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step5c evaluation metric checks passed")


if __name__ == "__main__":
    main()
