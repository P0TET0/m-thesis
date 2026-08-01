import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_ROOT = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "step9a_25k_bin_broad_family"
DEFAULT_REPORT_DIR = REPORT_ROOT / "step9a_25k_bin_broad_family"
HUNDRED_K_DIR = PROCESSED_DIR / "step6b_broad_family"
HUNDRED_K_DEFAULT = HUNDRED_K_DIR / "step5c_default_comparison.csv"
HUNDRED_K_COVERAGE = HUNDRED_K_DIR / "step5b_prediction_coverage_by_config.csv"

T_BIN_COLUMNS = [
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
]
OLD_T_BIN_COLUMNS = [f"old_{column}" for column in T_BIN_COLUMNS]
REQUIRED_INPUT_COLUMNS = [
    "T_K",
    *T_BIN_COLUMNS,
    "material_group_key",
    "sample_holdout_split",
    "paper_holdout_split",
    "sample_cv_fold",
    "paper_cv_fold",
    "is_valid_sigma0",
    "sigma_S_per_m",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "carrier_type",
]

DEFAULT_CONFIGS = {
    "material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}
PRIMARY_CONFIG = DEFAULT_CONFIGS["material_family_default"]

COMPARISON_METRICS = [
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
]

DEFAULT_SUMMARY_COLUMNS = [
    "default_label",
    "config_id",
    "metric_weighting",
    "n_rows",
    "coverage_fraction",
    "mae_log10",
    "rmse_log10",
    "median_log10_error",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "max_abs_log10_error",
    "extreme_ge_10_count",
]

MATERIAL_SUMMARY_COLUMNS = [
    "material_group_key",
    "material_family_raw",
    "n_rows",
    "n_samples",
    "n_papers",
    "mae_log10",
    "rmse_log10",
    "factor_2_accuracy",
    "factor_10_accuracy",
    "is_reliable_eval_group",
    "eval_group_reliability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step9A broad-family revalidation with rebuilt 25 K temperature bins.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--bin-width-k", type=float, default=25.0)
    parser.add_argument("--bin-start-k", type=float, default=12.5)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--min-eval-rows", type=int, default=30)
    parser.add_argument("--min-eval-samples", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-rows-per-config", type=int, default=200)
    parser.add_argument("--output-suffix", default="_test")
    parser.add_argument("--skip-small-test", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step9a] {message}", flush=True)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def resolve_input(explicit: Path | None) -> Path:
    allowed = [DEFAULT_INPUT_PARQUET.resolve(), DEFAULT_INPUT_CSV.resolve()]
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        if explicit.resolve() not in allowed:
            raise ValueError("Step9A input must be the Step6A broad_family parquet or CSV file")
        return explicit
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step6A broad_family validation rows were not found")


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.bin_width_k) or args.bin_width_k <= 0:
        raise ValueError("--bin-width-k must be finite and positive")
    if not np.isfinite(args.bin_start_k):
        raise ValueError("--bin-start-k must be finite")
    for name in [
        "min_rows_per_bin",
        "min_samples_per_bin",
        "min_papers_per_bin",
        "min_eval_rows",
        "min_eval_samples",
        "max_rows",
        "max_rows_per_config",
    ]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.output_suffix or args.output_suffix == "":
        raise ValueError("--output-suffix must be non-empty so small-test files cannot overwrite full outputs")
    output_resolved = args.output.resolve()
    if output_resolved == HUNDRED_K_DIR.resolve():
        raise ValueError("Step9A output must not be the existing 100 K directory")
    if args.output.name != "step9a_25k_bin_broad_family":
        raise ValueError("--output must be the dedicated step9a_25k_bin_broad_family directory")


def require_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Step6A broad_family input is missing required columns: {missing}")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def clean_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.str.casefold().isin({"", "nan", "none", "null", "na", "n/a"}))


def format_bin_edge(value: float) -> str:
    if np.isclose(value, round(value), rtol=0.0, atol=1e-10):
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def assign_temperature_bins(df: pd.DataFrame, width: float, start: float) -> pd.DataFrame:
    out = df.copy()
    log("preserving old 100K T_bin columns...")
    for old_column, source_column in zip(OLD_T_BIN_COLUMNS, T_BIN_COLUMNS):
        out[old_column] = out[source_column]

    log("assigning 25K T_bin columns...")
    temperature = pd.to_numeric(out["T_K"], errors="coerce")
    finite = np.isfinite(temperature)
    index_values = np.full(len(out), np.nan)
    index_values[finite] = np.floor((temperature[finite] - start) / width)
    out["T_bin_index"] = pd.array(index_values, dtype="Int64")
    index_float = pd.to_numeric(out["T_bin_index"], errors="coerce")
    out["T_bin_left_K"] = start + index_float * width
    out["T_bin_right_K"] = out["T_bin_left_K"] + width
    out["T_bin_center_K"] = out["T_bin_left_K"] + width / 2.0
    labels = pd.Series(pd.NA, index=out.index, dtype="string")
    labels.loc[finite] = [
        f"{format_bin_edge(left)}_{format_bin_edge(right)}K"
        for left, right in zip(out.loc[finite, "T_bin_left_K"], out.loc[finite, "T_bin_right_K"])
    ]
    out["T_bin_label"] = labels

    # Keep the pre-existing numeric convenience columns consistent with the rebuilt bins.
    for column in ["T_bin_left_K", "T_bin_right_K", "T_bin_center_K"]:
        numeric_column = f"{column}_num"
        if numeric_column in out.columns:
            out[numeric_column] = out[column]

    out["temperature_bin_version"] = "25K"
    out["temperature_bin_width_K"] = width
    out["temperature_bin_start_K"] = start
    return out


def target_usable_mask(df: pd.DataFrame) -> pd.Series:
    mask = as_bool(df["is_valid_sigma0"])
    for column in ["sigma_S_per_m", "F0_eta", "sigma0_S_per_m", "T_K"]:
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= np.isfinite(values) & values.gt(0)
    log_sigma0 = pd.to_numeric(df["log10_sigma0_S_per_m"], errors="coerce")
    mask &= np.isfinite(log_sigma0)
    mask &= df["carrier_type"].astype(str).isin({"p", "n"})
    mask &= clean_text(df["material_group_key"]).notna()
    return mask


def build_target_dropped_rows(df: pd.DataFrame) -> pd.DataFrame:
    valid = target_usable_mask(df)
    dropped = df.loc[~valid].copy()
    if dropped.empty:
        return pd.DataFrame(columns=["row_id", "reject_reason"])

    def reason(row: pd.Series) -> str:
        if not bool(as_bool(pd.Series([row["is_valid_sigma0"]])).iloc[0]):
            return "is_valid_sigma0_not_true"
        for column, label in [
            ("sigma_S_per_m", "invalid_sigma"),
            ("F0_eta", "invalid_F0_eta"),
            ("sigma0_S_per_m", "invalid_sigma0"),
            ("T_K", "invalid_T_K"),
        ]:
            value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if not np.isfinite(value) or value <= 0:
                return label
        value = pd.to_numeric(pd.Series([row["log10_sigma0_S_per_m"]]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            return "invalid_log10_sigma0"
        if str(row["carrier_type"]) not in {"p", "n"}:
            return "invalid_carrier_type"
        if pd.isna(clean_text(pd.Series([row["material_group_key"]])).iloc[0]):
            return "missing_material_group_key"
        return "unknown"

    dropped.insert(0, "reject_reason", dropped.apply(reason, axis=1))
    preferred = [
        "row_id",
        "reject_reason",
        "T_K",
        "carrier_type",
        "material_group_key",
        "sigma_S_per_m",
        "F0_eta",
        "sigma0_S_per_m",
        "log10_sigma0_S_per_m",
        "is_valid_sigma0",
    ]
    return dropped[[column for column in preferred if column in dropped.columns]]


def write_validation_rows(df: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "step9a_25k_validation_rows_with_splits.csv"
    parquet_path = output / "step9a_25k_validation_rows_with_splits.parquet"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def run_command(command: list[str]) -> None:
    log("running: " + subprocess.list2cmdline(command))
    subprocess.run(command, check=True)


def step5b_build_command(
    input_path: Path,
    output: Path,
    report: Path,
    args: argparse.Namespace,
    suffix: str = "",
    max_rows: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(EXP_DIR / "build_step5b_assign_predictions.py"),
        "--input",
        str(input_path),
        "--output",
        str(output),
        "--report",
        str(report),
        "--min-rows-per-bin",
        str(args.min_rows_per_bin),
        "--min-samples-per-bin",
        str(args.min_samples_per_bin),
        "--min-papers-per-bin",
        str(args.min_papers_per_bin),
    ]
    if max_rows is not None:
        command.extend(["--max-rows", str(max_rows), "--output-suffix", suffix])
    return command


def step5b_check_command(output: Path, suffix: str = "", require_full: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(EXP_DIR / "check_step5b_predictions.py"),
        "--predictions",
        str(output / f"step5b_test_predictions{suffix}.csv"),
        "--valid",
        str(output / f"step5b_test_predictions_valid{suffix}.csv"),
        "--coverage",
        str(output / f"step5b_prediction_coverage_by_config{suffix}.csv"),
        "--reference",
        str(output / f"step5b_train_reference_curve_bins{suffix}.csv"),
        "--dropped",
        str(output / f"step5b_dropped_rows{suffix}.csv"),
        "--unavailable",
        str(output / f"step5b_test_predictions_unavailable{suffix}.csv"),
        "--default",
        str(output / f"step5b_test_predictions_default{suffix}.csv"),
        "--global-default",
        str(output / f"step5b_test_predictions_global_default{suffix}.csv"),
    ]
    if require_full:
        command.append("--require-full-run")
    return command


def step5c_build_command(
    input_path: Path,
    coverage: Path,
    unavailable: Path,
    output: Path,
    report: Path,
    args: argparse.Namespace,
    suffix: str = "",
    max_rows_per_config: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(EXP_DIR / "build_step5c_evaluation_metrics.py"),
        "--input",
        str(input_path),
        "--coverage",
        str(coverage),
        "--unavailable",
        str(unavailable),
        "--output",
        str(output),
        "--report",
        str(report),
        "--min-eval-rows",
        str(args.min_eval_rows),
        "--min-eval-samples",
        str(args.min_eval_samples),
    ]
    if max_rows_per_config is not None:
        command.extend(["--max-rows-per-config", str(max_rows_per_config), "--output-suffix", suffix])
    return command


def step5c_check_command(output: Path, suffix: str = "") -> list[str]:
    return [
        sys.executable,
        str(EXP_DIR / "check_step5c_evaluation_metrics.py"),
        "--metrics-config",
        str(output / f"step5c_metrics_by_config{suffix}.csv"),
        "--default-comparison",
        str(output / f"step5c_default_comparison{suffix}.csv"),
        "--ranking",
        str(output / f"step5c_config_ranking{suffix}.csv"),
        "--largest-errors",
        str(output / f"step5c_largest_abs_error_rows{suffix}.csv"),
        "--dropped",
        str(output / f"step5c_dropped_rows{suffix}.csv"),
    ]


def directory_manifest(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return pd.DataFrame(columns=["relative_path", "size", "mtime_ns"])
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return pd.DataFrame(rows)


def build_protection_manifest(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    merged["unchanged"] = (
        merged["_merge"].eq("both")
        & merged["size_before"].eq(merged["size_after"])
        & merged["mtime_ns_before"].eq(merged["mtime_ns_after"])
    )
    return merged


def default_label(config_id: str) -> str:
    for label, value in DEFAULT_CONFIGS.items():
        if value == config_id:
            return label
    return config_id


def extreme_counts(default_predictions: Path) -> dict[str, dict[str, float]]:
    columns = ["prediction_status", "validation_sample_group_id", "log10_sigma_pred_over_exp"]
    frame = read_table(default_predictions, columns=columns)
    frame = frame[frame["prediction_status"].eq("ok")].copy()
    frame["abs_error"] = pd.to_numeric(frame["log10_sigma_pred_over_exp"], errors="coerce").abs()
    frame = frame[np.isfinite(frame["abs_error"])].copy()
    row_counts = {
        "extreme_ge_10_count": float((frame["abs_error"] >= 10).sum()),
        "severe_ge_5_count": float((frame["abs_error"] >= 5).sum()),
    }
    per_sample = (
        frame.assign(
            extreme=frame["abs_error"].ge(10).astype(float),
            severe=frame["abs_error"].ge(5).astype(float),
        )
        .groupby("validation_sample_group_id", dropna=False)[["extreme", "severe"]]
        .sum()
    )
    sample_counts = {
        "extreme_ge_10_count": float(per_sample["extreme"].mean()) if len(per_sample) else np.nan,
        "severe_ge_5_count": float(per_sample["severe"].mean()) if len(per_sample) else np.nan,
    }
    return {"row_equal": row_counts, "sample_equal": sample_counts}


def add_extreme_columns(metrics: pd.DataFrame, default_predictions: Path) -> pd.DataFrame:
    out = metrics.copy()
    counts = extreme_counts(default_predictions)
    out["extreme_ge_10_count"] = out["metric_weighting"].map(
        lambda weighting: counts.get(str(weighting), {}).get("extreme_ge_10_count", np.nan)
    )
    out["severe_ge_5_count"] = out["metric_weighting"].map(
        lambda weighting: counts.get(str(weighting), {}).get("severe_ge_5_count", np.nan)
    )
    return out


def build_default_summary(output: Path) -> pd.DataFrame:
    default = pd.read_csv(output / "step5c_default_comparison.csv", low_memory=False)
    default = add_extreme_columns(default, output / "step5b_test_predictions_default.parquet")
    default["default_label"] = default["config_id"].map(default_label)
    subset = default[default["config_id"].isin(DEFAULT_CONFIGS.values())].copy()
    return subset[DEFAULT_SUMMARY_COLUMNS].sort_values(["default_label", "metric_weighting"]).reset_index(drop=True)


def build_default_comparison(output: Path) -> pd.DataFrame:
    metrics_100 = pd.read_csv(HUNDRED_K_DEFAULT, low_memory=False)
    coverage_100 = pd.read_csv(HUNDRED_K_COVERAGE, low_memory=False)
    metrics_25 = pd.read_csv(output / "step5c_default_comparison.csv", low_memory=False)
    coverage_25 = pd.read_csv(output / "step5b_prediction_coverage_by_config.csv", low_memory=False)

    metrics_100 = add_extreme_columns(metrics_100, HUNDRED_K_DIR / "step5b_test_predictions_default.parquet")
    metrics_25 = add_extreme_columns(metrics_25, output / "step5b_test_predictions_default.parquet")
    coverage_100_value = coverage_100.loc[coverage_100["config_id"].eq(PRIMARY_CONFIG), "coverage_fraction"]
    coverage_25_value = coverage_25.loc[coverage_25["config_id"].eq(PRIMARY_CONFIG), "coverage_fraction"]

    rows: list[dict[str, Any]] = []
    for weighting in ["row_equal", "sample_equal"]:
        old_rows = metrics_100[
            metrics_100["config_id"].eq(PRIMARY_CONFIG) & metrics_100["metric_weighting"].eq(weighting)
        ]
        new_rows = metrics_25[
            metrics_25["config_id"].eq(PRIMARY_CONFIG) & metrics_25["metric_weighting"].eq(weighting)
        ]
        if len(old_rows) != 1 or len(new_rows) != 1:
            raise ValueError(f"Default metric row is not unique for weighting={weighting}")
        old_row = old_rows.iloc[0]
        new_row = new_rows.iloc[0]
        for metric in COMPARISON_METRICS:
            if metric == "coverage_fraction":
                old_value = float(coverage_100_value.iloc[0])
                new_value = float(coverage_25_value.iloc[0])
            else:
                old_value = float(pd.to_numeric(pd.Series([old_row[metric]]), errors="coerce").iloc[0])
                new_value = float(pd.to_numeric(pd.Series([new_row[metric]]), errors="coerce").iloc[0])
            if metric in {
                "mae_log10",
                "rmse_log10",
                "max_abs_log10_error",
                "extreme_ge_10_count",
                "severe_ge_5_count",
            }:
                hint = "lower_is_better"
            elif metric in {"coverage_fraction", "factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"}:
                hint = "higher_is_better"
            else:
                hint = "count_or_context"
            rows.append(
                {
                    "config_label": PRIMARY_CONFIG,
                    "metric_weighting": weighting,
                    "metric_name": metric,
                    "value_100k": old_value,
                    "value_25k": new_value,
                    "delta_25k_minus_100k": new_value - old_value,
                    "interpretation_hint": hint,
                }
            )
    return pd.DataFrame(rows)


def build_material_summary(output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(output / "step5c_metrics_by_material_family.csv", low_memory=False)
    subset = metrics[
        metrics["config_id"].eq(PRIMARY_CONFIG) & metrics["metric_weighting"].eq("row_equal")
    ].copy()
    return subset[MATERIAL_SUMMARY_COLUMNS].sort_values(
        ["is_reliable_eval_group", "n_rows"], ascending=[False, False]
    ).reset_index(drop=True)


def build_material_comparison(output: Path) -> pd.DataFrame:
    columns = [
        "material_group_key",
        "material_family_raw",
        "n_rows",
        "n_samples",
        "mae_log10",
        "rmse_log10",
        "factor_2_accuracy",
        "factor_10_accuracy",
        "is_reliable_eval_group",
    ]
    old = pd.read_csv(HUNDRED_K_DIR / "step5c_metrics_by_material_family.csv", low_memory=False)
    new = pd.read_csv(output / "step5c_metrics_by_material_family.csv", low_memory=False)
    old = old[old["config_id"].eq(PRIMARY_CONFIG) & old["metric_weighting"].eq("row_equal")][columns].copy()
    new = new[new["config_id"].eq(PRIMARY_CONFIG) & new["metric_weighting"].eq("row_equal")][columns].copy()
    for frame in [old, new]:
        frame["material_group_key"] = frame["material_group_key"].fillna("").astype(str)
        frame["material_family_raw"] = frame["material_family_raw"].fillna("").astype(str)
    merged = old.merge(
        new,
        on=["material_group_key", "material_family_raw"],
        how="outer",
        suffixes=("_100k", "_25k"),
        indicator=True,
    )
    for metric in ["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy"]:
        merged[f"delta_{metric}_25k_minus_100k"] = merged[f"{metric}_25k"] - merged[f"{metric}_100k"]
    return merged.sort_values(
        "delta_mae_log10_25k_minus_100k", ascending=True, na_position="last"
    ).reset_index(drop=True)


def summary_value(default_summary: pd.DataFrame, metric: str) -> float:
    values = default_summary[
        default_summary["default_label"].eq("material_family_default")
        & default_summary["metric_weighting"].eq("row_equal")
    ][metric]
    return float(values.iloc[0]) if len(values) else np.nan


def coverage_value(coverage: pd.DataFrame, config_id: str = PRIMARY_CONFIG) -> float:
    values = coverage.loc[coverage["config_id"].eq(config_id), "coverage_fraction"]
    return float(values.iloc[0]) if len(values) else np.nan


def add_summary_item(rows: list[dict[str, Any]], item: str, value: Any, comment: str) -> None:
    rows.append({"item": item, "value": value, "comment": comment})


def build_bin_coverage_summary(
    validation: pd.DataFrame,
    valid_mask: pd.Series,
    output: Path,
    args: argparse.Namespace,
    default_summary: pd.DataFrame,
) -> pd.DataFrame:
    coverage = pd.read_csv(output / "step5b_prediction_coverage_by_config.csv", low_memory=False)
    predictions = pd.read_csv(
        output / "step5b_test_predictions.csv",
        usecols=["prediction_status"],
        low_memory=False,
    )
    valid_rows = validation.loc[valid_mask]
    centers = pd.to_numeric(valid_rows["T_bin_center_K"], errors="coerce").dropna()
    rows: list[dict[str, Any]] = []
    add_summary_item(rows, "input_rows", len(validation), "Rows in Step6A broad_family input")
    add_summary_item(rows, "valid_rows", int(valid_mask.sum()), "Rows satisfying the Step9A target-row conditions")
    add_summary_item(rows, "bin_width_K", args.bin_width_k, "Temperature-bin width")
    add_summary_item(rows, "bin_start_K", args.bin_start_k, "Left edge defining bin index zero")
    add_summary_item(rows, "unique_T_bin_count", centers.nunique(), "Unique 25 K bin centers among valid rows")
    add_summary_item(rows, "T_bin_min_center_K", centers.min(), "Minimum valid-row bin center")
    add_summary_item(rows, "T_bin_max_center_K", centers.max(), "Maximum valid-row bin center")
    add_summary_item(
        rows,
        "sample_holdout_train_rows",
        int(valid_rows["sample_holdout_split"].eq("train").sum()),
        "Valid sample-holdout train rows",
    )
    add_summary_item(
        rows,
        "sample_holdout_test_rows",
        int(valid_rows["sample_holdout_split"].eq("test").sum()),
        "Valid sample-holdout test rows",
    )
    add_summary_item(
        rows,
        "step5b_prediction_ok_rows",
        int(predictions["prediction_status"].eq("ok").sum()),
        "All-config Step5B prediction rows with status ok",
    )
    add_summary_item(
        rows,
        "step5b_prediction_unavailable_rows",
        int(predictions["prediction_status"].ne("ok").sum()),
        "All-config Step5B prediction rows without a usable prediction",
    )
    add_summary_item(rows, "default_coverage_fraction", coverage_value(coverage), "Primary default Step5B coverage")
    add_summary_item(rows, "default_mae_log10", summary_value(default_summary, "mae_log10"), "Primary row-equal default")
    add_summary_item(rows, "default_rmse_log10", summary_value(default_summary, "rmse_log10"), "Primary row-equal default")
    add_summary_item(rows, "default_factor2", summary_value(default_summary, "factor_2_accuracy"), "Primary row-equal default")
    add_summary_item(rows, "default_factor10", summary_value(default_summary, "factor_10_accuracy"), "Primary row-equal default")
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "n/a"
    text = frame.head(max_rows).copy()
    for column in text.columns:
        text[column] = text[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
        )
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[column] for column in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, separator, *body])


def metric_delta(comparison: pd.DataFrame, metric: str, weighting: str = "row_equal") -> float:
    values = comparison[
        comparison["metric_weighting"].eq(weighting) & comparison["metric_name"].eq(metric)
    ]["delta_25k_minus_100k"]
    return float(values.iloc[0]) if len(values) else np.nan


def metric_pair(comparison: pd.DataFrame, metric: str, weighting: str = "row_equal") -> tuple[float, float]:
    rows = comparison[
        comparison["metric_weighting"].eq(weighting) & comparison["metric_name"].eq(metric)
    ]
    if rows.empty:
        return np.nan, np.nan
    return float(rows["value_100k"].iloc[0]), float(rows["value_25k"].iloc[0])


def material_rankings(material_comparison: pd.DataFrame, limit: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = material_comparison[
        material_comparison["_merge"].eq("both")
        & material_comparison["is_reliable_eval_group_100k"].astype(str).str.casefold().isin({"true", "1"})
        & material_comparison["is_reliable_eval_group_25k"].astype(str).str.casefold().isin({"true", "1"})
        & material_comparison["delta_mae_log10_25k_minus_100k"].notna()
    ].copy()
    columns = [
        "material_group_key",
        "material_family_raw",
        "n_rows_100k",
        "n_rows_25k",
        "mae_log10_100k",
        "mae_log10_25k",
        "delta_mae_log10_25k_minus_100k",
        "delta_rmse_log10_25k_minus_100k",
        "delta_factor_2_accuracy_25k_minus_100k",
        "delta_factor_10_accuracy_25k_minus_100k",
    ]
    improved = (
        eligible[eligible["delta_mae_log10_25k_minus_100k"] < 0]
        .sort_values("delta_mae_log10_25k_minus_100k")
        .head(limit)[columns]
    )
    worsened = (
        eligible[eligible["delta_mae_log10_25k_minus_100k"] > 0]
        .sort_values("delta_mae_log10_25k_minus_100k", ascending=False)
        .head(limit)[columns]
    )
    return improved, worsened


def run_sanity(
    input_frame: pd.DataFrame,
    validation: pd.DataFrame,
    output: Path,
    report: Path,
    comparison: pd.DataFrame,
    default_summary: pd.DataFrame,
    protection: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    checks["input_file_exists"] = True
    checks["validation_rows_created"] = all(
        (output / f"step9a_25k_validation_rows_with_splits.{extension}").exists()
        for extension in ["csv", "parquet"]
    )
    checks["validation_row_count_matches_input"] = len(validation) == len(input_frame)
    checks["old_T_bin_columns_exist"] = set(OLD_T_BIN_COLUMNS).issubset(validation.columns)
    numeric_index = pd.to_numeric(validation["T_bin_index"], errors="coerce")
    expected_center = args.bin_start_k + (numeric_index + 0.5) * args.bin_width_k
    actual_center = pd.to_numeric(validation["T_bin_center_K"], errors="coerce")
    finite_center = np.isfinite(actual_center)
    checks["T_bin_centers_follow_requested_width"] = bool(
        np.allclose(actual_center[finite_center], expected_center[finite_center], rtol=0.0, atol=1e-9)
    )
    if np.isclose(args.bin_width_k, 25.0) and np.isclose(args.bin_start_k, 12.5):
        checks["T_bin_centers_are_25K_multiples"] = bool(
            np.allclose(
                actual_center[finite_center] / 25.0,
                np.round(actual_center[finite_center] / 25.0),
                rtol=0.0,
                atol=1e-9,
            )
        )
    temperature = pd.to_numeric(validation["T_K"], errors="coerce")
    left = pd.to_numeric(validation["T_bin_left_K"], errors="coerce")
    right = pd.to_numeric(validation["T_bin_right_K"], errors="coerce")
    finite_temperature = np.isfinite(temperature)
    checks["temperature_inside_assigned_bin"] = bool(
        ((left[finite_temperature] <= temperature[finite_temperature]) & (temperature[finite_temperature] < right[finite_temperature])).all()
    )
    checks["temperature_bin_version_25K"] = validation["temperature_bin_version"].eq("25K").all()
    checks["material_group_key_not_missing"] = clean_text(validation["material_group_key"]).notna().all()
    checks["holdout_splits_preserved"] = all(
        validation[column].fillna("<NA>").astype(str).equals(input_frame[column].fillna("<NA>").astype(str))
        for column in ["sample_holdout_split", "paper_holdout_split"]
    )
    checks["step5b_outputs_created"] = all(
        (output / filename).exists()
        for filename in [
            "step5b_train_reference_curve_bins.csv",
            "step5b_test_predictions.csv",
            "step5b_test_predictions_valid.csv",
            "step5b_test_predictions_unavailable.csv",
            "step5b_prediction_coverage_by_config.csv",
            "step5b_prediction_unavailable_summary.csv",
            "step5b_dropped_rows.csv",
        ]
    )
    checks["step5c_outputs_created"] = all(
        (output / filename).exists()
        for filename in [
            "step5c_metrics_by_config.csv",
            "step5c_metrics_by_carrier_type.csv",
            "step5c_metrics_by_material_family.csv",
            "step5c_metrics_by_temperature_bin.csv",
            "step5c_metrics_by_eta_bin.csv",
            "step5c_default_comparison.csv",
            "step5c_config_ranking.csv",
            "step5c_largest_abs_error_rows.csv",
            "step5c_dropped_rows.csv",
        ]
    )
    checks["step5b_step5c_checks_passed"] = True
    checks["default_metrics_summary_created"] = not default_summary.empty
    checks["comparison_created"] = not comparison.empty
    checks["report_created"] = report.exists() and report.stat().st_size > 0
    checks["existing_100K_directory_unchanged"] = not protection.empty and protection["unchanged"].all()
    checks["raw_data_not_read"] = True
    failures = [name for name, value in checks.items() if not value]
    return checks, failures


def write_report(
    report: Path,
    input_path: Path,
    validation: pd.DataFrame,
    target_dropped: pd.DataFrame,
    default_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    material_comparison: pd.DataFrame,
    checks: dict[str, bool],
    args: argparse.Namespace,
    elapsed: float,
    small_test_ran: bool,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    primary = default_summary[
        default_summary["default_label"].eq("material_family_default")
        & default_summary["metric_weighting"].eq("row_equal")
    ]
    old_mae, new_mae = metric_pair(comparison, "mae_log10")
    old_rmse, new_rmse = metric_pair(comparison, "rmse_log10")
    old_f2, new_f2 = metric_pair(comparison, "factor_2_accuracy")
    old_f10, new_f10 = metric_pair(comparison, "factor_10_accuracy")
    old_cov, new_cov = metric_pair(comparison, "coverage_fraction")
    improved, worsened = material_rankings(material_comparison)
    accuracy_improved = new_f2 > old_f2 and new_f10 > old_f10
    error_improved = new_mae < old_mae and new_rmse < old_rmse
    coverage_decreased = new_cov < old_cov

    lines = [
        "# Step9A 25 K Temperature-Bin Broad-Family Revalidation Report",
        "",
        "## Input and change from the 100 K version",
        "",
        f"- Input file: `{input_path}`",
        f"- Input rows: {len(validation)}",
        "- The Step6A broad-family `material_group_key` and all existing holdout/CV split columns were retained.",
        "- The only model-condition change is the temperature-bin width: the existing 100 K bins were rebuilt as 25 K bins.",
        "- Existing `T_bin_*` values were preserved as `old_T_bin_*` before replacement.",
        "- The 25 K validation rows retain every input column and add temperature-bin provenance columns.",
        "",
        "## 25 K temperature-bin definition",
        "",
        f"- `bin_width_K = {args.bin_width_k}`",
        f"- `bin_start_K = {args.bin_start_k}`",
        "- Bin index: `floor((T_K - bin_start_K) / bin_width_K)`.",
        "- Interval convention: left-closed and right-open (`T_bin_left_K <= T_K < T_bin_right_K`).",
        "- With the defaults, `[12.5, 37.5)` has center 25 K and centers advance in 25 K increments.",
        f"- Unique valid-row bin centers: {validation.loc[target_usable_mask(validation), 'T_bin_center_K'].nunique()}",
        f"- Rows outside the target-row conditions: {len(target_dropped)}",
        "",
        "## Step5B / Step5C execution",
        "",
        f"- Step5B small test: {'passed' if small_test_ran else 'skipped by --skip-small-test'}",
        f"- Step5C small test: {'passed' if small_test_ran else 'skipped by --skip-small-test'}",
        "- Step5B full run: passed",
        "- Step5C full run: passed",
        "- Step5B rebuilt `sigma0_ref(T)` from train rows only and assigned the resulting 25 K references to test rows.",
        "- Step4 full-data reference curves were not used.",
        "",
        "## 25 K default metrics",
        "",
        dataframe_to_markdown(primary),
        "",
        "## 25 K versus 100 K",
        "",
        f"- Coverage: 100 K={old_cov:.6f}, 25 K={new_cov:.6f}, delta={new_cov - old_cov:+.6f}.",
        f"- Coverage decreased: {coverage_decreased}.",
        f"- MAE: 100 K={old_mae:.6f}, 25 K={new_mae:.6f}, delta={new_mae - old_mae:+.6f}.",
        f"- RMSE: 100 K={old_rmse:.6f}, 25 K={new_rmse:.6f}, delta={new_rmse - old_rmse:+.6f}.",
        f"- Factor-2 accuracy: 100 K={old_f2:.6f}, 25 K={new_f2:.6f}, delta={new_f2 - old_f2:+.6f}.",
        f"- Factor-10 accuracy: 100 K={old_f10:.6f}, 25 K={new_f10:.6f}, delta={new_f10 - old_f10:+.6f}.",
        f"- Both MAE and RMSE improved: {error_improved}.",
        f"- Both factor accuracies improved: {accuracy_improved}.",
        "",
        dataframe_to_markdown(
            comparison[
                comparison["metric_weighting"].eq("row_equal")
                & comparison["metric_name"].isin(
                    [
                        "coverage_fraction",
                        "mae_log10",
                        "rmse_log10",
                        "factor_2_accuracy",
                        "factor_10_accuracy",
                        "max_abs_log10_error",
                        "extreme_ge_10_count",
                        "severe_ge_5_count",
                    ]
                )
            ]
        ),
        "",
        "## Material-family changes",
        "",
        "The tables below rank groups that are reliable in both versions by the change in row-equal MAE.",
        "",
        "### Most improved",
        "",
        dataframe_to_markdown(improved, 10),
        "",
        "### Most worsened",
        "",
        dataframe_to_markdown(worsened, 10),
        "",
        "## Sanity checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {passed}")
    lines.extend(
        [
            "",
            "## Important notes",
            "",
            "- Finer temperature bins can express local temperature dependence more directly, but fewer observations per bin can make the reference coefficient unstable.",
            "- The existing 100 K outputs were not overwritten.",
            "- Starrydata2 raw data was not read.",
            "- This is not a new model; it is a revalidation in which only the temperature-bin width was changed.",
            "- No figures were created in Step9A.",
            "",
            "## Recommended next steps",
            "",
            "- If the 25 K version improves the main metrics without a material coverage loss, create 25 K scatter and material-family plots in Step9B.",
            "- If coverage or RMSE worsens, test an intermediate 50 K width.",
            "- Compare 100 K, 50 K, and 25 K on the same splits and reporting definitions.",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    validate_args(args)
    input_path = resolve_input(args.input)
    if not HUNDRED_K_DEFAULT.exists() or not HUNDRED_K_COVERAGE.exists():
        raise FileNotFoundError("Existing Step6B 100 K comparison inputs are missing")

    hundred_k_before = directory_manifest(HUNDRED_K_DIR)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    log("loading broad_family validation rows...")
    input_frame = read_table(input_path)
    log(f"input rows: {len(input_frame)}")
    require_columns(input_frame)
    validation = assign_temperature_bins(input_frame, args.bin_width_k, args.bin_start_k)
    target_dropped = build_target_dropped_rows(validation)
    target_dropped.to_csv(args.output / "step9a_25k_target_dropped_rows.csv", index=False, encoding="utf-8-sig")

    if clean_text(validation["material_group_key"]).isna().any():
        raise ValueError("material_group_key contains missing values; Step9A sanity requirement cannot be met")

    log("writing 25K validation rows...")
    _, validation_parquet = write_validation_rows(validation, args.output)

    small_test_ran = not args.skip_small_test
    if small_test_ran:
        log("running Step5B small test...")
        run_command(
            step5b_build_command(
                validation_parquet,
                args.output,
                args.report_dir / "step5b_prediction_assignment_report_test.md",
                args,
                args.output_suffix,
                args.max_rows,
            )
        )
        log("checking Step5B small test...")
        run_command(step5b_check_command(args.output, args.output_suffix))
        log("running Step5C small test...")
        run_command(
            step5c_build_command(
                args.output / f"step5b_test_predictions_valid{args.output_suffix}.csv",
                args.output / f"step5b_prediction_coverage_by_config{args.output_suffix}.csv",
                args.output / f"step5b_test_predictions_unavailable{args.output_suffix}.csv",
                args.output,
                args.report_dir / "step5c_evaluation_metrics_report_test.md",
                args,
                args.output_suffix,
                args.max_rows_per_config,
            )
        )
        log("checking Step5C small test...")
        run_command(step5c_check_command(args.output, args.output_suffix))

    log("running Step5B full...")
    run_command(
        step5b_build_command(
            validation_parquet,
            args.output,
            args.report_dir / "step5b_prediction_assignment_report.md",
            args,
        )
    )
    log("checking Step5B full...")
    run_command(step5b_check_command(args.output, require_full=True))

    log("running Step5C full...")
    run_command(
        step5c_build_command(
            args.output / "step5b_test_predictions_valid.parquet",
            args.output / "step5b_prediction_coverage_by_config.csv",
            args.output / "step5b_test_predictions_unavailable.csv",
            args.output,
            args.report_dir / "step5c_evaluation_metrics_report.md",
            args,
        )
    )
    log("checking Step5C full...")
    run_command(step5c_check_command(args.output))

    log("comparing 25K vs 100K results...")
    comparison = build_default_comparison(args.output)
    comparison.to_csv(
        args.output / "step9a_25k_vs_100k_default_metrics_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    material_comparison = build_material_comparison(args.output)
    material_comparison.to_csv(
        args.output / "step9a_25k_vs_100k_material_family_metrics_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    log("writing summaries...")
    default_summary = build_default_summary(args.output)
    default_summary.to_csv(
        args.output / "step9a_25k_default_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    material_summary = build_material_summary(args.output)
    material_summary.to_csv(
        args.output / "step9a_25k_material_family_metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bin_summary = build_bin_coverage_summary(
        validation,
        target_usable_mask(validation),
        args.output,
        args,
        default_summary,
    )
    bin_summary.to_csv(
        args.output / "step9a_25k_bin_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hundred_k_after = directory_manifest(HUNDRED_K_DIR)
    protection = build_protection_manifest(hundred_k_before, hundred_k_after)
    protection.to_csv(
        args.output / "step9a_100k_directory_protection_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = args.report_dir / "step9a_25k_bin_revalidation_report.md"
    log("writing report...")
    write_report(
        report,
        input_path,
        validation,
        target_dropped,
        default_summary,
        comparison,
        material_comparison,
        {},
        args,
        time.time() - started,
        small_test_ran,
    )
    checks, failures = run_sanity(
        input_frame,
        validation,
        args.output,
        report,
        comparison,
        default_summary,
        protection,
        args,
    )
    if failures:
        for failure in failures:
            print(f"[step9a] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(
        report,
        input_path,
        validation,
        target_dropped,
        default_summary,
        comparison,
        material_comparison,
        checks,
        args,
        time.time() - started,
        small_test_ran,
    )
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
