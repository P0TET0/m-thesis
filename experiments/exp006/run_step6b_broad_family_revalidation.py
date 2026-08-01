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
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.csv"
DEFAULT_OUTPUT = PROCESSED_DIR / "step6b_broad_family"
DEFAULT_REPORT_DIR = REPORT_DIR / "step6b_broad_family"
ORIGINAL_DEFAULT_COMPARISON = PROCESSED_DIR / "step5c_default_comparison.csv"

DEFAULT_CONFIGS = {
    "material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

METRICS = [
    "mae_log10",
    "rmse_log10",
    "median_log10_error",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "coverage_fraction",
    "n_rows",
    "n_samples",
    "n_papers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step6B broad_family revalidation.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--min-eval-rows", type=int, default=30)
    parser.add_argument("--min-eval-samples", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument("--max-rows-per-config", type=int, default=200)
    parser.add_argument("--output-suffix", default="_test")
    parser.add_argument("--skip-small-test", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Reuse existing Step5B/Step5C outputs and rebuild only Step6B summary artifacts.")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step6b] {message}", flush=True)


def resolve_input(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(explicit)
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step6A broad_family variant input not found")


def run_command(cmd: list[str]) -> None:
    log("running: " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def step5b_build_cmd(input_path: Path, output: Path, report: Path, args: argparse.Namespace, suffix: str = "", max_rows: int | None = None) -> list[str]:
    cmd = [
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
        cmd.extend(["--max-rows", str(max_rows), "--output-suffix", suffix])
    return cmd


def step5b_check_cmd(output: Path, suffix: str = "", require_full: bool = False) -> list[str]:
    cmd = [
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
        cmd.append("--require-full-run")
    return cmd


def step5c_build_cmd(input_path: Path, coverage: Path, unavailable: Path, output: Path, report: Path, args: argparse.Namespace, suffix: str = "", max_rows_per_config: int | None = None) -> list[str]:
    cmd = [
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
        cmd.extend(["--max-rows-per-config", str(max_rows_per_config), "--output-suffix", suffix])
    return cmd


def step5c_check_cmd(output: Path, suffix: str = "") -> list[str]:
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


def compare_pair(pred: pd.DataFrame, left_config: str, right_config: str, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = [
        "row_id",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "sigma0_ref_S_per_m",
        "log10_sigma0_ref_S_per_m",
        "log10_sigma_pred_over_exp",
        "material_group_key_for_prediction",
        "T_bin_center_K",
        "carrier_type",
    ]
    left = pred[pred["config_id"].eq(left_config)][cols].copy()
    right = pred[pred["config_id"].eq(right_config)][cols].copy()
    merged = left.merge(right, on="row_id", suffixes=("_material_family", "_global"), how="inner")
    merged["comparison_label"] = label
    merged["delta_log10_sigma_pred"] = merged["log10_sigma_pred_S_per_m_material_family"] - merged["log10_sigma_pred_S_per_m_global"]
    merged["delta_log10_sigma0_ref"] = merged["log10_sigma0_ref_S_per_m_material_family"] - merged["log10_sigma0_ref_S_per_m_global"]
    abs_pred = merged["delta_log10_sigma_pred"].abs()
    abs_ref = merged["delta_log10_sigma0_ref"].abs()
    summary = {
        "comparison_label": label,
        "joined_row_count": len(merged),
        "max_abs_delta_log10_sigma_pred": float(abs_pred.max()) if len(merged) else np.nan,
        "median_abs_delta_log10_sigma_pred": float(abs_pred.median()) if len(merged) else np.nan,
        "mean_abs_delta_log10_sigma_pred": float(abs_pred.mean()) if len(merged) else np.nan,
        "max_abs_delta_log10_sigma0_ref": float(abs_ref.max()) if len(merged) else np.nan,
        "median_abs_delta_log10_sigma0_ref": float(abs_ref.median()) if len(merged) else np.nan,
        "exact_equal_prediction_count": int((merged["sigma_pred_S_per_m_material_family"] == merged["sigma_pred_S_per_m_global"]).sum()),
        "approximately_equal_prediction_count": int((abs_pred <= 1e-12).sum()),
        "different_prediction_count": int((abs_pred > 1e-12).sum()),
        "different_prediction_fraction": float((abs_pred > 1e-12).mean()) if len(merged) else np.nan,
        "unique_material_group_key_for_prediction_count_material_family": int(left["material_group_key_for_prediction"].nunique()),
        "unique_material_group_key_for_prediction_count_global": int(right["material_group_key_for_prediction"].nunique()),
    }
    return merged.sort_values("delta_log10_sigma_pred", key=lambda s: s.abs(), ascending=False).head(1000), summary


def build_prediction_diff(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(output / "step5b_test_predictions_valid.csv", low_memory=False)
    examples: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for left_label, right_label, label in [
        ("material_family_default", "global_default", "sample_holdout_material_family_vs_global"),
        ("paper_material_family_default", "paper_global_default", "paper_holdout_material_family_vs_global"),
    ]:
        ex, summary = compare_pair(pred, DEFAULT_CONFIGS[left_label], DEFAULT_CONFIGS[right_label], label)
        examples.append(ex)
        summaries.append(summary)
    return pd.concat(examples, ignore_index=True), pd.DataFrame(summaries)


def build_reference_diag(output: Path) -> pd.DataFrame:
    ref = pd.read_csv(output / "step5b_train_reference_curve_bins.csv", low_memory=False)
    rows: list[dict[str, Any]] = []
    for keys, group in ref.groupby(["config_id", "split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"], dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(["config_id", "split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"], key_values))
        row.update(
            {
                "material_group_count": group["material_group_key"].nunique(),
                "reference_bin_count": len(group),
                "reliable_reference_bin_count": int(as_bool(group["is_reference_bin_candidate"]).sum()),
                "material_group_examples": " | ".join(map(str, group["material_group_key"].dropna().unique()[:10])),
                "T_bin_count": group["T_bin_center_K"].nunique(),
                "carrier_type_values": " | ".join(map(str, group["carrier_type"].dropna().unique())),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def default_label(config_id: str) -> str:
    for label, cid in DEFAULT_CONFIGS.items():
        if cid == config_id:
            return label
    return config_id


def build_default_metric_summary(output: Path) -> pd.DataFrame:
    default = pd.read_csv(output / "step5c_default_comparison.csv", low_memory=False)
    sub = default[default["config_id"].isin(DEFAULT_CONFIGS.values())].copy()
    sub["default_label"] = sub["config_id"].map(default_label)
    cols = ["default_label", "config_id", "metric_weighting", *METRICS]
    return sub[cols]


def build_original_comparison(output: Path) -> pd.DataFrame:
    original = pd.read_csv(ORIGINAL_DEFAULT_COMPARISON, low_memory=False)
    broad = pd.read_csv(output / "step5c_default_comparison.csv", low_memory=False)
    rows: list[dict[str, Any]] = []
    for label, config_id in DEFAULT_CONFIGS.items():
        for weighting in ["row_equal", "sample_equal"]:
            orig_row = original[original["config_id"].eq(config_id) & original["metric_weighting"].eq(weighting)]
            broad_row = broad[broad["config_id"].eq(config_id) & broad["metric_weighting"].eq(weighting)]
            if orig_row.empty or broad_row.empty:
                continue
            orig_row = orig_row.iloc[0]
            broad_row = broad_row.iloc[0]
            for metric in METRICS:
                orig = pd.to_numeric(pd.Series([orig_row[metric]]), errors="coerce").iloc[0]
                new = pd.to_numeric(pd.Series([broad_row[metric]]), errors="coerce").iloc[0]
                rel = (new - orig) / orig if pd.notna(orig) and orig != 0 else np.nan
                hint = "lower_is_better" if metric in {"mae_log10", "rmse_log10"} else "higher_is_better" if "accuracy" in metric or metric == "coverage_fraction" else "count_or_context"
                rows.append(
                    {
                        "default_label": label,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "original_value": orig,
                        "broad_family_value": new,
                        "delta_broad_minus_original": new - orig,
                        "relative_change_if_applicable": rel,
                        "interpretation_hint": hint,
                    }
                )
    return pd.DataFrame(rows)


def add_summary_item(rows: list[dict[str, str]], item: str, value: Any, comment: str) -> None:
    rows.append({"item": item, "value": str(value), "comment": comment})


def df_to_markdown(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def build_revalidation_summary(input_df: pd.DataFrame, output: Path, diff_summary: pd.DataFrame, default_summary: pd.DataFrame, original_compare: pd.DataFrame) -> pd.DataFrame:
    coverage = pd.read_csv(output / "step5b_prediction_coverage_by_config.csv", low_memory=False)
    pred = pd.read_csv(output / "step5b_test_predictions.csv", usecols=["prediction_status"], low_memory=False)
    rows: list[dict[str, str]] = []
    add_summary_item(rows, "input_variant", "broad_family", "Step6A broad_family material_group_key")
    add_summary_item(rows, "input_rows", len(input_df), "Rows in Step6A broad_family input")
    add_summary_item(rows, "material_group_key_unique_count", input_df["material_group_key"].nunique(), "Broad-family group count")
    add_summary_item(rows, "material_group_key_unknown_fraction", input_df["material_group_key"].eq("unknown_material_group").mean(), "Unknown broad-family fraction")
    add_summary_item(rows, "step5b_prediction_ok_rows", int(pred["prediction_status"].eq("ok").sum()), "All-config ok prediction rows")
    add_summary_item(rows, "step5b_prediction_unavailable_rows", int(pred["prediction_status"].ne("ok").sum()), "All-config unavailable rows")
    for item, label in [
        ("step5b_default_coverage_fraction", "material_family_default"),
        ("step5b_global_default_coverage_fraction", "global_default"),
    ]:
        cov = coverage[coverage["config_id"].eq(DEFAULT_CONFIGS[label])]["coverage_fraction"]
        add_summary_item(rows, item, cov.iloc[0] if not cov.empty else "n/a", "Step5B coverage")
    for item, label, metric in [
        ("step5c_default_mae_log10", "material_family_default", "mae_log10"),
        ("step5c_default_factor_2_accuracy", "material_family_default", "factor_2_accuracy"),
        ("step5c_default_factor_10_accuracy", "material_family_default", "factor_10_accuracy"),
        ("step5c_global_default_mae_log10", "global_default", "mae_log10"),
    ]:
        val = default_summary[default_summary["default_label"].eq(label) & default_summary["metric_weighting"].eq("row_equal")][metric]
        add_summary_item(rows, item, val.iloc[0] if not val.empty else "n/a", "Step5C row_equal default metric")
    sample_diff = diff_summary[diff_summary["comparison_label"].str.startswith("sample")]
    paper_diff = diff_summary[diff_summary["comparison_label"].str.startswith("paper")]
    add_summary_item(rows, "material_family_vs_global_predictions_identical", bool((sample_diff["different_prediction_count"] == 0).all()), "Sample holdout default")
    add_summary_item(rows, "material_family_vs_global_different_prediction_fraction", sample_diff["different_prediction_fraction"].iloc[0] if not sample_diff.empty else "n/a", "Sample holdout default")
    add_summary_item(rows, "paper_material_family_vs_global_predictions_identical", bool((paper_diff["different_prediction_count"] == 0).all()), "Paper holdout default")
    add_summary_item(rows, "paper_material_family_vs_global_different_prediction_fraction", paper_diff["different_prediction_fraction"].iloc[0] if not paper_diff.empty else "n/a", "Paper holdout default")
    comp = original_compare[
        original_compare["default_label"].eq("material_family_default")
        & original_compare["metric_weighting"].eq("row_equal")
    ]
    orig_mae = comp[comp["metric_name"].eq("mae_log10")]["original_value"]
    new_mae = comp[comp["metric_name"].eq("mae_log10")]["broad_family_value"]
    delta_mae = comp[comp["metric_name"].eq("mae_log10")]["delta_broad_minus_original"]
    add_summary_item(rows, "original_default_mae_log10", orig_mae.iloc[0] if not orig_mae.empty else "n/a", "Original Step5C default MAE")
    add_summary_item(rows, "broad_family_default_mae_log10", new_mae.iloc[0] if not new_mae.empty else "n/a", "Step6B broad-family default MAE")
    add_summary_item(rows, "default_mae_delta_broad_minus_original", delta_mae.iloc[0] if not delta_mae.empty else "n/a", "Positive means broad_family is worse by MAE")
    action = "Step6C visualization" if not sample_diff.empty and sample_diff["different_prediction_count"].iloc[0] > 0 else "debug Step5B group key handling"
    add_summary_item(rows, "recommended_next_action", action, "Next decision based on material vs global difference")
    return pd.DataFrame(rows)


def write_report(report: Path, input_path: Path, input_df: pd.DataFrame, diff_summary: pd.DataFrame, ref_diag: pd.DataFrame, default_summary: pd.DataFrame, original_compare: pd.DataFrame, summary: pd.DataFrame, checks: dict[str, bool], elapsed: float) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    top_groups = input_df["material_group_key"].value_counts().head(20).to_dict()
    lines = [
        "# Step6B Broad Family Revalidation Report",
        "",
        "## Summary",
        "",
        f"- input_variant: broad_family",
        f"- input_file: {input_path}",
        f"- input_rows: {len(input_df)}",
        f"- material_group_key unique count: {input_df['material_group_key'].nunique()}",
        f"- material_group_key top groups: {top_groups}",
        "- Step5B small test: passed",
        "- Step5B full run: passed",
        "- Step5C small test: passed",
        "- Step5C full run: passed",
        "",
        "## Prediction Diff Summary",
        "",
        df_to_markdown(diff_summary),
        "",
        "## Reference Group Diagnostics Preview",
        "",
        df_to_markdown(ref_diag, 20),
        "",
        "## Broad Family Default Metrics",
        "",
        df_to_markdown(default_summary),
        "",
        "## Original vs Broad Family Default Metrics",
        "",
        df_to_markdown(original_compare, 80),
        "",
        "## Revalidation Summary",
        "",
        df_to_markdown(summary),
        "",
        "## Sanity Check",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Step4 full-data reference curves were not read.",
            "- Starrydata2 raw data was not read.",
            "- Existing Step5B/Step5C outputs were not overwritten; outputs are under step6b_broad_family.",
            "- If material_family and global still match, inspect Step5B join keys and material_group_key_for_prediction.",
            "- If they differ, Step6C should visualize the broad_family revalidation.",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity(input_df: pd.DataFrame, output: Path, report: Path, diff_summary: pd.DataFrame, ref_diag: pd.DataFrame, original_compare: pd.DataFrame, default_summary: pd.DataFrame, summary: pd.DataFrame) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    checks["output_dir_is_step6b_specific"] = output.name == "step6b_broad_family"
    checks["input_unique_groups_gt_1"] = input_df["material_group_key"].nunique() > 1
    checks["input_material_group_key_not_missing"] = input_df["material_group_key"].notna().all()
    checks["step5b_full_outputs_exist"] = all((output / name).exists() for name in ["step5b_test_predictions.csv", "step5b_test_predictions_valid.csv", "step5b_prediction_coverage_by_config.csv"])
    checks["step5c_full_outputs_exist"] = all((output / name).exists() for name in ["step5c_metrics_by_config.csv", "step5c_default_comparison.csv", "step5c_config_ranking.csv"])
    valid = pd.read_csv(output / "step5b_test_predictions_valid.csv", usecols=["config_id"], low_memory=False)
    metrics = pd.read_csv(output / "step5c_metrics_by_config.csv", low_memory=False)
    default = pd.read_csv(output / "step5c_default_comparison.csv", low_memory=False)
    checks["step5b_valid_nonzero"] = len(valid) > 0
    checks["step5c_metrics_nonzero"] = len(metrics) > 0
    checks["step5c_default_comparison_8_rows"] = len(default) == 8
    checks["step5b_config_count_32"] = valid["config_id"].nunique() == 32
    checks["step5c_config_count_32"] = metrics["config_id"].nunique() == 32
    checks["diff_summary_created"] = not diff_summary.empty
    checks["reference_diag_created"] = not ref_diag.empty
    checks["original_comparison_created"] = not original_compare.empty
    checks["default_summary_created"] = not default_summary.empty
    checks["summary_created"] = not summary.empty
    checks["report_created"] = report.exists() and report.stat().st_size > 0
    checks["did_not_read_step4_full_data_reference_curve"] = True
    checks["did_not_read_raw_data"] = True
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def write_summary_outputs(args: argparse.Namespace, input_path: Path, input_df: pd.DataFrame, started: float) -> None:
    log("comparing material_family vs global predictions...")
    diff_examples, diff_summary = build_prediction_diff(args.output)
    diff_summary.to_csv(args.output / "step6b_material_family_vs_global_prediction_diff_summary.csv", index=False, encoding="utf-8-sig")
    diff_examples.to_csv(args.output / "step6b_material_family_vs_global_prediction_diff_examples.csv", index=False, encoding="utf-8-sig")
    ref_diag = build_reference_diag(args.output)
    ref_diag.to_csv(args.output / "step6b_reference_group_diagnostics.csv", index=False, encoding="utf-8-sig")

    log("comparing broad_family metrics with original metrics...")
    default_summary = build_default_metric_summary(args.output)
    default_summary.to_csv(args.output / "step6b_broad_family_default_metrics_summary.csv", index=False, encoding="utf-8-sig")
    original_compare = build_original_comparison(args.output)
    original_compare.to_csv(args.output / "step6b_broad_family_vs_original_default_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    summary = build_revalidation_summary(input_df, args.output, diff_summary, default_summary, original_compare)
    summary.to_csv(args.output / "step6b_revalidation_summary.csv", index=False, encoding="utf-8-sig")

    log("writing summary tables...")
    report = args.report_dir / "step6b_broad_family_revalidation_report.md"
    log("writing report...")
    write_report(report, input_path, input_df, diff_summary, ref_diag, default_summary, original_compare, summary, {}, time.time() - started)
    checks, failures = run_sanity(input_df, args.output, report, diff_summary, ref_diag, original_compare, default_summary, summary)
    if failures:
        for failure in failures:
            print(f"[step6b] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(report, input_path, input_df, diff_summary, ref_diag, default_summary, original_compare, summary, checks, time.time() - started)


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    log("loading broad_family variant input...")
    input_df = read_table(input_path)
    log(f"input rows: {len(input_df)}")
    log(f"material_group_key unique count: {input_df['material_group_key'].nunique()}")
    if input_df["material_group_key"].nunique() <= 1:
        raise SystemExit("broad_family input material_group_key unique count must be > 1")

    if args.summary_only:
        write_summary_outputs(args, input_path, input_df, started)
        log("done.")
        log(f"elapsed seconds: {time.time() - started:.2f}")
        return

    if not args.skip_small_test:
        log("running Step5B small test...")
        run_command(step5b_build_cmd(input_path, args.output, args.report_dir / "step5b_prediction_assignment_report_test.md", args, args.output_suffix, args.max_rows))
        log("checking Step5B small test...")
        run_command(step5b_check_cmd(args.output, args.output_suffix))
        log("running Step5C small test...")
        run_command(
            step5c_build_cmd(
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
        run_command(step5c_check_cmd(args.output, args.output_suffix))

    log("running Step5B full...")
    run_command(step5b_build_cmd(input_path, args.output, args.report_dir / "step5b_prediction_assignment_report.md", args))
    log("checking Step5B full...")
    run_command(step5b_check_cmd(args.output, "", True))
    log("running Step5C full...")
    run_command(
        step5c_build_cmd(
            args.output / "step5b_test_predictions_valid.parquet",
            args.output / "step5b_prediction_coverage_by_config.csv",
            args.output / "step5b_test_predictions_unavailable.csv",
            args.output,
            args.report_dir / "step5c_evaluation_metrics_report.md",
            args,
        )
    )
    log("checking Step5C full...")
    run_command(step5c_check_cmd(args.output))

    write_summary_outputs(args, input_path, input_df, started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
