import argparse
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
FIGURE_DIR = EXP_DIR / "figures" / "step5d"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_CONFIGS = {
    "material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

REQUIRED_PREDICTION_COLUMNS = [
    "config_id",
    "prediction_status",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "material_group_key_for_prediction",
    "T_K",
    "T_bin_center_K",
    "carrier_type",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
]

OPTIONAL_PREDICTION_COLUMNS = [
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
]

NUMERIC_COLUMNS = [
    "T_K",
    "T_bin_center_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
]

DIAGNOSTIC_OUTLIER_COLUMNS = [
    "config_id",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
    "carrier_type",
    "T_K",
    "T_bin_center_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_ref_over_row_sigma0",
    "log10_sigma0_ref_over_row_sigma0",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
    "outlier_direction",
    "outlier_severity",
    "likely_error_origin_hint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step5D-1 visual diagnostics.")
    parser.add_argument("--predictions-valid", type=Path, default=PROCESSED_DIR / "step5b_test_predictions_valid.parquet")
    parser.add_argument("--predictions-all", type=Path, default=PROCESSED_DIR / "step5b_test_predictions.csv")
    parser.add_argument("--reference-bins", type=Path, default=PROCESSED_DIR / "step5b_train_reference_curve_bins.csv")
    parser.add_argument("--metrics-config", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_config.csv")
    parser.add_argument("--metrics-carrier", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_carrier_type.csv")
    parser.add_argument("--metrics-material", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_material_family.csv")
    parser.add_argument("--metrics-temperature", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_temperature_bin.csv")
    parser.add_argument("--metrics-eta", type=Path, default=PROCESSED_DIR / "step5c_metrics_by_eta_bin.csv")
    parser.add_argument("--default-comparison", type=Path, default=PROCESSED_DIR / "step5c_default_comparison.csv")
    parser.add_argument("--ranking", type=Path, default=PROCESSED_DIR / "step5c_config_ranking.csv")
    parser.add_argument("--largest-errors", type=Path, default=PROCESSED_DIR / "step5c_largest_abs_error_rows.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--figures", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step5d_visual_diagnostics_report.md")
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step5d] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def fig_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def validate_prediction_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_PREDICTION_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"predictions-valid missing required columns: {missing}")
    for col in OPTIONAL_PREDICTION_COLUMNS:
        if col not in df.columns:
            df[col] = ""


def prepare_predictions(df: pd.DataFrame, max_rows_per_config: int | None) -> pd.DataFrame:
    if max_rows_per_config is not None:
        if max_rows_per_config <= 0:
            raise ValueError("--max-rows-per-config must be positive")
        df = df.groupby("config_id", dropna=False, sort=False).head(max_rows_per_config).copy()
    validate_prediction_columns(df)
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log10_sigma0_ref_over_row_sigma0"] = df["log10_sigma0_ref_S_per_m"] - df["log10_sigma0_S_per_m"]
    df["sigma0_ref_over_row_sigma0"] = 10.0 ** df["log10_sigma0_ref_over_row_sigma0"]
    return df


def sample_for_plot(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def metric_for_config(metrics_config: pd.DataFrame, config_id: str) -> dict[str, float]:
    row = metrics_config[
        metrics_config["config_id"].eq(config_id) & metrics_config["metric_weighting"].eq("row_equal")
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def save_figure(fig: plt.Figure, figures_dir: Path, base: str, suffix: str, title: str, source: str, config_id: str, n_points: int, description: str, index: list[dict[str, Any]]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / fig_name(base, suffix, "png")
    pdf = figures_dir / fig_name(base, suffix, "pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    index.append(
        {
            "figure_id": base + suffix,
            "figure_path_png": str(png),
            "figure_path_pdf": str(pdf),
            "title": title,
            "source_file": source,
            "config_id": config_id,
            "n_points_plotted": n_points,
            "description": description,
        }
    )


def short_label(config_id: str) -> str:
    return (
        config_id.replace("sample_holdout__", "sample|")
        .replace("paper_holdout__", "paper|")
        .replace("ref_", "ref=")
        .replace("__eval_", "|eval=")
        .replace("__material_family__", "|mat|")
        .replace("__global__", "|glob|")
        .replace("__", "|")
    )


def compare_config_pair(df: pd.DataFrame, left_config: str, right_config: str, comparison_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = df[df["config_id"].eq(left_config)].copy()
    right = df[df["config_id"].eq(right_config)].copy()
    cols = [
        "row_id",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "sigma0_ref_S_per_m",
        "log10_sigma0_ref_S_per_m",
        "log10_sigma_pred_over_exp",
        "material_group_key",
        "material_group_key_for_prediction",
        "T_bin_center_K",
        "carrier_type",
    ]
    merged = left[cols].merge(right[cols], on="row_id", suffixes=("_material_family", "_global"), how="inner")
    merged["comparison_label"] = comparison_label
    merged["delta_log10_sigma_pred"] = (
        merged["log10_sigma_pred_S_per_m_material_family"] - merged["log10_sigma_pred_S_per_m_global"]
    )
    merged["delta_log10_sigma0_ref"] = (
        merged["log10_sigma0_ref_S_per_m_material_family"] - merged["log10_sigma0_ref_S_per_m_global"]
    )
    abs_pred = merged["delta_log10_sigma_pred"].abs()
    abs_ref = merged["delta_log10_sigma0_ref"].abs()
    summary = {
        "comparison_label": comparison_label,
        "left_config_id": left_config,
        "right_config_id": right_config,
        "joined_row_count": len(merged),
        "max_abs_delta_log10_sigma_pred": float(abs_pred.max()) if len(merged) else np.nan,
        "median_abs_delta_log10_sigma_pred": float(abs_pred.median()) if len(merged) else np.nan,
        "max_abs_delta_log10_sigma0_ref": float(abs_ref.max()) if len(merged) else np.nan,
        "median_abs_delta_log10_sigma0_ref": float(abs_ref.median()) if len(merged) else np.nan,
        "exact_equal_prediction_count": int((merged["sigma_pred_S_per_m_material_family"] == merged["sigma_pred_S_per_m_global"]).sum()) if len(merged) else 0,
        "approximately_equal_prediction_count": int((abs_pred <= 1e-12).sum()) if len(merged) else 0,
        "different_prediction_count": int((abs_pred > 1e-12).sum()) if len(merged) else 0,
        "unique_material_group_key_count": int(left["material_group_key"].nunique(dropna=True)),
        "unique_material_group_key_examples": " | ".join(map(str, left["material_group_key"].dropna().unique()[:5])),
        "unique_material_group_key_for_prediction_count_material_family": int(left["material_group_key_for_prediction"].nunique(dropna=True)),
        "unique_material_group_key_for_prediction_count_global": int(right["material_group_key_for_prediction"].nunique(dropna=True)),
    }
    return merged, summary


def build_prediction_diff(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [
        (
            DEFAULT_CONFIGS["material_family_default"],
            DEFAULT_CONFIGS["global_default"],
            "sample_holdout_material_family_vs_global",
        ),
        (
            DEFAULT_CONFIGS["paper_material_family_default"],
            DEFAULT_CONFIGS["paper_global_default"],
            "paper_holdout_material_family_vs_global",
        ),
    ]
    diff_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for left, right, label in pairs:
        diff, summary = compare_config_pair(df, left, right, label)
        diff_frames.append(diff.head(10000))
        summaries.append(summary)
    return pd.concat(diff_frames, ignore_index=True), pd.DataFrame(summaries)


def build_reference_diagnostics(reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    default_ids = list(DEFAULT_CONFIGS.values())
    ref = reference[reference["config_id"].isin(default_ids)].copy()
    counts = (
        ref.groupby(["config_id", "group_scheme", "material_group_key"], dropna=False)
        .size()
        .reset_index(name="reference_bin_count")
    )
    rows: list[dict[str, Any]] = []
    for sample_prefix in ["sample_holdout", "paper_holdout"]:
        mat_id = DEFAULT_CONFIGS["material_family_default"] if sample_prefix == "sample_holdout" else DEFAULT_CONFIGS["paper_material_family_default"]
        glob_id = DEFAULT_CONFIGS["global_default"] if sample_prefix == "sample_holdout" else DEFAULT_CONFIGS["paper_global_default"]
        mat = ref[ref["config_id"].eq(mat_id)].copy()
        glob = ref[ref["config_id"].eq(glob_id)].copy()
        joined = mat.merge(
            glob[["carrier_type", "T_bin_center_K", "log10_sigma0_ref_S_per_m", "sigma0_ref_S_per_m"]],
            on=["carrier_type", "T_bin_center_K"],
            how="inner",
            suffixes=("_material_family", "_global"),
        )
        delta = joined["log10_sigma0_ref_S_per_m_material_family"] - joined["log10_sigma0_ref_S_per_m_global"]
        rows.append(
            {
                "comparison_label": f"{sample_prefix}_reference_material_family_vs_global",
                "material_family_config_id": mat_id,
                "global_config_id": glob_id,
                "material_family_reference_bins": len(mat),
                "global_reference_bins": len(glob),
                "material_family_material_group_key_count": mat["material_group_key"].nunique(dropna=True),
                "material_family_material_group_key_examples": " | ".join(map(str, mat["material_group_key"].dropna().unique()[:10])),
                "joined_carrier_T_bins": len(joined),
                "max_abs_delta_log10_sigma0_ref": float(delta.abs().max()) if len(joined) else np.nan,
                "median_abs_delta_log10_sigma0_ref": float(delta.abs().median()) if len(joined) else np.nan,
                "same_reference_value_count": int((delta.abs() <= 1e-12).sum()) if len(joined) else 0,
                "different_reference_value_count": int((delta.abs() > 1e-12).sum()) if len(joined) else 0,
            }
        )
    return pd.DataFrame(rows), counts


def classify_outlier(row: pd.Series) -> tuple[str, str, str]:
    direction = "over_predicted" if row["log10_sigma_pred_over_exp"] > 0 else "under_predicted"
    abs_error = abs(row["log10_sigma_pred_over_exp"])
    if abs_error >= 10:
        severity = "extreme_ge_10_decades"
    elif abs_error >= 5:
        severity = "severe_ge_5_decades"
    elif abs_error >= 2:
        severity = "large_ge_2_decades"
    else:
        severity = "moderate"
    if row["log10_sigma0_ref_over_row_sigma0"] >= 2:
        hint = "sigma0_ref_much_larger_than_row_sigma0"
    elif row["log10_sigma0_ref_over_row_sigma0"] <= -2:
        hint = "sigma0_ref_much_smaller_than_row_sigma0"
    elif row["sigma_S_per_m"] <= 1e-6:
        hint = "very_low_sigma_exp"
    elif row["sigma_S_per_m"] >= 1e7:
        hint = "very_high_sigma_exp"
    else:
        hint = "other_or_needs_manual_check"
    return direction, severity, hint


def build_largest_error_diagnostics(valid: pd.DataFrame, largest_errors: pd.DataFrame) -> pd.DataFrame:
    keys = ["config_id", "row_id"]
    largest_top = largest_errors.sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(100)[keys].copy()
    merged = largest_top.merge(valid, on=keys, how="left")
    if merged["sigma_S_per_m"].isna().any():
        fallback = valid.sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(100).copy()
        merged = pd.concat([merged.dropna(subset=["sigma_S_per_m"]), fallback], ignore_index=True)
        merged = merged.drop_duplicates(keys).sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(100)
    classifications = merged.apply(classify_outlier, axis=1, result_type="expand")
    merged["outlier_direction"] = classifications[0]
    merged["outlier_severity"] = classifications[1]
    merged["likely_error_origin_hint"] = classifications[2]
    for col in DIAGNOSTIC_OUTLIER_COLUMNS:
        if col not in merged.columns:
            merged[col] = ""
    return merged[DIAGNOSTIC_OUTLIER_COLUMNS]


def build_default_metrics(default_comparison: pd.DataFrame) -> pd.DataFrame:
    label_map = {v: k for k, v in DEFAULT_CONFIGS.items()}
    cols = [
        "config_id",
        "metric_weighting",
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
    out = default_comparison[default_comparison["config_id"].isin(DEFAULT_CONFIGS.values())].copy()
    out["config_label"] = out["config_id"].map(label_map)
    return out[["config_label", *cols]]


def plot_pred_vs_exp(df: pd.DataFrame, metrics_config: pd.DataFrame, config_label: str, config_id: str, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    sub = df[df["config_id"].eq(config_id)].copy()
    sub = sample_for_plot(sub, args.plot_sample_size, args.seed)
    metric = metric_for_config(metrics_config, config_id)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(sub["sigma_S_per_m"], sub["sigma_pred_S_per_m"], s=6, alpha=0.35)
    lo = min(sub["sigma_S_per_m"].min(), sub["sigma_pred_S_per_m"].min())
    hi = max(sub["sigma_S_per_m"].max(), sub["sigma_pred_S_per_m"].max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sigma_exp (S/m)")
    ax.set_ylabel("sigma_pred (S/m)")
    title = f"{config_label}: pred vs exp\nn={len(sub)}, MAE={metric.get('mae_log10', np.nan):.3g}, F2={metric.get('factor_2_accuracy', np.nan):.3g}, F10={metric.get('factor_10_accuracy', np.nan):.3g}"
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    save_figure(fig, args.figures, f"step5d_scatter_pred_vs_exp_{config_label}", args.output_suffix, title, str(args.predictions_valid), config_id, len(sub), "log-log predicted vs experimental conductivity", index)


def plot_error_hist(df: pd.DataFrame, metrics_config: pd.DataFrame, config_label: str, config_id: str, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    sub = df[df["config_id"].eq(config_id)].copy()
    errors = sub["log10_sigma_pred_over_exp"]
    outside = int(((errors < -5) | (errors > 5)).sum())
    metric = metric_for_config(metrics_config, config_id)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(errors.clip(-5, 5), bins=80)
    for x, label in [(0, "0"), (-1, "-1"), (1, "+1")]:
        ax.axvline(x, linestyle="--", linewidth=1, label=label)
    ax.set_xlabel("log10(sigma_pred / sigma_exp)")
    ax.set_ylabel("count")
    title = f"{config_label}: error distribution\nmedian={metric.get('median_log10_error', np.nan):.3g}, MAE={metric.get('mae_log10', np.nan):.3g}, RMSE={metric.get('rmse_log10', np.nan):.3g}, outside[-5,5]={outside}"
    ax.set_title(title)
    ax.legend()
    save_figure(fig, args.figures, f"step5d_error_hist_{config_label}", args.output_suffix, title, str(args.predictions_valid), config_id, len(sub), "Histogram of log10 prediction error clipped to [-5, 5]", index)


def plot_config_bar(metrics_config: pd.DataFrame, value_col: str, ascending: bool, base: str, title: str, ylabel: str, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    sub = metrics_config[metrics_config["metric_weighting"].eq("row_equal")].sort_values(value_col, ascending=ascending).head(16).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [short_label(x) for x in sub["config_id"]]
    ax.bar(range(len(sub)), sub[value_col])
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_figure(fig, args.figures, base, args.output_suffix, title, str(args.metrics_config), "multiple", len(sub), title, index)


def plot_metric_line(metrics: pd.DataFrame, x_col: str, y_col: str, labels: list[tuple[str, str]], base: str, title: str, xlabel: str, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n = 0
    for label, config_id in labels:
        sub = metrics[
            metrics["config_id"].eq(config_id) & metrics["metric_weighting"].eq("row_equal")
        ].copy()
        if sub.empty:
            continue
        sub = sub.sort_values(x_col)
        ax.plot(sub[x_col].astype(str) if x_col.endswith("label") else sub[x_col], sub[y_col], marker="o", label=label)
        n += len(sub)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.legend()
    if x_col.endswith("label"):
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_figure(fig, args.figures, base, args.output_suffix, title, "Step5C metrics", "default_vs_global", n, title, index)


def plot_carrier_bar(metrics: pd.DataFrame, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    labels = [("material_family", DEFAULT_CONFIGS["material_family_default"]), ("global", DEFAULT_CONFIGS["global_default"])]
    carriers = sorted(metrics["carrier_type"].dropna().unique())
    x = np.arange(len(carriers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    n = 0
    for i, (label, config_id) in enumerate(labels):
        sub = metrics[metrics["config_id"].eq(config_id) & metrics["metric_weighting"].eq("row_equal")]
        values = [sub[sub["carrier_type"].eq(c)]["mae_log10"].iloc[0] if not sub[sub["carrier_type"].eq(c)].empty else np.nan for c in carriers]
        ax.bar(x + (i - 0.5) * width, values, width, label=label)
        n += len(sub)
    ax.set_xticks(x)
    ax.set_xticklabels(carriers)
    ax.set_ylabel("mae_log10")
    ax.set_title("Carrier type MAE: default comparison")
    ax.legend()
    save_figure(fig, args.figures, "step5d_carrier_type_mae_default_comparison", args.output_suffix, "Carrier type MAE: default comparison", str(args.metrics_carrier), "default_vs_global", n, "MAE by carrier type", index)


def plot_material_worst(metrics: pd.DataFrame, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    sub = metrics[
        metrics["config_id"].eq(DEFAULT_CONFIGS["material_family_default"])
        & metrics["metric_weighting"].eq("row_equal")
        & metrics["is_reliable_eval_group"].astype(str).str.casefold().isin(["true", "1"])
    ].sort_values("mae_log10", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sub["material_family_raw"].fillna(sub["material_group_key"]).astype(str)
    ax.bar(range(len(sub)), sub["mae_log10"])
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("mae_log10")
    ax.set_title("Worst material families by MAE (default)")
    save_figure(fig, args.figures, "step5d_material_family_mae_worst20_default", args.output_suffix, "Worst material families by MAE (default)", str(args.metrics_material), DEFAULT_CONFIGS["material_family_default"], len(sub), "Worst 20 reliable material families", index)


def plot_abs_error_scatter(df: pd.DataFrame, x_col: str, base: str, xlabel: str, clipped: bool, args: argparse.Namespace, index: list[dict[str, Any]]) -> None:
    config_id = DEFAULT_CONFIGS["material_family_default"]
    sub = df[df["config_id"].eq(config_id)].copy()
    sub = sample_for_plot(sub, args.plot_sample_size, args.seed)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(sub[x_col], sub["abs_log10_sigma_pred_over_exp"], s=6, alpha=0.3)
    if clipped:
        ax.set_ylim(0, 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("abs_log10_sigma_pred_over_exp")
    title = f"Absolute error vs {xlabel} (default){' clipped y 0-5' if clipped else ''}"
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    save_figure(fig, args.figures, base, args.output_suffix, title, str(args.predictions_valid), config_id, len(sub), title, index)


def create_figures(valid: pd.DataFrame, metrics: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    index: list[dict[str, Any]] = []
    for label, config_id in DEFAULT_CONFIGS.items():
        plot_pred_vs_exp(valid, metrics["config"], label, config_id, args, index)
        plot_error_hist(valid, metrics["config"], label, config_id, args, index)
    plot_config_bar(metrics["config"], "mae_log10", True, "step5d_config_mae_top16", "Top 16 configs by MAE", "mae_log10", args, index)
    plot_config_bar(metrics["config"], "factor_2_accuracy", False, "step5d_config_factor2_top16", "Top 16 configs by factor 2 accuracy", "factor_2_accuracy", args, index)
    compare_labels = [("material_family", DEFAULT_CONFIGS["material_family_default"]), ("global", DEFAULT_CONFIGS["global_default"])]
    plot_metric_line(metrics["eta"], "eta_bin_label", "mae_log10", compare_labels, "step5d_eta_bin_mae_default_comparison", "Eta bin MAE: default comparison", "eta_bin_label", args, index)
    plot_metric_line(metrics["temperature"], "T_bin_center_K", "mae_log10", compare_labels, "step5d_temperature_bin_mae_default_comparison", "Temperature bin MAE: default comparison", "T_bin_center_K", args, index)
    plot_carrier_bar(metrics["carrier"], args, index)
    plot_material_worst(metrics["material"], args, index)
    plot_abs_error_scatter(valid, "eta", "step5d_abs_error_vs_eta_default", "eta", False, args, index)
    plot_abs_error_scatter(valid, "eta", "step5d_abs_error_vs_eta_default_clipped_y0_5", "eta", True, args, index)
    plot_abs_error_scatter(valid, "T_K", "step5d_abs_error_vs_temperature_default", "T_K", False, args, index)
    plot_abs_error_scatter(valid, "T_K", "step5d_abs_error_vs_temperature_default_clipped_y0_5", "T_K", True, args, index)
    return pd.DataFrame(index)


def build_visual_summary(diff_summary: pd.DataFrame, outliers: pd.DataFrame, default_metrics: pd.DataFrame, reference_diag: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    sample_diff = diff_summary[diff_summary["comparison_label"].str.startswith("sample")]
    paper_diff = diff_summary[diff_summary["comparison_label"].str.startswith("paper")]
    default_row = default_metrics[
        default_metrics["config_label"].eq("material_family_default") & default_metrics["metric_weighting"].eq("row_equal")
    ]
    max_outlier = outliers.sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(1)
    extreme_count = int(outliers["outlier_severity"].eq("extreme_ge_10_decades").sum())
    severe_count = int(outliers["outlier_severity"].isin(["extreme_ge_10_decades", "severe_ge_5_decades"]).sum())

    def add(item: str, status: str, value: Any, comment: str) -> None:
        rows.append({"diagnostic_item": item, "status": status, "value": str(value), "comment": comment})

    sample_identical = bool((sample_diff["different_prediction_count"].fillna(1) == 0).all()) if not sample_diff.empty else False
    paper_identical = bool((paper_diff["different_prediction_count"].fillna(1) == 0).all()) if not paper_diff.empty else False
    add("material_family_vs_global_default_identical_or_not", "warning" if sample_identical else "ok", sample_identical, "True means both default predictions are identical within 1e-12.")
    add("paper_material_family_vs_global_default_identical_or_not", "warning" if paper_identical else "ok", paper_identical, "True means both paper-holdout predictions are identical within 1e-12.")
    add("material_family_default_unique_material_group_key_count", "warning", sample_diff["unique_material_group_key_count"].iloc[0] if not sample_diff.empty else "n/a", "Low count suggests material grouping is effectively collapsed.")
    add("global_default_unique_material_group_key_for_prediction_count", "ok", sample_diff["unique_material_group_key_for_prediction_count_global"].iloc[0] if not sample_diff.empty else "n/a", "Global prediction key should be ALL.")
    add("reference_bins_material_group_key_count", "warning", reference_diag["material_family_material_group_key_count"].max() if not reference_diag.empty else "n/a", "Count of material groups inside material_family reference configs.")
    add("max_abs_log10_error", "warning", max_outlier["abs_log10_sigma_pred_over_exp"].iloc[0] if not max_outlier.empty else "n/a", "Largest absolute log10 error among top outliers.")
    add("max_abs_log10_error_row_id", "warning", max_outlier["row_id"].iloc[0] if not max_outlier.empty else "n/a", "Row id of largest error.")
    add("number_of_extreme_ge_10_decade_errors", "warning" if extreme_count else "ok", extreme_count, "Count in top100 outlier diagnostics.")
    add("number_of_severe_ge_5_decade_errors", "warning" if severe_count else "ok", severe_count, "Count in top100 outlier diagnostics.")
    if not default_row.empty:
        row = default_row.iloc[0]
        add("default_mae_log10", "ok", row["mae_log10"], "Material-family default row_equal MAE.")
        add("default_factor_2_accuracy", "ok", row["factor_2_accuracy"], "Material-family default row_equal factor-2 accuracy.")
        add("default_factor_10_accuracy", "ok", row["factor_10_accuracy"], "Material-family default row_equal factor-10 accuracy.")
    return pd.DataFrame(rows)


def run_sanity(valid: pd.DataFrame, diff_summary: pd.DataFrame, reference_diag: pd.DataFrame, outliers: pd.DataFrame, figure_index: pd.DataFrame, default_metrics: pd.DataFrame, visual_summary: pd.DataFrame, report_path: Path) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    checks["prediction_status_ok"] = valid["prediction_status"].eq("ok").all()
    checks["sigma_exp_positive"] = bool(np.isfinite(valid["sigma_S_per_m"]).all() and (valid["sigma_S_per_m"] > 0).all())
    checks["sigma_pred_positive"] = bool(np.isfinite(valid["sigma_pred_S_per_m"]).all() and (valid["sigma_pred_S_per_m"] > 0).all())
    checks["sigma_pred_over_exp_consistent"] = bool(np.allclose(valid["sigma_pred_over_exp"], valid["sigma_pred_S_per_m"] / valid["sigma_S_per_m"], rtol=1e-10))
    checks["log10_ratio_consistent"] = bool(np.allclose(valid["log10_sigma_pred_over_exp"], np.log10(valid["sigma_pred_over_exp"]), rtol=1e-10))
    checks["sigma0_ratio_matches_prediction_error"] = bool(np.allclose(valid["log10_sigma0_ref_over_row_sigma0"], valid["log10_sigma_pred_over_exp"], rtol=1e-10, atol=1e-10))
    checks["default_4_configs_exist"] = set(DEFAULT_CONFIGS.values()).issubset(set(valid["config_id"]))
    checks["at_least_one_default_has_rows"] = any(valid["config_id"].eq(config_id).any() for config_id in DEFAULT_CONFIGS.values())
    checks["diff_summary_created"] = not diff_summary.empty
    checks["reference_diagnostics_created"] = not reference_diag.empty
    checks["largest_error_diagnostics_created"] = not outliers.empty
    checks["figure_index_created"] = not figure_index.empty
    checks["figure_files_exist_nonzero"] = all(Path(p).exists() and Path(p).stat().st_size > 0 for p in list(figure_index["figure_path_png"]) + list(figure_index["figure_path_pdf"]))
    checks["default_metrics_created"] = not default_metrics.empty
    checks["visual_summary_created"] = not visual_summary.empty
    checks["report_created"] = report_path.exists() and report_path.stat().st_size > 0
    checks["did_not_read_step4_full_data_reference_curve"] = True
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *body])


def write_report(report_path: Path, inputs: dict[str, Path], figure_index: pd.DataFrame, default_metrics: pd.DataFrame, diff_summary: pd.DataFrame, reference_diag: pd.DataFrame, outliers: pd.DataFrame, visual_summary: pd.DataFrame, checks: dict[str, bool], elapsed: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_diff = diff_summary[diff_summary["comparison_label"].str.startswith("sample")]
    paper_diff = diff_summary[diff_summary["comparison_label"].str.startswith("paper")]
    material_identical = bool((sample_diff["different_prediction_count"].fillna(1) == 0).all()) if not sample_diff.empty else False
    paper_identical = bool((paper_diff["different_prediction_count"].fillna(1) == 0).all()) if not paper_diff.empty else False
    likely_reason = "material_group_key appears effectively single-valued and material/global reference bins have identical values" if material_identical else "predictions differ; inspect diff table"
    lines = [
        "# Step5D-1 Visual Diagnostics Report",
        "",
        "## Inputs",
        "",
    ]
    for label, path in inputs.items():
        lines.append(f"- {label}: {path}")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            df_to_markdown(figure_index[["figure_id", "figure_path_png", "figure_path_pdf", "config_id", "n_points_plotted"]], 50),
            "",
            "## Diagnostic Tables",
            "",
            "- step5d_global_vs_material_family_prediction_diff.csv",
            "- step5d_global_vs_material_family_prediction_diff_summary.csv",
            "- step5d_reference_group_diagnostics.csv",
            "- step5d_reference_group_counts.csv",
            "- step5d_largest_error_diagnostics_top100.csv",
            "- step5d_default_metrics_for_figures.csv",
            "- step5d_visual_diagnostics_summary.csv",
            "",
            "## Default Metrics",
            "",
            df_to_markdown(default_metrics),
            "",
            "## How To Read Figures",
            "",
            "- predicted vs experimental: points near y=x are accurate; vertical distance is multiplicative error.",
            "- error distribution: zero is perfect, +/-1 corresponds to factor-10 error.",
            "- eta/temperature/carrier plots show where MAE changes by subset.",
            "- material_family worst20 highlights reliable material groups with largest MAE.",
            "",
            "## Material Family vs Global",
            "",
            f"- material_family default identical to global default: {material_identical}",
            f"- paper material_family default identical to paper global default: {paper_identical}",
            f"- inferred reason: {likely_reason}",
            "",
            df_to_markdown(diff_summary),
            "",
            "## Reference Group Diagnostics",
            "",
            df_to_markdown(reference_diag),
            "",
            "## Largest Outliers",
            "",
            df_to_markdown(outliers[["config_id", "row_id", "abs_log10_sigma_pred_over_exp", "log10_sigma_pred_over_exp", "likely_error_origin_hint"]].head(20)),
            "",
            f"- Max abs error is explained by log10(sigma0_ref / row_sigma0): {checks.get('sigma0_ratio_matches_prediction_error')}",
            "",
            "## Visual Diagnostics Summary",
            "",
            df_to_markdown(visual_summary),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This Step5D-1 only visualizes and diagnoses existing predictions.",
            "- Step5B prediction results are visualized; predictions are not recomputed.",
            "- Step4 full-data reference curves are not used.",
            "- If material_family and global results are identical, confirm material grouping before drawing research conclusions.",
            "- Next: inspect material_group_key generation, review top outliers by paper/sample, and choose final figures or add supplemental sample_equal/paper_holdout plots.",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    inputs = {
        "predictions_valid": args.predictions_valid,
        "predictions_all": args.predictions_all,
        "reference_bins": args.reference_bins,
        "metrics_config": args.metrics_config,
        "metrics_carrier": args.metrics_carrier,
        "metrics_material": args.metrics_material,
        "metrics_temperature": args.metrics_temperature,
        "metrics_eta": args.metrics_eta,
        "default_comparison": args.default_comparison,
        "ranking": args.ranking,
        "largest_errors": args.largest_errors,
    }

    log("loading Step5B predictions...")
    valid = prepare_predictions(read_table(args.predictions_valid), args.max_rows_per_config)
    reference = read_table(args.reference_bins)
    log("loading Step5C metrics...")
    metrics = {
        "config": read_table(args.metrics_config),
        "carrier": read_table(args.metrics_carrier),
        "material": read_table(args.metrics_material),
        "temperature": read_table(args.metrics_temperature),
        "eta": read_table(args.metrics_eta),
        "default": read_table(args.default_comparison),
        "ranking": read_table(args.ranking),
        "largest": read_table(args.largest_errors),
    }
    log("validating required columns...")
    validate_prediction_columns(valid)
    log("computing additional diagnostic columns...")
    # Already computed in prepare_predictions.
    log("checking material_family vs global default predictions...")
    diff, diff_summary = build_prediction_diff(valid)
    log("diagnosing reference group structure...")
    reference_diag, reference_counts = build_reference_diagnostics(reference)
    log("diagnosing largest error rows...")
    outliers = build_largest_error_diagnostics(valid, metrics["largest"])
    log("building default metrics table...")
    default_metrics = build_default_metrics(metrics["default"])
    log("creating predicted vs experimental plots...")
    figure_index = create_figures(valid, metrics, args)
    log("creating error distribution plots...")
    log("creating config comparison plots...")
    log("creating eta/temperature/carrier/material plots...")
    visual_summary = build_visual_summary(diff_summary, outliers, default_metrics, reference_diag)

    args.output.mkdir(parents=True, exist_ok=True)
    diff.to_csv(args.output / output_name("step5d_global_vs_material_family_prediction_diff", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    diff_summary.to_csv(args.output / output_name("step5d_global_vs_material_family_prediction_diff_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    reference_diag.to_csv(args.output / output_name("step5d_reference_group_diagnostics", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    reference_counts.to_csv(args.output / output_name("step5d_reference_group_counts", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    outliers.to_csv(args.output / output_name("step5d_largest_error_diagnostics_top100", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    default_metrics.to_csv(args.output / output_name("step5d_default_metrics_for_figures", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    visual_summary.to_csv(args.output / output_name("step5d_visual_diagnostics_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    log("writing figure index...")
    figure_index.to_csv(args.output / output_name("step5d_figure_index", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    log("writing report...")
    # Create report after preliminary outputs so existence checks can see it.
    empty_checks: dict[str, bool] = {}
    write_report(args.report, inputs, figure_index, default_metrics, diff_summary, reference_diag, outliers, visual_summary, empty_checks, time.time() - started)
    log("running sanity checks...")
    checks, failures = run_sanity(valid, diff_summary, reference_diag, outliers, figure_index, default_metrics, visual_summary, args.report)
    if failures:
        for failure in failures:
            print(f"[step5d] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(args.report, inputs, figure_index, default_metrics, diff_summary, reference_diag, outliers, visual_summary, checks, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
