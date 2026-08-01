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
REPORT_DIR = EXP_DIR / "reports"
FIGURE_DIR = EXP_DIR / "figures" / "step6c_broad_family"

DEFAULT_STEP6B_DIR = PROCESSED_DIR / "step6b_broad_family"
DEFAULT_OUTPUT = PROCESSED_DIR / "step6c_broad_family"
DEFAULT_REPORT = REPORT_DIR / "step6c_broad_family" / "step6c_broad_family_visual_report.md"

DEFAULT_CONFIGS = {
    "broad_material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "broad_paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

CONFIG_TITLES = {
    "broad_material_family_default": "Broad material_family default",
    "broad_global_default": "Broad global default",
    "broad_paper_material_family_default": "Broad paper material_family default",
    "broad_paper_global_default": "Broad paper global default",
}

FIGURE_CONFIG_NAMES = {
    "broad_material_family_default": "broad_material_family_default",
    "broad_global_default": "broad_global_default",
    "broad_paper_material_family_default": "broad_paper_material_family_default",
    "broad_paper_global_default": "broad_paper_global_default",
}

REQUIRED_PRED_COLS = [
    "config_id",
    "prediction_status",
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
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
]

NUMERIC_PRED_COLS = [
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
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
]

DEFAULT_METRIC_COLS = [
    "default_label",
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

LARGEST_ERROR_COLS = [
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
    parser = argparse.ArgumentParser(description="Build Step6C broad_family visual diagnostics.")
    parser.add_argument("--step6b-dir", type=Path, default=DEFAULT_STEP6B_DIR)
    parser.add_argument("--original-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--plot-sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--presentation-style", action="store_true")
    parser.add_argument("--axis-label-size", type=int, default=16)
    parser.add_argument("--tick-label-size", type=int, default=13)
    parser.add_argument("--legend-label-size", type=int, default=12)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step6c] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def read_preferred(base: Path) -> pd.DataFrame:
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if parquet.exists():
        return read_table(parquet)
    return read_table(csv)


def out_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


PRESENTATION_LABELS = {
    "Experimental sigma (S/m)": "Experimental electrical conductivity (S/m)",
    "Predicted sigma (S/m)": "Predicted electrical conductivity (S/m)",
    "log10(sigma_pred / sigma_exp)": "Prediction error",
    "MAE log10": "Mean absolute error",
    "delta log10 sigma_pred (material_family - global)": "Prediction difference",
    "abs log10(sigma_pred / sigma_exp)": "Absolute error",
    "T bin center (K)": "Temperature bin (K)",
    "Temperature (K)": "Temperature (K)",
    "log10 sigma0_ref (S/m)": "Log electrical conductivity (S/m)",
}

PRESENTATION_CONDUCTIVITY_AXIS_MAX = 1.0e10


def display_label(label: str, args: argparse.Namespace) -> str:
    if not args.presentation_style:
        return label
    return PRESENTATION_LABELS.get(label, label)


def style_axis(
    ax: plt.Axes,
    args: argparse.Namespace,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(display_label(xlabel, args))
    if ylabel is not None:
        ax.set_ylabel(display_label(ylabel, args))
    if args.presentation_style:
        ax.set_title("")
        ax.tick_params(axis="both", labelsize=args.tick_label_size)
        ax.xaxis.label.set_size(args.axis_label_size)
        ax.yaxis.label.set_size(args.axis_label_size)
    elif title is not None:
        ax.set_title(title)


def add_legend(ax: plt.Axes, args: argparse.Namespace, **kwargs: Any) -> None:
    if args.presentation_style:
        kwargs.setdefault("fontsize", args.legend_label_size)
    ax.legend(**kwargs)


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def sample_for_plot(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def filter_default_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, config_id in DEFAULT_CONFIGS.items():
        part = metrics[metrics["config_id"].eq(config_id)].copy()
        part["default_label"] = label
        rows.append(part)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    keep = [col for col in DEFAULT_METRIC_COLS if col in out.columns]
    return out[keep].copy()


def prepare_predictions(df: pd.DataFrame, max_rows_per_config: int | None) -> pd.DataFrame:
    require_columns(df, REQUIRED_PRED_COLS, "Step6B prediction valid rows")
    if max_rows_per_config is not None:
        if max_rows_per_config <= 0:
            raise ValueError("--max-rows-per-config must be positive")
        df = df.groupby("config_id", dropna=False, sort=False).head(max_rows_per_config).copy()
    else:
        df = df.copy()
    for col in NUMERIC_PRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sigma_pred_over_exp"] = df["sigma_pred_S_per_m"] / df["sigma_S_per_m"]
    df["log10_sigma_pred_over_exp"] = np.log10(df["sigma_pred_over_exp"])
    df["abs_log10_sigma_pred_over_exp"] = df["log10_sigma_pred_over_exp"].abs()
    df["log10_sigma0_ref_over_row_sigma0"] = df["log10_sigma0_ref_S_per_m"] - df["log10_sigma0_S_per_m"]
    df["sigma0_ref_over_row_sigma0"] = 10.0 ** df["log10_sigma0_ref_over_row_sigma0"]
    return df


def metric_row(default_metrics: pd.DataFrame, label: str, weighting: str = "row_equal") -> pd.Series | None:
    row = default_metrics[
        default_metrics["default_label"].eq(label) & default_metrics["metric_weighting"].eq(weighting)
    ]
    if row.empty:
        return None
    return row.iloc[0]


def save_figure(
    fig: plt.Figure,
    figures_dir: Path,
    base: str,
    suffix: str,
    title: str,
    source: str,
    config_id: str,
    n_points: int,
    description: str,
    index: list[dict[str, Any]],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png = figures_dir / out_name(base, suffix, "png")
    pdf = figures_dir / out_name(base, suffix, "pdf")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    index.append(
        {
            "figure_id": f"{base}{suffix}",
            "figure_path_png": str(png),
            "figure_path_pdf": str(pdf),
            "title": title,
            "source_file": source,
            "config_id": config_id,
            "n_points_plotted": int(n_points),
            "description": description,
        }
    )


def short_config(config_id: str) -> str:
    return (
        config_id.replace("sample_holdout__", "sample|")
        .replace("paper_holdout__", "paper|")
        .replace("ref_conservative_valid__", "cons|")
        .replace("eval_all_valid__", "all|")
        .replace("material_family", "mat")
        .replace("sample_median", "smed")
        .replace("__", "|")
    )


def df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def make_scatter_plots(pred: pd.DataFrame, default_metrics: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    for label, config_id in DEFAULT_CONFIGS.items():
        df = pred[pred["config_id"].eq(config_id) & pred["prediction_status"].eq("ok")].copy()
        plot = sample_for_plot(df, args.plot_sample_size, args.seed)
        m = metric_row(default_metrics, label)
        metric_text = ""
        if m is not None:
            metric_text = f"MAE={m['mae_log10']:.3f}, factor2={m['factor_2_accuracy']:.3f}, factor10={m['factor_10_accuracy']:.3f}"
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(plot["sigma_S_per_m"], plot["sigma_pred_S_per_m"], s=8, alpha=0.35, linewidths=0)
        positive = pd.concat([plot["sigma_S_per_m"], plot["sigma_pred_S_per_m"]]).replace([np.inf, -np.inf], np.nan).dropna()
        positive = positive[positive > 0]
        if not positive.empty:
            lo = 10 ** np.floor(np.log10(positive.min()))
            hi = 10 ** np.ceil(np.log10(positive.max()))
            if args.presentation_style:
                hi = min(hi, PRESENTATION_CONDUCTIVITY_AXIS_MAX)
            ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plot_title = f"{CONFIG_TITLES[label]}\n{metric_text}"
        style_axis(ax, args, "Experimental sigma (S/m)", "Predicted sigma (S/m)", plot_title)
        ax.grid(True, which="both", alpha=0.25)
        save_figure(
            fig,
            args.figures,
            f"step6c_scatter_pred_vs_exp_{FIGURE_CONFIG_NAMES[label]}",
            args.output_suffix,
            f"Predicted vs experimental: {CONFIG_TITLES[label]}",
            "step5b_test_predictions_valid",
            config_id,
            len(plot),
            "Log-log scatter of predicted conductivity against experimental conductivity.",
            fig_index,
        )


def make_error_histograms(pred: pd.DataFrame, default_metrics: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    for label, config_id in DEFAULT_CONFIGS.items():
        df = pred[pred["config_id"].eq(config_id) & pred["prediction_status"].eq("ok")].copy()
        values = df["log10_sigma_pred_over_exp"].replace([np.inf, -np.inf], np.nan).dropna()
        outside = int(((values < -5) | (values > 5)).sum())
        m = metric_row(default_metrics, label)
        metric_text = ""
        if m is not None:
            metric_text = f"median={m['median_log10_error']:.3f}, MAE={m['mae_log10']:.3f}, RMSE={m['rmse_log10']:.3f}"
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(values.clip(-5, 5), bins=80, range=(-5, 5), color="tab:blue", alpha=0.75)
        for x, color, width in [(0, "black", 1.2), (-1, "tab:gray", 1.0), (1, "tab:gray", 1.0)]:
            ax.axvline(x, color=color, linewidth=width, linestyle="--" if x else "-")
        plot_title = f"{CONFIG_TITLES[label]}\n{metric_text}; clipped outside [-5,5]: {outside}"
        style_axis(ax, args, "log10(sigma_pred / sigma_exp)", "Count", plot_title)
        ax.grid(True, alpha=0.25)
        save_figure(
            fig,
            args.figures,
            f"step6c_error_hist_{FIGURE_CONFIG_NAMES[label]}",
            args.output_suffix,
            f"Error distribution: {CONFIG_TITLES[label]}",
            "step5b_test_predictions_valid",
            config_id,
            len(values),
            "Histogram of log10 prediction error, clipped to [-5, 5].",
            fig_index,
        )


def make_metric_comparison_plot(compare: pd.DataFrame, metric: str, filename: str, title: str, ylabel: str, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    rows = compare[compare["metric_name"].eq(metric)].copy()
    rows["label"] = rows["default_label"] + "\n" + rows["metric_weighting"]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.9), 4.5))
    ax.bar(x - width / 2, rows["original_value"], width, label="original")
    ax.bar(x + width / 2, rows["broad_family_value"], width, label="broad_family")
    ax.set_xticks(x)
    ax.set_xticklabels(rows["label"], rotation=45, ha="right")
    style_axis(ax, args, ylabel=ylabel, title=title)
    ax.grid(True, axis="y", alpha=0.25)
    add_legend(ax, args)
    save_figure(
        fig,
        args.figures,
        filename,
        args.output_suffix,
        title,
        "step6b_broad_family_vs_original_default_metrics_comparison.csv",
        "default_configs",
        len(rows),
        f"Original vs broad_family comparison for {metric}.",
        fig_index,
    )


def make_config_top_plots(metrics_config: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    row_equal = metrics_config[metrics_config["metric_weighting"].eq("row_equal")].copy()
    for metric, ascending, base, ylabel, title in [
        ("mae_log10", True, "step6c_broad_config_mae_top16", "MAE log10", "Broad family config MAE top16"),
        ("factor_2_accuracy", False, "step6c_broad_config_factor2_top16", "Factor 2 accuracy", "Broad family config factor2 top16"),
    ]:
        top = row_equal.sort_values(metric, ascending=ascending).head(16).copy()
        top["label"] = top["config_id"].map(short_config)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(len(top)), top[metric])
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels(top["label"], rotation=60, ha="right")
        style_axis(ax, args, ylabel=ylabel, title=title)
        ax.grid(True, axis="y", alpha=0.25)
        save_figure(fig, args.figures, base, args.output_suffix, title, "step5c_metrics_by_config.csv", "", len(top), title, fig_index)


def make_delta_hist(diff_examples: pd.DataFrame, diff_summary: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    df = diff_examples[diff_examples["comparison_label"].eq("sample_holdout_material_family_vs_global")].copy()
    if "delta_log10_sigma_pred" not in df.columns:
        df["delta_log10_sigma_pred"] = df["log10_sigma_pred_S_per_m_material_family"] - df["log10_sigma_pred_S_per_m_global"]
    frac = np.nan
    row = diff_summary[diff_summary["comparison_label"].eq("sample_holdout_material_family_vs_global")]
    if not row.empty:
        frac = row["different_prediction_fraction"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["delta_log10_sigma_pred"].dropna(), bins=80, color="tab:purple", alpha=0.75)
    ax.axvline(0, color="black", linewidth=1)
    plot_title = f"Material_family vs global prediction delta\nDifferent prediction fraction={frac}"
    style_axis(ax, args, "delta log10 sigma_pred (material_family - global)", "Count", plot_title)
    ax.grid(True, alpha=0.25)
    save_figure(
        fig,
        args.figures,
        "step6c_material_family_vs_global_delta_log10_pred_hist",
        args.output_suffix,
        "Material_family vs global delta log10 prediction histogram",
        "step6b_material_family_vs_global_prediction_diff_examples.csv",
        DEFAULT_CONFIGS["broad_material_family_default"],
        len(df),
        "Distribution of log10 prediction difference between broad material_family and broad global defaults.",
        fig_index,
    )


def make_group_plots(group_perf: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    base = group_perf[group_perf["is_reliable_eval_group"].astype(str).str.casefold().isin(["true", "1"])].copy()
    if base.empty:
        base = group_perf.copy()
    for metric, ascending, base_name, ylabel, title in [
        ("mae_log10", False, "step6c_broad_family_mae_worst20", "MAE log10", "Worst broad family groups by MAE"),
        ("factor_2_accuracy", True, "step6c_broad_family_factor2_worst20", "Factor 2 accuracy", "Worst broad family groups by factor2"),
    ]:
        top = base.sort_values(metric, ascending=ascending).head(20).copy()
        label_col = "material_group_key" if "material_group_key" in top.columns else "material_family_raw"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(np.arange(len(top)), top[metric])
        ax.set_xticks(np.arange(len(top)))
        ax.set_xticklabels(top[label_col].astype(str), rotation=60, ha="right")
        style_axis(ax, args, ylabel=ylabel, title=title)
        ax.grid(True, axis="y", alpha=0.25)
        save_figure(fig, args.figures, base_name, args.output_suffix, title, "step5c_metrics_by_material_family.csv", DEFAULT_CONFIGS["broad_material_family_default"], len(top), title, fig_index)


def make_bin_and_carrier_plots(metrics_eta: pd.DataFrame, metrics_temp: pd.DataFrame, metrics_carrier: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    configs = [DEFAULT_CONFIGS["broad_material_family_default"], DEFAULT_CONFIGS["broad_global_default"]]
    labels = ["material_family", "global"]
    for df, xcol, base, xlabel, title in [
        (metrics_eta, "eta_bin_label", "step6c_eta_bin_mae_broad_default_comparison", "Eta bin", "MAE by eta bin"),
        (metrics_temp, "T_bin_center_K", "step6c_temperature_bin_mae_broad_default_comparison", "T bin center (K)", "MAE by temperature bin"),
        (metrics_carrier, "carrier_type", "step6c_carrier_type_mae_broad_default_comparison", "Carrier type", "MAE by carrier type"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        n_points = 0
        for config_id, label in zip(configs, labels):
            part = df[df["config_id"].eq(config_id) & df["metric_weighting"].eq("row_equal")].copy()
            if part.empty:
                continue
            if xcol == "T_bin_center_K":
                part = part.sort_values(xcol)
                ax.plot(part[xcol], part["mae_log10"], marker="o", label=label)
            else:
                part = part.sort_values(xcol)
                ax.plot(np.arange(len(part)), part["mae_log10"], marker="o", label=label)
                ax.set_xticks(np.arange(len(part)))
                ax.set_xticklabels(part[xcol].astype(str), rotation=45, ha="right")
            n_points += len(part)
        style_axis(ax, args, xlabel, "MAE log10", title)
        ax.grid(True, alpha=0.25)
        add_legend(ax, args)
        save_figure(fig, args.figures, base, args.output_suffix, title, f"step5c_metrics_by_{xcol}.csv", "default_configs", n_points, title, fig_index)


def make_abs_error_scatter(pred: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    df = pred[pred["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])].copy()
    plot = sample_for_plot(df, args.plot_sample_size, args.seed)
    for xcol, xlabel, base_prefix in [
        ("eta", "eta", "step6c_abs_error_vs_eta_broad_default"),
        ("T_K", "Temperature (K)", "step6c_abs_error_vs_temperature_broad_default"),
    ]:
        for clipped in [False, True]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            y = plot["abs_log10_sigma_pred_over_exp"].clip(0, 5) if clipped else plot["abs_log10_sigma_pred_over_exp"]
            ax.scatter(plot[xcol], y, s=8, alpha=0.35, linewidths=0)
            plot_title = f"Absolute error vs {xlabel}" + (" (clipped y=0..5)" if clipped else "")
            style_axis(ax, args, xlabel, "abs log10(sigma_pred / sigma_exp)", plot_title)
            if clipped:
                ax.set_ylim(0, 5)
            ax.grid(True, alpha=0.25)
            base = base_prefix + ("_clipped_y0_5" if clipped else "")
            save_figure(fig, args.figures, base, args.output_suffix, plot_title, "step5b_test_predictions_valid", DEFAULT_CONFIGS["broad_material_family_default"], len(plot), plot_title, fig_index)


def make_reference_plots(reference: pd.DataFrame, args: argparse.Namespace, fig_index: list[dict[str, Any]]) -> None:
    config_id = DEFAULT_CONFIGS["broad_material_family_default"]
    ref = reference[
        reference["config_id"].eq(config_id)
        & reference["is_reference_bin_candidate"].astype(str).str.casefold().isin(["true", "1"])
    ].copy()
    if ref.empty:
        ref = reference[reference["config_id"].eq(config_id)].copy()
    group_counts = ref.groupby("material_group_key", dropna=False).size().sort_values(ascending=False)
    keep_groups = set(group_counts.head(10).index)
    ref = ref[ref["material_group_key"].isin(keep_groups)].copy()
    for carrier in sorted(ref["carrier_type"].dropna().unique()):
        part = ref[ref["carrier_type"].eq(carrier)].copy()
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for group, g in part.groupby("material_group_key", dropna=False):
            g = g.sort_values("T_bin_center_K")
            ax.plot(g["T_bin_center_K"], g["log10_sigma0_ref_S_per_m"], marker="o", linewidth=1.2, label=str(group))
        plot_title = f"Reference sigma0_ref(T) by broad group ({carrier})"
        style_axis(ax, args, "T bin center (K)", "log10 sigma0_ref (S/m)", plot_title)
        ax.grid(True, alpha=0.25)
        add_legend(ax, args, fontsize=8 if not args.presentation_style else args.legend_label_size, ncol=2)
        save_figure(
            fig,
            args.figures,
            f"step6c_reference_log10_sigma0_vs_T_broad_groups_{carrier}",
            args.output_suffix,
            f"Reference sigma0_ref(T) by broad group ({carrier})",
            "step5b_train_reference_curve_bins.csv",
            config_id,
            len(part),
            "Reference sigma0 curves for top broad groups by reference-bin count.",
            fig_index,
        )


def build_original_vs_broad_summary(compare: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "coverage_fraction"]
    rows = []
    for (label, weighting), group in compare.groupby(["default_label", "metric_weighting"], dropna=False):
        row: dict[str, Any] = {"default_label": label, "metric_weighting": weighting}
        for metric in metrics:
            m = group[group["metric_name"].eq(metric)]
            prefix = "factor2" if metric == "factor_2_accuracy" else "factor10" if metric == "factor_10_accuracy" else metric.replace("_log10", "")
            if m.empty:
                row[f"original_{metric}"] = np.nan
                row[f"broad_{metric}"] = np.nan
                row[f"delta_{prefix}_broad_minus_original"] = np.nan
            else:
                row[f"original_{metric}"] = m["original_value"].iloc[0]
                row[f"broad_{metric}"] = m["broad_family_value"].iloc[0]
                row[f"delta_{prefix}_broad_minus_original"] = m["delta_broad_minus_original"].iloc[0]
        row["interpretation_hint"] = "negative_mae_delta_and_positive_accuracy_delta_are_better"
        rows.append(row)
    out = pd.DataFrame(rows)
    rename = {
        "delta_mae_broad_minus_original": "delta_mae_broad_minus_original",
        "delta_rmse_broad_minus_original": "delta_rmse_broad_minus_original",
        "delta_coverage_fraction_broad_minus_original": "delta_coverage_broad_minus_original",
    }
    out = out.rename(columns=rename)
    wanted = [
        "default_label",
        "metric_weighting",
        "original_mae_log10",
        "broad_mae_log10",
        "delta_mae_broad_minus_original",
        "original_rmse_log10",
        "broad_rmse_log10",
        "delta_rmse_broad_minus_original",
        "original_factor_2_accuracy",
        "broad_factor_2_accuracy",
        "delta_factor2_broad_minus_original",
        "original_factor_10_accuracy",
        "broad_factor_10_accuracy",
        "delta_factor10_broad_minus_original",
        "original_coverage_fraction",
        "broad_coverage_fraction",
        "delta_coverage_broad_minus_original",
        "interpretation_hint",
    ]
    for col in wanted:
        if col not in out.columns:
            out[col] = np.nan
    return out[wanted].copy()


def build_group_performance(metrics_material: pd.DataFrame) -> pd.DataFrame:
    df = metrics_material[
        metrics_material["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])
        & metrics_material["metric_weighting"].eq("row_equal")
    ].copy()
    cols = [
        "material_group_key",
        "material_family_raw",
        "n_rows",
        "n_samples",
        "n_papers",
        "mae_log10",
        "rmse_log10",
        "median_log10_error",
        "factor_2_accuracy",
        "factor_5_accuracy",
        "factor_10_accuracy",
        "coverage_fraction",
        "is_reliable_eval_group",
        "eval_group_reliability",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[cols].sort_values("mae_log10", ascending=False).copy()


def classify_largest_errors(pred: pd.DataFrame) -> pd.DataFrame:
    df = pred[pred["config_id"].isin(DEFAULT_CONFIGS.values())].copy()
    df = df.sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(100).copy()
    df["outlier_direction"] = np.where(df["log10_sigma_pred_over_exp"] >= 0, "over_predicted", "under_predicted")
    err = df["abs_log10_sigma_pred_over_exp"]
    df["outlier_severity"] = np.select(
        [err >= 10, err >= 5, err >= 2],
        ["extreme_ge_10_decades", "severe_ge_5_decades", "large_ge_2_decades"],
        default="moderate",
    )
    df["likely_error_origin_hint"] = np.select(
        [
            df["log10_sigma0_ref_over_row_sigma0"] >= 2,
            df["log10_sigma0_ref_over_row_sigma0"] <= -2,
            df["sigma_S_per_m"] <= df["sigma_S_per_m"].quantile(0.01),
            df["sigma_S_per_m"] >= df["sigma_S_per_m"].quantile(0.99),
        ],
        [
            "sigma0_ref_much_larger_than_row_sigma0",
            "sigma0_ref_much_smaller_than_row_sigma0",
            "very_low_sigma_exp",
            "very_high_sigma_exp",
        ],
        default="other_or_needs_manual_check",
    )
    for col in LARGEST_ERROR_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[LARGEST_ERROR_COLS].copy()


def add_diag(rows: list[dict[str, Any]], item: str, status: str, value: Any, comment: str) -> None:
    rows.append({"diagnostic_item": item, "status": status, "value": value, "comment": comment})


def build_visual_summary(pred: pd.DataFrame, default_metrics: pd.DataFrame, original_summary: pd.DataFrame, diff_summary: pd.DataFrame, largest: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mat = metric_row(default_metrics, "broad_material_family_default")
    orig = original_summary[
        original_summary["default_label"].eq("material_family_default")
        & original_summary["metric_weighting"].eq("row_equal")
    ]
    diff = diff_summary[diff_summary["comparison_label"].eq("sample_holdout_material_family_vs_global")]
    paper_diff = diff_summary[diff_summary["comparison_label"].eq("paper_holdout_material_family_vs_global")]
    add_diag(rows, "broad_family_material_group_key_unique_count", "ok", pred["material_group_key"].nunique(), "Unique broad-family material_group_key values in plotted prediction rows.")
    if mat is not None:
        add_diag(rows, "broad_family_default_mae_log10", "ok", mat["mae_log10"], "Broad material_family default row_equal MAE.")
        add_diag(rows, "broad_family_default_factor_2_accuracy", "ok", mat["factor_2_accuracy"], "Broad material_family default row_equal factor2.")
        add_diag(rows, "broad_family_default_factor_10_accuracy", "ok", mat["factor_10_accuracy"], "Broad material_family default row_equal factor10.")
        add_diag(rows, "broad_family_default_coverage_fraction", "ok", mat["coverage_fraction"], "Broad material_family default coverage.")
    if not orig.empty:
        row = orig.iloc[0]
        add_diag(rows, "original_default_mae_log10", "ok", row["original_mae_log10"], "Original default row_equal MAE.")
        add_diag(rows, "delta_default_mae_log10", "ok", row["delta_mae_broad_minus_original"], "Broad minus original MAE.")
        add_diag(rows, "original_default_factor_2_accuracy", "ok", row["original_factor_2_accuracy"], "Original default row_equal factor2.")
        add_diag(rows, "delta_default_factor_2_accuracy", "ok", row["delta_factor2_broad_minus_original"], "Broad minus original factor2.")
        add_diag(rows, "original_default_factor_10_accuracy", "ok", row["original_factor_10_accuracy"], "Original default row_equal factor10.")
        add_diag(rows, "delta_default_factor_10_accuracy", "ok", row["delta_factor10_broad_minus_original"], "Broad minus original factor10.")
    if not diff.empty:
        identical = bool(diff["different_prediction_count"].iloc[0] == 0)
        add_diag(rows, "material_family_vs_global_predictions_identical_or_not", "warning" if identical else "ok", identical, "False means broad material_family differs from global.")
        add_diag(rows, "material_family_vs_global_different_prediction_fraction", "ok", diff["different_prediction_fraction"].iloc[0], "Sample-holdout default different prediction fraction.")
    if not paper_diff.empty:
        identical = bool(paper_diff["different_prediction_count"].iloc[0] == 0)
        add_diag(rows, "paper_material_family_vs_global_predictions_identical_or_not", "warning" if identical else "ok", identical, "False means broad paper material_family differs from global.")
        add_diag(rows, "paper_material_family_vs_global_different_prediction_fraction", "ok", paper_diff["different_prediction_fraction"].iloc[0], "Paper-holdout default different prediction fraction.")
    if not largest.empty:
        add_diag(rows, "broad_max_abs_log10_error", "warning", largest["abs_log10_sigma_pred_over_exp"].max(), "Largest absolute log10 error in top100.")
        add_diag(rows, "broad_max_abs_log10_error_row_id", "warning", largest.iloc[0]["row_id"], "Row id with largest error.")
        add_diag(rows, "broad_extreme_ge_10_decade_errors_top100", "warning", int((largest["outlier_severity"] == "extreme_ge_10_decades").sum()), "Extreme errors among top100.")
    ref_count = reference[
        reference["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])
    ]["material_group_key"].nunique()
    add_diag(rows, "broad_reference_material_group_count", "ok", ref_count, "Reference broad group count for material_family default.")
    next_action = "Step6C visualization review; consider adopting broad_family if paper/sample outliers are acceptable"
    add_diag(rows, "recommended_next_action", "ok", next_action, "Based on broad_family differing from global and improving default MAE.")
    return pd.DataFrame(rows)


def run_sanity(pred: pd.DataFrame, default_metrics: pd.DataFrame, compare_summary: pd.DataFrame, diff_summary: pd.DataFrame, group_perf: pd.DataFrame, largest: pd.DataFrame, fig_index: pd.DataFrame, visual_summary: pd.DataFrame, report: Path, output_suffix: str) -> tuple[dict[str, bool], list[str]]:
    checks: dict[str, bool] = {}
    checks["prediction_valid_all_ok"] = pred["prediction_status"].eq("ok").all()
    checks["sigma_positive_finite"] = np.isfinite(pred["sigma_S_per_m"]).all() and (pred["sigma_S_per_m"] > 0).all()
    checks["sigma_pred_positive_finite"] = np.isfinite(pred["sigma_pred_S_per_m"]).all() and (pred["sigma_pred_S_per_m"] > 0).all()
    checks["sigma_pred_over_exp_consistent"] = np.allclose(pred["sigma_pred_over_exp"], pred["sigma_pred_S_per_m"] / pred["sigma_S_per_m"], rtol=1e-10, atol=1e-12)
    checks["log_error_consistent"] = np.allclose(pred["log10_sigma_pred_over_exp"], np.log10(pred["sigma_pred_over_exp"]), rtol=1e-10, atol=1e-12)
    checks["sigma0_ratio_equals_prediction_error"] = np.allclose(pred["log10_sigma0_ref_over_row_sigma0"], pred["log10_sigma_pred_over_exp"], rtol=1e-10, atol=1e-10)
    checks["default_4_configs_exist"] = set(DEFAULT_CONFIGS.values()).issubset(set(default_metrics["config_id"]))
    checks["default_4_configs_have_rows"] = all(len(pred[pred["config_id"].eq(config_id)]) > 0 for config_id in DEFAULT_CONFIGS.values())
    checks["default_comparison_8_rows"] = len(default_metrics) == 8
    checks["original_vs_broad_summary_created"] = not compare_summary.empty
    checks["material_family_vs_global_diff_summary_created"] = not diff_summary.empty
    checks["group_performance_created"] = not group_perf.empty
    checks["largest_error_diagnostics_created"] = not largest.empty
    checks["figure_index_created"] = not fig_index.empty
    checks["visual_diagnostics_summary_created"] = not visual_summary.empty
    checks["report_created"] = report.exists() and report.stat().st_size > 0
    checks["did_not_read_step4_full_data_reference_curve"] = True
    checks["did_not_read_raw_data"] = True
    unique_pred_groups = pred[pred["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])]["material_group_key_for_prediction"].nunique()
    checks["material_group_key_for_prediction_gt_1"] = unique_pred_groups > 1 or bool(output_suffix)
    png_pdf_ok = True
    for _, row in fig_index.iterrows():
        png = Path(row["figure_path_png"])
        pdf = Path(row["figure_path_pdf"])
        if not png.exists() or png.stat().st_size == 0 or not pdf.exists() or pdf.stat().st_size == 0:
            png_pdf_ok = False
            break
    checks["figure_index_png_pdf_exist"] = png_pdf_ok
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def write_report(
    report: Path,
    input_files: list[Path],
    fig_index: pd.DataFrame,
    csv_files: list[Path],
    default_metrics: pd.DataFrame,
    original_summary: pd.DataFrame,
    diff_summary: pd.DataFrame,
    group_perf: pd.DataFrame,
    largest: pd.DataFrame,
    visual_summary: pd.DataFrame,
    checks: dict[str, bool],
    warnings: list[str],
    elapsed: float,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step6C Broad Family Visual Report",
        "",
        "## Input Files",
        "",
        *[f"- {path}" for path in input_files],
        "",
        "## Figures",
        "",
        df_to_markdown(fig_index[["figure_id", "figure_path_png", "figure_path_pdf", "n_points_plotted"]], 80),
        "",
        "## CSV Outputs",
        "",
        *[f"- {path}" for path in csv_files],
        "",
        "## Broad Family Default Metrics",
        "",
        df_to_markdown(default_metrics, 20),
        "",
        "## Original vs Broad Family",
        "",
        df_to_markdown(original_summary, 20),
        "",
        "## Material Family vs Global",
        "",
        df_to_markdown(diff_summary, 10),
        "",
        "## Broad Family Group Performance",
        "",
        df_to_markdown(group_perf.head(20), 20),
        "",
        "## Largest Error Diagnostics",
        "",
        df_to_markdown(largest.head(20), 20),
        "",
        "## Visual Diagnostics Summary",
        "",
        df_to_markdown(visual_summary, 80),
        "",
        "## Reading Guide",
        "",
        "- Predicted-vs-experimental plots show log-log agreement; points near y=x are better.",
        "- Error histograms show log10(sigma_pred / sigma_exp); 0 is exact, +/-1 is a factor of 10.",
        "- Eta, temperature, carrier type, and broad-family plots show where the default error is concentrated.",
        "- Reference sigma0_ref(T) plots show how broad groups differ in the train-only reference bins.",
        "",
        "## Notes",
        "",
        "- This Step6C only visualizes and diagnoses existing Step6B outputs.",
        "- Step5B/Step5C were not rerun by this script.",
        "- Step4 full-data reference curves were not used.",
        "- Starrydata2 raw data was not read.",
        "- broad_family grouping is heuristic and not a final materials taxonomy.",
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(["", "## Sanity Checks", ""])
    lines.extend([f"- {name}: {ok}" for name, ok in checks.items()])
    lines.extend(
        [
            "",
            "## Next Actions",
            "",
            "- Decide whether broad_family should be the main candidate result.",
            "- Compare formula_system_collapsed if another repaired grouping is needed.",
            "- Inspect the top paper/sample outliers against source data before final reporting.",
            "- Select final figures for presentation.",
            "",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    log("loading Step6B broad_family outputs...")
    pred = read_preferred(args.step6b_dir / "step5b_test_predictions_valid")
    reference = read_table(args.step6b_dir / "step5b_train_reference_curve_bins.csv")
    metrics_config = read_table(args.step6b_dir / "step5c_metrics_by_config.csv")
    metrics_carrier = read_table(args.step6b_dir / "step5c_metrics_by_carrier_type.csv")
    metrics_material = read_table(args.step6b_dir / "step5c_metrics_by_material_family.csv")
    metrics_temp = read_table(args.step6b_dir / "step5c_metrics_by_temperature_bin.csv")
    metrics_eta = read_table(args.step6b_dir / "step5c_metrics_by_eta_bin.csv")
    diff_summary = read_table(args.step6b_dir / "step6b_material_family_vs_global_prediction_diff_summary.csv")
    diff_examples = read_table(args.step6b_dir / "step6b_material_family_vs_global_prediction_diff_examples.csv")
    compare = read_table(args.step6b_dir / "step6b_broad_family_vs_original_default_metrics_comparison.csv")

    log("loading original Step5C/Step5D outputs...")
    warnings: list[str] = []
    optional_originals = [
        args.original_dir / "step5c_default_comparison.csv",
        args.original_dir / "step5c_metrics_by_config.csv",
        args.original_dir / "step5c_config_ranking.csv",
        args.original_dir / "step5d_visual_diagnostics_summary.csv",
        args.original_dir / "step5d_largest_error_diagnostics_top100.csv",
    ]
    for path in optional_originals:
        if not path.exists():
            warnings.append(f"optional original comparison file missing: {path}")
        else:
            _ = read_table(path)

    log("validating required columns...")
    require_columns(pred, REQUIRED_PRED_COLS, "Step6B prediction valid rows")
    require_columns(metrics_config, ["config_id", "metric_weighting", "mae_log10", "factor_2_accuracy"], "Step6B metrics by config")

    log("computing diagnostic columns...")
    pred = prepare_predictions(pred, args.max_rows_per_config)

    log("preparing default metrics tables...")
    default_metrics = filter_default_metrics(metrics_config)
    default_metrics.to_csv(args.output / out_name("step6c_broad_family_default_metrics_for_figures", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    log("comparing original vs broad_family metrics...")
    original_summary = build_original_vs_broad_summary(compare)
    original_summary.to_csv(args.output / out_name("step6c_original_vs_broad_metrics_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    log("preparing material_family vs global diagnostics...")
    diff_summary.to_csv(args.output / out_name("step6c_material_family_vs_global_diff_summary_for_report", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    log("diagnosing largest error rows...")
    group_perf = build_group_performance(metrics_material)
    group_perf.to_csv(args.output / out_name("step6c_broad_family_group_performance_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")
    largest = classify_largest_errors(pred)
    largest.to_csv(args.output / out_name("step6c_broad_largest_error_diagnostics_top100", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    fig_index_rows: list[dict[str, Any]] = []
    log("creating predicted vs experimental plots...")
    make_scatter_plots(pred, default_metrics, args, fig_index_rows)
    log("creating error distribution plots...")
    make_error_histograms(pred, default_metrics, args, fig_index_rows)
    log("creating original vs broad comparison plots...")
    make_metric_comparison_plot(compare, "mae_log10", "step6c_original_vs_broad_default_mae", "Original vs broad default MAE", "MAE log10", args, fig_index_rows)
    make_metric_comparison_plot(compare, "factor_2_accuracy", "step6c_original_vs_broad_default_factor2", "Original vs broad default factor2", "Factor 2 accuracy", args, fig_index_rows)
    make_metric_comparison_plot(compare, "factor_10_accuracy", "step6c_original_vs_broad_default_factor10", "Original vs broad default factor10", "Factor 10 accuracy", args, fig_index_rows)
    log("creating config comparison plots...")
    make_config_top_plots(metrics_config, args, fig_index_rows)
    make_delta_hist(diff_examples, diff_summary, args, fig_index_rows)
    log("creating eta/temperature/carrier plots...")
    make_bin_and_carrier_plots(metrics_eta, metrics_temp, metrics_carrier, args, fig_index_rows)
    make_abs_error_scatter(pred, args, fig_index_rows)
    log("creating broad family group plots...")
    make_group_plots(group_perf, args, fig_index_rows)
    log("creating reference sigma0(T) plots...")
    make_reference_plots(reference, args, fig_index_rows)

    log("writing figure index...")
    fig_index = pd.DataFrame(fig_index_rows)
    fig_index.to_csv(args.output / out_name("step6c_figure_index", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    log("writing summary CSVs...")
    visual_summary = build_visual_summary(pred, default_metrics, original_summary, diff_summary, largest, reference)
    visual_summary.to_csv(args.output / out_name("step6c_visual_diagnostics_summary", args.output_suffix, "csv"), index=False, encoding="utf-8-sig")

    csv_files = [
        args.output / out_name("step6c_figure_index", args.output_suffix, "csv"),
        args.output / out_name("step6c_broad_family_default_metrics_for_figures", args.output_suffix, "csv"),
        args.output / out_name("step6c_original_vs_broad_metrics_summary", args.output_suffix, "csv"),
        args.output / out_name("step6c_material_family_vs_global_diff_summary_for_report", args.output_suffix, "csv"),
        args.output / out_name("step6c_broad_family_group_performance_summary", args.output_suffix, "csv"),
        args.output / out_name("step6c_broad_largest_error_diagnostics_top100", args.output_suffix, "csv"),
        args.output / out_name("step6c_visual_diagnostics_summary", args.output_suffix, "csv"),
    ]
    input_files = [
        args.step6b_dir / "step5b_test_predictions_valid.parquet",
        args.step6b_dir / "step5b_train_reference_curve_bins.csv",
        args.step6b_dir / "step5c_metrics_by_config.csv",
        args.step6b_dir / "step6b_broad_family_vs_original_default_metrics_comparison.csv",
    ]

    log("writing report...")
    write_report(args.report, input_files, fig_index, csv_files, default_metrics, original_summary, diff_summary, group_perf, largest, visual_summary, {}, warnings, time.time() - started)

    log("running sanity checks...")
    checks, failures = run_sanity(pred, default_metrics, original_summary, diff_summary, group_perf, largest, fig_index, visual_summary, args.report, args.output_suffix)
    if failures:
        write_report(args.report, input_files, fig_index, csv_files, default_metrics, original_summary, diff_summary, group_perf, largest, visual_summary, checks, warnings, time.time() - started)
        for failure in failures:
            print(f"[step6c] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(args.report, input_files, fig_index, csv_files, default_metrics, original_summary, diff_summary, group_perf, largest, visual_summary, checks, warnings, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
