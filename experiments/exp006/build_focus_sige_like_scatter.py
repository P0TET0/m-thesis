import argparse
import math
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

DEFAULT_PREDICTIONS_PARQUET = (
    PROCESSED_DIR / "step6b_broad_family" / "step5b_test_predictions_valid.parquet"
)
DEFAULT_PREDICTIONS_CSV = (
    PROCESSED_DIR / "step6b_broad_family" / "step5b_test_predictions_valid.csv"
)
DEFAULT_OUTPUT = PROCESSED_DIR / "focus_sige_like"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_sige_like"
DEFAULT_REPORT = EXP_DIR / "reports" / "focus_sige_like" / "focus_sige_like_report.md"

DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
)
DEFAULT_TARGET_MATERIAL_GROUP = "broad::SiGe_like"

OPTIONAL_INPUTS = [
    PROCESSED_DIR
    / "step6c_broad_family"
    / "step6c_broad_family_group_performance_summary.csv",
    PROCESSED_DIR
    / "step6b_broad_family"
    / "step5c_metrics_by_material_family.csv",
    PROCESSED_DIR
    / "step6b_broad_family"
    / "step5c_metrics_by_carrier_type.csv",
]

REQUESTED_COLUMNS = [
    "config_id",
    "prediction_status",
    "row_id",
    "carrier_type",
    "material_group_key",
    "material_group_key_for_prediction",
    "material_family_raw",
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
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "paper_id",
    "sample_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
]

CRITICAL_COLUMNS = [
    "config_id",
    "carrier_type",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_over_exp",
]

MATERIAL_FILTER_COLUMNS = [
    "material_group_key_for_prediction",
    "material_group_key",
    "material_family_raw",
]

NUMERIC_COLUMNS = [
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
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
]

METRIC_COLUMNS = [
    "subset",
    "n_rows",
    "n_samples",
    "n_papers",
    "T_min_K",
    "T_max_K",
    "eta_min",
    "eta_median",
    "eta_max",
    "sigma_exp_min_S_per_m",
    "sigma_exp_median_S_per_m",
    "sigma_exp_max_S_per_m",
    "sigma_pred_min_S_per_m",
    "sigma_pred_median_S_per_m",
    "sigma_pred_max_S_per_m",
    "mean_log10_error",
    "median_log10_error",
    "mae_log10",
    "rmse_log10",
    "max_abs_log10_error",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "overprediction_fraction",
    "underprediction_fraction",
]

LARGEST_ERROR_COLUMNS = [
    "row_id",
    "carrier_type",
    "material_group_key",
    "material_group_key_for_prediction",
    "T_K",
    "S_uV_per_K",
    "eta",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_ref_S_per_m",
    "sigma0_S_per_m",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build focus plots for broad::SiGe_like prediction rows."
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--target-material-group", default=DEFAULT_TARGET_MATERIAL_GROUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[focus_sige] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def resolve_predictions(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"prediction file not found: {path}")
        return path
    if DEFAULT_PREDICTIONS_PARQUET.exists():
        return DEFAULT_PREDICTIONS_PARQUET
    if DEFAULT_PREDICTIONS_CSV.exists():
        return DEFAULT_PREDICTIONS_CSV
    raise FileNotFoundError(
        f"prediction file not found: {DEFAULT_PREDICTIONS_PARQUET} or {DEFAULT_PREDICTIONS_CSV}"
    )


def out_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def ensure_columns(df: pd.DataFrame) -> None:
    missing_critical = sorted(set(CRITICAL_COLUMNS) - set(df.columns))
    if missing_critical:
        raise ValueError(f"prediction input missing critical columns: {missing_critical}")
    if not (
        "material_group_key" in df.columns
        or "material_group_key_for_prediction" in df.columns
    ):
        raise ValueError(
            "prediction input must include material_group_key or material_group_key_for_prediction"
        )
    for col in REQUESTED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    if "prediction_status" not in df.columns:
        df["prediction_status"] = pd.NA
    if "abs_log10_sigma_pred_over_exp" not in df.columns:
        df["abs_log10_sigma_pred_over_exp"] = pd.to_numeric(
            df["log10_sigma_pred_over_exp"], errors="coerce"
        ).abs()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_optional_inputs() -> list[dict[str, Any]]:
    loaded = []
    for path in OPTIONAL_INPUTS:
        if not path.exists():
            loaded.append({"path": str(path), "status": "missing", "rows": 0, "columns": 0})
            continue
        table = read_table(path)
        loaded.append(
            {
                "path": str(path),
                "status": "loaded",
                "rows": len(table),
                "columns": len(table.columns),
            }
        )
    return loaded


def top_values(df: pd.DataFrame, col: str, n: int = 20) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame({"value": [], "count": []})
    vc = df[col].fillna("<NA>").astype(str).value_counts().head(n)
    return pd.DataFrame({"value": vc.index, "count": vc.values})


def suggest_candidates(df: pd.DataFrame, target: str) -> list[str]:
    tokens = [part for part in target.replace("::", "_").replace("-", "_").split("_") if part]
    tokens = [token.casefold() for token in tokens]
    suggestions: list[str] = []
    for col in MATERIAL_FILTER_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).unique()
        for value in values:
            lower = value.casefold()
            if any(token in lower for token in tokens) and value not in suggestions:
                suggestions.append(value)
            if len(suggestions) >= 30:
                return suggestions
    return suggestions


def write_zero_row_report(
    report_path: Path,
    predictions_path: Path,
    config_id: str,
    target: str,
    config_df: pd.DataFrame,
    optional_inputs: list[dict[str, Any]],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Focus SiGe_like Report",
        "",
        "No rows were extracted, so figures were not created.",
        "",
        f"- Input file: `{predictions_path}`",
        f"- Target config: `{config_id}`",
        f"- Target material group: `{target}`",
        "",
        "## Top Unique Values",
    ]
    for col in MATERIAL_FILTER_COLUMNS:
        lines.append(f"### {col}")
        values = top_values(config_df, col)
        if values.empty:
            lines.append("- column missing or empty")
        else:
            for _, row in values.iterrows():
                lines.append(f"- `{row['value']}`: {row['count']}")
    candidates = suggest_candidates(config_df, target)
    lines.extend(["", "## Candidate Names"])
    if candidates:
        lines.extend([f"- `{value}`" for value in candidates])
    else:
        lines.append("- No likely candidate names found.")
    lines.extend(["", "## Optional Inputs"])
    for item in optional_inputs:
        lines.append(
            f"- `{item['path']}`: {item['status']} rows={item['rows']} columns={item['columns']}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_material_filter(
    config_df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, str | None]:
    for col in MATERIAL_FILTER_COLUMNS:
        if col not in config_df.columns:
            continue
        rows = config_df[config_df[col].astype("string").eq(target)]
        if len(rows) > 0:
            return rows.copy(), col
    return config_df.iloc[0:0].copy(), None


def clean_rows(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows = rows[rows["prediction_status"].astype("string").eq("ok")]
    rows = rows[np.isfinite(rows["sigma_S_per_m"]) & (rows["sigma_S_per_m"] > 0)]
    rows = rows[
        np.isfinite(rows["sigma_pred_S_per_m"]) & (rows["sigma_pred_S_per_m"] > 0)
    ]
    rows = rows[rows["carrier_type"].astype("string").isin(["p", "n"])]
    rows["computed_log10_sigma_pred_over_exp"] = np.log10(
        rows["sigma_pred_S_per_m"] / rows["sigma_S_per_m"]
    )
    rows["log10_error_delta"] = (
        rows["log10_sigma_pred_over_exp"]
        - rows["computed_log10_sigma_pred_over_exp"]
    )
    rows["abs_log10_sigma_pred_over_exp"] = rows[
        "log10_sigma_pred_over_exp"
    ].abs()
    return rows


def maybe_limit_rows(rows: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows.sort_values(["carrier_type", "row_id"], kind="mergesort").head(max_rows)


def finite_stat(series: pd.Series, op: str) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    if op == "min":
        return float(np.min(values))
    if op == "max":
        return float(np.max(values))
    if op == "median":
        return float(np.median(values))
    if op == "mean":
        return float(np.mean(values))
    raise ValueError(op)


def unique_count(df: pd.DataFrame, columns: list[str]) -> int:
    for col in columns:
        if col in df.columns:
            return int(df[col].nunique(dropna=True))
    return 0


def compute_metrics(df: pd.DataFrame, subset: str) -> dict[str, Any]:
    err = pd.to_numeric(df["log10_sigma_pred_over_exp"], errors="coerce").to_numpy(
        dtype=float
    )
    err = err[np.isfinite(err)]
    abs_err = np.abs(err)
    n = len(df)
    if len(err) == 0:
        mean_err = median_err = mae = rmse = max_abs = math.nan
        factor2 = factor5 = factor10 = over = under = math.nan
    else:
        mean_err = float(np.mean(err))
        median_err = float(np.median(err))
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(err**2)))
        max_abs = float(np.max(abs_err))
        factor2 = float(np.mean(abs_err <= math.log10(2)))
        factor5 = float(np.mean(abs_err <= math.log10(5)))
        factor10 = float(np.mean(abs_err <= 1.0))
        over = float(np.mean(err > 0))
        under = float(np.mean(err < 0))
    return {
        "subset": subset,
        "n_rows": int(n),
        "n_samples": unique_count(df, ["sample_key", "sample_id"]),
        "n_papers": unique_count(df, ["paper_id"]),
        "T_min_K": finite_stat(df["T_K"], "min"),
        "T_max_K": finite_stat(df["T_K"], "max"),
        "eta_min": finite_stat(df["eta"], "min"),
        "eta_median": finite_stat(df["eta"], "median"),
        "eta_max": finite_stat(df["eta"], "max"),
        "sigma_exp_min_S_per_m": finite_stat(df["sigma_S_per_m"], "min"),
        "sigma_exp_median_S_per_m": finite_stat(df["sigma_S_per_m"], "median"),
        "sigma_exp_max_S_per_m": finite_stat(df["sigma_S_per_m"], "max"),
        "sigma_pred_min_S_per_m": finite_stat(df["sigma_pred_S_per_m"], "min"),
        "sigma_pred_median_S_per_m": finite_stat(df["sigma_pred_S_per_m"], "median"),
        "sigma_pred_max_S_per_m": finite_stat(df["sigma_pred_S_per_m"], "max"),
        "mean_log10_error": mean_err,
        "median_log10_error": median_err,
        "mae_log10": mae,
        "rmse_log10": rmse,
        "max_abs_log10_error": max_abs,
        "factor_2_accuracy": factor2,
        "factor_5_accuracy": factor5,
        "factor_10_accuracy": factor10,
        "overprediction_fraction": over,
        "underprediction_fraction": under,
    }


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def num(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def plot_limits(df: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [
            df["sigma_S_per_m"].to_numpy(dtype=float),
            df["sigma_pred_S_per_m"].to_numpy(dtype=float),
        ]
    )
    values = values[np.isfinite(values) & (values > 0)]
    lower = 10 ** math.floor(math.log10(values.min()))
    upper = 10 ** math.ceil(math.log10(values.max()))
    if lower == upper:
        upper = lower * 10
    return float(lower), float(upper)


def save_scatter(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    title_prefix: str,
    png_path: Path,
    pdf_path: Path,
    split_carrier: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    markers = {"p": "o", "n": "^"}
    colors = {"p": "#1f77b4", "n": "#d62728"}
    if split_carrier:
        for carrier in ["p", "n"]:
            subset = df[df["carrier_type"].astype("string").eq(carrier)]
            if subset.empty:
                continue
            ax.scatter(
                subset["sigma_S_per_m"],
                subset["sigma_pred_S_per_m"],
                s=24,
                alpha=0.72,
                marker=markers[carrier],
                color=colors[carrier],
                edgecolors="none",
                label=f"{carrier}-type",
            )
    else:
        carrier = str(df["carrier_type"].iloc[0]) if len(df) else ""
        ax.scatter(
            df["sigma_S_per_m"],
            df["sigma_pred_S_per_m"],
            s=26,
            alpha=0.76,
            marker=markers.get(carrier, "o"),
            color=colors.get(carrier, "#333333"),
            edgecolors="none",
            label=f"{carrier}-type" if carrier else None,
        )
    lower, upper = plot_limits(df)
    line = np.array([lower, upper])
    ax.plot(line, line, color="black", linewidth=1.2, label="y=x")
    ax.plot(line, line * 2, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.plot(line, line / 2, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.plot(line, line * 10, color="gray", linewidth=0.7, linestyle=":", alpha=0.45)
    ax.plot(line, line / 10, color="gray", linewidth=0.7, linestyle=":", alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("sigma_exp = sigma_S_per_m")
    ax.set_ylabel("sigma_pred = sigma_pred_S_per_m")
    ax.set_title(
        f"{title_prefix}\n"
        f"n={metrics['n_rows']}, MAE(log10)={num(metrics['mae_log10'])}, "
        f"factor2={pct(metrics['factor_2_accuracy'])}, "
        f"factor10={pct(metrics['factor_10_accuracy'])}"
    )
    ax.grid(True, which="both", linewidth=0.4, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def save_error_hist(
    df: pd.DataFrame,
    metrics: dict[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    bins = np.linspace(
        min(-1.2, float(df["log10_sigma_pred_over_exp"].min())),
        max(1.2, float(df["log10_sigma_pred_over_exp"].max())),
        32,
    )
    colors = {"p": "#1f77b4", "n": "#d62728"}
    for carrier in ["p", "n"]:
        subset = df[df["carrier_type"].astype("string").eq(carrier)]
        if subset.empty:
            continue
        ax.hist(
            subset["log10_sigma_pred_over_exp"],
            bins=bins,
            alpha=0.52,
            label=f"{carrier}-type",
            color=colors[carrier],
        )
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(-1, color="gray", linewidth=0.9, linestyle="--")
    ax.axvline(1, color="gray", linewidth=0.9, linestyle="--")
    ax.set_xlabel("log10_sigma_pred_over_exp")
    ax.set_ylabel("Row count")
    ax.set_title(
        "SiGe_like error histogram\n"
        f"median={num(metrics['median_log10_error'])}, "
        f"MAE={num(metrics['mae_log10'])}, RMSE={num(metrics['rmse_log10'])}"
    )
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_report(
    report_path: Path,
    predictions_path: Path,
    config_id: str,
    target: str,
    filter_col: str,
    rows: pd.DataFrame,
    metrics_df: pd.DataFrame,
    largest: pd.DataFrame,
    figures: list[dict[str, Any]],
    optional_inputs: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric = {row["subset"]: row for _, row in metrics_df.iterrows()}
    all_m = metric["all"]
    p_m = metric["p"]
    n_m = metric["n"]
    max_row = largest.iloc[0] if not largest.empty else None
    lines = [
        "# Focus SiGe_like Report",
        "",
        "## Inputs",
        f"- Prediction input: `{predictions_path}`",
        f"- Target config: `{config_id}`",
        f"- Target material group: `{target}`",
        f"- Material group filter column: `{filter_col}`",
        "",
        "## Optional Inputs",
    ]
    for item in optional_inputs:
        lines.append(
            f"- `{item['path']}`: {item['status']} rows={item['rows']} columns={item['columns']}"
        )
    lines.extend(
        [
            "",
            "## Extracted Rows",
            f"- Total rows: {len(rows)}",
            f"- p-type rows: {int((rows['carrier_type'].astype('string') == 'p').sum())}",
            f"- n-type rows: {int((rows['carrier_type'].astype('string') == 'n').sum())}",
            "",
            "## Metrics",
            (
                f"- all: MAE={num(all_m['mae_log10'])}, RMSE={num(all_m['rmse_log10'])}, "
                f"factor2={pct(all_m['factor_2_accuracy'])}, factor10={pct(all_m['factor_10_accuracy'])}"
            ),
            (
                f"- p: MAE={num(p_m['mae_log10'])}, RMSE={num(p_m['rmse_log10'])}, "
                f"factor2={pct(p_m['factor_2_accuracy'])}, factor10={pct(p_m['factor_10_accuracy'])}"
            ),
            (
                f"- n: MAE={num(n_m['mae_log10'])}, RMSE={num(n_m['rmse_log10'])}, "
                f"factor2={pct(n_m['factor_2_accuracy'])}, factor10={pct(n_m['factor_10_accuracy'])}"
            ),
            "",
            "## Largest Outlier",
        ]
    )
    if max_row is None:
        lines.append("- No largest outlier row available.")
    else:
        lines.extend(
            [
                f"- row_id: `{max_row.get('row_id', '')}`",
                f"- carrier_type: `{max_row.get('carrier_type', '')}`",
                f"- abs_log10_sigma_pred_over_exp: {num(max_row.get('abs_log10_sigma_pred_over_exp'))}",
                f"- log10_sigma_pred_over_exp: {num(max_row.get('log10_sigma_pred_over_exp'))}",
                f"- sigma_exp_S_per_m: {num(max_row.get('sigma_S_per_m'))}",
                f"- sigma_pred_S_per_m: {num(max_row.get('sigma_pred_S_per_m'))}",
                f"- sample_key: `{max_row.get('sample_key', '')}`",
                f"- paper_id: `{max_row.get('paper_id', '')}`",
            ]
        )
    lines.extend(["", "## Figures"])
    for fig in figures:
        lines.append(
            f"- `{fig['figure_id']}`: `{fig['figure_path_png']}` / `{fig['figure_path_pdf']}`"
        )
    lines.extend(
        [
            "",
            "## How To Read The Scatter Plots",
            "- Points closer to y=x are better.",
            "- Points above y=x are overpredictions.",
            "- Points below y=x are underpredictions.",
            "",
            "## Notes",
            "- This is a focus analysis that filters existing prediction results by material group.",
            "- No new sigma_pred values are calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data is not read.",
            "- SiGe_like results are provisional material-group checks based on broad_family classification.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {warning}" for warning in warnings])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    rows: pd.DataFrame,
    config_id: str,
    metrics_path: Path,
    figure_index_path: Path,
    report_path: Path,
    figures: list[dict[str, Any]],
) -> None:
    if rows.empty:
        raise ValueError("sanity check failed: extracted rows are empty")
    if set(rows["config_id"].astype(str)) != {config_id}:
        raise ValueError("sanity check failed: config_id mismatch")
    if not rows["prediction_status"].astype("string").eq("ok").all():
        raise ValueError("sanity check failed: prediction_status is not all ok")
    if not (np.isfinite(rows["sigma_S_per_m"]) & (rows["sigma_S_per_m"] > 0)).all():
        raise ValueError("sanity check failed: invalid sigma_S_per_m")
    if not (
        np.isfinite(rows["sigma_pred_S_per_m"]) & (rows["sigma_pred_S_per_m"] > 0)
    ).all():
        raise ValueError("sanity check failed: invalid sigma_pred_S_per_m")
    if not rows["carrier_type"].astype("string").isin(["p", "n"]).all():
        raise ValueError("sanity check failed: carrier_type not limited to p/n")
    delta = rows["log10_error_delta"].abs().max()
    if pd.isna(delta) or float(delta) > 1e-8:
        raise ValueError(
            f"sanity check failed: log10 error mismatch max delta={delta}"
        )
    for path in [metrics_path, figure_index_path, report_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"sanity check failed: missing or empty output {path}")
    figure_ids = {fig["figure_id"] for fig in figures}
    if "scatter_all" not in figure_ids:
        raise ValueError("sanity check failed: all scatter figure missing")
    if not ({"scatter_p", "scatter_n"} & figure_ids):
        raise ValueError("sanity check failed: no p/n scatter figure created")
    for fig in figures:
        for key in ["figure_path_png", "figure_path_pdf"]:
            path = Path(fig[key])
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"sanity check failed: missing figure {path}")


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    predictions_path = resolve_predictions(args.predictions)

    log("loading broad_family predictions...")
    predictions = read_table(predictions_path)
    optional_inputs = load_optional_inputs()
    ensure_columns(predictions)

    log("filtering target config...")
    config_df = predictions[predictions["config_id"].astype("string").eq(args.config_id)].copy()

    log("filtering target material group...")
    material_rows, filter_col = choose_material_filter(
        config_df, args.target_material_group
    )
    if material_rows.empty or filter_col is None:
        write_zero_row_report(
            args.report,
            predictions_path,
            args.config_id,
            args.target_material_group,
            config_df,
            optional_inputs,
        )
        raise SystemExit(
            f"No rows found for target material group {args.target_material_group}; report written to {args.report}"
        )

    rows = clean_rows(material_rows)
    rows = maybe_limit_rows(rows, args.max_rows)
    if rows.empty:
        write_zero_row_report(
            args.report,
            predictions_path,
            args.config_id,
            args.target_material_group,
            config_df,
            optional_inputs,
        )
        raise SystemExit(
            f"No valid rows after quality filters for {args.target_material_group}; report written to {args.report}"
        )

    p_rows = rows[rows["carrier_type"].astype("string").eq("p")].copy()
    n_rows = rows[rows["carrier_type"].astype("string").eq("n")].copy()
    log(f"extracted rows: {len(rows)}")
    log(f"p rows: {len(p_rows)}")
    log(f"n rows: {len(n_rows)}")

    log("computing metrics...")
    metrics_df = pd.DataFrame(
        [
            compute_metrics(rows, "all"),
            compute_metrics(p_rows, "p"),
            compute_metrics(n_rows, "n"),
        ],
        columns=METRIC_COLUMNS,
    )
    metrics_by_subset = {
        row["subset"]: row.to_dict() for _, row in metrics_df.iterrows()
    }

    log("writing filtered rows...")
    rows_csv = args.output / out_name("focus_sige_like_prediction_rows", args.output_suffix, "csv")
    rows_parquet = args.output / out_name(
        "focus_sige_like_prediction_rows", args.output_suffix, "parquet"
    )
    rows_p_csv = args.output / out_name(
        "focus_sige_like_prediction_rows_p", args.output_suffix, "csv"
    )
    rows_n_csv = args.output / out_name(
        "focus_sige_like_prediction_rows_n", args.output_suffix, "csv"
    )
    metrics_csv = args.output / out_name(
        "focus_sige_like_metrics_summary", args.output_suffix, "csv"
    )
    largest_csv = args.output / out_name(
        "focus_sige_like_largest_error_rows", args.output_suffix, "csv"
    )
    figure_index_csv = args.output / out_name(
        "focus_sige_like_figure_index", args.output_suffix, "csv"
    )
    rows.to_csv(rows_csv, index=False)
    rows.to_parquet(rows_parquet, index=False)
    p_rows.to_csv(rows_p_csv, index=False)
    n_rows.to_csv(rows_n_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)

    for col in LARGEST_ERROR_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.NA
    largest = (
        rows.sort_values("abs_log10_sigma_pred_over_exp", ascending=False)
        .head(100)[LARGEST_ERROR_COLUMNS]
        .copy()
    )
    largest.to_csv(largest_csv, index=False)

    figures: list[dict[str, Any]] = []
    warnings: list[str] = []

    log("creating all scatter plot...")
    all_png = args.figures / out_name(
        "focus_sige_like_scatter_pred_vs_exp_all", args.output_suffix, "png"
    )
    all_pdf = args.figures / out_name(
        "focus_sige_like_scatter_pred_vs_exp_all", args.output_suffix, "pdf"
    )
    save_scatter(
        rows,
        metrics_by_subset["all"],
        f"{args.target_material_group} all",
        all_png,
        all_pdf,
        split_carrier=True,
    )
    figures.append(
        {
            "figure_id": "scatter_all",
            "figure_path_png": str(all_png),
            "figure_path_pdf": str(all_pdf),
            "title": f"{args.target_material_group} all pred vs exp",
            "source_file": str(rows_csv),
            "n_points_plotted": len(rows),
            "description": "Log-log scatter of measured sigma versus predicted sigma for all p/n rows.",
        }
    )

    log("creating p-type scatter plot...")
    if p_rows.empty:
        warnings.append("p-type rows are zero; p-type scatter plot was not created.")
    else:
        p_png = args.figures / out_name(
            "focus_sige_like_scatter_pred_vs_exp_p", args.output_suffix, "png"
        )
        p_pdf = args.figures / out_name(
            "focus_sige_like_scatter_pred_vs_exp_p", args.output_suffix, "pdf"
        )
        save_scatter(
            p_rows,
            metrics_by_subset["p"],
            f"{args.target_material_group} p-type",
            p_png,
            p_pdf,
            split_carrier=False,
        )
        figures.append(
            {
                "figure_id": "scatter_p",
                "figure_path_png": str(p_png),
                "figure_path_pdf": str(p_pdf),
                "title": f"{args.target_material_group} p-type pred vs exp",
                "source_file": str(rows_p_csv),
                "n_points_plotted": len(p_rows),
                "description": "Log-log scatter of measured sigma versus predicted sigma for p-type rows.",
            }
        )

    log("creating n-type scatter plot...")
    if n_rows.empty:
        warnings.append("n-type rows are zero; n-type scatter plot was not created.")
    else:
        n_png = args.figures / out_name(
            "focus_sige_like_scatter_pred_vs_exp_n", args.output_suffix, "png"
        )
        n_pdf = args.figures / out_name(
            "focus_sige_like_scatter_pred_vs_exp_n", args.output_suffix, "pdf"
        )
        save_scatter(
            n_rows,
            metrics_by_subset["n"],
            f"{args.target_material_group} n-type",
            n_png,
            n_pdf,
            split_carrier=False,
        )
        figures.append(
            {
                "figure_id": "scatter_n",
                "figure_path_png": str(n_png),
                "figure_path_pdf": str(n_pdf),
                "title": f"{args.target_material_group} n-type pred vs exp",
                "source_file": str(rows_n_csv),
                "n_points_plotted": len(n_rows),
                "description": "Log-log scatter of measured sigma versus predicted sigma for n-type rows.",
            }
        )

    log("creating error histogram...")
    hist_png = args.figures / out_name(
        "focus_sige_like_error_hist_all", args.output_suffix, "png"
    )
    hist_pdf = args.figures / out_name(
        "focus_sige_like_error_hist_all", args.output_suffix, "pdf"
    )
    save_error_hist(rows, metrics_by_subset["all"], hist_png, hist_pdf)
    figures.append(
        {
            "figure_id": "error_hist_all",
            "figure_path_png": str(hist_png),
            "figure_path_pdf": str(hist_pdf),
            "title": f"{args.target_material_group} log10 prediction error histogram",
            "source_file": str(rows_csv),
            "n_points_plotted": len(rows),
            "description": "Histogram of log10_sigma_pred_over_exp with p/n rows overlaid.",
        }
    )

    log("writing figure index...")
    figure_index = pd.DataFrame(
        figures,
        columns=[
            "figure_id",
            "figure_path_png",
            "figure_path_pdf",
            "title",
            "source_file",
            "n_points_plotted",
            "description",
        ],
    )
    figure_index.to_csv(figure_index_csv, index=False)

    log("writing report...")
    write_report(
        args.report,
        predictions_path,
        args.config_id,
        args.target_material_group,
        filter_col,
        rows,
        metrics_df,
        largest,
        figures,
        optional_inputs,
        warnings,
    )

    log("running sanity checks...")
    run_sanity_checks(
        rows,
        args.config_id,
        metrics_csv,
        figure_index_csv,
        args.report,
        figures,
    )
    elapsed = time.perf_counter() - start
    log(f"done. elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
