import argparse
import math
import re
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
DEFAULT_PERFORMANCE_SUMMARY = (
    PROCESSED_DIR
    / "step6c_broad_family"
    / "step6c_broad_family_group_performance_summary.csv"
)
FALLBACK_PERFORMANCE_SUMMARY = (
    PROCESSED_DIR / "step6b_broad_family" / "step5c_metrics_by_material_family.csv"
)
DEFAULT_OUTPUT = PROCESSED_DIR / "focus_broad_families"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_broad_families"
DEFAULT_REPORT = (
    EXP_DIR / "reports" / "focus_broad_families" / "focus_broad_families_report.md"
)

DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
)

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

SUMMARY_GROUP_COLUMNS = [
    "material_group_key",
    "material_family_raw",
    "material_group_key_for_prediction",
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
    "material_group_key",
    "safe_group_name",
    "carrier_subset",
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
    "p_row_count",
    "n_row_count",
]

LARGEST_ERROR_COLUMNS = [
    "material_group_key",
    "row_id",
    "carrier_type",
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
        description="Build per-material broad_family prediction scatter plots."
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--performance-summary", type=Path, default=None)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument(
        "--selection-mode",
        choices=[
            "all_from_summary",
            "reliable_from_summary",
            "top_by_mae",
            "top_by_factor2",
            "manual_list",
        ],
        default="reliable_from_summary",
    )
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-papers", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--target-groups", nargs="*", default=[])
    parser.add_argument("--exclude-groups", nargs="*", default=[])
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-rows-per-group", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[focus_broad] {message}", flush=True)


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


def resolve_performance_summary(path: Path | None) -> Path | None:
    if path is not None:
        return path if path.exists() else None
    if DEFAULT_PERFORMANCE_SUMMARY.exists():
        return DEFAULT_PERFORMANCE_SUMMARY
    if FALLBACK_PERFORMANCE_SUMMARY.exists():
        return FALLBACK_PERFORMANCE_SUMMARY
    return None


def out_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def safe_group_name(group: str) -> str:
    safe = group.replace("::", "__")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "unknown_group"


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


def group_col_from_summary(summary: pd.DataFrame) -> str | None:
    for col in SUMMARY_GROUP_COLUMNS:
        if col in summary.columns:
            return col
    return None


def build_summary_from_predictions(config_df: pd.DataFrame) -> pd.DataFrame:
    col = (
        "material_group_key_for_prediction"
        if "material_group_key_for_prediction" in config_df.columns
        else "material_group_key"
    )
    rows = []
    for group, part in config_df.groupby(col, dropna=True):
        clean = clean_rows(part)
        rows.append(
            {
                "material_group_key": group,
                "material_family_raw": "",
                "n_rows": len(clean),
                "n_samples": unique_count(clean, ["sample_key", "sample_id"]),
                "n_papers": unique_count(clean, ["paper_id"]),
                "mae_log10": pd.to_numeric(
                    clean["log10_sigma_pred_over_exp"], errors="coerce"
                )
                .abs()
                .mean(),
                "factor_2_accuracy": (
                    pd.to_numeric(clean["log10_sigma_pred_over_exp"], errors="coerce")
                    .abs()
                    <= math.log10(2)
                ).mean(),
                "factor_10_accuracy": (
                    pd.to_numeric(clean["log10_sigma_pred_over_exp"], errors="coerce")
                    .abs()
                    <= 1.0
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def select_groups(
    summary: pd.DataFrame,
    mode: str,
    min_rows: int,
    min_samples: int,
    min_papers: int,
    top_n: int,
    target_groups: list[str],
    exclude_groups: list[str],
    max_groups: int | None,
) -> pd.DataFrame:
    group_col = group_col_from_summary(summary)
    if mode == "manual_list":
        selected = pd.DataFrame({group_col or "material_group_key": target_groups})
        if group_col is None:
            group_col = "material_group_key"
        selected["selection_reason"] = "manual_list"
    else:
        if group_col is None:
            raise ValueError("performance summary has no material group column")
        selected = summary.copy()
        if mode == "reliable_from_summary":
            if {
                "is_reliable_eval_group",
                "eval_group_reliability",
            }.issubset(selected.columns):
                reliable_flag = selected["is_reliable_eval_group"].fillna(False).astype(bool)
                reliable_label = selected["eval_group_reliability"].astype("string").isin(
                    ["high", "medium", "low"]
                )
                selected = selected[reliable_flag | reliable_label].copy()
                selected["selection_reason"] = "reliable_from_summary"
            else:
                selected = selected[
                    (pd.to_numeric(selected.get("n_rows"), errors="coerce") >= min_rows)
                    & (
                        pd.to_numeric(selected.get("n_samples"), errors="coerce")
                        >= min_samples
                    )
                    & (
                        pd.to_numeric(selected.get("n_papers"), errors="coerce")
                        >= min_papers
                    )
                ].copy()
                selected["selection_reason"] = "fallback_min_rows_samples_papers"
        elif mode == "all_from_summary":
            selected = selected.copy()
            selected["selection_reason"] = "all_from_summary"
        elif mode == "top_by_mae":
            selected = selected.sort_values("mae_log10", ascending=True).head(top_n).copy()
            selected["selection_reason"] = f"top_by_mae_top_{top_n}"
        elif mode == "top_by_factor2":
            selected = (
                selected.sort_values("factor_2_accuracy", ascending=False)
                .head(top_n)
                .copy()
            )
            selected["selection_reason"] = f"top_by_factor2_top_{top_n}"
        else:
            raise ValueError(mode)

    selected = selected[~selected[group_col].astype("string").isin(exclude_groups)].copy()
    selected = selected.dropna(subset=[group_col]).drop_duplicates(subset=[group_col])
    if max_groups is not None and max_groups > 0:
        selected = selected.head(max_groups).copy()

    selected_out = pd.DataFrame(
        {
            "material_group_key": selected[group_col].astype(str),
            "safe_group_name": selected[group_col].astype(str).map(safe_group_name),
            "selection_reason": selected.get("selection_reason", pd.Series([""] * len(selected))).values,
            "source_summary_rows": 1,
            "expected_n_rows_from_summary": pd.to_numeric(
                selected.get("n_rows"), errors="coerce"
            ),
            "expected_n_samples_from_summary": pd.to_numeric(
                selected.get("n_samples"), errors="coerce"
            ),
            "summary_mae_log10": pd.to_numeric(selected.get("mae_log10"), errors="coerce"),
            "summary_factor_2_accuracy": pd.to_numeric(
                selected.get("factor_2_accuracy"), errors="coerce"
            ),
            "summary_factor_10_accuracy": pd.to_numeric(
                selected.get("factor_10_accuracy"), errors="coerce"
            ),
        }
    )
    return selected_out.reset_index(drop=True)


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


def compute_metrics(
    df: pd.DataFrame,
    material_group: str,
    safe_name: str,
    subset: str,
    p_count: int,
    n_count: int,
) -> dict[str, Any]:
    err = pd.to_numeric(df["log10_sigma_pred_over_exp"], errors="coerce").to_numpy(
        dtype=float
    )
    err = err[np.isfinite(err)]
    abs_err = np.abs(err)
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
        "material_group_key": material_group,
        "safe_group_name": safe_name,
        "carrier_subset": subset,
        "n_rows": int(len(df)),
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
        "p_row_count": p_count,
        "n_row_count": n_count,
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
                s=20,
                alpha=0.68,
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
            s=22,
            alpha=0.72,
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
    by_carrier: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    errors = pd.to_numeric(df["log10_sigma_pred_over_exp"], errors="coerce")
    outside = int(((errors < -5) | (errors > 5)).sum())
    bins = np.linspace(-5, 5, 41)
    colors = {"p": "#1f77b4", "n": "#d62728"}
    if by_carrier:
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
        ax.legend(frameon=False)
        title_prefix = "error histogram by carrier"
    else:
        ax.hist(errors, bins=bins, alpha=0.72, color="#4c78a8")
        title_prefix = "error histogram"
    ax.axvline(0, color="black", linewidth=1.2)
    ax.axvline(-1, color="gray", linewidth=0.9, linestyle="--")
    ax.axvline(1, color="gray", linewidth=0.9, linestyle="--")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("log10_sigma_pred_over_exp")
    ax.set_ylabel("Row count")
    ax.set_title(
        f"{metrics['material_group_key']} {title_prefix}\n"
        f"median={num(metrics['median_log10_error'])}, "
        f"MAE={num(metrics['mae_log10'])}, RMSE={num(metrics['rmse_log10'])}, "
        f"outside[-5,5]={outside}"
    )
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranking = metrics_df[metrics_df["carrier_subset"].eq("all")].copy()
    ranking["rank_by_mae_log10"] = ranking["mae_log10"].rank(
        method="min", ascending=True
    )
    ranking["rank_by_factor_2_accuracy"] = ranking["factor_2_accuracy"].rank(
        method="min", ascending=False
    )
    ranking["rank_by_factor_10_accuracy"] = ranking["factor_10_accuracy"].rank(
        method="min", ascending=False
    )
    ranking["rank_by_n_rows"] = ranking["n_rows"].rank(method="min", ascending=False)
    rank_cols = [
        "rank_by_mae_log10",
        "rank_by_factor_2_accuracy",
        "rank_by_factor_10_accuracy",
        "rank_by_n_rows",
    ]
    ranking[rank_cols] = ranking[rank_cols].astype("Int64")
    return ranking.sort_values("rank_by_mae_log10").reset_index(drop=True)


def write_report(
    report_path: Path,
    predictions_path: Path,
    performance_path: Path | None,
    config_id: str,
    selection_mode: str,
    min_rows: int,
    min_samples: int,
    min_papers: int,
    exclude_groups: list[str],
    selected_groups: pd.DataFrame,
    processing: pd.DataFrame,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    largest: pd.DataFrame,
    figures: pd.DataFrame,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    all_metrics = metrics[metrics["carrier_subset"].eq("all")].copy()
    skipped = processing[processing["status"].astype(str).str.startswith("skipped")]
    lines = [
        "# Focus Broad Families Report",
        "",
        "## Inputs",
        f"- Prediction input: `{predictions_path}`",
        f"- Performance summary input: `{performance_path or 'not found; built from predictions'}`",
        f"- Target config: `{config_id}`",
        f"- selection_mode: `{selection_mode}`",
        f"- min_rows: {min_rows}",
        f"- min_samples: {min_samples}",
        f"- min_papers: {min_papers}",
        f"- exclude_groups: `{', '.join(exclude_groups) if exclude_groups else ''}`",
        "",
        "## Selected Material Groups",
    ]
    lines.extend([f"- `{group}`" for group in selected_groups["material_group_key"]])
    lines.extend(["", "## Skipped Material Groups"])
    if skipped.empty:
        lines.append("- None")
    else:
        for _, row in skipped.iterrows():
            lines.append(
                f"- `{row['material_group_key']}`: {row['status']} ({row['report_note']})"
            )
    lines.extend(["", "## Per-Group Metrics"])
    for _, row in all_metrics.sort_values("mae_log10").iterrows():
        lines.append(
            f"- `{row['material_group_key']}`: rows={int(row['n_rows'])}, "
            f"p={int(row['p_row_count'])}, n={int(row['n_row_count'])}, "
            f"MAE={num(row['mae_log10'])}, RMSE={num(row['rmse_log10'])}, "
            f"factor2={pct(row['factor_2_accuracy'])}, "
            f"factor10={pct(row['factor_10_accuracy'])}"
        )
    lines.extend(["", "## Best By MAE"])
    for _, row in ranking.sort_values("rank_by_mae_log10").head(10).iterrows():
        lines.append(f"- `{row['material_group_key']}`: MAE={num(row['mae_log10'])}")
    lines.extend(["", "## Best By Factor2"])
    for _, row in ranking.sort_values("rank_by_factor_2_accuracy").head(10).iterrows():
        lines.append(
            f"- `{row['material_group_key']}`: factor2={pct(row['factor_2_accuracy'])}"
        )
    lines.extend(["", "## Best By Factor10"])
    for _, row in ranking.sort_values("rank_by_factor_10_accuracy").head(10).iterrows():
        lines.append(
            f"- `{row['material_group_key']}`: factor10={pct(row['factor_10_accuracy'])}"
        )
    lines.extend(["", "## Largest Outlier Groups"])
    if largest.empty:
        lines.append("- None")
    else:
        top = (
            largest.groupby("material_group_key")["abs_log10_sigma_pred_over_exp"]
            .max()
            .sort_values(ascending=False)
            .head(10)
        )
        for group, value in top.items():
            lines.append(f"- `{group}`: max abs log10 error={num(value)}")
    lines.extend(["", "## Figures"])
    for _, row in figures.iterrows():
        lines.append(
            f"- `{row['figure_id']}`: `{row['figure_path_png']}` / `{row['figure_path_pdf']}`"
        )
    lines.extend(
        [
            "",
            "## How To Read The Scatter Plots",
            "- Points closer to y=x are better.",
            "- Points above y=x are overpredictions.",
            "- Points below y=x are underpredictions.",
            "- Points within the factor2 guide lines are within a factor of 2.",
            "- Points within the factor10 guide lines are within a factor of 10.",
            "",
            "## Notes",
            "- This is a focus analysis that filters existing prediction results by material group.",
            "- No new sigma_pred values are calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data is not read.",
            "- broad_family classification is heuristic and is not a strict material taxonomy.",
            "",
            "## Next Checks",
            "- Inspect groups with small MAE and sufficient n_rows/n_samples.",
            "- Compare p-type and n-type performance within each group.",
            "- Treat groups with scatter close to y=x as candidates where the model works well.",
            "- For poor scatter groups, consider whether scattering mechanisms or band assumptions differ.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    config_df: pd.DataFrame,
    selected: pd.DataFrame,
    all_rows: pd.DataFrame,
    processing: pd.DataFrame,
    config_id: str,
    metrics_path: Path,
    ranking_path: Path,
    figure_index_path: Path,
    report_path: Path,
    figures: pd.DataFrame,
) -> None:
    if config_df.empty:
        raise ValueError("sanity check failed: target config has no rows")
    if selected.empty:
        raise ValueError("sanity check failed: selected groups are empty")
    ok_groups = processing[
        ~processing["status"].astype(str).str.startswith("skipped")
    ]["material_group_key"].tolist()
    if len(ok_groups) == 0:
        raise ValueError("sanity check failed: no ok material groups")
    for group in ok_groups:
        if all_rows[all_rows["selected_material_group_key"].eq(group)].empty:
            raise ValueError(f"sanity check failed: no extracted rows for {group}")
    if set(all_rows["config_id"].astype(str)) != {config_id}:
        raise ValueError("sanity check failed: config_id mismatch")
    if not all_rows["prediction_status"].astype("string").eq("ok").all():
        raise ValueError("sanity check failed: prediction_status is not all ok")
    if not (
        np.isfinite(all_rows["sigma_S_per_m"]) & (all_rows["sigma_S_per_m"] > 0)
    ).all():
        raise ValueError("sanity check failed: invalid sigma_S_per_m")
    if not (
        np.isfinite(all_rows["sigma_pred_S_per_m"])
        & (all_rows["sigma_pred_S_per_m"] > 0)
    ).all():
        raise ValueError("sanity check failed: invalid sigma_pred_S_per_m")
    if not all_rows["carrier_type"].astype("string").isin(["p", "n"]).all():
        raise ValueError("sanity check failed: carrier_type not limited to p/n")
    delta = all_rows["log10_error_delta"].abs().max()
    if pd.isna(delta) or float(delta) > 1e-8:
        raise ValueError(
            f"sanity check failed: log10 error mismatch max delta={delta}"
        )
    for path in [metrics_path, ranking_path, figure_index_path, report_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"sanity check failed: missing or empty output {path}")
    scatter_all = figures[figures["carrier_subset"].eq("all")]
    if set(scatter_all["material_group_key"]) != set(ok_groups):
        raise ValueError("sanity check failed: not all ok groups have all scatter")
    for _, row in figures.iterrows():
        for key in ["figure_path_png", "figure_path_pdf"]:
            path = Path(row[key])
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"sanity check failed: missing figure {path}")


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    predictions_path = resolve_predictions(args.predictions)
    performance_path = resolve_performance_summary(args.performance_summary)

    log("loading broad_family predictions...")
    predictions = read_table(predictions_path)
    ensure_columns(predictions)

    log("loading performance summary...")
    performance = read_table(performance_path) if performance_path is not None else None

    log("filtering target config...")
    config_df = predictions[predictions["config_id"].astype("string").eq(args.config_id)].copy()
    if config_df.empty:
        raise ValueError(f"target config has no rows: {args.config_id}")

    if performance is None:
        performance = build_summary_from_predictions(config_df)

    log("selecting material groups...")
    selected_groups = select_groups(
        performance,
        args.selection_mode,
        args.min_rows,
        args.min_samples,
        args.min_papers,
        args.top_n,
        args.target_groups,
        args.exclude_groups,
        args.max_groups,
    )
    if selected_groups.empty:
        raise ValueError("selected groups are empty")
    log(f"selected groups: {len(selected_groups)}")

    selected_groups_path = args.output / out_name(
        "focus_broad_families_selected_groups", args.output_suffix, "csv"
    )
    all_rows_path = args.output / out_name(
        "focus_broad_families_prediction_rows", args.output_suffix, "csv"
    )
    all_rows_parquet_path = args.output / out_name(
        "focus_broad_families_prediction_rows", args.output_suffix, "parquet"
    )
    metrics_path = args.output / out_name(
        "focus_broad_families_metrics_summary", args.output_suffix, "csv"
    )
    ranking_path = args.output / out_name(
        "focus_broad_families_group_ranking", args.output_suffix, "csv"
    )
    largest_path = args.output / out_name(
        "focus_broad_families_largest_error_rows", args.output_suffix, "csv"
    )
    figure_index_path = args.output / out_name(
        "focus_broad_families_figure_index", args.output_suffix, "csv"
    )
    processing_path = args.output / out_name(
        "focus_broad_families_processing_summary", args.output_suffix, "csv"
    )
    selected_groups.to_csv(selected_groups_path, index=False)

    all_rows: list[pd.DataFrame] = []
    all_metrics: list[dict[str, Any]] = []
    all_largest: list[pd.DataFrame] = []
    figure_rows: list[dict[str, Any]] = []
    processing_rows: list[dict[str, Any]] = []

    for _, group_row in selected_groups.iterrows():
        group = str(group_row["material_group_key"])
        safe = str(group_row["safe_group_name"])
        group_fig_dir = args.figures / safe
        group_fig_dir.mkdir(parents=True, exist_ok=True)
        log(f"processing material group: {group}")
        material_rows, filter_col = choose_material_filter(config_df, group)
        rows = clean_rows(material_rows)
        rows = maybe_limit_rows(rows, args.max_rows_per_group)
        if rows.empty:
            processing_rows.append(
                {
                    "material_group_key": group,
                    "safe_group_name": safe,
                    "status": "skipped_no_rows",
                    "extracted_rows": 0,
                    "p_rows": 0,
                    "n_rows": 0,
                    "figures_created": 0,
                    "warnings": "",
                    "report_note": "No rows after config, material group, status, finite sigma, and carrier filters.",
                }
            )
            continue
        rows["selected_material_group_key"] = group
        rows["selected_material_filter_column"] = filter_col
        p_rows = rows[rows["carrier_type"].astype("string").eq("p")].copy()
        n_rows = rows[rows["carrier_type"].astype("string").eq("n")].copy()
        log(f"extracted rows: {len(rows)}")
        log(f"p rows: {len(p_rows)}")
        log(f"n rows: {len(n_rows)}")

        log("computing metrics...")
        metrics_all = compute_metrics(rows, group, safe, "all", len(p_rows), len(n_rows))
        metrics_p = compute_metrics(p_rows, group, safe, "p", len(p_rows), len(n_rows))
        metrics_n = compute_metrics(n_rows, group, safe, "n", len(p_rows), len(n_rows))
        all_metrics.extend([metrics_all, metrics_p, metrics_n])

        log("creating scatter plots...")
        figures_created = 0
        group_rows_csv = args.output / out_name(
            f"focus_broad_families_prediction_rows_{safe}", args.output_suffix, "csv"
        )
        rows.to_csv(group_rows_csv, index=False)

        scatter_all_png = group_fig_dir / out_name(
            f"{safe}_scatter_pred_vs_exp_all", args.output_suffix, "png"
        )
        scatter_all_pdf = group_fig_dir / out_name(
            f"{safe}_scatter_pred_vs_exp_all", args.output_suffix, "pdf"
        )
        save_scatter(
            rows,
            metrics_all,
            f"{group} all",
            scatter_all_png,
            scatter_all_pdf,
            split_carrier=True,
        )
        figures_created += 1
        figure_rows.append(
            {
                "figure_id": f"{safe}_scatter_all",
                "material_group_key": group,
                "safe_group_name": safe,
                "figure_path_png": str(scatter_all_png),
                "figure_path_pdf": str(scatter_all_pdf),
                "title": f"{group} all pred vs exp",
                "source_file": str(group_rows_csv),
                "n_points_plotted": len(rows),
                "carrier_subset": "all",
                "description": "Log-log scatter of measured sigma versus predicted sigma for all p/n rows.",
            }
        )

        warnings = []
        if p_rows.empty:
            warnings.append("p-type rows are zero; p-type scatter plot was not created.")
        else:
            scatter_p_png = group_fig_dir / out_name(
                f"{safe}_scatter_pred_vs_exp_p", args.output_suffix, "png"
            )
            scatter_p_pdf = group_fig_dir / out_name(
                f"{safe}_scatter_pred_vs_exp_p", args.output_suffix, "pdf"
            )
            save_scatter(
                p_rows,
                metrics_p,
                f"{group} p-type",
                scatter_p_png,
                scatter_p_pdf,
                split_carrier=False,
            )
            figures_created += 1
            figure_rows.append(
                {
                    "figure_id": f"{safe}_scatter_p",
                    "material_group_key": group,
                    "safe_group_name": safe,
                    "figure_path_png": str(scatter_p_png),
                    "figure_path_pdf": str(scatter_p_pdf),
                    "title": f"{group} p-type pred vs exp",
                    "source_file": str(group_rows_csv),
                    "n_points_plotted": len(p_rows),
                    "carrier_subset": "p",
                    "description": "Log-log scatter of measured sigma versus predicted sigma for p-type rows.",
                }
            )
        if n_rows.empty:
            warnings.append("n-type rows are zero; n-type scatter plot was not created.")
        else:
            scatter_n_png = group_fig_dir / out_name(
                f"{safe}_scatter_pred_vs_exp_n", args.output_suffix, "png"
            )
            scatter_n_pdf = group_fig_dir / out_name(
                f"{safe}_scatter_pred_vs_exp_n", args.output_suffix, "pdf"
            )
            save_scatter(
                n_rows,
                metrics_n,
                f"{group} n-type",
                scatter_n_png,
                scatter_n_pdf,
                split_carrier=False,
            )
            figures_created += 1
            figure_rows.append(
                {
                    "figure_id": f"{safe}_scatter_n",
                    "material_group_key": group,
                    "safe_group_name": safe,
                    "figure_path_png": str(scatter_n_png),
                    "figure_path_pdf": str(scatter_n_pdf),
                    "title": f"{group} n-type pred vs exp",
                    "source_file": str(group_rows_csv),
                    "n_points_plotted": len(n_rows),
                    "carrier_subset": "n",
                    "description": "Log-log scatter of measured sigma versus predicted sigma for n-type rows.",
                }
            )

        log("creating error histograms...")
        hist_all_png = group_fig_dir / out_name(
            f"{safe}_error_hist_all", args.output_suffix, "png"
        )
        hist_all_pdf = group_fig_dir / out_name(
            f"{safe}_error_hist_all", args.output_suffix, "pdf"
        )
        save_error_hist(rows, metrics_all, hist_all_png, hist_all_pdf, by_carrier=False)
        figures_created += 1
        figure_rows.append(
            {
                "figure_id": f"{safe}_error_hist_all",
                "material_group_key": group,
                "safe_group_name": safe,
                "figure_path_png": str(hist_all_png),
                "figure_path_pdf": str(hist_all_pdf),
                "title": f"{group} log10 prediction error histogram",
                "source_file": str(group_rows_csv),
                "n_points_plotted": len(rows),
                "carrier_subset": "all",
                "description": "Histogram of log10_sigma_pred_over_exp for all rows.",
            }
        )
        if not p_rows.empty and not n_rows.empty:
            hist_by_carrier_png = group_fig_dir / out_name(
                f"{safe}_error_hist_by_carrier", args.output_suffix, "png"
            )
            hist_by_carrier_pdf = group_fig_dir / out_name(
                f"{safe}_error_hist_by_carrier", args.output_suffix, "pdf"
            )
            save_error_hist(
                rows,
                metrics_all,
                hist_by_carrier_png,
                hist_by_carrier_pdf,
                by_carrier=True,
            )
            figures_created += 1
            figure_rows.append(
                {
                    "figure_id": f"{safe}_error_hist_by_carrier",
                    "material_group_key": group,
                    "safe_group_name": safe,
                    "figure_path_png": str(hist_by_carrier_png),
                    "figure_path_pdf": str(hist_by_carrier_pdf),
                    "title": f"{group} log10 prediction error histogram by carrier",
                    "source_file": str(group_rows_csv),
                    "n_points_plotted": len(rows),
                    "carrier_subset": "by_carrier",
                    "description": "Overlaid p-type and n-type histograms of log10_sigma_pred_over_exp.",
                }
            )

        for col in LARGEST_ERROR_COLUMNS:
            if col not in rows.columns:
                rows[col] = pd.NA
        group_largest = (
            rows.sort_values("abs_log10_sigma_pred_over_exp", ascending=False)
            .head(50)[LARGEST_ERROR_COLUMNS]
            .copy()
        )
        all_largest.append(group_largest)
        all_rows.append(rows)

        status = "ok"
        if p_rows.empty:
            status = "warning_no_p"
        elif n_rows.empty:
            status = "warning_no_n"
        log("writing per-group outputs...")
        processing_rows.append(
            {
                "material_group_key": group,
                "safe_group_name": safe,
                "status": status,
                "extracted_rows": len(rows),
                "p_rows": len(p_rows),
                "n_rows": len(n_rows),
                "figures_created": figures_created,
                "warnings": " ".join(warnings),
                "report_note": f"Filtered with {filter_col}.",
            }
        )

    if not all_rows:
        raise ValueError("no material groups produced rows")

    all_rows_df = pd.concat(all_rows, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics, columns=METRIC_COLUMNS)
    processing_df = pd.DataFrame(processing_rows)
    figure_index = pd.DataFrame(
        figure_rows,
        columns=[
            "figure_id",
            "material_group_key",
            "safe_group_name",
            "figure_path_png",
            "figure_path_pdf",
            "title",
            "source_file",
            "n_points_plotted",
            "carrier_subset",
            "description",
        ],
    )
    largest_df = pd.concat(all_largest, ignore_index=True) if all_largest else pd.DataFrame(columns=LARGEST_ERROR_COLUMNS)
    largest_df = (
        largest_df.sort_values("abs_log10_sigma_pred_over_exp", ascending=False)
        .head(1000)
        .copy()
    )

    all_rows_df.to_csv(all_rows_path, index=False)
    all_rows_df.to_parquet(all_rows_parquet_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    largest_df.to_csv(largest_path, index=False)
    processing_df.to_csv(processing_path, index=False)

    log("building ranking...")
    ranking = build_ranking(metrics_df)
    ranking.to_csv(ranking_path, index=False)

    log("writing figure index...")
    figure_index.to_csv(figure_index_path, index=False)

    log("writing report...")
    write_report(
        args.report,
        predictions_path,
        performance_path,
        args.config_id,
        args.selection_mode,
        args.min_rows,
        args.min_samples,
        args.min_papers,
        args.exclude_groups,
        selected_groups,
        processing_df,
        metrics_df,
        ranking,
        largest_df,
        figure_index,
    )

    log("running sanity checks...")
    run_sanity_checks(
        config_df,
        selected_groups,
        all_rows_df,
        processing_df,
        args.config_id,
        metrics_path,
        ranking_path,
        figure_index_path,
        args.report,
        figure_index,
    )
    elapsed = time.perf_counter() - start
    log(f"done. elapsed_sec={elapsed:.2f}")


if __name__ == "__main__":
    main()
