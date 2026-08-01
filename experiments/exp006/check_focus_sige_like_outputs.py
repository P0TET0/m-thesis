import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ROW_COLUMNS = [
    "config_id",
    "prediction_status",
    "row_id",
    "carrier_type",
    "material_group_key",
    "material_group_key_for_prediction",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_over_exp",
]

REQUIRED_METRIC_COLUMNS = [
    "subset",
    "n_rows",
    "n_samples",
    "n_papers",
    "mae_log10",
    "rmse_log10",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
]

REQUIRED_FIGURE_COLUMNS = [
    "figure_id",
    "figure_path_png",
    "figure_path_pdf",
    "title",
    "source_file",
    "n_points_plotted",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check focus SiGe_like output files.")
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_sige_like/focus_sige_like_prediction_rows.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_sige_like/focus_sige_like_metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--figure-index",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_sige_like/focus_sige_like_figure_index.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/exp006/reports/focus_sige_like/focus_sige_like_report.md"
        ),
    )
    return parser.parse_args()


def require_columns(
    df: pd.DataFrame, columns: list[str], label: str, failures: list[str]
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.rows, args.metrics, args.figure_index, args.report]:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    rows = pd.read_csv(args.rows, low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)
    figure_index = pd.read_csv(args.figure_index, low_memory=False)
    report_text = args.report.read_text(encoding="utf-8")

    require_columns(rows, REQUIRED_ROW_COLUMNS, "rows", failures)
    require_columns(metrics, REQUIRED_METRIC_COLUMNS, "metrics", failures)
    require_columns(figure_index, REQUIRED_FIGURE_COLUMNS, "figure_index", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if rows.empty:
        failures.append("rows table is empty")
    if metrics.empty:
        failures.append("metrics table is empty")
    if figure_index.empty:
        failures.append("figure index is empty")

    if rows["config_id"].nunique(dropna=False) != 1:
        failures.append("rows contain multiple config_id values")
    if not rows["prediction_status"].astype("string").eq("ok").all():
        failures.append("prediction_status is not all ok")
    sigma = pd.to_numeric(rows["sigma_S_per_m"], errors="coerce")
    sigma_pred = pd.to_numeric(rows["sigma_pred_S_per_m"], errors="coerce")
    log_error = pd.to_numeric(rows["log10_sigma_pred_over_exp"], errors="coerce")
    if not (np.isfinite(sigma) & (sigma > 0)).all():
        failures.append("sigma_S_per_m contains non-finite or non-positive values")
    if not (np.isfinite(sigma_pred) & (sigma_pred > 0)).all():
        failures.append("sigma_pred_S_per_m contains non-finite or non-positive values")
    if not rows["carrier_type"].astype("string").isin(["p", "n"]).all():
        failures.append("carrier_type contains values other than p/n")
    recomputed = np.log10(sigma_pred / sigma)
    max_delta = np.nanmax(np.abs(log_error - recomputed))
    if not math.isfinite(float(max_delta)) or float(max_delta) > 1e-8:
        failures.append(f"log10 error relation mismatch: max_delta={max_delta}")

    expected_subsets = ["all", "p", "n"]
    if list(metrics["subset"]) != expected_subsets:
        failures.append(f"metrics subset rows must be {expected_subsets}")

    figure_ids = set(figure_index["figure_id"].astype(str))
    if "scatter_all" not in figure_ids:
        failures.append("all scatter figure is missing from figure index")
    if not ({"scatter_p", "scatter_n"} & figure_ids):
        failures.append("neither p nor n scatter figure is present")
    if "error_hist_all" not in figure_ids:
        failures.append("error histogram is missing from figure index")
    for _, row in figure_index.iterrows():
        for key in ["figure_path_png", "figure_path_pdf"]:
            path = Path(row[key])
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"missing or empty figure: {path}")

    required_report_phrases = [
        "No new sigma_pred values are calculated.",
        "Step4 full-data reference curves are not used.",
        "Starrydata2 raw data is not read.",
    ]
    for phrase in required_report_phrases:
        if phrase not in report_text:
            failures.append(f"report missing note: {phrase}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    all_metric = metrics[metrics["subset"].eq("all")].iloc[0]
    p_metric = metrics[metrics["subset"].eq("p")].iloc[0]
    n_metric = metrics[metrics["subset"].eq("n")].iloc[0]
    print(f"rows: {len(rows)}")
    print(f"config_id: {rows['config_id'].iloc[0]}")
    print(f"p rows: {(rows['carrier_type'].astype('string') == 'p').sum()}")
    print(f"n rows: {(rows['carrier_type'].astype('string') == 'n').sum()}")
    print(
        "all metrics: "
        f"MAE={all_metric['mae_log10']:.6g}, "
        f"RMSE={all_metric['rmse_log10']:.6g}, "
        f"factor2={all_metric['factor_2_accuracy']:.6g}, "
        f"factor10={all_metric['factor_10_accuracy']:.6g}"
    )
    print(
        "p metrics: "
        f"MAE={p_metric['mae_log10']:.6g}, "
        f"RMSE={p_metric['rmse_log10']:.6g}, "
        f"factor2={p_metric['factor_2_accuracy']:.6g}, "
        f"factor10={p_metric['factor_10_accuracy']:.6g}"
    )
    print(
        "n metrics: "
        f"MAE={n_metric['mae_log10']:.6g}, "
        f"RMSE={n_metric['rmse_log10']:.6g}, "
        f"factor2={n_metric['factor_2_accuracy']:.6g}, "
        f"factor10={n_metric['factor_10_accuracy']:.6g}"
    )
    print(f"figures: {len(figure_index)}")
    print("focus SiGe_like output checks passed")


if __name__ == "__main__":
    main()
