import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SELECTED_COLUMNS = [
    "material_group_key",
    "safe_group_name",
    "selection_reason",
    "source_summary_rows",
    "expected_n_rows_from_summary",
    "expected_n_samples_from_summary",
    "summary_mae_log10",
    "summary_factor_2_accuracy",
    "summary_factor_10_accuracy",
]

REQUIRED_ROW_COLUMNS = [
    "config_id",
    "prediction_status",
    "row_id",
    "carrier_type",
    "material_group_key",
    "material_group_key_for_prediction",
    "selected_material_group_key",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_over_exp",
]

REQUIRED_METRIC_COLUMNS = [
    "material_group_key",
    "carrier_subset",
    "n_rows",
    "n_samples",
    "n_papers",
    "mae_log10",
    "rmse_log10",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "p_row_count",
    "n_row_count",
]

REQUIRED_RANKING_COLUMNS = [
    "material_group_key",
    "carrier_subset",
    "mae_log10",
    "factor_2_accuracy",
    "factor_10_accuracy",
    "rank_by_mae_log10",
    "rank_by_factor_2_accuracy",
    "rank_by_factor_10_accuracy",
    "rank_by_n_rows",
]

REQUIRED_FIGURE_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check focus broad-family outputs.")
    parser.add_argument(
        "--selected-groups",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_broad_families/focus_broad_families_selected_groups.csv"
        ),
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_broad_families/focus_broad_families_prediction_rows.csv"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_broad_families/focus_broad_families_metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_broad_families/focus_broad_families_group_ranking.csv"
        ),
    )
    parser.add_argument(
        "--figure-index",
        type=Path,
        default=Path(
            "experiments/exp006/data/processed/focus_broad_families/focus_broad_families_figure_index.csv"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/exp006/reports/focus_broad_families/focus_broad_families_report.md"
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
    for path in [
        args.selected_groups,
        args.rows,
        args.metrics,
        args.ranking,
        args.figure_index,
        args.report,
    ]:
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    selected = pd.read_csv(args.selected_groups, low_memory=False)
    rows = pd.read_csv(args.rows, low_memory=False)
    metrics = pd.read_csv(args.metrics, low_memory=False)
    ranking = pd.read_csv(args.ranking, low_memory=False)
    figure_index = pd.read_csv(args.figure_index, low_memory=False)
    report_text = args.report.read_text(encoding="utf-8")

    require_columns(selected, REQUIRED_SELECTED_COLUMNS, "selected_groups", failures)
    require_columns(rows, REQUIRED_ROW_COLUMNS, "rows", failures)
    require_columns(metrics, REQUIRED_METRIC_COLUMNS, "metrics", failures)
    require_columns(ranking, REQUIRED_RANKING_COLUMNS, "ranking", failures)
    require_columns(figure_index, REQUIRED_FIGURE_COLUMNS, "figure_index", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if selected.empty:
        failures.append("selected groups table is empty")
    if rows.empty:
        failures.append("rows table is empty")
    if metrics.empty:
        failures.append("metrics table is empty")
    if ranking.empty:
        failures.append("ranking table is empty")
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

    row_groups = set(rows["selected_material_group_key"].astype(str))
    selected_groups = set(selected["material_group_key"].astype(str))
    if not row_groups.issubset(selected_groups):
        failures.append("rows contain material groups not present in selected groups")

    subset_counts = metrics.groupby("material_group_key")["carrier_subset"].apply(set)
    for group, subsets in subset_counts.items():
        if subsets != {"all", "p", "n"}:
            failures.append(f"metrics for {group} must contain all/p/n rows")

    scatter_all = figure_index[
        (figure_index["carrier_subset"].eq("all"))
        & figure_index["figure_id"].astype(str).str.endswith("_scatter_all")
    ]
    if set(scatter_all["material_group_key"].astype(str)) != row_groups:
        failures.append("not every processed group has an all scatter figure")
    if not figure_index["figure_id"].is_unique:
        failures.append("figure_id values are not unique")
    for _, row in figure_index.iterrows():
        for key in ["figure_path_png", "figure_path_pdf"]:
            path = Path(row[key])
            if not path.exists() or path.stat().st_size == 0:
                failures.append(f"missing or empty figure: {path}")

    required_report_phrases = [
        "No new sigma_pred values are calculated.",
        "Step4 full-data reference curves are not used.",
        "Starrydata2 raw data is not read.",
        "broad_family classification is heuristic",
    ]
    for phrase in required_report_phrases:
        if phrase not in report_text:
            failures.append(f"report missing note: {phrase}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    all_metrics = metrics[metrics["carrier_subset"].eq("all")]
    print(f"selected groups: {len(selected)}")
    print(f"processed groups: {len(row_groups)}")
    print(f"rows: {len(rows)}")
    print(f"figures: {len(figure_index)}")
    print("top by MAE:")
    print(
        ranking.sort_values("rank_by_mae_log10")[
            ["material_group_key", "n_rows", "mae_log10", "factor_2_accuracy", "factor_10_accuracy"]
        ]
        .head(5)
        .to_string(index=False)
    )
    print("carrier counts:")
    print(
        all_metrics[
            ["material_group_key", "n_rows", "p_row_count", "n_row_count"]
        ].to_string(index=False)
    )
    print("focus broad-family output checks passed")


if __name__ == "__main__":
    main()
