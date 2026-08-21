"""Quantify how closely Step33 scatter points follow y=x in log-log space."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/"
    "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv"
)
DEFAULT_INDEX = (
    PROJECT_ROOT
    / "data/output/starrydata2_step32_sigma_ml_by_material_system/material_system_index_step32.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/starrydata2_step34_sigma_ml_y_equals_x_metrics"
AXIS_MIN = 1e2
AXIS_MAX = 1e8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate per-material-system agreement with the y=x line for Step33 scatter data."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--material-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")


def family_order(data: pd.DataFrame, index_path: Path) -> list[str]:
    present = set(data["material_group_key_broad_family"].astype(str))
    ordered: list[str] = []
    if index_path.exists():
        index = pd.read_csv(index_path)
        if "material_group_key_broad_family" in index.columns:
            ordered = list(dict.fromkeys(index["material_group_key_broad_family"].astype(str)))
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x))
    var_y = float(np.var(y))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0 else math.nan


def identity_metrics(data: pd.DataFrame) -> dict[str, Any]:
    if data.empty:
        return {
            "point_count": 0,
            "mae_log10": math.nan,
            "rmse_log10": math.nan,
            "median_abs_error_log10": math.nan,
            "bias_log10": math.nan,
            "mean_perpendicular_distance_log10": math.nan,
            "rmse_perpendicular_distance_log10": math.nan,
            "mean_multiplicative_error_factor": math.nan,
            "median_multiplicative_error_factor": math.nan,
            "rmse_multiplicative_error_factor": math.nan,
            "within_factor_1_25_rate": math.nan,
            "within_factor_1_5_rate": math.nan,
            "within_factor_2_rate": math.nan,
            "within_factor_3_rate": math.nan,
            "within_factor_10_rate": math.nan,
            "identity_r2_log10": math.nan,
            "concordance_correlation_log10": math.nan,
            "pearson_r_log10": math.nan,
            "fitted_slope_log10": math.nan,
            "fitted_intercept_log10": math.nan,
        }
    x = np.log10(data["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
    y = np.log10(data["sigma_pred_ML_S_per_m_step32"].to_numpy(dtype=float))
    error = y - x
    absolute = np.abs(error)
    mae = float(np.mean(absolute))
    rmse = float(np.sqrt(np.mean(error**2)))
    median_abs = float(np.median(absolute))
    denominator = float(np.sum((x - np.mean(x)) ** 2))
    identity_r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else math.nan
    if len(x) >= 2 and np.std(x) > 0 and np.std(y) > 0:
        pearson = float(np.corrcoef(x, y)[0, 1])
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(slope)
        intercept = float(intercept)
    else:
        pearson = math.nan
        slope = math.nan
        intercept = math.nan
    return {
        "point_count": int(len(data)),
        "mae_log10": mae,
        "rmse_log10": rmse,
        "median_abs_error_log10": median_abs,
        "bias_log10": float(np.mean(error)),
        "mean_perpendicular_distance_log10": mae / math.sqrt(2.0),
        "rmse_perpendicular_distance_log10": rmse / math.sqrt(2.0),
        "mean_multiplicative_error_factor": float(10**mae),
        "median_multiplicative_error_factor": float(10**median_abs),
        "rmse_multiplicative_error_factor": float(10**rmse),
        "within_factor_1_25_rate": float(np.mean(absolute <= math.log10(1.25))),
        "within_factor_1_5_rate": float(np.mean(absolute <= math.log10(1.5))),
        "within_factor_2_rate": float(np.mean(absolute <= math.log10(2.0))),
        "within_factor_3_rate": float(np.mean(absolute <= math.log10(3.0))),
        "within_factor_10_rate": float(np.mean(absolute <= 1.0)),
        "identity_r2_log10": identity_r2,
        "concordance_correlation_log10": concordance_correlation(x, y),
        "pearson_r_log10": pearson,
        "fitted_slope_log10": slope,
        "fitted_intercept_log10": intercept,
    }


def prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def reliability_label(point_count: int) -> str:
    if point_count >= 100:
        return "high_n_ge_100"
    if point_count >= 30:
        return "moderate_n_30_to_99"
    if point_count > 0:
        return "low_n_lt_30"
    return "not_evaluated"


def build_metrics(data: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    groups = [(family, data[data["material_group_key_broad_family"].eq(family)]) for family in families]
    groups.append(("__all_material_systems__", data))
    for family, subset in groups:
        visible_mask = (
            subset["sigma_obs_S_per_m_step11"].between(AXIS_MIN, AXIS_MAX)
            & subset["sigma_pred_ML_S_per_m_step32"].between(AXIS_MIN, AXIS_MAX)
        )
        visible = subset.loc[visible_mask]
        record: dict[str, Any] = {
            "material_group_key_broad_family": family,
            "total_point_count": int(len(subset)),
            "visible_point_count_1e2_1e8": int(len(visible)),
            "clipped_point_count_1e2_1e8": int((~visible_mask).sum()),
            "visible_metric_reliability": reliability_label(len(visible)),
        }
        record.update(prefix_metrics(identity_metrics(subset), "all_points"))
        record.update(prefix_metrics(identity_metrics(visible), "visible_1e2_1e8"))
        records.append(record)
    return pd.DataFrame(records)


def build_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("mae_log10", "mean(|log10(predicted) - log10(measured)|)", "0 is ideal; 10^value is the mean multiplicative-error factor"),
            ("rmse_log10", "sqrt(mean((log10(predicted) - log10(measured))^2))", "0 is ideal; emphasizes large departures"),
            ("median_abs_error_log10", "median(|log10(predicted) - log10(measured)|)", "0 is ideal; robust typical departure"),
            ("bias_log10", "mean(log10(predicted) - log10(measured))", "0 is ideal; positive=overprediction, negative=underprediction"),
            ("mean_perpendicular_distance_log10", "mae_log10 / sqrt(2)", "mean perpendicular distance from y=x in equal log10 axes"),
            ("rmse_perpendicular_distance_log10", "rmse_log10 / sqrt(2)", "RMS perpendicular distance from y=x in equal log10 axes"),
            ("mean_multiplicative_error_factor", "10^mae_log10", "1 is ideal; e.g. 2 means factor-of-2 mean log error"),
            ("within_factor_2_rate", "fraction with 0.5 <= predicted/measured <= 2", "1 is ideal"),
            ("identity_r2_log10", "1 - sum(log-error^2)/sum((log10(measured)-mean)^2)", "1 is ideal; may be negative"),
            ("concordance_correlation_log10", "Lin concordance correlation on log10 values", "1 is ideal; combines correlation and y=x agreement"),
            ("pearson_r_log10", "Pearson correlation on log10 values", "Correlation only; does not by itself measure y=x agreement"),
            ("fitted_slope_log10", "OLS slope of log10(predicted) vs log10(measured)", "1 is ideal together with intercept=0"),
            ("fitted_intercept_log10", "OLS intercept of log10(predicted) vs log10(measured)", "0 is ideal together with slope=1"),
        ],
        columns=["metric", "definition", "interpretation"],
    )


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def write_report(path: Path, metrics: pd.DataFrame) -> None:
    overall = metrics[metrics["material_group_key_broad_family"].eq("__all_material_systems__")].iloc[0]
    ranked = metrics[
        ~metrics["material_group_key_broad_family"].eq("__all_material_systems__")
        & metrics["visible_1e2_1e8_point_count"].gt(0)
    ].sort_values(["visible_1e2_1e8_rmse_log10", "material_group_key_broad_family"])
    lines = [
        "Step34: y=x線への一致度（exp005 ML sigma）",
        "=========================================",
        "",
        "対象",
        "----",
        "Step33散布図と同じDOIグループ・ホールドアウト点を使用。",
        "visible_1e2_1e8_* は散布図に表示された10^2～10^8 S/m内の点、",
        "all_points_* は表示範囲外を含む全正値点に対する指標。",
        "",
        "主要指標",
        "--------",
        "- RMSE(log10): 0に近いほどy=xに近い。大きな外れを強く評価する。",
        "- MAE(log10): 0に近いほどy=xに近い。10^MAEが平均的な倍率誤差。",
        "- Factor-2率: 予測/実測が0.5～2に入る割合。1に近いほど良い。",
        "- CCC(log10): 相関とy=xへの一致を同時評価。1が理想。",
        "",
        "全材料系（表示範囲内）",
        "----------------------",
        f"点数: {int(overall['visible_1e2_1e8_point_count']):,}",
        f"RMSE(log10): {format_value(overall['visible_1e2_1e8_rmse_log10'])}",
        f"MAE(log10): {format_value(overall['visible_1e2_1e8_mae_log10'])}",
        f"平均倍率誤差: {format_value(overall['visible_1e2_1e8_mean_multiplicative_error_factor'])}",
        f"Factor-2率: {format_value(overall['visible_1e2_1e8_within_factor_2_rate'])}",
        f"CCC(log10): {format_value(overall['visible_1e2_1e8_concordance_correlation_log10'])}",
        "",
        "材料系別（表示範囲内、RMSE(log10)昇順）",
        "-----------------------------------------",
    ]
    for row in ranked.itertuples(index=False):
        lines.append(
            f"- {row.material_group_key_broad_family}: n={int(row.visible_1e2_1e8_point_count)}, "
            f"RMSE={format_value(row.visible_1e2_1e8_rmse_log10)}, "
            f"MAE={format_value(row.visible_1e2_1e8_mae_log10)}, "
            f"倍率誤差={format_value(row.visible_1e2_1e8_mean_multiplicative_error_factor)}, "
            f"Factor-2={format_value(row.visible_1e2_1e8_within_factor_2_rate)}, "
            f"CCC={format_value(row.visible_1e2_1e8_concordance_correlation_log10)}, "
            f"信頼度={row.visible_metric_reliability}"
        )
    lines.extend(
        [
            "",
            "注意",
            "----",
            "点数が30未満の材料系は指標が不安定。特に分類不能はn=2。",
            "Pearson相関が高くても、傾き1・切片0でなければy=xへの一致が良いとは限らない。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists; pass --replace-output to replace it: {output}")
    output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.input, low_memory=False)
    require_columns(
        data,
        [
            "material_group_key_broad_family",
            "sigma_obs_S_per_m_step11",
            "sigma_pred_ML_S_per_m_step32",
        ],
    )
    for column in ["sigma_obs_S_per_m_step11", "sigma_pred_ML_S_per_m_step32"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        np.isfinite(data["sigma_obs_S_per_m_step11"])
        & data["sigma_obs_S_per_m_step11"].gt(0)
        & np.isfinite(data["sigma_pred_ML_S_per_m_step32"])
        & data["sigma_pred_ML_S_per_m_step32"].gt(0)
    ].copy()

    families = family_order(data, args.material_index)
    metrics = build_metrics(data, families)
    definitions = build_definitions()
    ranked = metrics[
        ~metrics["material_group_key_broad_family"].eq("__all_material_systems__")
    ].sort_values(
        ["visible_1e2_1e8_rmse_log10", "material_group_key_broad_family"],
        na_position="last",
    )
    metrics.to_csv(output / "y_equals_x_metrics_by_material_system_step34.csv", index=False)
    ranked.to_csv(output / "y_equals_x_metrics_ranked_visible_step34.csv", index=False)
    definitions.to_csv(output / "y_equals_x_metric_definitions_step34.csv", index=False)
    write_report(output / "step34_y_equals_x_metrics_report.txt", metrics)

    overall = metrics[metrics["material_group_key_broad_family"].eq("__all_material_systems__")]
    if len(overall) != 1 or int(overall.iloc[0]["all_points_point_count"]) != len(data):
        raise AssertionError("Overall metric row does not match input row count")
    print("Done.")
    print(f"Output: {output}")
    print(f"Material systems: {len(families)}")
    print(f"All points: {len(data):,}")
    print(f"Visible points: {int(overall.iloc[0]['visible_1e2_1e8_point_count']):,}")
    print(f"Visible RMSE(log10): {overall.iloc[0]['visible_1e2_1e8_rmse_log10']:.6g}")
    print(f"Visible factor-2 rate: {overall.iloc[0]['visible_1e2_1e8_within_factor_2_rate']:.6g}")


if __name__ == "__main__":
    main()
