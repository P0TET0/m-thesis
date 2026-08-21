import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_focus_broad_families_uniform_axis_scatter import (
    DEFAULT_BROAD_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_SIGE_DATA,
    FIXED_LOWER,
    FIXED_UPPER,
    GROUP_SPECS,
    clean_rows,
    source_path,
    subset_rows,
)


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "focus_broad_families_uniform_axes"
    / "focus_broad_families_y_equals_x_metrics_report.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate agreement metrics with the y=x line for all focus "
            "material-classification scatter plots."
        )
    )
    parser.add_argument("--broad-data", type=Path, default=DEFAULT_BROAD_DATA)
    parser.add_argument("--sige-data", type=Path, default=DEFAULT_SIGE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--axis-lower", type=float, default=FIXED_LOWER)
    parser.add_argument("--axis-upper", type=float, default=FIXED_UPPER)
    return parser.parse_args()


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.mean((x - mean_x) ** 2))
    var_y = float(np.mean((y - mean_y) ** 2))
    covariance = float(np.mean((x - mean_x) * (y - mean_y)))
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator == 0.0:
        return float("nan")
    return 2.0 * covariance / denominator


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calculate_metrics(
    df: pd.DataFrame, axis_lower: float, axis_upper: float
) -> dict[str, Any]:
    log_exp = np.log10(df["sigma_S_per_m"].to_numpy(dtype=float))
    log_pred = np.log10(df["sigma_pred_S_per_m"].to_numpy(dtype=float))
    residual = log_pred - log_exp
    abs_residual = np.abs(residual)
    orthogonal_distance = abs_residual / math.sqrt(2.0)
    inside = df["sigma_S_per_m"].between(axis_lower, axis_upper) & df[
        "sigma_pred_S_per_m"
    ].between(axis_lower, axis_upper)

    mean_abs_log10 = float(np.mean(abs_residual))
    median_abs_log10 = float(np.median(abs_residual))
    mean_bias = float(np.mean(residual))
    return {
        "n_rows": int(len(df)),
        "n_visible_in_fixed_axes": int(inside.sum()),
        "visible_fraction_in_fixed_axes": float(inside.mean()),
        # In equal-scale log10 coordinates, |log10(pred)-log10(exp)|/sqrt(2)
        # is the shortest distance from a point to y=x.
        "mean_orthogonal_distance_log10_decades": float(
            np.mean(orthogonal_distance)
        ),
        "median_orthogonal_distance_log10_decades": float(
            np.median(orthogonal_distance)
        ),
        "rmse_orthogonal_distance_log10_decades": float(
            np.sqrt(np.mean(orthogonal_distance**2))
        ),
        "mae_log10_pred_over_exp": mean_abs_log10,
        "median_abs_log10_pred_over_exp": median_abs_log10,
        "rmse_log10_pred_over_exp": float(np.sqrt(np.mean(residual**2))),
        "mean_absolute_error_factor": float(10.0**mean_abs_log10),
        "median_absolute_error_factor": float(10.0**median_abs_log10),
        "mean_log10_bias_pred_over_exp": mean_bias,
        "multiplicative_bias_pred_over_exp": float(10.0**mean_bias),
        "factor_2_fraction": float(np.mean(abs_residual <= math.log10(2.0))),
        "factor_5_fraction": float(np.mean(abs_residual <= math.log10(5.0))),
        "factor_10_fraction": float(np.mean(abs_residual <= 1.0)),
        "overprediction_fraction": float(np.mean(residual > 0.0)),
        "underprediction_fraction": float(np.mean(residual < 0.0)),
        "lins_ccc_log10": lins_ccc(log_exp, log_pred),
        "pearson_r_log10": pearson_r(log_exp, log_pred),
    }


def format_value(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def write_report(
    report_path: Path,
    metrics: pd.DataFrame,
    ranking: pd.DataFrame,
    metrics_path: Path,
    ranking_path: Path,
) -> None:
    lines = [
        "# y=x proximity metrics for focus broad families",
        "",
        "## 指標の意味",
        "",
        "- `mean_orthogonal_distance_log10_decades`: log10 座標上で各点から y=x までの最短距離の平均。0に近いほど良い。主ランキング指標。",
        "- `mean_absolute_error_factor`: 平均絶対log10誤差を倍率へ戻した値。1に近いほど良い。例: 2なら平均的なずれが約2倍。",
        "- `lins_ccc_log10`: log10(実測値)とlog10(予測値)の一致相関係数。1に近いほどy=xへの一致が良い。",
        "- `factor_2/5/10_fraction`: 予測値が実測値の各倍率以内に入った割合。1に近いほど良い。",
        "- `multiplicative_bias_pred_over_exp`: 系統的な予測/実測比。1が無偏り、1超は過大予測、1未満は過小予測。",
        "- 指標は軸範囲外も含む全有効点から算出。`n_visible_in_fixed_axes` は固定軸図に表示される点数。",
        "",
        "## 出力",
        "",
        f"- 全 all/p/n 指標: `{metrics_path}`",
        f"- 材料系 all のランキング: `{ranking_path}`",
        "",
        "## 材料系ランキング（all）",
        "",
        "| Rank | 材料系 | n | 平均直交距離 | 平均倍率誤差 | Lin's CCC | factor 2 | factor 5 | factor 10 | バイアス倍率 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"| {int(row['rank_by_mean_orthogonal_distance'])} | "
            f"{row['material_classification_ja']} | {int(row['n_rows'])} | "
            f"{format_value(row['mean_orthogonal_distance_log10_decades'])} | "
            f"{format_value(row['mean_absolute_error_factor'], 2)} | "
            f"{format_value(row['lins_ccc_log10'])} | "
            f"{100.0 * row['factor_2_fraction']:.1f}% | "
            f"{100.0 * row['factor_5_fraction']:.1f}% | "
            f"{100.0 * row['factor_10_fraction']:.1f}% | "
            f"{format_value(row['multiplicative_bias_pred_over_exp'], 2)} |"
        )
    lines.extend(
        [
            "",
            "注: 直交距離は図上のx軸・y軸が同じlog10スケールであることを前提とする。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.axis_lower < args.axis_upper):
        raise ValueError("axis limits must satisfy 0 < lower < upper")
    args.output.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    for material_order, (
        classification_ja,
        group_key,
        safe_name,
        source_kind,
    ) in enumerate(GROUP_SPECS, start=1):
        path = source_path(args.broad_data, args.sige_data, safe_name, source_kind)
        if not path.exists():
            raise FileNotFoundError(f"missing source for {classification_ja}: {path}")
        rows = clean_rows(pd.read_csv(path, low_memory=False))
        for subset_order, subset in enumerate(["all", "p", "n"]):
            current = subset_rows(rows, subset)
            if current.empty:
                continue
            result_rows.append(
                {
                    "material_order": material_order,
                    "subset_order": subset_order,
                    "material_classification_ja": classification_ja,
                    "material_group_key": group_key,
                    "safe_group_name": safe_name,
                    "carrier_subset": subset,
                    **calculate_metrics(current, args.axis_lower, args.axis_upper),
                    "source_csv": str(path),
                }
            )

    metrics = pd.DataFrame(result_rows).sort_values(
        ["material_order", "subset_order"]
    )
    metrics_path = args.output / "focus_broad_families_y_equals_x_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    ranking = metrics[metrics["carrier_subset"].eq("all")].copy()
    ranking = ranking.sort_values(
        "mean_orthogonal_distance_log10_decades", ascending=True
    ).reset_index(drop=True)
    ranking.insert(0, "rank_by_mean_orthogonal_distance", np.arange(1, len(ranking) + 1))
    ranking_path = args.output / "focus_broad_families_y_equals_x_ranking_all.csv"
    ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    write_report(args.report, metrics, ranking, metrics_path, ranking_path)

    if len(metrics) != 42:
        raise ValueError(f"expected 42 all/p/n metric rows, got {len(metrics)}")
    if len(ranking) != len(GROUP_SPECS):
        raise ValueError(f"expected 15 all rankings, got {len(ranking)}")
    numeric = metrics.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise ValueError("metrics contain infinite values")
    if not metrics["mean_orthogonal_distance_log10_decades"].ge(0.0).all():
        raise ValueError("orthogonal distance must be non-negative")
    for path in [metrics_path, ranking_path, args.report]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {path}")

    print(f"[y_equals_x_metrics] metric rows: {len(metrics)}", flush=True)
    print(f"[y_equals_x_metrics] ranking rows: {len(ranking)}", flush=True)
    print(f"[y_equals_x_metrics] metrics: {metrics_path}", flush=True)
    print(f"[y_equals_x_metrics] ranking: {ranking_path}", flush=True)
    print(f"[y_equals_x_metrics] report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
