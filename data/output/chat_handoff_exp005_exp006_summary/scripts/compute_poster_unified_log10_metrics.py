#!/usr/bin/env python3
"""Recalculate poster metrics for sigma and ZT on a common log10-ratio scale."""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIGMA_INPUT = (
    ROOT
    / "experiments/exp006/data/processed/step6b_broad_family"
    / "step5b_test_predictions.parquet"
)
DEFAULT_SIGMA_REFERENCE = (
    ROOT
    / "experiments/exp006/data/processed/step6c_broad_family"
    / "step6c_broad_family_default_metrics_for_figures.csv"
)
DEFAULT_ZT_INPUT = (
    ROOT
    / "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison"
    / "three_method_row_comparisons.csv"
)
DEFAULT_ZT_REFERENCE = (
    ROOT
    / "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison"
    / "three_method_model_metrics.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/output/poster_unified_log10_metrics"

SIGMA_CONFIG = (
    "sample_holdout__ref_conservative_valid__eval_all_valid__"
    "material_family__sample_median"
)
SIGMA_EXP_COLUMN = "sigma_S_per_m"
SIGMA_PRED_COLUMN = "sigma_pred_S_per_m"
ZT_EXP_COLUMN = "ZT_obs"
ZT_METHODS = (
    ("exp005_DF", "exp005 DF (reference)", "ZT_pred_exp005_DF"),
    ("exp005_ML", "exp005 ML", "ZT_pred_exp005_ML"),
    ("exp006", "exp006", "ZT_pred_exp006"),
)
EXPECTED_ZT_ROWS = 383
EXPECTED_ZT_SAMPLES = 109
VALIDATION_ATOL = 1e-12


def finite_numeric(series: pd.Series) -> pd.Series:
    """Convert a series to numeric and replace +/- infinity with NaN."""

    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def compute_log10_ratio_metrics(
    y_exp: pd.Series, y_pred: pd.Series
) -> dict[str, float | int | dict[str, int]]:
    """
    Calculate log10(pred/exp) metrics after excluding nonfinite/nonpositive rows.

    Return:
      n_total
      n_valid
      n_excluded
      rmse_log10_pred_over_exp
      mae_log10_pred_over_exp
      factor2_accuracy
      factor10_accuracy
      median_abs_log10_error
      max_abs_log10_error

    The additional ``exclusion_reasons`` mapping records missing, infinity, and
    nonpositive counts. Individual reason counts can overlap between exp/pred.
    """

    if len(y_exp) != len(y_pred):
        raise ValueError("y_exp and y_pred must have the same length")

    exp_raw = pd.to_numeric(y_exp, errors="coerce")
    pred_raw = pd.to_numeric(y_pred, errors="coerce")
    exp = finite_numeric(y_exp)
    pred = finite_numeric(y_pred)

    valid = exp.notna() & pred.notna() & (exp > 0) & (pred > 0)
    n_total = int(len(exp))
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("No finite, positive exp/pred pairs remain after filtering")

    log10_error = np.log10(
        pred.loc[valid].to_numpy(dtype=float) / exp.loc[valid].to_numpy(dtype=float)
    )
    abs_error = np.abs(log10_error)

    reasons = {
        "exp_missing_or_nonnumeric": int(exp_raw.isna().sum()),
        "pred_missing_or_nonnumeric": int(pred_raw.isna().sum()),
        "exp_inf": int(np.isinf(exp_raw.to_numpy(dtype=float)).sum()),
        "pred_inf": int(np.isinf(pred_raw.to_numpy(dtype=float)).sum()),
        "exp_nonpositive": int(((exp_raw <= 0) & np.isfinite(exp_raw)).sum()),
        "pred_nonpositive": int(((pred_raw <= 0) & np.isfinite(pred_raw)).sum()),
    }

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_excluded": n_total - n_valid,
        "rmse_log10_pred_over_exp": float(np.sqrt(np.mean(log10_error**2))),
        "mae_log10_pred_over_exp": float(np.mean(abs_error)),
        "factor2_accuracy": float(np.mean(abs_error <= np.log10(2.0))),
        "factor10_accuracy": float(np.mean(abs_error <= 1.0)),
        "median_abs_log10_error": float(np.median(abs_error)),
        "max_abs_log10_error": float(np.max(abs_error)),
        "exclusion_reasons": reasons,
    }


def metric_columns(metrics: dict[str, object]) -> dict[str, object]:
    """Return only the public metric columns, excluding diagnostic metadata."""

    return {key: value for key, value in metrics.items() if key != "exclusion_reasons"}


def relative_source(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns in {source}: {missing}")


def load_sigma_metrics(
    sigma_input: Path, sigma_reference: Path
) -> tuple[pd.DataFrame, dict[str, object], list[str]]:
    columns = ["config_id", SIGMA_EXP_COLUMN, SIGMA_PRED_COLUMN]
    frame = pd.read_parquet(sigma_input, columns=columns)
    selected = frame.loc[frame["config_id"] == SIGMA_CONFIG].copy()
    if selected.empty:
        raise RuntimeError(f"No sigma rows found for poster config: {SIGMA_CONFIG}")

    metrics = compute_log10_ratio_metrics(
        selected[SIGMA_EXP_COLUMN], selected[SIGMA_PRED_COLUMN]
    )

    reference = pd.read_csv(sigma_reference)
    require_columns(
        reference,
        [
            "default_label",
            "metric_weighting",
            "rmse_log10",
            "mae_log10",
            "factor_2_accuracy",
            "factor_10_accuracy",
        ],
        sigma_reference,
    )
    reference_row = reference.loc[
        (reference["default_label"] == "broad_material_family_default")
        & (reference["metric_weighting"] == "row_equal")
    ]
    if len(reference_row) != 1:
        raise RuntimeError("Could not uniquely identify the sigma poster reference row")

    checks = {
        "rmse_log10_pred_over_exp": "rmse_log10",
        "mae_log10_pred_over_exp": "mae_log10",
        "factor2_accuracy": "factor_2_accuracy",
        "factor10_accuracy": "factor_10_accuracy",
    }
    validation_lines: list[str] = []
    for calculated_name, reference_name in checks.items():
        calculated = float(metrics[calculated_name])
        expected = float(reference_row.iloc[0][reference_name])
        difference = abs(calculated - expected)
        validation_lines.append(
            f"sigma {calculated_name}: calculated={calculated:.15g}, "
            f"reference={expected:.15g}, abs_diff={difference:.3g}"
        )
        if not math.isclose(calculated, expected, rel_tol=0.0, abs_tol=VALIDATION_ATOL):
            raise RuntimeError(
                f"Sigma poster metric mismatch for {calculated_name}: "
                f"calculated={calculated}, reference={expected}"
            )

    output = pd.DataFrame(
        [
            {
                "target": "sigma",
                "dataset": "exp006_step6c_broad_family_sample_holdout_row_equal",
                "method": "exp006_broad_family",
                **metric_columns(metrics),
                "source_file": relative_source(sigma_input),
                "y_exp_column": SIGMA_EXP_COLUMN,
                "y_pred_column": SIGMA_PRED_COLUMN,
                "notes": (
                    "Poster main result; broad material-family sample-median "
                    "prediction; sample holdout; row-equal evaluation."
                ),
            }
        ]
    )
    return output, metrics, validation_lines


def load_zt_metrics(
    zt_input: Path, zt_reference: Path
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], list[str]]:
    frame = pd.read_csv(zt_input)
    required = [ZT_EXP_COLUMN, "sample_key", *[item[2] for item in ZT_METHODS]]
    require_columns(frame, required, zt_input)

    n_samples = int(frame["sample_key"].nunique(dropna=True))
    if len(frame) != EXPECTED_ZT_ROWS or n_samples != EXPECTED_ZT_SAMPLES:
        raise RuntimeError(
            "ZT source does not match the requested common cohort: "
            f"rows={len(frame)} (expected {EXPECTED_ZT_ROWS}), "
            f"samples={n_samples} (expected {EXPECTED_ZT_SAMPLES})"
        )

    references = pd.read_csv(zt_reference).set_index("method_key")
    require_columns(
        references.reset_index(),
        ["method_key", "row_log_RMSE", "row_within_factor_2_rate"],
        zt_reference,
    )

    output_rows: list[dict[str, object]] = []
    metrics_by_method: dict[str, dict[str, object]] = {}
    validation_lines: list[str] = []
    for method_key, method_label, pred_column in ZT_METHODS:
        metrics = compute_log10_ratio_metrics(frame[ZT_EXP_COLUMN], frame[pred_column])
        metrics_by_method[method_key] = metrics
        if method_key not in references.index:
            raise RuntimeError(f"ZT reference row is missing for {method_key}")

        old_ln_rmse = float(references.loc[method_key, "row_log_RMSE"])
        converted_log10_rmse = old_ln_rmse / np.log(10.0)
        calculated_rmse = float(metrics["rmse_log10_pred_over_exp"])
        old_factor2 = float(references.loc[method_key, "row_within_factor_2_rate"])
        calculated_factor2 = float(metrics["factor2_accuracy"])
        rmse_difference = abs(calculated_rmse - converted_log10_rmse)
        factor2_difference = abs(calculated_factor2 - old_factor2)
        validation_lines.append(
            f"ZT {method_key} RMSE: direct_log10={calculated_rmse:.15g}, "
            f"existing_ln/ln(10)={converted_log10_rmse:.15g}, "
            f"abs_diff={rmse_difference:.3g}"
        )
        validation_lines.append(
            f"ZT {method_key} factor-2: calculated={calculated_factor2:.15g}, "
            f"reference={old_factor2:.15g}, abs_diff={factor2_difference:.3g}"
        )
        if not math.isclose(
            calculated_rmse, converted_log10_rmse, rel_tol=0.0, abs_tol=VALIDATION_ATOL
        ):
            raise RuntimeError(f"ZT ln-to-log10 RMSE validation failed for {method_key}")
        if not math.isclose(
            calculated_factor2, old_factor2, rel_tol=0.0, abs_tol=VALIDATION_ATOL
        ):
            raise RuntimeError(f"ZT factor-2 validation failed for {method_key}")

        note = "Reference only; DF is not emphasized as an unknown-sample predictor."
        if method_key == "exp005_ML":
            note = "Poster comparison method; common-cohort row-level evaluation."
        elif method_key == "exp006":
            note = "Current method; common-cohort row-level evaluation."
        output_rows.append(
            {
                "target": "ZT",
                "dataset": "common_cohort_383_rows_109_samples_reported_ZT_pm2K",
                "evaluation_level": "row",
                "method": method_label,
                **metric_columns(metrics),
                "source_file": relative_source(zt_input),
                "y_exp_column": ZT_EXP_COLUMN,
                "y_pred_column": pred_column,
                "notes": note,
            }
        )

    return pd.DataFrame(output_rows), metrics_by_method, validation_lines


def make_summary(sigma: pd.DataFrame, zt: pd.DataFrame) -> str:
    sigma_row = sigma.iloc[0]
    zt_by_method = zt.set_index("method")
    ml = zt_by_method.loc["exp005 ML"]
    current = zt_by_method.loc["exp006"]

    return "\n".join(
        [
            "# Poster unified log10-ratio metrics",
            "",
            "All metrics below use `log10(pred/exp)` and only finite, positive pairs.",
            "",
            "## Electrical conductivity prediction",
            "",
            "| Metric | exp006 |",
            "|---|---:|",
            f"| RMSE(log10 σpred/σexp) | {sigma_row['rmse_log10_pred_over_exp']:.3f} |",
            f"| MAE(log10 σpred/σexp) | {sigma_row['mae_log10_pred_over_exp']:.3f} |",
            f"| Factor-2 accuracy | {100 * sigma_row['factor2_accuracy']:.1f}% |",
            f"| Factor-10 accuracy | {100 * sigma_row['factor10_accuracy']:.1f}% |",
            "",
            (
                f"Valid pairs: {int(sigma_row['n_valid']):,} / "
                f"{int(sigma_row['n_total']):,} rows "
                f"({int(sigma_row['n_excluded']):,} excluded)."
            ),
            "",
            "## ZT prediction on common cohort",
            "",
            "| Metric | exp005 ML | exp006 |",
            "|---|---:|---:|",
            (
                "| RMSE(log10 ZTpred/ZTexp) | "
                f"{ml['rmse_log10_pred_over_exp']:.3f} | "
                f"{current['rmse_log10_pred_over_exp']:.3f} |"
            ),
            (
                "| MAE(log10 ZTpred/ZTexp) | "
                f"{ml['mae_log10_pred_over_exp']:.3f} | "
                f"{current['mae_log10_pred_over_exp']:.3f} |"
            ),
            (
                f"| Factor-2 accuracy | {100 * ml['factor2_accuracy']:.1f}% | "
                f"{100 * current['factor2_accuracy']:.1f}% |"
            ),
            (
                f"| Factor-10 accuracy | {100 * ml['factor10_accuracy']:.1f}% | "
                f"{100 * current['factor10_accuracy']:.1f}% |"
            ),
            "",
            (
                f"Valid pairs per method: {int(ml['n_valid'])} / "
                f"{int(ml['n_total'])} rows ({int(ml['n_excluded'])} excluded)."
            ),
            "",
            "The exp005 DF row in the CSV is included only as a reference and is not "
            "part of the poster's main ML-versus-exp006 comparison.",
            "",
        ]
    )


def make_zt_figure(zt: pd.DataFrame, output_path: Path) -> None:
    main = zt.loc[zt["method"].isin(["exp005 ML", "exp006"])].set_index("method")
    methods = ["exp005 ML", "exp006"]
    colors = ["#E69F00", "#0072B2"]
    panels = [
        ("rmse_log10_pred_over_exp", "RMSE of log10 ratio", False),
        ("mae_log10_pred_over_exp", "MAE of log10 ratio", False),
        ("factor2_accuracy", "Factor-2 accuracy", True),
        ("factor10_accuracy", "Factor-10 accuracy", True),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 13,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (column, title, is_accuracy) in zip(axes.flat, panels):
        values = [float(main.loc[method, column]) for method in methods]
        plotted = [100.0 * value for value in values] if is_accuracy else values
        bars = axis.bar(methods, plotted, color=colors, width=0.62)
        axis.set_title(title, pad=10, weight="bold")
        axis.set_ylabel("Accuracy (%)" if is_accuracy else "log10 units")
        axis.set_ylim(0, max(plotted) * 1.28 if max(plotted) > 0 else 1)
        axis.grid(axis="y", alpha=0.25, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, plotted):
            label = f"{value:.1f}%" if is_accuracy else f"{value:.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + axis.get_ylim()[1] * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=15,
                weight="bold",
            )
    fig.suptitle(
        "ZT prediction: common cohort row-level comparison\n"
        "383 rows / 109 samples / reported ZT within ±2 K",
        fontsize=20,
        weight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def reason_lines(label: str, metrics: dict[str, object]) -> list[str]:
    reasons = metrics["exclusion_reasons"]
    assert isinstance(reasons, dict)
    lines = [
        f"{label}: n_total={metrics['n_total']}, n_valid={metrics['n_valid']}, "
        f"n_excluded={metrics['n_excluded']}"
    ]
    lines.extend(f"  {reason}: {count}" for reason, count in reasons.items())
    lines.append("  Note: reason counts may overlap between experimental and predicted values.")
    return lines


def write_validation_log(
    output_path: Path,
    sigma_input: Path,
    sigma_reference: Path,
    zt_input: Path,
    zt_reference: Path,
    sigma_metrics: dict[str, object],
    zt_metrics: dict[str, dict[str, object]],
    sigma_validation: list[str],
    zt_validation: list[str],
) -> None:
    lines = [
        "Poster unified log10-ratio metrics validation",
        "=" * 48,
        "",
        "Inputs",
        f"sigma prediction rows: {sigma_input.resolve()}",
        f"sigma existing metrics reference: {sigma_reference.resolve()}",
        f"ZT common-cohort rows: {zt_input.resolve()}",
        f"ZT existing metrics reference: {zt_reference.resolve()}",
        "",
        "Columns and evaluation definitions",
        f"sigma: exp={SIGMA_EXP_COLUMN}, pred={SIGMA_PRED_COLUMN}",
        f"sigma config_id={SIGMA_CONFIG}",
        f"ZT: exp={ZT_EXP_COLUMN}",
        *[f"ZT {key}: pred={column}" for key, _, column in ZT_METHODS],
        "ZT evaluation level=row; common cohort=383 rows / 109 samples / ±2 K.",
        "Sample-max and common DOI-group CV results were not loaded or mixed in.",
        "",
        "Sigma reproduction checks (PASS)",
        *sigma_validation,
        "",
        "ZT consistency checks (PASS)",
        "RMSE values below are recalculated directly with np.log10; the existing",
        "natural-log RMSE is used only for an independent ln(10) consistency check.",
        *zt_validation,
        "",
        "Exclusions",
        *reason_lines("sigma exp006_broad_family", sigma_metrics),
    ]
    for method_key, _, _ in ZT_METHODS:
        lines.extend(reason_lines(f"ZT {method_key}", zt_metrics[method_key]))
    lines.extend(
        [
            "",
            "Final status: PASS",
            "No PPTX files or pre-existing result CSV/figure files were modified.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-input", type=Path, default=DEFAULT_SIGMA_INPUT)
    parser.add_argument("--sigma-reference", type=Path, default=DEFAULT_SIGMA_REFERENCE)
    parser.add_argument("--zt-input", type=Path, default=DEFAULT_ZT_INPUT)
    parser.add_argument("--zt-reference", type=Path, default=DEFAULT_ZT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        args.sigma_input,
        args.sigma_reference,
        args.zt_input,
        args.zt_reference,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required input files not found: {missing}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Use --overwrite to replace it."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    sigma, sigma_metrics, sigma_validation = load_sigma_metrics(
        args.sigma_input, args.sigma_reference
    )
    zt, zt_metrics, zt_validation = load_zt_metrics(args.zt_input, args.zt_reference)

    staging = Path(
        tempfile.mkdtemp(prefix="poster_unified_log10_metrics_", dir=output_dir.parent)
    )
    try:
        sigma.to_csv(staging / "sigma_unified_log10_metrics.csv", index=False, encoding="utf-8-sig")
        zt.to_csv(
            staging / "zt_common_cohort_unified_log10_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (staging / "poster_unified_log10_metrics_summary.md").write_text(
            make_summary(sigma, zt), encoding="utf-8"
        )
        write_validation_log(
            staging / "validation_log.txt",
            args.sigma_input,
            args.sigma_reference,
            args.zt_input,
            args.zt_reference,
            sigma_metrics,
            zt_metrics,
            sigma_validation,
            zt_validation,
        )
        make_zt_figure(zt, staging / "figure_zt_unified_log10_metrics.png")

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Wrote unified poster metrics to: {output_dir}")
    print(sigma.to_string(index=False))
    print(zt.to_string(index=False))


if __name__ == "__main__":
    main()
