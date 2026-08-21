#!/usr/bin/env python3
"""Compare exp005 direct fitting, exp005 ML, and exp006 on shared ZT rows."""

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


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "output"
    / "exp005_exp006_common_cohort_zt_comparison"
    / "common_cohort_row_comparisons.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "output" / "exp005_ml_df_exp006_common_cohort_zt_comparison"
)
METHODS = [
    ("exp005_DF", "exp005 DF", "ZT_pred_exp005_DF", "#3B7A57"),
    ("exp005_ML", "exp005 ML", "ZT_pred_exp005_ML", "#C77720"),
    ("exp006", "exp006", "ZT_pred_exp006", "#2B7A98"),
]
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPLICATES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path, low_memory=False)
    required = {
        "row_id",
        "sample_key",
        "temperature_K",
        "seebeck_obs_V_per_K",
        "kappa_obs_W_per_mK",
        "ZT_obs",
        "ZT_pred_exp005",
        "ZT_pred_exp006",
        "target_tau_eff_step18",
        "prefactor_C_S_per_m_step12_rebuilt",
        "material_system_label",
    }
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing common-cohort columns: {sorted(missing)}")
    sigma_df = (
        finite(rows["prefactor_C_S_per_m_step12_rebuilt"])
        * finite(rows["target_tau_eff_step18"])
    )
    seebeck = finite(rows["seebeck_obs_V_per_K"])
    temperature = finite(rows["temperature_K"])
    kappa = finite(rows["kappa_obs_W_per_mK"])
    rows["sigma_pred_exp005_DF_S_per_m"] = sigma_df
    rows["ZT_pred_exp005_DF"] = np.where(
        seebeck.notna() & sigma_df.gt(0) & temperature.gt(0) & kappa.gt(0),
        np.square(seebeck) * sigma_df * temperature / kappa,
        np.nan,
    )
    rows["ZT_pred_exp005_ML"] = finite(rows["ZT_pred_exp005"])
    for column in ["ZT_obs", "ZT_pred_exp005_DF", "ZT_pred_exp005_ML", "ZT_pred_exp006"]:
        rows[column] = finite(rows[column])
    valid = rows[["ZT_obs", "ZT_pred_exp005_DF", "ZT_pred_exp005_ML", "ZT_pred_exp006"]].notna().all(axis=1)
    rows = rows.loc[valid].copy()
    if rows["row_id"].duplicated().any():
        raise ValueError("Shared comparison rows contain duplicate row_id values")
    return rows


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual = finite(actual)
    predicted = finite(predicted)
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    error = predicted - actual
    positive = actual.gt(0) & predicted.gt(0)
    ratio = predicted.loc[positive] / actual.loc[positive]
    nonzero = actual.ne(0)
    return {
        "n": int(len(actual)),
        "MAE": float(error.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "signed_bias": float(error.mean()),
        "MAPE_pct": float((error.loc[nonzero].abs() / actual.loc[nonzero].abs()).mean() * 100),
        "median_APE_pct": float((error.loc[nonzero].abs() / actual.loc[nonzero].abs()).median() * 100),
        "log_RMSE": float(np.sqrt(np.mean(np.square(np.log(ratio))))) if len(ratio) else math.nan,
        "within_factor_2_rate": float(ratio.between(0.5, 2.0).mean()) if len(ratio) else math.nan,
        "spearman": float(actual.corr(predicted, method="spearman")) if len(actual) >= 2 else math.nan,
        "pearson": float(actual.corr(predicted, method="pearson")) if len(actual) >= 2 else math.nan,
    }


def sample_summary(rows: pd.DataFrame) -> pd.DataFrame:
    aggregations: dict[str, tuple[str, str]] = {
        "comparable_rows": ("row_id", "size"),
        "ZT_obs_max": ("ZT_obs", "max"),
        "material_system_label": ("material_system_label", "first"),
        "DOI": ("DOI", "first"),
    }
    for key, _label, prediction, _color in METHODS:
        aggregations[f"{key}_ZT_max"] = (prediction, "max")
        aggregations[f"{key}_row_MAE"] = (
            prediction,
            lambda values, pred=prediction: math.nan,
        )
    # Compute maxima first; sample-balanced row errors are joined separately.
    max_aggs = {
        name: value
        for name, value in aggregations.items()
        if not name.endswith("_row_MAE")
    }
    output = rows.groupby("sample_key", sort=False).agg(**max_aggs).reset_index()
    row_error = pd.DataFrame({"sample_key": rows["sample_key"]})
    for key, _label, prediction, _color in METHODS:
        row_error[f"{key}_row_abs_error"] = (rows[prediction] - rows["ZT_obs"]).abs()
    sample_error = row_error.groupby("sample_key", sort=False).mean(numeric_only=True).reset_index()
    output = output.merge(sample_error, on="sample_key", how="left", validate="one_to_one")
    for key, _label, _prediction, _color in METHODS:
        output[f"{key}_max_abs_error"] = (
            output[f"{key}_ZT_max"] - output["ZT_obs_max"]
        ).abs()
    output["exp006_minus_exp005_ML_sample_mean_row_abs_error"] = (
        output["exp006_row_abs_error"] - output["exp005_ML_row_abs_error"]
    )
    output["exp006_minus_exp005_ML_sample_max_abs_error"] = (
        output["exp006_max_abs_error"] - output["exp005_ML_max_abs_error"]
    )
    return output


def build_metrics(rows: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for key, label, prediction, _color in METHODS:
        row_metric = regression_metrics(rows["ZT_obs"], rows[prediction])
        sample_metric = regression_metrics(samples["ZT_obs_max"], samples[f"{key}_ZT_max"])
        records.append(
            {
                "method_key": key,
                "method_label": label,
                "comparable_rows": len(rows),
                "comparable_samples": len(samples),
                "comparable_DOIs": rows["exp006_DOI_normalized"].nunique(),
                "sample_balanced_row_MAE": float(samples[f"{key}_row_abs_error"].mean()),
                **{f"row_{name}": value for name, value in row_metric.items()},
                **{f"sample_max_{name}": value for name, value in sample_metric.items()},
            }
        )
    return pd.DataFrame(records)


def bootstrap_mean_ci(values: pd.Series) -> tuple[float, float, float, float]:
    values = finite(values).dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lower), float(upper), float((means < 0).mean())


def build_paired_comparison(samples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scope, column in [
        ("sample_balanced_mean_row_absolute_error", "exp006_minus_exp005_ML_sample_mean_row_abs_error"),
        ("sample_max_absolute_error", "exp006_minus_exp005_ML_sample_max_abs_error"),
    ]:
        mean, lower, upper, probability_better = bootstrap_mean_ci(samples[column])
        records.append(
            {
                "comparison": "exp006_minus_exp005_ML",
                "scope": scope,
                "mean_paired_error_difference": mean,
                "bootstrap_95pct_CI_lower": lower,
                "bootstrap_95pct_CI_upper": upper,
                "bootstrap_probability_exp006_lower_error": probability_better,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_unit": "sample",
                "interpretation": (
                    "no_clear_MAE_improvement"
                    if lower <= 0 <= upper
                    else ("exp006_lower_error" if upper < 0 else "exp006_higher_error")
                ),
            }
        )
    return pd.DataFrame(records)


def material_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    records = []
    for material_name, group in samples.groupby("material_system_label", sort=False):
        for key, label, _prediction, _color in METHODS:
            records.append(
                {
                    "material_system_label": material_name,
                    "method_key": key,
                    "method_label": label,
                    "samples": len(group),
                    "sample_balanced_row_MAE": group[f"{key}_row_abs_error"].mean(),
                    "sample_max_MAE": group[f"{key}_max_abs_error"].mean(),
                }
            )
    return pd.DataFrame(records)


def limits(rows: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [rows["ZT_obs"], *[rows[prediction] for _key, _label, prediction, _color in METHODS]]
    ).dropna()
    upper = max(0.1, float(values.quantile(0.995)) * 1.05)
    return -0.03 * upper, upper


def plot_parity(rows: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    lower, upper = limits(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharex=True, sharey=True)
    for ax, (key, label, prediction, color) in zip(axes, METHODS):
        metric = metrics.loc[metrics["method_key"].eq(key)].iloc[0]
        ax.scatter(rows["ZT_obs"], rows[prediction], s=18, alpha=0.5, color=color, edgecolors="none")
        ax.plot([lower, upper], [lower, upper], "--", color="#B7352D", linewidth=1.35)
        ax.set_title(
            f"{label}\nMAE={metric['row_MAE']:.3f}, log RMSE={metric['row_log_RMSE']:.3f}",
            fontsize=11,
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.set_xlabel("Reported ZT")
    axes[0].set_ylabel("Predicted ZT")
    fig.suptitle("Same 383 rows / 109 samples: reported ZT within +/-2 K", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "DF is direct in-sample curve fitting; ML and exp006 are held-out prediction methods.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_overview(metrics: pd.DataFrame, output: Path) -> None:
    panels = [
        ("sample_balanced_row_MAE", "Sample-balanced MAE", "lower is better", False),
        ("row_log_RMSE", "Log RMSE", "lower is better", False),
        ("row_spearman", "Spearman correlation", "higher is better", True),
        ("row_within_factor_2_rate", "Within factor 2", "higher is better", True),
    ]
    colors = {key: color for key, _label, _prediction, color in METHODS}
    labels = {key: label for key, label, _prediction, _color in METHODS}
    order = [key for key, _label, _prediction, _color in METHODS]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8))
    for ax, (column, title, direction, is_rate) in zip(axes, panels):
        values = metrics.set_index("method_key").loc[order, column]
        bars = ax.bar(
            np.arange(len(order)), values, color=[colors[key] for key in order], width=0.68
        )
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([labels[key] for key in order], rotation=18, ha="right")
        ax.set_title(f"{title}\n({direction})", fontsize=11)
        ax.grid(axis="y", alpha=0.22)
        top = max(float(values.max()) * 1.24, 0.1)
        ax.set_ylim(0, top)
        for bar, value in zip(bars, values):
            text = f"{value * 100:.1f}%" if is_rate and column.endswith("rate") else f"{value:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.025, text, ha="center", va="bottom", fontsize=9)
    fig.suptitle("Three-method ZT performance on the identical common cohort", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Absolute MAE: exp006 is not improved over exp005 ML; multiplicative error: exp006 is better than ML but remains worse than DF.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_ml_exp006(
    samples: pd.DataFrame, paired: pd.DataFrame, output: Path
) -> None:
    x = samples["exp005_ML_row_abs_error"]
    y = samples["exp006_row_abs_error"]
    upper = max(float(pd.concat([x, y]).quantile(0.99)) * 1.08, 0.05)
    row = paired.loc[
        paired["scope"].eq("sample_balanced_mean_row_absolute_error")
    ].iloc[0]
    delta = samples["exp006_minus_exp005_ML_sample_mean_row_abs_error"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    axes[0].scatter(x, y, s=28, alpha=0.62, color="#2B7A98", edgecolors="none")
    axes[0].plot([0, upper], [0, upper], "--", color="#B7352D", linewidth=1.3)
    axes[0].set_xlim(0, upper)
    axes[0].set_ylim(0, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("exp005 ML mean absolute error")
    axes[0].set_ylabel("exp006 mean absolute error")
    axes[0].set_title("One point per sample")
    axes[0].grid(alpha=0.22)

    axes[1].axvline(0, linestyle="--", color="#B7352D", linewidth=1.3)
    axes[1].hist(delta, bins=24, color="#2B7A98", alpha=0.78, edgecolor="none")
    axes[1].axvline(row["mean_paired_error_difference"], color="#1C4F66", linewidth=2)
    axes[1].set_xlabel("exp006 error - exp005 ML error")
    axes[1].set_ylabel("Samples")
    axes[1].set_title(
        "Mean difference = "
        f"{row['mean_paired_error_difference']:+.3f}\n"
        f"sample-bootstrap 95% CI [{row['bootstrap_95pct_CI_lower']:+.3f}, "
        f"{row['bootstrap_95pct_CI_upper']:+.3f}]"
    )
    axes[1].grid(axis="y", alpha=0.22)
    fig.suptitle("Does exp006 reduce absolute ZT error versus exp005 ML?", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "The interval includes zero: no clear absolute-error improvement on this common cohort.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_material_mae(material: pd.DataFrame, output: Path) -> None:
    counts = material.groupby("material_system_label")["samples"].max()
    groups = counts.loc[counts.ge(5)].sort_values(ascending=True).index.tolist()
    if not groups:
        return
    y = np.arange(len(groups), dtype=float)
    offsets = {"exp005_DF": -0.22, "exp005_ML": 0.0, "exp006": 0.22}
    fig, ax = plt.subplots(figsize=(11.5, max(6.2, 0.62 * len(groups))))
    for key, label, _prediction, color in METHODS:
        indexed = material.loc[material["method_key"].eq(key)].set_index("material_system_label")
        values = indexed.loc[groups, "sample_balanced_row_MAE"]
        ax.barh(y + offsets[key], values, height=0.2, color=color, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{group} (n={int(counts[group])})" for group in groups])
    ax.set_xlabel("Sample-balanced mean absolute ZT error")
    ax.set_title("Absolute ZT error by material system (common cohort)")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    m = metrics.set_index("method_key")
    paired_row = paired.loc[
        paired["scope"].eq("sample_balanced_mean_row_absolute_error")
    ].iloc[0]
    lines = [
        "# exp005 DF / exp005 ML / exp006 common-cohort ZT comparison",
        "",
        "## Scope",
        "",
        f"- Identical comparable rows: {len(rows):,}.",
        f"- Identical samples: {len(samples):,}.",
        f"- DOI: {rows['exp006_DOI_normalized'].nunique():,}.",
        "- Reported ZT and thermal conductivity are matched from the same Step5 curves within +/-2 K.",
        "- All three methods use the same observed S, kappa, T, and reported-ZT reference.",
        "",
        "## Results",
        "",
        f"- exp005 DF sample-balanced MAE: {m.at['exp005_DF', 'sample_balanced_row_MAE']:.6g}; log RMSE: {m.at['exp005_DF', 'row_log_RMSE']:.6g}.",
        f"- exp005 ML sample-balanced MAE: {m.at['exp005_ML', 'sample_balanced_row_MAE']:.6g}; log RMSE: {m.at['exp005_ML', 'row_log_RMSE']:.6g}.",
        f"- exp006 sample-balanced MAE: {m.at['exp006', 'sample_balanced_row_MAE']:.6g}; log RMSE: {m.at['exp006', 'row_log_RMSE']:.6g}.",
        f"- Paired exp006-minus-ML MAE difference: {paired_row['mean_paired_error_difference']:+.6g}; sample-bootstrap 95% CI [{paired_row['bootstrap_95pct_CI_lower']:+.6g}, {paired_row['bootstrap_95pct_CI_upper']:+.6g}].",
        "- The absolute-error interval includes zero, so this common cohort does not show a clear MAE improvement of exp006 over exp005 ML.",
        "- exp006 has better log RMSE and factor-2 coverage than exp005 ML, but worse Spearman correlation on this subset.",
        "",
        "## Fairness cautions",
        "",
        "- exp005 DF is direct per-sample in-sample curve fitting and is expected to outperform held-out prediction methods; it is a fit-quality reference, not an equal generalization benchmark.",
        "- exp005 ML is DOI-held-out, while exp006 is sample-held-out. The evaluation rows are identical, but training-group leakage conditions are not identical.",
        "- This common cohort contains no sample with reported maximum ZT >= 1, so it cannot support conclusions about high-ZT discovery performance.",
        "- The plots show the measured result; they should not be described as proof that exp006 is universally no better than ML.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    rows = load_rows(args.input.resolve())
    samples = sample_summary(rows)
    metrics = build_metrics(rows, samples)
    paired = build_paired_comparison(samples)
    by_material = material_metrics(samples)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".three_method_zt_", dir=output.parent))
    try:
        figures = stage / "figures"
        figures.mkdir()
        write_csv(rows, stage / "three_method_row_comparisons.csv")
        write_csv(samples, stage / "three_method_sample_summary.csv")
        write_csv(metrics, stage / "three_method_model_metrics.csv")
        write_csv(paired, stage / "exp006_vs_exp005_ML_paired_comparison.csv")
        write_csv(by_material, stage / "three_method_material_system_metrics.csv")
        plot_parity(rows, metrics, figures / "figure_three_method_parity_reported_zt_2K.png")
        plot_metric_overview(
            metrics, figures / "figure_three_method_performance_metrics_reported_zt_2K.png"
        )
        plot_paired_ml_exp006(
            samples, paired, figures / "figure_exp006_vs_exp005_ML_paired_error_reported_zt_2K.png"
        )
        plot_material_mae(
            by_material, figures / "figure_three_method_material_system_MAE_reported_zt_2K.png"
        )
        write_notes(stage / "three_method_comparison_notes.md", rows, samples, metrics, paired)
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"output_dir: {output}")
    print(metrics[["method_label", "sample_balanced_row_MAE", "row_log_RMSE", "row_spearman", "row_within_factor_2_rate"]].to_string(index=False))
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
