#!/usr/bin/env python3
"""Reference comparison of native full-range exp005 and exp006 ZT evaluations.

This is intentionally non-paired: exp005 uses its primary DOI test cohort and
exp006 uses its native sample-holdout cohort.  The script recomputes identical
metric definitions from the saved row-level predictions and keeps the cohort
difference explicit in every output.
"""

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
DEFAULT_EXP005 = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step22_fitting_vs_ml_comparison"
    / "step22_row_level_comparison.csv"
)
DEFAULT_EXP006 = (
    ROOT
    / "data"
    / "output"
    / "exp006_pf_zt_from_sigma_prediction_full_curves"
    / "exp006_pf_zt_row_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "output" / "exp005_exp006_native_full_range_zt_comparison"

METHODS = [
    ("exp005_DF", "exp005 DF", "#3B7A57"),
    ("exp005_ML", "exp005 ML", "#C77720"),
    ("exp006", "exp006", "#2B7A98"),
]
METHOD_ORDER = [item[0] for item in METHODS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp005-rows", type=Path, default=DEFAULT_EXP005)
    parser.add_argument("--exp006-rows", type=Path, default=DEFAULT_EXP006)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def standardized_rows(
    source: pd.DataFrame,
    method_key: str,
    method_label: str,
    cohort: str,
    observation_column: str,
    prediction_column: str,
    doi_column: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "method_key": method_key,
            "method_label": method_label,
            "native_evaluation_cohort": cohort,
            "sample_key": source["sample_key"].astype(str),
            "DOI": source[doi_column] if doi_column in source else "",
            "temperature_K": finite(source["temperature_K"]),
            "ZT_observed": finite(source[observation_column]),
            "ZT_predicted": finite(source[prediction_column]),
        }
    )
    return output.loc[
        output["ZT_observed"].notna() & output["ZT_predicted"].notna()
    ].copy()


def load_native_rows(exp005_path: Path, exp006_path: Path) -> pd.DataFrame:
    exp005_columns = [
        "sample_key",
        "DOI",
        "temperature_K",
        "zt_obs_dimensionless_step11",
        "zt_pred_fitting_step21",
        "zt_pred_ML_step21",
    ]
    exp006_columns = [
        "sample_key",
        "DOI",
        "temperature_K",
        "ZT_obs",
        "ZT_pred_exp006",
    ]
    exp005 = pd.read_csv(exp005_path, usecols=exp005_columns, low_memory=False)
    exp006 = pd.read_csv(exp006_path, usecols=exp006_columns, low_memory=False)

    # Keep DF and ML on identical exp005 rows before standardizing.
    valid_exp005 = (
        finite(exp005["zt_obs_dimensionless_step11"]).notna()
        & finite(exp005["zt_pred_fitting_step21"]).notna()
        & finite(exp005["zt_pred_ML_step21"]).notna()
    )
    exp005 = exp005.loc[valid_exp005].copy()
    frames = [
        standardized_rows(
            exp005,
            "exp005_DF",
            "exp005 DF",
            "primary DOI-group test",
            "zt_obs_dimensionless_step11",
            "zt_pred_fitting_step21",
            "DOI",
        ),
        standardized_rows(
            exp005,
            "exp005_ML",
            "exp005 ML",
            "primary DOI-group test",
            "zt_obs_dimensionless_step11",
            "zt_pred_ML_step21",
            "DOI",
        ),
        standardized_rows(
            exp006,
            "exp006",
            "exp006",
            "native sample-holdout test",
            "ZT_obs",
            "ZT_pred_exp006",
            "DOI",
        ),
    ]
    return pd.concat(frames, ignore_index=True)


def confusion(actual: pd.Series, predicted: pd.Series, threshold: float = 1.0) -> dict[str, float | int]:
    truth = finite(actual).ge(threshold)
    pred = finite(predicted).ge(threshold)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "observed_ZT_ge_1_samples": int(truth.sum()),
        "predicted_ZT_ge_1_samples": int(pred.sum()),
        "ZT_ge_1_true_positive": tp,
        "ZT_ge_1_false_positive": fp,
        "ZT_ge_1_false_negative": fn,
        "ZT_ge_1_true_negative": tn,
        "ZT_ge_1_precision": precision,
        "ZT_ge_1_recall": recall,
        "ZT_ge_1_F1": f1,
    }


def sample_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["method_key", "method_label", "native_evaluation_cohort", "sample_key"],
            sort=False,
            dropna=False,
        )
        .agg(
            DOI=("DOI", "first"),
            comparable_rows=("temperature_K", "size"),
            temperature_min_K=("temperature_K", "min"),
            temperature_max_K=("temperature_K", "max"),
            observed_ZT_max=("ZT_observed", "max"),
            predicted_ZT_max=("ZT_predicted", "max"),
        )
        .reset_index()
    )


def build_metrics(rows: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method_key in METHOD_ORDER:
        part = rows.loc[rows["method_key"].eq(method_key)].copy()
        sample_part = samples.loc[samples["method_key"].eq(method_key)].copy()
        observed = finite(part["ZT_observed"])
        predicted = finite(part["ZT_predicted"])
        error = predicted - observed
        positive = observed.gt(0) & predicted.gt(0)
        ratio = predicted.loc[positive] / observed.loc[positive]
        nonzero = observed.ne(0)
        sample_observed = finite(sample_part["observed_ZT_max"])
        sample_predicted = finite(sample_part["predicted_ZT_max"])
        label = str(part["method_label"].iloc[0])
        record: dict[str, object] = {
            "method_key": method_key,
            "method_label": label,
            "native_evaluation_cohort": str(part["native_evaluation_cohort"].iloc[0]),
            "comparable_rows": int(len(part)),
            "comparable_samples": int(len(sample_part)),
            "comparable_DOIs": int(part["DOI"].fillna("").astype(str).replace("", np.nan).nunique()),
            "observed_ZT_row_min": float(observed.min()),
            "observed_ZT_row_median": float(observed.median()),
            "observed_ZT_row_max": float(observed.max()),
            "row_MAE": float(error.abs().mean()),
            "row_RMSE": float(np.sqrt(np.mean(np.square(error)))),
            "row_MAPE_pct": float((error.loc[nonzero].abs() / observed.loc[nonzero].abs()).mean() * 100),
            "row_median_APE_pct": float((error.loc[nonzero].abs() / observed.loc[nonzero].abs()).median() * 100),
            "row_log_RMSE": float(np.sqrt(np.mean(np.square(np.log(ratio))))) if len(ratio) else math.nan,
            "row_within_factor_2_rate": float(ratio.between(0.5, 2.0).mean()) if len(ratio) else math.nan,
            "row_spearman": float(observed.corr(predicted, method="spearman")),
            "sample_max_spearman": float(sample_observed.corr(sample_predicted, method="spearman")),
            **confusion(sample_observed, sample_predicted, 1.0),
        }
        records.append(record)
    return pd.DataFrame(records)


def plot_metric_comparison(metrics: pd.DataFrame, output: Path) -> None:
    panels = [
        ("row_log_RMSE", "Log RMSE", "lower is better", False),
        ("row_within_factor_2_rate", "Within factor 2", "higher is better", True),
        ("sample_max_spearman", "Sample-max Spearman", "higher is better", False),
        ("ZT_ge_1_F1", "ZT >= 1 F1", "higher is better", False),
    ]
    colors = {key: color for key, _label, color in METHODS}
    labels = {key: label for key, label, _color in METHODS}
    indexed = metrics.set_index("method_key").loc[METHOD_ORDER]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.9))
    for ax, (column, title, direction, percent) in zip(axes, panels):
        values = indexed[column]
        bars = ax.bar(
            np.arange(len(METHOD_ORDER)),
            values,
            color=[colors[key] for key in METHOD_ORDER],
            width=0.68,
        )
        ax.set_xticks(np.arange(len(METHOD_ORDER)))
        ax.set_xticklabels([labels[key] for key in METHOD_ORDER], rotation=18, ha="right")
        ax.set_title(f"{title}\n({direction})", fontsize=11)
        ax.grid(axis="y", alpha=0.22)
        upper = max(float(values.max()) * 1.25, 0.1)
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            label = f"{value * 100:.1f}%" if percent else f"{value:.3f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + upper * 0.025,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle("Native full-range ZT comparison (non-paired reference)", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "exp005: DOI-group test; exp006: sample-holdout test. Cohorts and high-ZT prevalence differ.",
        ha="center",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_ratio(rows: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharex=True, sharey=True)
    colors = {key: color for key, _label, color in METHODS}
    labels = {key: label for key, label, _color in METHODS}
    y_values: list[pd.Series] = []
    prepared: dict[str, pd.DataFrame] = {}
    for key in METHOD_ORDER:
        part = rows.loc[rows["method_key"].eq(key)].copy()
        valid = part["ZT_observed"].gt(0) & part["ZT_predicted"].gt(0)
        part = part.loc[valid].copy()
        part["log10_prediction_ratio"] = np.log10(
            part["ZT_predicted"] / part["ZT_observed"]
        )
        prepared[key] = part
        y_values.append(part["log10_prediction_ratio"])
    y_all = pd.concat(y_values)
    y_lower = math.floor(float(y_all.min()) * 2) / 2
    y_upper = math.ceil(float(y_all.max()) * 2) / 2
    factor2 = math.log10(2.0)
    for ax, key in zip(axes, METHOD_ORDER):
        part = prepared[key]
        metric = metrics.loc[metrics["method_key"].eq(key)].iloc[0]
        ax.axhspan(-factor2, factor2, color="#D7D7D7", alpha=0.45)
        ax.axhline(0, linestyle="--", color="#B7352D", linewidth=1.3)
        ax.axvline(1.0, linestyle=":", color="#555555", linewidth=1.1)
        ax.scatter(
            part["ZT_observed"],
            part["log10_prediction_ratio"],
            s=10,
            alpha=0.25,
            color=colors[key],
            edgecolors="none",
        )
        ax.set_xscale("log")
        ax.set_ylim(y_lower, y_upper)
        ax.set_title(
            f"{labels[key]}\nn={len(part):,}, log RMSE={metric['row_log_RMSE']:.3f}",
            fontsize=10.5,
        )
        ax.set_xlabel("Observed ZT (log scale)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("log10(predicted ZT / observed ZT)")
    fig.suptitle("Prediction error across the full positive-ZT range", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "Gray band: within factor 2. Vertical line: observed ZT = 1. Nonpositive rows are omitted from this ratio plot only.",
        ha="center",
        fontsize=9.2,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_ranges(samples: pd.DataFrame, metrics: pd.DataFrame, output: Path) -> None:
    exp005 = samples.loc[samples["method_key"].eq("exp005_ML")].copy()
    exp006 = samples.loc[samples["method_key"].eq("exp006")].copy()
    distributions = [exp005["observed_ZT_max"], exp006["observed_ZT_max"]]
    labels = [
        f"exp005 DOI test\n(n={len(exp005)}, ZT>=1: {int(exp005['observed_ZT_max'].ge(1).sum())})",
        f"exp006 sample holdout\n(n={len(exp006)}, ZT>=1: {int(exp006['observed_ZT_max'].ge(1).sum())})",
    ]
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    boxes = ax.boxplot(
        distributions,
        tick_labels=labels,
        patch_artist=True,
        showfliers=True,
        widths=0.55,
        medianprops={"color": "#333333", "linewidth": 1.5},
    )
    boxes["boxes"][0].set(facecolor="#E6A45F", alpha=0.72)
    boxes["boxes"][1].set(facecolor="#74AFC5", alpha=0.72)
    rng = np.random.default_rng(20260818)
    for index, values in enumerate(distributions, start=1):
        jitter = rng.normal(index, 0.045, size=len(values))
        ax.scatter(jitter, values, s=9, alpha=0.18, color="#555555", edgecolors="none")
    ax.axhline(1.0, linestyle="--", color="#B7352D", linewidth=1.3, label="ZT = 1")
    ax.set_ylabel("Observed sample-maximum ZT")
    ax.set_title("Native evaluation cohorts have different ZT distributions")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, metrics: pd.DataFrame) -> None:
    m = metrics.set_index("method_key")
    lines = [
        "# Native full-range exp005 / exp006 ZT comparison",
        "",
        "## Scope",
        "",
        "- exp005 DF and ML use the original Step22 primary DOI-group test rows.",
        "- exp006 uses its original full sample-holdout reported-ZT rows.",
        "- No common-sample or ZT<1 restriction is applied.",
        "- Metrics are recomputed with the same definitions, but the evaluation cohorts are not paired.",
        "",
        "## Coverage",
        "",
        f"- exp005: {int(m.at['exp005_ML', 'comparable_rows']):,} rows, {int(m.at['exp005_ML', 'comparable_samples']):,} samples, observed ZT max {m.at['exp005_ML', 'observed_ZT_row_max']:.6g}.",
        f"- exp006: {int(m.at['exp006', 'comparable_rows']):,} rows, {int(m.at['exp006', 'comparable_samples']):,} samples, observed ZT max {m.at['exp006', 'observed_ZT_row_max']:.6g}.",
        f"- Observed ZT>=1 samples: exp005={int(m.at['exp005_ML', 'observed_ZT_ge_1_samples'])}, exp006={int(m.at['exp006', 'observed_ZT_ge_1_samples'])}.",
        "",
        "## Main results",
        "",
        f"- log RMSE: DF={m.at['exp005_DF', 'row_log_RMSE']:.6g}, ML={m.at['exp005_ML', 'row_log_RMSE']:.6g}, exp006={m.at['exp006', 'row_log_RMSE']:.6g}.",
        f"- within factor 2: DF={m.at['exp005_DF', 'row_within_factor_2_rate']:.1%}, ML={m.at['exp005_ML', 'row_within_factor_2_rate']:.1%}, exp006={m.at['exp006', 'row_within_factor_2_rate']:.1%}.",
        f"- sample-max Spearman: DF={m.at['exp005_DF', 'sample_max_spearman']:.6g}, ML={m.at['exp005_ML', 'sample_max_spearman']:.6g}, exp006={m.at['exp006', 'sample_max_spearman']:.6g}.",
        f"- ZT>=1 F1: DF={m.at['exp005_DF', 'ZT_ge_1_F1']:.6g}, ML={m.at['exp005_ML', 'ZT_ge_1_F1']:.6g}, exp006={m.at['exp006', 'ZT_ge_1_F1']:.6g}.",
        "",
        "## Interpretation",
        "",
        "- On these native full-range evaluations, exp006 does not outperform exp005 ML on log RMSE, factor-2 coverage, sample-max ranking, or ZT>=1 F1.",
        "- This supports a descriptive statement that exp006 did not show a clear overall improvement over the exp005 ML result.",
        "- It is not a causal or paired superiority test because exp005 is DOI-held-out and exp006 is sample-held-out.",
        "- exp006 has only a small number of observed high-ZT samples, so its F1 estimate is unstable.",
        "- exp005 DF is an in-sample fitting upper reference, not an equal unknown-material prediction baseline.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    rows = load_native_rows(args.exp005_rows.resolve(), args.exp006_rows.resolve())
    samples = sample_summary(rows)
    metrics = build_metrics(rows, samples)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".native_full_zt_", dir=output.parent))
    try:
        figures = stage / "figures"
        figures.mkdir()
        write_csv(rows, stage / "native_full_range_row_comparisons.csv")
        write_csv(samples, stage / "native_full_range_sample_summary.csv")
        write_csv(metrics, stage / "native_full_range_model_metrics.csv")
        plot_metric_comparison(
            metrics, figures / "figure_native_full_range_metric_comparison.png"
        )
        plot_error_ratio(
            rows, metrics, figures / "figure_native_full_range_error_ratio.png"
        )
        plot_cohort_ranges(
            samples, metrics, figures / "figure_native_cohort_ZT_range_comparison.png"
        )
        write_notes(stage / "native_full_range_comparison_notes.md", metrics)
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"output_dir: {output}")
    print(
        metrics[
            [
                "method_label",
                "comparable_rows",
                "comparable_samples",
                "observed_ZT_ge_1_samples",
                "row_log_RMSE",
                "row_within_factor_2_rate",
                "sample_max_spearman",
                "ZT_ge_1_F1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
