#!/usr/bin/env python3
"""Color exp006 common DOI-CV ZT predictions by Step31 material system."""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import compare_exp006_zt_by_material_system as material


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = (
    ROOT
    / "data"
    / "output"
    / "exp005_exp006_common_doi_group_cv"
    / "common_doi_cv_sample_max_predictions.csv"
)
DEFAULT_CLASSIFICATION = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step31_reclassified_ct_curves"
    / "material_classification_keys_step31.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "output"
    / "exp005_exp006_common_doi_group_cv"
    / "material_system_colored_exp006"
)

ZT_THRESHOLD = 1.0

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def load_samples(path: Path, classification_path: Path) -> pd.DataFrame:
    wanted = [
        "sample_key",
        "fold",
        "DOI",
        "composition",
        "max_ZT_obs",
        "max_ZT_pred_exp006",
    ]
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(wanted).difference(header)
    if missing:
        raise ValueError(f"Missing common DOI-CV sample columns: {sorted(missing)}")
    samples = pd.read_csv(path, usecols=wanted, low_memory=False)
    samples["sample_key"] = samples["sample_key"].astype(str)
    samples["max_ZT_obs"] = finite(samples["max_ZT_obs"])
    samples["max_ZT_pred_exp006"] = finite(samples["max_ZT_pred_exp006"])
    samples = samples.loc[
        samples["max_ZT_obs"].notna() & samples["max_ZT_pred_exp006"].notna()
    ].copy()
    classification = material.load_sample_classification(classification_path)
    samples = material.add_classification(samples, classification)
    samples["material_system_label"] = samples["material_system_label"].fillna(
        "Unclassified"
    )
    samples["ZT_error_exp006"] = (
        samples["max_ZT_pred_exp006"] - samples["max_ZT_obs"]
    )
    samples["abs_ZT_error_exp006"] = samples["ZT_error_exp006"].abs()
    return samples


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual = finite(actual)
    predicted = finite(predicted)
    ok = actual.notna() & predicted.notna()
    actual = actual.loc[ok].to_numpy(float)
    predicted = predicted.loc[ok].to_numpy(float)
    error = predicted - actual
    positive = (actual > 0) & (predicted > 0)
    ratio = predicted[positive] / actual[positive]
    log_error = np.log10(ratio)
    high_actual = actual >= ZT_THRESHOLD
    high_pred = predicted >= ZT_THRESHOLD
    tp = int(np.sum(high_actual & high_pred))
    fp = int(np.sum(~high_actual & high_pred))
    fn = int(np.sum(high_actual & ~high_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "sample_count": int(len(actual)),
        "high_ZT_observed_samples": int(high_actual.sum()),
        "high_ZT_predicted_samples": int(high_pred.sum()),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "log10_RMSE": float(np.sqrt(np.mean(log_error**2))) if len(log_error) else np.nan,
        "within_factor_2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if len(ratio) else np.nan,
        "Spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")) if len(actual) > 1 else np.nan,
        "high_ZT_precision": precision,
        "high_ZT_recall": recall,
        "high_ZT_F1": f1,
        "median_ZT_observed": float(np.median(actual)),
        "median_ZT_predicted": float(np.median(predicted)),
        "max_ZT_observed": float(np.max(actual)),
        "max_ZT_predicted": float(np.max(predicted)),
    }


def build_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in samples.groupby("material_system_label", sort=False):
        record: dict[str, object] = {
            "material_system_label": label,
            "material_system_key": group["material_system_key"].iloc[0],
            "classification_status": group["classification_status"].iloc[0],
            "DOI_count": int(group["DOI"].nunique()),
        }
        record.update(
            regression_metrics(group["max_ZT_obs"], group["max_ZT_pred_exp006"])
        )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["sample_count", "material_system_label"], ascending=[False, True]
    ).reset_index(drop=True)


def style_maps(labels: list[str]) -> tuple[dict[str, object], dict[str, str]]:
    palette = list(plt.get_cmap("tab20").colors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p", "8", "d", "H", "x"]
    colors = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
    shapes = {label: markers[index % len(markers)] for index, label in enumerate(labels)}
    return colors, shapes


def plot_colored_parity(
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
) -> None:
    labels = metrics["material_system_label"].tolist()
    colors, shapes = style_maps(labels)
    fig, ax = plt.subplots(figsize=(17.5, 11.3))
    for label in labels:
        group = samples.loc[samples["material_system_label"].eq(label)]
        ax.scatter(
            group["max_ZT_obs"],
            group["max_ZT_pred_exp006"],
            s=42,
            alpha=0.56,
            color=colors[label],
            marker=shapes[label],
            edgecolors="none" if shapes[label] not in {"x"} else colors[label],
            label=f"{label} (n={len(group):,})",
        )
    all_values = pd.concat(
        [samples["max_ZT_obs"], samples["max_ZT_pred_exp006"]], ignore_index=True
    ).dropna()
    low = min(-0.03, float(all_values.min()) - 0.02)
    high = float(all_values.max()) * 1.05
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2, color="#4D4D4D", label="Ideal: predicted = reported")
    ax.axvline(ZT_THRESHOLD, color="#777777", linewidth=0.9, linestyle=":")
    ax.axhline(ZT_THRESHOLD, color="#777777", linewidth=0.9, linestyle=":")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reported sample maximum ZT", fontsize=19)
    ax.set_ylabel("exp006 OOF predicted sample maximum ZT", fontsize=19)
    ax.set_title(
        "exp006 ZT prediction by Step31 material system\nCommon DOI-group 5-fold CV; same evaluation samples",
        fontsize=23,
    )
    ax.grid(alpha=0.18)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
        ncol=1,
        fontsize=14,
        handletextpad=0.5,
        labelspacing=0.55,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_material_medians(metrics: pd.DataFrame, path: Path) -> None:
    ordered = metrics.sort_values("median_ZT_observed", ascending=True).reset_index(drop=True)
    labels = metrics["material_system_label"].tolist()
    colors, _ = style_maps(labels)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(15.5, 11.2))
    for index, row in ordered.iterrows():
        color = colors[row["material_system_label"]]
        ax.plot(
            [row["median_ZT_observed"], row["median_ZT_predicted"]],
            [index, index],
            color=color,
            linewidth=3.2,
            alpha=0.75,
        )
        ax.scatter(row["median_ZT_observed"], index, color=color, marker="o", s=85, edgecolors="#333333", linewidths=0.5, zorder=3)
        ax.scatter(row["median_ZT_predicted"], index, color=color, marker="D", s=76, edgecolors="#333333", linewidths=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{row.material_system_label}  (n={int(row.sample_count):,})" for row in ordered.itertuples()],
        fontsize=15,
    )
    ax.set_xlabel("Median sample maximum ZT", fontsize=19)
    ax.set_title("Material-system medians: reported versus exp006 OOF prediction", fontsize=22)
    ax.grid(axis="x", alpha=0.2)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="#333333", label="Reported median", markersize=10),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#777777", markeredgecolor="#333333", label="exp006 predicted median", markersize=9),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(path: Path, samples: pd.DataFrame, metrics: pd.DataFrame) -> None:
    classified = int(samples["classification_status"].eq("classified_step31").sum())
    lines = [
        "# exp006 common DOI-CV ZT by material system",
        "",
        f"- Evaluation: sample-maximum reported ZT versus exp006 out-of-fold predicted ZT.",
        f"- Samples: {len(samples):,}.",
        f"- Step31-classified samples: {classified:,}; unclassified: {len(samples) - classified:,}.",
        f"- Material-system labels shown: {metrics['material_system_label'].nunique()}.",
        "- Source prediction: common DOI-group 5-fold CV strict shared cohort.",
        "- No ZT<1 cutoff is applied.",
        "- Color and marker shape both encode material system.",
        "",
        "## Interpretation caution",
        "",
        "- Groups with small n have unstable correlations and high-ZT metrics; consult the metrics CSV before comparing systems.",
        "- `Other formula systems`, `Unknown material group`, and `Unclassified` are retained rather than silently removed.",
        "- These categories describe the evaluation samples. The current exp006 Step5A reference key itself is mostly `unknown_material_family`, so this plot is a post-hoc performance breakdown, not evidence that exp006 trained a separate model for every displayed material system.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def publish(stage: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; rerun with --overwrite: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage), str(destination))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    stage_parent = Path(tempfile.mkdtemp(prefix="exp006_zt_material_", dir=output.parent))
    stage = stage_parent / output.name
    stage.mkdir(parents=True)
    try:
        samples = load_samples(args.samples.resolve(), args.classification.resolve())
        metrics = build_metrics(samples)
        write_csv(samples, stage / "exp006_zt_samples_with_material_system.csv")
        write_csv(metrics, stage / "exp006_zt_metrics_by_material_system.csv")
        plot_colored_parity(
            samples,
            metrics,
            stage / "figure_exp006_zt_parity_colored_by_material_system.png",
        )
        plot_material_medians(
            metrics,
            stage / "figure_exp006_zt_medians_by_material_system.png",
        )
        write_notes(stage / "README_exp006_zt_by_material_system.md", samples, metrics)
        publish(stage, output, args.overwrite)
    except Exception:
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    else:
        shutil.rmtree(stage_parent, ignore_errors=True)
    print(f"output: {output}")
    print(metrics[["material_system_label", "sample_count", "MAE", "log10_RMSE", "Spearman"]].to_string(index=False))


if __name__ == "__main__":
    main()
