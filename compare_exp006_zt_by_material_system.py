#!/usr/bin/env python3
"""Compare exp006 predicted and observed ZT by existing Step31 material systems."""

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


ROOT = Path(__file__).resolve().parent
DEFAULT_ROWS = (
    ROOT
    / "data"
    / "output"
    / "exp006_pf_zt_from_sigma_prediction_full_curves"
    / "exp006_pf_zt_row_predictions.csv"
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
    / "exp006_pf_zt_from_sigma_prediction_full_curves"
    / "material_system_comparison"
)

ROW_REQUIRED = {
    "row_id",
    "sample_key",
    "temperature_K",
    "ZT_obs",
    "ZT_pred_exp006",
}
CLASS_REQUIRED = {
    "sample_key",
    "material_system_reclass_step31",
    "final_material_group_key",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-predictions", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-main-samples", type=int, default=10)
    parser.add_argument("--high-zt-threshold", type=float, default=1.0)
    return parser.parse_args()


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def display_name(key: object) -> str:
    if pd.isna(key) or not str(key).strip():
        return "Unclassified"
    value = str(key).strip().removeprefix("host::")
    labels = {
        "other_formula_system": "Other formula systems",
        "CoSb_skutterudite": "CoSb skutterudite",
        "unknown_material_group": "Unknown material group",
    }
    return labels.get(value, value)


def load_rows(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = ROW_REQUIRED.difference(header)
    if missing:
        raise ValueError(f"Missing ZT row columns: {sorted(missing)}")
    wanted = [
        "row_id",
        "sample_key",
        "sample_id",
        "DOI",
        "composition",
        "temperature_K",
        "kappa_matched_temperature_K",
        "kappa_temperature_delta_K",
        "zt_matched_temperature_K",
        "zt_temperature_delta_K",
        "ZT_obs",
        "ZT_pred_exp006",
        "ZT_calc_from_obs",
    ]
    df = pd.read_csv(path, usecols=[c for c in wanted if c in header], low_memory=False)
    df["sample_key"] = df["sample_key"].astype(str)
    df["ZT_obs"] = finite(df["ZT_obs"])
    df["ZT_pred_exp006"] = finite(df["ZT_pred_exp006"])
    return df.loc[df["ZT_obs"].notna() & df["ZT_pred_exp006"].notna()].copy()


def load_sample_classification(path: Path) -> pd.DataFrame:
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = CLASS_REQUIRED.difference(header)
    if missing:
        raise ValueError(f"Missing Step31 classification columns: {sorted(missing)}")
    wanted = [
        "sample_key",
        "material_system_original",
        "material_system_reclass_step31",
        "host_family_key",
        "final_material_group_key",
        "group_backoff_level",
        "major_element_system_key",
        "classification_reason",
        "parse_status",
    ]
    raw = pd.read_csv(path, usecols=[c for c in wanted if c in header], low_memory=False)
    raw["sample_key"] = raw["sample_key"].astype(str)
    check_columns = [
        c
        for c in [
            "material_system_reclass_step31",
            "host_family_key",
            "final_material_group_key",
            "group_backoff_level",
        ]
        if c in raw
    ]
    uniqueness = raw.groupby("sample_key", sort=False)[check_columns].nunique(dropna=False)
    ambiguous = uniqueness.gt(1).any(axis=1)
    if ambiguous.any():
        raise ValueError(
            f"Step31 classification is ambiguous for {int(ambiguous.sum())} sample keys"
        )
    return raw.drop_duplicates("sample_key", keep="first")


def add_classification(rows: pd.DataFrame, classification: pd.DataFrame) -> pd.DataFrame:
    merged = rows.merge(classification, on="sample_key", how="left", validate="many_to_one")
    merged["material_system_key"] = merged["material_system_reclass_step31"]
    merged["material_system_label"] = merged["material_system_key"].map(display_name)
    merged["classification_status"] = np.where(
        merged["material_system_reclass_step31"].notna(), "classified_step31", "unclassified"
    )
    return merged


def aggregate_samples(rows: pd.DataFrame) -> pd.DataFrame:
    carry = [
        "sample_id",
        "DOI",
        "composition",
        "material_system_key",
        "material_system_label",
        "classification_status",
        "final_material_group_key",
        "group_backoff_level",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "comparable_temperature_rows": ("temperature_K", "size"),
        "temperature_min_K": ("temperature_K", "min"),
        "temperature_max_K": ("temperature_K", "max"),
        "max_ZT_obs": ("ZT_obs", "max"),
        "max_ZT_pred_exp006": ("ZT_pred_exp006", "max"),
        "median_ZT_obs": ("ZT_obs", "median"),
        "median_ZT_pred_exp006": ("ZT_pred_exp006", "median"),
    }
    for column in carry:
        if column in rows:
            aggregations[column] = (column, "first")
    samples = rows.groupby("sample_key", dropna=False).agg(**aggregations).reset_index()
    samples["max_ZT_signed_error"] = samples["max_ZT_pred_exp006"] - samples["max_ZT_obs"]
    samples["max_ZT_abs_error"] = samples["max_ZT_signed_error"].abs()
    samples["max_ZT_absolute_percentage_error_pct"] = np.where(
        samples["max_ZT_obs"].ne(0),
        samples["max_ZT_abs_error"] / samples["max_ZT_obs"].abs() * 100.0,
        np.nan,
    )
    return samples


def confusion(actual: pd.Series, predicted: pd.Series, threshold: float) -> dict[str, float | int]:
    truth = actual.ge(threshold)
    pred = predicted.ge(threshold)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "high_ZT_true_positive": tp,
        "high_ZT_false_positive": fp,
        "high_ZT_false_negative": fn,
        "high_ZT_true_negative": tn,
        "high_ZT_precision": precision,
        "high_ZT_recall": recall,
        "high_ZT_F1": f1,
    }


def build_metrics(
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    minimum_main_samples: int,
    high_zt_threshold: float,
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for label, sample_group in samples.groupby("material_system_label", sort=False):
        row_group = rows.loc[rows["material_system_label"].eq(label)]
        row_error = row_group["ZT_pred_exp006"] - row_group["ZT_obs"]
        max_error = sample_group["max_ZT_pred_exp006"] - sample_group["max_ZT_obs"]
        row_nonzero = row_group["ZT_obs"].ne(0)
        sample_nonzero = sample_group["max_ZT_obs"].ne(0)
        classified = sample_group["classification_status"].eq("classified_step31").all()
        output.append(
            {
                "material_system_label": label,
                "material_system_key": sample_group["material_system_key"].iloc[0],
                "classification_status": sample_group["classification_status"].iloc[0],
                "main_analysis_group": bool(
                    classified and len(sample_group) >= minimum_main_samples
                ),
                "comparable_rows": int(len(row_group)),
                "comparable_samples": int(len(sample_group)),
                "row_ZT_obs_median": float(row_group["ZT_obs"].median()),
                "row_ZT_pred_median": float(row_group["ZT_pred_exp006"].median()),
                "row_ZT_signed_bias": float(row_error.mean()),
                "row_ZT_MAE": float(row_error.abs().mean()),
                "row_ZT_RMSE": float(np.sqrt(np.mean(np.square(row_error)))),
                "row_ZT_MAPE_pct": float(
                    (row_error[row_nonzero].abs() / row_group.loc[row_nonzero, "ZT_obs"].abs()).mean()
                    * 100.0
                )
                if row_nonzero.any()
                else math.nan,
                "row_ZT_spearman": float(
                    row_group["ZT_obs"].rank().corr(row_group["ZT_pred_exp006"].rank())
                )
                if len(row_group) >= 2
                else math.nan,
                "sample_max_ZT_obs_median": float(sample_group["max_ZT_obs"].median()),
                "sample_max_ZT_pred_median": float(
                    sample_group["max_ZT_pred_exp006"].median()
                ),
                "sample_max_ZT_signed_bias": float(max_error.mean()),
                "sample_max_ZT_MAE": float(max_error.abs().mean()),
                "sample_max_ZT_RMSE": float(np.sqrt(np.mean(np.square(max_error)))),
                "sample_max_ZT_MAPE_pct": float(
                    (
                        max_error[sample_nonzero].abs()
                        / sample_group.loc[sample_nonzero, "max_ZT_obs"].abs()
                    ).mean()
                    * 100.0
                )
                if sample_nonzero.any()
                else math.nan,
                "sample_max_ZT_spearman": float(
                    sample_group["max_ZT_obs"].rank().corr(
                        sample_group["max_ZT_pred_exp006"].rank()
                    )
                )
                if len(sample_group) >= 2
                else math.nan,
                "high_ZT_threshold": high_zt_threshold,
                **confusion(
                    sample_group["max_ZT_obs"],
                    sample_group["max_ZT_pred_exp006"],
                    high_zt_threshold,
                ),
            }
        )
    result = pd.DataFrame(output)
    return result.sort_values(
        ["main_analysis_group", "comparable_samples", "comparable_rows"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def plot_limits(actual: pd.Series, predicted: pd.Series) -> tuple[float, float]:
    values = pd.concat([finite(actual), finite(predicted)]).dropna()
    upper = max(0.1, float(values.quantile(0.995)) * 1.05)
    return -0.03 * upper, upper


def plot_row_parity_facets(
    rows: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    groups = metrics.loc[metrics["main_analysis_group"], "material_system_label"].tolist()
    columns = 4
    row_count = math.ceil(len(groups) / columns)
    lower, upper = plot_limits(rows["ZT_obs"], rows["ZT_pred_exp006"])
    fig, axes = plt.subplots(
        row_count, columns, figsize=(15, 3.6 * row_count), sharex=True, sharey=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes_array, groups):
        part = rows.loc[rows["material_system_label"].eq(label)]
        ax.scatter(
            part["ZT_obs"],
            part["ZT_pred_exp006"],
            s=15,
            alpha=0.45,
            color="#2B7A98",
            edgecolors="none",
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#C53B2C", linewidth=1.4)
        ax.set_title(f"{label}\n{part['sample_key'].nunique()} samples / {len(part)} rows", fontsize=10)
        ax.grid(alpha=0.22)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_array[len(groups) :]:
        ax.axis("off")
    fig.supxlabel("Observed ZT")
    fig.supylabel("exp006 predicted ZT")
    fig.suptitle("ZT prediction parity by Step31 material system", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_max_distribution(
    samples: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    main = metrics.loc[metrics["main_analysis_group"]].copy()
    main = main.sort_values("sample_max_ZT_obs_median", ascending=True)
    labels = main["material_system_label"].tolist()
    observed = [samples.loc[samples["material_system_label"].eq(x), "max_ZT_obs"] for x in labels]
    predicted = [
        samples.loc[samples["material_system_label"].eq(x), "max_ZT_pred_exp006"]
        for x in labels
    ]
    y = np.arange(len(labels), dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(6.5, len(labels) * 0.55)))
    obs_box = ax.boxplot(
        observed,
        positions=y + 0.17,
        vert=False,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    pred_box = ax.boxplot(
        predicted,
        positions=y - 0.17,
        vert=False,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for box in obs_box["boxes"]:
        box.set(facecolor="#7FB3C8", edgecolor="#2B7A98", alpha=0.85)
    for box in pred_box["boxes"]:
        box.set(facecolor="#E7A45B", edgecolor="#B76516", alpha=0.82)
    for item in obs_box["medians"]:
        item.set(color="#164B60", linewidth=1.6)
    for item in pred_box["medians"]:
        item.set(color="#7A3D00", linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{label}  (n={int(main.loc[main.material_system_label.eq(label), 'comparable_samples'].iloc[0])})"
            for label in labels
        ]
    )
    ax.set_xlabel("Sample maximum ZT")
    ax.set_title("Observed and predicted sample-maximum ZT by material system")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#2B7A98", linewidth=7, label="Observed max ZT"),
            Line2D([0], [0], color="#B76516", linewidth=7, label="Predicted max ZT"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_max_parity_facets(
    samples: pd.DataFrame, metrics: pd.DataFrame, output: Path
) -> None:
    groups = metrics.loc[metrics["main_analysis_group"], "material_system_label"].tolist()
    columns = 4
    row_count = math.ceil(len(groups) / columns)
    lower, upper = plot_limits(samples["max_ZT_obs"], samples["max_ZT_pred_exp006"])
    fig, axes = plt.subplots(
        row_count, columns, figsize=(15, 3.6 * row_count), sharex=True, sharey=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes_array, groups):
        part = samples.loc[samples["material_system_label"].eq(label)]
        ax.scatter(
            part["max_ZT_obs"],
            part["max_ZT_pred_exp006"],
            s=24,
            alpha=0.62,
            color="#6A51A3",
            edgecolors="none",
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#C53B2C", linewidth=1.4)
        metric = metrics.loc[metrics["material_system_label"].eq(label)].iloc[0]
        ax.set_title(
            f"{label}\nn={len(part)}, MAE={metric['sample_max_ZT_MAE']:.3f}", fontsize=10
        )
        ax.grid(alpha=0.22)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_array[len(groups) :]:
        ax.axis("off")
    fig.supxlabel("Observed sample maximum ZT")
    fig.supylabel("Predicted sample maximum ZT")
    fig.suptitle("Sample-maximum ZT parity by Step31 material system", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
    minimum_main_samples: int,
    high_zt_threshold: float,
) -> None:
    classified_rows = int(rows["classification_status"].eq("classified_step31").sum())
    classified_samples = int(
        samples["classification_status"].eq("classified_step31").sum()
    )
    main_labels = metrics.loc[metrics["main_analysis_group"], "material_system_label"].tolist()
    excluded = metrics.loc[~metrics["main_analysis_group"], ["material_system_label", "comparable_samples"]]
    lines = [
        "# ZT comparison by existing Step31 material system",
        "",
        "## Scope",
        "",
        "- ZT values come from the full-curve exp006 PF/ZT output and are not recomputed here.",
        "- Step31 material-system labels are joined by `sample_key` only.",
        "- Row-level comparisons require both `ZT_obs` and `ZT_pred_exp006` at the same exp006 anchor row.",
        "- Sample-level comparisons use maximum ZT within those comparable rows.",
        f"- High-ZT classification threshold: {high_zt_threshold:g}.",
        "",
        "## Coverage",
        "",
        f"- Comparable rows: {len(rows):,}; classified by Step31: {classified_rows:,} ({classified_rows / len(rows):.1%}).",
        f"- Comparable samples: {len(samples):,}; classified by Step31: {classified_samples:,} ({classified_samples / len(samples):.1%}).",
        f"- Main figures include classified groups with at least {minimum_main_samples} comparable samples.",
        f"- Main groups ({len(main_labels)}): {', '.join(main_labels)}.",
        "",
        "## Reference-only groups",
        "",
        *[
            f"- {row.material_system_label}: {int(row.comparable_samples)} samples"
            for row in excluded.itertuples(index=False)
        ],
        "",
        "## Interpretation cautions",
        "",
        "- The number of temperature rows differs by sample, so material-system conclusions should prioritize the sample-level tables and figures.",
        "- MAPE excludes observed ZT equal to zero; MAE and RMSE retain those rows.",
        "- Results describe propagation of the existing sigma prediction with observed S and kappa, not a fully independent ZT model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    args = parse_args()
    if args.minimum_main_samples < 1:
        raise ValueError("--minimum-main-samples must be at least 1")
    rows_path = args.row_predictions.resolve()
    classification_path = args.classification.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    rows = add_classification(
        load_rows(rows_path), load_sample_classification(classification_path)
    )
    samples = aggregate_samples(rows)
    metrics = build_metrics(
        rows, samples, args.minimum_main_samples, args.high_zt_threshold
    )
    sample_counts = metrics.set_index("material_system_label")["comparable_samples"]
    rows["material_system_sample_count"] = rows["material_system_label"].map(sample_counts)
    samples["material_system_sample_count"] = samples["material_system_label"].map(sample_counts)
    main_map = metrics.set_index("material_system_label")["main_analysis_group"]
    rows["main_analysis_group"] = rows["material_system_label"].map(main_map)
    samples["main_analysis_group"] = samples["material_system_label"].map(main_map)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".zt_material_stage_", dir=output_dir.parent))
    try:
        figures = stage_dir / "figures"
        figures.mkdir()
        write_csv(rows, stage_dir / "zt_material_system_row_comparisons.csv")
        write_csv(samples, stage_dir / "zt_material_system_sample_summary.csv")
        write_csv(metrics, stage_dir / "zt_material_system_metrics.csv")
        plot_row_parity_facets(
            rows, metrics, figures / "figure_zt_parity_by_material_system.png"
        )
        plot_sample_max_distribution(
            samples,
            metrics,
            figures / "figure_sample_max_zt_distribution_by_material_system.png",
        )
        plot_sample_max_parity_facets(
            samples,
            metrics,
            figures / "figure_sample_max_zt_parity_by_material_system.png",
        )
        write_notes(
            stage_dir / "zt_material_system_comparison_notes.md",
            rows,
            samples,
            metrics,
            args.minimum_main_samples,
            args.high_zt_threshold,
        )
        stage_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    classified_rows = int(rows["classification_status"].eq("classified_step31").sum())
    classified_samples = int(
        samples["classification_status"].eq("classified_step31").sum()
    )
    print(f"Created: {output_dir}")
    print(f"Comparable rows: {len(rows):,}; Step31 classified: {classified_rows:,}")
    print(f"Comparable samples: {len(samples):,}; Step31 classified: {classified_samples:,}")
    print(f"Main material groups: {int(metrics['main_analysis_group'].sum())}")


if __name__ == "__main__":
    main()
