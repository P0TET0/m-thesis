#!/usr/bin/env python3
"""Build the recommended three-scenario material-system ZT comparison."""

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

import compare_exp006_zt_by_material_system as material
import compute_pf_zt_from_exp006_sigma as pfzt


ROOT = Path(__file__).resolve().parent
DEFAULT_STRICT_ROWS = (
    ROOT
    / "data"
    / "output"
    / "exp006_pf_zt_from_sigma_prediction_full_curves"
    / "exp006_pf_zt_row_predictions.csv"
)
DEFAULT_PREDICTION = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step5b_test_predictions_default.csv"
)
DEFAULT_CURVES = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step5_core_properties"
    / "candidate_core_curves_step5.csv"
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
    / "material_system_comparison_recommended"
)

SCENARIOS = [
    {
        "key": "reported_zt_2K",
        "label": "Reported ZT, ±2 K",
        "short_label": "Reported ±2 K",
        "reference_type": "reported_ZT",
        "temperature_tolerance_K": 2.0,
        "primary_analysis": True,
        "color": "#2B7A98",
    },
    {
        "key": "calculated_zt_2K",
        "label": "Calculated ZT reference, ±2 K",
        "short_label": "Calculated ±2 K",
        "reference_type": "ZT_calc_from_observed_sigma_S_kappa",
        "temperature_tolerance_K": 2.0,
        "primary_analysis": False,
        "color": "#C77720",
    },
    {
        "key": "reported_zt_3K",
        "label": "Reported ZT sensitivity, ±3 K",
        "short_label": "Reported ±3 K",
        "reference_type": "reported_ZT",
        "temperature_tolerance_K": 3.0,
        "primary_analysis": False,
        "color": "#6A51A3",
    },
]
SCENARIO_BY_KEY = {item["key"]: item for item in SCENARIOS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-row-predictions", type=Path, default=DEFAULT_STRICT_ROWS)
    parser.add_argument("--prediction-file", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--curve-file", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-main-samples", type=int, default=10)
    parser.add_argument("--high-zt-threshold", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def strict_input_columns(path: Path) -> list[str]:
    header = set(pd.read_csv(path, nrows=0).columns)
    required = {
        "row_id",
        "sample_key",
        "temperature_K",
        "ZT_obs",
        "ZT_calc_from_obs",
        "ZT_pred_exp006",
    }
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing strict-output columns: {sorted(missing)}")
    wanted = [
        "row_id",
        "sample_key",
        "sample_id",
        "DOI",
        "composition",
        "temperature_K",
        "kappa_obs_W_per_mK",
        "kappa_matched_temperature_K",
        "kappa_temperature_delta_K",
        "zt_matched_temperature_K",
        "zt_temperature_delta_K",
        "ZT_obs",
        "ZT_calc_from_obs",
        "ZT_pred_exp006",
    ]
    return [column for column in wanted if column in header]


def add_scenario_metadata(df: pd.DataFrame, scenario_key: str) -> pd.DataFrame:
    scenario = SCENARIO_BY_KEY[scenario_key]
    output = df.copy()
    output["scenario_key"] = scenario_key
    output["scenario_label"] = scenario["label"]
    output["reference_type"] = scenario["reference_type"]
    output["temperature_tolerance_K"] = scenario["temperature_tolerance_K"]
    output["primary_analysis"] = scenario["primary_analysis"]
    return output


def load_strict_scenarios(path: Path) -> dict[str, pd.DataFrame]:
    source = pd.read_csv(path, usecols=strict_input_columns(path), low_memory=False)
    source["sample_key"] = source["sample_key"].astype(str)
    for column in ["ZT_obs", "ZT_calc_from_obs", "ZT_pred_exp006"]:
        source[column] = material.finite(source[column])

    reported = source.loc[
        source["ZT_obs"].notna() & source["ZT_pred_exp006"].notna()
    ].copy()
    reported["ZT_reference"] = reported["ZT_obs"]
    reported["reported_ZT_obs"] = reported["ZT_obs"]
    reported["calculated_ZT_reference"] = reported["ZT_calc_from_obs"]
    reported = add_scenario_metadata(reported, "reported_zt_2K")

    calculated = source.loc[
        source["ZT_calc_from_obs"].notna() & source["ZT_pred_exp006"].notna()
    ].copy()
    calculated["ZT_reference"] = calculated["ZT_calc_from_obs"]
    calculated["reported_ZT_obs"] = calculated["ZT_obs"]
    calculated["calculated_ZT_reference"] = calculated["ZT_calc_from_obs"]
    calculated = add_scenario_metadata(calculated, "calculated_zt_2K")
    return {
        "reported_zt_2K": reported,
        "calculated_zt_2K": calculated,
    }


def build_reported_3k_scenario(prediction_path: Path, curve_path: Path) -> pd.DataFrame:
    predictions = pfzt.load_predictions(prediction_path)
    aggregated, _ = pfzt.load_curve_observations(
        curve_path, set(predictions["sample_key"]), 3
    )
    matches, _ = pfzt.match_curve_observations(predictions, aggregated, 3.0)
    frame = pd.DataFrame(index=predictions.index)
    for column in [
        "row_id",
        "sample_key",
        "sample_id",
        "doi",
        "formula_raw",
        "T_K",
        "S_uV_per_K",
        "sigma_S_per_m",
        "sigma_pred_S_per_m",
    ]:
        if column in predictions:
            frame[column] = predictions[column]
    frame = frame.rename(
        columns={
            "doi": "DOI",
            "formula_raw": "composition",
            "T_K": "temperature_K",
        }
    )
    for column in [
        "kappa_obs_W_per_mK",
        "kappa_matched_temperature_K",
        "kappa_temperature_delta_K",
        "zt_matched_temperature_K",
        "zt_temperature_delta_K",
        "ZT_obs",
    ]:
        frame[column] = matches[column]
    seebeck = material.finite(frame["S_uV_per_K"]) * 1.0e-6
    sigma_pred = material.finite(frame["sigma_pred_S_per_m"])
    sigma_obs = material.finite(frame["sigma_S_per_m"])
    temperature = material.finite(frame["temperature_K"])
    kappa = material.finite(frame["kappa_obs_W_per_mK"])
    valid_pred = seebeck.notna() & sigma_pred.gt(0) & temperature.gt(0) & kappa.gt(0)
    valid_calc = seebeck.notna() & sigma_obs.gt(0) & temperature.gt(0) & kappa.gt(0)
    frame["ZT_pred_exp006"] = np.where(
        valid_pred, np.square(seebeck) * sigma_pred * temperature / kappa, np.nan
    )
    frame["ZT_calc_from_obs"] = np.where(
        valid_calc, np.square(seebeck) * sigma_obs * temperature / kappa, np.nan
    )
    frame["ZT_reference"] = material.finite(frame["ZT_obs"])
    frame["reported_ZT_obs"] = frame["ZT_reference"]
    frame["calculated_ZT_reference"] = frame["ZT_calc_from_obs"]
    frame = frame.loc[
        material.finite(frame["ZT_reference"]).notna()
        & material.finite(frame["ZT_pred_exp006"]).notna()
    ].copy()
    return add_scenario_metadata(frame, "reported_zt_3K")


def normalize_for_material_analysis(
    scenario_rows: pd.DataFrame, classification: pd.DataFrame
) -> pd.DataFrame:
    frame = scenario_rows.copy()
    frame["ZT_obs"] = material.finite(frame["ZT_reference"])
    frame["ZT_pred_exp006"] = material.finite(frame["ZT_pred_exp006"])
    return material.add_classification(frame, classification)


def analyze_scenario(
    rows: pd.DataFrame,
    minimum_main_samples: int,
    high_zt_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_key = str(rows["scenario_key"].iloc[0])
    samples = material.aggregate_samples(rows)
    metrics = material.build_metrics(
        rows, samples, minimum_main_samples, high_zt_threshold
    )
    scenario = SCENARIO_BY_KEY[scenario_key]
    for frame in [samples, metrics]:
        frame["scenario_key"] = scenario_key
        frame["scenario_label"] = scenario["label"]
        frame["reference_type"] = scenario["reference_type"]
        frame["temperature_tolerance_K"] = scenario["temperature_tolerance_K"]
        frame["primary_analysis"] = scenario["primary_analysis"]
    samples = samples.rename(
        columns={
            "max_ZT_obs": "max_ZT_reference",
            "median_ZT_obs": "median_ZT_reference",
        }
    )
    metrics = metrics.rename(
        columns={
            "row_ZT_obs_median": "row_ZT_reference_median",
            "sample_max_ZT_obs_median": "sample_max_ZT_reference_median",
        }
    )
    return samples, metrics


def coverage_table(
    rows_by_scenario: dict[str, pd.DataFrame],
    samples_by_scenario: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        key = scenario["key"]
        rows = rows_by_scenario[key]
        samples = samples_by_scenario[key]
        output.append(
            {
                "scenario_key": key,
                "scenario_label": scenario["label"],
                "reference_type": scenario["reference_type"],
                "temperature_tolerance_K": scenario["temperature_tolerance_K"],
                "primary_analysis": scenario["primary_analysis"],
                "comparable_rows": len(rows),
                "comparable_samples": len(samples),
                "step31_classified_rows": int(
                    rows["classification_status"].eq("classified_step31").sum()
                ),
                "step31_classified_samples": int(
                    samples["classification_status"].eq("classified_step31").sum()
                ),
            }
        )
    return pd.DataFrame(output)


def shared_limit(frames: list[pd.DataFrame], actual: str, predicted: str) -> tuple[float, float]:
    values = pd.concat(
        [
            pd.concat([material.finite(frame[actual]), material.finite(frame[predicted])])
            for frame in frames
        ]
    ).dropna()
    upper = max(0.1, float(values.quantile(0.995)) * 1.05)
    return -0.03 * upper, upper


def plot_coverage(
    metrics: pd.DataFrame, main_groups: list[str], output: Path
) -> None:
    pivot = (
        metrics.loc[metrics["material_system_label"].isin(main_groups)]
        .pivot(
            index="material_system_label",
            columns="scenario_key",
            values="comparable_samples",
        )
        .reindex(main_groups)
    )
    y = np.arange(len(main_groups), dtype=float)
    height = 0.23
    fig, ax = plt.subplots(figsize=(12, max(7, len(main_groups) * 0.52)))
    for index, scenario in enumerate(SCENARIOS):
        values = pivot[scenario["key"]].fillna(0).to_numpy()
        position = y + (index - 1) * height
        bars = ax.barh(
            position,
            values,
            height=height,
            label=scenario["short_label"],
            color=scenario["color"],
            alpha=0.88,
        )
        ax.bar_label(bars, labels=[f"{int(value)}" for value in values], padding=2, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(main_groups)
    ax.invert_yaxis()
    ax.set_xlabel("Comparable samples")
    ax.set_title("Material-system sample coverage by ZT comparison scenario")
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parity_facets(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    main_groups: list[str],
    limits: tuple[float, float],
    output: Path,
) -> None:
    key = str(rows["scenario_key"].iloc[0])
    scenario = SCENARIO_BY_KEY[key]
    columns = 4
    row_count = math.ceil(len(main_groups) / columns)
    fig, axes = plt.subplots(
        row_count, columns, figsize=(15, 3.6 * row_count), sharex=True, sharey=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    lower, upper = limits
    for ax, label in zip(axes_array, main_groups):
        part = rows.loc[rows["material_system_label"].eq(label)]
        metric_rows = metrics.loc[metrics["material_system_label"].eq(label)]
        mae = float(metric_rows["row_ZT_MAE"].iloc[0]) if len(metric_rows) else math.nan
        ax.scatter(
            part["ZT_obs"],
            part["ZT_pred_exp006"],
            s=15,
            alpha=0.45,
            color=scenario["color"],
            edgecolors="none",
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#C53B2C", linewidth=1.3)
        ax.set_title(
            f"{label}\nn={part['sample_key'].nunique()}, MAE={mae:.3f}", fontsize=8.8
        )
        ax.grid(alpha=0.2)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
    for ax in axes_array[len(main_groups) :]:
        ax.axis("off")
    reference_label = "Calculated reference ZT" if key == "calculated_zt_2K" else "Reported ZT"
    fig.supxlabel(reference_label)
    fig.supylabel("exp006 predicted ZT")
    fig.suptitle(scenario["label"], fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.98))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_median_summary(
    metrics: pd.DataFrame, main_groups: list[str], output: Path
) -> None:
    baseline = metrics.loc[
        metrics["scenario_key"].eq("reported_zt_2K")
        & metrics["material_system_label"].isin(main_groups)
    ].set_index("material_system_label")
    order = baseline["sample_max_ZT_reference_median"].sort_values(ascending=False).index.tolist()
    y = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(17, max(7, len(order) * 0.52)), sharex=True, sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        part = (
            metrics.loc[
                metrics["scenario_key"].eq(scenario["key"])
                & metrics["material_system_label"].isin(order)
            ]
            .set_index("material_system_label")
            .reindex(order)
        )
        reference = part["sample_max_ZT_reference_median"].to_numpy(float)
        predicted = part["sample_max_ZT_pred_median"].to_numpy(float)
        for yi, x0, x1 in zip(y, reference, predicted):
            ax.plot([x0, x1], [yi, yi], color="#909090", linewidth=1.2, zorder=1)
        ax.scatter(reference, y, color="#2B7A98", marker="o", s=40, label="Reference median", zorder=2)
        ax.scatter(predicted, y, color="#C77720", marker="s", s=36, label="Predicted median", zorder=2)
        ax.set_title(scenario["short_label"])
        ax.grid(axis="x", alpha=0.22)
        ax.set_xlabel("Median sample-maximum ZT")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    axes[-1].legend(loc="lower right")
    fig.suptitle("Material-system median sample-maximum ZT across comparison scenarios", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    coverage: pd.DataFrame,
    main_groups: list[str],
    minimum_main_samples: int,
) -> None:
    lines = [
        "# Recommended material-system ZT comparison",
        "",
        "## Scenario roles",
        "",
        "- `reported_zt_2K`: primary analysis. Reported ZT and kappa are independently matched within ±2 K.",
        "- `calculated_zt_2K`: coverage extension. The reference is calculated from observed sigma, S, and kappa at ±2 K; it is not reported ZT.",
        "- `reported_zt_3K`: temperature-tolerance sensitivity analysis. It is not substituted for the primary ±2 K result.",
        "- No interpolation or extrapolation is used in any scenario.",
        "",
        "## Coverage",
        "",
        *[
            f"- {row.scenario_label}: {int(row.comparable_rows):,} rows, {int(row.comparable_samples):,} samples, {int(row.step31_classified_samples):,} Step31-classified samples."
            for row in coverage.itertuples(index=False)
        ],
        "",
        "## Material-system figures",
        "",
        f"- The common main-group set is defined from the primary scenario using at least {minimum_main_samples} classified samples.",
        f"- Main groups ({len(main_groups)}): {', '.join(main_groups)}.",
        "- Parity plots use a common x/y scale across all three scenarios.",
        "- Sample-level summaries use maximum ZT within rows where both the scenario reference and prediction are available.",
        "",
        "## Interpretation",
        "",
        "- Prefer the reported-ZT ±2 K result for conclusions.",
        "- Use calculated-ZT ±2 K to assess whether conclusions persist with greater coverage.",
        "- Use reported-ZT ±3 K only to assess sensitivity to the matching tolerance.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    args = parse_args()
    if args.minimum_main_samples < 1:
        raise ValueError("--minimum-main-samples must be at least 1")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    classification = material.load_sample_classification(args.classification.resolve())
    scenario_sources = load_strict_scenarios(args.strict_row_predictions.resolve())
    scenario_sources["reported_zt_3K"] = build_reported_3k_scenario(
        args.prediction_file.resolve(), args.curve_file.resolve()
    )

    rows_by_scenario: dict[str, pd.DataFrame] = {}
    samples_by_scenario: dict[str, pd.DataFrame] = {}
    metrics_by_scenario: dict[str, pd.DataFrame] = {}
    for scenario in SCENARIOS:
        key = scenario["key"]
        rows = normalize_for_material_analysis(scenario_sources[key], classification)
        samples, metrics = analyze_scenario(
            rows, args.minimum_main_samples, args.high_zt_threshold
        )
        rows_by_scenario[key] = rows
        samples_by_scenario[key] = samples
        metrics_by_scenario[key] = metrics

    baseline_metrics = metrics_by_scenario["reported_zt_2K"]
    main_groups = baseline_metrics.loc[
        baseline_metrics["main_analysis_group"], "material_system_label"
    ].tolist()
    for frame in [*rows_by_scenario.values(), *samples_by_scenario.values()]:
        frame["common_main_analysis_group"] = frame["material_system_label"].isin(main_groups)
    for frame in metrics_by_scenario.values():
        frame["scenario_specific_main_group"] = frame["main_analysis_group"]
        frame["common_main_analysis_group"] = frame["material_system_label"].isin(main_groups)

    combined_rows = pd.concat(
        [rows_by_scenario[item["key"]] for item in SCENARIOS], ignore_index=True
    )
    combined_samples = pd.concat(
        [samples_by_scenario[item["key"]] for item in SCENARIOS], ignore_index=True
    )
    combined_metrics = pd.concat(
        [metrics_by_scenario[item["key"]] for item in SCENARIOS], ignore_index=True
    )
    coverage = coverage_table(rows_by_scenario, samples_by_scenario)
    limits = shared_limit(
        [rows_by_scenario[item["key"]] for item in SCENARIOS], "ZT_obs", "ZT_pred_exp006"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=".zt_recommended_stage_", dir=output_dir.parent))
    try:
        figures = stage_dir / "figures"
        figures.mkdir()
        write_csv(combined_rows, stage_dir / "zt_scenario_row_comparisons.csv")
        write_csv(combined_samples, stage_dir / "zt_scenario_sample_summary.csv")
        write_csv(combined_metrics, stage_dir / "zt_scenario_material_system_metrics.csv")
        write_csv(coverage, stage_dir / "zt_scenario_coverage_summary.csv")
        plot_coverage(
            combined_metrics,
            main_groups,
            figures / "figure_material_system_sample_coverage_by_scenario.png",
        )
        for scenario in SCENARIOS:
            key = scenario["key"]
            plot_parity_facets(
                rows_by_scenario[key],
                metrics_by_scenario[key],
                main_groups,
                limits,
                figures / f"figure_material_system_parity_{key}.png",
            )
        plot_sample_median_summary(
            combined_metrics,
            main_groups,
            figures / "figure_material_system_sample_max_medians_by_scenario.png",
        )
        write_notes(
            stage_dir / "zt_material_system_recommended_notes.md",
            coverage,
            main_groups,
            args.minimum_main_samples,
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        stage_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    print(f"Created: {output_dir}")
    print(coverage.to_string(index=False))
    print(f"Common main material groups: {len(main_groups)}")


if __name__ == "__main__":
    main()
