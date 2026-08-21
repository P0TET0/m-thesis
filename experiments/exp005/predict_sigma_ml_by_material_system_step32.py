"""Create exp005 ML sigma predictions grouped exactly like exp006 broad families.

The electrical-conductivity model is the existing exp005 Step19/20 method:

    sigma_ML(T) = prefactor_C(T) * tau_eff_ML

Step12 point CSVs are not required.  The same prefactor is recovered without
approximation from the retained Step13 validation outputs as
``sigma_pred_validation / tau_eff_train``.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the exp006 implementation, rather than maintaining a second taxonomy.
from experiments.exp006.build_step6a_material_group_keys import build_candidates


DEFAULT_STEP14 = (
    PROJECT_ROOT
    / "data/output/starrydata2_step14_pf_zt_prediction/thermoelectric_validation_predictions_step14.csv"
)
DEFAULT_STEP13_RESULTS = (
    PROJECT_ROOT
    / "data/output/starrydata2_step13_sigma_validation/tau_validation_primary_results_step13.csv"
)
DEFAULT_STEP19_DIR = PROJECT_ROOT / "data/output/starrydata2_step19_tau_eff_ml_model"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/output/starrydata2_step32_sigma_ml_by_material_system"
PRIMARY_SPLIT = "split_doi_group_80_20_step18"
PRIMARY_VALIDATION_METHOD = "high_temperature_holdout"
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply exp005 ML sigma prediction and aggregate it by exp006 broad_family."
    )
    parser.add_argument("--step14-points", type=Path, default=DEFAULT_STEP14)
    parser.add_argument("--step13-results", type=Path, default=DEFAULT_STEP13_RESULTS)
    parser.add_argument("--step19-dir", type=Path, default=DEFAULT_STEP19_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--primary-split", default=PRIMARY_SPLIT)
    parser.add_argument("--evaluation-model", default="auto")
    parser.add_argument("--bin-width-K", type=float, default=25.0)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=2)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--replace-output", action="store_true")
    return parser.parse_args()


def read_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    kwargs: dict[str, Any] = {"low_memory": False}
    if usecols is not None:
        columns = pd.read_csv(path, nrows=0).columns
        kwargs["usecols"] = [column for column in usecols if column in columns]
    return pd.read_csv(path, **kwargs)


def require_columns(data: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def first_text(values: pd.Series) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def safe_slug(value: str) -> str:
    value = value.replace("::", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def select_evaluation_model(model_comparison: pd.DataFrame, split: str, requested: str) -> str:
    if requested != "auto":
        if requested not in set(model_comparison["model_name"].astype(str)):
            raise ValueError(f"Requested evaluation model is absent: {requested}")
        return requested
    rows = model_comparison[
        model_comparison["split_name"].astype(str).eq(split)
        & model_comparison["split_role"].astype(str).eq("test")
    ].copy()
    rows["log_tau_rmse"] = pd.to_numeric(rows["log_tau_rmse"], errors="coerce")
    rows = rows[np.isfinite(rows["log_tau_rmse"])]
    if rows.empty:
        raise ValueError(f"No test metrics found for split {split}")
    return str(rows.sort_values("log_tau_rmse", kind="mergesort").iloc[0]["model_name"])


def load_base_points(step14_path: Path, step13_path: Path) -> pd.DataFrame:
    point_columns = [
        "sample_key",
        "temperature_K",
        "SID",
        "DOI",
        "doi_url",
        "sample_id",
        "composition",
        "material_system",
        "n_or_p",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_for_pf_zt_S_per_m_step14",
        "prediction_source_step14",
        "validation_method_step13",
        "split_role_step13",
    ]
    points = read_csv(step14_path, point_columns)
    require_columns(
        points,
        [
            "sample_key",
            "temperature_K",
            "composition",
            "material_system",
            "n_or_p",
            "sigma_obs_S_per_m_step11",
            "sigma_pred_for_pf_zt_S_per_m_step14",
        ],
        step14_path.name,
    )
    results = read_csv(
        step13_path,
        ["sample_key", "validation_method_step13", "tau_eff_train_step13", "validation_status_step13"],
    )
    require_columns(
        results,
        ["sample_key", "validation_method_step13", "tau_eff_train_step13", "validation_status_step13"],
        step13_path.name,
    )
    results = results[
        results["validation_method_step13"].astype(str).eq(PRIMARY_VALIDATION_METHOD)
        & results["validation_status_step13"].astype(str).eq("ok")
    ].copy()
    if results["sample_key"].duplicated().any():
        raise ValueError("Step13 primary results contain duplicate sample_key values")
    points = points.merge(
        results[["sample_key", "tau_eff_train_step13"]],
        on="sample_key",
        how="inner",
        validate="many_to_one",
    )
    numeric_columns = [
        "temperature_K",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_for_pf_zt_S_per_m_step14",
        "tau_eff_train_step13",
    ]
    for column in numeric_columns:
        points[column] = pd.to_numeric(points[column], errors="coerce")
    valid = (
        np.isfinite(points["temperature_K"])
        & np.isfinite(points["sigma_obs_S_per_m_step11"])
        & points["sigma_obs_S_per_m_step11"].gt(0)
        & np.isfinite(points["sigma_pred_for_pf_zt_S_per_m_step14"])
        & points["sigma_pred_for_pf_zt_S_per_m_step14"].gt(0)
        & np.isfinite(points["tau_eff_train_step13"])
        & points["tau_eff_train_step13"].gt(0)
    )
    points = points.loc[valid].copy()
    points["prefactor_C_S_per_m_step32"] = (
        points["sigma_pred_for_pf_zt_S_per_m_step14"] / points["tau_eff_train_step13"]
    )
    points["prefactor_source_step32"] = (
        "exact Step13 recovery: sigma_pred_validation_S_per_m_step13 / tau_eff_train_step13"
    )
    points.insert(0, "prediction_row_id_step32", np.arange(len(points), dtype=np.int64))
    return points.reset_index(drop=True)


def classify_samples(points: pd.DataFrame) -> pd.DataFrame:
    samples = (
        points.groupby("sample_key", sort=True, as_index=False)
        .agg(
            composition=("composition", first_text),
            material_system=("material_system", first_text),
            n_or_p=("n_or_p", first_text),
            sample_id=("sample_id", first_text) if "sample_id" in points.columns else ("sample_key", "first"),
        )
    )
    adapter = pd.DataFrame(
        {
            "row_id": np.arange(len(samples), dtype=np.int64),
            "sample_key": samples["sample_key"],
            "validation_sample_group_id": samples["sample_key"],
            "formula_raw": samples["composition"],
            "material_name_raw": samples["composition"],
            "material_family_raw": samples["material_system"],
            "material_group_key": samples["material_system"],
            "sample_label": samples["sample_id"],
            "source_notes": "exp005 Step32 adapter for exp006 broad_family",
            "carrier_type": samples["n_or_p"],
        }
    )
    classified = build_candidates(adapter)
    keep = [
        "sample_key",
        "formula_raw",
        "material_family_raw",
        "parsed_elements",
        "parsed_element_count",
        "formula_parse_status",
        "formula_parse_source",
        "formula_system_key",
        "material_group_key_broad_family",
        "material_group_key_broad_family_reason",
    ]
    classified = classified[keep].copy()
    if classified["sample_key"].duplicated().any():
        raise AssertionError("exp006 classifier adapter produced duplicate sample keys")
    return classified


def add_prediction_columns(
    points: pd.DataFrame,
    tau_predictions: pd.DataFrame,
    tau_column: str,
    model_name: str,
    scope: str,
) -> pd.DataFrame:
    require_columns(tau_predictions, ["sample_key", tau_column], f"tau predictions ({scope})")
    tau = tau_predictions[["sample_key", tau_column]].copy()
    tau[tau_column] = pd.to_numeric(tau[tau_column], errors="coerce")
    tau = tau[np.isfinite(tau[tau_column]) & tau[tau_column].gt(0)].drop_duplicates("sample_key")
    output = points.merge(tau, on="sample_key", how="inner", validate="many_to_one")
    output = output.rename(columns={tau_column: "tau_eff_pred_ML_step32"})
    output["sigma_pred_ML_S_per_m_step32"] = (
        output["prefactor_C_S_per_m_step32"] * output["tau_eff_pred_ML_step32"]
    )
    output["model_name_step32"] = model_name
    output["prediction_scope_step32"] = scope
    obs = output["sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
    pred = output["sigma_pred_ML_S_per_m_step32"].to_numpy(dtype=float)
    output["sigma_ML_log_error_step32"] = np.log(pred) - np.log(obs)
    output["sigma_ML_relative_error_step32"] = np.abs(pred - obs) / np.maximum(np.abs(obs), EPS)
    return output


def metric_record(group: pd.DataFrame) -> dict[str, Any]:
    obs = pd.to_numeric(group["sigma_obs_S_per_m_step11"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(group["sigma_pred_ML_S_per_m_step32"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    obs = obs[valid]
    pred = pred[valid]
    if not len(obs):
        return {"row_count": 0, "sample_count": 0}
    log_error = np.log(pred) - np.log(obs)
    relative = np.abs(pred - obs) / np.maximum(obs, EPS)
    log_obs = np.log(obs)
    denominator = np.sum((log_obs - log_obs.mean()) ** 2)
    return {
        "row_count": int(len(obs)),
        "sample_count": int(group.loc[valid, "sample_key"].nunique()),
        "sigma_log_rmse_step32": float(np.sqrt(np.mean(log_error**2))),
        "sigma_log_mae_step32": float(np.mean(np.abs(log_error))),
        "sigma_mape_step32": float(np.mean(relative)),
        "sigma_log_bias_step32": float(np.mean(log_error)),
        "sigma_r2_log_step32": float(1 - np.sum(log_error**2) / denominator) if denominator > 0 else math.nan,
        "within_factor_2_rate_step32": float(np.mean((pred / obs >= 0.5) & (pred / obs <= 2.0))),
    }


def build_metrics(evaluation: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for family, group in evaluation.groupby("material_group_key_broad_family", sort=True):
        record = {"material_group_key_broad_family": family}
        record.update(metric_record(group))
        records.append(record)
    overall = {"material_group_key_broad_family": "__all_material_systems__"}
    overall.update(metric_record(evaluation))
    records.append(overall)
    return pd.DataFrame(records).sort_values("material_group_key_broad_family", kind="mergesort")


def bin_center(temperature: pd.Series, width: float) -> pd.Series:
    return np.floor(pd.to_numeric(temperature, errors="coerce") / width + 0.5) * width


def aggregate_curves(screening: pd.DataFrame, width: float, min_rows: int, min_samples: int) -> pd.DataFrame:
    data = screening.copy()
    data["temperature_bin_center_K_step32"] = bin_center(data["temperature_K"], width)
    curves = (
        data.groupby(
            ["material_group_key_broad_family", "temperature_bin_center_K_step32"],
            sort=True,
        )
        .agg(
            row_count=("sample_key", "size"),
            sample_count=("sample_key", "nunique"),
            sigma_pred_ML_median_S_per_m_step32=("sigma_pred_ML_S_per_m_step32", "median"),
            sigma_pred_ML_q25_S_per_m_step32=("sigma_pred_ML_S_per_m_step32", lambda values: values.quantile(0.25)),
            sigma_pred_ML_q75_S_per_m_step32=("sigma_pred_ML_S_per_m_step32", lambda values: values.quantile(0.75)),
            sigma_obs_median_S_per_m_step32=("sigma_obs_S_per_m_step11", "median"),
            sigma_obs_q25_S_per_m_step32=("sigma_obs_S_per_m_step11", lambda values: values.quantile(0.25)),
            sigma_obs_q75_S_per_m_step32=("sigma_obs_S_per_m_step11", lambda values: values.quantile(0.75)),
        )
        .reset_index()
    )
    curves["plot_in_material_curve_step32"] = curves["row_count"].ge(min_rows) & curves[
        "sample_count"
    ].ge(min_samples)
    return curves


def build_index(screening: pd.DataFrame, evaluation: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    screen_counts = screening.groupby("material_group_key_broad_family").agg(
        screening_row_count=("sample_key", "size"),
        screening_sample_count=("sample_key", "nunique"),
        temperature_min_K=("temperature_K", "min"),
        temperature_max_K=("temperature_K", "max"),
    )
    eval_counts = evaluation.groupby("material_group_key_broad_family").agg(
        evaluation_row_count=("sample_key", "size"),
        evaluation_sample_count=("sample_key", "nunique"),
    )
    curve_counts = curves.groupby("material_group_key_broad_family").agg(
        curve_bin_count=("temperature_bin_center_K_step32", "size"),
        plotted_curve_bin_count=("plot_in_material_curve_step32", "sum"),
    )
    index = screen_counts.join(eval_counts, how="left").join(curve_counts, how="left").reset_index()
    index["output_slug"] = index["material_group_key_broad_family"].map(safe_slug)
    return index.sort_values(["screening_sample_count", "material_group_key_broad_family"], ascending=[False, True])


def create_figures(
    curves: pd.DataFrame,
    evaluation: pd.DataFrame,
    index: pd.DataFrame,
    figure_dir: Path,
    dpi: int,
) -> None:
    curve_dir = figure_dir / "material_curves"
    scatter_dir = figure_dir / "evaluation_scatter"
    curve_dir.mkdir(parents=True, exist_ok=True)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    families = index["material_group_key_broad_family"].tolist()
    for family in families:
        slug = safe_slug(family)
        curve = curves[
            curves["material_group_key_broad_family"].eq(family)
            & curves["plot_in_material_curve_step32"]
        ].sort_values("temperature_bin_center_K_step32")
        if not curve.empty:
            x = curve["temperature_bin_center_K_step32"].to_numpy(dtype=float)
            pred = curve["sigma_pred_ML_median_S_per_m_step32"].to_numpy(dtype=float)
            pred_q25 = curve["sigma_pred_ML_q25_S_per_m_step32"].to_numpy(dtype=float)
            pred_q75 = curve["sigma_pred_ML_q75_S_per_m_step32"].to_numpy(dtype=float)
            obs = curve["sigma_obs_median_S_per_m_step32"].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(8.0, 5.2))
            ax.plot(x, obs, color="#202020", marker="o", ms=3, lw=1.5, label="measured median")
            ax.plot(x, pred, color="#d95f02", marker="o", ms=3, lw=1.8, label="exp005 ML median")
            ax.fill_between(x, pred_q25, pred_q75, color="#d95f02", alpha=0.18, label="ML IQR")
            ax.set_yscale("log")
            ax.set_xlabel("Temperature (K)")
            ax.set_ylabel("Electrical conductivity (S/m)")
            ax.set_title(family)
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(curve_dir / f"sigma_ml_curve_{slug}.png", dpi=dpi)
            plt.close(fig)

        scatter = evaluation[evaluation["material_group_key_broad_family"].eq(family)]
        if not scatter.empty:
            obs_values = scatter["sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
            pred_values = scatter["sigma_pred_ML_S_per_m_step32"].to_numpy(dtype=float)
            lower = max(float(min(obs_values.min(), pred_values.min())), 1e-12)
            upper = float(max(obs_values.max(), pred_values.max()))
            fig, ax = plt.subplots(figsize=(5.8, 5.5))
            ax.scatter(obs_values, pred_values, s=12, alpha=0.35, color="#1b9e77", edgecolors="none")
            ax.plot([lower, upper], [lower, upper], color="#333333", linestyle="--", lw=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Measured sigma (S/m)")
            ax.set_ylabel("exp005 ML predicted sigma (S/m)")
            ax.set_title(f"{family} | DOI holdout")
            ax.grid(True, which="both", alpha=0.25)
            fig.tight_layout()
            fig.savefig(scatter_dir / f"sigma_ml_scatter_{slug}.png", dpi=dpi)
            plt.close(fig)

    plotted = [family for family in families if curves.loc[
        curves["material_group_key_broad_family"].eq(family), "plot_in_material_curve_step32"
    ].any()]
    if plotted:
        ncols = 3
        nrows = math.ceil(len(plotted) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.1 * nrows), squeeze=False)
        for ax, family in zip(axes.flat, plotted):
            curve = curves[
                curves["material_group_key_broad_family"].eq(family)
                & curves["plot_in_material_curve_step32"]
            ].sort_values("temperature_bin_center_K_step32")
            x = curve["temperature_bin_center_K_step32"]
            ax.plot(x, curve["sigma_obs_median_S_per_m_step32"], color="#202020", lw=1.3, label="measured")
            ax.plot(x, curve["sigma_pred_ML_median_S_per_m_step32"], color="#d95f02", lw=1.5, label="ML")
            ax.set_yscale("log")
            ax.set_title(family, fontsize=10)
            ax.set_xlabel("T (K)")
            ax.set_ylabel("sigma (S/m)")
            ax.grid(True, which="both", alpha=0.22)
        for ax in axes.flat[len(plotted) :]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figure_dir / "sigma_ml_material_system_curves_grid_step32.png", dpi=dpi)
        plt.close(fig)

    # DOI-group holdout scatter plots: one combined view and one family grid.
    evaluated_families = [
        family
        for family in families
        if evaluation["material_group_key_broad_family"].eq(family).any()
    ]
    if evaluated_families:
        all_obs = evaluation["sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
        all_pred = evaluation["sigma_pred_ML_S_per_m_step32"].to_numpy(dtype=float)
        lower = max(float(min(all_obs.min(), all_pred.min())), 1e-12)
        upper = float(max(all_obs.max(), all_pred.max()))

        fig, ax = plt.subplots(figsize=(8.8, 7.0))
        colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(evaluated_families)))
        for color, family in zip(colors, evaluated_families):
            data = evaluation[evaluation["material_group_key_broad_family"].eq(family)]
            ax.scatter(
                data["sigma_obs_S_per_m_step11"],
                data["sigma_pred_ML_S_per_m_step32"],
                s=11,
                alpha=0.35,
                color=color,
                edgecolors="none",
                label=family,
            )
        ax.plot([lower, upper], [lower, upper], color="#222222", linestyle="--", lw=1.3, label="ideal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_xlabel("Measured sigma (S/m)")
        ax.set_ylabel("exp005 ML predicted sigma (S/m)")
        ax.set_title("DOI-group holdout: all exp006 material systems")
        ax.grid(True, which="both", alpha=0.22)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=True)
        fig.tight_layout()
        fig.savefig(figure_dir / "sigma_ml_doi_holdout_scatter_all_step32.png", dpi=dpi)
        plt.close(fig)

        ncols = 3
        nrows = math.ceil(len(evaluated_families) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 4.7 * nrows), squeeze=False)
        for ax, family in zip(axes.flat, evaluated_families):
            data = evaluation[evaluation["material_group_key_broad_family"].eq(family)]
            obs_values = data["sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
            pred_values = data["sigma_pred_ML_S_per_m_step32"].to_numpy(dtype=float)
            family_lower = max(float(min(obs_values.min(), pred_values.min())), 1e-12)
            family_upper = float(max(obs_values.max(), pred_values.max()))
            ax.scatter(obs_values, pred_values, s=11, alpha=0.35, color="#1b9e77", edgecolors="none")
            ax.plot(
                [family_lower, family_upper],
                [family_lower, family_upper],
                color="#333333",
                linestyle="--",
                lw=1.1,
            )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(family_lower, family_upper)
            ax.set_ylim(family_lower, family_upper)
            ax.set_title(family, fontsize=10)
            ax.set_xlabel("Measured sigma (S/m)")
            ax.set_ylabel("ML predicted sigma (S/m)")
            ax.grid(True, which="both", alpha=0.22)
        for ax in axes.flat[len(evaluated_families) :]:
            ax.axis("off")
        fig.suptitle("DOI-group holdout scatter by exp006 material system", fontsize=15, y=1.0)
        fig.tight_layout()
        fig.savefig(figure_dir / "sigma_ml_doi_holdout_scatter_grid_step32.png", dpi=dpi)
        plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    base: pd.DataFrame,
    screening: pd.DataFrame,
    evaluation: pd.DataFrame,
    metrics: pd.DataFrame,
    index: pd.DataFrame,
    evaluation_model: str,
    final_model: str,
) -> None:
    overall = metrics[metrics["material_group_key_broad_family"].eq("__all_material_systems__")].iloc[0]
    lines = [
        "Step32 exp005 ML sigma prediction by exp006 material system",
        "===========================================================",
        "",
        "Method",
        "------",
        "sigma_ML(T) = prefactor_C(T) * tau_eff_ML.",
        "The broad-family classification calls experiments/exp006/build_step6a_material_group_keys.py directly.",
        "The Step12 prefactor is recovered exactly from retained Step13 values:",
        "prefactor_C(T) = sigma_pred_validation(T) / tau_eff_train.",
        "",
        "Scope",
        "-----",
        f"Step14/Step13 valid point rows: {len(base):,}",
        f"screening rows: {len(screening):,}",
        f"screening samples: {screening['sample_key'].nunique():,}",
        f"screening final model: {final_model}",
        "screening note: model was fitted on all recommended ML samples; these rows are not unbiased evaluation.",
        f"DOI holdout evaluation rows: {len(evaluation):,}",
        f"DOI holdout evaluation samples: {evaluation['sample_key'].nunique():,}",
        f"DOI holdout evaluation model: {evaluation_model}",
        f"DOI holdout split: {args.primary_split}",
        "",
        "Overall DOI holdout metrics",
        "----------------------------",
        f"sigma log RMSE: {overall['sigma_log_rmse_step32']:.6g}",
        f"sigma log MAE: {overall['sigma_log_mae_step32']:.6g}",
        f"sigma MAPE: {overall['sigma_mape_step32']:.6g}",
        f"within factor 2: {overall['within_factor_2_rate_step32']:.6g}",
        "",
        "Material systems",
        "----------------",
    ]
    metric_lookup = metrics.set_index("material_group_key_broad_family")
    for row in index.itertuples(index=False):
        family = row.material_group_key_broad_family
        metric = metric_lookup.loc[family] if family in metric_lookup.index else None
        metric_text = "not evaluated"
        if metric is not None:
            metric_text = f"log_RMSE={metric.sigma_log_rmse_step32:.4g}, factor2={metric.within_factor_2_rate_step32:.3f}"
        lines.append(
            f"- {family}: screening samples={int(row.screening_sample_count)}, "
            f"evaluation samples={int(row.evaluation_sample_count) if pd.notna(row.evaluation_sample_count) else 0}, "
            f"plotted bins={int(row.plotted_curve_bin_count)}, {metric_text}"
        )
    lines.extend(
        [
            "",
            "Caveats",
            "-------",
            "The material-system curves are medians of sample predictions in 25 K bins, not predictions for a single composition.",
            "The exp006 broad_family taxonomy is heuristic and includes an unknown/catch-all group.",
            "tau_eff is the exp005 relative effective scale, not a physical relaxation time in seconds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate(
    base: pd.DataFrame,
    screening: pd.DataFrame,
    evaluation: pd.DataFrame,
    curves: pd.DataFrame,
    index: pd.DataFrame,
    output: Path,
) -> None:
    if base.empty or screening.empty or evaluation.empty:
        raise AssertionError("base, screening, and evaluation outputs must be non-empty")
    if screening["prediction_row_id_step32"].duplicated().any():
        raise AssertionError("screening output duplicated point rows")
    calculated = screening["prefactor_C_S_per_m_step32"] * screening["tau_eff_pred_ML_step32"]
    if not np.allclose(calculated, screening["sigma_pred_ML_S_per_m_step32"], rtol=1e-12, atol=0):
        raise AssertionError("sigma_ML formula validation failed")
    if screening["material_group_key_broad_family"].isna().any():
        raise AssertionError("material group classification contains nulls")
    if set(index["material_group_key_broad_family"]) != set(screening["material_group_key_broad_family"]):
        raise AssertionError("material-system index is incomplete")
    if int(curves["row_count"].sum()) != len(screening):
        raise AssertionError("curve aggregation lost rows")
    required = [
        "sigma_ml_screening_predictions_by_material_system_step32.csv",
        "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv",
        "sigma_ml_material_system_curves_step32.csv",
        "sigma_ml_metrics_by_material_system_step32.csv",
        "material_system_index_step32.csv",
        "sample_material_system_classification_step32.csv",
        "step32_sigma_ml_by_material_system_report.txt",
        "figures/sigma_ml_material_system_curves_grid_step32.png",
        "figures/sigma_ml_doi_holdout_scatter_all_step32.png",
        "figures/sigma_ml_doi_holdout_scatter_grid_step32.png",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise AssertionError(f"Required Step32 outputs are missing: {missing}")


def install_staging(staging: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    staged_files = [path for path in staging.rglob("*") if path.is_file()]
    staged_relatives = {path.relative_to(staging) for path in staged_files}
    existing_files = [path for path in output.rglob("*") if path.is_file()]
    for source in staged_files:
        relative = source.relative_to(staging)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    for existing in existing_files:
        if existing.relative_to(output) not in staged_relatives:
            existing.unlink()
    for directory in sorted(
        [path for path in output.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    shutil.rmtree(staging)


def main() -> None:
    args = parse_args()
    if args.bin_width_K <= 0:
        raise ValueError("--bin-width-K must be positive")
    if min(args.min_rows_per_bin, args.min_samples_per_bin) < 1:
        raise ValueError("minimum bin counts must be at least 1")
    output = args.output_dir.resolve()
    if output.exists() and not args.replace_output:
        raise FileExistsError(f"Output exists; pass --replace-output to replace it: {output}")

    base = load_base_points(args.step14_points, args.step13_results)
    classification = classify_samples(base)
    base = base.merge(classification, on="sample_key", how="left", validate="many_to_one")

    model_comparison = read_csv(args.step19_dir / "tau_eff_ml_model_comparison_step19.csv")
    evaluation_model = select_evaluation_model(model_comparison, args.primary_split, args.evaluation_model)

    evaluation_tau = read_csv(
        args.step19_dir / "tau_eff_ml_predictions_step19.csv",
        ["sample_key", "model_name", "split_name", "split_role", "pred_tau_eff_step19"],
    )
    evaluation_tau = evaluation_tau[
        evaluation_tau["model_name"].astype(str).eq(evaluation_model)
        & evaluation_tau["split_name"].astype(str).eq(args.primary_split)
        & evaluation_tau["split_role"].astype(str).eq("test")
    ].copy()
    evaluation = add_prediction_columns(
        base,
        evaluation_tau,
        "pred_tau_eff_step19",
        evaluation_model,
        "unbiased DOI-group holdout evaluation",
    )

    final_tau = read_csv(
        args.step19_dir / "tau_eff_ml_final_all_samples_predictions_step19.csv",
        ["sample_key", "pred_tau_eff_final_model_step19", "final_model_name_step19"],
    )
    final_models = sorted(set(final_tau["final_model_name_step19"].dropna().astype(str)))
    final_model = ";".join(final_models) if final_models else "unknown"
    screening = add_prediction_columns(
        base,
        final_tau,
        "pred_tau_eff_final_model_step19",
        final_model,
        "all-sample screening; model trained on all recommended ML samples",
    )

    metrics = build_metrics(evaluation)
    curves = aggregate_curves(
        screening,
        args.bin_width_K,
        args.min_rows_per_bin,
        args.min_samples_per_bin,
    )
    index = build_index(screening, evaluation, curves)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".step32_staging_", dir=output.parent))
    try:
        screening.to_csv(staging / "sigma_ml_screening_predictions_by_material_system_step32.csv", index=False)
        evaluation.to_csv(staging / "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv", index=False)
        curves.to_csv(staging / "sigma_ml_material_system_curves_step32.csv", index=False)
        metrics.to_csv(staging / "sigma_ml_metrics_by_material_system_step32.csv", index=False)
        index.to_csv(staging / "material_system_index_step32.csv", index=False)
        classification.to_csv(staging / "sample_material_system_classification_step32.csv", index=False)

        by_material = staging / "by_material_system"
        by_material.mkdir(parents=True, exist_ok=True)
        for row in index.itertuples(index=False):
            family = row.material_group_key_broad_family
            slug = row.output_slug
            screening[screening["material_group_key_broad_family"].eq(family)].to_csv(
                by_material / f"sigma_ml_screening_{slug}.csv", index=False
            )
            evaluation[evaluation["material_group_key_broad_family"].eq(family)].to_csv(
                by_material / f"sigma_ml_doi_holdout_{slug}.csv", index=False
            )

        create_figures(curves, evaluation, index, staging / "figures", args.dpi)
        write_report(
            staging / "step32_sigma_ml_by_material_system_report.txt",
            args,
            base,
            screening,
            evaluation,
            metrics,
            index,
            evaluation_model,
            final_model,
        )
        validate(base, screening, evaluation, curves, index, staging)
        install_staging(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    overall = metrics[metrics["material_group_key_broad_family"].eq("__all_material_systems__")].iloc[0]
    print("Done.")
    print(f"Output: {output}")
    print(f"Material systems: {screening['material_group_key_broad_family'].nunique()}")
    print(f"Screening rows/samples: {len(screening):,} / {screening['sample_key'].nunique():,}")
    print(f"Evaluation model: {evaluation_model}")
    print(f"Evaluation rows/samples: {len(evaluation):,} / {evaluation['sample_key'].nunique():,}")
    print(f"Overall DOI-holdout sigma log RMSE: {overall['sigma_log_rmse_step32']:.6g}")
    print(f"Overall DOI-holdout within-factor-2: {overall['within_factor_2_rate_step32']:.6g}")


if __name__ == "__main__":
    main()
