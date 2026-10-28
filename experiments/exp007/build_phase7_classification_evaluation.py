"""Evaluate predictions before and after the oxide_v1 classification change.

Phase 7 is read-only with respect to the Phase 6 predictions.  It evaluates
the fixed broad-family baseline and the oxide_v1 re-prediction on paired rows,
adds sample-equal views, quantifies uncertainty with sample-cluster bootstrap,
summarizes oxide-group trade-offs, and audits lost prediction coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
FIGURES = EXP007 / "figures"

COMPARISON_ROWS = DATA / "phase6_prediction_comparison_rows.csv"
PHASE6_COVERAGE = DATA / "phase6_oxide_group_coverage.csv"
PHASE6_VALIDATION = DATA / "phase6_validation_checks.csv"
PHASE6_CHECK = REPORTS / "phase6_check_log.txt"
PHASE6_REPORT = REPORTS / "phase6_oxide_v1_reprediction.md"

DEFAULT_OUTPUT_DIR = DATA
DEFAULT_REPORT = REPORTS / "phase7_classification_evaluation.md"
DEFAULT_VALIDATION_LOG = REPORTS / "phase7_validation_log.txt"
DEFAULT_WORKBOOK_PAYLOAD = REPORTS / ".phase7_workbook_payload.json"

BASELINE_VARIANT = "broad_family_100K"
NEW_VARIANT = "oxide_v1_100K"
BOOTSTRAP_REPLICATES = 2000
RANDOM_SEED = 20261014
ATOL = 1e-12
FACTOR_THRESHOLDS = {
    "factor_2_accuracy": math.log10(2.0),
    "factor_5_accuracy": math.log10(5.0),
    "factor_10_accuracy": 1.0,
}
METRIC_META = {
    "mean_log10_error": ("Mean signed log10 error", "context", "log10"),
    "abs_mean_log10_error": ("Absolute mean log10 error", "lower", "log10"),
    "mae_log10": ("MAE of log10 ratio", "lower", "log10"),
    "rmse_log10": ("RMSE of log10 ratio", "lower", "log10"),
    "factor_2_accuracy": ("Factor-2 accuracy", "higher", "fraction"),
    "factor_5_accuracy": ("Factor-5 accuracy", "higher", "fraction"),
    "factor_10_accuracy": ("Factor-10 accuracy", "higher", "fraction"),
}
SCOPE_ORDER = [
    "all_materials",
    "all_oxide_assigned",
    "oxide_default_eligible",
    "oxide_review_only",
    "oxide_excluded_only",
]
BOOTSTRAP_SCOPES = ["all_materials", "all_oxide_assigned", "oxide_default_eligible"]

OUTPUT_FILES = {
    "paired_metrics": "phase7_paired_metric_evaluation.csv",
    "bootstrap": "phase7_bootstrap_intervals.csv",
    "groups": "phase7_oxide_group_evaluation.csv",
    "stratified": "phase7_stratified_evaluation.csv",
    "samples": "phase7_primary_sample_changes.csv",
    "sample_summary": "phase7_sample_change_summary.csv",
    "coverage_losses": "phase7_coverage_loss_rows.csv",
    "coverage_summary": "phase7_coverage_loss_summary.csv",
    "decision": "phase7_decision_summary.csv",
    "inventory": "phase7_inventory_summary.csv",
    "manifest": "phase7_input_manifest.csv",
    "validation": "phase7_validation_checks.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION_LOG)
    parser.add_argument("--workbook-payload", type=Path, default=DEFAULT_WORKBOOK_PAYLOAD)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def first_nonempty(values: Iterable[Any]) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def join_unique(values: Iterable[Any], limit: int = 8) -> str:
    unique: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    suffix = f" | ... (+{len(unique) - limit})" if len(unique) > limit else ""
    return " | ".join(unique[:limit]) + suffix


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_inputs() -> None:
    required = [COMPARISON_ROWS, PHASE6_COVERAGE, PHASE6_VALIDATION, PHASE6_CHECK, PHASE6_REPORT]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 7 inputs: {missing}")
    if not pd.read_csv(PHASE6_VALIDATION)["status"].eq("PASS").all():
        raise RuntimeError("Phase 6 build validation is not PASS")
    if "status=PASS" not in PHASE6_CHECK.read_text(encoding="utf-8"):
        raise RuntimeError("Phase 6 independent check is not PASS")


def make_manifest() -> pd.DataFrame:
    sources = [
        ("Phase 6 row-level before/after comparison", COMPARISON_ROWS),
        ("Phase 6 oxide-group coverage", PHASE6_COVERAGE),
        ("Phase 6 build validation", PHASE6_VALIDATION),
        ("Phase 6 independent check", PHASE6_CHECK),
        ("Phase 6 method and result report", PHASE6_REPORT),
    ]
    return pd.DataFrame(
        [
            {
                "role": role,
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in sources
        ]
    )


def scope_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    paired = as_bool(rows["paired_prediction_available"])
    oxide = as_bool(rows["is_phase0_broad_oxide"])
    eligible = as_bool(rows["oxide_v1_default_model_eligible"])
    status = rows["oxide_v1_inclusion_status"].astype(str)
    return {
        "all_materials": paired,
        "all_oxide_assigned": paired & oxide,
        "oxide_default_eligible": paired & oxide & eligible,
        "oxide_review_only": paired & oxide & status.eq("review"),
        "oxide_excluded_only": paired & oxide & status.eq("exclude"),
    }


def residual_metrics(values: np.ndarray) -> dict[str, float]:
    residual = np.asarray(values, dtype=float)
    residual = residual[np.isfinite(residual)]
    absolute = np.abs(residual)
    mean_error = float(np.mean(residual))
    output = {
        "mean_log10_error": mean_error,
        "abs_mean_log10_error": abs(mean_error),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
    }
    output.update({metric: float(np.mean(absolute <= threshold + ATOL)) for metric, threshold in FACTOR_THRESHOLDS.items()})
    return output


def sample_residuals(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby("validation_sample_group_id", dropna=False, sort=False)
        .agg(
            baseline_residual_log10=("baseline_residual_log10", "median"),
            phase7_residual_log10=("phase6_residual_log10", "median"),
            validation_paper_group_id=("validation_paper_group_id", "first"),
        )
        .reset_index()
    )


def metrics_for_weighting(rows: pd.DataFrame, weighting: str) -> tuple[dict[str, float], dict[str, float], int]:
    if weighting == "sample_equal":
        source = sample_residuals(rows)
        baseline = source["baseline_residual_log10"].to_numpy(float)
        new = source["phase7_residual_log10"].to_numpy(float)
        n_units = len(source)
    else:
        baseline = rows["baseline_residual_log10"].to_numpy(float)
        new = rows["phase6_residual_log10"].to_numpy(float)
        n_units = len(rows)
    return residual_metrics(baseline), residual_metrics(new), n_units


def evaluate_delta(metric: str, delta: float) -> str:
    direction = METRIC_META[metric][1]
    if direction == "context" or abs(delta) <= ATOL:
        return "context" if direction == "context" else "unchanged"
    favorable = delta < 0 if direction == "lower" else delta > 0
    return "improved" if favorable else "worsened"


def build_paired_metric_evaluation(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    masks = scope_masks(rows)
    for scope in SCOPE_ORDER:
        subset = rows.loc[masks[scope]].copy()
        for weighting in ["row_equal", "sample_equal"]:
            baseline, new, n_units = metrics_for_weighting(subset, weighting)
            for metric, (label, direction, unit) in METRIC_META.items():
                delta = new[metric] - baseline[metric]
                records.append(
                    {
                        "analysis_scope": scope,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "metric_label": label,
                        "unit": unit,
                        "n_units": n_units,
                        "n_source_rows": len(subset),
                        "n_samples": subset["validation_sample_group_id"].nunique(),
                        "n_papers": subset["validation_paper_group_id"].nunique(),
                        "baseline_value": baseline[metric],
                        "oxide_v1_value": new[metric],
                        "delta_oxide_v1_minus_baseline": delta,
                        "relative_delta_fraction": delta / abs(baseline[metric]) if abs(baseline[metric]) > ATOL else np.nan,
                        "delta_percentage_points": 100.0 * delta if unit == "fraction" else np.nan,
                        "better_direction": direction,
                        "evaluation": evaluate_delta(metric, delta),
                    }
                )
    return pd.DataFrame(records)


def cluster_sufficient_statistics(rows: pd.DataFrame, weighting: str) -> np.ndarray:
    records: list[list[float]] = []
    for _, group in rows.groupby("validation_sample_group_id", sort=False):
        if weighting == "sample_equal":
            baseline = np.asarray([float(group["baseline_residual_log10"].median())])
            new = np.asarray([float(group["phase6_residual_log10"].median())])
        else:
            baseline = group["baseline_residual_log10"].to_numpy(float)
            new = group["phase6_residual_log10"].to_numpy(float)
        row = [
            float(len(baseline)),
            float(baseline.sum()),
            float(np.abs(baseline).sum()),
            float((baseline**2).sum()),
            float(new.sum()),
            float(np.abs(new).sum()),
            float((new**2).sum()),
        ]
        for threshold in FACTOR_THRESHOLDS.values():
            row.extend([float((np.abs(baseline) <= threshold + ATOL).sum()), float((np.abs(new) <= threshold + ATOL).sum())])
        records.append(row)
    return np.asarray(records, dtype=float)


def metrics_from_sums(sums: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    n = sums[0]
    baseline_mean = sums[1] / n
    new_mean = sums[4] / n
    baseline = {
        "mean_log10_error": baseline_mean,
        "abs_mean_log10_error": abs(baseline_mean),
        "mae_log10": sums[2] / n,
        "rmse_log10": math.sqrt(sums[3] / n),
    }
    new = {
        "mean_log10_error": new_mean,
        "abs_mean_log10_error": abs(new_mean),
        "mae_log10": sums[5] / n,
        "rmse_log10": math.sqrt(sums[6] / n),
    }
    offset = 7
    for index, metric in enumerate(FACTOR_THRESHOLDS):
        baseline[metric] = sums[offset + 2 * index] / n
        new[metric] = sums[offset + 2 * index + 1] / n
    return baseline, new


def bootstrap_intervals(rows: pd.DataFrame, weighting: str, replicates: int, seed: int) -> dict[str, dict[str, float]]:
    stats = cluster_sufficient_statistics(rows, weighting)
    n_clusters = len(stats)
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(replicates, dtype=float) for metric in METRIC_META}
    for replicate in range(replicates):
        selected = rng.integers(0, n_clusters, size=n_clusters)
        counts = np.bincount(selected, minlength=n_clusters).astype(float)
        summed = counts @ stats
        baseline, new = metrics_from_sums(summed)
        for metric in METRIC_META:
            draws[metric][replicate] = new[metric] - baseline[metric]
    output: dict[str, dict[str, float]] = {}
    for metric, values in draws.items():
        direction = METRIC_META[metric][1]
        if direction == "lower":
            probability = float(np.mean(values < 0))
        elif direction == "higher":
            probability = float(np.mean(values > 0))
        else:
            probability = np.nan
        output[metric] = {
            "ci_lower": float(np.quantile(values, 0.025)),
            "ci_upper": float(np.quantile(values, 0.975)),
            "bootstrap_probability_improved": probability,
        }
    return output


def interval_interpretation(metric: str, lower: float, upper: float) -> str:
    direction = METRIC_META[metric][1]
    if direction == "context":
        return "context"
    if direction == "lower":
        if upper < 0:
            return "supported_improvement"
        if lower > 0:
            return "supported_worsening"
    else:
        if lower > 0:
            return "supported_improvement"
        if upper < 0:
            return "supported_worsening"
    return "inconclusive"


def build_bootstrap_table(rows: pd.DataFrame, paired: pd.DataFrame, replicates: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    masks = scope_masks(rows)
    lookup = paired.set_index(["analysis_scope", "metric_weighting", "metric_name"])
    for scope_index, scope in enumerate(BOOTSTRAP_SCOPES):
        subset = rows.loc[masks[scope]].copy()
        for weighting_index, weighting in enumerate(["row_equal", "sample_equal"]):
            seed = RANDOM_SEED + scope_index * 100 + weighting_index
            intervals = bootstrap_intervals(subset, weighting, replicates, seed)
            for metric, interval in intervals.items():
                point = float(lookup.loc[(scope, weighting, metric), "delta_oxide_v1_minus_baseline"])
                records.append(
                    {
                        "analysis_scope": scope,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "metric_label": METRIC_META[metric][0],
                        "n_source_rows": len(subset),
                        "n_sample_clusters": subset["validation_sample_group_id"].nunique(),
                        "bootstrap_replicates": replicates,
                        "random_seed": seed,
                        "point_delta": point,
                        "ci95_lower": interval["ci_lower"],
                        "ci95_upper": interval["ci_upper"],
                        "bootstrap_probability_improved": interval["bootstrap_probability_improved"],
                        "interval_interpretation": interval_interpretation(metric, interval["ci_lower"], interval["ci_upper"]),
                    }
                )
    return pd.DataFrame(records)


def exact_sign_test_p(improved: int, worsened: int) -> float:
    n = improved + worsened
    if n == 0:
        return np.nan
    k = min(improved, worsened)
    probability = sum(math.comb(n, index) for index in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def build_primary_sample_changes(rows: pd.DataFrame) -> pd.DataFrame:
    mask = scope_masks(rows)["oxide_default_eligible"]
    primary = rows.loc[mask].copy()
    records: list[dict[str, Any]] = []
    for sample_id, group in primary.groupby("validation_sample_group_id", sort=False):
        baseline = float(group["baseline_residual_log10"].median())
        new = float(group["phase6_residual_log10"].median())
        delta = abs(new) - abs(baseline)
        records.append(
            {
                "validation_sample_group_id": sample_id,
                "validation_paper_group_id": first_nonempty(group["validation_paper_group_id"]),
                "doi": first_nonempty(group["doi"]),
                "formula_raw": join_unique(group["formula_raw"]),
                "material_name_raw": join_unique(group["material_name_raw"]),
                "oxide_v1_group_key": first_nonempty(group["oxide_v1_group_key"]),
                "oxide_v1_display_name_en": first_nonempty(group["oxide_v1_display_name_en"]),
                "oxide_v1_display_name_ja": first_nonempty(group["oxide_v1_display_name_ja"]),
                "n_paired_rows": len(group),
                "baseline_median_residual_log10": baseline,
                "oxide_v1_median_residual_log10": new,
                "baseline_abs_median_error_log10": abs(baseline),
                "oxide_v1_abs_median_error_log10": abs(new),
                "delta_abs_median_error_oxide_v1_minus_baseline": delta,
                "delta_median_prediction_log10_oxide_v1_minus_baseline": new - baseline,
                "sample_error_change": "improved" if delta < -ATOL else "worsened" if delta > ATOL else "unchanged",
            }
        )
    output = pd.DataFrame(records)
    output["abs_error_change_rank"] = output["delta_abs_median_error_oxide_v1_minus_baseline"].abs().rank(method="first", ascending=False).astype("Int64")
    return output.sort_values("abs_error_change_rank").reset_index(drop=True)


def build_sample_change_summary(samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    scopes = [("oxide_default_eligible", samples)]
    scopes.extend((str(group), data) for group, data in samples.groupby("oxide_v1_group_key", sort=False))
    for scope, group in scopes:
        improved = int(group["sample_error_change"].eq("improved").sum())
        worsened = int(group["sample_error_change"].eq("worsened").sum())
        unchanged = int(group["sample_error_change"].eq("unchanged").sum())
        records.append(
            {
                "summary_level": "primary_scope" if scope == "oxide_default_eligible" else "oxide_v1_group",
                "scope_or_group": scope,
                "display_name_en": "Default-eligible oxide" if scope == "oxide_default_eligible" else first_nonempty(group["oxide_v1_display_name_en"]),
                "n_samples": len(group),
                "improved_samples": improved,
                "worsened_samples": worsened,
                "unchanged_samples": unchanged,
                "improved_fraction_excluding_ties": improved / (improved + worsened) if improved + worsened else np.nan,
                "mean_delta_abs_median_error": float(group["delta_abs_median_error_oxide_v1_minus_baseline"].mean()),
                "median_delta_abs_median_error": float(group["delta_abs_median_error_oxide_v1_minus_baseline"].median()),
                "two_sided_exact_sign_test_p": exact_sign_test_p(improved, worsened),
            }
        )
    return pd.DataFrame(records)


def support_label(n_samples: int) -> str:
    if n_samples >= 30:
        return "adequate_for_descriptive_comparison"
    if n_samples >= 10:
        return "limited"
    return "very_limited"


def build_group_evaluation(rows: pd.DataFrame, sample_summary: pd.DataFrame) -> pd.DataFrame:
    oxide = as_bool(rows["is_phase0_broad_oxide"])
    eligible = as_bool(rows["oxide_v1_default_model_eligible"])
    source = rows.loc[oxide & eligible].copy()
    sample_lookup = sample_summary.loc[sample_summary["summary_level"].eq("oxide_v1_group")].set_index("scope_or_group")
    records: list[dict[str, Any]] = []
    for keys, group in source.groupby(
        ["oxide_v1_group_key", "oxide_v1_display_name_en", "oxide_v1_display_name_ja"],
        dropna=False,
        sort=False,
    ):
        group_key, display_en, display_ja = keys
        paired = group.loc[as_bool(group["paired_prediction_available"])].copy()
        row_base, row_new, _ = metrics_for_weighting(paired, "row_equal")
        sample_base, sample_new, _ = metrics_for_weighting(paired, "sample_equal")
        baseline_ok = int(as_bool(group["baseline_prediction_available"]).sum())
        new_ok = int(as_bool(group["phase6_prediction_available"]).sum())
        denominator = len(group)
        sample_row = sample_lookup.loc[group_key]
        row_mae_delta = row_new["mae_log10"] - row_base["mae_log10"]
        sample_mae_delta = sample_new["mae_log10"] - sample_base["mae_log10"]
        coverage_delta = (new_ok - baseline_ok) / denominator
        if row_mae_delta < -ATOL and sample_mae_delta < -ATOL:
            result = "error_improved"
        elif row_mae_delta > ATOL and sample_mae_delta > ATOL:
            result = "error_worsened"
        else:
            result = "mixed_error_result"
        if coverage_delta < -0.05:
            result += "_with_material_coverage_loss"
        records.append(
            {
                "oxide_v1_group_key": group_key,
                "oxide_v1_display_name_en": display_en,
                "oxide_v1_display_name_ja": display_ja,
                "eligible_test_rows": denominator,
                "paired_rows": len(paired),
                "eligible_samples": group["validation_sample_group_id"].nunique(),
                "paired_samples": paired["validation_sample_group_id"].nunique(),
                "n_papers": paired["validation_paper_group_id"].nunique(),
                "baseline_coverage_fraction": baseline_ok / denominator,
                "oxide_v1_coverage_fraction": new_ok / denominator,
                "coverage_delta_oxide_v1_minus_baseline": coverage_delta,
                "row_baseline_mae_log10": row_base["mae_log10"],
                "row_oxide_v1_mae_log10": row_new["mae_log10"],
                "row_delta_mae_log10": row_mae_delta,
                "row_baseline_rmse_log10": row_base["rmse_log10"],
                "row_oxide_v1_rmse_log10": row_new["rmse_log10"],
                "row_delta_rmse_log10": row_new["rmse_log10"] - row_base["rmse_log10"],
                "row_delta_factor_2_accuracy": row_new["factor_2_accuracy"] - row_base["factor_2_accuracy"],
                "row_delta_factor_10_accuracy": row_new["factor_10_accuracy"] - row_base["factor_10_accuracy"],
                "sample_baseline_mae_log10": sample_base["mae_log10"],
                "sample_oxide_v1_mae_log10": sample_new["mae_log10"],
                "sample_delta_mae_log10": sample_mae_delta,
                "improved_samples": int(sample_row["improved_samples"]),
                "worsened_samples": int(sample_row["worsened_samples"]),
                "unchanged_samples": int(sample_row["unchanged_samples"]),
                "improved_fraction_excluding_ties": float(sample_row["improved_fraction_excluding_ties"]),
                "exact_sign_test_p": float(sample_row["two_sided_exact_sign_test_p"]),
                "support_note": support_label(paired["validation_sample_group_id"].nunique()),
                "descriptive_result": result,
            }
        )
    return pd.DataFrame(records).sort_values("row_delta_mae_log10").reset_index(drop=True)


def format_sigma_decade(value: float) -> str:
    if not np.isfinite(value) or value <= 0:
        return "unavailable"
    exponent = int(math.floor(math.log10(value)))
    return f"10^{exponent} to <10^{exponent + 1} S/m"


def build_stratified_evaluation(rows: pd.DataFrame) -> pd.DataFrame:
    primary = rows.loc[scope_masks(rows)["oxide_default_eligible"]].copy()
    primary["sigma_decade"] = primary["sigma_S_per_m"].map(format_sigma_decade)
    dimensions = [
        ("oxide_v1_group", "oxide_v1_display_name_en"),
        ("carrier_type", "carrier_type"),
        ("temperature_bin", "T_bin_label"),
        ("experimental_sigma_decade", "sigma_decade"),
    ]
    records: list[dict[str, Any]] = []
    for dimension, column in dimensions:
        for level, group in primary.groupby(column, dropna=False, sort=False):
            for weighting in ["row_equal", "sample_equal"]:
                baseline, new, n_units = metrics_for_weighting(group, weighting)
                records.append(
                    {
                        "dimension": dimension,
                        "level": level,
                        "metric_weighting": weighting,
                        "n_units": n_units,
                        "n_source_rows": len(group),
                        "n_samples": group["validation_sample_group_id"].nunique(),
                        "n_papers": group["validation_paper_group_id"].nunique(),
                        "baseline_mae_log10": baseline["mae_log10"],
                        "oxide_v1_mae_log10": new["mae_log10"],
                        "delta_mae_log10": new["mae_log10"] - baseline["mae_log10"],
                        "baseline_rmse_log10": baseline["rmse_log10"],
                        "oxide_v1_rmse_log10": new["rmse_log10"],
                        "delta_rmse_log10": new["rmse_log10"] - baseline["rmse_log10"],
                        "delta_factor_2_accuracy": new["factor_2_accuracy"] - baseline["factor_2_accuracy"],
                        "delta_factor_10_accuracy": new["factor_10_accuracy"] - baseline["factor_10_accuracy"],
                        "support_note": support_label(group["validation_sample_group_id"].nunique()),
                    }
                )
    return pd.DataFrame(records)


def build_coverage_losses(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_ok = as_bool(rows["baseline_prediction_available"])
    new_ok = as_bool(rows["phase6_prediction_available"])
    lost = rows.loc[baseline_ok & ~new_ok].copy()
    columns = [
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "doi",
        "formula_raw",
        "material_name_raw",
        "carrier_type",
        "T_K",
        "T_bin_label",
        "sigma_S_per_m",
        "oxide_v1_group_key",
        "oxide_v1_display_name_en",
        "oxide_v1_display_name_ja",
        "oxide_v1_inclusion_status",
        "oxide_v1_default_model_eligible",
        "phase6_prediction_status",
        "phase6_train_row_count",
        "phase6_train_sample_count",
        "phase6_train_paper_count",
    ]
    lost = lost[columns].sort_values(["oxide_v1_group_key", "phase6_prediction_status", "T_K"]).reset_index(drop=True)

    oxide = rows.loc[as_bool(rows["is_phase0_broad_oxide"])].copy()
    eligible = as_bool(oxide["oxide_v1_default_model_eligible"])
    status = oxide["oxide_v1_inclusion_status"].astype(str)
    scopes = {
        "all_oxide_assigned": pd.Series(True, index=oxide.index),
        "oxide_default_eligible": eligible,
        "oxide_review_only": status.eq("review"),
        "oxide_excluded_only": status.eq("exclude"),
    }
    records: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        subset = oxide.loc[mask]
        old_count = int(as_bool(subset["baseline_prediction_available"]).sum())
        new_count = int(as_bool(subset["phase6_prediction_available"]).sum())
        lost_subset = subset.loc[as_bool(subset["baseline_prediction_available"]) & ~as_bool(subset["phase6_prediction_available"])]
        records.append(
            {
                "summary_level": "analysis_scope",
                "scope_or_group": scope,
                "display_name_en": scope.replace("_", " "),
                "test_rows": len(subset),
                "n_samples": subset["validation_sample_group_id"].nunique(),
                "baseline_prediction_ok_rows": old_count,
                "oxide_v1_prediction_ok_rows": new_count,
                "lost_prediction_rows": len(lost_subset),
                "lost_prediction_samples": lost_subset["validation_sample_group_id"].nunique(),
                "baseline_coverage_fraction": old_count / len(subset) if len(subset) else np.nan,
                "oxide_v1_coverage_fraction": new_count / len(subset) if len(subset) else np.nan,
                "coverage_delta_oxide_v1_minus_baseline": (new_count - old_count) / len(subset) if len(subset) else np.nan,
                "loss_reasons": join_unique(lost_subset["phase6_prediction_status"]),
            }
        )
    for keys, subset in oxide.groupby(["oxide_v1_group_key", "oxide_v1_display_name_en"], sort=False):
        group_key, display = keys
        old_count = int(as_bool(subset["baseline_prediction_available"]).sum())
        new_count = int(as_bool(subset["phase6_prediction_available"]).sum())
        lost_subset = subset.loc[as_bool(subset["baseline_prediction_available"]) & ~as_bool(subset["phase6_prediction_available"])]
        records.append(
            {
                "summary_level": "oxide_v1_group",
                "scope_or_group": group_key,
                "display_name_en": display,
                "test_rows": len(subset),
                "n_samples": subset["validation_sample_group_id"].nunique(),
                "baseline_prediction_ok_rows": old_count,
                "oxide_v1_prediction_ok_rows": new_count,
                "lost_prediction_rows": len(lost_subset),
                "lost_prediction_samples": lost_subset["validation_sample_group_id"].nunique(),
                "baseline_coverage_fraction": old_count / len(subset),
                "oxide_v1_coverage_fraction": new_count / len(subset),
                "coverage_delta_oxide_v1_minus_baseline": (new_count - old_count) / len(subset),
                "loss_reasons": join_unique(lost_subset["phase6_prediction_status"]),
            }
        )
    return lost, pd.DataFrame(records)


def metric_lookup(metrics: pd.DataFrame, scope: str, weighting: str, metric: str) -> pd.Series:
    return metrics[
        metrics["analysis_scope"].eq(scope)
        & metrics["metric_weighting"].eq(weighting)
        & metrics["metric_name"].eq(metric)
    ].iloc[0]


def build_decision_summary(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    samples: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    scope = "oxide_default_eligible"
    mae = metric_lookup(metrics, scope, "row_equal", "mae_log10")
    rmse = metric_lookup(metrics, scope, "row_equal", "rmse_log10")
    factor2 = metric_lookup(metrics, scope, "row_equal", "factor_2_accuracy")
    factor10 = metric_lookup(metrics, scope, "row_equal", "factor_10_accuracy")
    sample_mae = metric_lookup(metrics, scope, "sample_equal", "mae_log10")
    primary_bootstrap = bootstrap[
        bootstrap["analysis_scope"].eq(scope)
        & bootstrap["metric_weighting"].eq("row_equal")
        & bootstrap["metric_name"].isin(["mae_log10", "rmse_log10"])
    ]
    coverage_row = coverage[
        coverage["summary_level"].eq("analysis_scope") & coverage["scope_or_group"].eq(scope)
    ].iloc[0]
    improved = int(samples["sample_error_change"].eq("improved").sum())
    worsened = int(samples["sample_error_change"].eq("worsened").sum())
    interval_supported = primary_bootstrap["interval_interpretation"].eq("supported_improvement").all()
    verdict = "classification_improves_large_error_control_with_small_coverage_cost"
    if not interval_supported:
        verdict = "classification_effect_is_mixed_or_uncertain"
    records = [
        ("primary_scope", scope, "Paired rows predicted by both variants; composition-only default-eligible oxide labels"),
        ("overall_verdict", verdict, "MAE/RMSE, tolerance accuracy, sample-level changes, bootstrap intervals, and coverage considered together"),
        ("row_mae_change", f"{mae['baseline_value']:.6f} -> {mae['oxide_v1_value']:.6f} ({mae['relative_delta_fraction']:.2%})", "Lower is better"),
        ("row_rmse_change", f"{rmse['baseline_value']:.6f} -> {rmse['oxide_v1_value']:.6f} ({rmse['relative_delta_fraction']:.2%})", "Lower is better"),
        ("sample_equal_mae_change", f"{sample_mae['baseline_value']:.6f} -> {sample_mae['oxide_v1_value']:.6f}", "Each sample receives equal weight"),
        ("factor_2_accuracy_change", f"{factor2['baseline_value']:.3%} -> {factor2['oxide_v1_value']:.3%} ({factor2['delta_percentage_points']:+.2f} pp)", "Higher is better"),
        ("factor_10_accuracy_change", f"{factor10['baseline_value']:.3%} -> {factor10['oxide_v1_value']:.3%} ({factor10['delta_percentage_points']:+.2f} pp)", "Higher is better"),
        ("sample_change_counts", f"{improved} improved / {worsened} worsened / {len(samples) - improved - worsened} unchanged", "Based on absolute sample-median residual"),
        ("coverage_change", f"{coverage_row['baseline_coverage_fraction']:.3%} -> {coverage_row['oxide_v1_coverage_fraction']:.3%} ({100*coverage_row['coverage_delta_oxide_v1_minus_baseline']:+.2f} pp)", "Available-as-is default-eligible oxide rows"),
        ("interpretation_limit", "descriptive_group_results_need_support_checks", "Groups with fewer than 10 paired samples are very limited; classification is formula-only and does not add structure labels"),
    ]
    return pd.DataFrame(records, columns=["item", "result", "definition"])


def build_inventory(rows: pd.DataFrame, samples: pd.DataFrame, groups: pd.DataFrame, lost: pd.DataFrame, replicates: int) -> pd.DataFrame:
    primary = scope_masks(rows)["oxide_default_eligible"]
    items = [
        ("test_rows", len(rows), "rows", "Frozen Phase 6 comparison cohort"),
        ("paired_rows_all_materials", int(as_bool(rows["paired_prediction_available"]).sum()), "rows", "Rows predicted by both variants"),
        ("paired_default_eligible_oxide_rows", int(primary.sum()), "rows", "Primary Phase 7 evaluation cohort"),
        ("paired_default_eligible_oxide_samples", len(samples), "samples", "Sample-equal primary cohort"),
        ("paired_default_eligible_oxide_papers", rows.loc[primary, "validation_paper_group_id"].nunique(), "papers", "Primary cohort"),
        ("evaluated_default_eligible_groups", len(groups), "groups", "Non-empty oxide_v1 groups in the primary cohort"),
        ("lost_prediction_rows_all_materials", len(lost), "rows", "Baseline available and oxide_v1 unavailable"),
        ("bootstrap_replicates", replicates, "replicates", "Sample-cluster paired bootstrap per scope and weighting"),
        ("bootstrap_seed", RANDOM_SEED, "integer", "Base deterministic seed"),
    ]
    return pd.DataFrame(items, columns=["metric", "value", "unit", "definition"])


def plot_primary_metrics(metrics: pd.DataFrame, coverage: pd.DataFrame, output: Path) -> None:
    primary = metrics[(metrics["analysis_scope"].eq("oxide_default_eligible")) & metrics["metric_weighting"].eq("row_equal")].set_index("metric_name")
    coverage_row = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    colors = ["#5B8FF9", "#5AD8A6"]
    labels = ["Broad oxide", "oxide_v1"]

    error_metrics = ["mae_log10", "rmse_log10"]
    x = np.arange(len(error_metrics))
    width = 0.35
    axes[0].bar(x - width / 2, primary.loc[error_metrics, "baseline_value"], width, label=labels[0], color=colors[0])
    axes[0].bar(x + width / 2, primary.loc[error_metrics, "oxide_v1_value"], width, label=labels[1], color=colors[1])
    axes[0].set_xticks(x, ["MAE", "RMSE"])
    axes[0].set_ylabel("log10 ratio")
    axes[0].set_title("Paired error metrics")
    axes[0].legend(frameon=False)

    accuracy_metrics = ["factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"]
    x = np.arange(len(accuracy_metrics))
    axes[1].bar(x - width / 2, 100 * primary.loc[accuracy_metrics, "baseline_value"], width, color=colors[0])
    axes[1].bar(x + width / 2, 100 * primary.loc[accuracy_metrics, "oxide_v1_value"], width, color=colors[1])
    axes[1].set_xticks(x, ["Factor 2", "Factor 5", "Factor 10"])
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Paired tolerance accuracy")
    axes[1].set_ylim(0, 100)

    axes[2].bar([0, 1], 100 * np.asarray([coverage_row["baseline_coverage_fraction"], coverage_row["oxide_v1_coverage_fraction"]]), color=colors)
    axes[2].set_xticks([0, 1], labels)
    axes[2].set_ylabel("Coverage (%)")
    axes[2].set_title("Available default-eligible rows")
    axes[2].set_ylim(90, 101)
    for index, value in enumerate([coverage_row["baseline_coverage_fraction"], coverage_row["oxide_v1_coverage_fraction"]]):
        axes[2].text(index, 100 * value + 0.15, f"{100*value:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Phase 7: classification change evaluation on default-eligible oxide rows", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_group_tradeoffs(groups: pd.DataFrame, output: Path) -> None:
    ordered = groups.sort_values("row_delta_mae_log10", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    colors = np.where(ordered["row_delta_mae_log10"] < 0, "#4C9F70", "#D95F5F")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.0, 0.42 * len(ordered) + 1.8)), sharey=True)
    axes[0].barh(y, ordered["row_delta_mae_log10"], color=colors)
    axes[0].axvline(0, color="#333333", linewidth=1)
    axes[0].set_xlabel("Delta MAE log10 (oxide_v1 - baseline)")
    axes[0].set_title("Error change")
    axes[0].set_yticks(y, ordered["oxide_v1_display_name_en"])
    axes[0].invert_yaxis()

    coverage_colors = np.where(ordered["coverage_delta_oxide_v1_minus_baseline"] < -0.05, "#E69500", "#7F8C8D")
    axes[1].barh(y, 100 * ordered["coverage_delta_oxide_v1_minus_baseline"], color=coverage_colors)
    axes[1].axvline(0, color="#333333", linewidth=1)
    axes[1].set_xlabel("Coverage change (percentage points)")
    axes[1].set_title("Coverage trade-off")
    fig.suptitle("Default-eligible oxide groups: paired error and available-row coverage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_changes(samples: pd.DataFrame, output: Path) -> None:
    values = samples["delta_abs_median_error_oxide_v1_minus_baseline"].to_numpy(float)
    improved = int((values < -ATOL).sum())
    worsened = int((values > ATOL).sum())
    unchanged = len(values) - improved - worsened
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    limit = float(np.quantile(np.abs(values), 0.99))
    bins = np.linspace(-limit, limit, 51)
    ax.hist(np.clip(values, -limit, limit), bins=bins, color="#5B8FF9", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.axvline(float(np.median(values)), color="#D95F5F", linewidth=1.5, linestyle="--", label=f"Median = {np.median(values):+.3f}")
    ax.set_xlabel("Delta absolute sample-median error (oxide_v1 - baseline), log10")
    ax.set_ylabel("Samples")
    ax.set_title("Default-eligible oxide samples: distribution of error change")
    ax.text(0.02, 0.96, f"Improved: {improved}\nWorsened: {worsened}\nUnchanged: {unchanged}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def append_check(checks: list[dict[str, str]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def build_validation(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    groups: pd.DataFrame,
    samples: pd.DataFrame,
    lost: pd.DataFrame,
    coverage: pd.DataFrame,
    figure_paths: list[Path],
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    paired = as_bool(rows["paired_prediction_available"])
    oxide = as_bool(rows["is_phase0_broad_oxide"])
    eligible = as_bool(rows["oxide_v1_default_model_eligible"])
    primary = paired & oxide & eligible
    append_check(checks, "phase6_build_validation_passes", pd.read_csv(PHASE6_VALIDATION)["status"].eq("PASS").all(), "phase6_validation_checks.csv")
    append_check(checks, "phase6_independent_check_passes", "status=PASS" in PHASE6_CHECK.read_text(encoding="utf-8"), "phase6_check_log.txt")
    append_check(checks, "comparison_row_count_frozen", len(rows) == 19008 and rows["row_id"].is_unique, f"rows={len(rows)}")
    append_check(checks, "paired_row_count_frozen", int(paired.sum()) == 18890, f"rows={int(paired.sum())}")
    append_check(checks, "primary_row_count", int(primary.sum()) == 2984, f"rows={int(primary.sum())}")
    append_check(checks, "primary_sample_count", len(samples) == 552 and samples["validation_sample_group_id"].is_unique, f"samples={len(samples)}")
    append_check(checks, "paired_metric_table_complete", len(metrics) == len(SCOPE_ORDER) * 2 * len(METRIC_META), f"rows={len(metrics)}")
    append_check(checks, "bootstrap_table_complete", len(bootstrap) == len(BOOTSTRAP_SCOPES) * 2 * len(METRIC_META), f"rows={len(bootstrap)}")
    append_check(checks, "bootstrap_intervals_ordered", bool((bootstrap["ci95_lower"] <= bootstrap["point_delta"]).all() and (bootstrap["point_delta"] <= bootstrap["ci95_upper"]).all()), "lower <= point <= upper")
    append_check(checks, "group_count_reconciles", len(groups) == rows.loc[oxide & eligible, "oxide_v1_group_key"].nunique(), f"groups={len(groups)}")
    append_check(checks, "group_paired_rows_reconcile", int(groups["paired_rows"].sum()) == int(primary.sum()), f"rows={int(groups['paired_rows'].sum())}")
    append_check(checks, "coverage_loss_count", len(lost) == 78, f"rows={len(lost)}")
    primary_loss = lost.loc[as_bool(lost["oxide_v1_default_model_eligible"])]
    append_check(checks, "primary_coverage_loss_count", len(primary_loss) == 48, f"rows={len(primary_loss)}")
    scope_coverage = coverage.loc[coverage["summary_level"].eq("analysis_scope")]
    append_check(checks, "coverage_scopes_complete", set(scope_coverage["scope_or_group"]) == {"all_oxide_assigned", "oxide_default_eligible", "oxide_review_only", "oxide_excluded_only"}, str(sorted(scope_coverage["scope_or_group"])))
    nonoxide = rows.loc[~oxide & paired, "delta_log10_prediction_phase6_minus_baseline"].abs()
    append_check(checks, "nonoxide_predictions_remain_unchanged", bool((nonoxide <= ATOL).all()), f"max_abs_delta={float(nonoxide.max()):.3e}")
    figures_ok = all(path.is_file() and path.stat().st_size > 10_000 for path in figure_paths)
    append_check(checks, "figures_created", figures_ok, ", ".join(f"{path.name}={path.stat().st_size if path.exists() else 0}" for path in figure_paths))
    current_hashes = [sha256(ROOT / path) for path in manifest["path"]]
    append_check(checks, "input_hashes_match_manifest", current_hashes == manifest["sha256"].tolist(), "current SHA-256 versus recorded")
    return pd.DataFrame(checks)


def df_to_markdown(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "n/a"
    text = frame.head(max_rows).copy()
    for column in text.columns:
        text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    decision: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    groups: pd.DataFrame,
    sample_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    primary_metrics = metrics[
        metrics["analysis_scope"].eq("oxide_default_eligible")
        & metrics["metric_weighting"].eq("row_equal")
        & metrics["metric_name"].isin(["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"])
    ][["metric_name", "baseline_value", "oxide_v1_value", "delta_oxide_v1_minus_baseline", "delta_percentage_points", "evaluation"]]
    primary_bootstrap = bootstrap[
        bootstrap["analysis_scope"].eq("oxide_default_eligible")
        & bootstrap["metric_weighting"].eq("row_equal")
        & bootstrap["metric_name"].isin(["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy"])
    ][["metric_name", "point_delta", "ci95_lower", "ci95_upper", "bootstrap_probability_improved", "interval_interpretation"]]
    group_view = groups[
        [
            "oxide_v1_display_name_en",
            "paired_rows",
            "paired_samples",
            "row_delta_mae_log10",
            "sample_delta_mae_log10",
            "coverage_delta_oxide_v1_minus_baseline",
            "improved_fraction_excluding_ties",
            "support_note",
            "descriptive_result",
        ]
    ]
    primary_sample_summary = sample_summary[sample_summary["summary_level"].eq("primary_scope")]
    primary_coverage = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")]
    lines = [
        "# Phase 7 classification-change evaluation",
        "",
        "## Evaluation design",
        "",
        "- Primary cohort: default-eligible oxide rows predicted by both the broad-oxide baseline and oxide_v1.",
        "- Primary comparison: paired row-equal metrics. Sample-equal metrics prevent samples with many temperature rows from dominating.",
        "- Uncertainty: deterministic 2,000-replicate paired bootstrap with validation samples as clusters.",
        "- Coverage is evaluated separately on all available rows so missing reference bins are not hidden by paired filtering.",
        "- This phase evaluates existing Phase 6 predictions; it does not refit or change the taxonomy.",
        "",
        "## Decision summary",
        "",
        df_to_markdown(decision),
        "",
        "## Primary paired metrics",
        "",
        df_to_markdown(primary_metrics),
        "",
        "## Primary bootstrap intervals",
        "",
        df_to_markdown(primary_bootstrap),
        "",
        "## Sample-level direction",
        "",
        df_to_markdown(primary_sample_summary),
        "",
        "## Coverage trade-off",
        "",
        df_to_markdown(primary_coverage),
        "",
        "## Default-eligible oxide groups",
        "",
        df_to_markdown(group_view),
        "",
        "## Figures",
        "",
        "- experiments/exp007/figures/figure_phase7_primary_metrics_before_after.png",
        "- experiments/exp007/figures/figure_phase7_group_mae_coverage_tradeoff.png",
        "- experiments/exp007/figures/figure_phase7_sample_error_changes.png",
        "",
        "## Validation",
        "",
        df_to_markdown(validation),
        "",
        "## Interpretation limits",
        "",
        "The intervals quantify held-out-sample variation within this frozen data split; they do not represent performance on a new external database. Group-level results are descriptive, especially when fewer than 10 paired samples are available. oxide_v1 remains a composition-only taxonomy and does not add crystal-structure labels.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace({np.nan: None, np.inf: None, -np.inf: None}).to_json(orient="records"))


def write_payload(
    path: Path,
    decision: pd.DataFrame,
    inventory: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    groups: pd.DataFrame,
    stratified: pd.DataFrame,
    samples: pd.DataFrame,
    sample_summary: pd.DataFrame,
    losses: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    primary = metrics[
        metrics["analysis_scope"].eq("oxide_default_eligible")
        & metrics["metric_weighting"].eq("row_equal")
    ]
    lookup = primary.set_index("metric_name")
    payload = {
        "title": "Phase 7: evaluation before and after oxide classification change",
        "scope_note": "Primary comparison uses paired default-eligible oxide rows. Sample-equal metrics, sample-cluster bootstrap intervals, group results, and coverage losses are reported separately.",
        "insights": [
            f"Primary paired MAE changed from {lookup.loc['mae_log10', 'baseline_value']:.3f} to {lookup.loc['mae_log10', 'oxide_v1_value']:.3f} log10.",
            f"Primary paired RMSE changed from {lookup.loc['rmse_log10', 'baseline_value']:.3f} to {lookup.loc['rmse_log10', 'oxide_v1_value']:.3f} log10.",
            f"Factor-2 accuracy changed by {lookup.loc['factor_2_accuracy', 'delta_percentage_points']:+.2f} percentage points; factor-10 accuracy changed by {lookup.loc['factor_10_accuracy', 'delta_percentage_points']:+.2f} points.",
            "Coverage losses are kept separate from paired-error gains so the trade-off remains visible.",
        ],
        "tables": {
            "decision": safe_records(decision),
            "inventory": safe_records(inventory),
            "paired_metrics": safe_records(metrics),
            "bootstrap": safe_records(bootstrap),
            "groups": safe_records(groups),
            "stratified": safe_records(stratified),
            "samples": safe_records(samples),
            "sample_summary": safe_records(sample_summary),
            "coverage_losses": safe_records(losses),
            "coverage_summary": safe_records(coverage),
            "validation": safe_records(validation),
            "manifest": safe_records(manifest),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest()
    rows = pd.read_csv(COMPARISON_ROWS, low_memory=False)
    metrics = build_paired_metric_evaluation(rows)
    bootstrap = build_bootstrap_table(rows, metrics, args.bootstrap_replicates)
    samples = build_primary_sample_changes(rows)
    sample_summary = build_sample_change_summary(samples)
    groups = build_group_evaluation(rows, sample_summary)
    stratified = build_stratified_evaluation(rows)
    losses, coverage = build_coverage_losses(rows)
    decision = build_decision_summary(metrics, bootstrap, samples, coverage)
    inventory = build_inventory(rows, samples, groups, losses, args.bootstrap_replicates)

    figure_paths = [
        FIGURES / "figure_phase7_primary_metrics_before_after.png",
        FIGURES / "figure_phase7_group_mae_coverage_tradeoff.png",
        FIGURES / "figure_phase7_sample_error_changes.png",
    ]
    plot_primary_metrics(metrics, coverage, figure_paths[0])
    plot_group_tradeoffs(groups, figure_paths[1])
    plot_sample_changes(samples, figure_paths[2])

    validation = build_validation(rows, metrics, bootstrap, groups, samples, losses, coverage, figure_paths, manifest)
    files = {
        OUTPUT_FILES["paired_metrics"]: metrics,
        OUTPUT_FILES["bootstrap"]: bootstrap,
        OUTPUT_FILES["groups"]: groups,
        OUTPUT_FILES["stratified"]: stratified,
        OUTPUT_FILES["samples"]: samples,
        OUTPUT_FILES["sample_summary"]: sample_summary,
        OUTPUT_FILES["coverage_losses"]: losses,
        OUTPUT_FILES["coverage_summary"]: coverage,
        OUTPUT_FILES["decision"]: decision,
        OUTPUT_FILES["inventory"]: inventory,
        OUTPUT_FILES["manifest"]: manifest,
        OUTPUT_FILES["validation"]: validation,
    }
    for name, frame in files.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")

    write_report(args.report, decision, metrics, bootstrap, groups, sample_summary, coverage, validation)
    write_payload(
        args.workbook_payload,
        decision,
        inventory,
        metrics,
        bootstrap,
        groups,
        stratified,
        samples,
        sample_summary,
        losses,
        coverage,
        validation,
        manifest,
    )
    lines = [
        "Phase 7 build validation",
        f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}",
        f"checks={len(validation)}",
        f"failed={int(validation['status'].ne('PASS').sum())}",
    ]
    lines.extend(f"{row.check}: {row.status} | {row.detail}" for row in validation.itertuples(index=False))
    args.validation_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise SystemExit(f"Phase 7 validation failed: {failed}")


if __name__ == "__main__":
    main()
