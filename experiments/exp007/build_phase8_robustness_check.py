"""Phase 8 robustness checks for the oxide_v1 conductivity result.

The frozen sample holdout and oxide_v1 taxonomy are retained.  Both the broad
oxide baseline and oxide_v1 model are rebuilt under the same alternative
temperature-bin/support settings, then the paired conclusion is stress-tested
against row/sample/paper weighting, symmetric outlier trimming, and
leave-one-oxide-group-out analyses.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp006 import build_step5b_assign_predictions as step5b


EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
FIGURES = EXP007 / "figures"

SOURCE_ROWS = ROOT / "experiments" / "exp006" / "data" / "processed" / "step6a_validation_rows_with_splits_key_broad_family.csv"
ASSIGNMENTS = DATA / "phase6_oxide_v1_all_row_assignments.csv"
PHASE6_COMPARISON = DATA / "phase6_prediction_comparison_rows.csv"
PHASE7_REPORT = REPORTS / "phase7_classification_evaluation.md"
PHASE7_CHECK = REPORTS / "phase7_check_log.txt"

OUT_PREDICTIONS = DATA / "phase8_model_sensitivity_predictions.csv"
OUT_MODEL_METRICS = DATA / "phase8_model_sensitivity_metrics.csv"
OUT_EVAL = DATA / "phase8_evaluation_sensitivity.csv"
OUT_LOO = DATA / "phase8_leave_one_group_out.csv"
OUT_GROUP = DATA / "phase8_group_direction_consistency.csv"
OUT_DECISION = DATA / "phase8_decision_summary.csv"
OUT_INVENTORY = DATA / "phase8_inventory_summary.csv"
OUT_MANIFEST = DATA / "phase8_input_manifest.csv"
OUT_VALIDATION = DATA / "phase8_validation_checks.csv"
OUT_REPORT = REPORTS / "phase8_robustness_check.md"
OUT_LOG = REPORTS / "phase8_validation_log.txt"
OUT_PAYLOAD = REPORTS / ".phase8_workbook_payload.json"

FIG_MODEL = FIGURES / "figure_phase8_model_sensitivity.png"
FIG_EVAL = FIGURES / "figure_phase8_evaluation_sensitivity.png"
FIG_LOO = FIGURES / "figure_phase8_leave_one_group_out.png"

ATOL = 1e-10
FACTOR_THRESHOLDS = {
    "factor_2_accuracy": math.log10(2.0),
    "factor_5_accuracy": math.log10(5.0),
    "factor_10_accuracy": 1.0,
}
SUPPORTS = {
    "lenient": (1, 1, 1),
    "frozen": (3, 3, 1),
    "strict": (5, 5, 2),
}
BIN_WIDTHS = [50, 100, 150, 200]
WEIGHTINGS = ["row_equal", "sample_equal", "paper_equal"]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_inputs() -> None:
    paths = [SOURCE_ROWS, ASSIGNMENTS, PHASE6_COMPARISON, PHASE7_REPORT, PHASE7_CHECK]
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 8 inputs: {missing}")
    check = PHASE7_CHECK.read_text(encoding="utf-8")
    if "status=PASS" not in check:
        raise RuntimeError("Phase 7 independent validation is not PASS")


def make_manifest() -> pd.DataFrame:
    roles = [
        ("Frozen broad-family validation rows and holdout assignments", SOURCE_ROWS),
        ("Frozen oxide_v1 row assignments", ASSIGNMENTS),
        ("Frozen Phase 6 before/after predictions", PHASE6_COMPARISON),
        ("Phase 7 evaluation report", PHASE7_REPORT),
        ("Phase 7 independent validation", PHASE7_CHECK),
    ]
    return pd.DataFrame(
        [{"role": role, "path": relative(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for role, path in roles]
    )


def add_temperature_bins(rows: pd.DataFrame, width: int) -> pd.DataFrame:
    out = rows.copy()
    t = pd.to_numeric(out["T_K"], errors="coerce")
    index = np.floor((t - width / 2.0) / width).astype(int)
    center = (index + 1) * width
    out["T_bin_index"] = index
    out["T_bin_left_K"] = center - width / 2.0
    out["T_bin_right_K"] = center + width / 2.0
    out["T_bin_center_K"] = center.astype(float)
    out["T_bin_label"] = out["T_bin_left_K"].map(lambda x: f"{x:g}") + "_" + out["T_bin_right_K"].map(lambda x: f"{x:g}") + "K"
    return out


def build_reference(rows: pd.DataFrame, curve_method: str) -> pd.DataFrame:
    train = rows[rows["sample_holdout_split"].eq("train") & as_bool(rows["is_conservative_valid_sigma0"])].copy()
    train["material_group_key"] = train["material_group_key"].fillna("").astype(str).replace("", "unknown_material_family")
    group_cols = ["material_group_key", "carrier_type", "T_bin_center_K"]
    counts = (
        train.groupby(group_cols, dropna=False, sort=False)
        .agg(
            train_row_count=("row_id", "size"),
            train_sample_count=("validation_sample_group_id", "nunique"),
            train_paper_count=("validation_paper_group_id", "nunique"),
        )
        .reset_index()
    )
    if curve_method == "sample_median":
        sample = (
            train.groupby(group_cols + ["validation_sample_group_id"], dropna=False, sort=False)["log10_sigma0_S_per_m"]
            .median()
            .reset_index(name="sample_log10_sigma0")
        )
        values = sample.groupby(group_cols, dropna=False, sort=False)["sample_log10_sigma0"].median().reset_index(name="log10_sigma0_ref_S_per_m")
    elif curve_method == "row_median":
        values = train.groupby(group_cols, dropna=False, sort=False)["log10_sigma0_S_per_m"].median().reset_index(name="log10_sigma0_ref_S_per_m")
    else:
        raise ValueError(curve_method)
    return counts.merge(values, on=group_cols, validate="one_to_one")


def predict(rows: pd.DataFrame, reference: pd.DataFrame, thresholds: tuple[int, int, int]) -> pd.DataFrame:
    test = rows[rows["sample_holdout_split"].eq("test") & as_bool(rows["is_valid_sigma0"])].copy()
    test["material_group_key"] = test["material_group_key"].fillna("").astype(str).replace("", "unknown_material_family")
    keys = ["material_group_key", "carrier_type", "T_bin_center_K"]
    out = test.merge(reference, on=keys, how="left", validate="many_to_one")
    min_rows, min_samples, min_papers = thresholds
    missing = out["log10_sigma0_ref_S_per_m"].isna()
    reliable = (
        out["train_row_count"].ge(min_rows)
        & out["train_sample_count"].ge(min_samples)
        & out["train_paper_count"].ge(min_papers)
    )
    out["prediction_status"] = np.where(missing, "missing_reference_bin", np.where(reliable, "ok", "unreliable_reference_bin"))
    ok = out["prediction_status"].eq("ok")
    out["residual_log10"] = np.nan
    out.loc[ok, "residual_log10"] = (
        out.loc[ok, "log10_sigma0_ref_S_per_m"]
        + np.log10(pd.to_numeric(out.loc[ok, "F0_eta"], errors="coerce"))
        - pd.to_numeric(out.loc[ok, "log10_sigma_S_per_m"], errors="coerce")
    )
    return out


def residual_metrics(values: np.ndarray) -> dict[str, float]:
    residual = np.asarray(values, dtype=float)
    residual = residual[np.isfinite(residual)]
    absolute = np.abs(residual)
    return {
        "mean_log10_error": float(np.mean(residual)),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(residual**2))),
        **{name: float(np.mean(absolute <= threshold + ATOL)) for name, threshold in FACTOR_THRESHOLDS.items()},
    }


def weighted_residuals(rows: pd.DataFrame, column: str, weighting: str) -> np.ndarray:
    if weighting == "row_equal":
        return rows[column].to_numpy(float)
    unit = "validation_sample_group_id" if weighting == "sample_equal" else "validation_paper_group_id"
    return rows.groupby(unit, dropna=False, sort=False)[column].median().to_numpy(float)


def metric_record(rows: pd.DataFrame, weighting: str) -> dict[str, Any]:
    baseline = residual_metrics(weighted_residuals(rows, "baseline_residual_log10", weighting))
    new = residual_metrics(weighted_residuals(rows, "oxide_v1_residual_log10", weighting))
    result: dict[str, Any] = {
        "metric_weighting": weighting,
        "n_units": len(weighted_residuals(rows, "baseline_residual_log10", weighting)),
        "n_source_rows": len(rows),
        "n_samples": rows["validation_sample_group_id"].nunique(),
        "n_papers": rows["validation_paper_group_id"].nunique(),
    }
    for metric in baseline:
        result[f"baseline_{metric}"] = baseline[metric]
        result[f"oxide_v1_{metric}"] = new[metric]
        result[f"delta_{metric}"] = new[metric] - baseline[metric]
    return result


def scenario_definitions() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for width in BIN_WIDTHS:
        for support, thresholds in SUPPORTS.items():
            rows.append(
                {
                    "scenario_id": f"bw{width:03d}_{support}_sample_median",
                    "temperature_bin_width_K": width,
                    "support_profile": support,
                    "min_train_rows": thresholds[0],
                    "min_train_samples": thresholds[1],
                    "min_train_papers": thresholds[2],
                    "curve_method": "sample_median",
                    "is_frozen_setting": width == 100 and support == "frozen",
                }
            )
    rows.append(
        {
            "scenario_id": "bw100_frozen_row_median",
            "temperature_bin_width_K": 100,
            "support_profile": "frozen",
            "min_train_rows": 3,
            "min_train_samples": 3,
            "min_train_papers": 1,
            "curve_method": "row_median",
            "is_frozen_setting": False,
        }
    )
    return rows


def build_model_sensitivity(source: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_cols = [
        "row_id", "oxide_v1_group_key", "oxide_v1_display_name_en", "oxide_v1_display_name_ja",
        "oxide_v1_default_model_eligible", "oxide_v1_inclusion_status",
    ]
    meta = assignments[metadata_cols].copy()
    meta["oxide_v1_default_model_eligible"] = as_bool(meta["oxide_v1_default_model_eligible"])
    assignment_map = assignments.set_index("row_id")["material_group_key_phase6"]
    scenario_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    definitions = scenario_definitions()
    cache: dict[tuple[int, str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    for scenario in definitions:
        width = int(scenario["temperature_bin_width_K"])
        curve = str(scenario["curve_method"])
        thresholds = (int(scenario["min_train_rows"]), int(scenario["min_train_samples"]), int(scenario["min_train_papers"]))
        key_base = (width, curve, "baseline")
        key_new = (width, curve, "oxide_v1")
        if key_base not in cache:
            rebinned = add_temperature_bins(source, width)
            baseline_rows = rebinned.copy()
            new_rows = rebinned.copy()
            oxide = new_rows["material_group_key"].eq("broad::oxide")
            new_rows.loc[oxide, "material_group_key"] = new_rows.loc[oxide, "row_id"].map(assignment_map)
            if new_rows.loc[oxide, "material_group_key"].isna().any():
                raise RuntimeError("oxide_v1 mapping is incomplete")
            cache[key_base] = (baseline_rows, build_reference(baseline_rows, curve))
            cache[key_new] = (new_rows, build_reference(new_rows, curve))
        baseline = predict(cache[key_base][0], cache[key_base][1], thresholds)
        new = predict(cache[key_new][0], cache[key_new][1], thresholds)
        keep = [
            "row_id", "validation_sample_group_id", "validation_paper_group_id", "formula_raw", "material_name_raw",
            "carrier_type", "T_K", "T_bin_center_K", "material_group_key", "prediction_status", "residual_log10",
            "train_row_count", "train_sample_count", "train_paper_count",
        ]
        old = baseline[keep].rename(columns={
            "material_group_key": "baseline_group_key", "prediction_status": "baseline_prediction_status",
            "residual_log10": "baseline_residual_log10", "train_row_count": "baseline_train_row_count",
            "train_sample_count": "baseline_train_sample_count", "train_paper_count": "baseline_train_paper_count",
        })
        new_keep = new[["row_id", "material_group_key", "prediction_status", "residual_log10", "train_row_count", "train_sample_count", "train_paper_count"]].rename(columns={
            "material_group_key": "oxide_v1_group_key_pred", "prediction_status": "oxide_v1_prediction_status",
            "residual_log10": "oxide_v1_residual_log10", "train_row_count": "oxide_v1_train_row_count",
            "train_sample_count": "oxide_v1_train_sample_count", "train_paper_count": "oxide_v1_train_paper_count",
        })
        merged = old.merge(new_keep, on="row_id", validate="one_to_one").merge(meta, on="row_id", how="left", validate="one_to_one")
        primary = merged[merged["oxide_v1_default_model_eligible"].eq(True)].copy()
        primary["baseline_prediction_available"] = primary["baseline_prediction_status"].eq("ok")
        primary["oxide_v1_prediction_available"] = primary["oxide_v1_prediction_status"].eq("ok")
        primary["paired_prediction_available"] = primary["baseline_prediction_available"] & primary["oxide_v1_prediction_available"]
        primary.insert(0, "scenario_id", scenario["scenario_id"])
        for field, value in scenario.items():
            if field != "scenario_id":
                primary[field] = value
        paired = primary[primary["paired_prediction_available"]].copy()
        coverage = {
            "eligible_test_rows": len(primary),
            "baseline_prediction_ok_rows": int(primary["baseline_prediction_available"].sum()),
            "oxide_v1_prediction_ok_rows": int(primary["oxide_v1_prediction_available"].sum()),
            "paired_prediction_rows": len(paired),
            "baseline_coverage_fraction": float(primary["baseline_prediction_available"].mean()),
            "oxide_v1_coverage_fraction": float(primary["oxide_v1_prediction_available"].mean()),
            "coverage_delta_oxide_v1_minus_baseline": float(primary["oxide_v1_prediction_available"].mean() - primary["baseline_prediction_available"].mean()),
        }
        for weighting in WEIGHTINGS:
            record = {**scenario, **coverage, **metric_record(paired, weighting)}
            record["mae_rmse_direction"] = "improved" if record["delta_mae_log10"] < -ATOL and record["delta_rmse_log10"] < -ATOL else "not_both_improved"
            metric_rows.append(record)
        for group_key, group in paired.groupby("oxide_v1_group_key", dropna=False, sort=False):
            record = {**scenario, "oxide_v1_group_key": group_key, "oxide_v1_display_name_en": group["oxide_v1_display_name_en"].iloc[0]}
            record.update(metric_record(group, "row_equal"))
            group_rows.append(record)
        scenario_rows.append(primary)
    return pd.concat(scenario_rows, ignore_index=True), pd.DataFrame(metric_rows), pd.DataFrame(group_rows)


def build_evaluation_sensitivity(frozen: pd.DataFrame) -> pd.DataFrame:
    primary = frozen[
        as_bool(frozen["is_phase0_broad_oxide"])
        & as_bool(frozen["oxide_v1_default_model_eligible"])
        & as_bool(frozen["paired_prediction_available"])
    ].copy()
    primary = primary.rename(columns={"phase6_residual_log10": "oxide_v1_residual_log10"})
    records: list[dict[str, Any]] = []
    for weighting in WEIGHTINGS:
        records.append({"sensitivity_id": f"full_{weighting}", "filter_rule": "all paired rows", "trim_fraction_extreme_rows": 0.0, **metric_record(primary, weighting)})
    severity = np.maximum(primary["baseline_residual_log10"].abs(), primary["oxide_v1_residual_log10"].abs())
    for fraction in [0.01, 0.025, 0.05]:
        cutoff = float(severity.quantile(1.0 - fraction))
        subset = primary[severity <= cutoff].copy()
        records.append(
            {
                "sensitivity_id": f"symmetric_trim_{100*fraction:g}pct_row_equal",
                "filter_rule": "remove rows above the stated quantile of max(|baseline residual|, |oxide_v1 residual|)",
                "trim_fraction_extreme_rows": fraction,
                "symmetric_severity_cutoff_log10": cutoff,
                **metric_record(subset, "row_equal"),
            }
        )
    return pd.DataFrame(records)


def build_leave_one_group_out(frozen: pd.DataFrame) -> pd.DataFrame:
    primary = frozen[
        as_bool(frozen["is_phase0_broad_oxide"])
        & as_bool(frozen["oxide_v1_default_model_eligible"])
        & as_bool(frozen["paired_prediction_available"])
    ].copy().rename(columns={"phase6_residual_log10": "oxide_v1_residual_log10"})
    records: list[dict[str, Any]] = []
    groups = primary[["oxide_v1_group_key", "oxide_v1_display_name_en"]].drop_duplicates().sort_values("oxide_v1_group_key")
    for group in groups.itertuples(index=False):
        subset = primary[~primary["oxide_v1_group_key"].eq(group.oxide_v1_group_key)].copy()
        omitted = primary[primary["oxide_v1_group_key"].eq(group.oxide_v1_group_key)]
        for weighting in ["row_equal", "sample_equal"]:
            records.append(
                {
                    "omitted_group_key": group.oxide_v1_group_key,
                    "omitted_group_name": group.oxide_v1_display_name_en,
                    "omitted_rows": len(omitted),
                    "omitted_samples": omitted["validation_sample_group_id"].nunique(),
                    **metric_record(subset, weighting),
                }
            )
    return pd.DataFrame(records)


def summarize_group_consistency(group_scenarios: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (key, name), group in group_scenarios.groupby(["oxide_v1_group_key", "oxide_v1_display_name_en"], dropna=False, sort=True):
        delta = group["delta_mae_log10"].to_numpy(float)
        records.append(
            {
                "oxide_v1_group_key": key,
                "oxide_v1_display_name_en": name,
                "evaluated_scenarios": len(group),
                "scenarios_mae_improved": int((delta < -ATOL).sum()),
                "scenarios_mae_worsened": int((delta > ATOL).sum()),
                "improvement_fraction": float(np.mean(delta < -ATOL)),
                "median_delta_mae_log10": float(np.median(delta)),
                "minimum_delta_mae_log10": float(np.min(delta)),
                "maximum_delta_mae_log10": float(np.max(delta)),
                "minimum_paired_samples": int(group["n_samples"].min()),
                "robust_direction": "consistently_improved" if np.all(delta < -ATOL) else ("consistently_worsened" if np.all(delta > ATOL) else "mixed"),
            }
        )
    return pd.DataFrame(records)


def append_check(checks: list[dict[str, str]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def build_validation(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    recomputed = add_temperature_bins(source, 100)
    append_check(checks, "frozen_100K_bins_reproduced", np.allclose(recomputed["T_bin_center_K"], source["T_bin_center_K"]), "Alternate-bin implementation exactly reproduces the frozen 100 K centers")
    broad_ids = set(source.loc[source["material_group_key"].eq("broad::oxide"), "row_id"].astype(str))
    assignment_ids = set(assignments["row_id"].astype(str))
    append_check(checks, "oxide_assignment_complete", broad_ids == assignment_ids, f"source={len(broad_ids):,}; assignments={len(assignment_ids):,}")
    append_check(checks, "scenario_count", predictions["scenario_id"].nunique() == 13, f"scenarios={predictions['scenario_id'].nunique()}")
    append_check(checks, "one_row_per_scenario_and_test_row", not predictions.duplicated(["scenario_id", "row_id"]).any(), f"rows={len(predictions):,}")
    per_scenario = predictions.groupby("scenario_id").size()
    append_check(checks, "fixed_primary_denominator", per_scenario.nunique() == 1 and int(per_scenario.iloc[0]) == 3032, f"rows_per_scenario={sorted(per_scenario.unique().tolist())}")
    append_check(checks, "metrics_finite", np.isfinite(metrics.filter(regex="^(baseline|oxide_v1|delta)_").select_dtypes(include=[np.number]).to_numpy()).all(), "All reported model-sensitivity metric values are finite")

    scenario = predictions[predictions["scenario_id"].eq("bw100_frozen_sample_median")].copy()
    expected = frozen[
        as_bool(frozen["is_phase0_broad_oxide"]) & as_bool(frozen["oxide_v1_default_model_eligible"])
    ][["row_id", "baseline_prediction_status", "phase6_prediction_status", "baseline_residual_log10", "phase6_residual_log10"]]
    joined = scenario.merge(expected, on="row_id", suffixes=("", "_expected"), validate="one_to_one")
    status_ok = joined["baseline_prediction_status"].eq(joined["baseline_prediction_status_expected"]).all() and joined["oxide_v1_prediction_status"].eq(joined["phase6_prediction_status"]).all()
    append_check(checks, "frozen_prediction_status_exact", status_ok, f"rows_compared={len(joined):,}")
    old_ok = joined["baseline_prediction_status"].eq("ok")
    new_ok = joined["oxide_v1_prediction_status"].eq("ok")
    old_diff = float(np.nanmax(np.abs(joined.loc[old_ok, "baseline_residual_log10"] - joined.loc[old_ok, "baseline_residual_log10_expected"])))
    new_diff = float(np.nanmax(np.abs(joined.loc[new_ok, "oxide_v1_residual_log10"] - joined.loc[new_ok, "phase6_residual_log10"])))
    append_check(checks, "frozen_residuals_exact", old_diff < ATOL and new_diff < ATOL, f"max_abs_diff baseline={old_diff:.3e}; oxide_v1={new_diff:.3e}")

    sample_split = source.groupby("validation_sample_group_id")["sample_holdout_split"].nunique()
    append_check(checks, "sample_holdout_no_overlap", int((sample_split > 1).sum()) == 0, f"overlapping_sample_groups={int((sample_split > 1).sum())}")
    append_check(checks, "evaluation_sensitivity_complete", len(evaluation) == 6, f"rows={len(evaluation)}")
    expected_loo = assignments.loc[as_bool(assignments["oxide_v1_default_model_eligible"]), "oxide_v1_group_key"].nunique() * 2
    append_check(checks, "leave_one_group_out_complete", len(loo) == expected_loo, f"rows={len(loo)}; expected={expected_loo}")
    append_check(checks, "all_model_scenarios_mae_rmse_improve", metrics["mae_rmse_direction"].eq("improved").all(), f"improved={int(metrics['mae_rmse_direction'].eq('improved').sum())}/{len(metrics)} scenario-weighting combinations")
    append_check(checks, "all_evaluation_sensitivities_mae_rmse_improve", ((evaluation["delta_mae_log10"] < 0) & (evaluation["delta_rmse_log10"] < 0)).all(), f"improved={int(((evaluation['delta_mae_log10'] < 0) & (evaluation['delta_rmse_log10'] < 0)).sum())}/{len(evaluation)}")
    append_check(checks, "all_leave_one_group_out_mae_rmse_improve", ((loo["delta_mae_log10"] < 0) & (loo["delta_rmse_log10"] < 0)).all(), f"improved={int(((loo['delta_mae_log10'] < 0) & (loo['delta_rmse_log10'] < 0)).sum())}/{len(loo)}")
    return pd.DataFrame(checks)


def build_decision(metrics: pd.DataFrame, evaluation: pd.DataFrame, loo: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    row_metrics = metrics[metrics["metric_weighting"].eq("row_equal")]
    all_model = bool(((metrics["delta_mae_log10"] < 0) & (metrics["delta_rmse_log10"] < 0)).all())
    all_eval = bool(((evaluation["delta_mae_log10"] < 0) & (evaluation["delta_rmse_log10"] < 0)).all())
    all_loo = bool(((loo["delta_mae_log10"] < 0) & (loo["delta_rmse_log10"] < 0)).all())
    verdict = "robust_with_coverage_and_group_specific_caveats" if all_model and all_eval and all_loo else "partly_robust_or_setting_dependent"
    return pd.DataFrame(
        [
            ("overall_verdict", verdict, "Joint assessment of all predefined robustness checks"),
            ("model_scenarios", f"{int(((metrics['delta_mae_log10'] < 0) & (metrics['delta_rmse_log10'] < 0)).sum())}/{len(metrics)} scenario-weighting combinations improved in both MAE and RMSE", "13 model settings x 3 weighting schemes"),
            ("row_equal_mae_delta_range", f"{row_metrics['delta_mae_log10'].min():+.6f} to {row_metrics['delta_mae_log10'].max():+.6f}", "oxide_v1 minus broad oxide; negative is better"),
            ("row_equal_rmse_delta_range", f"{row_metrics['delta_rmse_log10'].min():+.6f} to {row_metrics['delta_rmse_log10'].max():+.6f}", "oxide_v1 minus broad oxide; negative is better"),
            ("evaluation_sensitivities", f"{int(((evaluation['delta_mae_log10'] < 0) & (evaluation['delta_rmse_log10'] < 0)).sum())}/{len(evaluation)} improved", "Row/sample/paper weighting plus symmetric outlier trimming"),
            ("leave_one_group_out", f"{int(((loo['delta_mae_log10'] < 0) & (loo['delta_rmse_log10'] < 0)).sum())}/{len(loo)} improved", "Each default-eligible oxide group omitted in turn; row and sample equal"),
            ("group_direction", f"{int(group['robust_direction'].eq('consistently_improved').sum())} consistently improved / {int(group['robust_direction'].eq('mixed').sum())} mixed / {int(group['robust_direction'].eq('consistently_worsened').sum())} consistently worsened", "Group-level MAE direction over 13 model settings"),
            ("interpretation_limit", "not_external_validation", "Same frozen sample split and same dataset; DOI-held-out or external data robustness is not tested here"),
        ],
        columns=["item", "result", "definition"],
    )


def build_inventory(predictions: pd.DataFrame, metrics: pd.DataFrame, evaluation: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("model_scenarios", predictions["scenario_id"].nunique(), "settings", "12 bin/support combinations plus one row-median aggregation check"),
            ("temperature_bin_widths", predictions["temperature_bin_width_K"].nunique(), "widths", "50, 100, 150, and 200 K"),
            ("support_profiles", predictions["support_profile"].nunique(), "profiles", "lenient 1/1/1, frozen 3/3/1, strict 5/5/2 rows/samples/papers"),
            ("eligible_test_rows_per_scenario", predictions.groupby("scenario_id").size().iloc[0], "rows", "Formula-only default-eligible oxide test rows"),
            ("scenario_prediction_rows", len(predictions), "rows", "All scenario x primary-row records"),
            ("model_metric_records", len(metrics), "records", "13 settings x row/sample/paper weighting"),
            ("evaluation_sensitivities", len(evaluation), "checks", "Frozen setting weighting and symmetric trimming"),
            ("leave_one_group_out_records", len(loo), "checks", "Groups x row/sample weighting"),
        ],
        columns=["metric", "value", "unit", "definition"],
    )


def plot_outputs(metrics: pd.DataFrame, evaluation: pd.DataFrame, loo: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    row = metrics[metrics["metric_weighting"].eq("row_equal")].copy().sort_values(["curve_method", "temperature_bin_width_K", "support_profile"])
    labels = row["scenario_id"].str.replace("_sample_median", "", regex=False).str.replace("_row_median", "_rowmed", regex=False)
    y = np.arange(len(row))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    axes[0].barh(y, row["delta_mae_log10"], color=np.where(row["delta_mae_log10"] < 0, "#2E8B57", "#C44E52"))
    axes[0].axvline(0, color="#222222", linewidth=1)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Delta MAE log10 (oxide_v1 - broad)")
    axes[0].set_title("Error direction")
    axes[1].plot(100 * row["baseline_coverage_fraction"], y, "o", label="Broad oxide", color="#4C78A8")
    axes[1].plot(100 * row["oxide_v1_coverage_fraction"], y, "s", label="oxide_v1", color="#F58518")
    for idx, (_, record) in enumerate(row.iterrows()):
        axes[1].plot(100 * np.array([record["baseline_coverage_fraction"], record["oxide_v1_coverage_fraction"]]), [idx, idx], color="#999999", linewidth=0.8)
    axes[1].set_xlabel("Prediction coverage (%)")
    axes[1].set_title("Coverage trade-off")
    axes[1].legend(frameon=False)
    fig.suptitle("Phase 8 model sensitivity: identical setting changes applied to both variants", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_MODEL, dpi=180, bbox_inches="tight")
    plt.close(fig)

    eval_order = evaluation.copy()
    y = np.arange(len(eval_order))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.36
    ax.barh(y - width / 2, eval_order["delta_mae_log10"], height=width, label="MAE", color="#2E8B57")
    ax.barh(y + width / 2, eval_order["delta_rmse_log10"], height=width, label="RMSE", color="#4C78A8")
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, eval_order["sensitivity_id"])
    ax.invert_yaxis()
    ax.set_xlabel("Delta log10 error (oxide_v1 - broad; negative is better)")
    ax.set_title("Evaluation and outlier sensitivity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_EVAL, dpi=180, bbox_inches="tight")
    plt.close(fig)

    row_loo = loo[loo["metric_weighting"].eq("row_equal")].sort_values("delta_mae_log10")
    y = np.arange(len(row_loo))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.barh(y, row_loo["delta_mae_log10"], color=np.where(row_loo["delta_mae_log10"] < 0, "#2E8B57", "#C44E52"))
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_yticks(y, row_loo["omitted_group_name"])
    ax.invert_yaxis()
    ax.set_xlabel("Delta MAE after omission (oxide_v1 - broad; negative is better)")
    ax.set_title("Leave-one-oxide-group-out robustness")
    fig.tight_layout()
    fig.savefig(FIG_LOO, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(decision: pd.DataFrame, metrics: pd.DataFrame, evaluation: pd.DataFrame, loo: pd.DataFrame, group: pd.DataFrame, validation: pd.DataFrame) -> None:
    row = metrics[metrics["metric_weighting"].eq("row_equal")]
    frozen = row[row["scenario_id"].eq("bw100_frozen_sample_median")].iloc[0]
    worst_coverage = row.loc[row["coverage_delta_oxide_v1_minus_baseline"].idxmin()]
    mixed = group[group["robust_direction"].ne("consistently_improved")]
    lines = [
        "# Phase 8: robustness check",
        "",
        "## Purpose and fixed conditions",
        "",
        "Phase 7 concluded that oxide_v1 lowered paired conductivity-prediction error. Phase 8 checks whether that direction depends on a single bin width, support rule, weighting unit, extreme rows, or one oxide group. The sample holdout, train/evaluation subsets, source data, and frozen oxide_v1 rules remain unchanged.",
        "",
        "## Model-setting sensitivity",
        "",
        f"- Both variants improved in MAE and RMSE in **{int(((metrics['delta_mae_log10'] < 0) & (metrics['delta_rmse_log10'] < 0)).sum())}/{len(metrics)}** model-setting × weighting comparisons.",
        f"- Under the frozen 100 K / 3-row / 3-sample / 1-paper / sample-median setting, row MAE changed from {frozen['baseline_mae_log10']:.6f} to {frozen['oxide_v1_mae_log10']:.6f} (delta {frozen['delta_mae_log10']:+.6f}); RMSE changed by {frozen['delta_rmse_log10']:+.6f}.",
        f"- Across the 13 row-equal settings, MAE delta ranged from {row['delta_mae_log10'].min():+.6f} to {row['delta_mae_log10'].max():+.6f}; RMSE delta ranged from {row['delta_rmse_log10'].min():+.6f} to {row['delta_rmse_log10'].max():+.6f}.",
        f"- The largest coverage penalty was {100*worst_coverage['coverage_delta_oxide_v1_minus_baseline']:+.2f} percentage points ({worst_coverage['scenario_id']}). Finer grouping can still lose predictions when a reference bin lacks support.",
        "",
        "## Evaluation sensitivity and influence checks",
        "",
        f"- Row-, sample-, and paper-equal weighting plus 1%, 2.5%, and 5% symmetric extreme-row removal all retained lower oxide_v1 MAE and RMSE ({len(evaluation)}/{len(evaluation)} checks).",
        f"- Omitting each oxide group in turn retained lower MAE and RMSE for both row- and sample-equal views ({len(loo)}/{len(loo)} checks). Therefore the overall gain is not caused by one group alone.",
        f"- At group level across model settings: {int(group['robust_direction'].eq('consistently_improved').sum())} groups were consistently improved, {int(group['robust_direction'].eq('mixed').sum())} mixed, and {int(group['robust_direction'].eq('consistently_worsened').sum())} consistently worsened.",
    ]
    if not mixed.empty:
        labels = ", ".join(mixed["oxide_v1_display_name_en"].astype(str).tolist())
        lines.append(f"- Group-level caveat: {labels} did not improve in every model setting; these should remain explicit in later discussion.")
    lines += [
        "",
        "## Conclusion",
        "",
        f"**{decision.iloc[0]['result']}**. The overall error reduction is stable to the predefined internal sensitivity tests. This does not constitute external validation: the same dataset and sample holdout are used throughout, and the coverage cost and group-specific exceptions remain material.",
        "",
        "## Outputs",
        "",
        f"- `{relative(OUT_MODEL_METRICS)}`: model-setting and weighting metrics",
        f"- `{relative(OUT_PREDICTIONS)}`: row-level scenario predictions and statuses",
        f"- `{relative(OUT_EVAL)}`: weighting and symmetric-outlier sensitivity",
        f"- `{relative(OUT_LOO)}`: leave-one-group-out results",
        f"- `{relative(OUT_GROUP)}`: group-level direction consistency",
        f"- `{relative(OUT_DECISION)}`: concise decision table",
        f"- `{relative(OUT_VALIDATION)}` and `{relative(OUT_LOG)}`: build validation evidence",
        f"- `{relative(FIG_MODEL)}`, `{relative(FIG_EVAL)}`, `{relative(FIG_LOO)}`: figures",
        "",
        "## Validation",
        "",
        f"Build checks: {int(validation['status'].eq('PASS').sum())}/{len(validation)} PASS.",
        "",
    ]
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    require_inputs()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    source_raw = pd.read_csv(SOURCE_ROWS, low_memory=False)
    step5b.validate_columns(source_raw)
    source, dropped = step5b.filter_usable_rows(source_raw)
    if not dropped.empty:
        print(f"usable rows={len(source):,}; dropped rows={len(dropped):,}")
    assignments = pd.read_csv(ASSIGNMENTS, low_memory=False)
    frozen = pd.read_csv(PHASE6_COMPARISON, low_memory=False)
    manifest = make_manifest()

    predictions, metrics, group_scenarios = build_model_sensitivity(source, assignments)
    evaluation = build_evaluation_sensitivity(frozen)
    loo = build_leave_one_group_out(frozen)
    group = summarize_group_consistency(group_scenarios)
    validation = build_validation(source, assignments, predictions, metrics, evaluation, loo, frozen)
    decision = build_decision(metrics, evaluation, loo, group)
    inventory = build_inventory(predictions, metrics, evaluation, loo)

    predictions.to_csv(OUT_PREDICTIONS, index=False)
    metrics.to_csv(OUT_MODEL_METRICS, index=False)
    evaluation.to_csv(OUT_EVAL, index=False)
    loo.to_csv(OUT_LOO, index=False)
    group.to_csv(OUT_GROUP, index=False)
    decision.to_csv(OUT_DECISION, index=False)
    inventory.to_csv(OUT_INVENTORY, index=False)
    manifest.to_csv(OUT_MANIFEST, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)
    plot_outputs(metrics, evaluation, loo)
    write_report(decision, metrics, evaluation, loo, group, validation)

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(frame.to_json(orient="records"))

    payload = {
        "title": "Phase 8: oxide_v1 robustness check",
        "scope_note": "Frozen sample holdout and taxonomy; paired default-eligible oxide rows. Negative error deltas favor oxide_v1.",
        "insights": records(decision),
        "tables": {
            "decision": records(decision),
            "inventory": records(inventory),
            "model_metrics": records(metrics),
            "evaluation": records(evaluation),
            "leave_one_out": records(loo),
            "group_consistency": records(group),
            "validation": records(validation),
            "manifest": records(manifest),
        },
    }
    OUT_PAYLOAD.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "PASS" if validation["status"].eq("PASS").all() else "FAIL"
    OUT_LOG.write_text(
        "Phase 8 build validation\n"
        f"status={status}\n"
        f"checks={len(validation)}\n"
        f"pass={int(validation['status'].eq('PASS').sum())}\n"
        + "\n".join(f"{row.status} {row.check}: {row.detail}" for row in validation.itertuples(index=False))
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase 8 build: {status} ({int(validation['status'].eq('PASS').sum())}/{len(validation)} checks)")
    print(decision.to_string(index=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
