#!/usr/bin/env python3
"""Compare exp005 and exp006 ZT on a shared held-out sample/DOI cohort.

The evaluation grid is the existing exp006 test-prediction temperature grid.  The
reported ZT and kappa values are the Step5 full-curve observations already matched
within +/-2 K by ``compute_pf_zt_from_exp006_sigma.py``.  exp005 sigma predictions
are re-projected onto that same grid from the Step19 DOI-test tau predictions and a
Step12-compatible empirical prefactor rebuilt from the same Step5 curves.
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import compare_exp006_zt_by_material_system as material
from experiments.exp005 import build_training_dataset_step10 as step10
from experiments.exp005.normalize_units_step11 import convert_quantity


ROOT = Path(__file__).resolve().parent
DEFAULT_EXP006_ROWS = (
    ROOT
    / "data"
    / "output"
    / "exp006_pf_zt_from_sigma_prediction_full_curves"
    / "exp006_pf_zt_row_predictions.csv"
)
DEFAULT_EXP005_TAU = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step19_tau_eff_ml_model"
    / "tau_eff_ml_predictions_step19.csv"
)
DEFAULT_ELECTRICAL_CURVES = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step5_core_properties"
    / "starrydata2_step5_core_properties.xlsx"
)
DEFAULT_TAU_FIT_RESULTS = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step12_tau_fit"
    / "tau_fit_results_step12.csv"
)
DEFAULT_CLASSIFICATION = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step31_reclassified_ct_curves"
    / "material_classification_keys_step31.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "output" / "exp005_exp006_common_cohort_zt_comparison"

EXP005_MODEL = "gradient_boosting"
EXP005_SPLIT = "split_doi_group_80_20_step18"
EXP005_ROLE = "test"
TEMPERATURE_TOLERANCE_K = 2.0
TEMPERATURE_ROUND_DECIMALS = 3
TEMPERATURE_BIN_WIDTH_K = 25.0
MIN_BASELINE_SAMPLE_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp006-rows", type=Path, default=DEFAULT_EXP006_ROWS)
    parser.add_argument("--exp005-tau", type=Path, default=DEFAULT_EXP005_TAU)
    parser.add_argument(
        "--electrical-curve-source", type=Path, default=DEFAULT_ELECTRICAL_CURVES
    )
    parser.add_argument("--tau-fit-results", type=Path, default=DEFAULT_TAU_FIT_RESULTS)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-main-samples", type=int, default=5)
    parser.add_argument("--high-zt-threshold", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_doi(values: pd.Series) -> pd.Series:
    output = values.fillna("").astype(str).str.strip().str.casefold()
    output = output.str.replace(r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
    output = output.str.replace(r"^doi:\s*", "", regex=True)
    return output.str.rstrip("/.")


def load_exp006_rows(path: Path) -> pd.DataFrame:
    wanted = [
        "row_id",
        "sample_key",
        "sample_id",
        "DOI",
        "composition",
        "temperature_K",
        "sigma_obs_S_per_m",
        "sigma_pred_exp006_S_per_m",
        "seebeck_obs_V_per_K",
        "kappa_obs_W_per_mK",
        "kappa_matched_temperature_K",
        "kappa_temperature_delta_K",
        "zt_matched_temperature_K",
        "zt_temperature_delta_K",
        "ZT_obs",
        "ZT_calc_from_obs",
        "ZT_pred_exp006",
        "config_id",
        "prediction_status",
    ]
    header = set(pd.read_csv(path, nrows=0).columns)
    required = {
        "row_id",
        "sample_key",
        "DOI",
        "temperature_K",
        "seebeck_obs_V_per_K",
        "kappa_obs_W_per_mK",
        "ZT_obs",
        "ZT_pred_exp006",
    }
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing exp006 row columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=[c for c in wanted if c in header], low_memory=False)
    frame["sample_key"] = frame["sample_key"].astype(str)
    for column in [
        "temperature_K",
        "sigma_obs_S_per_m",
        "sigma_pred_exp006_S_per_m",
        "seebeck_obs_V_per_K",
        "kappa_obs_W_per_mK",
        "kappa_temperature_delta_K",
        "zt_temperature_delta_K",
        "ZT_obs",
        "ZT_calc_from_obs",
        "ZT_pred_exp006",
    ]:
        if column in frame:
            frame[column] = finite(frame[column])
    frame["exp006_DOI_normalized"] = normalize_doi(frame["DOI"])
    return frame


def load_exp005_tau(path: Path) -> pd.DataFrame:
    wanted = [
        "sample_key",
        "model_name",
        "split_name",
        "split_role",
        "DOI",
        "target_tau_eff_step18",
        "pred_tau_eff_step19",
        "target_log_tau_eff_step18",
        "pred_log_tau_eff_step19",
        "composition",
        "material_system",
        "n_or_p",
    ]
    header = set(pd.read_csv(path, nrows=0).columns)
    required = {
        "sample_key",
        "model_name",
        "split_name",
        "split_role",
        "DOI",
        "pred_tau_eff_step19",
    }
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Missing exp005 tau columns: {sorted(missing)}")
    frame = pd.read_csv(path, usecols=[c for c in wanted if c in header], low_memory=False)
    frame = frame.loc[
        frame["model_name"].eq(EXP005_MODEL)
        & frame["split_name"].eq(EXP005_SPLIT)
        & frame["split_role"].eq(EXP005_ROLE)
    ].copy()
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["pred_tau_eff_step19"] = finite(frame["pred_tau_eff_step19"])
    frame["exp005_DOI"] = frame["DOI"]
    frame["exp005_DOI_normalized"] = normalize_doi(frame["DOI"])
    frame = frame.drop(columns=["DOI"])
    frame = frame.rename(
        columns={
            "composition": "exp005_tau_composition",
            "material_system": "exp005_tau_material_system",
            "n_or_p": "exp005_tau_n_or_p",
        }
    )
    duplicated = frame["sample_key"].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f"exp005 primary DOI test has duplicate sample keys: {frame.loc[duplicated, 'sample_key'].nunique()}"
        )
    return frame


def read_electrical_curves(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        header = pd.read_excel(path, sheet_name="property_core_curves", nrows=0).columns
        wanted = [
            *step10.PROPERTY_SOURCE_COLUMNS,
            *step10.POINT_METADATA_COLUMNS,
            "x_values_json",
            "y_values_json",
            "is_electrical_conductivity_curve",
            "is_electrical_resistivity_curve",
        ]
        usecols = list(dict.fromkeys(column for column in wanted if column in header))
        return pd.read_excel(
            path,
            sheet_name="property_core_curves",
            usecols=usecols,
            engine="openpyxl",
        )
    return pd.read_csv(path, low_memory=False)


def load_tau_fit_source_map(path: Path) -> pd.DataFrame:
    wanted = [
        "sample_key",
        "material_system",
        "n_or_p",
        "fitting_source_actual_step10",
        "fit_status_step12",
    ]
    header = set(pd.read_csv(path, nrows=0).columns)
    missing = set(wanted).difference(header)
    if missing:
        raise ValueError(f"Missing Step12 tau-fit columns: {sorted(missing)}")
    mapping = pd.read_csv(path, usecols=wanted, low_memory=False)
    mapping["sample_key"] = mapping["sample_key"].astype(str)
    mapping = mapping.loc[mapping["fit_status_step12"].eq("ok")].copy()
    mapping = mapping.drop_duplicates("sample_key", keep="first")
    return mapping.rename(
        columns={
            "material_system": "material_system_step12",
            "n_or_p": "n_or_p_step12_source",
            "fitting_source_actual_step10": "fitting_source_actual_step10_step12",
        }
    )


def normalized_electrical_baseline(
    curve_path: Path, tau_fit_path: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    curves = read_electrical_curves(curve_path)
    source_map = load_tau_fit_source_map(tau_fit_path)
    if "fitting_source_preference_step8" not in curves:
        curves["fitting_source_preference_step8"] = ""
    property_name = curves.apply(step10.classify_property, axis=1)
    curves = curves.loc[
        property_name.isin([step10.SIGMA_PROPERTY, step10.RHO_PROPERTY])
    ].copy()
    curves["sample_key"] = curves["sample_key"].astype(str)
    curves = curves.loc[curves["sample_key"].isin(source_map["sample_key"])].copy()
    points, _ = step10.expand_curves_to_points(curves)
    points = step10.add_sigma_rho_flags(points)
    points["sample_key"] = points["sample_key"].astype(str)
    points = points.merge(source_map, on="sample_key", how="inner", validate="many_to_one")
    points["fitting_source_actual_step10"] = points[
        "fitting_source_actual_step10_step12"
    ]
    selected = points.loc[
        points["property_step10"].eq(points["fitting_source_actual_step10"])
        & points["usable_for_tau_fit_step10"].map(step10.normalize_bool)
    ].copy()
    selected["temperature_K"] = finite(selected["temperature_K"])
    selected["value_raw"] = finite(selected["value_raw"])
    selected = selected.loc[
        selected["temperature_K"].notna() & selected["value_raw"].gt(0)
    ].copy()

    sigma = convert_quantity(selected["value_raw"], selected["unit_y"], "sigma", True)
    rho = convert_quantity(selected["value_raw"], selected["unit_y"], "rho", True)
    sigma_from_sigma = finite(sigma["converted_value"])
    rho_si = finite(rho["converted_value"])
    sigma_from_rho = pd.Series(
        np.where(rho_si.gt(0), 1.0 / rho_si, np.nan), index=selected.index
    )
    selected["sigma_obs_S_per_m_step11"] = np.where(
        selected["property_step10"].eq(step10.SIGMA_PROPERTY),
        sigma_from_sigma,
        sigma_from_rho,
    )
    selected["temperature_K"] = selected["temperature_K"].round(
        TEMPERATURE_ROUND_DECIMALS
    )
    selected = selected.loc[finite(selected["sigma_obs_S_per_m_step11"]).gt(0)].copy()

    selected["material_system"] = selected["material_system_step12"].fillna("unknown").astype(str)
    selected["n_or_p"] = selected["n_or_p_step12_source"].fillna("unknown").astype(str)
    for column in ["material_system", "n_or_p"]:
        selected.loc[selected[column].str.strip().eq(""), column] = "unknown"

    baseline = (
        selected.groupby(
            [
                "sample_key",
                "temperature_K",
                "material_system",
                "n_or_p",
                "fitting_source_actual_step10",
            ],
            dropna=False,
            sort=False,
        )["sigma_obs_S_per_m_step11"]
        .median()
        .reset_index()
    )
    baseline["temperature_bin_K_step12"] = (
        np.round(baseline["temperature_K"] / TEMPERATURE_BIN_WIDTH_K)
        * TEMPERATURE_BIN_WIDTH_K
    )
    stats = {
        "source_electrical_curve_rows": int(len(curves)),
        "expanded_electrical_point_rows": int(len(points)),
        "selected_electrical_point_rows": int(len(selected)),
        "baseline_electrical_rows": int(len(baseline)),
        "baseline_electrical_samples": int(baseline["sample_key"].nunique()),
        "step12_fit_source_samples": int(source_map["sample_key"].nunique()),
    }
    return baseline, stats


def add_target_metadata(target: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        baseline.sort_values(["sample_key", "temperature_K"])
        .drop_duplicates("sample_key", keep="first")[[
            "sample_key",
            "material_system",
            "n_or_p",
            "fitting_source_actual_step10",
        ]]
    )
    output = target.merge(metadata, on="sample_key", how="left", validate="many_to_one")
    for column in ["material_system", "n_or_p", "fitting_source_actual_step10"]:
        output[column] = output[column].fillna("unknown").astype(str)
    output["temperature_bin_K_step12"] = (
        np.round(output["temperature_K"] / TEMPERATURE_BIN_WIDTH_K)
        * TEMPERATURE_BIN_WIDTH_K
    )
    return output


def add_targeted_prefactors(target: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    output = target.copy()
    output["prefactor_C_S_per_m_step12_rebuilt"] = np.nan
    output["prefactor_source_step12_rebuilt"] = ""
    output["prefactor_other_sample_count_step12_rebuilt"] = 0
    levels = [
        (
            [
                "material_system",
                "n_or_p",
                "fitting_source_actual_step10",
                "temperature_bin_K_step12",
            ],
            "material_system+n_or_p+fitting_source+temperature_bin",
        ),
        (
            ["material_system", "n_or_p", "temperature_bin_K_step12"],
            "material_system+n_or_p+temperature_bin",
        ),
        (["n_or_p", "temperature_bin_K_step12"], "n_or_p+temperature_bin"),
        (["temperature_bin_K_step12"], "temperature_bin"),
        ([], "global_median"),
    ]
    assigned = pd.Series(False, index=output.index)
    for group_columns, source in levels:
        remaining = output.loc[~assigned]
        if remaining.empty:
            break
        if group_columns:
            target_groups = remaining[group_columns].drop_duplicates()
            eligible = baseline.merge(target_groups, on=group_columns, how="inner")
            base_groups = eligible.groupby(group_columns, dropna=False, sort=False)
            target_grouped = remaining.groupby(group_columns, dropna=False, sort=False)
        else:
            base_groups = [((), baseline)]
            target_grouped = [((), remaining)]

        base_lookup = {key: group for key, group in base_groups}
        for key, target_group in target_grouped:
            lookup_key = key
            base_group = base_lookup.get(lookup_key)
            if base_group is None or base_group.empty:
                continue
            for sample_key, sample_targets in target_group.groupby("sample_key", sort=False):
                if not base_group["sample_key"].astype(str).eq(str(sample_key)).any():
                    continue
                others = base_group.loc[
                    ~base_group["sample_key"].astype(str).eq(str(sample_key))
                ]
                other_sample_count = int(others["sample_key"].nunique())
                if other_sample_count < MIN_BASELINE_SAMPLE_COUNT:
                    continue
                values = finite(others["sigma_obs_S_per_m_step11"])
                values = values.loc[values.gt(0)]
                if values.empty:
                    continue
                value = float(values.median())
                indices = sample_targets.index
                output.loc[indices, "prefactor_C_S_per_m_step12_rebuilt"] = value
                output.loc[indices, "prefactor_source_step12_rebuilt"] = source
                output.loc[
                    indices, "prefactor_other_sample_count_step12_rebuilt"
                ] = other_sample_count
                assigned.loc[indices] = True
    output["prefactor_status_step12_rebuilt"] = np.where(
        finite(output["prefactor_C_S_per_m_step12_rebuilt"]).gt(0), "ok", "unavailable"
    )
    return output


def build_common_rows(
    exp006_rows: pd.DataFrame,
    exp005_tau: pd.DataFrame,
    baseline: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    common = exp006_rows.merge(exp005_tau, on="sample_key", how="inner", validate="many_to_one")
    doi_equal = (
        common["exp006_DOI_normalized"].ne("")
        & common["exp005_DOI_normalized"].ne("")
        & common["exp006_DOI_normalized"].eq(common["exp005_DOI_normalized"])
    )
    mismatched_doi_rows = int((~doi_equal).sum())
    common = common.loc[doi_equal].copy()
    common = add_target_metadata(common, baseline)
    common = add_targeted_prefactors(common, baseline)

    common["sigma_pred_exp005_S_per_m"] = (
        finite(common["prefactor_C_S_per_m_step12_rebuilt"])
        * finite(common["pred_tau_eff_step19"])
    )
    seebeck = finite(common["seebeck_obs_V_per_K"])
    kappa = finite(common["kappa_obs_W_per_mK"])
    temperature = finite(common["temperature_K"])
    sigma005 = finite(common["sigma_pred_exp005_S_per_m"])
    valid005 = seebeck.notna() & sigma005.gt(0) & temperature.gt(0) & kappa.gt(0)
    common["ZT_pred_exp005"] = np.where(
        valid005, np.square(seebeck) * sigma005 * temperature / kappa, np.nan
    )

    valid_shared = (
        finite(common["ZT_obs"]).notna()
        & finite(common["ZT_pred_exp005"]).notna()
        & finite(common["ZT_pred_exp006"]).notna()
        & finite(common["kappa_temperature_delta_K"]).le(TEMPERATURE_TOLERANCE_K)
        & finite(common["zt_temperature_delta_K"]).le(TEMPERATURE_TOLERANCE_K)
    )
    comparable = common.loc[valid_shared].copy()
    comparable["comparison_scope"] = "common_sample_and_DOI_same_exp006_temperature_row"
    comparable["reference_type"] = "reported_ZT"
    comparable["temperature_tolerance_K"] = TEMPERATURE_TOLERANCE_K
    comparable["temperature_round_decimals"] = TEMPERATURE_ROUND_DECIMALS
    comparable["exp005_model"] = EXP005_MODEL
    comparable["exp005_split"] = EXP005_SPLIT
    comparable["exp005_split_role"] = EXP005_ROLE
    stats = {
        "exp006_input_rows": int(len(exp006_rows)),
        "exp006_input_samples": int(exp006_rows["sample_key"].nunique()),
        "exp005_primary_test_samples": int(exp005_tau["sample_key"].nunique()),
        "common_sample_rows_before_doi_check": int(len(common) + mismatched_doi_rows),
        "common_samples_before_doi_check": int(
            exp006_rows.loc[
                exp006_rows["sample_key"].isin(exp005_tau["sample_key"]), "sample_key"
            ].nunique()
        ),
        "doi_mismatched_or_missing_rows": mismatched_doi_rows,
        "common_sample_and_doi_rows": int(len(common)),
        "common_sample_and_doi_samples": int(common["sample_key"].nunique()),
        "common_sample_and_doi_DOIs": int(common["exp006_DOI_normalized"].nunique()),
        "prefactor_available_rows": int(common["prefactor_status_step12_rebuilt"].eq("ok").sum()),
        "reported_ZT_shared_comparable_rows": int(len(comparable)),
        "reported_ZT_shared_comparable_samples": int(comparable["sample_key"].nunique()),
        "reported_ZT_shared_comparable_DOIs": int(comparable["exp006_DOI_normalized"].nunique()),
    }
    return comparable, stats


def aggregate_samples(rows: pd.DataFrame) -> pd.DataFrame:
    carry = [
        "sample_id",
        "DOI",
        "composition",
        "exp006_DOI_normalized",
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
        "max_ZT_pred_exp005": ("ZT_pred_exp005", "max"),
        "max_ZT_pred_exp006": ("ZT_pred_exp006", "max"),
        "median_ZT_obs": ("ZT_obs", "median"),
        "median_ZT_pred_exp005": ("ZT_pred_exp005", "median"),
        "median_ZT_pred_exp006": ("ZT_pred_exp006", "median"),
    }
    for column in carry:
        if column in rows:
            aggregations[column] = (column, "first")
    output = rows.groupby("sample_key", dropna=False).agg(**aggregations).reset_index()
    for model in ["exp005", "exp006"]:
        error = output[f"max_ZT_pred_{model}"] - output["max_ZT_obs"]
        output[f"max_ZT_signed_error_{model}"] = error
        output[f"max_ZT_abs_error_{model}"] = error.abs()
    output["sample_max_abs_error_exp006_minus_exp005"] = (
        output["max_ZT_abs_error_exp006"] - output["max_ZT_abs_error_exp005"]
    )
    output["lower_sample_max_abs_error_model"] = np.select(
        [
            output["max_ZT_abs_error_exp005"] < output["max_ZT_abs_error_exp006"],
            output["max_ZT_abs_error_exp006"] < output["max_ZT_abs_error_exp005"],
        ],
        ["exp005", "exp006"],
        default="tie",
    )
    return output


def confusion(actual: pd.Series, predicted: pd.Series, threshold: float) -> dict[str, float | int]:
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
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": (tp + tn) / len(truth) if len(truth) else math.nan,
    }


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual = finite(actual)
    predicted = finite(predicted)
    valid = actual.notna() & predicted.notna()
    actual = actual.loc[valid]
    predicted = predicted.loc[valid]
    error = predicted - actual
    positive = actual.gt(0) & predicted.gt(0)
    return {
        "n": int(len(actual)),
        "MAE": float(error.abs().mean()) if len(error) else math.nan,
        "RMSE": float(np.sqrt(np.mean(np.square(error)))) if len(error) else math.nan,
        "signed_bias": float(error.mean()) if len(error) else math.nan,
        "MAPE_pct": float((error.loc[actual.ne(0)].abs() / actual.loc[actual.ne(0)].abs()).mean() * 100)
        if actual.ne(0).any()
        else math.nan,
        "log_RMSE": float(
            np.sqrt(np.mean(np.square(np.log(predicted.loc[positive]) - np.log(actual.loc[positive]))))
        )
        if positive.any()
        else math.nan,
        "spearman": float(actual.corr(predicted, method="spearman")) if len(actual) >= 2 else math.nan,
        "pearson": float(actual.corr(predicted, method="pearson")) if len(actual) >= 2 else math.nan,
    }


def build_metrics(
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    threshold: float,
    minimum_main_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_records: list[dict[str, object]] = []
    material_records: list[dict[str, object]] = []
    for model in ["exp005", "exp006"]:
        row_metrics = regression_metrics(rows["ZT_obs"], rows[f"ZT_pred_{model}"])
        sample_metrics = regression_metrics(
            samples["max_ZT_obs"], samples[f"max_ZT_pred_{model}"]
        )
        overall_records.append(
            {
                "model": model,
                "comparable_rows": int(len(rows)),
                "comparable_samples": int(len(samples)),
                "comparable_DOIs": int(samples["exp006_DOI_normalized"].nunique()),
                **{f"row_{key}": value for key, value in row_metrics.items()},
                **{f"sample_max_{key}": value for key, value in sample_metrics.items()},
                "high_ZT_threshold": threshold,
                **{
                    f"high_ZT_{key}": value
                    for key, value in confusion(
                        samples["max_ZT_obs"], samples[f"max_ZT_pred_{model}"], threshold
                    ).items()
                },
            }
        )
        for label, sample_group in samples.groupby("material_system_label", sort=False):
            row_group = rows.loc[rows["material_system_label"].eq(label)]
            row_metrics = regression_metrics(
                row_group["ZT_obs"], row_group[f"ZT_pred_{model}"]
            )
            sample_metrics = regression_metrics(
                sample_group["max_ZT_obs"], sample_group[f"max_ZT_pred_{model}"]
            )
            classified = sample_group["classification_status"].eq("classified_step31").all()
            material_records.append(
                {
                    "model": model,
                    "material_system_label": label,
                    "material_system_key": sample_group["material_system_key"].iloc[0],
                    "classification_status": sample_group["classification_status"].iloc[0],
                    "main_analysis_group": bool(
                        classified and len(sample_group) >= minimum_main_samples
                    ),
                    "comparable_rows": int(len(row_group)),
                    "comparable_samples": int(len(sample_group)),
                    "comparable_DOIs": int(sample_group["exp006_DOI_normalized"].nunique()),
                    **{f"row_{key}": value for key, value in row_metrics.items()},
                    **{f"sample_max_{key}": value for key, value in sample_metrics.items()},
                    "high_ZT_threshold": threshold,
                    **{
                        f"high_ZT_{key}": value
                        for key, value in confusion(
                            sample_group["max_ZT_obs"],
                            sample_group[f"max_ZT_pred_{model}"],
                            threshold,
                        ).items()
                    },
                }
            )
    overall = pd.DataFrame(overall_records)
    by_material = pd.DataFrame(material_records).sort_values(
        ["main_analysis_group", "comparable_samples", "model"],
        ascending=[False, False, True],
    )
    return overall, by_material


def shared_limits(actual: pd.Series, *predicted: pd.Series) -> tuple[float, float]:
    values = pd.concat([finite(actual), *[finite(x) for x in predicted]]).dropna()
    upper = max(0.1, float(values.quantile(0.995)) * 1.05)
    return -0.03 * upper, upper


def plot_overall_parity(rows: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    lower, upper = shared_limits(
        rows["ZT_obs"], rows["ZT_pred_exp005"], rows["ZT_pred_exp006"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3), sharex=True, sharey=True)
    styles = [("exp005", "#C77720"), ("exp006", "#2B7A98")]
    for ax, (model, color) in zip(axes, styles):
        metric = metrics.loc[metrics["model"].eq(model)].iloc[0]
        ax.scatter(
            rows["ZT_obs"], rows[f"ZT_pred_{model}"], s=17, alpha=0.48, color=color, edgecolors="none"
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#B7352D", linewidth=1.4)
        ax.set_title(
            f"{model}\nrows={len(rows):,}, samples={rows['sample_key'].nunique():,}, "
            f"MAE={metric['row_MAE']:.3f}"
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.22)
        ax.set_xlabel("Reported ZT")
    axes[0].set_ylabel("Predicted ZT")
    fig.suptitle("Common held-out sample/DOI cohort: reported ZT within +/-2 K", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_material_parity(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    path: Path,
    sample_level: bool = False,
) -> None:
    base = metrics.loc[
        metrics["model"].eq("exp006") & metrics["main_analysis_group"]
    ].copy()
    groups = base.sort_values("comparable_samples", ascending=False)[
        "material_system_label"
    ].tolist()
    if not groups:
        return
    if sample_level:
        plot_rows = aggregate_samples(rows)
        actual_col = "max_ZT_obs"
        pred005 = "max_ZT_pred_exp005"
        pred006 = "max_ZT_pred_exp006"
    else:
        plot_rows = rows
        actual_col = "ZT_obs"
        pred005 = "ZT_pred_exp005"
        pred006 = "ZT_pred_exp006"
    lower, upper = shared_limits(plot_rows[actual_col], plot_rows[pred005], plot_rows[pred006])
    columns = 4
    row_count = math.ceil(len(groups) / columns)
    fig, axes = plt.subplots(
        row_count, columns, figsize=(15, 3.7 * row_count), sharex=True, sharey=True
    )
    axes_array = np.atleast_1d(axes).ravel()
    for ax, label in zip(axes_array, groups):
        part = plot_rows.loc[plot_rows["material_system_label"].eq(label)]
        ax.scatter(
            part[actual_col], part[pred005], s=20, alpha=0.48, color="#C77720", edgecolors="none", label="exp005"
        )
        ax.scatter(
            part[actual_col], part[pred006], s=22, alpha=0.55, color="#2B7A98", marker="x", linewidths=0.8, label="exp006"
        )
        ax.plot([lower, upper], [lower, upper], "--", color="#B7352D", linewidth=1.2)
        m5 = metrics.loc[
            metrics["model"].eq("exp005") & metrics["material_system_label"].eq(label)
        ].iloc[0]
        m6 = metrics.loc[
            metrics["model"].eq("exp006") & metrics["material_system_label"].eq(label)
        ].iloc[0]
        prefix = "sample_max" if sample_level else "row"
        ax.set_title(
            f"{label}\nn={part['sample_key'].nunique()}, "
            f"MAE 005/006={m5[f'{prefix}_MAE']:.3f}/{m6[f'{prefix}_MAE']:.3f}",
            fontsize=8.7,
        )
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
    for ax in axes_array[len(groups) :]:
        ax.axis("off")
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", color="#C77720", label="exp005"),
            Line2D([0], [0], marker="x", linestyle="none", color="#2B7A98", label="exp006"),
            Line2D([0], [0], linestyle="--", color="#B7352D", label="ideal"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
    )
    fig.supxlabel("Reported sample-maximum ZT" if sample_level else "Reported ZT")
    fig.supylabel("Predicted sample-maximum ZT" if sample_level else "Predicted ZT")
    title = (
        "Common cohort sample-maximum ZT parity by material system"
        if sample_level
        else "Common cohort ZT parity by material system: reported ZT within +/-2 K"
    )
    fig.suptitle(title, fontsize=16, y=0.997)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(
    path: Path,
    stats: dict[str, int],
    baseline_stats: dict[str, int],
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    metric = metrics.set_index("model")
    observed_high_zt = int(samples["max_ZT_obs"].ge(1.0).sum())

    def f1_text(model: str) -> str:
        value = metric.at[model, "high_ZT_F1"]
        return f"{value:.6g}" if pd.notna(value) else "undefined"

    lines = [
        "# exp005 vs exp006 common-cohort ZT comparison",
        "",
        "## Primary comparison",
        "",
        f"- Shared comparable rows: {len(rows):,}.",
        f"- Shared samples: {len(samples):,}.",
        f"- Shared DOI: {samples['exp006_DOI_normalized'].nunique():,}.",
        "- Both models are evaluated on exactly the same exp006 temperature rows.",
        "- Reference is reported ZT from the Step5 full property curves, matched to the same sample within +/-2 K.",
        "- Both models use the same observed Seebeck coefficient, thermal conductivity, and temperature.",
        "- No interpolation or extrapolation is used.",
        "- Sample-level metrics use the maximum over the same shared comparable rows.",
        "",
        "## Models",
        "",
        f"- exp005: `{EXP005_MODEL}` from `{EXP005_SPLIT}` / `{EXP005_ROLE}`.",
        "- exp005 sigma is reconstructed as the Step12 empirical prefactor times the Step19 predicted relative tau_eff.",
        "- The Step12 prefactor hierarchy and 25 K bins are rebuilt from the same Step5 electrical curves used for this comparison.",
        "- exp006: existing sample-holdout sigma prediction; it is not retrained here.",
        "",
        "## Overall results",
        "",
        f"- exp005 row MAE: {metric.at['exp005', 'row_MAE']:.6g}; sample-max MAE: {metric.at['exp005', 'sample_max_MAE']:.6g}; ZT>=1 F1: {f1_text('exp005')}.",
        f"- exp006 row MAE: {metric.at['exp006', 'row_MAE']:.6g}; sample-max MAE: {metric.at['exp006', 'sample_max_MAE']:.6g}; ZT>=1 F1: {f1_text('exp006')}.",
        f"- Observed sample-maximum ZT>=1 count: {observed_high_zt}; F1 is undefined when this count is zero.",
        f"- Lower sample-max absolute error: exp005={int(samples['lower_sample_max_abs_error_model'].eq('exp005').sum())}, exp006={int(samples['lower_sample_max_abs_error_model'].eq('exp006').sum())}, tie={int(samples['lower_sample_max_abs_error_model'].eq('tie').sum())}.",
        "",
        "## Reconstruction and interpretation cautions",
        "",
        "- The old full Step20/Step21 prediction CSV files are not present; therefore exp005 is method-reexecuted from its saved Step19 DOI-test predictions rather than copied from the truncated Excel preview.",
        "- This is the preferred fair comparison for the current files, but it is not a byte-for-byte reproduction of the historical Step21 output because the shared Step5 source is deliberately used for both experiments.",
        "- exp005 is DOI-held-out; exp006 is sample-held-out. Restricting to shared held-out samples removes evaluation-row differences, but exp006 may still have other training samples from the same DOI.",
        "- The existing `figure_material_system_parity_reported_zt_2K.png` remains the full exp006-cohort result. Use the new common-cohort material-system figure for direct exp005-vs-exp006 claims.",
        "",
        "## Audit counts",
        "",
        *[f"- {key}: {value:,}" for key, value in {**stats, **baseline_stats}.items()],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}; pass --overwrite")

    exp006 = load_exp006_rows(args.exp006_rows.resolve())
    exp005 = load_exp005_tau(args.exp005_tau.resolve())
    baseline, baseline_stats = normalized_electrical_baseline(
        args.electrical_curve_source.resolve(), args.tau_fit_results.resolve()
    )
    rows, stats = build_common_rows(exp006, exp005, baseline)
    if rows.empty:
        raise ValueError("No shared reported-ZT rows remain after common-cohort filtering")

    classification = material.load_sample_classification(args.classification.resolve())
    rows = material.add_classification(rows, classification)
    samples = aggregate_samples(rows)
    overall, by_material = build_metrics(
        rows, samples, args.high_zt_threshold, args.minimum_main_samples
    )
    audit = pd.DataFrame(
        [(key, value) for key, value in {**stats, **baseline_stats}.items()],
        columns=["metric", "value"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".exp005_exp006_common_", dir=output.parent))
    try:
        figures = stage / "figures"
        figures.mkdir()
        write_csv(rows, stage / "common_cohort_row_comparisons.csv")
        write_csv(samples, stage / "common_cohort_sample_summary.csv")
        write_csv(overall, stage / "common_cohort_model_metrics.csv")
        write_csv(by_material, stage / "common_cohort_material_system_metrics.csv")
        write_csv(audit, stage / "common_cohort_audit.csv")
        plot_overall_parity(
            rows,
            overall,
            figures / "figure_common_cohort_overall_parity_reported_zt_2K.png",
        )
        plot_material_parity(
            rows,
            by_material,
            figures / "figure_common_cohort_material_system_parity_reported_zt_2K.png",
        )
        plot_material_parity(
            rows,
            by_material,
            figures / "figure_common_cohort_sample_max_material_system_parity_reported_zt_2K.png",
            sample_level=True,
        )
        write_notes(
            stage / "common_cohort_comparison_notes.md",
            stats,
            baseline_stats,
            rows,
            samples,
            overall,
        )
        if output.exists():
            shutil.rmtree(output)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"output_dir: {output}")
    print(f"shared comparable rows: {len(rows)}")
    print(f"shared samples: {len(samples)}")
    print(f"shared DOI: {samples['exp006_DOI_normalized'].nunique()}")
    for row in overall.itertuples(index=False):
        print(
            f"{row.model}: row_MAE={row.row_MAE:.6g}, "
            f"sample_max_MAE={row.sample_max_MAE:.6g}, F1={row.high_ZT_F1:.6g}"
        )


if __name__ == "__main__":
    main()
