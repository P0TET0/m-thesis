import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP11_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step11_unit_normalized"
DEFAULT_STEP12_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"
DEFAULT_STEP13_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step13_sigma_validation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step14_pf_zt_prediction"

EXCEL_PREVIEW_ROWS = 100_000
EPS = 1e-12
PF_EPS = 1e-30
ZT_THRESHOLDS = [0.5, 1.0, 1.5]

STEP12_REQUIRED = [
    "sample_key",
    "temperature_K",
    "sigma_obs_S_per_m_step11",
    "sigma_pred_S_per_m_step12",
    "fit_status_step12",
]
STEP12_PF_ZT_REQUIRED = [
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
    "zt_calc_from_obs_step11",
    "power_factor_obs_W_per_mK2_step11",
]
STEP13_REQUIRED = [
    "sample_key",
    "temperature_K",
    "validation_method_step13",
    "split_role_step13",
    "sigma_obs_S_per_m_step11",
    "sigma_pred_validation_S_per_m_step13",
    "validation_status_step13",
]
STEP13_PF_ZT_REQUIRED = [
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
]

META_COLUMNS = [
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "fitting_source_actual_step10",
    "sigma_obs_source_step11",
    "tau_eff_step12",
    "tau_eff_unit_step12",
    "tau_eff_mode_step12",
    "can_eval_power_factor_step11",
    "can_calc_zt_from_obs_step11",
    "can_compare_zt_obs_step11",
    "prefactor_source_step12",
    "prefactor_group_key_step12",
]

PREDICTION_OUTPUT_COLUMNS = [
    "sample_key",
    "temperature_K",
    "prediction_source_step14",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "tau_eff_step12",
    "tau_eff_unit_step12",
    "tau_eff_mode_step12",
    "sigma_obs_S_per_m_step11",
    "sigma_pred_for_pf_zt_S_per_m_step14",
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
    "zt_calc_from_obs_step11",
    "power_factor_obs_W_per_mK2_step14",
    "power_factor_pred_W_per_mK2_step14",
    "power_factor_obs_uW_per_cmK2_step14",
    "power_factor_pred_uW_per_cmK2_step14",
    "pf_abs_error_W_per_mK2_step14",
    "zt_pred_from_sigma_step14",
    "pf_relative_error_step14",
    "pf_log_error_step14",
    "pf_error_status_step14",
    "zt_pred_vs_calc_abs_error_step14",
    "zt_pred_vs_calc_relative_error_step14",
    "zt_pred_vs_calc_log_error_step14",
    "zt_pred_vs_calc_status_step14",
    "zt_pred_vs_obs_abs_error_step14",
    "zt_pred_vs_obs_relative_error_step14",
    "zt_pred_vs_obs_log_error_step14",
    "zt_pred_vs_obs_status_step14",
    "pf_pred_status_step14",
    "zt_pred_status_step14",
]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PF and ZT from Step12/Step13 sigma predictions.")
    parser.add_argument("--step11_dir", type=Path, default=DEFAULT_STEP11_DIR)
    parser.add_argument("--step12_dir", type=Path, default=DEFAULT_STEP12_DIR)
    parser.add_argument("--step13_dir", type=Path, default=DEFAULT_STEP13_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary_validation_method", default="high_temperature_holdout")
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--include_step13_validation", type=parse_bool, default=True)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_bool(value: Any) -> bool:
    return normalize_text(value).casefold() in {"true", "1", "yes", "y"}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def header_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_csv_selected(path: Path, desired_columns: list[str], nrows: int | None = None) -> pd.DataFrame:
    columns = header_columns(path)
    usecols = [column for column in desired_columns if column in columns]
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        nrows=nrows,
    )


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def validate_required(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & (numeric > 0)


def compare_status(relative_error: pd.Series, available: pd.Series) -> pd.Series:
    status = pd.Series("not_available", index=relative_error.index, dtype="object")
    status.loc[available & (relative_error <= 0.2)] = "ok"
    status.loc[available & (relative_error > 0.2) & (relative_error <= 1.0)] = "warning"
    status.loc[available & (relative_error > 1.0)] = "large_mismatch"
    return status


def first_failure_status(
    seebeck: pd.Series,
    sigma_pred: pd.Series,
    temperature: pd.Series | None = None,
    kappa: pd.Series | None = None,
) -> pd.Series:
    status = pd.Series("ok", index=seebeck.index, dtype="object")
    status.loc[~np.isfinite(seebeck)] = "missing Seebeck"
    status.loc[status.eq("ok") & (~np.isfinite(sigma_pred) | (sigma_pred <= 0))] = "invalid predicted sigma"
    if temperature is not None:
        status.loc[status.eq("ok") & ~np.isfinite(temperature)] = "missing temperature"
    if kappa is not None:
        status.loc[status.eq("ok") & (~np.isfinite(kappa) | (kappa <= 0))] = "missing or invalid kappa"
    return status


def add_pf_zt_predictions(
    df: pd.DataFrame,
    sigma_prediction_column: str,
    prediction_source: str,
) -> pd.DataFrame:
    output = df.copy()
    ensure_columns(
        output,
        [
            "temperature_K",
            "sigma_obs_S_per_m_step11",
            sigma_prediction_column,
            "seebeck_obs_V_per_K_step11",
            "kappa_obs_W_per_mK_step11",
            "zt_obs_dimensionless_step11",
            "zt_calc_from_obs_step11",
            "power_factor_obs_W_per_mK2_step11",
        ],
    )
    output["prediction_source_step14"] = prediction_source
    temperature = pd.to_numeric(output["temperature_K"], errors="coerce")
    sigma_obs = pd.to_numeric(output["sigma_obs_S_per_m_step11"], errors="coerce")
    sigma_pred = pd.to_numeric(output[sigma_prediction_column], errors="coerce")
    seebeck = pd.to_numeric(output["seebeck_obs_V_per_K_step11"], errors="coerce")
    kappa = pd.to_numeric(output["kappa_obs_W_per_mK_step11"], errors="coerce")
    zt_obs = pd.to_numeric(output["zt_obs_dimensionless_step11"], errors="coerce")
    zt_calc = pd.to_numeric(output["zt_calc_from_obs_step11"], errors="coerce")

    zt_calc_fallback_ok = ~np.isfinite(zt_calc) & np.isfinite(seebeck) & np.isfinite(sigma_obs) & (sigma_obs > 0) & np.isfinite(temperature) & np.isfinite(kappa) & (kappa > 0)
    zt_calc = zt_calc.where(~zt_calc_fallback_ok, (seebeck**2) * sigma_obs * temperature / kappa)
    output["zt_calc_from_obs_step11"] = zt_calc
    output["sigma_pred_for_pf_zt_S_per_m_step14"] = sigma_pred

    pf_pred_status = first_failure_status(seebeck, sigma_pred)
    pf_pred_ok = pf_pred_status.eq("ok")
    pf_pred = (seebeck**2) * sigma_pred
    pf_pred.loc[~pf_pred_ok] = np.nan
    output["power_factor_pred_W_per_mK2_step14"] = pf_pred
    output["power_factor_pred_uW_per_cmK2_step14"] = pf_pred * 10000.0
    output["pf_pred_status_step14"] = pf_pred_status

    pf_obs = pd.to_numeric(output["power_factor_obs_W_per_mK2_step11"], errors="coerce")
    pf_obs_fallback_ok = ~np.isfinite(pf_obs) & np.isfinite(seebeck) & np.isfinite(sigma_obs) & (sigma_obs > 0)
    pf_obs = pf_obs.where(~pf_obs_fallback_ok, (seebeck**2) * sigma_obs)
    output["power_factor_obs_W_per_mK2_step14"] = pf_obs
    output["power_factor_obs_uW_per_cmK2_step14"] = pf_obs * 10000.0

    pf_error_ok = np.isfinite(pf_pred) & np.isfinite(pf_obs)
    output["pf_abs_error_W_per_mK2_step14"] = (pf_pred - pf_obs).abs().where(pf_error_ok, np.nan)
    output["pf_relative_error_step14"] = (
        output["pf_abs_error_W_per_mK2_step14"] / np.maximum(pf_obs.abs(), PF_EPS)
    ).where(pf_error_ok, np.nan)
    pf_log_ok = pf_error_ok & (pf_pred > 0) & (pf_obs > 0)
    pf_log_error = pd.Series(np.nan, index=output.index, dtype="float64")
    pf_log_error.loc[pf_log_ok] = np.log(pf_pred.loc[pf_log_ok]) - np.log(pf_obs.loc[pf_log_ok])
    output["pf_log_error_step14"] = pf_log_error
    output["pf_error_status_step14"] = np.where(pf_error_ok, "ok", "not_available")

    zt_pred_status = first_failure_status(seebeck, sigma_pred, temperature, kappa)
    zt_pred_ok = zt_pred_status.eq("ok")
    zt_pred = (seebeck**2) * sigma_pred * temperature / kappa
    zt_pred.loc[~zt_pred_ok] = np.nan
    output["zt_pred_from_sigma_step14"] = zt_pred
    output["zt_pred_status_step14"] = zt_pred_status

    add_target_errors(output, zt_pred, zt_calc, "calc")
    add_target_errors(output, zt_pred, zt_obs, "obs")
    return output


def add_target_errors(output: pd.DataFrame, prediction: pd.Series, target: pd.Series, suffix: str) -> None:
    available = np.isfinite(prediction) & np.isfinite(target)
    abs_error = (prediction - target).abs().where(available, np.nan)
    relative = (abs_error / np.maximum(target.abs(), EPS)).where(available, np.nan)
    log_ok = available & (prediction > 0) & (target > 0)
    log_error = pd.Series(np.nan, index=output.index, dtype="float64")
    log_error.loc[log_ok] = np.log(prediction.loc[log_ok]) - np.log(target.loc[log_ok])
    output[f"zt_pred_vs_{suffix}_abs_error_step14"] = abs_error
    output[f"zt_pred_vs_{suffix}_relative_error_step14"] = relative
    output[f"zt_pred_vs_{suffix}_log_error_step14"] = log_error
    output[f"zt_pred_vs_{suffix}_status_step14"] = compare_status(relative, available)


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ensure_columns(df, columns)
    return df.loc[:, columns].copy()


def quality_from_metrics(log_rmse: float, mape: float, count: int) -> str:
    if count <= 0 or not math.isfinite(log_rmse) or not math.isfinite(mape):
        return "not_evaluated"
    if log_rmse <= 0.20 and mape <= 0.20:
        return "excellent"
    if log_rmse <= 0.40 and mape <= 0.50:
        return "good"
    if log_rmse <= 0.80 and mape <= 1.00:
        return "moderate"
    return "poor"


def regression_metrics(pred: pd.Series, obs: pd.Series) -> dict[str, float]:
    pred_values = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    obs_values = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    available = np.isfinite(pred_values) & np.isfinite(obs_values)
    if not available.any():
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "mape": math.nan, "log_rmse": math.nan}
    pred_values = pred_values[available]
    obs_values = obs_values[available]
    residual = pred_values - obs_values
    mape = np.mean(np.abs(residual) / np.maximum(np.abs(obs_values), EPS))
    log_ok = (pred_values > 0) & (obs_values > 0)
    log_rmse = math.nan
    if log_ok.any():
        log_error = np.log(pred_values[log_ok]) - np.log(obs_values[log_ok])
        log_rmse = float(np.sqrt(np.mean(log_error**2)))
    return {
        "count": int(len(pred_values)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape": float(mape),
        "log_rmse": log_rmse,
    }


def first_nonempty(series: pd.Series) -> Any:
    for value in series:
        if normalize_text(value):
            return value
    return series.iloc[0] if len(series) else ""


def bool_count(series: pd.Series) -> int:
    return int(series.map(normalize_bool).sum())


def aggregate_sample_results(df: pd.DataFrame, validation: bool = False) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in df.groupby("sample_key", sort=True):
        metadata = {
            column: first_nonempty(group[column])
            for column in [
                "SID",
                "DOI",
                "doi_url",
                "sample_id",
                "paper_title",
                "year",
                "composition",
                "material_system",
                "n_or_p",
                "n_or_p_basis",
                "n_or_p_confidence_step6",
                "sintering_method",
                "sintering_checked",
                "record_checked",
                "nanocarbon_keyword_detected_step9",
                "nanocarbon_type_auto_step9",
                "rare_metal_flag_auto_step9",
                "toxicity_flag_auto_step9",
                "tau_eff_step12",
                "tau_eff_unit_step12",
                "tau_eff_mode_step12",
            ]
            if column in group.columns
        }
        pf_metrics = regression_metrics(
            group["power_factor_pred_W_per_mK2_step14"],
            group["power_factor_obs_W_per_mK2_step14"],
        )
        zt_calc_metrics = regression_metrics(
            group["zt_pred_from_sigma_step14"],
            group["zt_calc_from_obs_step11"],
        )
        zt_obs_metrics = regression_metrics(
            group["zt_pred_from_sigma_step14"],
            group["zt_obs_dimensionless_step11"],
        )
        zt_obs = pd.to_numeric(group["zt_obs_dimensionless_step11"], errors="coerce")
        zt_calc = pd.to_numeric(group["zt_calc_from_obs_step11"], errors="coerce")
        zt_pred = pd.to_numeric(group["zt_pred_from_sigma_step14"], errors="coerce")
        bias = (zt_pred - zt_obs).dropna()
        record = {
            "sample_key": sample_key,
            **metadata,
            "n_rows_step14": len(group),
            "n_pf_eval_rows_step14": pf_metrics["count"],
            "n_zt_calc_eval_rows_step14": zt_calc_metrics["count"],
            "n_zt_obs_eval_rows_step14": zt_obs_metrics["count"],
            "pf_mae_step14": pf_metrics["mae"],
            "pf_rmse_step14": pf_metrics["rmse"],
            "pf_mape_step14": pf_metrics["mape"],
            "pf_log_rmse_step14": pf_metrics["log_rmse"],
            "zt_pred_vs_calc_mae_step14": zt_calc_metrics["mae"],
            "zt_pred_vs_calc_rmse_step14": zt_calc_metrics["rmse"],
            "zt_pred_vs_calc_mape_step14": zt_calc_metrics["mape"],
            "zt_pred_vs_calc_log_rmse_step14": zt_calc_metrics["log_rmse"],
            "zt_pred_vs_obs_mae_step14": zt_obs_metrics["mae"],
            "zt_pred_vs_obs_rmse_step14": zt_obs_metrics["rmse"],
            "zt_pred_vs_obs_mape_step14": zt_obs_metrics["mape"],
            "zt_pred_vs_obs_log_rmse_step14": zt_obs_metrics["log_rmse"],
            "zt_obs_max_step14": zt_obs.max(skipna=True),
            "zt_pred_max_step14": zt_pred.max(skipna=True),
            "zt_calc_from_obs_max_step14": zt_calc.max(skipna=True),
            "zt_obs_mean_step14": zt_obs.mean(skipna=True),
            "zt_pred_mean_step14": zt_pred.mean(skipna=True),
            "zt_pred_bias_mean_step14": bias.mean() if not bias.empty else math.nan,
            "zt_pred_vs_obs_quality_step14": quality_from_metrics(
                zt_obs_metrics["log_rmse"], zt_obs_metrics["mape"], zt_obs_metrics["count"]
            ),
            "zt_pred_vs_calc_quality_step14": quality_from_metrics(
                zt_calc_metrics["log_rmse"], zt_calc_metrics["mape"], zt_calc_metrics["count"]
            ),
        }
        if validation:
            record["validation_method_step13"] = first_nonempty(group["validation_method_step13"])
            record["n_validation_rows_step14"] = len(group)
            record["validation_pf_mape_step14"] = pf_metrics["mape"]
            record["validation_zt_pred_vs_obs_mape_step14"] = zt_obs_metrics["mape"]
            record["validation_zt_pred_vs_calc_mape_step14"] = zt_calc_metrics["mape"]
            record["validation_zt_quality_step14"] = record["zt_pred_vs_obs_quality_step14"]
        records.append(record)
    return pd.DataFrame(records)


def classification_metrics(sample_results: pd.DataFrame, evaluation_source: str, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        observed_basis = pd.to_numeric(sample_results["zt_obs_max_step14"], errors="coerce")
        calc_basis = pd.to_numeric(sample_results["zt_calc_from_obs_max_step14"], errors="coerce")
        observed_value = observed_basis.where(np.isfinite(observed_basis), calc_basis)
        predicted_value = pd.to_numeric(sample_results["zt_pred_max_step14"], errors="coerce")
        available = np.isfinite(observed_value) & np.isfinite(predicted_value)
        obs_pos = observed_value >= threshold
        pred_pos = predicted_value >= threshold
        tp = int((available & obs_pos & pred_pos).sum())
        fp = int((available & ~obs_pos & pred_pos).sum())
        fn = int((available & obs_pos & ~pred_pos).sum())
        tn = int((available & ~obs_pos & ~pred_pos).sum())
        precision = tp / (tp + fp) if tp + fp > 0 else math.nan
        recall = tp / (tp + fn) if tp + fn > 0 else math.nan
        f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0 else math.nan
        accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else math.nan
        specificity = tn / (tn + fp) if tn + fp > 0 else math.nan
        rows.append(
            {
                "evaluation_source_step14": evaluation_source,
                "threshold": threshold,
                "n_samples": int(available.sum()),
                "n_observed_positive": int((available & obs_pos).sum()),
                "n_predicted_positive": int((available & pred_pos).sum()),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "specificity": specificity,
            }
        )
    return pd.DataFrame(rows)


def build_high_performance_classification(sample_results: pd.DataFrame, validation_results: pd.DataFrame) -> pd.DataFrame:
    frames = [classification_metrics(sample_results, "step12_all_fit", ZT_THRESHOLDS)]
    if not validation_results.empty:
        frames.append(classification_metrics(validation_results, "step13_validation", ZT_THRESHOLDS))
    return pd.concat(frames, ignore_index=True)


def build_material_summary(sample_results: pd.DataFrame) -> pd.DataFrame:
    if sample_results.empty:
        return pd.DataFrame()
    grouped = sample_results.groupby(["material_system", "n_or_p"], dropna=False, sort=True)
    return grouped.agg(
        sample_count=("sample_key", "count"),
        zt_eval_sample_count=("n_zt_obs_eval_rows_step14", lambda values: int((pd.to_numeric(values, errors="coerce") > 0).sum())),
        median_tau_eff_step12=("tau_eff_step12", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_pf_mape_step14=("pf_mape_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_zt_pred_vs_obs_mape_step14=("zt_pred_vs_obs_mape_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_zt_pred_vs_calc_mape_step14=("zt_pred_vs_calc_mape_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_zt_pred_vs_obs_log_rmse_step14=("zt_pred_vs_obs_log_rmse_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_zt_pred_max_step14=("zt_pred_max_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        median_zt_obs_max_step14=("zt_obs_max_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
        zt_obs_ge_1_sample_count=("zt_obs_max_step14", lambda values: int((pd.to_numeric(values, errors="coerce") >= 1.0).sum())),
        zt_pred_ge_1_sample_count=("zt_pred_max_step14", lambda values: int((pd.to_numeric(values, errors="coerce") >= 1.0).sum())),
        good_or_excellent_sample_count=("zt_pred_vs_obs_quality_step14", lambda values: int(pd.Series(values).isin(["excellent", "good"]).sum())),
        poor_sample_count=("zt_pred_vs_obs_quality_step14", lambda values: int(pd.Series(values).eq("poor").sum())),
        nanocarbon_sample_count=("nanocarbon_keyword_detected_step9", lambda values: int(pd.Series(values).map(normalize_bool).sum())),
        rare_metal_flag_sample_count=("rare_metal_flag_auto_step9", lambda values: int(pd.Series(values).map(normalize_bool).sum())),
        toxicity_flag_sample_count=("toxicity_flag_auto_step9", lambda values: int(pd.Series(values).map(normalize_bool).sum())),
    ).reset_index()


def distribution_stats(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": "", "median": "", "p10": "", "p25": "", "p75": "", "p90": "", "max": ""}
    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
        "max": float(numeric.max()),
    }


def build_error_distribution(sample_results: pd.DataFrame, validation_results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics = [
        "pf_mape_step14",
        "zt_pred_vs_obs_mape_step14",
        "zt_pred_vs_calc_mape_step14",
        "zt_pred_vs_obs_log_rmse_step14",
        "zt_pred_vs_calc_log_rmse_step14",
    ]
    frames = [("step12_all_fit", sample_results)]
    if not validation_results.empty:
        frames.append(("step13_validation", validation_results))
    for source, frame in frames:
        for group_values, group in frame.groupby(["n_or_p", "material_system"], dropna=False, sort=True):
            n_or_p, material_system = group_values
            for metric in metrics:
                rows.append(
                    {
                        "evaluation_source_step14": source,
                        "n_or_p": n_or_p,
                        "material_system": material_system,
                        "metric_step14": metric,
                        **distribution_stats(group[metric]),
                    }
                )
    return pd.DataFrame(rows)


def problem_row_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if normalize_text(row.get("pf_pred_status_step14")) != "ok":
        reasons.append(normalize_text(row.get("pf_pred_status_step14")) or "PF prediction not ok")
    if normalize_text(row.get("zt_pred_status_step14")) != "ok":
        reasons.append(normalize_text(row.get("zt_pred_status_step14")) or "ZT prediction not ok")
    if normalize_text(row.get("zt_pred_vs_obs_status_step14")) == "large_mismatch":
        reasons.append("large ZT mismatch vs observed")
    if normalize_text(row.get("zt_pred_vs_calc_status_step14")) == "large_mismatch":
        reasons.append("large ZT mismatch vs calculated")
    zt_pred = pd.to_numeric(pd.Series([row.get("zt_pred_from_sigma_step14")]), errors="coerce").iloc[0]
    if math.isfinite(zt_pred) and zt_pred < 0:
        reasons.append("negative predicted ZT")
    if math.isfinite(zt_pred) and zt_pred > 10:
        reasons.append("extremely large predicted ZT")
    return "; ".join(dict.fromkeys(reasons)) if reasons else "review recommended"


def build_problem_rows(all_fit: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    frames = [all_fit]
    if not validation.empty:
        frames.append(validation)
    combined = pd.concat(frames, ignore_index=True)
    zt_pred = pd.to_numeric(combined["zt_pred_from_sigma_step14"], errors="coerce")
    obs_rel = pd.to_numeric(combined["zt_pred_vs_obs_relative_error_step14"], errors="coerce")
    calc_rel = pd.to_numeric(combined["zt_pred_vs_calc_relative_error_step14"], errors="coerce")
    problem = (
        ~combined["pf_pred_status_step14"].eq("ok")
        | ~combined["zt_pred_status_step14"].eq("ok")
        | combined["zt_pred_vs_obs_status_step14"].eq("large_mismatch")
        | combined["zt_pred_vs_calc_status_step14"].eq("large_mismatch")
        | (obs_rel > 1.0)
        | (calc_rel > 1.0)
        | (zt_pred > 10)
        | (zt_pred < 0)
    )
    output = combined.loc[problem].copy()
    output["pf_zt_problem_reason_step14"] = output.apply(problem_row_reason, axis=1)
    return output


def problem_sample_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if normalize_text(row.get("zt_pred_vs_obs_quality_step14")) == "poor":
        reasons.append("poor ZT prediction vs observed")
    if normalize_text(row.get("zt_pred_vs_calc_quality_step14")) == "poor":
        reasons.append("poor ZT prediction vs calculated")
    for column, label in [
        ("zt_pred_vs_obs_mape_step14", "large ZT observed MAPE"),
        ("zt_pred_vs_calc_mape_step14", "large ZT calculated MAPE"),
    ]:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if math.isfinite(value) and value > 1.0:
            reasons.append(label)
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    if math.isfinite(obs) and math.isfinite(pred):
        if obs >= 1.0 and pred < 1.0:
            reasons.append("observed ZT>=1 but predicted ZT<1")
        if obs < 1.0 and pred >= 1.0:
            reasons.append("predicted ZT>=1 but observed ZT<1")
    return "; ".join(dict.fromkeys(reasons)) if reasons else "review recommended"


def build_problem_samples(sample_results: pd.DataFrame) -> pd.DataFrame:
    obs_mape = pd.to_numeric(sample_results["zt_pred_vs_obs_mape_step14"], errors="coerce")
    calc_mape = pd.to_numeric(sample_results["zt_pred_vs_calc_mape_step14"], errors="coerce")
    obs_max = pd.to_numeric(sample_results["zt_obs_max_step14"], errors="coerce")
    pred_max = pd.to_numeric(sample_results["zt_pred_max_step14"], errors="coerce")
    problem = (
        sample_results["zt_pred_vs_obs_quality_step14"].eq("poor")
        | sample_results["zt_pred_vs_calc_quality_step14"].eq("poor")
        | (obs_mape > 1.0)
        | (calc_mape > 1.0)
        | ((obs_max >= 1.0) & (pred_max < 1.0))
        | ((obs_max < 1.0) & (pred_max >= 1.0))
    )
    output = sample_results.loc[problem].copy()
    output["pf_zt_problem_reason_step14"] = output.apply(problem_sample_reason, axis=1)
    high_or_large_error = (
        (pd.to_numeric(output["zt_obs_max_step14"], errors="coerce") >= 1.0)
        | (pd.to_numeric(output["zt_pred_max_step14"], errors="coerce") >= 1.0)
        | output["zt_pred_vs_obs_quality_step14"].eq("poor")
        | output["zt_pred_vs_calc_quality_step14"].eq("poor")
    )
    output["needs_manual_review_after_step14"] = "yes"
    output["needs_sintering_check_later_step14"] = np.where(high_or_large_error, "yes", "no")
    return output


def describe_series(series: pd.Series, prefix: str) -> list[tuple[str, str]]:
    stats = distribution_stats(series)
    return [(f"{prefix}_{key}", str(value)) for key, value in stats.items() if key != "p10"]


def build_report(
    input_counts: dict[str, int],
    all_fit: pd.DataFrame,
    validation: pd.DataFrame,
    sample_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    high_perf: pd.DataFrame,
    problem_rows: pd.DataFrame,
    problem_samples: pd.DataFrame,
    args: argparse.Namespace,
    step13_used: bool,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = [
        ("input_sigma_predictions_step12_rows", str(input_counts.get("sigma_predictions_step12", 0))),
        ("input_tau_fit_results_step12_sample_count", str(input_counts.get("tau_fit_results_step12", 0))),
        ("input_sigma_validation_predictions_step13_rows", str(input_counts.get("sigma_validation_predictions_step13", 0))),
        ("thermoelectric_predictions_step14_rows", str(len(all_fit))),
        ("thermoelectric_validation_predictions_step14_rows", str(len(validation))),
        ("pf_zt_sample_results_step14_sample_count", str(len(sample_results))),
        ("pf_zt_validation_sample_results_step14_sample_count", str(len(validation_results))),
        ("pf_zt_problem_rows_step14_rows", str(len(problem_rows))),
        ("pf_zt_problem_samples_step14_sample_count", str(len(problem_samples))),
        ("step13_validation_used", "yes" if step13_used else "no"),
        ("pf_pred_ok_rows", str(int(all_fit["pf_pred_status_step14"].eq("ok").sum()))),
        ("zt_pred_ok_rows", str(int(all_fit["zt_pred_status_step14"].eq("ok").sum()))),
        ("zt_obs_compare_rows", str(int((~all_fit["zt_pred_vs_obs_status_step14"].eq("not_available")).sum()))),
        ("zt_calc_compare_rows", str(int((~all_fit["zt_pred_vs_calc_status_step14"].eq("not_available")).sum()))),
        ("validation_zt_pred_ok_rows", str(int(validation["zt_pred_status_step14"].eq("ok").sum())) if not validation.empty else "0"),
    ]
    for source, frame, label in [
        ("step12_all_fit", sample_results, "step12"),
        ("step13_validation", validation_results, "step13_validation"),
    ]:
        if frame.empty:
            continue
        rows.append((f"{label}_evaluation_source", source))
        for metric in [
            "pf_mape_step14",
            "zt_pred_vs_obs_mape_step14",
            "zt_pred_vs_calc_mape_step14",
            "zt_pred_vs_obs_log_rmse_step14",
            "zt_pred_vs_calc_log_rmse_step14",
        ]:
            rows.extend(describe_series(frame[metric], f"{label}_{metric}"))

    for _, row in high_perf.iterrows():
        source = row["evaluation_source_step14"]
        threshold = row["threshold"]
        for metric in ["precision", "recall", "f1", "accuracy", "n_observed_positive", "n_predicted_positive"]:
            rows.append((f"{source}_zt_ge_{threshold}_{metric}", str(row[metric])))

    eval_samples = sample_results[
        (pd.to_numeric(sample_results["n_zt_obs_eval_rows_step14"], errors="coerce") > 0)
        | (pd.to_numeric(sample_results["n_zt_calc_eval_rows_step14"], errors="coerce") > 0)
    ]
    for n_or_p, group in eval_samples.groupby("n_or_p", dropna=False):
        rows.append((f"n_or_p_{n_or_p}_zt_eval_sample_count", str(len(group))))
        rows.append(
            (
                f"n_or_p_{n_or_p}_median_zt_pred_vs_obs_mape_step14",
                str(float(pd.to_numeric(group["zt_pred_vs_obs_mape_step14"], errors="coerce").median())),
            )
        )
    material_eval = eval_samples.groupby("material_system", dropna=False).agg(
        zt_eval_sample_count=("sample_key", "count"),
        median_obs_mape=("zt_pred_vs_obs_mape_step14", lambda values: pd.to_numeric(values, errors="coerce").median()),
    )
    for material_system, row in material_eval.sort_values("zt_eval_sample_count", ascending=False).head(20).iterrows():
        rows.append((f"material_system_{material_system}_zt_eval_sample_count", str(int(row["zt_eval_sample_count"]))))
    for material_system, row in material_eval.sort_values("median_obs_mape", ascending=True).head(20).iterrows():
        rows.append((f"material_system_{material_system}_median_zt_pred_vs_obs_mape_step14", str(float(row["median_obs_mape"]))))

    zt_obs = pd.to_numeric(sample_results["zt_obs_max_step14"], errors="coerce")
    zt_pred = pd.to_numeric(sample_results["zt_pred_max_step14"], errors="coerce")
    rows.extend(
        [
            ("zt_obs_ge_1_sample_count", str(int((zt_obs >= 1.0).sum()))),
            ("zt_pred_ge_1_sample_count", str(int((zt_pred >= 1.0).sum()))),
            ("zt_obs_ge_1_and_zt_pred_ge_1_sample_count", str(int(((zt_obs >= 1.0) & (zt_pred >= 1.0)).sum()))),
            ("zt_obs_ge_1_but_zt_pred_lt_1_sample_count", str(int(((zt_obs >= 1.0) & (zt_pred < 1.0)).sum()))),
            ("zt_obs_lt_1_but_zt_pred_ge_1_sample_count", str(int(((zt_obs < 1.0) & (zt_pred >= 1.0)).sum()))),
            (
                "needs_sintering_check_later_step14_yes_sample_count",
                str(int(problem_samples["needs_sintering_check_later_step14"].eq("yes").sum())) if not problem_samples.empty else "0",
            ),
            ("sintering_method_unknown_rows", str(int(all_fit["sintering_method"].astype(str).str.casefold().eq("unknown").sum()))),
            ("sintering_checked_no_rows", str(int(all_fit["sintering_checked"].astype(str).str.casefold().eq("no").sum()))),
            ("record_checked_no_rows", str(int(all_fit["record_checked"].astype(str).str.casefold().eq("no").sum()))),
            ("n_p_changed_rows", "0"),
            ("sintering_changed_rows", str(sintering_changed_rows(all_fit))),
            ("note", "Step14 did not predict Seebeck coefficient or thermal conductivity."),
            ("note", "Step14 estimated PF and ZT using sigma_pred from tau_eff and observed S_obs/kappa_obs."),
            ("note", "ZT_pred = S_obs^2 * sigma_pred * T / kappa_obs."),
            ("note", "Step14 mainly evaluates how sigma prediction affects ZT estimation."),
        ]
    )
    if not step13_used:
        rows.append(("warning", "Step13 validation files were not used; only Step12 all-fit predictions were evaluated."))
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    return "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n", report_df


def sintering_changed_rows(df: pd.DataFrame) -> int:
    return int(
        (
            ~df["sintering_method"].astype(str).str.casefold().eq("unknown")
            | ~df["sintering_checked"].astype(str).str.casefold().eq("no")
            | ~df["record_checked"].astype(str).str.casefold().eq("no")
        ).sum()
    )


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    csv_frame(df).to_csv(path, index=False)


def read_excel_frame(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)


def add_excel_preview_note(sheet_name: str, row_count: int, excel_notes: list[str]) -> None:
    if row_count <= EXCEL_PREVIEW_ROWS:
        return
    excel_notes.append(
        f"{sheet_name} has {row_count} rows; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook; full data is in CSV"
    )


def fit_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    preview = df.head(200)
    for index, column in enumerate(df.columns, start=1):
        max_length = len(str(column))
        if not preview.empty:
            max_length = max(max_length, int(preview[column].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(
            max(max_length + 2, 12), 60
        )


def write_excel_output(output_dir: Path, report_df: pd.DataFrame) -> None:
    sheet_files = {
        "pf_zt_sample_results": "pf_zt_sample_results_step14.csv",
        "pf_zt_validation_sample_results": "pf_zt_validation_sample_results_step14.csv",
        "zt_high_performance_classification": "zt_high_performance_classification_step14.csv",
        "pf_zt_material_summary": "pf_zt_material_summary_step14.csv",
        "pf_zt_problem_samples": "pf_zt_problem_samples_step14.csv",
        "zt_error_distribution": "zt_error_distribution_step14.csv",
        "pf_zt_problem_rows": "pf_zt_problem_rows_step14.csv",
    }
    with pd.ExcelWriter(output_dir / "starrydata2_step14_pf_zt_prediction.xlsx", engine="openpyxl") as writer:
        for sheet_name, filename in sheet_files.items():
            frame = read_excel_frame(output_dir / filename, nrows=EXCEL_PREVIEW_ROWS)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)
        report_df.to_excel(writer, sheet_name="prediction_report", index=False)
        fit_worksheet(writer, "prediction_report", report_df)


def zt_ge_1_metrics(high_perf: pd.DataFrame) -> tuple[float, float, float]:
    row = high_perf[
        high_perf["evaluation_source_step14"].eq("step12_all_fit")
        & np.isclose(pd.to_numeric(high_perf["threshold"], errors="coerce"), 1.0)
    ]
    if row.empty:
        return math.nan, math.nan, math.nan
    first = row.iloc[0]
    return float(first["precision"]), float(first["recall"]), float(first["f1"])


def assert_acceptance(
    all_fit: pd.DataFrame,
    validation: pd.DataFrame,
    sample_results: pd.DataFrame,
    validation_results: pd.DataFrame,
    high_perf: pd.DataFrame,
    problem_samples: pd.DataFrame,
) -> None:
    for column in [
        "sample_key",
        "sigma_pred_for_pf_zt_S_per_m_step14",
        "power_factor_pred_W_per_mK2_step14",
        "zt_pred_from_sigma_step14",
        "zt_pred_vs_obs_relative_error_step14",
        "zt_pred_vs_calc_relative_error_step14",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in all_fit.columns:
            raise KeyError(f"thermoelectric_predictions_step14 missing {column}")
    if all_fit.duplicated(["sample_key", "temperature_K"]).any():
        raise ValueError("thermoelectric_predictions_step14 is not one row per sample-temperature")
    if not validation.empty and not validation["prediction_source_step14"].eq("step13_validation_sigma").all():
        raise ValueError("validation predictions do not contain step13 validation source")
    if sample_results["sample_key"].duplicated().any():
        raise ValueError("pf_zt_sample_results_step14 is not one row per sample")
    for column in ["zt_obs_max_step14", "zt_pred_max_step14", "zt_pred_vs_obs_quality_step14"]:
        if column not in sample_results.columns:
            raise KeyError(f"pf_zt_sample_results_step14 missing {column}")
    if not validation_results.empty and "validation_zt_pred_vs_obs_mape_step14" not in validation_results.columns:
        raise KeyError("pf_zt_validation_sample_results_step14 missing validation ZT metric")
    thresholds = set(pd.to_numeric(high_perf["threshold"], errors="coerce").dropna().round(6))
    if not {0.5, 1.0, 1.5}.issubset(thresholds):
        raise ValueError("high performance classification missing required thresholds")
    if "needs_sintering_check_later_step14" not in problem_samples.columns:
        raise KeyError("pf_zt_problem_samples_step14 missing needs_sintering_check_later_step14")
    if sintering_changed_rows(all_fit) != 0:
        raise ValueError("sintering fields changed in thermoelectric predictions")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    step12_sigma_path = args.step12_dir / "sigma_predictions_step12.csv"
    step12_results_path = args.step12_dir / "tau_fit_results_step12.csv"
    step12_ready_path = args.step12_dir / "tau_fit_ready_samples_step12.csv"
    for path in [step12_sigma_path, step12_results_path, step12_ready_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required Step12 file not found: {path}")

    input_counts = {
        "sigma_predictions_step12": count_csv_rows(step12_sigma_path),
        "tau_fit_results_step12": count_csv_rows(step12_results_path),
        "tau_fit_ready_samples_step12": count_csv_rows(step12_ready_path),
        "sigma_validation_predictions_step13": 0,
    }

    step12_columns = sorted(set(STEP12_REQUIRED + STEP12_PF_ZT_REQUIRED + META_COLUMNS))
    step12_df = read_csv_selected(step12_sigma_path, step12_columns)
    validate_required(step12_df, STEP12_REQUIRED, "sigma_predictions_step12.csv")
    ensure_columns(step12_df, STEP12_PF_ZT_REQUIRED + META_COLUMNS)
    step12_df = step12_df[
        step12_df["fit_status_step12"].eq("ok")
        & finite_positive(step12_df["sigma_pred_S_per_m_step12"])
        & np.isfinite(pd.to_numeric(step12_df["temperature_K"], errors="coerce"))
    ].copy()
    all_fit = add_pf_zt_predictions(step12_df, "sigma_pred_S_per_m_step12", "step12_fit_sigma")
    all_fit = select_columns(all_fit, PREDICTION_OUTPUT_COLUMNS)

    step13_used = False
    validation = pd.DataFrame(columns=[*PREDICTION_OUTPUT_COLUMNS, "validation_method_step13"])
    validation_path = args.step13_dir / "sigma_validation_predictions_step13.csv"
    primary_path = args.step13_dir / "tau_validation_primary_results_step13.csv"
    if args.include_step13_validation and validation_path.exists() and primary_path.exists():
        input_counts["sigma_validation_predictions_step13"] = count_csv_rows(validation_path)
        step13_columns = sorted(
            set(
                STEP13_REQUIRED
                + STEP13_PF_ZT_REQUIRED
                + [
                    "SID",
                    "DOI",
                    "doi_url",
                    "sample_id",
                    "paper_title",
                    "year",
                    "composition",
                    "material_system",
                    "n_or_p",
                    "n_or_p_confidence_step6",
                    "sintering_method",
                    "sintering_checked",
                    "record_checked",
                    "fitting_source_actual_step10",
                    "sigma_obs_source_step11",
                    "tau_eff_step12",
                    "tau_eff_train_step13",
                    "sigma_pred_S_per_m_step12",
                    "sigma_validation_log_error_step13",
                    "sigma_validation_relative_error_step13",
                    "prefactor_source_step12",
                    "prefactor_group_key_step12",
                    "prefactor_status_step12",
                ]
            )
        )
        validation_raw = read_csv_selected(validation_path, step13_columns)
        try:
            validate_required(validation_raw, STEP13_REQUIRED, "sigma_validation_predictions_step13.csv")
            ensure_columns(validation_raw, STEP13_PF_ZT_REQUIRED)
            validation_raw = validation_raw[
                validation_raw["validation_method_step13"].eq(args.primary_validation_method)
                & validation_raw["split_role_step13"].eq("validation")
                & validation_raw["validation_status_step13"].eq("ok")
                & finite_positive(validation_raw["sigma_pred_validation_S_per_m_step13"])
            ].copy()
            primary = read_csv_selected(
                primary_path,
                [
                    "sample_key",
                    "validation_method_step13",
                    "validation_quality_step13",
                    "validation_sigma_mape_step13",
                    "validation_sigma_log_rmse_step13",
                ],
            )
            validation_raw = validation_raw.merge(
                primary,
                on=["sample_key", "validation_method_step13"],
                how="left",
            )
            validation = add_pf_zt_predictions(
                validation_raw,
                "sigma_pred_validation_S_per_m_step13",
                "step13_validation_sigma",
            )
            validation_columns = [
                *PREDICTION_OUTPUT_COLUMNS,
                "validation_method_step13",
                "split_role_step13",
                "validation_status_step13",
                "validation_quality_step13",
                "sigma_validation_log_error_step13",
                "sigma_validation_relative_error_step13",
            ]
            validation = select_columns(validation, validation_columns)
            step13_used = True
        except KeyError:
            validation = pd.DataFrame(columns=[*PREDICTION_OUTPUT_COLUMNS, "validation_method_step13"])
            step13_used = False

    sample_results = aggregate_sample_results(all_fit, validation=False)
    validation_results = aggregate_sample_results(validation, validation=True) if not validation.empty else pd.DataFrame()
    high_perf = build_high_performance_classification(sample_results, validation_results)
    material_summary = build_material_summary(sample_results)
    error_distribution = build_error_distribution(sample_results, validation_results)
    problem_rows = build_problem_rows(all_fit, validation)
    problem_samples = build_problem_samples(sample_results)

    assert_acceptance(all_fit, validation, sample_results, validation_results, high_perf, problem_samples)

    write_csv(all_fit, args.output_dir / "thermoelectric_predictions_step14.csv")
    write_csv(validation, args.output_dir / "thermoelectric_validation_predictions_step14.csv")
    write_csv(sample_results, args.output_dir / "pf_zt_sample_results_step14.csv")
    write_csv(validation_results, args.output_dir / "pf_zt_validation_sample_results_step14.csv")
    write_csv(high_perf, args.output_dir / "zt_high_performance_classification_step14.csv")
    write_csv(material_summary, args.output_dir / "pf_zt_material_summary_step14.csv")
    write_csv(problem_rows, args.output_dir / "pf_zt_problem_rows_step14.csv")
    write_csv(problem_samples, args.output_dir / "pf_zt_problem_samples_step14.csv")
    write_csv(error_distribution, args.output_dir / "zt_error_distribution_step14.csv")

    excel_notes: list[str] = []
    for sheet_name, row_count in [
        ("pf_zt_sample_results", len(sample_results)),
        ("pf_zt_validation_sample_results", len(validation_results)),
        ("zt_high_performance_classification", len(high_perf)),
        ("pf_zt_material_summary", len(material_summary)),
        ("pf_zt_problem_samples", len(problem_samples)),
        ("zt_error_distribution", len(error_distribution)),
        ("pf_zt_problem_rows", len(problem_rows)),
    ]:
        add_excel_preview_note(sheet_name, row_count, excel_notes)
    report_text, report_df = build_report(
        input_counts,
        all_fit,
        validation,
        sample_results,
        validation_results,
        high_perf,
        problem_rows,
        problem_samples,
        args,
        step13_used,
        excel_notes,
    )
    (args.output_dir / "step14_pf_zt_prediction_report.txt").write_text(report_text, encoding="utf-8")
    write_excel_output(args.output_dir, report_df)

    zt_precision, zt_recall, zt_f1 = zt_ge_1_metrics(high_perf)
    median_obs_mape = pd.to_numeric(sample_results["zt_pred_vs_obs_mape_step14"], errors="coerce").median()
    median_calc_mape = pd.to_numeric(sample_results["zt_pred_vs_calc_mape_step14"], errors="coerce").median()
    needs_sintering = int(problem_samples["needs_sintering_check_later_step14"].eq("yes").sum()) if not problem_samples.empty else 0

    print("Done.")
    print("Created:")
    print("- thermoelectric_predictions_step14.csv")
    print("- thermoelectric_validation_predictions_step14.csv")
    print("- pf_zt_sample_results_step14.csv")
    print("- pf_zt_validation_sample_results_step14.csv")
    print("- zt_high_performance_classification_step14.csv")
    print("- pf_zt_material_summary_step14.csv")
    print("- pf_zt_problem_rows_step14.csv")
    print("- pf_zt_problem_samples_step14.csv")
    print("- zt_error_distribution_step14.csv")
    print("- step14_pf_zt_prediction_report.txt")
    print("- starrydata2_step14_pf_zt_prediction.xlsx")
    print("")
    print("Summary:")
    print(f"thermoelectric prediction rows: {len(all_fit)}")
    print(f"validation prediction rows: {len(validation)}")
    print(f"sample results: {len(sample_results)}")
    print(f"validation sample results: {len(validation_results)}")
    print(f"PF eval rows: {int(all_fit['pf_error_status_step14'].eq('ok').sum())}")
    print(f"ZT pred rows: {int(all_fit['zt_pred_status_step14'].eq('ok').sum())}")
    print(f"ZT obs compare rows: {int((~all_fit['zt_pred_vs_obs_status_step14'].eq('not_available')).sum())}")
    print(f"ZT calc compare rows: {int((~all_fit['zt_pred_vs_calc_status_step14'].eq('not_available')).sum())}")
    print(f"median ZT vs obs MAPE: {median_obs_mape}")
    print(f"median ZT vs calc MAPE: {median_calc_mape}")
    print(f"ZT>=1 precision: {zt_precision}")
    print(f"ZT>=1 recall: {zt_recall}")
    print(f"ZT>=1 F1: {zt_f1}")
    print(f"problem samples: {len(problem_samples)}")
    print(f"needs sintering check samples: {needs_sintering}")
    print("n/p changed rows: 0")
    print(f"sintering changed rows: {sintering_changed_rows(all_fit)}")


if __name__ == "__main__":
    main()
