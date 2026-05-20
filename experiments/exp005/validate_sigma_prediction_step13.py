import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP12_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step13_sigma_validation"

REQUIRED_INPUT_FILES = {
    "predictions": "sigma_predictions_step12.csv",
    "tau_results": "tau_fit_results_step12.csv",
    "ready_samples": "tau_fit_ready_samples_step12.csv",
}
OPTIONAL_INPUT_FILES = {
    "initial_predictions": "initial_tau_fit_predictions_step12.csv",
    "problem_samples": "tau_fit_problem_samples_step12.csv",
    "holdout_eval": "tau_fit_holdout_eval_step12.csv",
    "material_summary": "tau_fit_material_summary_step12.csv",
}

SUPPORTED_METHODS = {
    "high_temperature_holdout",
    "low_temperature_holdout",
    "interleaved_holdout",
}
EXCEL_PREVIEW_ROWS = 100_000
EPS = 1e-12

PREDICTION_COLUMNS = [
    "sample_key",
    "temperature_K",
    "validation_method_step13",
    "split_role_step13",
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
    "sigma_obs_S_per_m_step11",
    "prefactor_C_S_per_m_step12",
    "tau_eff_step12",
    "tau_eff_train_step13",
    "log_tau_eff_train_step13",
    "sigma_pred_S_per_m_step12",
    "sigma_pred_validation_S_per_m_step13",
    "sigma_validation_residual_S_per_m_step13",
    "sigma_validation_abs_error_S_per_m_step13",
    "sigma_validation_relative_error_step13",
    "sigma_validation_log_error_step13",
    "prefactor_source_step12",
    "prefactor_group_key_step12",
    "prefactor_status_step12",
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
    "can_eval_power_factor_step11",
    "can_calc_zt_from_obs_step11",
    "can_compare_zt_obs_step11",
    "validation_status_step13",
]

RESULT_METADATA_COLUMNS = [
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
]

DATA_KEEP_COLUMNS = sorted(
    set(
        [
            "sample_key",
            "temperature_K",
            "sigma_obs_S_per_m_step11",
            "prefactor_C_S_per_m_step12",
            "fit_status_step12",
            "tau_eff_step12",
            "log_tau_eff_step12",
            "tau_eff_unit_step12",
            "tau_eff_mode_step12",
            "sigma_pred_S_per_m_step12",
            "sigma_log_error_step12",
            "sigma_relative_error_step12",
            "prefactor_source_step12",
            "prefactor_group_key_step12",
            "prefactor_status_step12",
            "seebeck_obs_V_per_K_step11",
            "kappa_obs_W_per_mK_step11",
            "zt_obs_dimensionless_step11",
            "power_factor_obs_W_per_mK2_step11",
            "zt_calc_from_obs_step11",
            "can_eval_power_factor_step11",
            "can_calc_zt_from_obs_step11",
            "can_compare_zt_obs_step11",
            *RESULT_METADATA_COLUMNS,
        ]
    )
)


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
    parser = argparse.ArgumentParser(description="Validate Step12 sigma prediction by temperature holdout.")
    parser.add_argument("--step12_dir", type=Path, default=DEFAULT_STEP12_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_train_points", type=int, default=5)
    parser.add_argument("--min_validation_points", type=int, default=2)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--primary_validation_method", default="high_temperature_holdout")
    parser.add_argument("--run_interleaved_validation", type=parse_bool, default=True)
    parser.add_argument("--run_low_temperature_validation", type=parse_bool, default=True)
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
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_csv_text(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)


def input_paths(step12_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, filename in REQUIRED_INPUT_FILES.items():
        path = step12_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required Step12 input file not found: {path}")
        paths[label] = path
    for label, filename in OPTIONAL_INPUT_FILES.items():
        path = step12_dir / filename
        if path.exists():
            paths[label] = path
    return paths


def validate_predictions(df: pd.DataFrame) -> None:
    required = [
        "sample_key",
        "temperature_K",
        "sigma_obs_S_per_m_step11",
        "prefactor_C_S_per_m_step12",
        "fit_status_step12",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"sigma_predictions_step12.csv missing required columns: {missing}")


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def first_nonempty(series: pd.Series) -> Any:
    for value in series:
        text = normalize_text(value)
        if text:
            return value
    return series.iloc[0] if len(series) else ""


def prepare_validation_input(predictions: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(predictions, DATA_KEEP_COLUMNS)
    data = predictions.loc[:, DATA_KEEP_COLUMNS].copy()
    data["temperature_K"] = pd.to_numeric(data["temperature_K"], errors="coerce")
    data["sigma_obs_S_per_m_step11"] = pd.to_numeric(
        data["sigma_obs_S_per_m_step11"], errors="coerce"
    )
    data["prefactor_C_S_per_m_step12"] = pd.to_numeric(
        data["prefactor_C_S_per_m_step12"], errors="coerce"
    )
    usable = (
        data["fit_status_step12"].eq("ok")
        & np.isfinite(data["sigma_obs_S_per_m_step11"])
        & (data["sigma_obs_S_per_m_step11"] > 0)
        & np.isfinite(data["prefactor_C_S_per_m_step12"])
        & (data["prefactor_C_S_per_m_step12"] > 0)
        & np.isfinite(data["temperature_K"])
    )
    data["usable_for_validation_step13"] = usable
    data["validation_input_reason_step13"] = np.where(
        usable, "usable for Step13 validation", "not usable for Step13 validation"
    )
    return data.loc[usable].copy().reset_index(drop=True)


def validation_methods(args: argparse.Namespace) -> list[str]:
    if args.primary_validation_method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported primary_validation_method: {args.primary_validation_method}")
    methods = [args.primary_validation_method]
    if args.run_low_temperature_validation:
        methods.append("low_temperature_holdout")
    if args.run_interleaved_validation:
        methods.append("interleaved_holdout")
    ordered: list[str] = []
    for method in methods:
        if method not in ordered:
            ordered.append(method)
    return ordered


def split_indices(
    sorted_index: np.ndarray,
    method: str,
    holdout_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(sorted_index)
    validation_count = max(1, int(math.ceil(n_rows * holdout_fraction)))
    validation_count = min(validation_count, max(n_rows - 1, 0))
    train_count = n_rows - validation_count
    if method == "high_temperature_holdout":
        train_idx = sorted_index[:train_count]
        validation_idx = sorted_index[train_count:]
    elif method == "low_temperature_holdout":
        validation_idx = sorted_index[:validation_count]
        train_idx = sorted_index[validation_count:]
    elif method == "interleaved_holdout":
        positions = np.round(np.linspace(0, n_rows - 1, validation_count)).astype(int)
        selected = []
        seen = set()
        for position in positions:
            position = int(position)
            if position not in seen:
                selected.append(position)
                seen.add(position)
        if len(selected) < validation_count:
            for position in range(n_rows):
                if position not in seen:
                    selected.append(position)
                    seen.add(position)
                    if len(selected) == validation_count:
                        break
        validation_positions = np.array(sorted(selected), dtype=int)
        train_positions = np.array(
            [position for position in range(n_rows) if position not in set(validation_positions)],
            dtype=int,
        )
        validation_idx = sorted_index[validation_positions]
        train_idx = sorted_index[train_positions]
    else:
        raise ValueError(f"Unsupported validation method: {method}")
    return train_idx, validation_idx


def regression_metrics(obs: pd.Series, pred: pd.Series) -> dict[str, float]:
    obs_values = pd.to_numeric(obs, errors="coerce").to_numpy(dtype=float)
    pred_values = pd.to_numeric(pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs_values) & np.isfinite(pred_values) & (obs_values > 0) & (pred_values > 0)
    if not mask.any():
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "log_mae": math.nan,
            "log_rmse": math.nan,
            "r2_log": math.nan,
            "bias_log": math.nan,
            "within_10": math.nan,
            "within_25": math.nan,
            "within_50": math.nan,
        }
    obs_values = obs_values[mask]
    pred_values = pred_values[mask]
    residual = pred_values - obs_values
    relative = np.abs(residual) / np.maximum(np.abs(obs_values), EPS)
    log_error = np.log(pred_values) - np.log(obs_values)
    log_true = np.log(obs_values)
    ss_res = float(np.sum(log_error**2))
    ss_tot = float(np.sum((log_true - np.mean(log_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(relative)),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "r2_log": math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot,
        "bias_log": float(np.mean(log_error)),
        "within_10": float(np.mean(relative <= 0.10)),
        "within_25": float(np.mean(relative <= 0.25)),
        "within_50": float(np.mean(relative <= 0.50)),
    }


def validation_quality(status: str, log_rmse: float, mape: float) -> str:
    if status != "ok":
        return "not_evaluated"
    if math.isfinite(log_rmse) and math.isfinite(mape):
        if log_rmse <= 0.20 and mape <= 0.20:
            return "excellent"
        if log_rmse <= 0.40 and mape <= 0.50:
            return "good"
        if log_rmse <= 0.80 and mape <= 1.00:
            return "moderate"
    return "poor"


def run_validation_method(
    data: pd.DataFrame,
    method: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = data.copy()
    output["validation_method_step13"] = method
    output["split_role_step13"] = "not_used"
    output["validation_status_step13"] = "not_enough_points"
    output["tau_eff_train_step13"] = np.nan
    output["log_tau_eff_train_step13"] = np.nan
    output["sigma_pred_validation_S_per_m_step13"] = np.nan
    output["sigma_validation_residual_S_per_m_step13"] = np.nan
    output["sigma_validation_abs_error_S_per_m_step13"] = np.nan
    output["sigma_validation_relative_error_step13"] = np.nan
    output["sigma_validation_log_error_step13"] = np.nan

    result_records: list[dict[str, Any]] = []
    diagnostic_records: list[dict[str, Any]] = []

    for sample_key, group in data.groupby("sample_key", sort=True):
        sorted_group = group.sort_values(["temperature_K", "sigma_obs_S_per_m_step11"])
        sorted_index = sorted_group.index.to_numpy()
        train_idx, validation_idx = split_indices(sorted_index, method, args.holdout_fraction)
        n_train = len(train_idx)
        n_validation = len(validation_idx)
        n_total = len(sorted_index)
        status = (
            "ok"
            if n_train >= args.min_train_points and n_validation >= args.min_validation_points
            else "not_enough_points"
        )
        output.loc[sorted_index, "validation_status_step13"] = status

        if status == "ok":
            output.loc[train_idx, "split_role_step13"] = "train"
            output.loc[validation_idx, "split_role_step13"] = "validation"
            train = output.loc[train_idx]
            log_tau = float(
                np.mean(
                    np.log(train["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
                    - np.log(train["prefactor_C_S_per_m_step12"].to_numpy(dtype=float))
                )
            )
            tau = float(np.exp(log_tau))
            output.loc[sorted_index, "tau_eff_train_step13"] = tau
            output.loc[sorted_index, "log_tau_eff_train_step13"] = log_tau
            pred = output.loc[sorted_index, "prefactor_C_S_per_m_step12"].to_numpy(dtype=float) * tau
            obs = output.loc[sorted_index, "sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
            output.loc[sorted_index, "sigma_pred_validation_S_per_m_step13"] = pred
            residual = pred - obs
            output.loc[sorted_index, "sigma_validation_residual_S_per_m_step13"] = residual
            output.loc[sorted_index, "sigma_validation_abs_error_S_per_m_step13"] = np.abs(residual)
            output.loc[sorted_index, "sigma_validation_relative_error_step13"] = np.abs(residual) / np.maximum(
                np.abs(obs), EPS
            )
            output.loc[sorted_index, "sigma_validation_log_error_step13"] = np.log(pred) - np.log(obs)
            train_metrics = regression_metrics(
                output.loc[train_idx, "sigma_obs_S_per_m_step11"],
                output.loc[train_idx, "sigma_pred_validation_S_per_m_step13"],
            )
            validation_metrics = regression_metrics(
                output.loc[validation_idx, "sigma_obs_S_per_m_step11"],
                output.loc[validation_idx, "sigma_pred_validation_S_per_m_step13"],
            )
            note = "validation completed"
        else:
            tau = math.nan
            log_tau = math.nan
            train_metrics = regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
            validation_metrics = regression_metrics(pd.Series(dtype=float), pd.Series(dtype=float))
            note = "not enough train or validation points"

        group_for_meta = output.loc[sorted_index]
        metadata = {
            column: first_nonempty(group_for_meta[column])
            for column in RESULT_METADATA_COLUMNS
            if column in group_for_meta.columns
        }
        temp_min = float(group_for_meta["temperature_K"].min()) if n_total else math.nan
        temp_max = float(group_for_meta["temperature_K"].max()) if n_total else math.nan
        train_temps = group_for_meta.loc[train_idx, "temperature_K"] if len(train_idx) else pd.Series(dtype=float)
        val_temps = (
            group_for_meta.loc[validation_idx, "temperature_K"]
            if len(validation_idx)
            else pd.Series(dtype=float)
        )
        step12_tau = pd.to_numeric(group_for_meta["tau_eff_step12"], errors="coerce").dropna()
        step12_tau_value = float(step12_tau.iloc[0]) if not step12_tau.empty else math.nan
        log_step12_tau = math.log(step12_tau_value) if math.isfinite(step12_tau_value) and step12_tau_value > 0 else math.nan
        log_diff = log_tau - log_step12_tau if math.isfinite(log_tau) and math.isfinite(log_step12_tau) else math.nan
        ratio = tau / step12_tau_value if math.isfinite(tau) and math.isfinite(step12_tau_value) and step12_tau_value > 0 else math.nan
        quality = validation_quality(status, validation_metrics["log_rmse"], validation_metrics["mape"])

        result_records.append(
            {
                "sample_key": sample_key,
                "validation_method_step13": method,
                **metadata,
                "tau_eff_step12": first_nonempty(group_for_meta["tau_eff_step12"]),
                "tau_eff_train_step13": tau,
                "log_tau_eff_train_step13": log_tau,
                "tau_eff_unit_step13": first_nonempty(group_for_meta.get("tau_eff_unit_step12", pd.Series(["relative_scale"]))),
                "tau_eff_mode_step13": first_nonempty(
                    group_for_meta.get("tau_eff_mode_step12", pd.Series(["empirical_group_baseline"]))
                ),
                "n_total_rows_step13": n_total,
                "n_train_rows_step13": n_train,
                "n_validation_rows_step13": n_validation,
                "temperature_min_step13": temp_min,
                "temperature_max_step13": temp_max,
                "temperature_span_step13": temp_max - temp_min if math.isfinite(temp_min) and math.isfinite(temp_max) else "",
                "validation_temperature_min_step13": float(val_temps.min()) if not val_temps.empty else "",
                "validation_temperature_max_step13": float(val_temps.max()) if not val_temps.empty else "",
                "train_sigma_mae_step13": train_metrics["mae"],
                "train_sigma_rmse_step13": train_metrics["rmse"],
                "train_sigma_mape_step13": train_metrics["mape"],
                "train_sigma_log_mae_step13": train_metrics["log_mae"],
                "train_sigma_log_rmse_step13": train_metrics["log_rmse"],
                "train_sigma_r2_log_step13": train_metrics["r2_log"],
                "validation_sigma_mae_step13": validation_metrics["mae"],
                "validation_sigma_rmse_step13": validation_metrics["rmse"],
                "validation_sigma_mape_step13": validation_metrics["mape"],
                "validation_sigma_log_mae_step13": validation_metrics["log_mae"],
                "validation_sigma_log_rmse_step13": validation_metrics["log_rmse"],
                "validation_sigma_r2_log_step13": validation_metrics["r2_log"],
                "validation_sigma_bias_log_step13": validation_metrics["bias_log"],
                "validation_within_10pct_rate_step13": validation_metrics["within_10"],
                "validation_within_25pct_rate_step13": validation_metrics["within_25"],
                "validation_within_50pct_rate_step13": validation_metrics["within_50"],
                "tau_eff_train_vs_step12_log_diff_step13": log_diff,
                "tau_eff_train_vs_step12_ratio_step13": ratio,
                "validation_status_step13": status,
                "validation_quality_step13": quality,
                "validation_note_step13": note,
            }
        )
        diagnostic_records.append(
            {
                "sample_key": sample_key,
                "validation_method_step13": method,
                "n_total_rows_step13": n_total,
                "n_train_rows_step13": n_train,
                "n_validation_rows_step13": n_validation,
                "temperature_min_step13": temp_min,
                "temperature_max_step13": temp_max,
                "temperature_span_step13": temp_max - temp_min if math.isfinite(temp_min) and math.isfinite(temp_max) else "",
                "train_temperature_min_step13": float(train_temps.min()) if not train_temps.empty else "",
                "train_temperature_max_step13": float(train_temps.max()) if not train_temps.empty else "",
                "validation_temperature_min_step13": float(val_temps.min()) if not val_temps.empty else "",
                "validation_temperature_max_step13": float(val_temps.max()) if not val_temps.empty else "",
                "split_status_step13": status,
                "split_note_step13": note,
            }
        )

    return (
        selected_columns(output, PREDICTION_COLUMNS),
        pd.DataFrame(result_records),
        pd.DataFrame(diagnostic_records),
    )


def selected_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ensure_columns(df, columns)
    return df.loc[:, columns]


def build_good_samples(primary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_key",
        "material_system",
        "n_or_p",
        "composition",
        "tau_eff_train_step13",
        "validation_sigma_log_rmse_step13",
        "validation_sigma_mape_step13",
        "validation_within_25pct_rate_step13",
        "validation_quality_step13",
        "temperature_span_step13",
        "n_train_rows_step13",
        "n_validation_rows_step13",
    ]
    good = primary[
        primary["validation_status_step13"].eq("ok")
        & primary["validation_quality_step13"].isin(["excellent", "good"])
    ].copy()
    return selected_columns(good, columns)


def validation_problem_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if normalize_text(row.get("validation_status_step13")) != "ok":
        reasons.append("not enough points")
    if normalize_text(row.get("validation_quality_step13")) == "poor":
        reasons.append("poor validation accuracy")
    log_rmse = pd.to_numeric(pd.Series([row.get("validation_sigma_log_rmse_step13")]), errors="coerce").iloc[0]
    mape = pd.to_numeric(pd.Series([row.get("validation_sigma_mape_step13")]), errors="coerce").iloc[0]
    ratio = pd.to_numeric(pd.Series([row.get("tau_eff_train_vs_step12_ratio_step13")]), errors="coerce").iloc[0]
    if math.isfinite(log_rmse) and log_rmse > 0.80:
        reasons.append("large log RMSE")
    if math.isfinite(mape) and mape > 1.00:
        reasons.append("large MAPE")
    if math.isfinite(ratio) and (ratio > 10.0 or ratio < 0.1):
        reasons.append("tau_eff train differs greatly from Step12")
    return "; ".join(reasons) if reasons else "review recommended"


def build_problem_samples(primary: pd.DataFrame) -> pd.DataFrame:
    log_rmse = pd.to_numeric(primary["validation_sigma_log_rmse_step13"], errors="coerce")
    mape = pd.to_numeric(primary["validation_sigma_mape_step13"], errors="coerce")
    ratio = pd.to_numeric(primary["tau_eff_train_vs_step12_ratio_step13"], errors="coerce")
    problem = (
        ~primary["validation_status_step13"].eq("ok")
        | primary["validation_quality_step13"].eq("poor")
        | (log_rmse > 0.80)
        | (mape > 1.00)
        | (ratio > 10.0)
        | (ratio < 0.1)
    )
    output = primary.loc[problem].copy()
    output["validation_problem_reason_step13"] = output.apply(validation_problem_reason, axis=1)
    return output


def bool_count(series: pd.Series) -> int:
    return int(series.map(normalize_bool).sum())


def build_material_summary(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby(["material_system", "n_or_p", "validation_method_step13"], dropna=False, sort=True)
    return grouped.agg(
        sample_count=("sample_key", "count"),
        validation_ok_sample_count=(
            "validation_status_step13",
            lambda values: int(pd.Series(values).eq("ok").sum()),
        ),
        excellent_sample_count=(
            "validation_quality_step13",
            lambda values: int(pd.Series(values).eq("excellent").sum()),
        ),
        good_sample_count=(
            "validation_quality_step13",
            lambda values: int(pd.Series(values).eq("good").sum()),
        ),
        moderate_sample_count=(
            "validation_quality_step13",
            lambda values: int(pd.Series(values).eq("moderate").sum()),
        ),
        poor_sample_count=(
            "validation_quality_step13",
            lambda values: int(pd.Series(values).eq("poor").sum()),
        ),
        median_validation_sigma_log_rmse_step13=(
            "validation_sigma_log_rmse_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_validation_sigma_mape_step13=(
            "validation_sigma_mape_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_validation_within_25pct_rate_step13=(
            "validation_within_25pct_rate_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_tau_eff_train_step13=(
            "tau_eff_train_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_temperature_span_step13=(
            "temperature_span_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        nanocarbon_sample_count=("nanocarbon_keyword_detected_step9", bool_count),
        rare_metal_flag_sample_count=("rare_metal_flag_auto_step9", bool_count),
        toxicity_flag_sample_count=("toxicity_flag_auto_step9", bool_count),
    ).reset_index()


def distribution_stats(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": "",
            "median": "",
            "p10": "",
            "p25": "",
            "p75": "",
            "p90": "",
            "max": "",
        }
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


def build_error_distribution(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "validation_sigma_log_rmse_step13",
        "validation_sigma_mape_step13",
        "validation_within_25pct_rate_step13",
    ]
    rows: list[dict[str, Any]] = []
    grouped = results.groupby(["validation_method_step13", "n_or_p", "material_system"], dropna=False, sort=True)
    for group_values, group in grouped:
        method, n_or_p, material_system = group_values
        for metric in metrics:
            rows.append(
                {
                    "validation_method_step13": method,
                    "n_or_p": n_or_p,
                    "material_system": material_system,
                    "metric_step13": metric,
                    **distribution_stats(group[metric]),
                }
            )
    return pd.DataFrame(rows)


def describe_series(series: pd.Series, prefix: str) -> list[tuple[str, str]]:
    stats = distribution_stats(series)
    return [(f"{prefix}_{key}", str(value)) for key, value in stats.items() if key != "p10"]


def value_counts_rows(prefix: str, series: pd.Series) -> list[tuple[str, str]]:
    counts = series.fillna("").astype(str).value_counts().sort_index()
    return [(f"{prefix}_{key}_count", str(int(value))) for key, value in counts.items()]


def build_report(
    input_counts: dict[str, int],
    validation_input_rows: int,
    validation_samples: int,
    methods: list[str],
    primary_method: str,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    primary: pd.DataFrame,
    good: pd.DataFrame,
    problem: pd.DataFrame,
    material_summary: pd.DataFrame,
    args: argparse.Namespace,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = [
        ("input_sigma_predictions_step12_rows", str(input_counts["predictions"])),
        ("input_tau_fit_results_step12_sample_count", str(input_counts["tau_results"])),
        ("input_tau_fit_ready_samples_step12_sample_count", str(input_counts["ready_samples"])),
        ("validation_input_rows", str(validation_input_rows)),
        ("validation_sample_count", str(validation_samples)),
        ("validation_methods", ";".join(methods)),
        ("primary_validation_method", primary_method),
        ("holdout_fraction", str(args.holdout_fraction)),
        ("min_train_points", str(args.min_train_points)),
        ("min_validation_points", str(args.min_validation_points)),
        ("sigma_validation_predictions_step13_rows", str(len(predictions))),
        ("tau_validation_results_step13_rows", str(len(results))),
        ("tau_validation_primary_results_step13_rows", str(len(primary))),
        ("tau_validation_good_samples_step13_rows", str(len(good))),
        ("tau_validation_problem_samples_step13_rows", str(len(problem))),
        ("tau_validation_material_summary_step13_rows", str(len(material_summary))),
    ]
    rows.extend(value_counts_rows("validation_status_step13", results["validation_status_step13"]))
    rows.extend(value_counts_rows("validation_quality_step13", results["validation_quality_step13"]))

    primary_ok = primary[primary["validation_status_step13"].eq("ok")]
    primary_quality_counts = primary["validation_quality_step13"].value_counts()
    rows.extend(
        [
            ("primary_ok_sample_count", str(len(primary_ok))),
            ("primary_excellent_sample_count", str(int(primary_quality_counts.get("excellent", 0)))),
            ("primary_good_sample_count", str(int(primary_quality_counts.get("good", 0)))),
            ("primary_moderate_sample_count", str(int(primary_quality_counts.get("moderate", 0)))),
            ("primary_poor_sample_count", str(int(primary_quality_counts.get("poor", 0)))),
            (
                "primary_not_evaluated_sample_count",
                str(int(primary_quality_counts.get("not_evaluated", 0))),
            ),
        ]
    )
    for metric in [
        "validation_sigma_log_rmse_step13",
        "validation_sigma_mape_step13",
        "validation_within_10pct_rate_step13",
        "validation_within_25pct_rate_step13",
        "validation_within_50pct_rate_step13",
    ]:
        rows.extend(describe_series(primary[metric], f"primary_{metric}"))

    for n_or_p, group in primary_ok.groupby("n_or_p", dropna=False):
        rows.append((f"n_or_p_{n_or_p}_primary_validation_ok_sample_count", str(len(group))))
        rows.append(
            (
                f"n_or_p_{n_or_p}_median_validation_sigma_log_rmse_step13",
                str(float(pd.to_numeric(group["validation_sigma_log_rmse_step13"], errors="coerce").median())),
            )
        )
    material_ok = primary_ok.groupby("material_system", dropna=False).agg(
        ok_sample_count=("sample_key", "count"),
        median_log_rmse=(
            "validation_sigma_log_rmse_step13",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
    )
    for material_system, row in material_ok.sort_values("ok_sample_count", ascending=False).head(20).iterrows():
        rows.append((f"material_system_{material_system}_primary_validation_ok_sample_count", str(int(row["ok_sample_count"]))))
    for material_system, row in material_ok.sort_values("median_log_rmse", ascending=True).head(20).iterrows():
        rows.append((f"material_system_{material_system}_median_validation_sigma_log_rmse_step13", str(float(row["median_log_rmse"]))))

    sintering_changed = (
        ~predictions["sintering_method"].astype(str).str.casefold().eq("unknown")
        | ~predictions["sintering_checked"].astype(str).str.casefold().eq("no")
        | ~predictions["record_checked"].astype(str).str.casefold().eq("no")
    )
    rows.extend(
        [
            (
                "sintering_method_unknown_rows",
                str(int(predictions["sintering_method"].astype(str).str.casefold().eq("unknown").sum())),
            ),
            (
                "sintering_checked_no_rows",
                str(int(predictions["sintering_checked"].astype(str).str.casefold().eq("no").sum())),
            ),
            (
                "record_checked_no_rows",
                str(int(predictions["record_checked"].astype(str).str.casefold().eq("no").sum())),
            ),
            ("n_p_changed_rows", "0"),
            ("sintering_changed_rows", str(int(sintering_changed.sum()))),
            ("note", "Step13 did not predict Seebeck coefficient, thermal conductivity, or ZT."),
            (
                "note",
                "Step13 validated sigma prediction accuracy from tau_eff by temperature-point holdout.",
            ),
            (
                "note",
                "Step13 tau_eff is a relative scalar against the empirical baseline by default, as in Step12.",
            ),
        ]
    )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    return "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n", report_df


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    csv_frame(df).to_csv(path, index=False)


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
        "tau_validation_results": "tau_validation_results_step13.csv",
        "tau_validation_primary_results": "tau_validation_primary_results_step13.csv",
        "tau_validation_good_samples": "tau_validation_good_samples_step13.csv",
        "tau_validation_problem_samples": "tau_validation_problem_samples_step13.csv",
        "tau_validation_material_summary": "tau_validation_material_summary_step13.csv",
        "temperature_split_diagnostics": "temperature_split_diagnostics_step13.csv",
        "error_distribution": "tau_validation_error_distribution_step13.csv",
    }
    with pd.ExcelWriter(output_dir / "starrydata2_step13_sigma_validation.xlsx", engine="openpyxl") as writer:
        for sheet_name, filename in sheet_files.items():
            frame = read_csv_text(output_dir / filename, nrows=EXCEL_PREVIEW_ROWS)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)
        report_df.to_excel(writer, sheet_name="validation_report", index=False)
        fit_worksheet(writer, "validation_report", report_df)


def assert_acceptance(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    primary: pd.DataFrame,
    good: pd.DataFrame,
    problem: pd.DataFrame,
    material_summary: pd.DataFrame,
    primary_method: str,
) -> None:
    for column in [
        "sample_key",
        "validation_method_step13",
        "split_role_step13",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_validation_S_per_m_step13",
        "sigma_validation_log_error_step13",
        "validation_status_step13",
    ]:
        if column not in predictions.columns:
            raise KeyError(f"sigma_validation_predictions_step13 missing {column}")
    if predictions.duplicated(["sample_key", "temperature_K", "validation_method_step13"]).any():
        raise ValueError("sigma_validation_predictions_step13 is not one row per sample-temperature-method")
    for column in [
        "tau_eff_train_step13",
        "validation_sigma_log_rmse_step13",
        "validation_sigma_mape_step13",
        "validation_quality_step13",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in results.columns:
            raise KeyError(f"tau_validation_results_step13 missing {column}")
    if results.duplicated(["sample_key", "validation_method_step13"]).any():
        raise ValueError("tau_validation_results_step13 is not one row per sample-method")
    if not primary["validation_method_step13"].eq(primary_method).all():
        raise ValueError("primary validation results contain non-primary methods")
    if not good.empty and not good["validation_quality_step13"].isin(["excellent", "good"]).all():
        raise ValueError("good samples contain non-good validation quality")
    if "validation_problem_reason_step13" not in problem.columns:
        raise KeyError("problem samples missing validation_problem_reason_step13")
    for column in ["material_system", "n_or_p", "validation_method_step13"]:
        if column not in material_summary.columns:
            raise KeyError(f"material summary missing {column}")
    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if not results[column].astype(str).str.casefold().eq(expected).all():
            raise ValueError(f"{column} changed from expected {expected}")


def main() -> None:
    args = parse_args()
    paths = input_paths(args.step12_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_counts = {label: count_csv_rows(path) for label, path in paths.items()}

    tau_results = read_csv_text(paths["tau_results"], nrows=1)
    ready_samples = read_csv_text(paths["ready_samples"], nrows=1)
    del tau_results, ready_samples

    predictions_raw = read_csv_text(paths["predictions"])
    validate_predictions(predictions_raw)
    validation_input = prepare_validation_input(predictions_raw)
    del predictions_raw

    methods = validation_methods(args)
    prediction_frames: list[pd.DataFrame] = []
    result_frames: list[pd.DataFrame] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for method in methods:
        method_predictions, method_results, method_diagnostics = run_validation_method(
            validation_input, method, args
        )
        prediction_frames.append(method_predictions)
        result_frames.append(method_results)
        diagnostic_frames.append(method_diagnostics)

    validation_predictions = pd.concat(prediction_frames, ignore_index=True)
    validation_results = pd.concat(result_frames, ignore_index=True)
    split_diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    primary_results = validation_results[
        validation_results["validation_method_step13"].eq(args.primary_validation_method)
    ].copy()
    good_samples = build_good_samples(primary_results)
    problem_samples = build_problem_samples(primary_results)
    material_summary = build_material_summary(validation_results)
    error_distribution = build_error_distribution(validation_results)

    assert_acceptance(
        validation_predictions,
        validation_results,
        primary_results,
        good_samples,
        problem_samples,
        material_summary,
        args.primary_validation_method,
    )

    write_csv(validation_predictions, args.output_dir / "sigma_validation_predictions_step13.csv")
    write_csv(validation_results, args.output_dir / "tau_validation_results_step13.csv")
    write_csv(primary_results, args.output_dir / "tau_validation_primary_results_step13.csv")
    write_csv(good_samples, args.output_dir / "tau_validation_good_samples_step13.csv")
    write_csv(problem_samples, args.output_dir / "tau_validation_problem_samples_step13.csv")
    write_csv(material_summary, args.output_dir / "tau_validation_material_summary_step13.csv")
    write_csv(error_distribution, args.output_dir / "tau_validation_error_distribution_step13.csv")
    write_csv(split_diagnostics, args.output_dir / "temperature_split_diagnostics_step13.csv")

    excel_notes: list[str] = []
    for sheet_name, row_count in [
        ("tau_validation_results", len(validation_results)),
        ("tau_validation_primary_results", len(primary_results)),
        ("tau_validation_good_samples", len(good_samples)),
        ("tau_validation_problem_samples", len(problem_samples)),
        ("tau_validation_material_summary", len(material_summary)),
        ("temperature_split_diagnostics", len(split_diagnostics)),
        ("error_distribution", len(error_distribution)),
    ]:
        add_excel_preview_note(sheet_name, row_count, excel_notes)

    report_text, report_df = build_report(
        input_counts,
        len(validation_input),
        validation_input["sample_key"].nunique(),
        methods,
        args.primary_validation_method,
        validation_predictions,
        validation_results,
        primary_results,
        good_samples,
        problem_samples,
        material_summary,
        args,
        excel_notes,
    )
    (args.output_dir / "step13_sigma_validation_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    write_excel_output(args.output_dir, report_df)

    primary_quality_counts = primary_results["validation_quality_step13"].value_counts()
    primary_ok = primary_results[primary_results["validation_status_step13"].eq("ok")]
    median_log_rmse = pd.to_numeric(
        primary_results["validation_sigma_log_rmse_step13"], errors="coerce"
    ).median()
    median_mape = pd.to_numeric(primary_results["validation_sigma_mape_step13"], errors="coerce").median()
    median_within25 = pd.to_numeric(
        primary_results["validation_within_25pct_rate_step13"], errors="coerce"
    ).median()
    sintering_changed = (
        ~validation_predictions["sintering_method"].astype(str).str.casefold().eq("unknown")
        | ~validation_predictions["sintering_checked"].astype(str).str.casefold().eq("no")
        | ~validation_predictions["record_checked"].astype(str).str.casefold().eq("no")
    )

    print("Done.")
    print("Created:")
    print("- sigma_validation_predictions_step13.csv")
    print("- tau_validation_results_step13.csv")
    print("- tau_validation_primary_results_step13.csv")
    print("- tau_validation_good_samples_step13.csv")
    print("- tau_validation_problem_samples_step13.csv")
    print("- tau_validation_material_summary_step13.csv")
    print("- tau_validation_error_distribution_step13.csv")
    print("- temperature_split_diagnostics_step13.csv")
    print("- step13_sigma_validation_report.txt")
    print("- starrydata2_step13_sigma_validation.xlsx")
    print("")
    print("Summary:")
    print(f"validation input rows: {len(validation_input)}")
    print(f"validation samples: {validation_input['sample_key'].nunique()}")
    print(f"validation methods: {';'.join(methods)}")
    print(f"primary method: {args.primary_validation_method}")
    print(f"primary ok samples: {len(primary_ok)}")
    print(f"primary excellent samples: {int(primary_quality_counts.get('excellent', 0))}")
    print(f"primary good samples: {int(primary_quality_counts.get('good', 0))}")
    print(f"primary moderate samples: {int(primary_quality_counts.get('moderate', 0))}")
    print(f"primary poor samples: {int(primary_quality_counts.get('poor', 0))}")
    print(f"primary median log RMSE: {median_log_rmse}")
    print(f"primary median MAPE: {median_mape}")
    print(f"primary median within 25pct rate: {median_within25}")
    print(f"problem samples: {len(problem_samples)}")
    print("n/p changed rows: 0")
    print(f"sintering changed rows: {int(sintering_changed.sum())}")


if __name__ == "__main__":
    main()
