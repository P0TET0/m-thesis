import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP11_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step11_unit_normalized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"

REQUIRED_INPUT_FILES = {
    "initial": "initial_tau_fit_training_normalized_step11.csv",
    "training": "training_dataset_normalized_step11.csv",
}
OPTIONAL_INPUT_FILES = {
    "review": "review_training_dataset_normalized_step11.csv",
    "sigma_rho_points": "sigma_rho_points_normalized_step11.csv",
    "unit_audit": "unit_conversion_audit_step11.csv",
    "zt_check": "zt_consistency_check_step11.csv",
}

EXCEL_PREVIEW_ROWS = 100_000
MIN_BASELINE_SAMPLE_COUNT = 3
EPS = 1e-12

METADATA_COLUMNS = [
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
    "fitting_source_preference_step8",
    "fitting_source_actual_step10",
    "sigma_obs_source_step11",
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
    "power_factor_obs_W_per_mK2_step11",
    "zt_calc_from_obs_step11",
    "can_eval_power_factor_step11",
    "can_calc_zt_from_obs_step11",
    "can_compare_zt_obs_step11",
    "row_quality_step11",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit Step12 relative tau_eff from sigma(T).")
    parser.add_argument("--step11_dir", type=Path, default=DEFAULT_STEP11_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_points_per_sample", type=int, default=5)
    parser.add_argument("--temperature_bin_width_K", type=float, default=25.0)
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--prefactor_column", default="none")
    parser.add_argument("--prefactor_mode", default="empirical_group_baseline")
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


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(normalize_bool).sum())


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def read_csv_text(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)


def input_paths(step11_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, filename in REQUIRED_INPUT_FILES.items():
        path = step11_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required Step11 input file not found: {path}")
        paths[label] = path
    for label, filename in OPTIONAL_INPUT_FILES.items():
        path = step11_dir / filename
        if path.exists():
            paths[label] = path
    return paths


def validate_initial(df: pd.DataFrame) -> None:
    required = [
        "sample_key",
        "temperature_K",
        "sigma_obs_S_per_m_step11",
        "can_use_for_initial_tau_fit_step11",
        "can_fit_tau_step11",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"initial_tau_fit_training_normalized_step11.csv missing required columns: {missing}"
        )


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & (numeric > 0)


def prepare_fit_data(initial: pd.DataFrame, temperature_bin_width: float) -> pd.DataFrame:
    ensure_columns(initial, METADATA_COLUMNS)
    data = initial.copy()
    data["temperature_K_numeric_step12"] = pd.to_numeric(data["temperature_K"], errors="coerce")
    data["sigma_obs_S_per_m_numeric_step12"] = pd.to_numeric(
        data["sigma_obs_S_per_m_step11"], errors="coerce"
    )
    usable = (
        data["can_use_for_initial_tau_fit_step11"].map(normalize_bool)
        & data["can_fit_tau_step11"].map(normalize_bool)
        & np.isfinite(data["sigma_obs_S_per_m_numeric_step12"])
        & (data["sigma_obs_S_per_m_numeric_step12"] > 0)
        & np.isfinite(data["temperature_K_numeric_step12"])
    )
    data["usable_for_tau_fit_step12"] = usable
    data["tau_fit_input_reason_step12"] = np.where(
        usable, "usable initial Step11 row", "not usable for initial tau fitting"
    )
    data = data.loc[usable].copy().reset_index(drop=True)
    data["temperature_K"] = data["temperature_K_numeric_step12"]
    data["sigma_obs_S_per_m_step11"] = data["sigma_obs_S_per_m_numeric_step12"]
    data["temperature_bin_K_step12"] = (
        np.round(data["temperature_K"] / temperature_bin_width) * temperature_bin_width
    )
    return data


def group_key_from_values(group_cols: list[str], values: tuple[Any, ...] | Any) -> str:
    if not group_cols:
        return "all_data"
    if len(group_cols) == 1:
        values_tuple = (values,)
    else:
        values_tuple = tuple(values)
    return "|".join(f"{column}={value}" for column, value in zip(group_cols, values_tuple))


def kth_excluding(sorted_all: np.ndarray, own_sorted: np.ndarray, rank: int) -> float:
    lo = 0
    hi = len(sorted_all) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        count_excluding = (mid + 1) - np.searchsorted(own_sorted, sorted_all[mid], side="right")
        if count_excluding > rank:
            hi = mid
        else:
            lo = mid + 1
    return float(sorted_all[lo])


def median_excluding(sorted_all: np.ndarray, own_sorted: np.ndarray) -> float:
    n_excluding = len(sorted_all) - len(own_sorted)
    if n_excluding <= 0:
        return math.nan
    if n_excluding % 2 == 1:
        return kth_excluding(sorted_all, own_sorted, n_excluding // 2)
    low = kth_excluding(sorted_all, own_sorted, n_excluding // 2 - 1)
    high = kth_excluding(sorted_all, own_sorted, n_excluding // 2)
    return (low + high) / 2.0


def leave_one_sample_baselines(
    data: pd.DataFrame,
    group_cols: list[str],
    source_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if group_cols:
        grouped = data.groupby(group_cols, sort=False, dropna=False)
    else:
        grouped = [((), data)]

    for group_values, group in grouped:
        sample_count_total = group["sample_key"].nunique()
        if sample_count_total <= MIN_BASELINE_SAMPLE_COUNT:
            continue
        values = group["sigma_obs_S_per_m_step11"].to_numpy(dtype=float)
        sorted_all = np.sort(values)
        group_key = group_key_from_values(group_cols, group_values)
        for sample_key, sample_group in group.groupby("sample_key", sort=False):
            own_values = np.sort(sample_group["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
            sample_count_excluding = sample_count_total - 1
            if sample_count_excluding < MIN_BASELINE_SAMPLE_COUNT:
                continue
            row_count_excluding = len(group) - len(sample_group)
            if row_count_excluding <= 0:
                continue
            median_value = median_excluding(sorted_all, own_values)
            if not math.isfinite(median_value) or median_value <= 0:
                continue
            record = {
                "sample_key": sample_key,
                "prefactor_C_S_per_m_step12_candidate": median_value,
                "prefactor_source_step12_candidate": source_name,
                "prefactor_group_key_step12_candidate": group_key,
                "prefactor_sample_count_step12_candidate": sample_count_excluding,
                "prefactor_row_count_step12_candidate": row_count_excluding,
            }
            for column in group_cols:
                record[column] = sample_group[column].iloc[0]
            records.append(record)
    return pd.DataFrame(records)


def apply_empirical_prefactors(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    output["prefactor_C_S_per_m_step12"] = np.nan
    output["prefactor_source_step12"] = ""
    output["prefactor_group_key_step12"] = ""
    output["prefactor_sample_count_step12"] = 0
    output["prefactor_row_count_step12"] = 0
    output["prefactor_status_step12"] = "unavailable"

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
    for group_cols, source_name in levels:
        candidates = leave_one_sample_baselines(output, group_cols, source_name)
        if candidates.empty:
            continue
        merge_cols = ["sample_key", *group_cols]
        merged = output[merge_cols].merge(candidates, on=merge_cols, how="left", sort=False)
        candidate_values = pd.to_numeric(
            merged["prefactor_C_S_per_m_step12_candidate"], errors="coerce"
        )
        fill = ~assigned & np.isfinite(candidate_values) & (candidate_values > 0)
        if not fill.any():
            continue
        output.loc[fill, "prefactor_C_S_per_m_step12"] = candidate_values.loc[fill].to_numpy()
        output.loc[fill, "prefactor_source_step12"] = merged.loc[
            fill, "prefactor_source_step12_candidate"
        ].to_numpy()
        output.loc[fill, "prefactor_group_key_step12"] = merged.loc[
            fill, "prefactor_group_key_step12_candidate"
        ].to_numpy()
        output.loc[fill, "prefactor_sample_count_step12"] = merged.loc[
            fill, "prefactor_sample_count_step12_candidate"
        ].to_numpy()
        output.loc[fill, "prefactor_row_count_step12"] = merged.loc[
            fill, "prefactor_row_count_step12_candidate"
        ].to_numpy()
        output.loc[fill, "prefactor_status_step12"] = "ok"
        assigned |= fill
    return output


def apply_external_prefactors(data: pd.DataFrame, prefactor_column: str) -> pd.DataFrame:
    output = data.copy()
    values = pd.to_numeric(output[prefactor_column], errors="coerce")
    ok = np.isfinite(values) & (values > 0)
    output["prefactor_C_S_per_m_step12"] = values.where(ok, np.nan)
    output["prefactor_source_step12"] = "external_prefactor"
    output["prefactor_group_key_step12"] = prefactor_column
    output["prefactor_sample_count_step12"] = ""
    output["prefactor_row_count_step12"] = ""
    output["prefactor_status_step12"] = np.where(ok, "ok", "unavailable")
    return output


def add_prefactors(
    data: pd.DataFrame,
    prefactor_column: str,
    prefactor_mode: str,
) -> tuple[pd.DataFrame, str]:
    if prefactor_column != "none":
        if prefactor_column not in data.columns:
            raise KeyError(f"--prefactor_column was specified but not found: {prefactor_column}")
        return apply_external_prefactors(data, prefactor_column), "external_prefactor"
    if prefactor_mode == "external_prefactor":
        raise ValueError("--prefactor_mode external_prefactor requires --prefactor_column")
    return apply_empirical_prefactors(data), "empirical_group_baseline"


def regression_metrics(sigma_obs: pd.Series, sigma_pred: pd.Series) -> dict[str, float]:
    obs = pd.to_numeric(sigma_obs, errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(sigma_pred, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if not mask.any():
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "log_mae": math.nan,
            "log_rmse": math.nan,
            "r2_log": math.nan,
            "bias_log": math.nan,
        }
    obs = obs[mask]
    pred = pred[mask]
    residual = pred - obs
    log_error = np.log(pred) - np.log(obs)
    log_true = np.log(obs)
    ss_res = float(np.sum(log_error**2))
    ss_tot = float(np.sum((log_true - np.mean(log_true)) ** 2))
    r2_log = math.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mape": float(np.mean(np.abs(residual) / np.maximum(np.abs(obs), EPS))),
        "log_mae": float(np.mean(np.abs(log_error))),
        "log_rmse": float(np.sqrt(np.mean(log_error**2))),
        "r2_log": r2_log,
        "bias_log": float(np.mean(log_error)),
    }


def first_nonempty(series: pd.Series) -> Any:
    for value in series:
        text = normalize_text(value)
        if text:
            return value
    return series.iloc[0] if len(series) else ""


def build_tau_fit_results(
    data: pd.DataFrame,
    min_points_per_sample: int,
    holdout_fraction: float,
    tau_eff_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    split_role = pd.Series("not_used", index=data.index, dtype="object")
    tau_train_by_sample: dict[str, float] = {}
    log_tau_train_by_sample: dict[str, float] = {}
    holdout_pred = pd.Series(np.nan, index=data.index, dtype="float64")
    holdout_log_error = pd.Series(np.nan, index=data.index, dtype="float64")

    available_mask = (
        data["prefactor_status_step12"].eq("ok")
        & np.isfinite(pd.to_numeric(data["prefactor_C_S_per_m_step12"], errors="coerce"))
        & (pd.to_numeric(data["prefactor_C_S_per_m_step12"], errors="coerce") > 0)
    )

    for sample_key, group in data.groupby("sample_key", sort=True):
        available = group.loc[available_mask.loc[group.index]].copy()
        metadata = {column: first_nonempty(group[column]) for column in METADATA_COLUMNS if column in group.columns}
        n_available = len(available)
        n_unavailable = len(group) - n_available
        fit_note = ""
        if n_available == 0:
            fit_status = "prefactor_unavailable"
            fit_note = "no rows with available prefactor"
        elif n_available < min_points_per_sample:
            fit_status = "insufficient_points"
            fit_note = f"only {n_available} rows with available prefactor"
        else:
            fit_status = "ok"
            fit_note = "fit completed in log space"

        record: dict[str, Any] = {
            "sample_key": sample_key,
            **metadata,
            "n_fit_rows_step12": n_available,
            "n_fit_temperature_points_step12": available["temperature_K"].nunique() if n_available else 0,
            "temperature_min_fit_step12": available["temperature_K"].min() if n_available else "",
            "temperature_max_fit_step12": available["temperature_K"].max() if n_available else "",
            "temperature_span_fit_step12": (
                available["temperature_K"].max() - available["temperature_K"].min() if n_available else ""
            ),
            "n_prefactor_available_rows_step12": n_available,
            "n_prefactor_unavailable_rows_step12": n_unavailable,
            "fit_status_step12": fit_status,
            "fit_note_step12": fit_note,
            "tau_eff_unit_step12": (
                "depends_on_external_prefactor"
                if tau_eff_mode == "external_prefactor"
                else "relative_scale"
            ),
            "tau_eff_mode_step12": tau_eff_mode,
            "prefactor_source_step12": first_nonempty(available["prefactor_source_step12"])
            if n_available
            else first_nonempty(group["prefactor_source_step12"]),
        }

        if fit_status == "ok":
            log_sigma = np.log(available["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
            log_prefactor = np.log(available["prefactor_C_S_per_m_step12"].to_numpy(dtype=float))
            log_tau_values = log_sigma - log_prefactor
            log_tau = float(np.mean(log_tau_values))
            tau = float(np.exp(log_tau))
            pred = available["prefactor_C_S_per_m_step12"].to_numpy(dtype=float) * tau
            metrics = regression_metrics(available["sigma_obs_S_per_m_step11"], pd.Series(pred, index=available.index))
            log_tau_std = float(np.std(log_tau_values, ddof=1)) if len(log_tau_values) > 1 else 0.0
            record.update(
                {
                    "tau_eff_step12": tau,
                    "log_tau_eff_step12": log_tau,
                    "sigma_fit_mae_step12": metrics["mae"],
                    "sigma_fit_rmse_step12": metrics["rmse"],
                    "sigma_fit_mape_step12": metrics["mape"],
                    "sigma_fit_log_mae_step12": metrics["log_mae"],
                    "sigma_fit_log_rmse_step12": metrics["log_rmse"],
                    "sigma_fit_r2_log_step12": metrics["r2_log"],
                    "sigma_fit_bias_log_step12": metrics["bias_log"],
                    "tau_eff_log_std_step12": log_tau_std,
                    "tau_eff_geometric_std_factor_step12": float(np.exp(log_tau_std)),
                }
            )
            holdout_info = holdout_evaluation(
                available, holdout_fraction, split_role, holdout_pred, holdout_log_error
            )
            tau_train_by_sample[sample_key] = holdout_info.get("tau_eff_train_step12", math.nan)
            log_tau_train_by_sample[sample_key] = holdout_info.get(
                "log_tau_eff_train_step12", math.nan
            )
            record.update(holdout_info)
        else:
            record.update(empty_fit_metric_values())
            record.update(empty_holdout_values())
        records.append(record)

    result = pd.DataFrame(records)
    data_with_splits = data.copy()
    data_with_splits["split_role_step12"] = split_role
    data_with_splits["tau_eff_train_step12"] = data_with_splits["sample_key"].map(tau_train_by_sample)
    data_with_splits["log_tau_eff_train_step12"] = data_with_splits["sample_key"].map(
        log_tau_train_by_sample
    )
    data_with_splits["sigma_pred_holdout_S_per_m_step12"] = holdout_pred
    data_with_splits["sigma_holdout_log_error_step12"] = holdout_log_error
    return result, data_with_splits


def empty_fit_metric_values() -> dict[str, Any]:
    return {
        "tau_eff_step12": "",
        "log_tau_eff_step12": "",
        "sigma_fit_mae_step12": "",
        "sigma_fit_rmse_step12": "",
        "sigma_fit_mape_step12": "",
        "sigma_fit_log_mae_step12": "",
        "sigma_fit_log_rmse_step12": "",
        "sigma_fit_r2_log_step12": "",
        "sigma_fit_bias_log_step12": "",
        "tau_eff_log_std_step12": "",
        "tau_eff_geometric_std_factor_step12": "",
    }


def empty_holdout_values() -> dict[str, Any]:
    return {
        "tau_eff_train_step12": "",
        "log_tau_eff_train_step12": "",
        "n_train_rows_step12": 0,
        "n_holdout_rows_step12": 0,
        "sigma_holdout_mae_step12": "",
        "sigma_holdout_rmse_step12": "",
        "sigma_holdout_mape_step12": "",
        "sigma_holdout_log_rmse_step12": "",
        "sigma_holdout_r2_log_step12": "",
        "holdout_eval_status_step12": "not_enough_points",
    }


def holdout_evaluation(
    available: pd.DataFrame,
    holdout_fraction: float,
    split_role: pd.Series,
    holdout_pred: pd.Series,
    holdout_log_error: pd.Series,
) -> dict[str, Any]:
    n_total = len(available)
    holdout_n = int(math.ceil(n_total * holdout_fraction))
    train_n = n_total - holdout_n
    if n_total < 8 or train_n < 5 or holdout_n < 2:
        split_role.loc[available.index] = "fit_all"
        values = empty_holdout_values()
        values["n_train_rows_step12"] = train_n
        values["n_holdout_rows_step12"] = holdout_n
        return values

    sorted_available = available.sort_values(["temperature_K", "sigma_obs_S_per_m_step11"])
    train = sorted_available.iloc[:train_n]
    holdout = sorted_available.iloc[train_n:]
    split_role.loc[train.index] = "train"
    split_role.loc[holdout.index] = "holdout"

    log_tau_train = float(
        np.mean(
            np.log(train["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
            - np.log(train["prefactor_C_S_per_m_step12"].to_numpy(dtype=float))
        )
    )
    tau_train = float(np.exp(log_tau_train))
    pred = holdout["prefactor_C_S_per_m_step12"].to_numpy(dtype=float) * tau_train
    holdout_pred.loc[holdout.index] = pred
    log_errors = np.log(pred) - np.log(holdout["sigma_obs_S_per_m_step11"].to_numpy(dtype=float))
    holdout_log_error.loc[holdout.index] = log_errors
    metrics = regression_metrics(holdout["sigma_obs_S_per_m_step11"], pd.Series(pred, index=holdout.index))
    return {
        "tau_eff_train_step12": tau_train,
        "log_tau_eff_train_step12": log_tau_train,
        "n_train_rows_step12": len(train),
        "n_holdout_rows_step12": len(holdout),
        "sigma_holdout_mae_step12": metrics["mae"],
        "sigma_holdout_rmse_step12": metrics["rmse"],
        "sigma_holdout_mape_step12": metrics["mape"],
        "sigma_holdout_log_rmse_step12": metrics["log_rmse"],
        "sigma_holdout_r2_log_step12": metrics["r2_log"],
        "holdout_eval_status_step12": "ok",
    }


def add_prediction_columns(data: pd.DataFrame, fit_results: pd.DataFrame) -> pd.DataFrame:
    merge_columns = [
        "sample_key",
        "tau_eff_step12",
        "log_tau_eff_step12",
        "tau_eff_unit_step12",
        "tau_eff_mode_step12",
        "fit_status_step12",
        "tau_eff_train_step12",
        "holdout_eval_status_step12",
    ]
    pred = data.merge(fit_results[merge_columns], on="sample_key", how="left")
    ok = pred["fit_status_step12"].eq("ok") & pred["prefactor_status_step12"].eq("ok")
    tau = pd.to_numeric(pred["tau_eff_step12"], errors="coerce")
    prefactor = pd.to_numeric(pred["prefactor_C_S_per_m_step12"], errors="coerce")
    sigma_obs = pd.to_numeric(pred["sigma_obs_S_per_m_step11"], errors="coerce")
    sigma_pred = prefactor * tau
    sigma_pred.loc[~ok] = np.nan
    pred["sigma_pred_S_per_m_step12"] = sigma_pred
    pred["sigma_residual_S_per_m_step12"] = sigma_pred - sigma_obs
    pred["sigma_abs_error_S_per_m_step12"] = (sigma_pred - sigma_obs).abs()
    pred["sigma_relative_error_step12"] = pred["sigma_abs_error_S_per_m_step12"] / np.maximum(
        sigma_obs.abs(), EPS
    )
    log_error = np.log(sigma_pred) - np.log(sigma_obs)
    log_error.loc[~ok | ~np.isfinite(log_error)] = np.nan
    pred["sigma_log_error_step12"] = log_error
    return pred


def build_ready_samples(results: pd.DataFrame) -> pd.DataFrame:
    tau_ok = finite_positive(results["tau_eff_step12"])
    log_rmse = pd.to_numeric(results["sigma_fit_log_rmse_step12"], errors="coerce")
    return results.loc[results["fit_status_step12"].eq("ok") & tau_ok & np.isfinite(log_rmse)].copy()


def problem_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if normalize_text(row.get("fit_status_step12")) != "ok":
        reasons.append(f"fit_status={row.get('fit_status_step12')}")
    if normalize_text(row.get("fit_status_step12")) == "prefactor_unavailable":
        reasons.append("prefactor unavailable")
    if normalize_text(row.get("fit_status_step12")) == "insufficient_points":
        reasons.append("insufficient points")
    for column, label in [
        ("sigma_fit_mape_step12", "large fit MAPE"),
        ("sigma_fit_log_rmse_step12", "large fit log RMSE"),
        ("sigma_holdout_log_rmse_step12", "large holdout log RMSE"),
    ]:
        value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
        if math.isfinite(value) and value > 1.0:
            reasons.append(label)
    return "; ".join(reasons) if reasons else "review recommended"


def build_problem_samples(results: pd.DataFrame) -> pd.DataFrame:
    mape = pd.to_numeric(results["sigma_fit_mape_step12"], errors="coerce")
    log_rmse = pd.to_numeric(results["sigma_fit_log_rmse_step12"], errors="coerce")
    holdout_log = pd.to_numeric(results["sigma_holdout_log_rmse_step12"], errors="coerce")
    problem = (
        ~results["fit_status_step12"].eq("ok")
        | (mape > 1.0)
        | (log_rmse > 1.0)
        | (holdout_log > 1.0)
    )
    output = results.loc[problem].copy()
    output["tau_fit_problem_reason_step12"] = output.apply(problem_reason, axis=1)
    return output


def normalize_flag_count(series: pd.Series) -> int:
    return int(series.map(normalize_bool).sum())


def build_material_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    grouped = results.groupby(["material_system", "n_or_p"], dropna=False, sort=True)
    return grouped.agg(
        sample_count=("sample_key", "count"),
        fit_ok_sample_count=("fit_status_step12", lambda values: int(pd.Series(values).eq("ok").sum())),
        median_tau_eff_step12=("tau_eff_step12", lambda values: pd.to_numeric(values, errors="coerce").median()),
        mean_tau_eff_step12=("tau_eff_step12", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        median_log_tau_eff_step12=(
            "log_tau_eff_step12",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_sigma_fit_log_rmse_step12=(
            "sigma_fit_log_rmse_step12",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_sigma_fit_mape_step12=(
            "sigma_fit_mape_step12",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_sigma_holdout_log_rmse_step12=(
            "sigma_holdout_log_rmse_step12",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        median_temperature_span_fit_step12=(
            "temperature_span_fit_step12",
            lambda values: pd.to_numeric(values, errors="coerce").median(),
        ),
        nanocarbon_sample_count=("nanocarbon_keyword_detected_step9", normalize_flag_count),
        rare_metal_flag_sample_count=("rare_metal_flag_auto_step9", normalize_flag_count),
        toxicity_flag_sample_count=("toxicity_flag_auto_step9", normalize_flag_count),
    ).reset_index()


def first_examples(series: pd.Series) -> str:
    return ";".join(series.dropna().astype(str).drop_duplicates().head(5))


def build_prefactor_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    valid = predictions[
        predictions["prefactor_status_step12"].eq("ok")
        & np.isfinite(pd.to_numeric(predictions["prefactor_C_S_per_m_step12"], errors="coerce"))
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "prefactor_group_key_step12",
                "temperature_bin_K_step12",
                "prefactor_source_step12",
                "row_count",
                "sample_count",
                "median_prefactor_C_S_per_m_step12",
                "min_prefactor_C_S_per_m_step12",
                "max_prefactor_C_S_per_m_step12",
                "example_sample_keys",
            ]
        )
    valid["prefactor_C_S_per_m_step12"] = pd.to_numeric(
        valid["prefactor_C_S_per_m_step12"], errors="coerce"
    )
    return valid.groupby(
        ["prefactor_group_key_step12", "temperature_bin_K_step12", "prefactor_source_step12"],
        dropna=False,
        sort=True,
    ).agg(
        row_count=("sample_key", "count"),
        sample_count=("sample_key", pd.Series.nunique),
        median_prefactor_C_S_per_m_step12=("prefactor_C_S_per_m_step12", "median"),
        min_prefactor_C_S_per_m_step12=("prefactor_C_S_per_m_step12", "min"),
        max_prefactor_C_S_per_m_step12=("prefactor_C_S_per_m_step12", "max"),
        example_sample_keys=("sample_key", first_examples),
    ).reset_index()


def build_holdout_eval(results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_key",
        "material_system",
        "n_or_p",
        "n_fit_rows_step12",
        "n_train_rows_step12",
        "n_holdout_rows_step12",
        "tau_eff_train_step12",
        "sigma_holdout_mae_step12",
        "sigma_holdout_rmse_step12",
        "sigma_holdout_mape_step12",
        "sigma_holdout_log_rmse_step12",
        "sigma_holdout_r2_log_step12",
        "holdout_eval_status_step12",
    ]
    ensure_columns(results, columns)
    return results.loc[:, columns].copy()


def describe_series(series: pd.Series, prefix: str) -> list[tuple[str, str]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return [
            (f"{prefix}_count", "0"),
            (f"{prefix}_mean", ""),
            (f"{prefix}_median", ""),
            (f"{prefix}_p25", ""),
            (f"{prefix}_p75", ""),
            (f"{prefix}_p90", ""),
            (f"{prefix}_max", ""),
        ]
    return [
        (f"{prefix}_count", str(int(values.count()))),
        (f"{prefix}_mean", str(float(values.mean()))),
        (f"{prefix}_median", str(float(values.median()))),
        (f"{prefix}_p25", str(float(values.quantile(0.25)))),
        (f"{prefix}_p75", str(float(values.quantile(0.75)))),
        (f"{prefix}_p90", str(float(values.quantile(0.90)))),
        (f"{prefix}_max", str(float(values.max()))),
    ]


def value_counts_rows(prefix: str, series: pd.Series, top_n: int | None = None) -> list[tuple[str, str]]:
    counts = series.fillna("").astype(str).value_counts().sort_values(ascending=False)
    if top_n is not None:
        counts = counts.head(top_n)
    return [(f"{prefix}_{key}_count", str(int(value))) for key, value in counts.items()]


def build_report(
    input_counts: dict[str, int],
    fit_data_rows: int,
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    ready: pd.DataFrame,
    problem: pd.DataFrame,
    args: argparse.Namespace,
    tau_eff_mode: str,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = [
        ("input_initial_tau_fit_training_normalized_step11_rows", str(input_counts["initial"])),
        ("input_training_dataset_normalized_step11_rows", str(input_counts["training"])),
        ("fit_data_step12_rows", str(fit_data_rows)),
        ("sigma_predictions_step12_rows", str(len(predictions))),
        ("tau_fit_results_step12_sample_count", str(len(results))),
        ("tau_fit_ready_samples_step12_sample_count", str(len(ready))),
        ("tau_fit_problem_samples_step12_sample_count", str(len(problem))),
        ("tau_eff_mode_step12", tau_eff_mode),
        ("prefactor_mode", args.prefactor_mode),
        ("temperature_bin_width_K", str(args.temperature_bin_width_K)),
        ("min_points_per_sample", str(args.min_points_per_sample)),
        ("holdout_fraction", str(args.holdout_fraction)),
    ]
    fit_status_counts = results["fit_status_step12"].fillna("").astype(str).value_counts()
    for status in ["ok", "insufficient_points", "prefactor_unavailable"]:
        rows.append((f"fit_status_step12_{status}_count", str(int(fit_status_counts.get(status, 0)))))
    holdout_status_counts = results["holdout_eval_status_step12"].fillna("").astype(str).value_counts()
    for status in ["ok", "not_enough_points"]:
        rows.append(
            (f"holdout_eval_status_step12_{status}_count", str(int(holdout_status_counts.get(status, 0))))
        )
    rows.extend(
        value_counts_rows(
            "n_or_p_fit_ok",
            results.loc[results["fit_status_step12"].eq("ok"), "n_or_p"],
        )
    )
    rows.extend(
        value_counts_rows(
            "material_system_fit_ok",
            results.loc[results["fit_status_step12"].eq("ok"), "material_system"],
            top_n=20,
        )
    )
    rows.extend(describe_series(results["sigma_fit_log_rmse_step12"], "sigma_fit_log_rmse_step12"))
    rows.extend(describe_series(results["sigma_fit_mape_step12"], "sigma_fit_mape_step12"))
    rows.extend(
        describe_series(results["sigma_holdout_log_rmse_step12"], "sigma_holdout_log_rmse_step12")
    )
    rows.extend(describe_series(results["sigma_holdout_mape_step12"], "sigma_holdout_mape_step12"))
    rows.extend(describe_series(results["tau_eff_step12"], "tau_eff_step12"))
    rows.extend(describe_series(results["log_tau_eff_step12"], "log_tau_eff_step12"))
    rows.extend(value_counts_rows("prefactor_source_step12", predictions["prefactor_source_step12"]))
    rows.extend(value_counts_rows("prefactor_status_step12", predictions["prefactor_status_step12"]))

    rows.extend(
        [
            (
                "sintering_method_unknown_rows",
                str(
                    int(
                        predictions["sintering_method"]
                        .fillna("")
                        .astype(str)
                        .str.casefold()
                        .eq("unknown")
                        .sum()
                    )
                ),
            ),
            (
                "sintering_checked_no_rows",
                str(
                    int(
                        predictions["sintering_checked"]
                        .fillna("")
                        .astype(str)
                        .str.casefold()
                        .eq("no")
                        .sum()
                    )
                ),
            ),
            (
                "record_checked_no_rows",
                str(
                    int(
                        predictions["record_checked"]
                        .fillna("")
                        .astype(str)
                        .str.casefold()
                        .eq("no")
                        .sum()
                    )
                ),
            ),
            ("n_or_p_changed_rows", "0"),
            ("note", "Step12 did not predict Seebeck coefficient, thermal conductivity, or ZT."),
            ("note", "Step12 fit tau_eff from electrical conductivity and calculated sigma_pred."),
            (
                "note",
                "Default tau_eff is a relative scalar against an empirical baseline, not a physical relaxation time in seconds.",
            ),
            ("note", "A physical relaxation time requires an external C(T) or transport prefactor."),
        ]
    )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    return "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n", report_df


def selected_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ensure_columns(df, columns)
    remaining = [column for column in df.columns if column not in columns]
    return df.loc[:, [*columns, *remaining]]


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
        "tau_fit_results": "tau_fit_results_step12.csv",
        "initial_tau_fit_predictions": "initial_tau_fit_predictions_step12.csv",
        "tau_fit_ready_samples": "tau_fit_ready_samples_step12.csv",
        "tau_fit_problem_samples": "tau_fit_problem_samples_step12.csv",
        "tau_fit_material_summary": "tau_fit_material_summary_step12.csv",
        "prefactor_baseline_audit": "prefactor_baseline_audit_step12.csv",
        "holdout_eval": "tau_fit_holdout_eval_step12.csv",
    }
    with pd.ExcelWriter(output_dir / "starrydata2_step12_tau_fit.xlsx", engine="openpyxl") as writer:
        for sheet_name, filename in sheet_files.items():
            frame = read_csv_text(output_dir / filename, nrows=EXCEL_PREVIEW_ROWS)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)
        report_df.to_excel(writer, sheet_name="tau_fit_report", index=False)
        fit_worksheet(writer, "tau_fit_report", report_df)


def assert_acceptance(
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    ready: pd.DataFrame,
    problem: pd.DataFrame,
    material_summary: pd.DataFrame,
) -> None:
    if results["sample_key"].duplicated().any():
        raise ValueError("tau_fit_results_step12.csv is not one row per sample_key")
    for column in [
        "sample_key",
        "tau_eff_step12",
        "tau_eff_mode_step12",
        "tau_eff_unit_step12",
        "sigma_fit_log_rmse_step12",
        "sigma_fit_mape_step12",
        "fit_status_step12",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in results.columns:
            raise KeyError(f"tau_fit_results_step12 missing {column}")
    for column in [
        "sigma_obs_S_per_m_step11",
        "sigma_pred_S_per_m_step12",
        "tau_eff_step12",
        "sigma_log_error_step12",
        "prefactor_C_S_per_m_step12",
    ]:
        if column not in predictions.columns:
            raise KeyError(f"sigma_predictions_step12 missing {column}")
    if not ready["fit_status_step12"].eq("ok").all():
        raise ValueError("tau_fit_ready_samples_step12 contains non-ok samples")
    if "tau_fit_problem_reason_step12" not in problem.columns:
        raise KeyError("tau_fit_problem_samples_step12 missing tau_fit_problem_reason_step12")
    for column in ["material_system", "n_or_p", "median_tau_eff_step12"]:
        if column not in material_summary.columns:
            raise KeyError(f"tau_fit_material_summary_step12 missing {column}")
    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if not predictions[column].fillna("").astype(str).str.casefold().eq(expected).all():
            raise ValueError(f"{column} changed from expected {expected}")


def main() -> None:
    args = parse_args()
    paths = input_paths(args.step11_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_counts = {label: count_csv_rows(path) for label, path in paths.items()}

    initial = read_csv_text(paths["initial"])
    validate_initial(initial)
    fit_data = prepare_fit_data(initial, args.temperature_bin_width_K)
    fit_data, tau_eff_mode = add_prefactors(fit_data, args.prefactor_column, args.prefactor_mode)
    fit_results, fit_data_with_splits = build_tau_fit_results(
        fit_data, args.min_points_per_sample, args.holdout_fraction, tau_eff_mode
    )
    predictions = add_prediction_columns(fit_data_with_splits, fit_results)

    prediction_columns = [
        "sample_key",
        "temperature_K",
        "temperature_bin_K_step12",
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
        "sigma_obs_S_per_m_step11",
        "prefactor_C_S_per_m_step12",
        "prefactor_source_step12",
        "prefactor_group_key_step12",
        "prefactor_status_step12",
        "tau_eff_step12",
        "sigma_pred_S_per_m_step12",
        "sigma_residual_S_per_m_step12",
        "sigma_abs_error_S_per_m_step12",
        "sigma_relative_error_step12",
        "sigma_log_error_step12",
        "split_role_step12",
        "tau_eff_train_step12",
        "sigma_pred_holdout_S_per_m_step12",
        "sigma_holdout_log_error_step12",
        "seebeck_obs_V_per_K_step11",
        "kappa_obs_W_per_mK_step11",
        "zt_obs_dimensionless_step11",
        "power_factor_obs_W_per_mK2_step11",
        "zt_calc_from_obs_step11",
        "can_eval_power_factor_step11",
        "can_calc_zt_from_obs_step11",
        "can_compare_zt_obs_step11",
        "fit_status_step12",
    ]
    predictions = selected_columns(predictions, prediction_columns)
    initial_predictions = predictions[predictions["fit_status_step12"].eq("ok")].copy()

    result_columns = [
        "sample_key",
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
        "fitting_source_preference_step8",
        "fitting_source_actual_step10",
        "sigma_obs_source_step11",
        "tau_eff_step12",
        "log_tau_eff_step12",
        "tau_eff_unit_step12",
        "tau_eff_mode_step12",
        "n_fit_rows_step12",
        "n_fit_temperature_points_step12",
        "temperature_min_fit_step12",
        "temperature_max_fit_step12",
        "temperature_span_fit_step12",
        "sigma_fit_mae_step12",
        "sigma_fit_rmse_step12",
        "sigma_fit_mape_step12",
        "sigma_fit_log_mae_step12",
        "sigma_fit_log_rmse_step12",
        "sigma_fit_r2_log_step12",
        "sigma_fit_bias_log_step12",
        "tau_eff_log_std_step12",
        "tau_eff_geometric_std_factor_step12",
        "tau_eff_train_step12",
        "n_train_rows_step12",
        "n_holdout_rows_step12",
        "sigma_holdout_mae_step12",
        "sigma_holdout_rmse_step12",
        "sigma_holdout_mape_step12",
        "sigma_holdout_log_rmse_step12",
        "sigma_holdout_r2_log_step12",
        "holdout_eval_status_step12",
        "fit_status_step12",
        "fit_note_step12",
        "prefactor_source_step12",
    ]
    fit_results = selected_columns(fit_results, result_columns)
    ready = build_ready_samples(fit_results)
    problem = build_problem_samples(fit_results)
    material_summary = build_material_summary(fit_results)
    prefactor_audit = build_prefactor_audit(predictions)
    holdout_eval = build_holdout_eval(fit_results)

    assert_acceptance(fit_results, predictions, ready, problem, material_summary)

    write_csv(fit_results, args.output_dir / "tau_fit_results_step12.csv")
    write_csv(predictions, args.output_dir / "sigma_predictions_step12.csv")
    write_csv(initial_predictions, args.output_dir / "initial_tau_fit_predictions_step12.csv")
    write_csv(ready, args.output_dir / "tau_fit_ready_samples_step12.csv")
    write_csv(problem, args.output_dir / "tau_fit_problem_samples_step12.csv")
    write_csv(material_summary, args.output_dir / "tau_fit_material_summary_step12.csv")
    write_csv(prefactor_audit, args.output_dir / "prefactor_baseline_audit_step12.csv")
    write_csv(holdout_eval, args.output_dir / "tau_fit_holdout_eval_step12.csv")

    excel_notes: list[str] = []
    for sheet_name, row_count in [
        ("tau_fit_results", len(fit_results)),
        ("initial_tau_fit_predictions", len(initial_predictions)),
        ("tau_fit_ready_samples", len(ready)),
        ("tau_fit_problem_samples", len(problem)),
        ("tau_fit_material_summary", len(material_summary)),
        ("prefactor_baseline_audit", len(prefactor_audit)),
        ("holdout_eval", len(holdout_eval)),
    ]:
        add_excel_preview_note(sheet_name, row_count, excel_notes)

    report_text, report_df = build_report(
        input_counts,
        len(fit_data),
        predictions,
        fit_results,
        ready,
        problem,
        args,
        tau_eff_mode,
        excel_notes,
    )
    (args.output_dir / "step12_tau_fit_report.txt").write_text(report_text, encoding="utf-8")
    write_excel_output(args.output_dir, report_df)

    median_log_rmse = pd.to_numeric(fit_results["sigma_fit_log_rmse_step12"], errors="coerce").median()
    median_mape = pd.to_numeric(fit_results["sigma_fit_mape_step12"], errors="coerce").median()
    median_holdout_log_rmse = pd.to_numeric(
        fit_results["sigma_holdout_log_rmse_step12"], errors="coerce"
    ).median()
    median_holdout_mape = pd.to_numeric(
        fit_results["sigma_holdout_mape_step12"], errors="coerce"
    ).median()
    fit_ok = fit_results["fit_status_step12"].eq("ok")
    n_fit_ok = int((fit_ok & fit_results["n_or_p"].astype(str).str.casefold().eq("n")).sum())
    p_fit_ok = int((fit_ok & fit_results["n_or_p"].astype(str).str.casefold().eq("p")).sum())
    sintering_changed_rows = int(
        (
            ~predictions["sintering_method"].astype(str).str.casefold().eq("unknown")
            | ~predictions["sintering_checked"].astype(str).str.casefold().eq("no")
            | ~predictions["record_checked"].astype(str).str.casefold().eq("no")
        ).sum()
    )

    print("Done.")
    print("Created:")
    print("- tau_fit_results_step12.csv")
    print("- sigma_predictions_step12.csv")
    print("- initial_tau_fit_predictions_step12.csv")
    print("- tau_fit_ready_samples_step12.csv")
    print("- tau_fit_problem_samples_step12.csv")
    print("- tau_fit_material_summary_step12.csv")
    print("- prefactor_baseline_audit_step12.csv")
    print("- tau_fit_holdout_eval_step12.csv")
    print("- step12_tau_fit_report.txt")
    print("- starrydata2_step12_tau_fit.xlsx")
    print("")
    print("Summary:")
    print(f"fit data rows: {len(fit_data)}")
    print(f"samples fitted: {len(fit_results)}")
    print(f"fit ok samples: {int(fit_ok.sum())}")
    print(f"problem samples: {len(problem)}")
    print(f"sigma prediction rows: {len(predictions)}")
    print(f"tau_eff mode: {tau_eff_mode}")
    print(f"median sigma_fit_log_rmse: {median_log_rmse}")
    print(f"median sigma_fit_mape: {median_mape}")
    print(f"median holdout_log_rmse: {median_holdout_log_rmse}")
    print(f"median holdout_mape: {median_holdout_mape}")
    print(f"n fit ok samples: {n_fit_ok}")
    print(f"p fit ok samples: {p_fit_ok}")
    print(f"sintering changed rows: {sintering_changed_rows}")
    print("n/p changed rows: 0")


if __name__ == "__main__":
    main()
