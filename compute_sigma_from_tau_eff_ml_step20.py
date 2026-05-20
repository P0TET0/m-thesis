import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP19_DIR = "data/output/starrydata2_step19_tau_eff_ml_model"
DEFAULT_STEP12_DIR = "data/output/starrydata2_step12_tau_fit"
DEFAULT_STEP18_DIR = "data/output/starrydata2_step18_tau_eff_ml_dataset"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step20_sigma_ml_prediction"
EXCEL_PREVIEW_ROWS = 100_000
EPS = 1e-12

STRING_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "composition",
    "material_system",
    "n_or_p",
    "model_name",
    "split_name",
    "split_role",
]

SIGMA_BASE_COLUMNS = [
    "sample_key",
    "temperature_K",
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
    "prefactor_C_S_per_m_step12",
    "sigma_obs_S_per_m_step11",
    "sigma_pred_S_per_m_step12",
    "sigma_relative_error_step12",
    "sigma_log_error_step12",
    "tau_eff_step12",
    "tau_eff_unit_step12",
    "tau_eff_mode_step12",
    "fit_status_step12",
    "fitting_source_actual_step10",
    "sigma_obs_source_step11",
    "seebeck_obs_V_per_K_step11",
    "kappa_obs_W_per_mK_step11",
    "zt_obs_dimensionless_step11",
    "zt_calc_from_obs_step11",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Step20 sigma_pred_ML from Step19 tau_eff predictions.")
    parser.add_argument("--step19_dir", default=DEFAULT_STEP19_DIR)
    parser.add_argument("--step12_dir", default=DEFAULT_STEP12_DIR)
    parser.add_argument("--step18_dir", default=DEFAULT_STEP18_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary_split", default="split_doi_group_80_20_step18")
    parser.add_argument("--selected_model_name", default="auto")
    parser.add_argument(
        "--evaluation_model_strategy",
        default="primary_split_best",
        choices=["primary_split_best", "step19_selected", "both"],
    )
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def read_csv(path, required=True, usecols=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return None
    kwargs = {"dtype": dtype_for_existing(path), "low_memory": False}
    if usecols is not None:
        header = pd.read_csv(path, nrows=0)
        kwargs["usecols"] = [c for c in usecols if c in header.columns]
    return pd.read_csv(path, **kwargs)


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def find_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def finite_positive(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0)


def safe_log(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.where(np.isfinite(values) & (values > 0), np.log(values), np.nan)


def r2_log(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    ok = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    if ok.sum() < 2:
        return np.nan
    yt = y_true_log[ok]
    yp = y_pred_log[ok]
    denom = np.sum((yt - yt.mean()) ** 2)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.sum((yt - yp) ** 2) / denom)


def sigma_metrics(df):
    ok = df["sigma_ML_error_status_step20"].eq("ok")
    sub = df[ok].copy()
    if sub.empty:
        return empty_metrics()
    obs = pd.to_numeric(sub["sigma_obs_S_per_m_step11"], errors="coerce")
    pred = pd.to_numeric(sub["sigma_pred_ML_S_per_m_step20"], errors="coerce")
    log_err = pd.to_numeric(sub["sigma_ML_log_error_step20"], errors="coerce")
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    log_err = log_err[valid]
    if len(obs) == 0:
        return empty_metrics()
    rel = np.abs(pred - obs) / np.maximum(np.abs(obs), EPS)
    ratio = pred / obs
    true_log = np.log(obs)
    pred_log = np.log(pred)
    sample_count = int(sub.loc[valid, "sample_key"].nunique()) if "sample_key" in sub.columns else np.nan
    return {
        "n_rows": int(len(obs)),
        "n_samples": sample_count,
        "sigma_mae_step20": float(np.mean(np.abs(pred - obs))),
        "sigma_rmse_step20": float(math.sqrt(np.mean((pred - obs) ** 2))),
        "sigma_mape_step20": float(np.mean(rel)),
        "sigma_log_mae_step20": float(np.mean(np.abs(log_err))),
        "sigma_log_rmse_step20": float(math.sqrt(np.mean(log_err**2))),
        "sigma_r2_log_step20": r2_log(true_log, pred_log),
        "sigma_bias_log_step20": float(np.mean(log_err)),
        "within_25pct_rate_step20": float(np.mean(rel <= 0.25)),
        "within_50pct_rate_step20": float(np.mean(rel <= 0.50)),
        "within_factor_2_rate_step20": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))),
    }


def empty_metrics():
    return {
        "n_rows": 0,
        "n_samples": 0,
        "sigma_mae_step20": np.nan,
        "sigma_rmse_step20": np.nan,
        "sigma_mape_step20": np.nan,
        "sigma_log_mae_step20": np.nan,
        "sigma_log_rmse_step20": np.nan,
        "sigma_r2_log_step20": np.nan,
        "sigma_bias_log_step20": np.nan,
        "within_25pct_rate_step20": np.nan,
        "within_50pct_rate_step20": np.nan,
        "within_factor_2_rate_step20": np.nan,
    }


def fitting_metrics(df):
    if "sigma_pred_S_per_m_step12" not in df.columns:
        return {"fitting_sigma_log_rmse": np.nan, "fitting_sigma_mape": np.nan}
    obs = pd.to_numeric(df["sigma_obs_S_per_m_step11"], errors="coerce")
    pred = pd.to_numeric(df["sigma_pred_S_per_m_step12"], errors="coerce")
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    if ok.sum() == 0:
        return {"fitting_sigma_log_rmse": np.nan, "fitting_sigma_mape": np.nan}
    log_err = np.log(pred[ok]) - np.log(obs[ok])
    rel = np.abs(pred[ok] - obs[ok]) / np.maximum(np.abs(obs[ok]), EPS)
    return {
        "fitting_sigma_log_rmse": float(math.sqrt(np.mean(log_err**2))),
        "fitting_sigma_mape": float(np.mean(rel)),
    }


def add_sigma_calculations(df, tau_col, log_col):
    pref = pd.to_numeric(df["prefactor_C_S_per_m_step12"], errors="coerce")
    tau = pd.to_numeric(df[tau_col], errors="coerce")
    obs = pd.to_numeric(df["sigma_obs_S_per_m_step11"], errors="coerce")
    df["sigma_pred_ML_S_per_m_step20"] = pref * tau
    status = np.full(len(df), "ok", dtype=object)
    status[pref.isna()] = "missing_prefactor"
    status[np.isfinite(pref) & (pref <= 0)] = "invalid_prefactor"
    status[tau.isna()] = "missing_tau_eff_pred"
    status[np.isfinite(tau) & (tau <= 0)] = "invalid_tau_eff_pred"
    status[~np.isfinite(df["sigma_pred_ML_S_per_m_step20"])] = "invalid_tau_eff_pred"
    df["sigma_pred_ML_status_step20"] = status
    df["sigma_pred_ML_note_step20"] = np.where(status == "ok", "ok", status)

    pred = pd.to_numeric(df["sigma_pred_ML_S_per_m_step20"], errors="coerce")
    df["sigma_ML_residual_S_per_m_step20"] = pred - obs
    df["sigma_ML_abs_error_S_per_m_step20"] = (pred - obs).abs()
    df["sigma_ML_relative_error_step20"] = (pred - obs).abs() / np.maximum(obs.abs(), EPS)
    valid_log = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)
    df["sigma_ML_log_error_step20"] = np.where(valid_log, np.log(pred) - np.log(obs), np.nan)
    df["sigma_ML_error_status_step20"] = np.where(valid_log & (status == "ok"), "ok", "not_evaluated")

    fit_pred = pd.to_numeric(df.get("sigma_pred_S_per_m_step12"), errors="coerce")
    df["sigma_fitting_vs_ML_abs_diff_step20"] = (fit_pred - pred).abs()
    df["sigma_fitting_vs_ML_relative_diff_step20"] = (fit_pred - pred).abs() / np.maximum(fit_pred.abs(), EPS)
    valid_fit = np.isfinite(fit_pred) & np.isfinite(pred) & (fit_pred > 0) & (pred > 0)
    df["sigma_fitting_vs_ML_log_diff_step20"] = np.where(valid_fit, np.log(pred) - np.log(fit_pred), np.nan)
    return df


def determine_models(model_comparison, selected_summary, primary_split, selected_arg):
    selected_model = selected_arg
    if selected_model == "auto":
        if "selected_model_name" in selected_summary.columns:
            selected_model = str(selected_summary["selected_model_name"].iloc[0])
        elif "best_model_name" in selected_summary.columns:
            selected_model = str(selected_summary["best_model_name"].iloc[0])
        else:
            selected_model = ""

    rmse_col = find_col(model_comparison, ["log_tau_rmse", "rmse_log_tau", "test_rmse_log_tau"])
    split_col = "split_name" if "split_name" in model_comparison.columns else "primary_split_name"
    role_col = "split_role" if "split_role" in model_comparison.columns else None
    primary = model_comparison[model_comparison[split_col].eq(primary_split)].copy()
    if role_col:
        primary = primary[primary[role_col].eq("test")]
    primary = primary[pd.to_numeric(primary[rmse_col], errors="coerce").notna()]
    best_model = str(primary.sort_values(rmse_col).iloc[0]["model_name"])
    baseline = primary[primary["model_name"].eq("baseline_mean")]
    selected = primary[primary["model_name"].eq(selected_model)]
    best = primary[primary["model_name"].eq(best_model)]
    baseline_rmse = float(baseline[rmse_col].iloc[0]) if len(baseline) else np.nan
    selected_rmse = float(selected[rmse_col].iloc[0]) if len(selected) else np.nan
    best_rmse = float(best[rmse_col].iloc[0]) if len(best) else np.nan
    return {
        "step19_selected_model_name_step20": selected_model,
        "primary_split_best_model_name_step20": best_model,
        "baseline_primary_rmse_step20": baseline_rmse,
        "selected_model_primary_rmse_step20": selected_rmse,
        "primary_best_model_rmse_step20": best_rmse,
        "selected_model_worse_than_baseline_step20": bool(np.isfinite(selected_rmse) and np.isfinite(baseline_rmse) and selected_rmse > baseline_rmse),
    }


def quality_from_metrics(row):
    rmse = row.get("sigma_ML_log_rmse_step20")
    mape = row.get("sigma_ML_mape_step20")
    if pd.isna(rmse) or pd.isna(mape):
        return "not_evaluated"
    if rmse <= 0.2 and mape <= 0.2:
        return "excellent"
    if rmse <= 0.4 and mape <= 0.5:
        return "good"
    if rmse <= 0.8 and mape <= 1.0:
        return "moderate"
    return "poor"


def problem_reason(row):
    reasons = ["large sigma ML error"]
    if row.get("sigma_fitting_log_rmse_step12", np.nan) < row.get("sigma_ML_log_rmse_step20", np.nan):
        reasons.append("direct fitting much better than ML")
    if str(row.get("material_system", "")).lower() == "unknown":
        reasons.append("material system poorly represented")
    reasons.append("tau_eff ML prediction error")
    return "; ".join(dict.fromkeys(reasons))


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"report": data.splitlines()})
            data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(len(v) for v in values) + 2, 60)


def make_notes():
    return """# Step20 sigma_pred_ML Notes

## Purpose
Compute machine-learning-derived electrical conductivity from Step19 tau_eff predictions.

## Formula
`sigma_pred_ML(T) = prefactor_C(T) * tau_eff_pred`.

## Inputs
The calculation uses Step19 tau_eff predictions and Step12 `prefactor_C_S_per_m_step12` temperature rows.

## Evaluation Data
The primary evaluation uses the DOI group split test rows. Final all-samples predictions are downstream-only.

## Selected Model vs DOI-best Model
Step20 distinguishes the Step19 selected model from the model with the best DOI split test RMSE.

## Main Results
See `step20_sigma_ml_report.txt`.

## Comparison with Direct Fitting
Direct fitting uses sigma_obs to fit tau_eff, so it is expected to outperform material-feature ML prediction.

## Important Caveats
Step20 does not calculate PF or ZT.
Step20 uses tau_eff predicted by ML to calculate sigma_pred_ML.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Direct fitting performance is expected to be better than ML prediction because it uses sigma_obs directly.
Final all-samples predictions are for downstream screening, not unbiased model evaluation.

## Next Step
Step21 should compute PF_pred_ML and ZT_pred_ML from `sigma_ml_downstream_ready_step20.csv`.
"""


def make_report(input_counts, output_counts, model_info, recommended_model, model_comparison, vs_fitting, material_summary, np_summary, problem_samples, np_changed, sintering_changed):
    lines = []
    lines.append("Step20 sigma ML report")
    lines.append("")
    lines.append(f"Input tau_eff_ml_predictions_step19 rows: {input_counts['tau_predictions']}")
    lines.append(f"Input sigma_predictions_step12 rows: {input_counts['sigma_step12']}")
    lines.append("")
    for key, value in output_counts.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append(f"Step19 selected model: {model_info['step19_selected_model_name_step20']}")
    lines.append(f"Primary DOI split best model: {model_info['primary_split_best_model_name_step20']}")
    lines.append(f"Recommended sigma evaluation model: {recommended_model}")
    lines.append(f"Selected model worse than baseline on DOI split: {'yes' if model_info['selected_model_worse_than_baseline_step20'] else 'no'}")
    lines.append("")
    lines.append("Primary DOI test sigma performance:")
    for _, row in model_comparison.sort_values("primary_test_sigma_log_rmse_step20").iterrows():
        lines.append(
            f"- {row['model_name']}: log_RMSE={row['primary_test_sigma_log_rmse_step20']}, "
            f"MAPE={row['primary_test_sigma_mape_step20']}, R2_log={row['primary_test_sigma_r2_log_step20']}, "
            f"within_factor_2={row['primary_test_within_factor_2_rate_step20']}"
        )
    lines.append("")
    baseline = model_comparison[model_comparison["model_name"].eq("baseline_mean")]
    rec = model_comparison[model_comparison["model_name"].eq(recommended_model)]
    baseline_rmse = baseline["primary_test_sigma_log_rmse_step20"].iloc[0] if len(baseline) else np.nan
    rec_rmse = rec["primary_test_sigma_log_rmse_step20"].iloc[0] if len(rec) else np.nan
    lines.append("Baseline comparison:")
    lines.append(f"- baseline sigma log RMSE: {baseline_rmse}")
    lines.append(f"- recommended model sigma log RMSE: {rec_rmse}")
    lines.append(f"- improvement over baseline: {baseline_rmse - rec_rmse if pd.notna(baseline_rmse) and pd.notna(rec_rmse) else np.nan}")
    lines.append("")
    rec_fit = vs_fitting[vs_fitting["model_name"].eq(recommended_model)]
    if len(rec_fit):
        row = rec_fit.iloc[0]
        lines.append("Fitting version comparison:")
        lines.append(f"- Step12 fitting sigma log RMSE: {row['fitting_sigma_log_rmse']}")
        lines.append(f"- Step20 ML sigma log RMSE: {row['ML_sigma_log_rmse']}")
        lines.append(f"- ML vs fitting gap: {row['ML_vs_fitting_log_rmse_gap']}")
    lines.append("")
    lines.append("Material summary:")
    for _, row in material_summary.sort_values("median_sigma_ML_mape_step20").head(20).iterrows():
        lines.append(f"- {row['material_system']} / {row['n_or_p']} / {row['model_name']}: median MAPE={row['median_sigma_ML_mape_step20']}, poor={row['poor_sample_count']}")
    lines.append("")
    lines.append("n/p summary:")
    for _, row in np_summary.iterrows():
        lines.append(f"- {row['n_or_p']} / {row['model_name']}: samples={row['sample_count']}, median MAPE={row['median_sigma_ML_mape_step20']}")
    lines.append("")
    lines.append("problem samples:")
    lines.append(f"- sigma_ml_problem_samples_step20 count: {len(problem_samples)}")
    lines.append(f"- needs_review_for_ML_sigma_step20 count: {int(problem_samples['needs_review_for_ML_sigma_step20'].astype(bool).sum()) if len(problem_samples) else 0}")
    lines.append("")
    lines.append(f"n/p changed rows: {np_changed}")
    lines.append(f"sintering changed rows: {sintering_changed}")
    lines.append("")
    lines.append("Notes:")
    lines.append("Step20 did not calculate PF or ZT.")
    lines.append("Step20 calculated only sigma_pred_ML from Step19 tau_eff_pred.")
    lines.append("Final all-samples predictions are for downstream screening and not unbiased evaluation.")
    lines.append("tau_eff is relative scale, not physical seconds.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    step19_dir = Path(args.step19_dir)
    step12_dir = Path(args.step12_dir)
    step18_dir = Path(args.step18_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tau_pred = read_csv(step19_dir / "tau_eff_ml_predictions_step19.csv")
    model_comp19 = read_csv(step19_dir / "tau_eff_ml_model_comparison_step19.csv")
    selected_summary19 = read_csv(step19_dir / "tau_eff_ml_selected_model_summary_step19.csv")
    final_tau_pred = read_csv(step19_dir / "tau_eff_ml_final_all_samples_predictions_step19.csv")
    tau_fit_results = read_csv(step12_dir / "tau_fit_results_step12.csv")
    metadata18 = read_csv(step18_dir / "tau_eff_ml_metadata_step18.csv", required=False)

    sigma_header = pd.read_csv(step12_dir / "sigma_predictions_step12.csv", nrows=0)
    sigma_cols = [c for c in SIGMA_BASE_COLUMNS if c in sigma_header.columns]
    sigma_step12 = read_csv(step12_dir / "sigma_predictions_step12.csv", usecols=sigma_cols)

    require_columns(tau_pred, ["sample_key", "model_name", "split_name", "split_role"], "tau_eff_ml_predictions_step19.csv")
    require_columns(final_tau_pred, ["sample_key", "pred_log_tau_eff_final_model_step19", "pred_tau_eff_final_model_step19", "final_model_name_step19"], "tau_eff_ml_final_all_samples_predictions_step19.csv")
    require_columns(sigma_step12, ["sample_key", "temperature_K", "sigma_obs_S_per_m_step11", "prefactor_C_S_per_m_step12"], "sigma_predictions_step12.csv")

    input_counts = {
        "tau_predictions": len(tau_pred),
        "final_tau_predictions": len(final_tau_pred),
        "sigma_step12": len(sigma_step12),
    }

    model_info = determine_models(model_comp19, selected_summary19, args.primary_split, args.selected_model_name)
    selected_model = model_info["step19_selected_model_name_step20"]
    primary_best_model = model_info["primary_split_best_model_name_step20"]
    if args.evaluation_model_strategy == "step19_selected":
        recommended_model = selected_model
        evaluation_models = [selected_model]
    elif args.evaluation_model_strategy == "both":
        evaluation_models = list(dict.fromkeys([selected_model, primary_best_model]))
        recommended_model = primary_best_model
    else:
        recommended_model = primary_best_model
        evaluation_models = [primary_best_model]

    log_col = find_col(tau_pred, ["pred_log_tau_eff_step19", "log_tau_eff_pred_step19"])
    tau_col = find_col(tau_pred, ["pred_tau_eff_step19", "tau_eff_pred_step19"])
    if log_col is None and tau_col is None:
        raise ValueError("No tau_eff prediction columns found in Step19 predictions.")
    tau_pred["log_tau_eff_pred_ml_step20"] = pd.to_numeric(tau_pred[log_col], errors="coerce") if log_col else safe_log(tau_pred[tau_col])
    tau_pred["tau_eff_pred_ml_step20"] = pd.to_numeric(tau_pred[tau_col], errors="coerce") if tau_col else np.exp(tau_pred["log_tau_eff_pred_ml_step20"])
    tau_pred["tau_eff_prediction_source_step20"] = f"step19:{tau_col or log_col}"

    tau_keep_cols = [
        "sample_key",
        "model_name",
        "split_name",
        "split_role",
        "target_log_tau_eff_step18",
        "target_tau_eff_step18",
        "pred_log_tau_eff_step19",
        "pred_tau_eff_step19",
        "log_tau_eff_pred_ml_step20",
        "tau_eff_pred_ml_step20",
        "tau_eff_prediction_source_step20",
        "n_or_p_final_step17",
        "target_quality_step18",
    ]
    tau_keep_cols = [c for c in tau_keep_cols if c in tau_pred.columns]

    sigma_ml = sigma_step12.merge(tau_pred[tau_keep_cols], on="sample_key", how="inner")
    sigma_ml = add_sigma_calculations(sigma_ml, "tau_eff_pred_ml_step20", "log_tau_eff_pred_ml_step20")

    ordered_cols = [
        "sample_key",
        "temperature_K",
        "model_name",
        "split_name",
        "split_role",
        "SID",
        "DOI",
        "doi_url",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p",
        "n_or_p_final_step17",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "prefactor_C_S_per_m_step12",
        "tau_eff_pred_ml_step20",
        "log_tau_eff_pred_ml_step20",
        "sigma_pred_ML_S_per_m_step20",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_S_per_m_step12",
        "sigma_ML_relative_error_step20",
        "sigma_ML_log_error_step20",
        "sigma_fitting_vs_ML_log_diff_step20",
        "sigma_pred_ML_status_step20",
        "tau_eff_prediction_source_step20",
        "fit_status_step12",
        "tau_eff_unit_step12",
        "tau_eff_mode_step12",
    ]
    sigma_ml = sigma_ml[[c for c in ordered_cols if c in sigma_ml.columns] + [c for c in sigma_ml.columns if c not in ordered_cols]]

    primary_test = sigma_ml[sigma_ml["split_name"].eq(args.primary_split) & sigma_ml["split_role"].eq("test")].copy()

    final_cols = [
        "sample_key",
        "pred_log_tau_eff_final_model_step19",
        "pred_tau_eff_final_model_step19",
        "final_model_name_step19",
    ]
    final_sigma = sigma_step12.merge(final_tau_pred[final_cols], on="sample_key", how="inner")
    final_sigma["sigma_pred_ML_all_samples_S_per_m_step20"] = (
        pd.to_numeric(final_sigma["prefactor_C_S_per_m_step12"], errors="coerce")
        * pd.to_numeric(final_sigma["pred_tau_eff_final_model_step19"], errors="coerce")
    )
    final_sigma["prediction_scope_step20"] = "downstream_all_samples"
    final_sigma["prediction_evaluation_note_step20"] = "not_unbiased_evaluation; model trained on all recommended ML samples"

    metrics_rows = []
    for keys, group in sigma_ml.groupby(["model_name", "split_name", "split_role"], dropna=False):
        row = {"model_name": keys[0], "split_name": keys[1], "split_role": keys[2]}
        row.update(sigma_metrics(group))
        metrics_rows.append(row)
    model_metrics = pd.DataFrame(metrics_rows)

    comp_rows = []
    primary_metrics = model_metrics[model_metrics["split_name"].eq(args.primary_split) & model_metrics["split_role"].eq("test")]
    baseline_row = primary_metrics[primary_metrics["model_name"].eq("baseline_mean")]
    baseline_sigma_rmse = baseline_row["sigma_log_rmse_step20"].iloc[0] if len(baseline_row) else np.nan
    for _, row in primary_metrics.iterrows():
        warning = []
        if row["model_name"] == selected_model and model_info["selected_model_worse_than_baseline_step20"]:
            warning.append("selected model worse than baseline on DOI split")
        if row["model_name"] == primary_best_model:
            warning.append("primary best model used for evaluation")
        comp_rows.append(
            {
                "model_name": row["model_name"],
                "primary_split": args.primary_split,
                "primary_test_n_samples": row["n_samples"],
                "primary_test_n_rows": row["n_rows"],
                "primary_test_sigma_log_rmse_step20": row["sigma_log_rmse_step20"],
                "primary_test_sigma_mape_step20": row["sigma_mape_step20"],
                "primary_test_sigma_r2_log_step20": row["sigma_r2_log_step20"],
                "primary_test_within_factor_2_rate_step20": row["within_factor_2_rate_step20"],
                "step19_selected_model": row["model_name"] == selected_model,
                "primary_split_best_tau_model": row["model_name"] == primary_best_model,
                "recommended_sigma_evaluation_model_step20": row["model_name"] in evaluation_models,
                "baseline_sigma_log_rmse_step20": baseline_sigma_rmse,
                "improvement_over_baseline_log_rmse_step20": baseline_sigma_rmse - row["sigma_log_rmse_step20"] if pd.notna(baseline_sigma_rmse) else np.nan,
                "model_warning_step20": "; ".join(warning),
            }
        )
    model_comparison = pd.DataFrame(comp_rows)

    vs_rows = []
    for keys, group in primary_test.groupby(["model_name", "split_name", "split_role"], dropna=False):
        ml = sigma_metrics(group)
        fit = fitting_metrics(group)
        gap_rmse = ml["sigma_log_rmse_step20"] - fit["fitting_sigma_log_rmse"]
        gap_mape = ml["sigma_mape_step20"] - fit["fitting_sigma_mape"]
        interpretation = "ML tau prediction is less accurate than direct tau fitting, as expected"
        if pd.notna(gap_rmse) and gap_rmse < 0.2:
            interpretation = "ML approximation is close to fitted tau performance"
        if pd.notna(gap_rmse) and gap_rmse > 1.0:
            interpretation = "ML model underperforms strongly; feature information may be insufficient"
        vs_rows.append(
            {
                "model_name": keys[0],
                "split_name": keys[1],
                "split_role": keys[2],
                "n_rows": ml["n_rows"],
                "n_samples": ml["n_samples"],
                "ML_sigma_log_rmse": ml["sigma_log_rmse_step20"],
                "ML_sigma_mape": ml["sigma_mape_step20"],
                "fitting_sigma_log_rmse": fit["fitting_sigma_log_rmse"],
                "fitting_sigma_mape": fit["fitting_sigma_mape"],
                "ML_vs_fitting_log_rmse_gap": gap_rmse,
                "ML_vs_fitting_mape_gap": gap_mape,
                "ML_worse_than_fitting_flag": bool(pd.notna(gap_rmse) and gap_rmse > 0),
                "interpretation_step20": interpretation,
            }
        )
    vs_fitting = pd.DataFrame(vs_rows)

    sample_rows = []
    for keys, group in primary_test.groupby(["sample_key", "model_name", "split_name", "split_role"], dropna=False):
        ml = sigma_metrics(group)
        fit = fitting_metrics(group)
        first = group.iloc[0]
        row = {
            "sample_key": keys[0],
            "model_name": keys[1],
            "split_name": keys[2],
            "split_role": keys[3],
            "DOI": first.get("DOI"),
            "doi_url": first.get("doi_url"),
            "sample_id": first.get("sample_id"),
            "paper_title": first.get("paper_title"),
            "composition": first.get("composition"),
            "material_system": first.get("material_system"),
            "n_or_p": first.get("n_or_p"),
            "n_or_p_final_step17": first.get("n_or_p_final_step17"),
            "n_temperature_rows_step20": len(group),
            "temperature_min_step20": pd.to_numeric(group["temperature_K"], errors="coerce").min(),
            "temperature_max_step20": pd.to_numeric(group["temperature_K"], errors="coerce").max(),
            "temperature_span_step20": pd.to_numeric(group["temperature_K"], errors="coerce").max() - pd.to_numeric(group["temperature_K"], errors="coerce").min(),
            "sigma_ML_mape_step20": ml["sigma_mape_step20"],
            "sigma_ML_log_rmse_step20": ml["sigma_log_rmse_step20"],
            "sigma_ML_r2_log_step20": ml["sigma_r2_log_step20"],
            "sigma_ML_bias_log_step20": ml["sigma_bias_log_step20"],
            "sigma_ML_within_factor_2_rate_step20": ml["within_factor_2_rate_step20"],
            "sigma_fitting_mape_step12": fit["fitting_sigma_mape"],
            "sigma_fitting_log_rmse_step12": fit["fitting_sigma_log_rmse"],
            "target_log_tau_eff_step18": first.get("target_log_tau_eff_step18"),
            "pred_log_tau_eff_step19": first.get("pred_log_tau_eff_step19"),
            "target_tau_eff_step18": first.get("target_tau_eff_step18"),
            "pred_tau_eff_step19": first.get("pred_tau_eff_step19"),
        }
        row["sigma_prediction_quality_step20"] = quality_from_metrics(row)
        row["sigma_prediction_note_step20"] = "primary split test sample aggregate"
        sample_rows.append(row)
    sample_results = pd.DataFrame(sample_rows)

    material_summary = (
        sample_results.groupby(["material_system", "n_or_p", "model_name"], dropna=False)
        .agg(
            sample_count=("sample_key", "nunique"),
            row_count=("n_temperature_rows_step20", "sum"),
            median_sigma_ML_mape_step20=("sigma_ML_mape_step20", "median"),
            median_sigma_ML_log_rmse_step20=("sigma_ML_log_rmse_step20", "median"),
            median_sigma_ML_within_factor_2_rate_step20=("sigma_ML_within_factor_2_rate_step20", "median"),
            excellent_sample_count=("sigma_prediction_quality_step20", lambda s: int((s == "excellent").sum())),
            good_sample_count=("sigma_prediction_quality_step20", lambda s: int((s == "good").sum())),
            moderate_sample_count=("sigma_prediction_quality_step20", lambda s: int((s == "moderate").sum())),
            poor_sample_count=("sigma_prediction_quality_step20", lambda s: int((s == "poor").sum())),
        )
        .reset_index()
    )
    np_summary = (
        sample_results.groupby(["n_or_p", "model_name"], dropna=False)
        .agg(
            sample_count=("sample_key", "nunique"),
            row_count=("n_temperature_rows_step20", "sum"),
            median_sigma_ML_mape_step20=("sigma_ML_mape_step20", "median"),
            median_sigma_ML_log_rmse_step20=("sigma_ML_log_rmse_step20", "median"),
            median_sigma_ML_within_factor_2_rate_step20=("sigma_ML_within_factor_2_rate_step20", "median"),
        )
        .reset_index()
    )

    problem_rows = sigma_ml[
        (pd.to_numeric(sigma_ml["sigma_ML_relative_error_step20"], errors="coerce") > 1.0)
        | (pd.to_numeric(sigma_ml["sigma_ML_log_error_step20"], errors="coerce").abs() > 1.0)
        | (~sigma_ml["sigma_pred_ML_status_step20"].eq("ok"))
    ].copy()
    problem_samples = sample_results[
        sample_results["sigma_prediction_quality_step20"].eq("poor")
        | (pd.to_numeric(sample_results["sigma_ML_log_rmse_step20"], errors="coerce") > 1.0)
        | (pd.to_numeric(sample_results["sigma_ML_mape_step20"], errors="coerce") > 1.0)
    ].copy()
    problem_samples["sigma_ML_problem_reason_step20"] = problem_samples.apply(problem_reason, axis=1)
    problem_samples["needs_review_for_ML_sigma_step20"] = True

    downstream_cols = [
        "sample_key",
        "temperature_K",
        "sigma_pred_ML_all_samples_S_per_m_step20",
        "pred_tau_eff_final_model_step19",
        "pred_log_tau_eff_final_model_step19",
        "final_model_name_step19",
        "sigma_obs_S_per_m_step11",
        "sigma_pred_S_per_m_step12",
        "seebeck_obs_V_per_K_step11",
        "kappa_obs_W_per_mK_step11",
        "zt_obs_dimensionless_step11",
        "zt_calc_from_obs_step11",
        "DOI",
        "doi_url",
        "sample_id",
        "paper_title",
        "composition",
        "material_system",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "prediction_scope_step20",
        "prediction_evaluation_note_step20",
    ]
    downstream_ready = final_sigma[[c for c in downstream_cols if c in final_sigma.columns]].copy()

    np_changed = 0
    sintering_changed = 0

    output_counts = {
        "sigma_ml_predictions_step20 rows": len(sigma_ml),
        "sigma_ml_primary_test_predictions_step20 rows": len(primary_test),
        "sigma_ml_all_samples_predictions_step20 rows": len(final_sigma),
        "sigma_ml_downstream_ready_step20 rows": len(downstream_ready),
    }
    report = make_report(
        input_counts,
        output_counts,
        model_info,
        recommended_model,
        model_comparison,
        vs_fitting,
        material_summary,
        np_summary,
        problem_samples,
        np_changed,
        sintering_changed,
    )
    notes = make_notes()

    sigma_ml.to_csv(output_dir / "sigma_ml_predictions_step20.csv", index=False)
    primary_test.to_csv(output_dir / "sigma_ml_primary_test_predictions_step20.csv", index=False)
    final_sigma.to_csv(output_dir / "sigma_ml_all_samples_predictions_step20.csv", index=False)
    model_metrics.to_csv(output_dir / "sigma_ml_model_metrics_step20.csv", index=False)
    model_comparison.to_csv(output_dir / "sigma_ml_model_comparison_step20.csv", index=False)
    vs_fitting.to_csv(output_dir / "sigma_ml_vs_fitting_comparison_step20.csv", index=False)
    sample_results.to_csv(output_dir / "sigma_ml_sample_results_step20.csv", index=False)
    material_summary.to_csv(output_dir / "sigma_ml_material_summary_step20.csv", index=False)
    np_summary.to_csv(output_dir / "sigma_ml_np_summary_step20.csv", index=False)
    problem_samples.to_csv(output_dir / "sigma_ml_problem_samples_step20.csv", index=False)
    problem_rows.to_csv(output_dir / "sigma_ml_problem_rows_step20.csv", index=False)
    downstream_ready.to_csv(output_dir / "sigma_ml_downstream_ready_step20.csv", index=False)
    (output_dir / "step20_sigma_ml_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step20_sigma_ml_notes.md").write_text(notes, encoding="utf-8")

    write_excel(
        output_dir / "starrydata2_step20_sigma_ml_prediction.xlsx",
        {
            "model_comparison": model_comparison,
            "model_metrics": model_metrics,
            "primary_test_predictions": primary_test,
            "sample_results": sample_results,
            "vs_fitting_comparison": vs_fitting,
            "material_summary": material_summary,
            "np_summary": np_summary,
            "problem_samples": problem_samples,
            "downstream_ready": downstream_ready,
            "sigma_ml_report": report,
        },
    )

    rec_row = model_comparison[model_comparison["model_name"].eq(recommended_model)]
    baseline_row = model_comparison[model_comparison["model_name"].eq("baseline_mean")]
    fit_rec = vs_fitting[vs_fitting["model_name"].eq(recommended_model)]
    rec_sigma_rmse = rec_row["primary_test_sigma_log_rmse_step20"].iloc[0] if len(rec_row) else np.nan
    rec_mape = rec_row["primary_test_sigma_mape_step20"].iloc[0] if len(rec_row) else np.nan
    baseline_sigma_rmse = baseline_row["primary_test_sigma_log_rmse_step20"].iloc[0] if len(baseline_row) else np.nan
    fitting_rmse = fit_rec["fitting_sigma_log_rmse"].iloc[0] if len(fit_rec) else np.nan
    ml_fit_gap = fit_rec["ML_vs_fitting_log_rmse_gap"].iloc[0] if len(fit_rec) else np.nan
    print("Done.")
    print("Created:")
    for name in [
        "sigma_ml_predictions_step20.csv",
        "sigma_ml_primary_test_predictions_step20.csv",
        "sigma_ml_all_samples_predictions_step20.csv",
        "sigma_ml_model_metrics_step20.csv",
        "sigma_ml_model_comparison_step20.csv",
        "sigma_ml_vs_fitting_comparison_step20.csv",
        "sigma_ml_sample_results_step20.csv",
        "sigma_ml_material_summary_step20.csv",
        "sigma_ml_np_summary_step20.csv",
        "sigma_ml_problem_samples_step20.csv",
        "sigma_ml_problem_rows_step20.csv",
        "sigma_ml_downstream_ready_step20.csv",
        "step20_sigma_ml_report.txt",
        "step20_sigma_ml_notes.md",
        "starrydata2_step20_sigma_ml_prediction.xlsx",
    ]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"sigma ML prediction rows: {len(sigma_ml)}")
    print(f"primary test prediction rows: {len(primary_test)}")
    print(f"downstream ready rows: {len(downstream_ready)}")
    print(f"Step19 selected model: {selected_model}")
    print(f"Primary DOI best model: {primary_best_model}")
    print(f"Recommended sigma evaluation model: {recommended_model}")
    print(f"Selected model worse than baseline: {model_info['selected_model_worse_than_baseline_step20']}")
    print(f"Recommended model primary sigma log RMSE: {rec_sigma_rmse}")
    print(f"Recommended model primary sigma MAPE: {rec_mape}")
    print(f"Baseline primary sigma log RMSE: {baseline_sigma_rmse}")
    print(f"Step12 fitting sigma log RMSE: {fitting_rmse}")
    print(f"ML vs fitting gap: {ml_fit_gap}")
    print(f"problem samples: {len(problem_samples)}")
    print(f"n/p changed rows: {np_changed}")
    print(f"sintering changed rows: {sintering_changed}")


if __name__ == "__main__":
    main()
