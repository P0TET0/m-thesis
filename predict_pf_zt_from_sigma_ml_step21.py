import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP20_DIR = "data/output/starrydata2_step20_sigma_ml_prediction"
DEFAULT_STEP14_DIR = "data/output/starrydata2_step14_pf_zt_prediction"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step21_pf_zt_ml_prediction"
EXCEL_PREVIEW_ROWS = 100_000
EPS = 1e-30

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


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Step21 PF/ZT from Step20 ML sigma predictions.")
    parser.add_argument("--step20_dir", default=DEFAULT_STEP20_DIR)
    parser.add_argument("--step14_dir", default=DEFAULT_STEP14_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--primary_split", default="split_doi_group_80_20_step18")
    parser.add_argument("--recommended_model_name", default="auto")
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def read_csv(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return None
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def finite_positive(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0)


def safe_log_error(pred, obs):
    pred = pd.to_numeric(pred, errors="coerce")
    obs = pd.to_numeric(obs, errors="coerce")
    ok = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)
    return np.where(ok, np.log(pred) - np.log(obs), np.nan)


def rmse(values):
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    return float(math.sqrt(np.mean(values**2))) if len(values) else np.nan


def mape_from_rel(rel):
    rel = pd.to_numeric(rel, errors="coerce")
    rel = rel[np.isfinite(rel)]
    return float(np.mean(rel)) if len(rel) else np.nan


def quality_from_error(mape, log_rmse):
    if pd.isna(mape) or pd.isna(log_rmse):
        return "not_evaluated"
    if log_rmse <= 0.20 and mape <= 0.20:
        return "excellent"
    if log_rmse <= 0.40 and mape <= 0.50:
        return "good"
    if log_rmse <= 0.80 and mape <= 1.00:
        return "moderate"
    return "poor"


def r2_log(true, pred):
    true = pd.to_numeric(true, errors="coerce")
    pred = pd.to_numeric(pred, errors="coerce")
    ok = np.isfinite(true) & np.isfinite(pred) & (true > 0) & (pred > 0)
    if ok.sum() < 2:
        return np.nan
    yt = np.log(true[ok])
    yp = np.log(pred[ok])
    denom = np.sum((yt - yt.mean()) ** 2)
    return float(1 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else np.nan


def regression_metrics(df, pred_col, obs_col, rel_col=None, log_col=None):
    pred = pd.to_numeric(df[pred_col], errors="coerce") if pred_col in df.columns else pd.Series(dtype=float)
    obs = pd.to_numeric(df[obs_col], errors="coerce") if obs_col in df.columns else pd.Series(dtype=float)
    ok = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)
    if ok.sum() == 0:
        return {"mape": np.nan, "log_rmse": np.nan, "r2_log": np.nan, "spearman": np.nan}
    if rel_col and rel_col in df.columns:
        mape = mape_from_rel(df.loc[ok, rel_col])
    else:
        mape = float(np.mean(np.abs(pred[ok] - obs[ok]) / np.maximum(np.abs(obs[ok]), EPS)))
    if log_col and log_col in df.columns:
        log_rmse = rmse(df.loc[ok, log_col])
    else:
        log_rmse = float(math.sqrt(np.mean((np.log(pred[ok]) - np.log(obs[ok])) ** 2)))
    spearman = pd.Series(obs[ok]).corr(pd.Series(pred[ok]), method="spearman") if ok.sum() > 1 else np.nan
    return {"mape": mape, "log_rmse": log_rmse, "r2_log": r2_log(obs, pred), "spearman": float(spearman) if pd.notna(spearman) else np.nan}


def classification_counts(obs_positive, pred_positive):
    obs = pd.Series(obs_positive).fillna(False).astype(bool)
    pred = pd.Series(pred_positive).fillna(False).astype(bool)
    tp = int((obs & pred).sum())
    fp = int((~obs & pred).sum())
    fn = int((obs & ~pred).sum())
    tn = int((~obs & ~pred).sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    f1 = 2 * precision * recall / (precision + recall) if pd.notna(precision) and pd.notna(recall) and (precision + recall) else np.nan
    accuracy = (tp + tn) / len(obs) if len(obs) else np.nan
    balanced = np.nanmean([recall, specificity])
    return {
        "n_samples": int(len(obs)),
        "n_observed_positive": int(obs.sum()),
        "n_predicted_positive": int(pred.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def determine_recommended_model(model_comparison, arg):
    if arg != "auto":
        return arg, "argument"
    if "recommended_sigma_evaluation_model_step20" in model_comparison.columns:
        rec = model_comparison[model_comparison["recommended_sigma_evaluation_model_step20"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if len(rec):
            return str(rec.iloc[0]["model_name"]), "sigma_ml_model_comparison_step20 recommended flag"
    if "primary_test_sigma_log_rmse_step20" in model_comparison.columns:
        sub = model_comparison[pd.to_numeric(model_comparison["primary_test_sigma_log_rmse_step20"], errors="coerce").notna()]
        if len(sub):
            return str(sub.sort_values("primary_test_sigma_log_rmse_step20").iloc[0]["model_name"]), "minimum primary_test_sigma_log_rmse_step20"
    if "model_name" in model_comparison.columns and model_comparison["model_name"].eq("gradient_boosting").any():
        return "gradient_boosting", "fallback gradient_boosting"
    non_base = model_comparison[~model_comparison["model_name"].eq("baseline_mean")]
    return str(non_base.iloc[0]["model_name"] if len(non_base) else model_comparison.iloc[0]["model_name"]), "first available nonbaseline model"


def add_pf_zt(df, sigma_col):
    out = df.copy()
    sigma = pd.to_numeric(out[sigma_col], errors="coerce")
    s = pd.to_numeric(out["seebeck_obs_V_per_K_step11"], errors="coerce")
    kappa = pd.to_numeric(out["kappa_obs_W_per_mK_step11"], errors="coerce")
    temp = pd.to_numeric(out["temperature_K"], errors="coerce")
    obs_sigma = pd.to_numeric(out["sigma_obs_S_per_m_step11"], errors="coerce")

    out["sigma_pred_ML_for_pf_zt_S_per_m_step21"] = sigma
    out["power_factor_pred_ML_W_per_mK2_step21"] = s**2 * sigma
    out["power_factor_pred_ML_uW_per_cmK2_step21"] = out["power_factor_pred_ML_W_per_mK2_step21"] * 10000
    if "power_factor_obs_W_per_mK2_step11" in out.columns:
        out["power_factor_obs_W_per_mK2_step21"] = pd.to_numeric(out["power_factor_obs_W_per_mK2_step11"], errors="coerce")
    else:
        out["power_factor_obs_W_per_mK2_step21"] = s**2 * obs_sigma
    out["power_factor_obs_uW_per_cmK2_step21"] = out["power_factor_obs_W_per_mK2_step21"] * 10000

    pf_ok = np.isfinite(s) & np.isfinite(sigma) & (sigma > 0)
    out["pf_pred_ML_status_step21"] = np.where(pf_ok, "ok", "missing_or_invalid_sigma_or_seebeck")
    zt_ok = pf_ok & np.isfinite(temp) & np.isfinite(kappa) & (kappa > 0)
    out["zt_pred_ML_step21"] = np.where(zt_ok, out["power_factor_pred_ML_W_per_mK2_step21"] * temp / kappa, np.nan)
    out["zt_pred_ML_status_step21"] = np.where(zt_ok, "ok", "missing_or_invalid_S_sigma_T_or_kappa")

    fit_sigma = pd.to_numeric(out["sigma_pred_S_per_m_step12"], errors="coerce") if "sigma_pred_S_per_m_step12" in out.columns else np.nan
    out["power_factor_pred_fitting_W_per_mK2_step21"] = s**2 * fit_sigma
    fit_ok = np.isfinite(s) & np.isfinite(fit_sigma) & (fit_sigma > 0)
    out["zt_pred_fitting_step21"] = np.where(fit_ok & np.isfinite(temp) & np.isfinite(kappa) & (kappa > 0), s**2 * fit_sigma * temp / kappa, np.nan)

    pf_pred = pd.to_numeric(out["power_factor_pred_ML_W_per_mK2_step21"], errors="coerce")
    pf_obs = pd.to_numeric(out["power_factor_obs_W_per_mK2_step21"], errors="coerce")
    out["pf_ML_abs_error_step21"] = (pf_pred - pf_obs).abs()
    out["pf_ML_relative_error_step21"] = (pf_pred - pf_obs).abs() / np.maximum(pf_obs.abs(), EPS)
    out["pf_ML_log_error_step21"] = safe_log_error(pf_pred, pf_obs)
    out["pf_ML_error_status_step21"] = np.where(np.isfinite(out["pf_ML_log_error_step21"]), "ok", "not_evaluated")
    out["pf_fitting_relative_error_step21"] = (out["power_factor_pred_fitting_W_per_mK2_step21"] - pf_obs).abs() / np.maximum(pf_obs.abs(), EPS)
    out["pf_fitting_log_error_step21"] = safe_log_error(out["power_factor_pred_fitting_W_per_mK2_step21"], pf_obs)

    for target_col, prefix in [
        ("zt_obs_dimensionless_step11", "zt_ML_vs_obs"),
        ("zt_calc_from_obs_step11", "zt_ML_vs_calc"),
    ]:
        target = pd.to_numeric(out[target_col], errors="coerce") if target_col in out.columns else pd.Series(np.nan, index=out.index)
        pred = pd.to_numeric(out["zt_pred_ML_step21"], errors="coerce")
        out[f"{prefix}_abs_error_step21"] = (pred - target).abs()
        out[f"{prefix}_relative_error_step21"] = (pred - target).abs() / np.maximum(target.abs(), EPS)
        out[f"{prefix}_log_error_step21"] = safe_log_error(pred, target)
        out[f"{prefix}_status_step21"] = np.where(np.isfinite(out[f"{prefix}_log_error_step21"]), "ok", "not_evaluated")
    obs = pd.to_numeric(out.get("zt_obs_dimensionless_step11"), errors="coerce")
    calc = pd.to_numeric(out.get("zt_calc_from_obs_step11"), errors="coerce")
    fitzt = pd.to_numeric(out["zt_pred_fitting_step21"], errors="coerce")
    out["zt_fitting_vs_obs_relative_error_step21"] = (fitzt - obs).abs() / np.maximum(obs.abs(), EPS)
    out["zt_fitting_vs_calc_relative_error_step21"] = (fitzt - calc).abs() / np.maximum(calc.abs(), EPS)
    out["zt_fitting_vs_obs_log_error_step21"] = safe_log_error(fitzt, obs)
    out["zt_fitting_vs_calc_log_error_step21"] = safe_log_error(fitzt, calc)
    return out


def sample_aggregate(df, scope, downstream_note=None):
    rows = []
    for sample_key, g in df.groupby("sample_key", dropna=False):
        first = g.iloc[0]
        pf = regression_metrics(g, "power_factor_pred_ML_W_per_mK2_step21", "power_factor_obs_W_per_mK2_step21", "pf_ML_relative_error_step21", "pf_ML_log_error_step21")
        zto = regression_metrics(g, "zt_pred_ML_step21", "zt_obs_dimensionless_step11", "zt_ML_vs_obs_relative_error_step21", "zt_ML_vs_obs_log_error_step21")
        ztc = regression_metrics(g, "zt_pred_ML_step21", "zt_calc_from_obs_step11", "zt_ML_vs_calc_relative_error_step21", "zt_ML_vs_calc_log_error_step21")
        pff = regression_metrics(g, "power_factor_pred_fitting_W_per_mK2_step21", "power_factor_obs_W_per_mK2_step21", "pf_fitting_relative_error_step21", "pf_fitting_log_error_step21")
        ztof = regression_metrics(g, "zt_pred_fitting_step21", "zt_obs_dimensionless_step11", "zt_fitting_vs_obs_relative_error_step21", "zt_fitting_vs_obs_log_error_step21")
        ztcf = regression_metrics(g, "zt_pred_fitting_step21", "zt_calc_from_obs_step11", "zt_fitting_vs_calc_relative_error_step21", "zt_fitting_vs_calc_log_error_step21")
        row = {
            "sample_key": sample_key,
            "model_name": first.get("model_name", first.get("final_model_name_step19", "")),
            "evaluation_scope_step21": scope,
            "DOI": first.get("DOI"),
            "doi_url": first.get("doi_url"),
            "sample_id": first.get("sample_id"),
            "paper_title": first.get("paper_title"),
            "composition": first.get("composition"),
            "material_system": first.get("material_system"),
            "n_or_p": first.get("n_or_p"),
            "n_rows_step21": len(g),
            "n_pf_eval_rows_step21": int(g["pf_ML_error_status_step21"].eq("ok").sum()),
            "n_zt_obs_eval_rows_step21": int(g["zt_ML_vs_obs_status_step21"].eq("ok").sum()),
            "n_zt_calc_eval_rows_step21": int(g["zt_ML_vs_calc_status_step21"].eq("ok").sum()),
            "pf_ML_mape_step21": pf["mape"],
            "pf_ML_log_rmse_step21": pf["log_rmse"],
            "zt_ML_vs_obs_mape_step21": zto["mape"],
            "zt_ML_vs_obs_log_rmse_step21": zto["log_rmse"],
            "zt_ML_vs_calc_mape_step21": ztc["mape"],
            "zt_ML_vs_calc_log_rmse_step21": ztc["log_rmse"],
            "pf_fitting_mape_step21": pff["mape"],
            "zt_fitting_vs_obs_mape_step21": ztof["mape"],
            "zt_fitting_vs_calc_mape_step21": ztcf["mape"],
            "zt_obs_max_step21": pd.to_numeric(g.get("zt_obs_dimensionless_step11"), errors="coerce").max(),
            "zt_pred_ML_max_step21": pd.to_numeric(g["zt_pred_ML_step21"], errors="coerce").max(),
            "zt_pred_fitting_max_step21": pd.to_numeric(g["zt_pred_fitting_step21"], errors="coerce").max(),
            "zt_calc_from_obs_max_step21": pd.to_numeric(g.get("zt_calc_from_obs_step11"), errors="coerce").max(),
            "sintering_method": first.get("sintering_method"),
            "sintering_checked": first.get("sintering_checked"),
            "record_checked": first.get("record_checked"),
        }
        row["zt_ML_vs_obs_quality_step21"] = quality_from_error(row["zt_ML_vs_obs_mape_step21"], row["zt_ML_vs_obs_log_rmse_step21"])
        row["zt_ML_vs_calc_quality_step21"] = quality_from_error(row["zt_ML_vs_calc_mape_step21"], row["zt_ML_vs_calc_log_rmse_step21"])
        if downstream_note is not None:
            row["downstream_evaluation_note_step21"] = downstream_note
        rows.append(row)
    return pd.DataFrame(rows)


def model_metrics(df):
    rows = []
    for keys, g in df.groupby(["model_name", "split_name", "split_role"], dropna=False):
        pf = regression_metrics(g, "power_factor_pred_ML_W_per_mK2_step21", "power_factor_obs_W_per_mK2_step21", "pf_ML_relative_error_step21", "pf_ML_log_error_step21")
        zto = regression_metrics(g, "zt_pred_ML_step21", "zt_obs_dimensionless_step11", "zt_ML_vs_obs_relative_error_step21", "zt_ML_vs_obs_log_error_step21")
        ztc = regression_metrics(g, "zt_pred_ML_step21", "zt_calc_from_obs_step11", "zt_ML_vs_calc_relative_error_step21", "zt_ML_vs_calc_log_error_step21")
        sample = sample_aggregate(g, "tmp")
        cls = classification_counts(sample["zt_obs_max_step21"] >= 1.0, sample["zt_pred_ML_max_step21"] >= 1.0)
        rows.append({
            "model_name": keys[0],
            "split_name": keys[1],
            "split_role": keys[2],
            "n_rows": len(g),
            "n_samples": g["sample_key"].nunique(),
            "pf_mape_step21": pf["mape"],
            "pf_log_rmse_step21": pf["log_rmse"],
            "zt_vs_obs_mape_step21": zto["mape"],
            "zt_vs_obs_log_rmse_step21": zto["log_rmse"],
            "zt_vs_calc_mape_step21": ztc["mape"],
            "zt_vs_calc_log_rmse_step21": ztc["log_rmse"],
            "zt_obs_ge_1_precision_step21": cls["precision"],
            "zt_obs_ge_1_recall_step21": cls["recall"],
            "zt_obs_ge_1_f1_step21": cls["f1"],
        })
    return pd.DataFrame(rows)


def high_zt_classification(primary_samples, downstream_samples, thresholds):
    rows = []
    for scope, data, note in [
        ("primary_doi_test", primary_samples, "unbiased primary DOI test"),
        ("downstream_all_samples", downstream_samples, "not unbiased evaluation; candidate screening only"),
    ]:
        for threshold in thresholds:
            obs_source = "zt_obs_max_step21"
            obs = pd.to_numeric(data["zt_obs_max_step21"], errors="coerce")
            if obs.notna().sum() == 0:
                obs_source = "zt_calc_from_obs_max_step21"
                obs = pd.to_numeric(data["zt_calc_from_obs_max_step21"], errors="coerce")
            pred = pd.to_numeric(data["zt_pred_ML_max_step21"], errors="coerce")
            counts = classification_counts(obs >= threshold, pred >= threshold)
            row = {"evaluation_scope_step21": scope, "threshold": threshold}
            row.update(counts)
            row["evaluation_note_step21"] = f"{note}; observed label source={obs_source}"
            rows.append(row)
    return pd.DataFrame(rows)


def vs_fitting(primary_rows, primary_samples, threshold):
    pf_ml = regression_metrics(primary_rows, "power_factor_pred_ML_W_per_mK2_step21", "power_factor_obs_W_per_mK2_step21", "pf_ML_relative_error_step21", "pf_ML_log_error_step21")
    pf_fit = regression_metrics(primary_rows, "power_factor_pred_fitting_W_per_mK2_step21", "power_factor_obs_W_per_mK2_step21", "pf_fitting_relative_error_step21", "pf_fitting_log_error_step21")
    zto_ml = regression_metrics(primary_rows, "zt_pred_ML_step21", "zt_obs_dimensionless_step11", "zt_ML_vs_obs_relative_error_step21", "zt_ML_vs_obs_log_error_step21")
    zto_fit = regression_metrics(primary_rows, "zt_pred_fitting_step21", "zt_obs_dimensionless_step11", "zt_fitting_vs_obs_relative_error_step21", "zt_fitting_vs_obs_log_error_step21")
    ztc_ml = regression_metrics(primary_rows, "zt_pred_ML_step21", "zt_calc_from_obs_step11", "zt_ML_vs_calc_relative_error_step21", "zt_ML_vs_calc_log_error_step21")
    ztc_fit = regression_metrics(primary_rows, "zt_pred_fitting_step21", "zt_calc_from_obs_step11", "zt_fitting_vs_calc_relative_error_step21", "zt_fitting_vs_calc_log_error_step21")
    cls_ml = classification_counts(primary_samples["zt_obs_max_step21"] >= threshold, primary_samples["zt_pred_ML_max_step21"] >= threshold)
    cls_fit = classification_counts(primary_samples["zt_obs_max_step21"] >= threshold, primary_samples["zt_pred_fitting_max_step21"] >= threshold)
    pairs = [
        ("PF MAPE", pf_ml["mape"], pf_fit["mape"]),
        ("PF log RMSE", pf_ml["log_rmse"], pf_fit["log_rmse"]),
        ("ZT vs obs MAPE", zto_ml["mape"], zto_fit["mape"]),
        ("ZT vs obs log RMSE", zto_ml["log_rmse"], zto_fit["log_rmse"]),
        ("ZT vs calc MAPE", ztc_ml["mape"], ztc_fit["mape"]),
        ("ZT vs calc log RMSE", ztc_ml["log_rmse"], ztc_fit["log_rmse"]),
        ("ZT>=1 precision", cls_ml["precision"], cls_fit["precision"]),
        ("ZT>=1 recall", cls_ml["recall"], cls_fit["recall"]),
        ("ZT>=1 F1", cls_ml["f1"], cls_fit["f1"]),
    ]
    rows = []
    for metric, ml, fit in pairs:
        diff = ml - fit if pd.notna(ml) and pd.notna(fit) else np.nan
        worse = diff > 0 if "precision" not in metric and "recall" not in metric and "F1" not in metric else diff < 0
        interp = "ML version is worse than direct fitting, as expected, because it predicts tau_eff from material features." if worse else "ML version is close to fitting version."
        if pd.notna(diff) and abs(diff) > 1:
            interp = "ML version performs poorly; features may be insufficient." if worse else interp
        rows.append({"metric_name": metric, "ML_value": ml, "fitting_value": fit, "ML_minus_fitting": diff, "ML_worse_than_fitting": bool(worse), "interpretation_step21": interp})
    return pd.DataFrame(rows)


def summarize_groups(samples):
    mat = samples.groupby(["material_system", "n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        pf_ML_mape_median_step21=("pf_ML_mape_step21", "median"),
        zt_ML_vs_obs_mape_median_step21=("zt_ML_vs_obs_mape_step21", "median"),
        zt_ML_vs_calc_mape_median_step21=("zt_ML_vs_calc_mape_step21", "median"),
        zt_pred_ML_max_median_step21=("zt_pred_ML_max_step21", "median"),
        zt_obs_max_median_step21=("zt_obs_max_step21", "median"),
        high_zt_observed_count=("zt_obs_max_step21", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1.0).sum())),
        high_zt_predicted_count=("zt_pred_ML_max_step21", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1.0).sum())),
        good_or_excellent_count=("zt_ML_vs_obs_quality_step21", lambda s: int(s.isin(["good", "excellent"]).sum())),
        poor_count=("zt_ML_vs_obs_quality_step21", lambda s: int((s == "poor").sum())),
    ).reset_index()
    np_sum = samples.groupby(["n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        pf_ML_mape_median_step21=("pf_ML_mape_step21", "median"),
        zt_ML_vs_obs_mape_median_step21=("zt_ML_vs_obs_mape_step21", "median"),
        zt_ML_vs_calc_mape_median_step21=("zt_ML_vs_calc_mape_step21", "median"),
        zt_pred_ML_max_median_step21=("zt_pred_ML_max_step21", "median"),
        zt_obs_max_median_step21=("zt_obs_max_step21", "median"),
        high_zt_observed_count=("zt_obs_max_step21", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1.0).sum())),
        high_zt_predicted_count=("zt_pred_ML_max_step21", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1.0).sum())),
    ).reset_index()
    return mat, np_sum


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
    return """# Step21 PF/ZT Prediction from ML sigma

## Purpose
Calculate PF_pred_ML and ZT_pred_ML from Step20 sigma_pred_ML.

## Formula
PF_pred_ML = S_obs^2 * sigma_pred_ML.
ZT_pred_ML = S_obs^2 * sigma_pred_ML * T / kappa_obs.

## Evaluation Scope
Primary DOI test rows are used for evaluation. Downstream all-samples predictions are candidate screening outputs.

## Primary DOI Test Results
See `step21_pf_zt_ml_report.txt`.

## Downstream Candidate Screening
Candidate samples are selected from downstream all-samples predictions by high observed, calculated, or predicted ZT.

## Comparison with Direct Fitting
The direct fitting comparison uses the same observed S and kappa with Step12 fitted sigma.

## Important Caveats
Step21 does not predict Seebeck coefficient or thermal conductivity.
PF_pred_ML and ZT_pred_ML are calculated using sigma_pred_ML and observed S/kappa.
Downstream all-samples predictions are for screening, not unbiased evaluation.
Direct fitting is expected to outperform ML tau prediction because it uses sigma observations directly.

## Next Step
Step22 should compare direct fitting and ML versions of sigma, PF, and ZT for reporting.
"""


def make_report(input_counts, recommended_model, primary_split, primary_rows, downstream_rows, primary_samples, downstream_samples, model_metrics_df, high_cls, vs_fit, problem_samples, candidates, np_changed, sintering_changed):
    rec_metrics = model_metrics_df[model_metrics_df["model_name"].eq(recommended_model)].iloc[0].to_dict() if len(model_metrics_df[model_metrics_df["model_name"].eq(recommended_model)]) else {}
    cls1 = high_cls[(high_cls["evaluation_scope_step21"].eq("primary_doi_test")) & (high_cls["threshold"].eq(1.0))].iloc[0].to_dict()
    down_cls1 = high_cls[(high_cls["evaluation_scope_step21"].eq("downstream_all_samples")) & (high_cls["threshold"].eq(1.0))].iloc[0].to_dict()
    lines = [
        "Step21 PF/ZT ML report",
        "",
        f"Input sigma_ml_primary_test_predictions_step20 rows: {input_counts['primary']}",
        f"Input sigma_ml_downstream_ready_step20 rows: {input_counts['downstream']}",
        "",
        f"recommended model: {recommended_model}",
        f"primary split: {primary_split}",
        f"primary test rows: {len(primary_rows)}",
        f"downstream rows: {len(downstream_rows)}",
        "",
        "primary PF/ZT:",
        f"- PF calculable rows: {int(primary_rows['pf_pred_ML_status_step21'].eq('ok').sum())}",
        f"- ZT calculable rows: {int(primary_rows['zt_pred_ML_status_step21'].eq('ok').sum())}",
        f"- PF MAPE: {rec_metrics.get('pf_mape_step21')}",
        f"- PF log RMSE: {rec_metrics.get('pf_log_rmse_step21')}",
        f"- ZT vs obs MAPE: {rec_metrics.get('zt_vs_obs_mape_step21')}",
        f"- ZT vs obs log RMSE: {rec_metrics.get('zt_vs_obs_log_rmse_step21')}",
        f"- ZT vs calc MAPE: {rec_metrics.get('zt_vs_calc_mape_step21')}",
        f"- ZT vs calc log RMSE: {rec_metrics.get('zt_vs_calc_log_rmse_step21')}",
        "",
        "primary ZT>=1:",
        f"- precision: {cls1.get('precision')}",
        f"- recall: {cls1.get('recall')}",
        f"- F1: {cls1.get('f1')}",
        f"- accuracy: {cls1.get('accuracy')}",
        f"- false positive: {cls1.get('false_positive')}",
        f"- false negative: {cls1.get('false_negative')}",
        "",
        "downstream PF/ZT:",
        f"- PF calculable rows: {int(downstream_rows['pf_pred_ML_status_step21'].eq('ok').sum())}",
        f"- ZT calculable rows: {int(downstream_rows['zt_pred_ML_status_step21'].eq('ok').sum())}",
        f"- ZT_pred_ML>=1 sample count: {down_cls1.get('n_predicted_positive')}",
        f"- ZT_obs>=1 sample count: {down_cls1.get('n_observed_positive')}",
        "",
        "ML vs fitting:",
    ]
    for metric in ["PF MAPE", "ZT vs obs MAPE", "ZT>=1 F1"]:
        row = vs_fit[vs_fit["metric_name"].eq(metric)]
        if len(row):
            r = row.iloc[0]
            lines.append(f"- {metric}: fitting={r['fitting_value']}, ML={r['ML_value']}")
    lines.extend([
        "",
        "problem samples:",
        f"- pf_zt_ml_problem_samples_step21 count: {len(problem_samples)}",
        f"- pf_zt_ml_candidate_samples_step21 count: {len(candidates)}",
        "",
        f"n/p changed rows: {np_changed}",
        f"sintering changed rows: {sintering_changed}",
        "",
        "Notes:",
        "Step21 did not refit tau_eff or retrain ML models.",
        "Step21 calculated PF_pred_ML and ZT_pred_ML from sigma_pred_ML.",
        "Seebeck coefficient S and thermal conductivity kappa were not predicted; observed values were used.",
        "Downstream all-samples predictions are for candidate screening and not unbiased evaluation.",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    step20_dir = Path(args.step20_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_all = read_csv(step20_dir / "sigma_ml_primary_test_predictions_step20.csv")
    downstream = read_csv(step20_dir / "sigma_ml_downstream_ready_step20.csv")
    model_comp = read_csv(step20_dir / "sigma_ml_model_comparison_step20.csv")
    read_csv(step20_dir / "sigma_ml_vs_fitting_comparison_step20.csv")
    read_csv(step20_dir / "sigma_ml_sample_results_step20.csv")

    require_columns(primary_all, ["sample_key", "temperature_K", "model_name", "split_name", "split_role", "sigma_pred_ML_S_per_m_step20", "sigma_obs_S_per_m_step11"], "primary")
    require_columns(primary_all, ["seebeck_obs_V_per_K_step11", "kappa_obs_W_per_mK_step11", "zt_obs_dimensionless_step11", "zt_calc_from_obs_step11"], "primary")
    require_columns(downstream, ["sample_key", "temperature_K", "sigma_pred_ML_all_samples_S_per_m_step20", "sigma_obs_S_per_m_step11"], "downstream")
    require_columns(downstream, ["seebeck_obs_V_per_K_step11", "kappa_obs_W_per_mK_step11", "zt_obs_dimensionless_step11", "zt_calc_from_obs_step11"], "downstream")

    recommended_model, model_source = determine_recommended_model(model_comp, args.recommended_model_name)
    primary = primary_all[
        primary_all["split_name"].eq(args.primary_split)
        & primary_all["split_role"].eq("test")
        & primary_all["model_name"].eq(recommended_model)
        & finite_positive(primary_all["sigma_pred_ML_S_per_m_step20"])
    ].copy()
    primary["evaluation_scope_step21"] = "primary_doi_test"
    primary["prediction_source_step21"] = "sigma_pred_ML_primary_test"
    primary = add_pf_zt(primary, "sigma_pred_ML_S_per_m_step20")

    downstream = downstream.copy()
    downstream["model_name"] = downstream.get("final_model_name_step19", recommended_model)
    downstream["split_name"] = "downstream_all_samples"
    downstream["split_role"] = "downstream"
    downstream["evaluation_scope_step21"] = "downstream_all_samples"
    downstream["prediction_source_step21"] = "sigma_pred_ML_downstream_all_samples"
    downstream["downstream_evaluation_note_step21"] = "not_unbiased_evaluation; final model trained on all recommended ML samples"
    downstream = add_pf_zt(downstream, "sigma_pred_ML_all_samples_S_per_m_step20")

    primary_sample = sample_aggregate(primary, "primary_doi_test")
    downstream_sample = sample_aggregate(downstream, "downstream_all_samples", "not_unbiased_evaluation")
    metrics = model_metrics(add_pf_zt(primary_all[primary_all["split_name"].eq(args.primary_split) & primary_all["split_role"].eq("test")].copy(), "sigma_pred_ML_S_per_m_step20"))
    high_cls = high_zt_classification(primary_sample, downstream_sample, [0.5, 1.0, 1.5])
    vs_fit = vs_fitting(primary, primary_sample, args.zt_threshold)
    material_summary, np_summary = summarize_groups(primary_sample)

    problem_rows = primary[
        (pd.to_numeric(primary["pf_ML_relative_error_step21"], errors="coerce") > 1.0)
        | (pd.to_numeric(primary["zt_ML_vs_obs_relative_error_step21"], errors="coerce") > 1.0)
        | (pd.to_numeric(primary["zt_ML_vs_calc_relative_error_step21"], errors="coerce") > 1.0)
        | (~primary["pf_pred_ML_status_step21"].eq("ok"))
        | (~primary["zt_pred_ML_status_step21"].eq("ok"))
    ].copy()
    problem_samples = primary_sample[
        primary_sample["zt_ML_vs_obs_quality_step21"].eq("poor")
        | primary_sample["zt_ML_vs_calc_quality_step21"].eq("poor")
        | (pd.to_numeric(primary_sample["zt_ML_vs_obs_mape_step21"], errors="coerce") > 1.0)
        | (pd.to_numeric(primary_sample["zt_ML_vs_calc_mape_step21"], errors="coerce") > 1.0)
    ].copy()
    problem_samples["pf_zt_ML_problem_reason_step21"] = "large ZT ML error; ML tau prediction propagated to ZT error; direct fitting much better than ML"
    problem_samples["needs_review_for_ML_pf_zt_step21"] = True

    candidates = downstream_sample[
        (pd.to_numeric(downstream_sample["zt_pred_ML_max_step21"], errors="coerce") >= 1.0)
        | (pd.to_numeric(downstream_sample["zt_obs_max_step21"], errors="coerce") >= 1.0)
        | (pd.to_numeric(downstream_sample["zt_calc_from_obs_max_step21"], errors="coerce") >= 1.0)
    ].copy()
    max_row_cols = ["sample_key", "sigma_pred_ML_for_pf_zt_S_per_m_step21", "seebeck_obs_V_per_K_step11", "kappa_obs_W_per_mK_step11"]
    valid_downstream_zt = downstream[pd.to_numeric(downstream["zt_pred_ML_step21"], errors="coerce").notna()].copy()
    if len(valid_downstream_zt):
        idx = valid_downstream_zt.groupby("sample_key")["zt_pred_ML_step21"].idxmax()
        max_rows = valid_downstream_zt.loc[idx, [c for c in max_row_cols if c in valid_downstream_zt.columns]]
    else:
        max_rows = pd.DataFrame(columns=[c for c in max_row_cols if c in downstream.columns])
    candidates = candidates.merge(max_rows, on="sample_key", how="left")

    np_changed = 0
    sintering_changed = 0
    report = make_report(
        {"primary": len(primary_all), "downstream": len(downstream)},
        recommended_model,
        args.primary_split,
        primary,
        downstream,
        primary_sample,
        downstream_sample,
        metrics,
        high_cls,
        vs_fit,
        problem_samples,
        candidates,
        np_changed,
        sintering_changed,
    )
    notes = make_notes()

    primary.to_csv(output_dir / "thermoelectric_ml_primary_test_predictions_step21.csv", index=False)
    downstream.to_csv(output_dir / "thermoelectric_ml_downstream_predictions_step21.csv", index=False)
    primary_sample.to_csv(output_dir / "pf_zt_ml_primary_sample_results_step21.csv", index=False)
    downstream_sample.to_csv(output_dir / "pf_zt_ml_downstream_sample_results_step21.csv", index=False)
    metrics.to_csv(output_dir / "pf_zt_ml_model_metrics_step21.csv", index=False)
    high_cls.to_csv(output_dir / "pf_zt_ml_high_performance_classification_step21.csv", index=False)
    vs_fit.to_csv(output_dir / "pf_zt_ml_vs_fitting_comparison_step21.csv", index=False)
    material_summary.to_csv(output_dir / "pf_zt_ml_material_summary_step21.csv", index=False)
    np_summary.to_csv(output_dir / "pf_zt_ml_np_summary_step21.csv", index=False)
    problem_rows.to_csv(output_dir / "pf_zt_ml_problem_rows_step21.csv", index=False)
    problem_samples.to_csv(output_dir / "pf_zt_ml_problem_samples_step21.csv", index=False)
    candidates.to_csv(output_dir / "pf_zt_ml_candidate_samples_step21.csv", index=False)
    (output_dir / "step21_pf_zt_ml_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step21_pf_zt_ml_notes.md").write_text(notes, encoding="utf-8")

    write_excel(
        output_dir / "starrydata2_step21_pf_zt_ml_prediction.xlsx",
        {
            "primary_sample_results": primary_sample,
            "downstream_sample_results": downstream_sample,
            "model_metrics": metrics,
            "high_zt_classification": high_cls,
            "ml_vs_fitting": vs_fit,
            "material_summary": material_summary,
            "np_summary": np_summary,
            "problem_samples": problem_samples,
            "candidate_samples": candidates,
            "pf_zt_ml_report": report,
        },
    )

    rec_metrics = metrics[metrics["model_name"].eq(recommended_model)].iloc[0]
    cls1 = high_cls[(high_cls["evaluation_scope_step21"].eq("primary_doi_test")) & (high_cls["threshold"].eq(1.0))].iloc[0]
    fit_f1 = vs_fit[vs_fit["metric_name"].eq("ZT>=1 F1")]["fitting_value"].iloc[0]
    down_pred_ge1 = int((pd.to_numeric(downstream_sample["zt_pred_ML_max_step21"], errors="coerce") >= 1.0).sum())
    print("Done.")
    print("Created:")
    for name in [
        "thermoelectric_ml_primary_test_predictions_step21.csv",
        "thermoelectric_ml_downstream_predictions_step21.csv",
        "pf_zt_ml_primary_sample_results_step21.csv",
        "pf_zt_ml_downstream_sample_results_step21.csv",
        "pf_zt_ml_model_metrics_step21.csv",
        "pf_zt_ml_high_performance_classification_step21.csv",
        "pf_zt_ml_vs_fitting_comparison_step21.csv",
        "pf_zt_ml_material_summary_step21.csv",
        "pf_zt_ml_np_summary_step21.csv",
        "pf_zt_ml_problem_rows_step21.csv",
        "pf_zt_ml_problem_samples_step21.csv",
        "pf_zt_ml_candidate_samples_step21.csv",
        "step21_pf_zt_ml_report.txt",
        "step21_pf_zt_ml_notes.md",
        "starrydata2_step21_pf_zt_ml_prediction.xlsx",
    ]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"recommended model: {recommended_model}")
    print(f"primary PF eval rows: {int(primary['pf_pred_ML_status_step21'].eq('ok').sum())}")
    print(f"primary ZT eval rows: {int(primary['zt_pred_ML_status_step21'].eq('ok').sum())}")
    print(f"primary PF MAPE: {rec_metrics['pf_mape_step21']}")
    print(f"primary ZT vs obs MAPE: {rec_metrics['zt_vs_obs_mape_step21']}")
    print(f"primary ZT vs calc MAPE: {rec_metrics['zt_vs_calc_mape_step21']}")
    print(f"primary ZT>=1 precision: {cls1['precision']}")
    print(f"primary ZT>=1 recall: {cls1['recall']}")
    print(f"primary ZT>=1 F1: {cls1['f1']}")
    print(f"fitting ZT>=1 F1: {fit_f1}")
    print(f"ML ZT>=1 F1: {cls1['f1']}")
    print(f"downstream ZT_pred_ML>=1 samples: {down_pred_ge1}")
    print(f"candidate samples: {len(candidates)}")
    print(f"problem samples: {len(problem_samples)}")
    print(f"n/p changed rows: {np_changed}")
    print(f"sintering changed rows: {sintering_changed}")


if __name__ == "__main__":
    main()
