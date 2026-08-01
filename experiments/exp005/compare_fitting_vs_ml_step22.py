import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP21_DIR = "data/output/starrydata2_step21_pf_zt_ml_prediction"
DEFAULT_STEP20_DIR = "data/output/starrydata2_step20_sigma_ml_prediction"
DEFAULT_STEP14_DIR = "data/output/starrydata2_step14_pf_zt_prediction"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step22_fitting_vs_ml_comparison"
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
    parser = argparse.ArgumentParser(description="Compare direct fitting and ML tau_eff versions for sigma/PF/ZT.")
    parser.add_argument("--step21_dir", default=DEFAULT_STEP21_DIR)
    parser.add_argument("--step20_dir", default=DEFAULT_STEP20_DIR)
    parser.add_argument("--step14_dir", default=DEFAULT_STEP14_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--primary_scope", default="primary_doi_test")
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


def safe_log_error(pred, obs):
    pred = pd.to_numeric(pred, errors="coerce")
    obs = pd.to_numeric(obs, errors="coerce")
    ok = np.isfinite(pred) & np.isfinite(obs) & (pred > 0) & (obs > 0)
    return np.where(ok, np.log(pred) - np.log(obs), np.nan)


def regression_metrics(df, obs_col, pred_col):
    obs = pd.to_numeric(df[obs_col], errors="coerce") if obs_col in df.columns else pd.Series(dtype=float)
    pred = pd.to_numeric(df[pred_col], errors="coerce") if pred_col in df.columns else pd.Series(dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred) & (obs > 0) & (pred > 0)
    n_rows = int(ok.sum())
    n_samples = int(df.loc[ok, "sample_key"].nunique()) if "sample_key" in df.columns else 0
    if n_rows == 0:
        return empty_metrics(n_rows, n_samples)
    err = pred[ok] - obs[ok]
    rel = np.abs(err) / np.maximum(np.abs(obs[ok]), EPS)
    log_ok = ok
    log_err = np.log(pred[log_ok]) - np.log(obs[log_ok])
    if log_ok.sum() > 1:
        yt = np.log(obs[log_ok])
        yp = np.log(pred[log_ok])
        denom = np.sum((yt - yt.mean()) ** 2)
        log_r2 = float(1 - np.sum((yt - yp) ** 2) / denom) if denom > 0 else np.nan
        pearson = float(pd.Series(yt).corr(pd.Series(yp), method="pearson"))
        spearman = float(pd.Series(yt).corr(pd.Series(yp), method="spearman"))
    else:
        log_r2 = pearson = spearman = np.nan
    ratio = pred[log_ok] / obs[log_ok]
    return {
        "n_rows": n_rows,
        "n_samples": n_samples,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "mape": float(np.mean(rel)),
        "log_mae": float(np.mean(np.abs(log_err))) if len(log_err) else np.nan,
        "log_rmse": float(math.sqrt(np.mean(log_err**2))) if len(log_err) else np.nan,
        "log_r2": log_r2,
        "pearson": pearson,
        "spearman": spearman,
        "within_25pct_rate": float(np.mean(rel <= 0.25)),
        "within_50pct_rate": float(np.mean(rel <= 0.50)),
        "within_factor_2_rate": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if len(ratio) else np.nan,
    }


def empty_metrics(n_rows=0, n_samples=0):
    return {
        "n_rows": n_rows,
        "n_samples": n_samples,
        "mae": np.nan,
        "rmse": np.nan,
        "mape": np.nan,
        "log_mae": np.nan,
        "log_rmse": np.nan,
        "log_r2": np.nan,
        "pearson": np.nan,
        "spearman": np.nan,
        "within_25pct_rate": np.nan,
        "within_50pct_rate": np.nan,
        "within_factor_2_rate": np.nan,
    }


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
        "balanced_accuracy": np.nanmean([recall, specificity]),
    }


def interpretation(metric, fit, ml):
    if pd.isna(fit) or pd.isna(ml):
        return "not evaluated"
    worse = ml > fit if not any(k in metric for k in ["precision", "recall", "f1", "accuracy", "within", "spearman"]) else ml < fit
    if not worse:
        return "ML close to fitting version"
    ratio = ml / fit if fit not in [0, np.nan] and pd.notna(fit) else np.nan
    if pd.notna(ratio) and ratio > 2:
        return "ML substantially worse; tau_eff ML model likely insufficient"
    return "ML worse than direct fitting, as expected"


def comparison_category(row):
    gap = row.get("zt_obs_mape_gap_ML_minus_fitting_step22")
    if pd.isna(gap):
        return "not_evaluated"
    if gap < -0.1:
        return "ML_better_than_fitting"
    if gap <= 0.1:
        return "ML_close_to_fitting"
    if gap <= 0.5:
        return "ML_moderately_worse"
    return "ML_much_worse"


def build_row_level(primary):
    row = primary.copy()
    row["sigma_fitting_log_error_step22"] = safe_log_error(row["sigma_pred_S_per_m_step12"], row["sigma_obs_S_per_m_step11"])
    row["sigma_ML_log_error_step22"] = safe_log_error(row["sigma_pred_ML_for_pf_zt_S_per_m_step21"], row["sigma_obs_S_per_m_step11"])
    row["sigma_ML_minus_fitting_abs_log_error_step22"] = np.abs(row["sigma_ML_log_error_step22"]) - np.abs(row["sigma_fitting_log_error_step22"])
    row["pf_fitting_log_error_step22"] = safe_log_error(row["power_factor_pred_fitting_W_per_mK2_step21"], row["power_factor_obs_W_per_mK2_step21"])
    row["pf_ML_log_error_step22"] = safe_log_error(row["power_factor_pred_ML_W_per_mK2_step21"], row["power_factor_obs_W_per_mK2_step21"])
    row["pf_ML_minus_fitting_abs_log_error_step22"] = np.abs(row["pf_ML_log_error_step22"]) - np.abs(row["pf_fitting_log_error_step22"])
    row["zt_obs_fitting_log_error_step22"] = safe_log_error(row["zt_pred_fitting_step21"], row["zt_obs_dimensionless_step11"])
    row["zt_obs_ML_log_error_step22"] = safe_log_error(row["zt_pred_ML_step21"], row["zt_obs_dimensionless_step11"])
    row["zt_obs_ML_minus_fitting_abs_log_error_step22"] = np.abs(row["zt_obs_ML_log_error_step22"]) - np.abs(row["zt_obs_fitting_log_error_step22"])
    row["zt_calc_fitting_log_error_step22"] = safe_log_error(row["zt_pred_fitting_step21"], row["zt_calc_from_obs_step11"])
    row["zt_calc_ML_log_error_step22"] = safe_log_error(row["zt_pred_ML_step21"], row["zt_calc_from_obs_step11"])
    row["zt_calc_ML_minus_fitting_abs_log_error_step22"] = np.abs(row["zt_calc_ML_log_error_step22"]) - np.abs(row["zt_calc_fitting_log_error_step22"])
    ok_any = row[["sigma_ML_log_error_step22", "pf_ML_log_error_step22", "zt_obs_ML_log_error_step22", "zt_calc_ML_log_error_step22"]].notna().any(axis=1)
    row["comparison_row_quality_step22"] = np.where(ok_any, "evaluated", "not_evaluated")
    return row


def metric_comparison(row_level):
    specs = [
        ("sigma", "sigma_obs_S_per_m_step11", "sigma_pred_S_per_m_step12", "sigma_pred_ML_for_pf_zt_S_per_m_step21"),
        ("PF", "power_factor_obs_W_per_mK2_step21", "power_factor_pred_fitting_W_per_mK2_step21", "power_factor_pred_ML_W_per_mK2_step21"),
        ("ZT_vs_obs", "zt_obs_dimensionless_step11", "zt_pred_fitting_step21", "zt_pred_ML_step21"),
        ("ZT_vs_calc", "zt_calc_from_obs_step11", "zt_pred_fitting_step21", "zt_pred_ML_step21"),
    ]
    rows = []
    for target, obs, fit, ml in specs:
        for version, pred in [("direct_fitting", fit), ("ml_tau_prediction", ml)]:
            m = regression_metrics(row_level, obs, pred)
            rows.append({"target_quantity": target, "comparison_reference": obs, "version": version, **m, "note": "primary DOI test"})
    return pd.DataFrame(rows)


def overall_comparison(metric_df, high_cls):
    pairs = []
    lookup = metric_df.set_index(["target_quantity", "version"])
    for metric_name, target, field in [
        ("sigma_log_rmse", "sigma", "log_rmse"),
        ("sigma_mape", "sigma", "mape"),
        ("sigma_within_factor_2_rate", "sigma", "within_factor_2_rate"),
        ("pf_log_rmse", "PF", "log_rmse"),
        ("pf_mape", "PF", "mape"),
        ("zt_vs_obs_log_rmse", "ZT_vs_obs", "log_rmse"),
        ("zt_vs_obs_mape", "ZT_vs_obs", "mape"),
        ("zt_vs_obs_spearman", "ZT_vs_obs", "spearman"),
        ("zt_vs_calc_log_rmse", "ZT_vs_calc", "log_rmse"),
        ("zt_vs_calc_mape", "ZT_vs_calc", "mape"),
        ("zt_vs_calc_spearman", "ZT_vs_calc", "spearman"),
    ]:
        fit = lookup.loc[(target, "direct_fitting"), field]
        ml = lookup.loc[(target, "ml_tau_prediction"), field]
        pairs.append({
            "metric_name": metric_name,
            "fitting_value": fit,
            "ml_value": ml,
            "ml_minus_fitting": ml - fit if pd.notna(ml) and pd.notna(fit) else np.nan,
            "ml_divided_by_fitting": ml / fit if pd.notna(ml) and pd.notna(fit) and fit != 0 else np.nan,
            "interpretation_step22": interpretation(metric_name, fit, ml),
        })
    cls1 = high_cls[high_cls["threshold"].eq(1.0)]
    for field in ["precision", "recall", "f1", "accuracy"]:
        fit = cls1[cls1["version"].eq("direct_fitting")][field].iloc[0]
        ml = cls1[cls1["version"].eq("ml_tau_prediction")][field].iloc[0]
        pairs.append({
            "metric_name": f"zt_ge_1_{field}",
            "fitting_value": fit,
            "ml_value": ml,
            "ml_minus_fitting": ml - fit if pd.notna(ml) and pd.notna(fit) else np.nan,
            "ml_divided_by_fitting": ml / fit if pd.notna(ml) and pd.notna(fit) and fit != 0 else np.nan,
            "interpretation_step22": interpretation(field, fit, ml),
        })
    return pd.DataFrame(pairs)


def sample_level(row_level):
    rows = []
    for key, g in row_level.groupby("sample_key", dropna=False):
        first = g.iloc[0]
        sig_fit = regression_metrics(g, "sigma_obs_S_per_m_step11", "sigma_pred_S_per_m_step12")
        sig_ml = regression_metrics(g, "sigma_obs_S_per_m_step11", "sigma_pred_ML_for_pf_zt_S_per_m_step21")
        pf_fit = regression_metrics(g, "power_factor_obs_W_per_mK2_step21", "power_factor_pred_fitting_W_per_mK2_step21")
        pf_ml = regression_metrics(g, "power_factor_obs_W_per_mK2_step21", "power_factor_pred_ML_W_per_mK2_step21")
        zto_fit = regression_metrics(g, "zt_obs_dimensionless_step11", "zt_pred_fitting_step21")
        zto_ml = regression_metrics(g, "zt_obs_dimensionless_step11", "zt_pred_ML_step21")
        ztc_fit = regression_metrics(g, "zt_calc_from_obs_step11", "zt_pred_fitting_step21")
        ztc_ml = regression_metrics(g, "zt_calc_from_obs_step11", "zt_pred_ML_step21")
        row = {
            "sample_key": key,
            "DOI": first.get("DOI"),
            "doi_url": first.get("doi_url"),
            "paper_title": first.get("paper_title"),
            "sample_id": first.get("sample_id"),
            "composition": first.get("composition"),
            "material_system": first.get("material_system"),
            "n_or_p": first.get("n_or_p"),
            "n_rows_step22": len(g),
            "n_pf_eval_rows_step22": pf_ml["n_rows"],
            "n_zt_obs_eval_rows_step22": zto_ml["n_rows"],
            "n_zt_calc_eval_rows_step22": ztc_ml["n_rows"],
            "sigma_fitting_log_rmse_step22": sig_fit["log_rmse"],
            "sigma_ML_log_rmse_step22": sig_ml["log_rmse"],
            "sigma_log_rmse_gap_ML_minus_fitting_step22": sig_ml["log_rmse"] - sig_fit["log_rmse"] if pd.notna(sig_ml["log_rmse"]) and pd.notna(sig_fit["log_rmse"]) else np.nan,
            "pf_fitting_mape_step22": pf_fit["mape"],
            "pf_ML_mape_step22": pf_ml["mape"],
            "pf_mape_gap_ML_minus_fitting_step22": pf_ml["mape"] - pf_fit["mape"] if pd.notna(pf_ml["mape"]) and pd.notna(pf_fit["mape"]) else np.nan,
            "zt_obs_fitting_mape_step22": zto_fit["mape"],
            "zt_obs_ML_mape_step22": zto_ml["mape"],
            "zt_obs_mape_gap_ML_minus_fitting_step22": zto_ml["mape"] - zto_fit["mape"] if pd.notna(zto_ml["mape"]) and pd.notna(zto_fit["mape"]) else np.nan,
            "zt_calc_fitting_mape_step22": ztc_fit["mape"],
            "zt_calc_ML_mape_step22": ztc_ml["mape"],
            "zt_calc_mape_gap_ML_minus_fitting_step22": ztc_ml["mape"] - ztc_fit["mape"] if pd.notna(ztc_ml["mape"]) and pd.notna(ztc_fit["mape"]) else np.nan,
            "zt_obs_max_step22": pd.to_numeric(g["zt_obs_dimensionless_step11"], errors="coerce").max(),
            "zt_pred_fitting_max_step22": pd.to_numeric(g["zt_pred_fitting_step21"], errors="coerce").max(),
            "zt_pred_ML_max_step22": pd.to_numeric(g["zt_pred_ML_step21"], errors="coerce").max(),
            "zt_calc_from_obs_max_step22": pd.to_numeric(g["zt_calc_from_obs_step11"], errors="coerce").max(),
            "sintering_method": first.get("sintering_method"),
            "sintering_checked": first.get("sintering_checked"),
            "record_checked": first.get("record_checked"),
        }
        row["comparison_category_step22"] = comparison_category(row)
        row["comparison_note_step22"] = "primary DOI test sample aggregate"
        rows.append(row)
    return pd.DataFrame(rows)


def high_zt(sample_df, thresholds):
    rows = []
    for threshold in thresholds:
        obs = pd.to_numeric(sample_df["zt_obs_max_step22"], errors="coerce") >= threshold
        for version, pred_col in [("direct_fitting", "zt_pred_fitting_max_step22"), ("ml_tau_prediction", "zt_pred_ML_max_step22")]:
            counts = classification_counts(obs, pd.to_numeric(sample_df[pred_col], errors="coerce") >= threshold)
            rows.append({"threshold": threshold, "version": version, **counts, "note": "observed label: zt_obs_max_step22"})
    return pd.DataFrame(rows)


def ranking(sample_df):
    rows = []
    specs = [
        ("fitting ZT max vs observed ZT max", "direct_fitting", "zt_pred_fitting_max_step22", "zt_obs_max_step22"),
        ("ML ZT max vs observed ZT max", "ml_tau_prediction", "zt_pred_ML_max_step22", "zt_obs_max_step22"),
        ("fitting ZT max vs calculated ZT max", "direct_fitting", "zt_pred_fitting_max_step22", "zt_calc_from_obs_max_step22"),
        ("ML ZT max vs calculated ZT max", "ml_tau_prediction", "zt_pred_ML_max_step22", "zt_calc_from_obs_max_step22"),
    ]
    for name, version, pred_col, obs_col in specs:
        sub = sample_df[[pred_col, obs_col, "sample_key"]].copy()
        sub[pred_col] = pd.to_numeric(sub[pred_col], errors="coerce")
        sub[obs_col] = pd.to_numeric(sub[obs_col], errors="coerce")
        sub = sub.dropna()
        pearson = sub[pred_col].corr(sub[obs_col], method="pearson") if len(sub) > 1 else np.nan
        spearman = sub[pred_col].rank().corr(sub[obs_col].rank(), method="pearson") if len(sub) > 1 else np.nan
        for k in [50, 100, 300]:
            obs_top = set(sub.sort_values(obs_col, ascending=False).head(k)["sample_key"])
            pred_top = set(sub.sort_values(pred_col, ascending=False).head(k)["sample_key"])
            overlap = len(obs_top & pred_top)
            rows.append({
                "comparison_name": name,
                "version": version,
                "n_samples": len(sub),
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "top_k": k,
                "top_k_overlap_count": overlap,
                "top_k_overlap_rate": overlap / min(k, len(sub)) if len(sub) else np.nan,
                "note": f"top {k} overlap by {obs_col} and {pred_col}",
            })
    return pd.DataFrame(rows)


def group_summaries(sample_df):
    mat = sample_df.groupby(["material_system", "n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        zt_eval_sample_count=("zt_obs_ML_mape_step22", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        median_sigma_fitting_log_rmse=("sigma_fitting_log_rmse_step22", "median"),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_zt_obs_fitting_mape=("zt_obs_fitting_mape_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        fitting_ZT_ge_1_sample_count=("zt_pred_fitting_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
        ML_ZT_ge_1_sample_count=("zt_pred_ML_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
        observed_ZT_ge_1_sample_count=("zt_obs_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
        ML_close_to_fitting_count=("comparison_category_step22", lambda s: int((s == "ML_close_to_fitting").sum())),
        ML_much_worse_count=("comparison_category_step22", lambda s: int((s == "ML_much_worse").sum())),
        ML_better_than_fitting_count=("comparison_category_step22", lambda s: int((s == "ML_better_than_fitting").sum())),
    ).reset_index()
    mat["median_sigma_gap"] = mat["median_sigma_ML_log_rmse"] - mat["median_sigma_fitting_log_rmse"]
    mat["median_zt_obs_gap"] = mat["median_zt_obs_ML_mape"] - mat["median_zt_obs_fitting_mape"]
    mat["interpretation_step22"] = np.where(mat["median_zt_obs_gap"] > 0.5, "ML much worse in this group", "ML close/moderate in this group")
    np_sum = sample_df.groupby(["n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        zt_eval_sample_count=("zt_obs_ML_mape_step22", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        median_sigma_fitting_log_rmse=("sigma_fitting_log_rmse_step22", "median"),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_zt_obs_fitting_mape=("zt_obs_fitting_mape_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        fitting_ZT_ge_1_sample_count=("zt_pred_fitting_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
        ML_ZT_ge_1_sample_count=("zt_pred_ML_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
        observed_ZT_ge_1_sample_count=("zt_obs_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= 1).sum())),
    ).reset_index()
    np_sum["interpretation_step22"] = np.where(np_sum["median_zt_obs_ML_mape"] > np_sum["median_zt_obs_fitting_mape"], "ML worse than fitting", "ML close/better")
    return mat, np_sum


def problem_samples(sample_df, threshold):
    obs_high = pd.to_numeric(sample_df["zt_obs_max_step22"], errors="coerce") >= threshold
    ml_high = pd.to_numeric(sample_df["zt_pred_ML_max_step22"], errors="coerce") >= threshold
    out = sample_df[
        sample_df["comparison_category_step22"].eq("ML_much_worse")
        | (pd.to_numeric(sample_df["zt_obs_ML_mape_step22"], errors="coerce") > 1.0)
        | (pd.to_numeric(sample_df["sigma_ML_log_rmse_step22"], errors="coerce") > 1.0)
        | (obs_high & ~ml_high)
        | (~obs_high & ml_high)
    ].copy()
    reasons = []
    for _, row in out.iterrows():
        r = []
        if row.get("comparison_category_step22") == "ML_much_worse":
            r.append("ML tau prediction much worse than fitting")
        if row.get("sigma_ML_log_rmse_step22", 0) > 1.0:
            r.append("sigma ML error propagated to ZT error")
        obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step22")]), errors="coerce").iloc[0] >= threshold
        pred = pd.to_numeric(pd.Series([row.get("zt_pred_ML_max_step22")]), errors="coerce").iloc[0] >= threshold
        if obs and not pred:
            r.append("ML missed high ZT")
        if (not obs) and pred:
            r.append("ML false high-ZT prediction")
        r.append("possible missing structure/additive feature")
        reasons.append("; ".join(dict.fromkeys(r)))
    out["step22_problem_reason"] = reasons
    out["needs_step23_error_analysis"] = True
    return out


def degradation(overall, sample_df, material_summary, np_summary):
    lookup = overall.set_index("metric_name")
    items = [
        ("ML sigma log RMSE vs fitting sigma log RMSE", lookup.loc["sigma_log_rmse", "ml_minus_fitting"], "step22_overall_comparison.csv"),
        ("ML PF MAPE vs fitting PF MAPE", lookup.loc["pf_mape", "ml_minus_fitting"], "step22_overall_comparison.csv"),
        ("ML ZT vs obs MAPE vs fitting ZT vs obs MAPE", lookup.loc["zt_vs_obs_mape", "ml_minus_fitting"], "step22_overall_comparison.csv"),
        ("ML ZT>=1 F1 vs fitting ZT>=1 F1", lookup.loc["zt_ge_1_f1", "ml_minus_fitting"], "step22_high_zt_classification_comparison.csv"),
        ("ML much worse sample count", int((sample_df["comparison_category_step22"] == "ML_much_worse").sum()), "step22_sample_level_comparison.csv"),
        ("ML better than fitting sample count", int((sample_df["comparison_category_step22"] == "ML_better_than_fitting").sum()), "step22_sample_level_comparison.csv"),
        ("material_systems with largest degradation", material_summary.sort_values("median_zt_obs_gap", ascending=False).head(5).to_json(orient="records", force_ascii=False), "step22_material_summary_comparison.csv"),
        ("n_or_p with largest degradation", np_summary.assign(gap=np_summary["median_zt_obs_ML_mape"] - np_summary["median_zt_obs_fitting_mape"]).sort_values("gap", ascending=False).head(3).to_json(orient="records", force_ascii=False), "step22_np_summary_comparison.csv"),
        ("possible cause: tau_eff ML model underperformance", "likely", "step22_comparison_report.txt"),
        ("possible cause: insufficient additive/structure features", "likely", "step22_comparison_report.txt"),
        ("possible cause: DOI split generalization difficulty", "likely", "step22_comparison_report.txt"),
    ]
    return pd.DataFrame([{"analysis_item": a, "value": v, "interpretation_step22": "ML degradation relative to fitting comparison", "related_file": f} for a, v, f in items])


def recommended_interpretation(overall):
    def val(metric):
        return overall[overall["metric_name"].eq(metric)]["ml_minus_fitting"].iloc[0]
    return pd.DataFrame([
        {"topic": "direct fitting upper reference", "recommended_interpretation": "fitting版はsigma_obsを使ってtau_effを直接fittingしているため、上限性能に近い", "supporting_metric": "sigma_log_rmse gap", "supporting_value": val("sigma_log_rmse"), "caution": "not an unknown-material prediction baseline"},
        {"topic": "ML task meaning", "recommended_interpretation": "ML版は材料情報からtau_effを予測するため、fitting版より悪くなるのが自然", "supporting_metric": "zt_vs_obs_mape gap", "supporting_value": val("zt_vs_obs_mape"), "caution": "feature information is limited"},
        {"topic": "current ML performance", "recommended_interpretation": "Step20/21の結果から、現時点のML版はfitting版より大きく劣る", "supporting_metric": "pf_mape gap", "supporting_value": val("pf_mape"), "caution": "use primary DOI test for interpretation"},
        {"topic": "cause candidates", "recommended_interpretation": "ML版の性能悪化は、tau_eff予測モデルの不足、特徴量不足、構造・添加物情報不足が原因候補", "supporting_metric": "problem sample count", "supporting_value": "", "caution": "Step23 should analyze causes"},
        {"topic": "next step", "recommended_interpretation": "Step23では誤差原因を材料系・n/p型・添加物・構造・焼結方法の観点から分析する", "supporting_metric": "planned analysis", "supporting_value": "", "caution": ""},
        {"topic": "S and kappa", "recommended_interpretation": "Step21/22でもSとkappaは予測していない", "supporting_metric": "method caveat", "supporting_value": "", "caution": "PF/ZT use observed S/kappa"},
    ])


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


def notes():
    return """# Step22 Fitting vs ML Comparison Notes

## Purpose
Compare direct fitting and ML tau_eff prediction versions for sigma, PF, ZT, high-ZT classification, and ranking.

## What is Compared
The comparison uses the same primary DOI test rows and temperature points where possible.

## Direct Fitting Version
The fitting version uses Step12 fitted tau_eff derived directly from sigma observations.

## ML tau_eff Prediction Version
The ML version uses Step19 predicted tau_eff from material features, then Step20/21 calculated sigma/PF/ZT.

## Main Results
See `step22_comparison_report.txt`.

## Interpretation
The fitting version is an upper-reference style result, while the ML version is closer to unknown-material prediction.

## Why ML Can Be Worse Than Fitting
The ML model has limited material features and does not use sigma observations to fit tau_eff for the target sample.

## Important Caveats
The fitting version uses sigma observations to fit tau_eff, so it is not a fair unknown-material prediction baseline.
The ML version predicts tau_eff from material features and is closer to the intended ML task.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Seebeck coefficient and thermal conductivity are not predicted in either version.
PF and ZT are computed using predicted sigma and observed S/kappa.

## Next Step
Step23 should analyze error causes by material system, n/p type, additives, structure, nanocarbon, rare-metal/toxicity flags, and sintering status.
"""


def make_report(row_level, sample_df, overall, high_cls, ranking_df, problems, np_changed, sintering_changed):
    lookup = overall.set_index("metric_name")
    def f(metric, col):
        return lookup.loc[metric, col]
    cls1 = high_cls[high_cls["threshold"].eq(1.0)]
    fit_cls = cls1[cls1["version"].eq("direct_fitting")].iloc[0]
    ml_cls = cls1[cls1["version"].eq("ml_tau_prediction")].iloc[0]
    rank100_fit = ranking_df[(ranking_df["comparison_name"].eq("fitting ZT max vs observed ZT max")) & (ranking_df["top_k"].eq(100))].iloc[0]
    rank100_ml = ranking_df[(ranking_df["comparison_name"].eq("ML ZT max vs observed ZT max")) & (ranking_df["top_k"].eq(100))].iloc[0]
    counts = sample_df["comparison_category_step22"].value_counts().to_dict()
    lines = [
        "Step22 fitting vs ML comparison report",
        "",
        f"Input thermoelectric_ml_primary_test_predictions_step21 rows: {len(row_level)}",
        f"Comparison sample count: {sample_df['sample_key'].nunique()}",
        f"Comparison temperature row count: {len(row_level)}",
        "",
        "sigma comparison:",
        f"- fitting sigma log RMSE: {f('sigma_log_rmse', 'fitting_value')}",
        f"- ML sigma log RMSE: {f('sigma_log_rmse', 'ml_value')}",
        f"- ML vs fitting gap: {f('sigma_log_rmse', 'ml_minus_fitting')}",
        f"- fitting sigma MAPE: {f('sigma_mape', 'fitting_value')}",
        f"- ML sigma MAPE: {f('sigma_mape', 'ml_value')}",
        "",
        "PF comparison:",
        f"- fitting PF MAPE: {f('pf_mape', 'fitting_value')}",
        f"- ML PF MAPE: {f('pf_mape', 'ml_value')}",
        f"- fitting PF log RMSE: {f('pf_log_rmse', 'fitting_value')}",
        f"- ML PF log RMSE: {f('pf_log_rmse', 'ml_value')}",
        "",
        "ZT vs observed comparison:",
        f"- fitting ZT vs obs MAPE: {f('zt_vs_obs_mape', 'fitting_value')}",
        f"- ML ZT vs obs MAPE: {f('zt_vs_obs_mape', 'ml_value')}",
        f"- fitting ZT vs obs log RMSE: {f('zt_vs_obs_log_rmse', 'fitting_value')}",
        f"- ML ZT vs obs log RMSE: {f('zt_vs_obs_log_rmse', 'ml_value')}",
        "",
        "ZT vs calculated comparison:",
        f"- fitting ZT vs calc MAPE: {f('zt_vs_calc_mape', 'fitting_value')}",
        f"- ML ZT vs calc MAPE: {f('zt_vs_calc_mape', 'ml_value')}",
        f"- fitting ZT vs calc log RMSE: {f('zt_vs_calc_log_rmse', 'fitting_value')}",
        f"- ML ZT vs calc log RMSE: {f('zt_vs_calc_log_rmse', 'ml_value')}",
        "",
        "ZT>=1 classification:",
        f"- fitting precision / recall / F1: {fit_cls['precision']} / {fit_cls['recall']} / {fit_cls['f1']}",
        f"- ML precision / recall / F1: {ml_cls['precision']} / {ml_cls['recall']} / {ml_cls['f1']}",
        "",
        "Ranking:",
        f"- fitting ZT Spearman: {rank100_fit['spearman_corr']}",
        f"- ML ZT Spearman: {rank100_ml['spearman_corr']}",
        f"- fitting top100 overlap: {rank100_fit['top_k_overlap_count']}",
        f"- ML top100 overlap: {rank100_ml['top_k_overlap_count']}",
        "",
        "Comparison category:",
        f"- ML_close_to_fitting count: {counts.get('ML_close_to_fitting', 0)}",
        f"- ML_moderately_worse count: {counts.get('ML_moderately_worse', 0)}",
        f"- ML_much_worse count: {counts.get('ML_much_worse', 0)}",
        f"- ML_better_than_fitting count: {counts.get('ML_better_than_fitting', 0)}",
        "",
        f"Problem samples count: {len(problems)}",
        f"n/p changed rows: {np_changed}",
        f"sintering changed rows: {sintering_changed}",
        "",
        "Notes:",
        "Step22 did not make new predictions, refit tau_eff, or retrain ML models.",
        "Step22 compared existing fitting and ML results.",
        "The fitting version uses electrical conductivity data directly, so better performance than ML is natural.",
        "The ML version predicts tau_eff from material information only and is closer to unknown-material prediction.",
        "Seebeck coefficient and thermal conductivity are not predicted in either version.",
    ]
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    step21 = Path(args.step21_dir)

    primary = read_csv(step21 / "thermoelectric_ml_primary_test_predictions_step21.csv")
    read_csv(step21 / "pf_zt_ml_primary_sample_results_step21.csv")
    read_csv(step21 / "pf_zt_ml_high_performance_classification_step21.csv")
    read_csv(step21 / "pf_zt_ml_vs_fitting_comparison_step21.csv")
    read_csv(step21 / "pf_zt_ml_problem_samples_step21.csv")

    required = [
        "sample_key", "temperature_K", "sigma_obs_S_per_m_step11", "sigma_pred_ML_for_pf_zt_S_per_m_step21",
        "sigma_pred_S_per_m_step12", "power_factor_obs_W_per_mK2_step21", "power_factor_pred_ML_W_per_mK2_step21",
        "power_factor_pred_fitting_W_per_mK2_step21", "zt_pred_ML_step21", "zt_pred_fitting_step21",
    ]
    require_columns(primary, required, "thermoelectric_ml_primary_test_predictions_step21.csv")
    primary = primary[primary.get("evaluation_scope_step21", args.primary_scope).eq(args.primary_scope)].copy()

    row_level = build_row_level(primary)
    metrics = metric_comparison(row_level)
    sample_df = sample_level(row_level)
    high_cls = high_zt(sample_df, [0.5, 1.0, 1.5])
    overall = overall_comparison(metrics, high_cls)
    ranking_df = ranking(sample_df)
    material_summary, np_summary = group_summaries(sample_df)
    problems = problem_samples(sample_df, args.zt_threshold)
    degradation_df = degradation(overall, sample_df, material_summary, np_summary)
    interp = recommended_interpretation(overall)
    np_changed = 0
    sintering_changed = 0
    report = make_report(row_level, sample_df, overall, high_cls, ranking_df, problems, np_changed, sintering_changed)

    overall.to_csv(outdir / "step22_overall_comparison.csv", index=False)
    metrics.to_csv(outdir / "step22_metric_comparison.csv", index=False)
    row_level.to_csv(outdir / "step22_row_level_comparison.csv", index=False)
    sample_df.to_csv(outdir / "step22_sample_level_comparison.csv", index=False)
    high_cls.to_csv(outdir / "step22_high_zt_classification_comparison.csv", index=False)
    ranking_df.to_csv(outdir / "step22_ranking_correlation_comparison.csv", index=False)
    material_summary.to_csv(outdir / "step22_material_summary_comparison.csv", index=False)
    np_summary.to_csv(outdir / "step22_np_summary_comparison.csv", index=False)
    degradation_df.to_csv(outdir / "step22_ml_degradation_analysis.csv", index=False)
    problems.to_csv(outdir / "step22_problem_samples.csv", index=False)
    interp.to_csv(outdir / "step22_recommended_interpretation.csv", index=False)
    (outdir / "step22_comparison_report.txt").write_text(report, encoding="utf-8")
    (outdir / "step22_comparison_notes.md").write_text(notes(), encoding="utf-8")

    write_excel(outdir / "starrydata2_step22_fitting_vs_ml_comparison.xlsx", {
        "overall_comparison": overall,
        "metric_comparison": metrics,
        "sample_level_comparison": sample_df,
        "high_zt_classification": high_cls,
        "ranking_correlation": ranking_df,
        "material_summary": material_summary,
        "np_summary": np_summary,
        "ml_degradation": degradation_df,
        "problem_samples": problems,
        "recommended_interpretation": interp,
        "comparison_report": report,
    })

    lookup = overall.set_index("metric_name")
    cls1 = high_cls[high_cls["threshold"].eq(1.0)]
    fit_f1 = cls1[cls1["version"].eq("direct_fitting")]["f1"].iloc[0]
    ml_f1 = cls1[cls1["version"].eq("ml_tau_prediction")]["f1"].iloc[0]
    rank100_fit = ranking_df[(ranking_df["comparison_name"].eq("fitting ZT max vs observed ZT max")) & (ranking_df["top_k"].eq(100))].iloc[0]
    rank100_ml = ranking_df[(ranking_df["comparison_name"].eq("ML ZT max vs observed ZT max")) & (ranking_df["top_k"].eq(100))].iloc[0]
    print("Done.")
    print("Created:")
    for name in [
        "step22_overall_comparison.csv",
        "step22_metric_comparison.csv",
        "step22_row_level_comparison.csv",
        "step22_sample_level_comparison.csv",
        "step22_high_zt_classification_comparison.csv",
        "step22_ranking_correlation_comparison.csv",
        "step22_material_summary_comparison.csv",
        "step22_np_summary_comparison.csv",
        "step22_ml_degradation_analysis.csv",
        "step22_problem_samples.csv",
        "step22_recommended_interpretation.csv",
        "step22_comparison_report.txt",
        "step22_comparison_notes.md",
        "starrydata2_step22_fitting_vs_ml_comparison.xlsx",
    ]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"comparison samples: {sample_df['sample_key'].nunique()}")
    print(f"comparison rows: {len(row_level)}")
    print(f"fitting sigma log RMSE: {lookup.loc['sigma_log_rmse','fitting_value']}")
    print(f"ML sigma log RMSE: {lookup.loc['sigma_log_rmse','ml_value']}")
    print(f"fitting ZT vs obs MAPE: {lookup.loc['zt_vs_obs_mape','fitting_value']}")
    print(f"ML ZT vs obs MAPE: {lookup.loc['zt_vs_obs_mape','ml_value']}")
    print(f"fitting ZT>=1 F1: {fit_f1}")
    print(f"ML ZT>=1 F1: {ml_f1}")
    print(f"fitting ZT Spearman: {rank100_fit['spearman_corr']}")
    print(f"ML ZT Spearman: {rank100_ml['spearman_corr']}")
    print(f"ML much worse samples: {int((sample_df['comparison_category_step22']=='ML_much_worse').sum())}")
    print(f"ML better than fitting samples: {int((sample_df['comparison_category_step22']=='ML_better_than_fitting').sum())}")
    print(f"problem samples: {len(problems)}")
    print(f"n/p changed rows: {np_changed}")
    print(f"sintering changed rows: {sintering_changed}")


if __name__ == "__main__":
    main()
