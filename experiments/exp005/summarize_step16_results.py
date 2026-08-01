import argparse
import math
import os
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STEP15_DIR = "data/output/starrydata2_step15_pf_zt_error_analysis"
DEFAULT_STEP14_DIR = "data/output/starrydata2_step14_pf_zt_prediction"
DEFAULT_STEP13_DIR = "data/output/starrydata2_step13_sigma_validation"
DEFAULT_STEP12_DIR = "data/output/starrydata2_step12_tau_fit"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step16_result_summary"

STRING_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "composition",
    "material_system",
]

REQUIRED_SAMPLE_COLUMNS = [
    "sample_key",
    "zt_obs_max_step14",
    "zt_pred_max_step14",
    "zt_pred_vs_obs_mape_step14",
    "zt_pred_vs_calc_mape_step14",
]

OPTIONAL_SAMPLE_COLUMNS = [
    "pf_mape_step14",
    "pf_log_rmse_step14",
    "zt_pred_vs_obs_log_rmse_step14",
    "zt_pred_vs_calc_log_rmse_step14",
    "zt_calc_from_obs_max_step14",
    "zt_pred_vs_obs_quality_step14",
    "zt_pred_vs_calc_quality_step14",
    "zt_error_analysis_category_step15",
    "manual_review_priority_score_step15",
    "manual_review_priority_tier_step15",
    "needs_manual_review_step15",
    "needs_sintering_check_later_step15",
    "sintering_check_reason_step15",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize Step15 PF/ZT error analysis results for Step16."
    )
    parser.add_argument("--step15_dir", default=DEFAULT_STEP15_DIR)
    parser.add_argument("--step14_dir", default=DEFAULT_STEP14_DIR)
    parser.add_argument("--step13_dir", default=DEFAULT_STEP13_DIR)
    parser.add_argument("--step12_dir", default=DEFAULT_STEP12_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_step17_targets", type=int, default=300)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def read_csv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file is missing: {path}")
        return None
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def ensure_columns(df, columns, source_name):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def has_col(df, col):
    return col in df.columns


def safe_median(df, col):
    if df is None or col not in df.columns:
        return np.nan
    values = numeric(df[col]).dropna()
    if values.empty:
        return np.nan
    return values.median()


def safe_count_nonnull(df, col):
    if df is None or col not in df.columns:
        return 0
    return int(numeric(df[col]).notna().sum())


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_yes_value(value):
    text = normalize_text(value).lower()
    return text in {"yes", "true", "1", "y", "t"}


def truthy_mask(df, col):
    if df is None or col not in df.columns:
        return pd.Series(False, index=df.index if df is not None else [])
    return df[col].map(is_yes_value)


def nonempty_mask(df, col):
    if df is None or col not in df.columns:
        return pd.Series(False, index=df.index if df is not None else [])
    return df[col].fillna("").astype(str).str.strip().ne("")


def metric_row(name, value, note=""):
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        value = ""
    return {"metric_name": name, "metric_value": value, "metric_note": note}


def format_number(value):
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def compute_overall_metrics(samples, manual, sintering, best, zt_threshold):
    pf_eval = safe_count_nonnull(samples, "pf_mape_step14")
    zt_eval = safe_count_nonnull(samples, "zt_pred_vs_obs_mape_step14")
    problem_count = 0
    if "pf_zt_problem_reason_step15" in samples.columns:
        problem_count = int(nonempty_mask(samples, "pf_zt_problem_reason_step15").sum())
    elif "zt_error_analysis_category_step15" in samples.columns:
        categories = samples["zt_error_analysis_category_step15"].fillna("").astype(str)
        problem_count = int(~categories.isin(["", "good_prediction", "missing_eval_data", "not_evaluated"]).sum())

    obs_ge = numeric(samples["zt_obs_max_step14"]) >= zt_threshold
    pred_ge = numeric(samples["zt_pred_max_step14"]) >= zt_threshold
    n_or_p = samples["n_or_p"].fillna("unknown").astype(str).str.lower() if "n_or_p" in samples.columns else pd.Series("unknown", index=samples.index)

    rows = [
        metric_row("total_sample_results", len(samples), "Rows in pf_zt_error_samples_step15.csv"),
        metric_row("zt_eval_sample_count", zt_eval, "Samples with ZT pred-vs-observed MAPE"),
        metric_row("pf_eval_sample_count", pf_eval, "Samples with PF MAPE"),
        metric_row("problem_sample_count", problem_count, "Samples with Step15 problem reason/category"),
        metric_row("manual_review_candidate_count", 0 if manual is None else len(manual), "Rows in manual_review_candidates_step15.csv"),
        metric_row("sintering_check_candidate_count", 0 if sintering is None else len(sintering), "Rows in sintering_check_candidates_step15.csv"),
        metric_row("best_candidate_count", 0 if best is None else len(best), "Rows in best_candidate_samples_step15.csv"),
        metric_row("median_pf_mape", safe_median(samples, "pf_mape_step14")),
        metric_row("median_pf_log_rmse", safe_median(samples, "pf_log_rmse_step14")),
        metric_row("median_zt_pred_vs_obs_mape", safe_median(samples, "zt_pred_vs_obs_mape_step14")),
        metric_row("median_zt_pred_vs_obs_log_rmse", safe_median(samples, "zt_pred_vs_obs_log_rmse_step14")),
        metric_row("median_zt_pred_vs_calc_mape", safe_median(samples, "zt_pred_vs_calc_mape_step14")),
        metric_row("median_zt_pred_vs_calc_log_rmse", safe_median(samples, "zt_pred_vs_calc_log_rmse_step14")),
        metric_row("zt_obs_ge_1_sample_count", int(obs_ge.sum()), f"Observed ZT >= {zt_threshold:g}"),
        metric_row("zt_pred_ge_1_sample_count", int(pred_ge.sum()), f"Predicted ZT >= {zt_threshold:g}"),
        metric_row("zt_true_positive_count", int((obs_ge & pred_ge).sum()), f"Observed and predicted ZT >= {zt_threshold:g}"),
        metric_row("zt_false_positive_count", int((~obs_ge & pred_ge).sum()), f"Predicted ZT >= {zt_threshold:g}, observed below threshold"),
        metric_row("zt_false_negative_count", int((obs_ge & ~pred_ge).sum()), f"Observed ZT >= {zt_threshold:g}, predicted below threshold"),
        metric_row("nanocarbon_sample_count", int(truthy_mask(samples, "nanocarbon_keyword_detected_step9").sum())),
        metric_row("rare_metal_flag_sample_count", int(truthy_mask(samples, "rare_metal_flag_auto_step9").sum())),
        metric_row("toxicity_flag_sample_count", int(truthy_mask(samples, "toxicity_flag_auto_step9").sum())),
        metric_row("n_type_sample_count", int(n_or_p.eq("n").sum())),
        metric_row("p_type_sample_count", int(n_or_p.eq("p").sum())),
        metric_row("mixed_sample_count", int(n_or_p.eq("mixed").sum())),
        metric_row("unknown_np_sample_count", int(n_or_p.isin(["unknown", "", "nan"]).sum())),
    ]
    return pd.DataFrame(rows)


def interpret_classification(row):
    precision = row.get("precision", np.nan)
    recall = row.get("recall", np.nan)
    if pd.isna(precision) or pd.isna(recall):
        return "classification metrics unavailable"
    if recall >= 0.75 and precision < 0.75:
        return "high recall; useful for screening but false positives remain"
    if precision >= 0.75 and recall < 0.75:
        return "high precision; conservative high-ZT prediction"
    if recall < 0.5:
        return "low recall; high-ZT samples are often missed"
    if precision >= 0.75 and recall >= 0.75:
        return "high precision and recall for this threshold"
    return "moderate screening performance; inspect false positives and false negatives"


def make_threshold_summary(classification):
    thresholds = [0.5, 1.0, 1.5]
    out = classification[classification["threshold"].astype(float).isin(thresholds)].copy()
    if "false_negative_rate" not in out.columns:
        out["false_negative_rate"] = out["false_negative"] / out["n_observed_positive"].replace(0, np.nan)
    if "false_positive_rate" not in out.columns:
        denom = out["false_positive"] + out["true_negative"]
        out["false_positive_rate"] = out["false_positive"] / denom.replace(0, np.nan)
    if "balanced_accuracy" not in out.columns:
        out["balanced_accuracy"] = (out["recall"] + out["specificity"]) / 2
    out["interpretation_step16"] = out.apply(interpret_classification, axis=1)
    columns = [
        "evaluation_source_step14",
        "threshold",
        "n_samples",
        "n_observed_positive",
        "n_predicted_positive",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
        "balanced_accuracy",
        "false_negative_rate",
        "false_positive_rate",
        "interpretation_step16",
    ]
    return out[[col for col in columns if col in out.columns]].sort_values(["evaluation_source_step14", "threshold"])


def corr_pair(df, x_col, y_col, comparison_name):
    rows = []
    valid = df[[x_col, y_col, "sample_key"]].copy()
    valid[x_col] = numeric(valid[x_col])
    valid[y_col] = numeric(valid[y_col])
    valid = valid.dropna(subset=[x_col, y_col])
    if len(valid) < 2:
        rows.append(
            {
                "comparison_name": comparison_name,
                "n_samples": len(valid),
                "pearson_corr": np.nan,
                "spearman_corr": np.nan,
                "top_k": "",
                "top_k_overlap_count": "",
                "top_k_overlap_rate": "",
                "note": "not enough paired samples",
            }
        )
        return rows

    pearson = valid[x_col].corr(valid[y_col])
    spearman = valid[x_col].rank(method="average").corr(valid[y_col].rank(method="average"))
    rows.append(
        {
            "comparison_name": comparison_name,
            "n_samples": len(valid),
            "pearson_corr": pearson,
            "spearman_corr": spearman,
            "top_k": "",
            "top_k_overlap_count": "",
            "top_k_overlap_rate": "",
            "note": "correlation over paired non-null samples",
        }
    )
    for top_k in [50, 100, 300]:
        k = min(top_k, len(valid))
        pred_top = set(valid.sort_values(x_col, ascending=False).head(k)["sample_key"])
        obs_top = set(valid.sort_values(y_col, ascending=False).head(k)["sample_key"])
        overlap = len(pred_top & obs_top)
        rows.append(
            {
                "comparison_name": comparison_name,
                "n_samples": len(valid),
                "pearson_corr": pearson,
                "spearman_corr": spearman,
                "top_k": top_k,
                "top_k_overlap_count": overlap,
                "top_k_overlap_rate": overlap / k if k else np.nan,
                "note": f"top-{top_k} overlap; denominator uses min(top_k, n_samples)",
            }
        )
    return rows


def make_ranking_correlation(samples):
    rows = []
    rows.extend(corr_pair(samples, "zt_pred_max_step14", "zt_obs_max_step14", "zt_pred_max_step14 vs zt_obs_max_step14"))
    if "zt_calc_from_obs_max_step14" in samples.columns:
        rows.extend(
            corr_pair(
                samples,
                "zt_pred_max_step14",
                "zt_calc_from_obs_max_step14",
                "zt_pred_max_step14 vs zt_calc_from_obs_max_step14",
            )
        )
    return pd.DataFrame(rows)


def summarize_by_category(samples, summary_type, col):
    if col not in samples.columns:
        return pd.DataFrame()
    tmp = samples.copy()
    tmp[col] = tmp[col].fillna("not_available").astype(str).replace({"": "not_available"})
    manual_mask = truthy_mask(tmp, "needs_manual_review_step15")
    sintering_mask = truthy_mask(tmp, "needs_sintering_check_later_step15")
    rows = []
    for category, group in tmp.groupby(col, dropna=False):
        idx = group.index
        rows.append(
            {
                "summary_type": summary_type,
                "category": category,
                "sample_count": len(group),
                "sample_fraction": len(group) / len(samples) if len(samples) else np.nan,
                "median_pf_mape": safe_median(group, "pf_mape_step14"),
                "median_zt_pred_vs_obs_mape": safe_median(group, "zt_pred_vs_obs_mape_step14"),
                "median_zt_pred_vs_calc_mape": safe_median(group, "zt_pred_vs_calc_mape_step14"),
                "median_zt_obs_max": safe_median(group, "zt_obs_max_step14"),
                "median_zt_pred_max": safe_median(group, "zt_pred_max_step14"),
                "manual_review_count": int(manual_mask.loc[idx].sum()),
                "sintering_check_count": int(sintering_mask.loc[idx].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_error_level_summary(samples):
    frames = []
    for col in [
        "pf_error_level_sample_step15",
        "zt_obs_error_level_sample_step15",
        "zt_calc_error_level_sample_step15",
        "zt_error_analysis_category_step15",
        "manual_review_priority_tier_step15",
    ]:
        frame = summarize_by_category(samples, col, col)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["summary_type", "sample_count"], ascending=[True, False])


def interpret_material(row):
    if row.get("sample_count", 0) < 20:
        return "limited samples; interpretation uncertain"
    high_zt = row.get("high_zt_observed_sample_count", 0)
    mape = row.get("median_zt_pred_vs_obs_mape_step15", np.nan)
    large = row.get("large_error_sample_count", 0)
    if high_zt > 0 and ((not pd.isna(mape) and mape >= 0.5) or large > 0):
        return "many high-ZT or large-error samples; prioritize manual review"
    if not pd.isna(mape) and mape <= 0.25 and high_zt > 0:
        return "low error and high candidate count; reliable group"
    return "useful group-level context for Step17 prioritization"


def make_material_np_summary(samples, zt_threshold):
    df = samples.copy()
    if "material_system" not in df.columns:
        df["material_system"] = "unknown"
    if "n_or_p" not in df.columns:
        df["n_or_p"] = "unknown"
    obs_ge = numeric(df["zt_obs_max_step14"]) >= zt_threshold
    pred_ge = numeric(df["zt_pred_max_step14"]) >= zt_threshold
    manual = truthy_mask(df, "needs_manual_review_step15")
    sintering = truthy_mask(df, "needs_sintering_check_later_step15")
    nano = truthy_mask(df, "nanocarbon_keyword_detected_step9")
    rare = truthy_mask(df, "rare_metal_flag_auto_step9")
    toxic = truthy_mask(df, "toxicity_flag_auto_step9")

    rows = []
    group_cols = ["material_system", "n_or_p"]
    for keys, group in df.groupby(group_cols, dropna=False):
        idx = group.index
        row = {
            "material_system": keys[0] if normalize_text(keys[0]) else "unknown",
            "n_or_p": keys[1] if normalize_text(keys[1]) else "unknown",
            "sample_count": len(group),
            "zt_eval_sample_count": safe_count_nonnull(group, "zt_pred_vs_obs_mape_step14"),
            "median_pf_mape_step15": safe_median(group, "pf_mape_step14"),
            "median_zt_pred_vs_obs_mape_step15": safe_median(group, "zt_pred_vs_obs_mape_step14"),
            "median_zt_pred_vs_calc_mape_step15": safe_median(group, "zt_pred_vs_calc_mape_step14"),
            "high_zt_observed_sample_count": int(obs_ge.loc[idx].sum()),
            "high_zt_predicted_sample_count": int(pred_ge.loc[idx].sum()),
            "high_zt_true_positive_count": int((obs_ge.loc[idx] & pred_ge.loc[idx]).sum()),
            "high_zt_false_negative_count": int((obs_ge.loc[idx] & ~pred_ge.loc[idx]).sum()),
            "high_zt_false_positive_count": int((~obs_ge.loc[idx] & pred_ge.loc[idx]).sum()),
            "manual_review_sample_count": int(manual.loc[idx].sum()),
            "sintering_check_sample_count": int(sintering.loc[idx].sum()),
            "nanocarbon_sample_count": int(nano.loc[idx].sum()),
            "rare_metal_flag_sample_count": int(rare.loc[idx].sum()),
            "toxicity_flag_sample_count": int(toxic.loc[idx].sum()),
        }
        row["large_error_sample_count"] = int(
            group.get("zt_error_analysis_category_step15", pd.Series("", index=group.index))
            .fillna("")
            .astype(str)
            .str.contains("large|false_negative|false_positive|sigma", case=False, regex=True)
            .sum()
        )
        row["interpretation_step16"] = interpret_material(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("sample_count", ascending=False)


def make_feature_flag_summary(samples, existing_feature_summary):
    if existing_feature_summary is not None:
        out = existing_feature_summary.copy()
    else:
        out = pd.DataFrame()

    target_features = [
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "n_or_p",
        "fitting_source_actual_step10",
        "tau_eff_mode_step12",
    ]
    existing_keys = set()
    if not out.empty and {"feature_name_step15", "feature_value_step15"}.issubset(out.columns):
        existing_keys = set(zip(out["feature_name_step15"].astype(str), out["feature_value_step15"].astype(str)))

    rows = []
    obs_ge = numeric(samples["zt_obs_max_step14"]) >= 1.0
    pred_ge = numeric(samples["zt_pred_max_step14"]) >= 1.0
    large_error = samples.get("zt_error_analysis_category_step15", pd.Series("", index=samples.index)).fillna("").astype(str).str.contains(
        "large|false_negative|false_positive|sigma", case=False, regex=True
    )
    manual = truthy_mask(samples, "needs_manual_review_step15")
    for feature in target_features:
        if feature not in samples.columns:
            continue
        values = samples[feature].fillna("not_available").astype(str).replace({"": "not_available"})
        for value, group in samples.groupby(values, dropna=False):
            key = (feature, str(value))
            if key in existing_keys:
                continue
            idx = group.index
            rows.append(
                {
                    "feature_name_step15": feature,
                    "feature_value_step15": value,
                    "sample_count": len(group),
                    "zt_eval_sample_count": safe_count_nonnull(group, "zt_pred_vs_obs_mape_step14"),
                    "median_zt_pred_vs_obs_mape_step15": safe_median(group, "zt_pred_vs_obs_mape_step14"),
                    "median_zt_pred_vs_calc_mape_step15": safe_median(group, "zt_pred_vs_calc_mape_step14"),
                    "median_pf_mape_step15": safe_median(group, "pf_mape_step14"),
                    "high_zt_observed_sample_count": int(obs_ge.loc[idx].sum()),
                    "high_zt_predicted_sample_count": int(pred_ge.loc[idx].sum()),
                    "large_error_sample_count": int(large_error.loc[idx].sum()),
                    "manual_review_sample_count": int(manual.loc[idx].sum()),
                }
            )
    if rows:
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True) if not out.empty else pd.DataFrame(rows)
    if out.empty:
        return out
    out["interpretation_step16"] = out.apply(interpret_feature, axis=1)
    return out.sort_values(["feature_name_step15", "sample_count"], ascending=[True, False])


def interpret_feature(row):
    count = row.get("sample_count", 0)
    if count < 10:
        return "limited samples; interpretation uncertain"
    large = row.get("large_error_sample_count", 0)
    manual = row.get("manual_review_sample_count", 0)
    if large > 0 or manual > 0:
        return "contains large-error or manual-review candidates"
    return "no strong Step16 warning from this grouped summary"


def top_examples(df, col="sample_key", n=5):
    if df is None or df.empty or col not in df.columns:
        return ""
    return "; ".join(df[col].dropna().astype(str).head(n).tolist())


def make_review_summary(df, label):
    columns = [
        "manual_review_priority_tier_step15",
        "zt_error_analysis_category_step15",
        "needs_sintering_check_later_step15",
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "n_or_p",
        "material_system",
    ]
    if label == "sintering":
        columns = ["sintering_check_reason_step15", "material_system", "n_or_p", "zt_error_analysis_category_step15"]
    rows = []
    if df is None:
        return pd.DataFrame(columns=["summary_type", "category", "sample_count", "top_example_sample_keys", "top_example_compositions", "note"])
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].fillna("not_available").astype(str).replace({"": "not_available"})
        for value, group in df.groupby(values, dropna=False):
            rows.append(
                {
                    "summary_type": col,
                    "category": value,
                    "sample_count": len(group),
                    "top_example_sample_keys": top_examples(group, "sample_key"),
                    "top_example_compositions": top_examples(group, "composition"),
                    "note": "Step16 summary only; original records should be checked in Step17",
                }
            )
    return pd.DataFrame(rows).sort_values(["summary_type", "sample_count"], ascending=[True, False])


def make_key_findings(overall, threshold_summary):
    metric = dict(zip(overall["metric_name"], overall["metric_value"]))
    threshold_1 = threshold_summary[np.isclose(pd.to_numeric(threshold_summary["threshold"]), 1.0)]
    if not threshold_1.empty:
        row1 = threshold_1.iloc[0]
        perf = f"P={format_number(row1.get('precision'))}, R={format_number(row1.get('recall'))}, F1={format_number(row1.get('f1'))}"
    else:
        perf = "NA"
    rows = [
        ("KF01", "PF/ZT prediction status", "tau_eff fitting after prior steps has already been used for PF/ZT estimates.", "step_status", "completed before Step16", "step15_error_analysis_report.txt", "Step16 does not create new predictions."),
        ("KF02", "ZT>=1 classification", "ZT>=1 precision/recall/F1 summarize high-ZT screening performance.", "precision_recall_f1", perf, "step16_zt_threshold_summary.csv", "False positives and false negatives remain."),
        ("KF03", "ZT error", "Median ZT prediction MAPE against observed ZT.", "median_zt_pred_vs_obs_mape", metric.get("median_zt_pred_vs_obs_mape", ""), "step16_overall_metrics.csv", "Uses Step15 sample-level metrics."),
        ("KF04", "PF error", "Median PF prediction MAPE.", "median_pf_mape", metric.get("median_pf_mape", ""), "step16_overall_metrics.csv", "Uses Step15 sample-level metrics."),
        ("KF05", "High-ZT errors", "High-ZT false positives and false negatives exist and need targeted inspection.", "fp_fn_counts", f"FP={metric.get('zt_false_positive_count', '')}, FN={metric.get('zt_false_negative_count', '')}", "step16_overall_metrics.csv", "Counts use the configured ZT threshold."),
        ("KF06", "Manual review", "Manual review candidates are available for Step17.", "manual_review_candidate_count", metric.get("manual_review_candidate_count", ""), "step16_manual_review_summary.csv", "Step16 does not inspect original papers."),
        ("KF07", "Sintering check", "Sintering check candidates are available for prioritized Step17 checks.", "sintering_check_candidate_count", metric.get("sintering_check_candidate_count", ""), "step16_sintering_check_summary.csv", "Sintering methods remain unknown in Step16."),
        ("KF08", "tau_eff caveat", "tau_eff is a relative scale and not a physical relaxation time in seconds.", "tau_eff_unit", "relative scalar", "step16_summary_notes.md", "Do not interpret tau_eff as seconds."),
        ("KF09", "S/kappa caveat", "S and kappa were not predicted; observed values were used.", "prediction_scope", "sigma_pred with observed S/kappa", "step16_summary_notes.md", "PF/ZT errors partly reflect the sigma prediction path."),
        ("KF10", "Step17 scope", "Step17 should inspect only high-ZT, large-error, and paper-candidate samples.", "review_target_file", "step16_next_step17_review_targets.csv", "step16_next_step17_review_targets.csv", "Targets are prioritized and capped by the script argument."),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "finding_id",
            "finding_topic",
            "finding_summary",
            "supporting_metric",
            "supporting_value",
            "related_file",
            "caution",
        ],
    )


def merge_candidate_sources(samples, manual, sintering, best, highzt, zt_threshold, top_n):
    base = samples.set_index("sample_key", drop=False)
    source_frames = [
        ("manual_review", manual),
        ("sintering_check", sintering),
        ("best_candidate", best),
        ("high_zt_error", highzt),
    ]
    keys = []
    source_map = {}
    for source, df in source_frames:
        if df is None or "sample_key" not in df.columns:
            continue
        for key in df["sample_key"].dropna().astype(str):
            keys.append(key)
            source_map.setdefault(key, set()).add(source)
    keys = list(dict.fromkeys(keys))
    rows = []
    manual_keys = set(manual["sample_key"].dropna().astype(str)) if manual is not None and "sample_key" in manual.columns else set()
    sintering_keys = set(sintering["sample_key"].dropna().astype(str)) if sintering is not None and "sample_key" in sintering.columns else set()
    best_keys = set(best["sample_key"].dropna().astype(str)) if best is not None and "sample_key" in best.columns else set()
    highzt_idx = highzt.set_index("sample_key", drop=False) if highzt is not None and "sample_key" in highzt.columns else pd.DataFrame()

    for key in keys:
        if key in base.index:
            rec = base.loc[key]
            if isinstance(rec, pd.DataFrame):
                rec = rec.iloc[0]
            row = rec.to_dict()
        else:
            row = {"sample_key": key}
        row["source_files_step16"] = ";".join(sorted(source_map.get(key, [])))

        score = 0
        reasons = []
        obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
        pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
        mape = pd.to_numeric(pd.Series([row.get("zt_pred_vs_obs_mape_step14")]), errors="coerce").iloc[0]
        category = normalize_text(row.get("zt_error_analysis_category_step15"))
        high_case = ""
        high_thresholds = []
        if not highzt_idx.empty and key in highzt_idx.index:
            high_rec = highzt_idx.loc[key]
            if isinstance(high_rec, pd.DataFrame):
                high_case = ";".join(high_rec.get("classification_case_step15", pd.Series(dtype=str)).dropna().astype(str).unique())
                high_thresholds = pd.to_numeric(high_rec.get("threshold_step15", pd.Series(dtype=float)), errors="coerce").dropna().tolist()
            else:
                high_case = normalize_text(high_rec.get("classification_case_step15"))
                th = pd.to_numeric(pd.Series([high_rec.get("threshold_step15")]), errors="coerce").iloc[0]
                if not pd.isna(th):
                    high_thresholds = [th]
        row["classification_case_step15"] = high_case or row.get("classification_case_step15", "")
        row["thresholds_in_high_zt_error_file_step15"] = ";".join(format_number(v) for v in sorted(set(high_thresholds)))

        if "false_negative" in high_case and any(np.isclose(high_thresholds, zt_threshold)):
            score += 1000
            reasons.append(f"ZT>={zt_threshold:g} false negative")
        if "false_positive" in high_case and any(np.isclose(high_thresholds, zt_threshold)):
            score += 900
            reasons.append(f"ZT>={zt_threshold:g} false positive")
        if not pd.isna(obs) and obs >= zt_threshold:
            score += 800
            reasons.append(f"observed ZT>={zt_threshold:g}")
        if (not pd.isna(mape) and mape >= 1.0) or any(token in category for token in ["large", "sigma", "false_negative", "false_positive"]):
            score += 500
            reasons.append("large ZT prediction error")
        if key in best_keys:
            score += 350
            reasons.append("best candidate sample")
        if is_yes_value(row.get("nanocarbon_keyword_detected_step9")):
            score += 250
            reasons.append("nanocarbon candidate")
        if key in manual_keys:
            score += 200
            reasons.append("manual review candidate")
        if key in sintering_keys:
            score += 150
            reasons.append("sintering check candidate")
        if not pd.isna(pred) and pred >= zt_threshold:
            score += 100
            reasons.append(f"predicted ZT>={zt_threshold:g}")

        row["step17_review_priority_score"] = score
        if score >= 1400:
            tier = "A"
        elif score >= 900:
            tier = "B"
        else:
            tier = "C"
        row["step17_review_priority_tier"] = tier
        row["step17_review_reason"] = "; ".join(dict.fromkeys(reasons))
        row["step17_check_additive"] = "yes" if key in manual_keys else "no"
        row["step17_check_structure"] = "yes" if key in manual_keys else "no"
        np_value = normalize_text(row.get("n_or_p")).lower()
        check_np = np_value in {"mixed", "unknown", ""} or "large ZT prediction error" in reasons or key in best_keys
        row["step17_check_np_type"] = "yes" if check_np else "no"
        row["step17_check_sintering"] = "yes" if key in sintering_keys else "no"
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    preferred = [
        "sample_key",
        "step17_review_priority_score",
        "step17_review_priority_tier",
        "step17_review_reason",
        "step17_check_additive",
        "step17_check_structure",
        "step17_check_np_type",
        "step17_check_sintering",
        "source_files_step16",
        "classification_case_step15",
        "thresholds_in_high_zt_error_file_step15",
        "material_system",
        "n_or_p",
        "n_or_p_basis",
        "n_or_p_step6",
        "n_or_p_basis_step6",
        "n_or_p_confidence_step6",
        "composition",
        "DOI",
        "doi_url",
        "paper_title",
        "zt_obs_max_step14",
        "zt_pred_max_step14",
        "zt_calc_from_obs_max_step14",
        "zt_pred_vs_obs_mape_step14",
        "zt_pred_vs_calc_mape_step14",
        "manual_review_priority_score_step15",
        "manual_review_priority_tier_step15",
        "zt_error_analysis_category_step15",
        "nanocarbon_keyword_detected_step9",
        "nanocarbon_type_auto_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "additive_auto_step9",
        "additive_manual_step9",
        "structure_auto_step9",
        "structure_manual_step9",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "sintering_check_reason_step15",
    ]
    extra = [col for col in out.columns if col not in preferred]
    out = out[[col for col in preferred + extra if col in out.columns]]
    return out.sort_values(["step17_review_priority_score", "zt_obs_max_step14"], ascending=[False, False]).head(top_n)


def make_notes(overall, threshold_summary, ranking, targets_count):
    metric = dict(zip(overall["metric_name"], overall["metric_value"]))
    zt1 = threshold_summary[np.isclose(pd.to_numeric(threshold_summary["threshold"]), 1.0)]
    zt1_text = "not available"
    if not zt1.empty:
        r = zt1.iloc[0]
        zt1_text = f"precision {format_number(r.get('precision'))}, recall {format_number(r.get('recall'))}, F1 {format_number(r.get('f1'))}"
    rank_row = ranking[ranking["comparison_name"].eq("zt_pred_max_step14 vs zt_obs_max_step14") & ranking["top_k"].astype(str).eq("")]
    rank_text = "not available"
    if not rank_row.empty:
        r = rank_row.iloc[0]
        rank_text = f"Pearson {format_number(r.get('pearson_corr'))}, Spearman {format_number(r.get('spearman_corr'))}"
    return f"""# Step16 Summary

## Purpose
Step16 summarizes PF/ZT error analysis and ZT>=1 classification performance from Step15 for reporting and Step17 review prioritization.

## Data Used
The main input is `pf_zt_error_samples_step15.csv` with {format_number(metric.get("total_sample_results"))} sample-level rows. Manual review, sintering-check, best-candidate, high-ZT classification, material, n/p, and feature-flag summaries from Step15 were also used when available.

## Main PF/ZT Error Results
- Median PF MAPE: {format_number(metric.get("median_pf_mape"))}
- Median ZT pred-vs-observed MAPE: {format_number(metric.get("median_zt_pred_vs_obs_mape"))}
- Median ZT pred-vs-calc-from-observed MAPE: {format_number(metric.get("median_zt_pred_vs_calc_mape"))}

## ZT>=1 Classification Performance
For the ZT>=1 threshold, classification performance is {zt1_text}.

## Ranking Correlation
ZT predicted-vs-observed ranking correlation is {rank_text}. Top-k overlap values are recorded in `step16_ranking_correlation.csv`.

## Important Caveats
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.

Seebeck coefficient and thermal conductivity were not predicted in Steps 14-16.

PF_pred and ZT_pred were calculated using sigma_pred and observed S/kappa.

Sintering methods are still unknown and will be checked only for prioritized samples in Step17.

## Manual Review Targets for Step17
`step16_next_step17_review_targets.csv` contains {targets_count} prioritized samples. The target list combines manual-review candidates, sintering-check candidates, best-candidate samples, and high-ZT false positive/false negative samples, deduplicated by `sample_key`.

## Sintering Check Policy
Step16 does not investigate sintering methods. `sintering_method=unknown`, `sintering_checked=no`, and `record_checked=no` are preserved. Step17 should check sintering only for prioritized samples marked `step17_check_sintering=yes`.

## Next Step
Use `step16_next_step17_review_targets.csv` to inspect original papers for high-ZT samples, large-error samples, ZT>=1 false negatives/false positives, paper candidates, selected sintering-check candidates, and samples where additive/structure metadata should be improved.
"""


def value_from_metric(overall, name):
    found = overall.loc[overall["metric_name"].eq(name), "metric_value"]
    return found.iloc[0] if len(found) else np.nan


def make_report(
    inputs,
    outputs,
    overall,
    threshold_summary,
    ranking,
    material_np,
    targets,
    samples,
    n_p_changed_rows,
    sintering_changed_rows,
):
    metric = dict(zip(overall["metric_name"], overall["metric_value"]))
    lines = []
    lines.append("Step16 PF/ZT result summary")
    lines.append("")
    for key, value in inputs.items():
        lines.append(f"input {key}: {value}")
    lines.append("")
    for key, value in outputs.items():
        lines.append(f"output {key}: {value}")
    lines.append("")
    lines.append("overall:")
    for name in [
        "total_sample_results",
        "pf_eval_sample_count",
        "zt_eval_sample_count",
        "problem_sample_count",
        "manual_review_candidate_count",
        "sintering_check_candidate_count",
    ]:
        lines.append(f"- {name}: {format_number(metric.get(name))}")
    lines.append("")
    lines.append("PF/ZT error:")
    for name in [
        "median_pf_mape",
        "median_zt_pred_vs_obs_mape",
        "median_zt_pred_vs_calc_mape",
        "median_zt_pred_vs_obs_log_rmse",
        "median_zt_pred_vs_calc_log_rmse",
    ]:
        lines.append(f"- {name}: {format_number(metric.get(name))}")
    lines.append("")
    lines.append("ZT threshold classification:")
    for threshold in [0.5, 1.0, 1.5]:
        subset = threshold_summary[np.isclose(pd.to_numeric(threshold_summary["threshold"]), threshold)]
        if subset.empty:
            lines.append(f"- threshold {threshold:g}: not available")
        else:
            row = subset.iloc[0]
            lines.append(
                f"- threshold {threshold:g}: precision={format_number(row.get('precision'))}, "
                f"recall={format_number(row.get('recall'))}, F1={format_number(row.get('f1'))}"
            )
    lines.append("")
    lines.append("Ranking correlation:")
    main_rank = ranking[ranking["comparison_name"].eq("zt_pred_max_step14 vs zt_obs_max_step14")]
    corr_row = main_rank[main_rank["top_k"].astype(str).eq("")]
    if not corr_row.empty:
        row = corr_row.iloc[0]
        lines.append(f"- Spearman zt_pred_max vs zt_obs_max: {format_number(row.get('spearman_corr'))}")
        lines.append(f"- Pearson zt_pred_max vs zt_obs_max: {format_number(row.get('pearson_corr'))}")
    for top_k in [50, 100, 300]:
        top_row = main_rank[main_rank["top_k"].astype(str).eq(str(top_k))]
        if not top_row.empty:
            row = top_row.iloc[0]
            lines.append(
                f"- top {top_k} overlap: {format_number(row.get('top_k_overlap_count'))} "
                f"({format_number(row.get('top_k_overlap_rate'))})"
            )
    lines.append("")
    lines.append("Review:")
    lines.append(f"- Step17 review target count: {len(targets)}")
    for col in ["step17_check_additive", "step17_check_structure", "step17_check_np_type", "step17_check_sintering"]:
        lines.append(f"- {col} count: {int(targets[col].eq('yes').sum()) if col in targets.columns else 0}")
    lines.append("")
    lines.append("n/p:")
    if "n_or_p" in samples.columns:
        for value, count in samples["n_or_p"].fillna("unknown").astype(str).value_counts().items():
            lines.append(f"- n_or_p sample count {value}: {count}")
    if "n_or_p" in material_np.columns:
        for _, row in material_np.groupby("n_or_p", dropna=False)["median_zt_pred_vs_obs_mape_step15"].median().reset_index().iterrows():
            lines.append(f"- n_or_p median ZT vs obs MAPE {row['n_or_p']}: {format_number(row['median_zt_pred_vs_obs_mape_step15'])}")
    lines.append("")
    lines.append("material:")
    if "material_system" in samples.columns:
        for value, count in samples["material_system"].fillna("unknown").astype(str).value_counts().head(20).items():
            lines.append(f"- material_system sample count {value}: {count}")
    for _, row in material_np.sort_values("median_zt_pred_vs_obs_mape_step15", ascending=False).head(20).iterrows():
        lines.append(
            f"- material_system median ZT vs obs MAPE {row.get('material_system')} / {row.get('n_or_p')}: "
            f"{format_number(row.get('median_zt_pred_vs_obs_mape_step15'))}"
        )
    lines.append("")
    lines.append("flags:")
    for name in ["nanocarbon_sample_count", "rare_metal_flag_sample_count", "toxicity_flag_sample_count"]:
        lines.append(f"- {name}: {format_number(metric.get(name))}")
    lines.append("")
    lines.append("sintering:")
    lines.append(f"- sintering_method=unknown rows: {int(samples.get('sintering_method', pd.Series([], dtype=str)).fillna('').astype(str).str.lower().eq('unknown').sum()) if 'sintering_method' in samples.columns else 0}")
    lines.append(f"- sintering_checked=no rows: {int(samples.get('sintering_checked', pd.Series([], dtype=str)).fillna('').astype(str).str.lower().eq('no').sum()) if 'sintering_checked' in samples.columns else 0}")
    lines.append(f"- record_checked=no rows: {int(samples.get('record_checked', pd.Series([], dtype=str)).fillna('').astype(str).str.lower().eq('no').sum()) if 'record_checked' in samples.columns else 0}")
    lines.append("")
    lines.append(f"n/p changed rows: {n_p_changed_rows}")
    lines.append(f"sintering changed rows: {sintering_changed_rows}")
    lines.append("")
    lines.append("sample_key duplicate check:")
    lines.append(f"- pf_zt_error_samples_step15 duplicate sample_key rows: {int(samples['sample_key'].duplicated().sum())}")
    lines.append(f"- step16_next_step17_review_targets duplicate sample_key rows: {int(targets['sample_key'].duplicated().sum()) if 'sample_key' in targets.columns else 0}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Step16 did not create new predictions, refit tau_eff, or recalculate PF/ZT.")
    lines.append("- Step16 summarized results through Step15.")
    lines.append("- tau_eff is a relative scale and not a physical relaxation time in seconds.")
    lines.append("- PF_pred/ZT_pred were calculated from sigma_pred and observed S_obs/kappa_obs; S and kappa were not predicted.")
    return "\n".join(lines) + "\n"


def write_excel(output_path, sheets):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.book[safe_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
            for column_cells in ws.columns:
                max_len = 0
                col_letter = column_cells[0].column_letter
                for cell in column_cells:
                    text = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(text), 60))
                ws.column_dimensions[col_letter].width = max(10, min(max_len + 2, 60))


def main():
    args = parse_args()
    step15_dir = Path(args.step15_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_csv(step15_dir / "pf_zt_error_samples_step15.csv", required=True)
    classification = read_csv(step15_dir / "high_zt_classification_analysis_step15.csv", required=True)
    highzt = read_csv(step15_dir / "high_zt_missed_and_false_positive_samples_step15.csv", required=True)
    best = read_csv(step15_dir / "best_candidate_samples_step15.csv", required=True)
    manual = read_csv(step15_dir / "manual_review_candidates_step15.csv", required=True)
    sintering = read_csv(step15_dir / "sintering_check_candidates_step15.csv", required=True)
    by_material = read_csv(step15_dir / "pf_zt_error_by_material_step15.csv", required=True)
    by_np = read_csv(step15_dir / "pf_zt_error_by_np_type_step15.csv", required=True)
    by_feature = read_csv(step15_dir / "pf_zt_error_by_feature_flags_step15.csv", required=True)
    validation = read_csv(step15_dir / "pf_zt_validation_error_samples_step15.csv", required=False)

    ensure_columns(samples, REQUIRED_SAMPLE_COLUMNS, "pf_zt_error_samples_step15.csv")
    ensure_columns(samples, ["sample_key"], "pf_zt_error_samples_step15.csv")
    for frame_name, frame in [
        ("manual_review_candidates_step15.csv", manual),
        ("sintering_check_candidates_step15.csv", sintering),
        ("best_candidate_samples_step15.csv", best),
        ("high_zt_missed_and_false_positive_samples_step15.csv", highzt),
    ]:
        ensure_columns(frame, ["sample_key"], frame_name)

    overall = compute_overall_metrics(samples, manual, sintering, best, args.zt_threshold)
    threshold_summary = make_threshold_summary(classification)
    ranking = make_ranking_correlation(samples)
    error_level = make_error_level_summary(samples)
    material_np = make_material_np_summary(samples, args.zt_threshold)
    feature_summary = make_feature_flag_summary(samples, by_feature)
    manual_summary = make_review_summary(manual, "manual")
    sintering_summary = make_review_summary(sintering, "sintering")
    key_findings = make_key_findings(overall, threshold_summary)
    targets = merge_candidate_sources(samples, manual, sintering, best, highzt, args.zt_threshold, args.top_n_step17_targets)

    # Step16 does not alter n/p labels. The target file is assembled from Step15 rows by sample_key.
    n_p_cols = ["n_or_p", "n_or_p_basis", "n_or_p_step6", "n_or_p_basis_step6", "n_or_p_confidence_step6"]
    n_p_changed_rows = 0
    if not targets.empty:
        sample_lookup = samples.set_index("sample_key")
        for _, row in targets.iterrows():
            key = row.get("sample_key")
            if key in sample_lookup.index:
                src = sample_lookup.loc[key]
                if isinstance(src, pd.DataFrame):
                    src = src.iloc[0]
                for col in n_p_cols:
                    if col in targets.columns and col in samples.columns and normalize_text(row.get(col)) != normalize_text(src.get(col)):
                        n_p_changed_rows += 1
                        break
    sintering_changed_rows = 0
    if not targets.empty:
        for col, expected in [("sintering_method", "unknown"), ("sintering_checked", "no"), ("record_checked", "no")]:
            if col in targets.columns:
                sintering_changed_rows += int((~targets[col].fillna("").astype(str).str.lower().eq(expected)).sum())
    input_counts = {
        "pf_zt_error_samples_step15 sample count": len(samples),
        "high_zt_classification_analysis_step15 row count": len(classification),
        "manual_review_candidates_step15 sample count": len(manual),
        "sintering_check_candidates_step15 sample count": len(sintering),
        "best_candidate_samples_step15 sample count": len(best),
    }
    output_counts = {
        "step16_overall_metrics rows": len(overall),
        "step16_zt_threshold_summary rows": len(threshold_summary),
        "step16_ranking_correlation rows": len(ranking),
        "step16_error_level_summary rows": len(error_level),
        "step16_material_np_summary rows": len(material_np),
        "step16_next_step17_review_targets sample count": len(targets),
    }
    report_text = make_report(
        input_counts,
        output_counts,
        overall,
        threshold_summary,
        ranking,
        material_np,
        targets,
        samples,
        n_p_changed_rows,
        sintering_changed_rows,
    )
    notes_text = make_notes(overall, threshold_summary, ranking, len(targets))

    outputs = {
        "step16_overall_metrics.csv": overall,
        "step16_zt_threshold_summary.csv": threshold_summary,
        "step16_ranking_correlation.csv": ranking,
        "step16_error_level_summary.csv": error_level,
        "step16_material_np_summary.csv": material_np,
        "step16_feature_flag_summary.csv": feature_summary,
        "step16_manual_review_summary.csv": manual_summary,
        "step16_sintering_check_summary.csv": sintering_summary,
        "step16_key_findings_table.csv": key_findings,
        "step16_next_step17_review_targets.csv": targets,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    (output_dir / "step16_summary_report.txt").write_text(report_text, encoding="utf-8")
    (output_dir / "step16_summary_notes.md").write_text(notes_text, encoding="utf-8")

    report_df = pd.DataFrame({"summary_report": report_text.splitlines()})
    write_excel(
        output_dir / "starrydata2_step16_result_summary.xlsx",
        {
            "overall_metrics": overall,
            "zt_threshold_summary": threshold_summary,
            "ranking_correlation": ranking,
            "error_level_summary": error_level,
            "material_np_summary": material_np,
            "feature_flag_summary": feature_summary,
            "manual_review_summary": manual_summary,
            "sintering_check_summary": sintering_summary,
            "key_findings": key_findings,
            "step17_review_targets": targets,
            "summary_report": report_df,
        },
    )

    zt1 = threshold_summary[np.isclose(pd.to_numeric(threshold_summary["threshold"]), 1.0)]
    zt1_row = zt1.iloc[0] if not zt1.empty else {}
    rank_main = ranking[
        ranking["comparison_name"].eq("zt_pred_max_step14 vs zt_obs_max_step14")
        & ranking["top_k"].astype(str).eq("")
    ]
    spearman = rank_main.iloc[0]["spearman_corr"] if not rank_main.empty else np.nan
    top100 = ranking[
        ranking["comparison_name"].eq("zt_pred_max_step14 vs zt_obs_max_step14")
        & ranking["top_k"].astype(str).eq("100")
    ]
    top100_value = top100.iloc[0]["top_k_overlap_count"] if not top100.empty else np.nan

    print("Done.")
    print("Created:")
    for filename in [
        "step16_overall_metrics.csv",
        "step16_zt_threshold_summary.csv",
        "step16_ranking_correlation.csv",
        "step16_error_level_summary.csv",
        "step16_material_np_summary.csv",
        "step16_feature_flag_summary.csv",
        "step16_manual_review_summary.csv",
        "step16_sintering_check_summary.csv",
        "step16_key_findings_table.csv",
        "step16_next_step17_review_targets.csv",
        "step16_summary_report.txt",
        "step16_summary_notes.md",
        "starrydata2_step16_result_summary.xlsx",
    ]:
        print(f"- {filename}")
    print("")
    print("Summary:")
    print(f"total samples: {format_number(value_from_metric(overall, 'total_sample_results'))}")
    print(f"median PF MAPE: {format_number(value_from_metric(overall, 'median_pf_mape'))}")
    print(f"median ZT vs obs MAPE: {format_number(value_from_metric(overall, 'median_zt_pred_vs_obs_mape'))}")
    print(f"ZT>=1 precision: {format_number(zt1_row.get('precision', np.nan))}")
    print(f"ZT>=1 recall: {format_number(zt1_row.get('recall', np.nan))}")
    print(f"ZT>=1 F1: {format_number(zt1_row.get('f1', np.nan))}")
    print(f"ZT ranking Spearman: {format_number(spearman)}")
    print(f"top100 ZT overlap: {format_number(top100_value)}")
    print(f"manual review candidates: {len(manual)}")
    print(f"sintering check candidates: {len(sintering)}")
    print(f"step17 review targets: {len(targets)}")
    print(f"n/p changed rows: {n_p_changed_rows}")
    print(f"sintering changed rows: {sintering_changed_rows}")


if __name__ == "__main__":
    main()
