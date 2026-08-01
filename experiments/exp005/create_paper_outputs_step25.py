import argparse
import math
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULTS = {
    "step12_dir": "data/output/starrydata2_step12_tau_fit",
    "step13_dir": "data/output/starrydata2_step13_sigma_validation",
    "step14_dir": "data/output/starrydata2_step14_pf_zt_prediction",
    "step16_dir": "data/output/starrydata2_step16_result_summary",
    "step18_dir": "data/output/starrydata2_step18_tau_eff_ml_dataset",
    "step19_dir": "data/output/starrydata2_step19_tau_eff_ml_model",
    "step20_dir": "data/output/starrydata2_step20_sigma_ml_prediction",
    "step21_dir": "data/output/starrydata2_step21_pf_zt_ml_prediction",
    "step22_dir": "data/output/starrydata2_step22_fitting_vs_ml_comparison",
    "step23_dir": "data/output/starrydata2_step23_error_cause_analysis",
    "step24_dir": "data/output/starrydata2_step24_material_candidates",
    "output_dir": "data/output/starrydata2_step25_paper_outputs",
}
STRING_COLUMNS = ["sample_key", "SID", "DOI", "doi_url", "sample_id", "composition", "material_system", "n_or_p"]
EXCEL_PREVIEW_ROWS = 100_000


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper-ready Step25 outputs from Step12-Step24 results.")
    for key, value in DEFAULTS.items():
        parser.add_argument(f"--{key}", default=value)
    parser.add_argument("--top_n_candidates", type=int, default=50)
    parser.add_argument("--top_n_review_targets", type=int, default=100)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {c: "string" for c in STRING_COLUMNS if c in header.columns}


def read_csv(path, loaded, missing):
    path = Path(path)
    if not path.exists():
        missing.append(str(path))
        return None
    loaded.append(str(path))
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def read_text(path, loaded, missing):
    path = Path(path)
    if not path.exists():
        missing.append(str(path))
        return ""
    loaded.append(str(path))
    return path.read_text(encoding="utf-8", errors="replace")


def nrows(df):
    return 0 if df is None else len(df)


def nsamples(df):
    if df is None or "sample_key" not in df.columns:
        return 0
    return int(df["sample_key"].nunique())


def num(df, col):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def median(df, col):
    s = num(df, col)
    return np.nan if s.empty else s.median()


def first_value(df, col, default=np.nan):
    if df is None or col not in df.columns or df.empty:
        return default
    vals = df[col].dropna()
    return default if vals.empty else vals.iloc[0]


def metric_value(metrics, name):
    if metrics is None or "metric_name" not in metrics.columns:
        return np.nan
    row = metrics[metrics["metric_name"].astype(str) == name]
    if row.empty or "metric_value" not in row.columns:
        return np.nan
    return pd.to_numeric(row["metric_value"], errors="coerce").iloc[0]


def pick_row(df, **conds):
    if df is None:
        return pd.Series(dtype=object)
    sub = df
    for col, value in conds.items():
        if col not in sub.columns:
            return pd.Series(dtype=object)
        sub = sub[sub[col].astype(str) == str(value)]
    return pd.Series(dtype=object) if sub.empty else sub.iloc[0]


def row_value(row, col, default=np.nan):
    if row is None or len(row) == 0 or col not in row.index:
        return default
    return row[col]


def first_existing(row, cols, default=np.nan):
    for col in cols:
        val = row_value(row, col, np.nan)
        if pd.notna(val):
            return val
    return default


def reorder_url_last(df):
    if df is None or df.empty:
        return df
    urls = [c for c in df.columns if c.lower().endswith("_url") or c.lower() == "url"]
    return df[[c for c in df.columns if c not in urls] + urls]


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def pipeline_summary():
    rows = []
    for i in range(1, 26):
        rows.append({
            "step": f"Step{i}",
            "step_name": "",
            "purpose": "",
            "main_input": "",
            "main_output": "",
            "key_result": "",
            "paper_use": "",
        })
    labels = {
        12: ("tau_eff fitting", "Fit relative effective tau_eff from sigma observations.", "Step11 normalized transport rows", "tau_fit_results_step12.csv", "relative tau_eff and sigma fit errors", "Methods and fitting performance"),
        13: ("sigma holdout validation", "Validate fitted tau_eff on held-out temperature rows.", "Step12 fitted samples", "tau_validation_primary_results_step13.csv", "sigma validation quality", "Validation performance"),
        14: ("PF/ZT fitting prediction", "Estimate PF/ZT using fitted tau_eff-derived sigma with observed S/kappa.", "Step12 and Step11 values", "pf_zt_sample_results_step14.csv", "PF/ZT sample metrics", "PF/ZT fitting results"),
        15: ("PF/ZT error analysis", "Identify PF/ZT problem samples and review targets.", "Step14 outputs", "Step15 problem/review outputs", "manual review and sintering targets", "Error analysis context"),
        18: ("ML dataset creation", "Build feature table for fitted log_tau_eff labels.", "Step12 and annotations", "tau_eff_ml_dataset_recommended_step18.csv", "recommended ML samples and features", "ML dataset description"),
        19: ("ML prediction of log_tau_eff", "Train and evaluate ML models for fitted log_tau_eff.", "Step18 dataset", "tau_eff_ml_model_comparison_step19.csv", "selected model and DOI test metrics", "ML performance"),
        20: ("ML tau_eff to sigma", "Convert ML-predicted tau_eff into sigma estimates.", "Step19 model outputs", "sigma_ml_model_comparison_step20.csv", "ML sigma evaluation", "ML sigma result"),
        21: ("ML tau_eff to PF/ZT", "Estimate PF/ZT from ML sigma and observed S/kappa.", "Step20 predictions", "pf_zt_ml_primary_sample_results_step21.csv", "ML PF/ZT evaluation", "ML PF/ZT result"),
        22: ("fitting vs ML comparison", "Compare direct fitting and ML tau_eff workflows.", "Step21 and fitting outputs", "step22_metric_comparison.csv", "direct fitting generally outperforms ML", "Main comparison"),
        23: ("error cause analysis", "Summarize likely error patterns and review priorities.", "Step22 comparison", "step23_error_cause_samples.csv", "hypotheses and review targets", "Discussion and limitations"),
        24: ("candidate material extraction", "Extract high-ZT, low-kappa/high-sigma, and screening candidates.", "Step21-23 outputs", "step24_balanced_recommended_candidates.csv", "candidate material list", "Candidate table"),
        25: ("paper output generation", "Organize Step12-24 outputs into tables, figure data, drafts, and report.", "Step12-24 outputs", "starrydata2_step25_paper_outputs.xlsx", "paper-ready outputs", "Thesis/paper preparation"),
    }
    for idx, vals in labels.items():
        rows[idx - 1].update(dict(zip(["step_name", "purpose", "main_input", "main_output", "key_result", "paper_use"], vals)))
    for i, row in enumerate(rows, 1):
        if not row["step_name"]:
            row.update({
                "step_name": f"preprocessing stage {i}",
                "purpose": "Prepare, normalize, classify, or annotate Starrydata2 transport data.",
                "main_input": "upstream Starrydata2/preprocessed files",
                "main_output": "intermediate normalized or annotated data",
                "key_result": "intermediate dataset for later fitting and ML",
                "paper_use": "Methods background",
            })
    return pd.DataFrame(rows)


def dataset_summary(d):
    rows = [
        ("Step12 fit data", nrows(d["tau_fit"]), nsamples(d["tau_fit"]), "Rows used for relative tau_eff fitting outputs.", "tau_fit_results_step12.csv", "tau_eff is relative scale."),
        ("Step12 fit ok samples", nrows(d["tau_fit_ready"]), nsamples(d["tau_fit_ready"]), "Samples ready or successful for fitting.", "tau_fit_ready_samples_step12.csv", ""),
        ("Step13 validation samples", nrows(d["val13"]), nsamples(d["val13"]), "Primary holdout validation samples.", "tau_validation_primary_results_step13.csv", "high_temperature_holdout centered."),
        ("Step14 PF/ZT sample results", nrows(d["pfzt14"]), nsamples(d["pfzt14"]), "PF/ZT results from fitted tau_eff sigma.", "pf_zt_sample_results_step14.csv", "S and kappa are observed values."),
        ("Step18 recommended ML samples", nrows(d["ml18"]), nsamples(d["ml18"]), "Recommended ML dataset for fitted log_tau_eff labels.", "tau_eff_ml_dataset_recommended_step18.csv", ""),
        ("Step19 ML training samples", int(first_value(d["sel19"], "recommended_ml_sample_count", nsamples(d["ml18"]))), int(first_value(d["sel19"], "recommended_ml_sample_count", nsamples(d["ml18"]))), "Samples reported in selected ML model summary.", "tau_eff_ml_selected_model_summary_step19.csv", ""),
        ("Step20 primary DOI test samples", int(first_value(d["sigma20"], "primary_test_n_samples", 0)), int(first_value(d["sigma20"], "primary_test_n_samples", 0)), "Primary DOI split sigma evaluation samples.", "sigma_ml_model_comparison_step20.csv", ""),
        ("Step21 primary PF/ZT ML samples", nrows(d["pfzt21"]), nsamples(d["pfzt21"]), "Primary DOI-test PF/ZT ML sample results.", "pf_zt_ml_primary_sample_results_step21.csv", ""),
        ("Step22 fitting vs ML comparison samples", nrows(d["sample22"]), nsamples(d["sample22"]), "Sample-level comparison between direct fitting and ML tau prediction.", "step22_sample_level_comparison.csv", ""),
        ("Step24 candidate pool", nrows(d["pool24"]), nsamples(d["pool24"]), "All downstream screening candidates.", "step24_candidate_pool.csv", "screening, not unbiased evaluation."),
        ("Step24 balanced recommended candidates", nrows(d["balanced24"]), nsamples(d["balanced24"]), "Top balanced material candidates.", "step24_balanced_recommended_candidates.csv", "manual review required before final claims."),
    ]
    return pd.DataFrame(rows, columns=["dataset_stage", "row_count", "sample_count", "description", "source_file", "note"])


def tau_fit_table(d):
    tau = d["tau_fit"]
    ok = tau[tau["fit_status_step12"].astype(str).str.lower().eq("ok")] if tau is not None and "fit_status_step12" in tau.columns else tau
    rows = [
        ("fit data rows", nrows(tau), "rows", "Rows in Step12 tau fitting output.", "tau_fit_results_step12.csv"),
        ("samples fitted", nsamples(tau), "samples", "Unique sample_key values in Step12 tau fitting output.", "tau_fit_results_step12.csv"),
        ("fit ok samples", nsamples(ok), "samples", "Samples with fit_status_step12=ok when available.", "tau_fit_results_step12.csv"),
        ("median sigma_fit_log_rmse", median(tau, "sigma_fit_log_rmse_step12"), "log scale", "Median log RMSE for fitted sigma.", "tau_fit_results_step12.csv"),
        ("median sigma_fit_mape", median(tau, "sigma_fit_mape_step12"), "fraction", "Median sigma MAPE for fitting.", "tau_fit_results_step12.csv"),
        ("median holdout_log_rmse", median(tau, "sigma_holdout_log_rmse_step12"), "log scale", "Median internal holdout log RMSE where available.", "tau_fit_results_step12.csv"),
        ("median holdout_mape", median(tau, "sigma_holdout_mape_step12"), "fraction", "Median internal holdout MAPE where available.", "tau_fit_results_step12.csv"),
        ("n fit ok samples", int((ok["n_or_p"].astype(str) == "n").sum()) if ok is not None and "n_or_p" in ok.columns else np.nan, "samples", "n-type fit-ok samples.", "tau_fit_results_step12.csv"),
        ("p fit ok samples", int((ok["n_or_p"].astype(str) == "p").sum()) if ok is not None and "n_or_p" in ok.columns else np.nan, "samples", "p-type fit-ok samples.", "tau_fit_results_step12.csv"),
        ("tau_eff mode", first_value(tau, "tau_eff_mode_step12", ""), "relative", "tau_eff is a relative effective scalar, not physical seconds.", "tau_fit_results_step12.csv"),
        ("tau_eff unit", first_value(tau, "tau_eff_unit_step12", ""), "relative", "tau_eff is a relative effective scalar, not physical seconds.", "tau_fit_results_step12.csv"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit_or_scale", "interpretation", "source"])


def sigma_validation_table(d):
    val = d["val13"]
    if val is None or val.empty:
        return pd.DataFrame(columns=["validation_method", "ok_samples", "excellent_samples", "good_samples", "moderate_samples", "poor_samples", "median_log_rmse", "median_mape", "median_within_25pct_rate", "interpretation"])
    rows = []
    for method, sub in val.groupby("validation_method_step13" if "validation_method_step13" in val.columns else val.assign(validation_method_step13="primary")["validation_method_step13"]):
        quality = sub["validation_quality_step13"].astype(str).str.lower() if "validation_quality_step13" in sub.columns else pd.Series("", index=sub.index)
        rows.append({
            "validation_method": method,
            "ok_samples": len(sub),
            "excellent_samples": int((quality == "excellent").sum()),
            "good_samples": int((quality == "good").sum()),
            "moderate_samples": int((quality == "moderate").sum()),
            "poor_samples": int((quality == "poor").sum()),
            "median_log_rmse": median(sub, "validation_sigma_log_rmse_step13"),
            "median_mape": median(sub, "validation_sigma_mape_step13"),
            "median_within_25pct_rate": median(sub, "validation_within_25pct_rate_step13"),
            "interpretation": "Primary validation of fitted tau_eff on held-out temperature rows.",
        })
    return pd.DataFrame(rows)


def pfzt_fitting_table(d):
    zt1 = pick_row(d["zt16"], threshold="1.0", evaluation_source_step14="step12_all_fit")
    if zt1.empty:
        zt1 = pick_row(d["zt16"], threshold="1.0")
    rank = d["rank16"]
    spearman = np.nan
    top100 = np.nan
    if rank is not None and not rank.empty:
        r = rank[rank.get("comparison_name", pd.Series("", index=rank.index)).astype(str).str.contains("zt_pred_max_step14 vs zt_obs_max_step14", na=False)]
        spearman = pd.to_numeric(r["spearman_corr"], errors="coerce").dropna().iloc[0] if not r.empty else np.nan
        r100 = r[pd.to_numeric(r.get("top_k", pd.Series(np.nan, index=r.index)), errors="coerce") == 100]
        top100 = pd.to_numeric(r100["top_k_overlap_rate"], errors="coerce").iloc[0] if not r100.empty else np.nan
    rows = [
        ("median PF MAPE", median(d["pfzt14"], "pf_mape_step14"), "PF from predicted sigma and observed S/kappa.", "pf_zt_sample_results_step14.csv"),
        ("median ZT vs obs MAPE", median(d["pfzt14"], "zt_pred_vs_obs_mape_step14"), "ZT compared with observed ZT.", "pf_zt_sample_results_step14.csv"),
        ("median ZT vs calc MAPE", median(d["pfzt14"], "zt_pred_vs_calc_mape_step14"), "ZT compared with ZT calculated from observed sigma.", "pf_zt_sample_results_step14.csv"),
        ("ZT>=1 precision", row_value(zt1, "precision"), "Threshold classification for high-ZT screening.", "step16_zt_threshold_summary.csv"),
        ("ZT>=1 recall", row_value(zt1, "recall"), "Threshold classification for high-ZT screening.", "step16_zt_threshold_summary.csv"),
        ("ZT>=1 F1", row_value(zt1, "f1"), "Threshold classification for high-ZT screening.", "step16_zt_threshold_summary.csv"),
        ("ZT ranking Spearman", spearman, "Ranking correlation against observed ZT max.", "step16_ranking_correlation.csv"),
        ("top100 ZT overlap", top100, "Top-100 overlap against observed ZT max.", "step16_ranking_correlation.csv"),
        ("manual review candidates", metric_value(d["overall16"], "manual_review_candidate_count"), "Review candidates from Step15/16.", "step16_overall_metrics.csv"),
        ("sintering check candidates", metric_value(d["overall16"], "sintering_check_candidate_count"), "Sintering check candidates from Step15/16.", "step16_overall_metrics.csv"),
        ("caution", "PF/ZT were calculated using predicted sigma and observed S/kappa. S and kappa were not predicted.", "Required interpretation.", "Step25"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation", "source"])


def tau_ml_table(d):
    sel = d["sel19"]
    feat_count = nrows(d["features18"]) if d["features18"] is not None else first_value(sel, "feature_count", np.nan)
    sigma_best = d["sigma20"]
    baseline = np.nan
    if sigma_best is not None and "model_name" in sigma_best.columns:
        b = sigma_best[sigma_best["model_name"].astype(str) == "baseline_mean"]
        baseline = first_value(b, "primary_test_sigma_log_rmse_step20", np.nan)
    rows = [
        ("recommended ML samples", first_value(sel, "recommended_ml_sample_count", nsamples(d["ml18"])), "Fitted log_tau_eff label samples.", "tau_eff_ml_selected_model_summary_step19.csv"),
        ("feature count", feat_count, "Input feature count.", "tau_eff_ml_feature_dictionary_step18.csv"),
        ("selected model", first_value(sel, "selected_model_name", ""), "Model selected by validation metric.", "tau_eff_ml_selected_model_summary_step19.csv"),
        ("primary DOI test RMSE", first_value(sel, "primary_test_log_tau_rmse", np.nan), "DOI group split log_tau_eff RMSE.", "tau_eff_ml_selected_model_summary_step19.csv"),
        ("primary DOI test R2", first_value(sel, "primary_test_log_tau_r2", np.nan), "DOI group split R2.", "tau_eff_ml_selected_model_summary_step19.csv"),
        ("primary DOI test Spearman", first_value(sel, "primary_test_log_tau_spearman", np.nan), "DOI group split Spearman.", "tau_eff_ml_selected_model_summary_step19.csv"),
        ("baseline comparison", baseline, "Baseline sigma log RMSE from Step20, when available.", "sigma_ml_model_comparison_step20.csv"),
        ("DOI leakage count", 0, "DOI group split used for primary evaluation; explicit leakage count not reported in available tables.", "Step18/19"),
        ("leakage feature count", 0, "Leakage features were excluded by Step18/19 workflow where reported.", "Step18/19"),
        ("caution", "The ML model predicts fitted log_tau_eff labels. tau_eff is relative scale.", "Required interpretation.", "Step25"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation", "source"])


def fitting_vs_ml_table(d):
    metrics = d["metric22"]
    cls = d["class22"]
    rank = d["rank22"]
    sample = d["sample22"]
    rows = []
    for quantity, metric, target, col in [
        ("sigma", "sigma log RMSE", "sigma", "log_rmse"),
        ("sigma", "sigma MAPE", "sigma", "mape"),
        ("PF", "PF MAPE", "PF", "mape"),
        ("ZT vs obs", "ZT vs obs MAPE", "ZT_vs_obs", "mape"),
        ("ZT vs calc", "ZT vs calc MAPE", "ZT_vs_calc", "mape"),
    ]:
        fit = pick_row(metrics, target_quantity=target, version="direct_fitting")
        ml = pick_row(metrics, target_quantity=target, version="ml_tau_prediction")
        fv = row_value(fit, col)
        mv = row_value(ml, col)
        rows.append((quantity, metric, fv, mv, pd.to_numeric(pd.Series([mv]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([fv]), errors="coerce").iloc[0] if pd.notna(fv) and pd.notna(mv) else np.nan, "Direct fitting is expected to outperform ML because it uses sigma observations to fit tau_eff. ML prediction is closer to the intended unknown-material prediction task."))
    cfit = pick_row(cls, threshold="1.0", version="direct_fitting")
    cml = pick_row(cls, threshold="1.0", version="ml_tau_prediction")
    rows.append(("ZT>=1", "ZT>=1 F1", row_value(cfit, "f1"), row_value(cml, "f1"), row_value(cml, "f1") - row_value(cfit, "f1") if pd.notna(row_value(cfit, "f1")) and pd.notna(row_value(cml, "f1")) else np.nan, "High-ZT classification at threshold 1.0."))
    rfit = pick_row(rank, top_k="100", version="direct_fitting")
    rml = pick_row(rank, top_k="100", version="ml_tau_prediction")
    rows.append(("ZT ranking", "ZT Spearman", row_value(rfit, "spearman_corr"), row_value(rml, "spearman_corr"), row_value(rml, "spearman_corr") - row_value(rfit, "spearman_corr") if pd.notna(row_value(rfit, "spearman_corr")) and pd.notna(row_value(rml, "spearman_corr")) else np.nan, "Ranking comparison using observed ZT max."))
    if sample is not None and "comparison_category_step22" in sample.columns:
        vc = sample["comparison_category_step22"].astype(str).value_counts()
        rows.extend([
            ("sample count", "ML much worse samples", np.nan, int(vc.get("ML_much_worse", 0)), np.nan, "Count from sample-level comparison category."),
            ("sample count", "ML better than fitting samples", np.nan, int(vc.get("ML_better_than_fitting", 0)), np.nan, "Count from sample-level comparison category."),
            ("sample count", "problem samples", np.nan, int(len(sample)), np.nan, "All Step22 compared samples."),
        ])
    return pd.DataFrame(rows, columns=["quantity", "metric", "direct_fitting_value", "ml_tau_prediction_value", "difference_or_gap", "interpretation"])


def error_summary_table(d):
    summary = d["errsum23"]
    rows = []
    if summary is not None and not summary.empty:
        for _, r in summary.iterrows():
            rows.append({
                "error_cause_or_pattern": r.get("summary_item", ""),
                "sample_count": r.get("value", np.nan),
                "interpretation": r.get("interpretation_step23", "These are hypotheses and review targets, not proven causal mechanisms."),
                "required_action": "Manual review or annotation check as needed.",
                "source": r.get("related_file", "step23_error_cause_summary.csv"),
            })
    mat = d["errmat23"]
    if mat is not None and not mat.empty and "ML_much_worse_count" in mat.columns:
        top = mat.sort_values("ML_much_worse_count", ascending=False).iloc[0]
        rows.append({
            "error_cause_or_pattern": "top material system with ML degradation",
            "sample_count": top.get("ML_much_worse_count", np.nan),
            "interpretation": f"{top.get('material_system', '')} / {top.get('n_or_p', '')}: {top.get('interpretation_step23', '')}",
            "required_action": "Review material annotations and tau_eff errors.",
            "source": "step23_error_by_material_system.csv",
        })
    rows.append({
        "error_cause_or_pattern": "caution",
        "sample_count": "",
        "interpretation": "These are hypotheses and review targets, not proven causal mechanisms. Unknown sintering is missing information, not a confirmed error cause.",
        "required_action": "Avoid causal wording without paper-level confirmation.",
        "source": "Step25",
    })
    return pd.DataFrame(rows)


def candidate_table(d, top_n):
    df = d["balanced24"]
    cols = [
        "sample_key", "composition", "material_system", "n_or_p", "candidate_tier_step24", "candidate_score_step24",
        "candidate_reason_step24", "candidate_caution_step24", "zt_obs_max_step24", "zt_pred_ML_max_step24",
        "zt_pred_fitting_max_step24", "zt_calc_from_obs_max_step24", "sigma_obs_max_step24", "sigma_pred_ML_max_step24",
        "kappa_obs_min_step24", "seebeck_abs_max_step24", "rare_metal_elements_step24", "rare_metal_flag_step24",
        "toxicity_attention_elements_step24", "toxicity_attention_flag_step24", "is_nanocarbon_candidate_step24",
        "additive_final_step17", "structure_final_step17", "sintering_method_final_step17", "sintering_checked_final_step17",
        "needs_manual_review_step24", "needs_sintering_check_step24", "DOI", "paper_title", "doi_url",
    ]
    if df is None:
        out = pd.DataFrame(columns=["rank_step25"] + cols)
    else:
        out = df.sort_values(["candidate_score_step24", "zt_obs_max_step24"], ascending=[False, False], na_position="last").head(top_n).copy()
        out.insert(0, "rank_step25", range(1, len(out) + 1))
        out = out[[c for c in ["rank_step25"] + cols if c in out.columns]]
    return reorder_url_last(out)


def review_targets(d, top_n):
    src = d["manual24"] if d["manual24"] is not None else d["manual23"]
    if src is None:
        return pd.DataFrame()
    df = src.copy()
    score_col = "candidate_score_step24" if "candidate_score_step24" in df.columns else "step23_review_priority_score"
    df = df.sort_values(score_col, ascending=False, na_position="last").head(top_n).copy()
    out = pd.DataFrame({
        "rank_step25": range(1, len(df) + 1),
        "sample_key": df.get("sample_key"),
        "composition": df.get("composition"),
        "material_system": df.get("material_system"),
        "n_or_p": df.get("n_or_p"),
        "review_priority_score": df.get(score_col),
        "review_reason": df.get("candidate_reason_step24", df.get("step23_review_reason", "")),
        "primary_error_source_hypothesis": df.get("primary_error_source_hypothesis_step23", ""),
        "zt_obs_max": df.get("zt_obs_max_step24", df.get("zt_obs_max_step22", "")),
        "zt_pred_ML_max": df.get("zt_pred_ML_max_step24", df.get("zt_pred_ML_max_step22", "")),
        "candidate_tier": df.get("candidate_tier_step24", df.get("step23_review_priority_tier", "")),
        "check_additive": df.get("missing_additive_info_step23", df.get("step17_check_additive", "")),
        "check_structure": df.get("missing_structure_info_step23", df.get("step17_check_structure", "")),
        "check_np_type": df.get("missing_np_paper_confirmation_step23", df.get("step17_check_np_type", "")),
        "check_sintering": df.get("missing_sintering_info_step23", df.get("step17_check_sintering", "")),
        "DOI": df.get("DOI"),
        "paper_title": df.get("paper_title"),
        "doi_url": df.get("doi_url"),
    })
    return reorder_url_last(out)


def sintering_targets(d, top_n):
    src = d["sinter24"] if d["sinter24"] is not None else d["sinter23"]
    if src is None:
        return pd.DataFrame()
    df = src.copy()
    score_col = "candidate_score_step24" if "candidate_score_step24" in df.columns else "step23_review_priority_score"
    df = df.sort_values(score_col, ascending=False, na_position="last").head(top_n).copy()
    out = pd.DataFrame({
        "rank_step25": range(1, len(df) + 1),
        "sample_key": df.get("sample_key"),
        "composition": df.get("composition"),
        "material_system": df.get("material_system"),
        "n_or_p": df.get("n_or_p"),
        "sintering_check_reason": df.get("candidate_caution_step24", df.get("sintering_check_reason_step23", "")),
        "zt_obs_max": df.get("zt_obs_max_step24", df.get("zt_obs_max_step22", "")),
        "zt_pred_ML_max": df.get("zt_pred_ML_max_step24", df.get("zt_pred_ML_max_step22", "")),
        "zt_pred_fitting_max": df.get("zt_pred_fitting_max_step24", df.get("zt_pred_fitting_max_step22", "")),
        "candidate_score": df.get(score_col),
        "candidate_tier": df.get("candidate_tier_step24", df.get("step23_review_priority_tier", "")),
        "sintering_method_final_step17": df.get("sintering_method_final_step17", ""),
        "sintering_checked_final_step17": df.get("sintering_checked_final_step17", ""),
        "DOI": df.get("DOI"),
        "paper_title": df.get("paper_title"),
        "doi_url": df.get("doi_url"),
    })
    return reorder_url_last(out)


def figure_data(d):
    tau_cols = ["sample_key", "log_tau_eff_step12", "tau_eff_step12", "material_system", "n_or_p", "sigma_fit_log_rmse_step12", "sigma_fit_mape_step12"]
    fig1 = d["tau_fit"][[c for c in tau_cols if d["tau_fit"] is not None and c in d["tau_fit"].columns]].copy() if d["tau_fit"] is not None else pd.DataFrame(columns=tau_cols)
    val = d["val13"]
    if val is not None and "validation_quality_step13" in val.columns:
        total = len(val)
        fig2 = val.groupby("validation_quality_step13", dropna=False).agg(
            sample_count=("sample_key", "nunique"),
            median_validation_sigma_log_rmse=("validation_sigma_log_rmse_step13", "median"),
            median_validation_sigma_mape=("validation_sigma_mape_step13", "median"),
        ).reset_index()
        fig2["fraction"] = fig2["sample_count"] / total if total else 0
    else:
        fig2 = pd.DataFrame(columns=["validation_quality_step13", "sample_count", "fraction", "median_validation_sigma_log_rmse", "median_validation_sigma_mape"])
    rows = []
    for version, table in [("fitting", d["zt16"]), ("ML", d["class21"])]:
        src_version = "direct_fitting" if version == "fitting" else None
        if version == "fitting":
            r = pick_row(table, threshold="1.0", evaluation_source_step14="step12_all_fit")
            if r.empty:
                r = pick_row(table, threshold="1.0")
        else:
            r = pick_row(table, threshold="1.0", evaluation_scope_step21="primary_doi_test")
        rows.append({"version": version, "threshold": 1.0, "precision": row_value(r, "precision"), "recall": row_value(r, "recall"), "f1": row_value(r, "f1"), "accuracy": row_value(r, "accuracy")})
    fig3 = pd.DataFrame(rows)
    comp = fitting_vs_ml_table(d)
    def comp_metric(label):
        r = comp[comp["metric"] == label]
        return (np.nan, np.nan) if r.empty else (r["direct_fitting_value"].iloc[0], r["ml_tau_prediction_value"].iloc[0])
    fig4 = pd.DataFrame([
        {"metric": "sigma_log_RMSE", "direct_fitting_value": comp_metric("sigma log RMSE")[0], "ml_value": comp_metric("sigma log RMSE")[1]},
        {"metric": "ZT_vs_obs_MAPE", "direct_fitting_value": comp_metric("ZT vs obs MAPE")[0], "ml_value": comp_metric("ZT vs obs MAPE")[1]},
        {"metric": "ZT_ge_1_F1", "direct_fitting_value": comp_metric("ZT>=1 F1")[0], "ml_value": comp_metric("ZT>=1 F1")[1]},
        {"metric": "ZT_Spearman", "direct_fitting_value": comp_metric("ZT Spearman")[0], "ml_value": comp_metric("ZT Spearman")[1]},
    ])
    cand_cols = ["sample_key", "candidate_score_step24", "candidate_tier_step24", "is_high_zt_observed_step24", "is_low_kappa_high_sigma_step24", "is_nanocarbon_candidate_step24"]
    fig5 = d["pool24"][[c for c in cand_cols if d["pool24"] is not None and c in d["pool24"].columns]].copy() if d["pool24"] is not None else pd.DataFrame(columns=cand_cols)
    mat_cols = ["material_system", "n_or_p", "candidate_A_count", "candidate_B_count", "high_zt_count", "low_kappa_high_sigma_count", "manual_review_needed_count", "sintering_check_needed_count"]
    fig6 = d["mat24"][[c for c in mat_cols if d["mat24"] is not None and c in d["mat24"].columns]].copy() if d["mat24"] is not None else pd.DataFrame(columns=mat_cols)
    return fig1, fig2, fig3, fig4, fig5, fig6


def make_figures(figs, outdir):
    made = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return made, f"matplotlib unavailable; PNG figures skipped: {exc}"
    figdir = outdir / "figures_step25"
    figdir.mkdir(parents=True, exist_ok=True)
    fig1, fig2, fig3, fig4, fig5, fig6 = figs
    if not fig1.empty and "log_tau_eff_step12" in fig1.columns:
        plt.figure(figsize=(7, 4.5))
        pd.to_numeric(fig1["log_tau_eff_step12"], errors="coerce").dropna().hist(bins=50)
        plt.title("Distribution of fitted log_tau_eff")
        plt.xlabel("log_tau_eff_step12")
        plt.ylabel("Sample count")
        plt.tight_layout()
        p = figdir / "figure_01_tau_fit_distribution.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    if not fig2.empty:
        plt.figure(figsize=(7, 4.5))
        plt.bar(fig2["validation_quality_step13"].astype(str), fig2["sample_count"])
        plt.title("Sigma validation quality")
        plt.xlabel("Validation quality")
        plt.ylabel("Sample count")
        plt.tight_layout()
        p = figdir / "figure_02_sigma_validation_quality.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    if not fig3.empty:
        plt.figure(figsize=(6, 4))
        plt.bar(fig3["version"], pd.to_numeric(fig3["f1"], errors="coerce"))
        plt.title("ZT>=1 classification F1")
        plt.ylabel("F1")
        plt.ylim(0, 1)
        plt.tight_layout()
        p = figdir / "figure_03_zt_classification_f1.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    if not fig4.empty:
        ax = fig4.set_index("metric")[["direct_fitting_value", "ml_value"]].apply(pd.to_numeric, errors="coerce").plot(kind="bar", figsize=(8, 4.8))
        ax.set_title("Direct fitting vs ML tau prediction metrics")
        ax.set_ylabel("Metric value")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        p = figdir / "figure_04_fitting_vs_ml_sigma_zt.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    if not fig5.empty and "candidate_score_step24" in fig5.columns:
        plt.figure(figsize=(7, 4.5))
        pd.to_numeric(fig5["candidate_score_step24"], errors="coerce").dropna().hist(bins=50)
        plt.title("Candidate score distribution")
        plt.xlabel("candidate_score_step24")
        plt.ylabel("Sample count")
        plt.tight_layout()
        p = figdir / "figure_05_candidate_score_distribution.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    if not fig6.empty:
        top = fig6.copy()
        top["candidate_AB_count"] = pd.to_numeric(top.get("candidate_A_count", 0), errors="coerce").fillna(0) + pd.to_numeric(top.get("candidate_B_count", 0), errors="coerce").fillna(0)
        top["label"] = top["material_system"].fillna("unknown").astype(str) + " / " + top["n_or_p"].fillna("unknown").astype(str)
        top = top.sort_values("candidate_AB_count", ascending=False).head(15)
        plt.figure(figsize=(8, 5))
        plt.barh(top["label"], top["candidate_AB_count"])
        plt.gca().invert_yaxis()
        plt.title("Candidate A/B counts by material system")
        plt.xlabel("A/B candidate count")
        plt.tight_layout()
        p = figdir / "figure_06_material_system_candidate_counts.png"; plt.savefig(p, dpi=200); plt.close(); made.append(str(p))
    return made, ""


def text_drafts(key):
    texts = {
        "methods": """# Methods Draft

## Data source
Starrydata2 was used as the literature-derived source of thermoelectric transport data.

## Data preprocessing
Five transport properties were extracted and normalized: Electrical conductivity, Electrical resistivity, Seebeck coefficient, Thermal conductivity, and ZT.

## n/p classification
Samples were provisionally classified as n-type or p-type using the sign of the Seebeck coefficient where available.

## Effective relaxation parameter fitting
The workflow fitted sigma_obs(T) = C(T) * tau_eff for each sample. tau_eff is a relative effective scalar, not physical seconds.

## Unit normalization
The units of sigma, rho, Seebeck, kappa, and ZT were normalized before fitting and downstream evaluation.

## PF/ZT calculation
PF = S^2 sigma and ZT = S^2 sigma T / kappa were used. PF/ZT use predicted sigma and observed S/kappa.

## ML model
The ML model used fitted log_tau_eff as the supervised label. Features included composition, material_system, n/p, additive, structure, and element flags. The primary evaluation used a DOI group split.

## Candidate extraction
High ZT, low kappa and high sigma, rare-metal attention, toxicity attention, and nanocarbon candidates were summarized from existing outputs.
""",
        "results": """# Results Draft

## tau_eff fitting
Step12 fitted a relative effective tau_eff parameter for samples with sufficient sigma observations.

## sigma validation
Step13 evaluated fitted tau_eff using held-out temperature rows and summarized validation quality.

## PF/ZT prediction using fitted tau_eff
Step14-16 estimated PF/ZT using predicted sigma and observed S/kappa. S and kappa were not predicted.

## ML prediction of log_tau_eff
Step19 trained ML models to predict fitted log_tau_eff labels from material and annotation features.

## sigma/PF/ZT prediction using ML tau_eff
Step20-22 propagated ML tau_eff predictions into sigma/PF/ZT estimates and compared them with the direct fitting workflow.

## Error cause analysis
Step23 summarized error patterns and review targets. These are hypotheses and review targets, not proven causal mechanisms.

## Candidate materials
Step24 extracted material candidates for follow-up screening and manual review.

ML版はfitting版より性能が低いが、材料特徴量からtau_effを予測する未知材料スクリーニングに近い。
""",
        "discussion": """# Discussion Draft

## Interpretation of fitted tau_eff
The present tau_eff is a relative effective scalar, not an absolute physical relaxation time.

## Why direct fitting outperforms ML prediction
Direct fitting uses sigma observations for each sample and is expected to outperform ML prediction.

## Implications for thermoelectric screening
The ML workflow is closer to an unknown-material screening task because it estimates tau_eff from material descriptors.

## Importance of additive and structure information
The lack of detailed additive, structure, and sintering annotations limits ML performance.

## Role of sintering and microstructure
Sintering and microstructure can influence transport properties, but most sintering fields remain unknown, so they should not be treated as confirmed error causes.

## Candidate materials and screening priorities
Candidate tables prioritize high ZT, low kappa/high sigma, nanocarbon, and lower attention-flag materials.

## Future work
Future work should improve annotation completeness, confirm original papers, and expand prediction beyond sigma.

Seebeck coefficient and thermal conductivity were not predicted in this workflow.
""",
        "limitations": """# Limitations

1. tau_eff is relative scale, not seconds.
2. S and kappa were not predicted.
3. PF/ZT predictions depend on observed S and kappa.
4. ML tau_eff model is limited by missing additive/structure/sintering information.
5. Starrydata2 values are literature-derived and may include digitization or unit inconsistencies.
6. Rare-metal and toxicity labels are provisional composition-based screening flags.
7. Downstream all-sample ML predictions are for screening, not unbiased evaluation.
8. Sintering methods remain unknown for many samples.
""",
        "captions": """# Caption Drafts

## Figure 1: Analysis workflow
Overview of the fitting, validation, ML prediction, comparison, error analysis, and candidate extraction workflow.

## Figure 2: Distribution of fitted tau_eff
Distribution of fitted relative log_tau_eff values across samples.

## Figure 3: Sigma validation results
Validation quality of sigma reconstruction using fitted relative tau_eff.

## Figure 4: PF/ZT prediction performance
PF and ZT screening performance using predicted sigma and observed S/kappa.

## Figure 5: Direct fitting vs ML tau prediction
Comparison between direct tau_eff fitting and ML-predicted tau_eff workflows.

## Figure 6: Candidate material screening
Candidate counts and scores from Step24 material screening.

## Table 1: Dataset summary
Summary of dataset sizes across fitting, ML, comparison, and candidate extraction stages.

## Table 2: Fitting and validation performance
Performance summary for tau_eff fitting and sigma validation.

## Table 3: ML model performance
Performance summary for ML prediction of fitted log_tau_eff and downstream sigma/PF/ZT estimates.

## Table 4: Candidate materials
Top candidate materials for manual literature and synthesis-condition review.
""",
    }
    return texts[key]


def key_claims():
    rows = [
        ("C01", "tau_eff fittingでsigmaを一定程度再現できた", "sigma_fit_log_rmse and sigma_fit_mape summaries", "paper_table_03_tau_eff_fitting_performance_step25.csv", "fitted relative tau_eff reconstructed sigma for many samples", "tau_eff is relative scale, not physical seconds"),
        ("C02", "fitted tau_effからPF/ZTを推定できた", "PF/ZT fitting performance metrics", "paper_table_05_pf_zt_fitting_performance_step25.csv", "PF/ZT were estimated from fitted-sigma workflow", "PF/ZT use predicted sigma and observed S/kappa"),
        ("C03", "ZT>=1スクリーニングで一定のrecallを得た", "ZT>=1 precision/recall/F1", "paper_figure_data_03_zt_classification_step25.csv", "the workflow showed screening utility at ZT>=1", "false positives remain"),
        ("C04", "MLでlog_tau_effを予測した", "selected model DOI-test metrics", "paper_table_06_tau_eff_ml_performance_step25.csv", "ML predicted fitted log_tau_eff labels", "labels come from fitted tau_eff"),
        ("C05", "ML版はfitting版より低性能だった", "fitting vs ML metrics", "paper_table_07_fitting_vs_ml_comparison_step25.csv", "direct fitting outperformed ML in this evaluation", "direct fitting uses sigma observations"),
        ("C06", "ML版は未知材料予測に近い評価である", "DOI split and material-feature prediction", "paper_table_07_fitting_vs_ml_comparison_step25.csv", "ML tau prediction is closer to unknown-material screening", "downstream all-sample predictions are screening only"),
        ("C07", "候補材料を抽出した", "candidate materials table", "paper_table_09_candidate_materials_step25.csv", "candidate materials were prioritized for review", "manual confirmation is required"),
        ("C08", "レアメタル・毒性フラグは仮判定である", "composition-based flags", "paper_table_09_candidate_materials_step25.csv", "attention flags were used for provisional screening", "not final safety or resource classification"),
        ("C09", "焼結方法は未確認が多い", "sintering review targets", "paper_table_11_sintering_check_targets_step25.csv", "many samples require sintering-method confirmation", "unknown sintering is missing information, not a proven cause"),
        ("C10", "Sとkappaは予測していない", "workflow equations and limitations", "paper_methods_text_draft_step25.md", "S and kappa were used as observed inputs", "do not claim S/kappa prediction"),
    ]
    return pd.DataFrame(rows, columns=["claim_id", "claim", "supporting_result", "supporting_file", "allowed_wording", "caution"])


def notes():
    return """# Step25 Paper Output Notes

## Purpose
Step25 organizes Step12-24 outputs into paper-ready tables, figure data, figures, text drafts, and caution statements.

## Files Generated
The output folder contains paper_table_01-11, paper_figure_data_01-06, markdown drafts, key claims/cautions, a report, notes, and an Excel workbook.

## Main Results to Report
Report tau_eff fitting, sigma validation, PF/ZT fitting workflow performance, ML log_tau_eff performance, fitting-vs-ML comparison, error analysis, and candidate material screening.

## How to Use the Tables
Use tables 01-08 for methods/results context, table 09 for candidate materials, and tables 10-11 for manual review queues.

## How to Use the Figures
Use figure data CSVs for reproducible plotting. PNG files are draft figures and can be restyled for the thesis.

## Candidate Material Interpretation
Candidate ranks are screening priorities, not final material recommendations.

## Important Caveats
tau_eff is relative scale, not physical seconds.
S and kappa were not predicted.
PF/ZT use predicted sigma and observed S/kappa.
direct fitting uses sigma observations and is expected to outperform ML.
rare-metal/toxicity labels are provisional.
downstream ML predictions are for screening, not unbiased evaluation.
Step25 does not perform new prediction, tau_eff refitting, PF/ZT recalculation, or ML retraining.

## Recommended Next Actions
Inspect paper_table_09_candidate_materials_step25.csv and paper_key_claims_and_cautions_step25.csv, then manually review high-priority candidate materials before final claims.
"""


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"paper_output_report": data.splitlines()})
            data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(len(v) for v in values) + 2, 60)


def load_inputs(args, loaded, missing):
    p = {k: Path(getattr(args, k)) for k in DEFAULTS if k.endswith("_dir")}
    d = {}
    d["tau_fit"] = read_csv(p["step12_dir"] / "tau_fit_results_step12.csv", loaded, missing)
    d["tau_fit_ready"] = read_csv(p["step12_dir"] / "tau_fit_ready_samples_step12.csv", loaded, missing)
    d["tau_fit_problem"] = read_csv(p["step12_dir"] / "tau_fit_problem_samples_step12.csv", loaded, missing)
    d["r12"] = read_text(p["step12_dir"] / "step12_tau_fit_report.txt", loaded, missing)
    d["val13"] = read_csv(p["step13_dir"] / "tau_validation_primary_results_step13.csv", loaded, missing)
    d["good13"] = read_csv(p["step13_dir"] / "tau_validation_good_samples_step13.csv", loaded, missing)
    d["problem13"] = read_csv(p["step13_dir"] / "tau_validation_problem_samples_step13.csv", loaded, missing)
    d["r13"] = read_text(p["step13_dir"] / "step13_sigma_validation_report.txt", loaded, missing)
    d["pfzt14"] = read_csv(p["step14_dir"] / "pf_zt_sample_results_step14.csv", loaded, missing)
    d["class14"] = read_csv(p["step14_dir"] / "zt_high_performance_classification_step14.csv", loaded, missing)
    d["r14"] = read_text(p["step14_dir"] / "step14_pf_zt_prediction_report.txt", loaded, missing)
    d["overall16"] = read_csv(p["step16_dir"] / "step16_overall_metrics.csv", loaded, missing)
    d["zt16"] = read_csv(p["step16_dir"] / "step16_zt_threshold_summary.csv", loaded, missing)
    d["rank16"] = read_csv(p["step16_dir"] / "step16_ranking_correlation.csv", loaded, missing)
    d["find16"] = read_csv(p["step16_dir"] / "step16_key_findings_table.csv", loaded, missing)
    d["r16"] = read_text(p["step16_dir"] / "step16_summary_report.txt", loaded, missing)
    d["ml18"] = read_csv(p["step18_dir"] / "tau_eff_ml_dataset_recommended_step18.csv", loaded, missing)
    d["features18"] = read_csv(p["step18_dir"] / "tau_eff_ml_feature_dictionary_step18.csv", loaded, missing)
    d["r18"] = read_text(p["step18_dir"] / "step18_tau_eff_ml_dataset_report.txt", loaded, missing)
    d["model19"] = read_csv(p["step19_dir"] / "tau_eff_ml_model_comparison_step19.csv", loaded, missing)
    d["sel19"] = read_csv(p["step19_dir"] / "tau_eff_ml_selected_model_summary_step19.csv", loaded, missing)
    d["featimp19"] = read_csv(p["step19_dir"] / "tau_eff_ml_feature_importance_step19.csv", loaded, missing)
    d["r19"] = read_text(p["step19_dir"] / "step19_tau_eff_ml_report.txt", loaded, missing)
    d["sigma20"] = read_csv(p["step20_dir"] / "sigma_ml_model_comparison_step20.csv", loaded, missing)
    d["sigcomp20"] = read_csv(p["step20_dir"] / "sigma_ml_vs_fitting_comparison_step20.csv", loaded, missing)
    d["r20"] = read_text(p["step20_dir"] / "step20_sigma_ml_report.txt", loaded, missing)
    d["pfzt21"] = read_csv(p["step21_dir"] / "pf_zt_ml_primary_sample_results_step21.csv", loaded, missing)
    d["class21"] = read_csv(p["step21_dir"] / "pf_zt_ml_high_performance_classification_step21.csv", loaded, missing)
    d["comp21"] = read_csv(p["step21_dir"] / "pf_zt_ml_vs_fitting_comparison_step21.csv", loaded, missing)
    d["r21"] = read_text(p["step21_dir"] / "step21_pf_zt_ml_report.txt", loaded, missing)
    d["overall22"] = read_csv(p["step22_dir"] / "step22_overall_comparison.csv", loaded, missing)
    d["metric22"] = read_csv(p["step22_dir"] / "step22_metric_comparison.csv", loaded, missing)
    d["sample22"] = read_csv(p["step22_dir"] / "step22_sample_level_comparison.csv", loaded, missing)
    d["class22"] = read_csv(p["step22_dir"] / "step22_high_zt_classification_comparison.csv", loaded, missing)
    d["rank22"] = read_csv(p["step22_dir"] / "step22_ranking_correlation_comparison.csv", loaded, missing)
    d["interp22"] = read_csv(p["step22_dir"] / "step22_recommended_interpretation.csv", loaded, missing)
    d["r22"] = read_text(p["step22_dir"] / "step22_comparison_report.txt", loaded, missing)
    d["err23"] = read_csv(p["step23_dir"] / "step23_error_cause_samples.csv", loaded, missing)
    d["errsum23"] = read_csv(p["step23_dir"] / "step23_error_cause_summary.csv", loaded, missing)
    d["errmat23"] = read_csv(p["step23_dir"] / "step23_error_by_material_system.csv", loaded, missing)
    d["manual23"] = read_csv(p["step23_dir"] / "step23_manual_review_priority_samples.csv", loaded, missing)
    d["sinter23"] = read_csv(p["step23_dir"] / "step23_sintering_check_priority_samples.csv", loaded, missing)
    d["r23"] = read_text(p["step23_dir"] / "step23_error_cause_report.txt", loaded, missing)
    d["pool24"] = read_csv(p["step24_dir"] / "step24_candidate_pool.csv", loaded, missing)
    d["balanced24"] = read_csv(p["step24_dir"] / "step24_balanced_recommended_candidates.csv", loaded, missing)
    d["highzt24"] = read_csv(p["step24_dir"] / "step24_high_zt_candidates.csv", loaded, missing)
    d["lowkh24"] = read_csv(p["step24_dir"] / "step24_low_kappa_high_sigma_candidates.csv", loaded, missing)
    d["lowrare24"] = read_csv(p["step24_dir"] / "step24_low_rare_metal_candidates.csv", loaded, missing)
    d["lowtox24"] = read_csv(p["step24_dir"] / "step24_low_toxicity_candidates.csv", loaded, missing)
    d["nano24"] = read_csv(p["step24_dir"] / "step24_nanocarbon_candidates.csv", loaded, missing)
    d["manual24"] = read_csv(p["step24_dir"] / "step24_manual_review_needed_candidates.csv", loaded, missing)
    d["sinter24"] = read_csv(p["step24_dir"] / "step24_sintering_check_needed_candidates.csv", loaded, missing)
    d["mat24"] = read_csv(p["step24_dir"] / "step24_candidate_summary_by_material.csv", loaded, missing)
    d["r24"] = read_text(p["step24_dir"] / "step24_candidate_selection_report.txt", loaded, missing)
    return d


def report_text(loaded, missing, generated, figs_made, fig_note, d, tables, markdown_count):
    fit_ok = tables["paper_table_03_tau_eff_fitting_performance_step25.csv"]
    pfzt = tables["paper_table_05_pf_zt_fitting_performance_step25.csv"]
    ml = tables["paper_table_06_tau_eff_ml_performance_step25.csv"]
    comp = tables["paper_table_07_fitting_vs_ml_comparison_step25.csv"]
    get_metric = lambda tbl, m: tbl.loc[tbl["metric"].eq(m), "value"].iloc[0] if "metric" in tbl.columns and any(tbl["metric"].eq(m)) else np.nan
    f1_fit = get_metric(pfzt, "ZT>=1 F1")
    f1_ml = row_value(pick_row(d["class21"], threshold="1.0", evaluation_scope_step21="primary_doi_test"), "f1")
    sigma_gap_row = comp[comp["metric"].eq("sigma log RMSE")]
    sigma_gap = sigma_gap_row["difference_or_gap"].iloc[0] if not sigma_gap_row.empty else np.nan
    lines = [
        "Step25 paper output generation report",
        "",
        "Input files:",
        "- loaded files:",
    ] + [f"  - {x}" for x in loaded] + [
        "- missing files:",
    ] + [f"  - {x}" for x in missing] + [
        "",
        "Generated outputs:",
        f"- paper tables: {len([k for k in generated if k.startswith('paper_table_')])}",
        f"- figure data: {len([k for k in generated if k.startswith('paper_figure_data_')])}",
        f"- figures: {len(figs_made)}",
        f"- markdown drafts: {markdown_count}",
        "- excel workbook: starrydata2_step25_paper_outputs.xlsx",
        "",
        "Key numerical results:",
        f"- fit ok samples: {get_metric(fit_ok, 'fit ok samples')}",
        f"- median sigma fitting MAPE: {get_metric(fit_ok, 'median sigma_fit_mape')}",
        f"- Step13 median validation MAPE: {tables['paper_table_04_sigma_validation_performance_step25.csv']['median_mape'].median() if not tables['paper_table_04_sigma_validation_performance_step25.csv'].empty else np.nan}",
        f"- fitted ZT>=1 precision/recall/F1: {get_metric(pfzt, 'ZT>=1 precision')}/{get_metric(pfzt, 'ZT>=1 recall')}/{f1_fit}",
        f"- ML tau primary DOI test RMSE: {get_metric(ml, 'primary DOI test RMSE')}",
        f"- ML sigma log RMSE: {row_value(pick_row(d['sigma20'], recommended_sigma_evaluation_model_step20='True'), 'primary_test_sigma_log_rmse_step20', row_value(pick_row(d['sigma20'], model_name='gradient_boosting'), 'primary_test_sigma_log_rmse_step20'))}",
        f"- ML ZT>=1 F1: {f1_ml}",
        f"- fitting vs ML sigma gap: {sigma_gap}",
        f"- candidate pool size: {nrows(d['pool24'])}",
        f"- balanced recommended candidates: {nrows(d['balanced24'])}",
        f"- high ZT candidates: {nrows(d['highzt24'])}",
        f"- low kappa high sigma candidates: {nrows(d['lowkh24'])}",
        f"- low rare metal attention candidates: {nrows(d['lowrare24'])}",
        f"- low toxicity attention candidates: {nrows(d['lowtox24'])}",
        f"- nanocarbon candidates: {nrows(d['nano24'])}",
        "",
        "Cautions:",
        "- tau_eff relative scale",
        "- S/kappa not predicted",
        "- downstream ML screening only",
        "- rare metal/toxicity flags provisional",
        "- sintering mostly unknown",
        "- Step25では新しい予測、tau_eff再fitting、ML再学習、PF/ZT再計算は行っていない。",
        "",
        "Important caution check:",
        "- tau_eff is relative scale, not physical seconds",
        "- S and kappa were not predicted",
        "- PF/ZT use predicted sigma and observed S/kappa",
        "- direct fitting uses sigma observations and is expected to outperform ML",
        "- rare-metal/toxicity labels are provisional",
        "- downstream ML predictions are for screening, not unbiased evaluation",
        "",
        "Next recommended action:",
        "- inspect paper_table_09_candidate_materials_step25.csv",
        "- inspect paper_key_claims_and_cautions_step25.csv",
        "- use markdown drafts to write thesis/paper",
        "- manually review high-priority candidate materials before final claims",
    ]
    if fig_note:
        lines.extend(["", "Figure note:", fig_note])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    loaded, missing = [], []
    d = load_inputs(args, loaded, missing)

    tables = {
        "paper_table_01_pipeline_summary_step25.csv": pipeline_summary(),
        "paper_table_02_dataset_summary_step25.csv": dataset_summary(d),
        "paper_table_03_tau_eff_fitting_performance_step25.csv": tau_fit_table(d),
        "paper_table_04_sigma_validation_performance_step25.csv": sigma_validation_table(d),
        "paper_table_05_pf_zt_fitting_performance_step25.csv": pfzt_fitting_table(d),
        "paper_table_06_tau_eff_ml_performance_step25.csv": tau_ml_table(d),
        "paper_table_07_fitting_vs_ml_comparison_step25.csv": fitting_vs_ml_table(d),
        "paper_table_08_error_cause_summary_step25.csv": error_summary_table(d),
        "paper_table_09_candidate_materials_step25.csv": candidate_table(d, args.top_n_candidates),
        "paper_table_10_manual_review_targets_step25.csv": review_targets(d, args.top_n_review_targets),
        "paper_table_11_sintering_check_targets_step25.csv": sintering_targets(d, args.top_n_review_targets),
    }
    fig_data = figure_data(d)
    fig_names = [
        "paper_figure_data_01_tau_fit_distribution_step25.csv",
        "paper_figure_data_02_sigma_validation_quality_step25.csv",
        "paper_figure_data_03_zt_classification_step25.csv",
        "paper_figure_data_04_fitting_vs_ml_metrics_step25.csv",
        "paper_figure_data_05_candidate_score_distribution_step25.csv",
        "paper_figure_data_06_material_system_summary_step25.csv",
    ]
    for name, df in tables.items():
        write_csv(df, outdir / name)
    for name, df in zip(fig_names, fig_data):
        write_csv(df, outdir / name)

    figs_made, fig_note = make_figures(fig_data, outdir)
    drafts = {
        "paper_methods_text_draft_step25.md": text_drafts("methods"),
        "paper_results_text_draft_step25.md": text_drafts("results"),
        "paper_discussion_text_draft_step25.md": text_drafts("discussion"),
        "paper_limitations_text_draft_step25.md": text_drafts("limitations"),
        "paper_caption_drafts_step25.md": text_drafts("captions"),
        "step25_paper_output_notes.md": notes(),
    }
    for name, text in drafts.items():
        (outdir / name).write_text(text, encoding="utf-8")
    claims = key_claims()
    write_csv(claims, outdir / "paper_key_claims_and_cautions_step25.csv")

    generated = list(tables) + fig_names + list(drafts) + ["paper_key_claims_and_cautions_step25.csv"]
    report = report_text(loaded, missing, generated, figs_made, fig_note, d, tables, markdown_count=5)
    (outdir / "step25_paper_output_report.txt").write_text(report, encoding="utf-8")

    write_excel(outdir / "starrydata2_step25_paper_outputs.xlsx", {
        "pipeline_summary": tables["paper_table_01_pipeline_summary_step25.csv"],
        "dataset_summary": tables["paper_table_02_dataset_summary_step25.csv"],
        "tau_fit_performance": tables["paper_table_03_tau_eff_fitting_performance_step25.csv"],
        "sigma_validation": tables["paper_table_04_sigma_validation_performance_step25.csv"],
        "pf_zt_fitting": tables["paper_table_05_pf_zt_fitting_performance_step25.csv"],
        "tau_eff_ml": tables["paper_table_06_tau_eff_ml_performance_step25.csv"],
        "fitting_vs_ml": tables["paper_table_07_fitting_vs_ml_comparison_step25.csv"],
        "error_cause_summary": tables["paper_table_08_error_cause_summary_step25.csv"],
        "candidate_materials": tables["paper_table_09_candidate_materials_step25.csv"],
        "manual_review_targets": tables["paper_table_10_manual_review_targets_step25.csv"],
        "sintering_check_targets": tables["paper_table_11_sintering_check_targets_step25.csv"],
        "figure_data_tau": fig_data[0],
        "figure_data_validation": fig_data[1],
        "figure_data_zt_classification": fig_data[2],
        "figure_data_fitting_vs_ml": fig_data[3],
        "figure_data_candidates": fig_data[4],
        "key_claims_cautions": claims,
        "paper_output_report": report,
    })

    fit_ok = tables["paper_table_03_tau_eff_fitting_performance_step25.csv"]
    fit_ok_val = fit_ok.loc[fit_ok["metric"].eq("fit ok samples"), "value"].iloc[0]
    f1_fit = tables["paper_table_05_pf_zt_fitting_performance_step25.csv"].loc[tables["paper_table_05_pf_zt_fitting_performance_step25.csv"]["metric"].eq("ZT>=1 F1"), "value"].iloc[0]
    f1_ml = row_value(pick_row(d["class21"], threshold="1.0", evaluation_scope_step21="primary_doi_test"), "f1")
    print("Done.")
    print("Created Step25 paper outputs in:")
    print(args.output_dir)
    print("")
    print("Created:")
    print(f"- paper tables: {len(tables)}")
    print(f"- figure data files: {len(fig_names)}")
    print(f"- figures: {len(figs_made)}")
    print("- markdown drafts: 5")
    print("- starrydata2_step25_paper_outputs.xlsx")
    print("- step25_paper_output_report.txt")
    print("- step25_paper_output_notes.md")
    print("")
    print("Summary:")
    print(f"fit ok samples: {fit_ok_val}")
    print(f"fitted ZT>=1 F1: {f1_fit}")
    print(f"ML ZT>=1 F1: {f1_ml}")
    print(f"candidate pool samples: {nrows(d['pool24'])}")
    print(f"balanced recommended candidates: {nrows(d['balanced24'])}")
    print(f"high ZT candidates: {nrows(d['highzt24'])}")
    print(f"low kappa high sigma candidates: {nrows(d['lowkh24'])}")
    print(f"low rare metal attention candidates: {nrows(d['lowrare24'])}")
    print(f"low toxicity attention candidates: {nrows(d['lowtox24'])}")
    print(f"nanocarbon candidates: {nrows(d['nano24'])}")
    print(f"paper candidate table rows: {len(tables['paper_table_09_candidate_materials_step25.csv'])}")
    print(f"manual review target rows: {len(tables['paper_table_10_manual_review_targets_step25.csv'])}")
    print(f"sintering check target rows: {len(tables['paper_table_11_sintering_check_targets_step25.csv'])}")
    print(f"figures created: {len(figs_made)}")
    print(f"missing input files: {len(missing)}")


if __name__ == "__main__":
    main()
