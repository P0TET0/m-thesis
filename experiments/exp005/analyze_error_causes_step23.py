import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP22_DIR = "data/output/starrydata2_step22_fitting_vs_ml_comparison"
DEFAULT_STEP17_DIR = "data/output/starrydata2_step17_literature_review"
DEFAULT_STEP18_DIR = "data/output/starrydata2_step18_tau_eff_ml_dataset"
DEFAULT_STEP19_DIR = "data/output/starrydata2_step19_tau_eff_ml_model"
DEFAULT_STEP15_DIR = "data/output/starrydata2_step15_pf_zt_error_analysis"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step23_error_cause_analysis"
EXCEL_PREVIEW_ROWS = 100_000

STRING_COLUMNS = ["sample_key", "SID", "DOI", "doi_url", "sample_id", "composition", "material_system", "n_or_p"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Step23 error-cause candidates from fitting-vs-ML comparison.")
    parser.add_argument("--step22_dir", default=DEFAULT_STEP22_DIR)
    parser.add_argument("--step17_dir", default=DEFAULT_STEP17_DIR)
    parser.add_argument("--step18_dir", default=DEFAULT_STEP18_DIR)
    parser.add_argument("--step19_dir", default=DEFAULT_STEP19_DIR)
    parser.add_argument("--step15_dir", default=DEFAULT_STEP15_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_manual_review", type=int, default=500)
    parser.add_argument("--top_n_sintering_check", type=int, default=300)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {c: "string" for c in STRING_COLUMNS if c in header.columns}


def read_csv(path, required=False, usecols=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        return None
    header = pd.read_csv(path, nrows=0)
    kwargs = {"dtype": dtype_for_existing(path), "low_memory": False}
    if usecols:
        kwargs["usecols"] = [c for c in usecols if c in header.columns]
    return pd.read_csv(path, **kwargs)


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def first_by_key(df):
    if df is None or "sample_key" not in df.columns:
        return None
    return df.drop_duplicates("sample_key", keep="first").copy()


def merge_optional(base, df, cols, suffix):
    df = first_by_key(df)
    if df is None:
        return base
    keep = ["sample_key"] + [c for c in cols if c in df.columns and c != "sample_key"]
    sub = df[keep].copy()
    rename = {c: f"{c}__{suffix}" for c in sub.columns if c != "sample_key" and c in base.columns}
    sub = sub.rename(columns=rename)
    return base.merge(sub, on="sample_key", how="left")


def unknown(series):
    return series.isna() | series.astype(str).str.strip().str.lower().isin(["", "unknown", "nan", "<na>", "not_checked"])


def bool_series(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def coalesce(df, out, candidates, default="unknown"):
    vals = pd.Series(default, index=df.index, dtype=object)
    set_any = pd.Series(False, index=df.index)
    for c in candidates:
        if c not in df.columns:
            continue
        s = df[c]
        ok = ~unknown(s)
        vals = vals.where(set_any | ~ok, s)
        set_any = set_any | ok
    df[out] = vals
    return df


def classify_pattern(row):
    sf = row.get("sigma_fitting_log_rmse_step22")
    sm = row.get("sigma_ML_log_rmse_step22")
    zf = row.get("zt_obs_fitting_mape_step22")
    zm = row.get("zt_obs_ML_mape_step22")
    vals = pd.to_numeric(pd.Series([sf, sm, zf, zm]), errors="coerce")
    if vals.isna().all():
        return "not_evaluable"
    fitting_good_ml_bad = ((pd.notna(zf) and pd.notna(zm) and zf <= 0.5 and zm > 1.0) or (pd.notna(sf) and pd.notna(sm) and sf <= 0.4 and sm > 1.0))
    if fitting_good_ml_bad:
        return "fitting_good_ML_bad"
    if pd.notna(zf) and pd.notna(zm) and zf > 1.0 and zm > 1.0:
        return "fitting_bad_ML_bad"
    if pd.notna(zf) and pd.notna(zm) and zf > 1.0 and zm <= 1.0:
        return "fitting_bad_ML_good"
    if pd.notna(zf) and pd.notna(zm) and zf <= 0.5 and zm <= 0.5:
        return "both_good"
    if pd.notna(sf) and pd.notna(sm) and sf > 1.0 and sm > 1.0:
        return "both_poor"
    return "both_poor" if row.get("comparison_category_step22") == "ML_much_worse" else "not_evaluable"


def hypotheses(row):
    pattern = row["error_pattern_step23"]
    missing_add = bool(row["missing_additive_info_step23"])
    missing_struct = bool(row["missing_structure_info_step23"])
    missing_sint = bool(row["missing_sintering_info_step23"])
    tau_err = pd.to_numeric(pd.Series([row.get("abs_error_log_tau_eff_step19")]), errors="coerce").iloc[0]
    sigma_fit = pd.to_numeric(pd.Series([row.get("sigma_fitting_log_rmse_step22")]), errors="coerce").iloc[0]
    zobs = pd.to_numeric(pd.Series([row.get("zt_obs_ML_mape_step22")]), errors="coerce").iloc[0]
    zcalc = pd.to_numeric(pd.Series([row.get("zt_calc_ML_mape_step22")]), errors="coerce").iloc[0]
    high_or_error = pd.to_numeric(pd.Series([row.get("zt_obs_max_step22")]), errors="coerce").iloc[0] >= 1.0 or row.get("comparison_category_step22") == "ML_much_worse"

    if pattern == "not_evaluable":
        primary = "not_evaluable"
    elif pattern == "fitting_good_ML_bad" and pd.notna(tau_err) and tau_err > 1.0:
        primary = "tau_eff_ML_prediction_error"
    elif pattern == "fitting_good_ML_bad" and (missing_add or missing_struct):
        primary = "insufficient_material_features"
    elif (missing_add or missing_struct) and row.get("comparison_category_step22") == "ML_much_worse":
        primary = "missing_additive_or_structure_annotation"
    elif pattern == "fitting_bad_ML_bad" and missing_sint and high_or_error:
        primary = "possible_sintering_or_microstructure_effect"
    elif pd.notna(zobs) and pd.notna(zcalc) and zobs > 1.0 and zcalc <= 1.0:
        primary = "possible_ZT_observation_or_unit_inconsistency"
    elif pd.notna(sigma_fit) and sigma_fit > 1.0:
        primary = "direct_fitting_or_prefactor_limitation"
    elif str(row.get("material_system", "")).lower() == "unknown":
        primary = "material_system_out_of_distribution"
    else:
        primary = "n_or_p_specific_error" if row.get("n_or_p") in ["n", "p"] else "insufficient_material_features"

    secondary = []
    if missing_add or missing_struct:
        secondary.append("missing_additive_or_structure_annotation")
    if missing_sint and high_or_error:
        secondary.append("possible_sintering_or_microstructure_effect")
    if pd.notna(tau_err) and tau_err > 1.0:
        secondary.append("tau_eff_ML_prediction_error")
    if pd.notna(sigma_fit) and sigma_fit > 1.0:
        secondary.append("direct_fitting_or_prefactor_limitation")
    return primary, "; ".join(dict.fromkeys([s for s in secondary if s != primary])) or "none"


def add_review_priority(df, zt_threshold):
    scores = []
    reasons = []
    for _, row in df.iterrows():
        score = 0
        reason = []
        high_zt = pd.to_numeric(pd.Series([row.get("zt_obs_max_step22")]), errors="coerce").iloc[0] >= zt_threshold
        ml_high = pd.to_numeric(pd.Series([row.get("zt_pred_ML_max_step22")]), errors="coerce").iloc[0] >= zt_threshold
        fit_high = pd.to_numeric(pd.Series([row.get("zt_pred_fitting_max_step22")]), errors="coerce").iloc[0] >= zt_threshold
        if row.get("comparison_category_step22") == "ML_much_worse":
            score += 30; reason.append("ML much worse than fitting")
        if row.get("error_pattern_step23") == "fitting_bad_ML_bad":
            score += 25; reason.append("fitting and ML both poor")
        if high_zt:
            score += 30; reason.append("observed high ZT")
        if high_zt and not ml_high:
            score += 30; reason.append("ZT_ML false negative")
        if (not high_zt) and ml_high:
            score += 25; reason.append("ZT_ML false positive")
        if row.get("missing_additive_info_step23") and row.get("comparison_category_step22") == "ML_much_worse":
            score += 15; reason.append("missing additive info and ML bad")
        if row.get("missing_structure_info_step23") and row.get("comparison_category_step22") == "ML_much_worse":
            score += 15; reason.append("missing structure info and ML bad")
        if row.get("missing_sintering_info_step23") and (high_zt or row.get("comparison_category_step22") == "ML_much_worse"):
            score += 20; reason.append("missing sintering info for high-ZT/error sample")
        if bool(row.get("nanocarbon_keyword_detected_step9")):
            score += 15; reason.append("nanocarbon candidate")
        if str(row.get("manual_review_status_step17", "unknown")).lower() in ["unknown", "not_checked", "nan", ""]:
            score += 10; reason.append("manual review not checked")
        scores.append(score)
        reasons.append("; ".join(reason) if reason else "low priority")
    df["step23_review_priority_score"] = scores
    df["step23_review_priority_tier"] = pd.cut(df["step23_review_priority_score"], bins=[-1, 24, 49, 79, 999], labels=["low", "C", "B", "A"]).astype(str)
    df["step23_review_reason"] = reasons
    return df


def summarize_group(df, group_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        sample_count=("sample_key", "nunique"),
        ML_much_worse_count=("comparison_category_step22", lambda s: int((s == "ML_much_worse").sum())),
        fitting_good_ML_bad_count=("error_pattern_step23", lambda s: int((s == "fitting_good_ML_bad").sum())),
        fitting_bad_ML_bad_count=("error_pattern_step23", lambda s: int((s == "fitting_bad_ML_bad").sum())),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        missing_additive_count=("missing_additive_info_step23", "sum"),
        missing_structure_count=("missing_structure_info_step23", "sum"),
        missing_sintering_count=("missing_sintering_info_step23", "sum"),
        manual_annotation_needed_count=("manual_annotation_needed_step23", "sum"),
    ).reset_index()
    out["interpretation_step23"] = np.where(out["ML_much_worse_count"] > 0, "review high-error ML degradation cases; do not infer causality", "limited ML degradation in this group")
    return out


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
    return """# Step23 Error Cause Analysis Notes

## Purpose
Organize candidate causes for fitting-vs-ML error differences using material annotations.

## Inputs
Step23 primarily uses Step22 sample-level comparison results and enriches them with Step17/18/19/15 metadata when available.

## Error Pattern Definitions
Patterns separate cases where fitting is good but ML is bad, both are bad, fitting is bad but ML is better, both are good, and not evaluable cases.

## Main Error Hypotheses
Hypotheses include ML tau_eff prediction error, insufficient material features, missing additive/structure annotations, possible sintering or microstructure effects, ZT observation/unit inconsistency, and direct fitting limitations.

## Material System Trends
See `step23_error_by_material_system.csv`.

## n/p Type Trends
See `step23_error_by_np_type.csv`.

## Additive and Structure Information
Unknown additive and structure groups are treated as missing information and review priorities, not proven causes.

## Sintering Information Policy
Sintering methods are mostly unknown; unknown sintering is treated as missing information, not as a confirmed error cause.

## High-ZT Error Cases
See `step23_high_zt_error_cases.csv`.

## Recommended Manual Review
See `step23_manual_review_priority_samples.csv` and `step23_sintering_check_priority_samples.csv`.

## Important Caveats
Step23 does not prove causal mechanisms.
Sintering methods are mostly unknown; unknown sintering is treated as missing information, not as a confirmed error cause.
ML errors may reflect insufficient features, missing annotations, or poor generalization under DOI split.
Seebeck coefficient and thermal conductivity were not predicted; PF/ZT errors depend on observed S and kappa.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.

## Next Step
Step24 should use these review priorities and feature flags to extract candidate materials.
"""


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    step22 = Path(args.step22_dir)
    samples = read_csv(step22 / "step22_sample_level_comparison.csv", required=True)
    problems22 = read_csv(step22 / "step22_problem_samples.csv", required=True)
    material22 = read_csv(step22 / "step22_material_summary_comparison.csv", required=True)
    read_csv(step22 / "step22_np_summary_comparison.csv", required=True)
    read_csv(step22 / "step22_ml_degradation_analysis.csv", required=True)

    require_columns(samples, ["sample_key", "material_system", "n_or_p", "composition", "comparison_category_step22"], "step22_sample_level_comparison.csv")

    step17_cols = [
        "sample_key", "n_or_p_final_step17", "n_or_p_source_step17", "n_or_p_confidence_final_step17",
        "additive_final_step17", "additive_source_step17", "additive_confidence_final_step17",
        "structure_final_step17", "structure_source_step17", "structure_confidence_final_step17",
        "sintering_method_final_step17", "sintering_condition_final_step17", "sintering_checked_final_step17", "sintering_source_step17", "sintering_confidence_final_step17",
        "paper_checked_step17", "manual_review_status_step17", "step17_check_additive", "step17_check_structure", "step17_check_np_type", "step17_check_sintering",
        "nanocarbon_keyword_detected_step9", "nanocarbon_type_auto_step9", "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9",
    ]
    step18_cols = step17_cols + ["target_quality_step18", "sigma_fit_log_rmse_step12", "validation_sigma_log_rmse_step13"]
    step19_cols = [
        "sample_key", "target_log_tau_eff_step18", "pred_log_tau_eff_step19", "residual_log_tau_eff_step19",
        "abs_error_log_tau_eff_step19", "tau_eff_ratio_pred_true_step19", "residual_category_step19",
        "manual_review_status_step17", "paper_checked_step17",
    ]
    step15_cols = ["sample_key", "manual_review_priority_score_step15", "manual_review_priority_tier_step15", "pf_zt_problem_reason_step15", "sintering_check_reason_step15"]

    df = samples.copy()
    df = merge_optional(df, read_csv(Path(args.step18_dir) / "tau_eff_ml_dataset_step18.csv"), step18_cols, "step18")
    df = merge_optional(df, read_csv(Path(args.step17_dir) / "step17_annotated_samples.csv"), step17_cols, "step17")
    df = merge_optional(df, read_csv(Path(args.step19_dir) / "tau_eff_ml_residual_analysis_step19.csv"), step19_cols, "step19")
    df = merge_optional(df, read_csv(Path(args.step15_dir) / "pf_zt_error_samples_step15.csv"), step15_cols, "step15")

    for col, cands in {
        "n_or_p_final_step17": ["n_or_p_final_step17", "n_or_p_final_step17__step17", "n_or_p_final_step17__step18", "n_or_p"],
        "additive_final_step17": ["additive_final_step17", "additive_final_step17__step17", "additive_final_step17__step18"],
        "additive_source_step17": ["additive_source_step17", "additive_source_step17__step17", "additive_source_step17__step18"],
        "additive_confidence_final_step17": ["additive_confidence_final_step17", "additive_confidence_final_step17__step17", "additive_confidence_final_step17__step18"],
        "structure_final_step17": ["structure_final_step17", "structure_final_step17__step17", "structure_final_step17__step18"],
        "structure_source_step17": ["structure_source_step17", "structure_source_step17__step17", "structure_source_step17__step18"],
        "structure_confidence_final_step17": ["structure_confidence_final_step17", "structure_confidence_final_step17__step17", "structure_confidence_final_step17__step18"],
        "sintering_method_final_step17": ["sintering_method_final_step17", "sintering_method_final_step17__step17", "sintering_method_final_step17__step18", "sintering_method"],
        "sintering_checked_final_step17": ["sintering_checked_final_step17", "sintering_checked_final_step17__step17", "sintering_checked_final_step17__step18", "sintering_checked"],
        "sintering_source_step17": ["sintering_source_step17", "sintering_source_step17__step17", "sintering_source_step17__step18"],
        "paper_checked_step17": ["paper_checked_step17", "paper_checked_step17__step17", "paper_checked_step17__step18", "paper_checked_step17__step19"],
        "manual_review_status_step17": ["manual_review_status_step17", "manual_review_status_step17__step17", "manual_review_status_step17__step18", "manual_review_status_step17__step19"],
    }.items():
        df = coalesce(df, col, cands)

    for flag in ["nanocarbon_keyword_detected_step9", "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9"]:
        if flag not in df.columns:
            df[flag] = False
        df[flag] = bool_series(df[flag])
    if "nanocarbon_type_auto_step9" not in df.columns:
        df["nanocarbon_type_auto_step9"] = "unknown"

    df["missing_additive_info_step23"] = unknown(df["additive_final_step17"])
    df["missing_structure_info_step23"] = unknown(df["structure_final_step17"])
    df["missing_sintering_info_step23"] = unknown(df["sintering_method_final_step17"])
    df["missing_np_paper_confirmation_step23"] = ~df["n_or_p_source_step17"].astype(str).str.lower().str.contains("paper|manual", na=False) if "n_or_p_source_step17" in df.columns else True
    high_zt = pd.to_numeric(df["zt_obs_max_step22"], errors="coerce") >= args.zt_threshold
    ml_bad = df["comparison_category_step22"].eq("ML_much_worse") | (pd.to_numeric(df["zt_obs_ML_mape_step22"], errors="coerce") > 1)
    df["manual_annotation_needed_step23"] = ml_bad | high_zt | df[["missing_additive_info_step23", "missing_structure_info_step23", "missing_sintering_info_step23", "missing_np_paper_confirmation_step23"]].any(axis=1)

    df["error_pattern_step23"] = df.apply(classify_pattern, axis=1)
    hyp = df.apply(hypotheses, axis=1)
    df["primary_error_source_hypothesis_step23"] = [h[0] for h in hyp]
    df["secondary_error_source_hypothesis_step23"] = [h[1] for h in hyp]
    df["error_cause_note_step23"] = "candidate hypothesis only; no causal mechanism proven"
    df = add_review_priority(df, args.zt_threshold)

    ordered = [
        "sample_key", "DOI", "paper_title", "sample_id", "composition", "material_system", "n_or_p", "n_or_p_final_step17",
        "zt_obs_max_step22", "zt_pred_fitting_max_step22", "zt_pred_ML_max_step22", "zt_calc_from_obs_max_step22",
        "sigma_fitting_log_rmse_step22", "sigma_ML_log_rmse_step22", "sigma_log_rmse_gap_ML_minus_fitting_step22",
        "zt_obs_fitting_mape_step22", "zt_obs_ML_mape_step22", "zt_obs_mape_gap_ML_minus_fitting_step22",
        "comparison_category_step22", "error_pattern_step23", "primary_error_source_hypothesis_step23", "secondary_error_source_hypothesis_step23", "error_cause_note_step23",
        "additive_final_step17", "additive_source_step17", "additive_confidence_final_step17",
        "structure_final_step17", "structure_source_step17", "structure_confidence_final_step17",
        "sintering_method", "sintering_checked", "record_checked", "sintering_method_final_step17", "sintering_checked_final_step17", "sintering_source_step17",
        "nanocarbon_keyword_detected_step9", "nanocarbon_type_auto_step9", "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9",
        "missing_additive_info_step23", "missing_structure_info_step23", "missing_sintering_info_step23", "missing_np_paper_confirmation_step23", "manual_annotation_needed_step23",
        "step23_review_priority_score", "step23_review_priority_tier", "step23_review_reason", "doi_url",
    ]
    error_samples = df[[c for c in ordered if c in df.columns] + [c for c in df.columns if c not in ordered]].copy()

    cause_matrix = df.groupby(["comparison_category_step22", "error_pattern_step23", "primary_error_source_hypothesis_step23"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        median_sigma_fitting_log_rmse=("sigma_fitting_log_rmse_step22", "median"),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_zt_obs_fitting_mape=("zt_obs_fitting_mape_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        missing_additive_count=("missing_additive_info_step23", "sum"),
        missing_structure_count=("missing_structure_info_step23", "sum"),
        missing_sintering_count=("missing_sintering_info_step23", "sum"),
        manual_review_needed_count=("manual_annotation_needed_step23", "sum"),
    ).reset_index()
    cause_matrix["interpretation_step23"] = "candidate cause grouping; not causal proof"

    by_material = df.groupby(["material_system", "n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        ML_much_worse_count=("comparison_category_step22", lambda s: int((s == "ML_much_worse").sum())),
        ML_better_than_fitting_count=("comparison_category_step22", lambda s: int((s == "ML_better_than_fitting").sum())),
        fitting_good_ML_bad_count=("error_pattern_step23", lambda s: int((s == "fitting_good_ML_bad").sum())),
        fitting_bad_ML_bad_count=("error_pattern_step23", lambda s: int((s == "fitting_bad_ML_bad").sum())),
        both_good_count=("error_pattern_step23", lambda s: int((s == "both_good").sum())),
        both_poor_count=("error_pattern_step23", lambda s: int((s == "both_poor").sum())),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_sigma_fitting_log_rmse=("sigma_fitting_log_rmse_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        median_zt_obs_fitting_mape=("zt_obs_fitting_mape_step22", "median"),
        missing_additive_count=("missing_additive_info_step23", "sum"),
        missing_structure_count=("missing_structure_info_step23", "sum"),
        missing_sintering_count=("missing_sintering_info_step23", "sum"),
        nanocarbon_count=("nanocarbon_keyword_detected_step9", "sum"),
        rare_metal_flag_count=("rare_metal_flag_auto_step9", "sum"),
        toxicity_flag_count=("toxicity_flag_auto_step9", "sum"),
        manual_annotation_needed_count=("manual_annotation_needed_step23", "sum"),
        step23_A_or_B_review_count=("step23_review_priority_tier", lambda s: int(s.isin(["A", "B"]).sum())),
    ).reset_index()
    by_material["interpretation_step23"] = np.where(by_material["ML_much_worse_count"] > 0, "ML degradation exists; review annotations and tau_eff errors", "limited degradation")

    by_np = summarize_group(df, ["n_or_p"])
    by_add = summarize_group(df, ["additive_final_step17", "additive_source_step17"])
    by_struct = summarize_group(df, ["structure_final_step17", "structure_source_step17"])
    by_sint = df.groupby(["sintering_method_final_step17", "sintering_checked_final_step17", "sintering_source_step17"], dropna=False).agg(
        sample_count=("sample_key", "nunique"),
        ML_much_worse_count=("comparison_category_step22", lambda s: int((s == "ML_much_worse").sum())),
        fitting_bad_ML_bad_count=("error_pattern_step23", lambda s: int((s == "fitting_bad_ML_bad").sum())),
        high_ZT_count=("zt_obs_max_step22", lambda s: int((pd.to_numeric(s, errors="coerce") >= args.zt_threshold).sum())),
        median_sigma_ML_log_rmse=("sigma_ML_log_rmse_step22", "median"),
        median_zt_obs_ML_mape=("zt_obs_ML_mape_step22", "median"),
        manual_annotation_needed_count=("manual_annotation_needed_step23", "sum"),
    ).reset_index()
    by_sint["interpretation_step23"] = np.where(unknown(by_sint["sintering_method_final_step17"]), "sintering unknown means review target, not confirmed cause", "sintering info available")

    flag_rows = []
    for flag in ["nanocarbon_keyword_detected_step9", "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9", "missing_additive_info_step23", "missing_structure_info_step23", "missing_sintering_info_step23", "manual_annotation_needed_step23"]:
        for val, g in df.groupby(flag, dropna=False):
            flag_rows.append({
                "feature_name": flag, "feature_value": val, "sample_count": g["sample_key"].nunique(),
                "ML_much_worse_count": int((g["comparison_category_step22"] == "ML_much_worse").sum()),
                "median_sigma_ML_log_rmse": g["sigma_ML_log_rmse_step22"].median(),
                "median_zt_obs_ML_mape": g["zt_obs_ML_mape_step22"].median(),
                "manual_annotation_needed_count": int(g["manual_annotation_needed_step23"].sum()),
                "interpretation_step23": "flag association only; no causal proof",
            })
    by_flags = pd.DataFrame(flag_rows)

    missing_rows = []
    for info, col in [
        ("additive", "missing_additive_info_step23"),
        ("structure", "missing_structure_info_step23"),
        ("sintering", "missing_sintering_info_step23"),
        ("n_or_p_paper_confirmation", "missing_np_paper_confirmation_step23"),
        ("manual_paper_review", "manual_annotation_needed_step23"),
    ]:
        miss = df[col].astype(bool)
        missing_rows.append({
            "information_type": info,
            "missing_count": int(miss.sum()),
            "available_count": int((~miss).sum()),
            "missing_fraction": float(miss.mean()),
            "high_error_missing_count": int((miss & df["comparison_category_step22"].eq("ML_much_worse")).sum()),
            "high_ZT_missing_count": int((miss & (pd.to_numeric(df["zt_obs_max_step22"], errors="coerce") >= args.zt_threshold)).sum()),
            "manual_review_priority": "high" if miss.mean() > 0.5 else "medium",
            "note": "missing information only; not confirmed cause",
        })
    missing_summary = pd.DataFrame(missing_rows)

    manual = df[df["manual_annotation_needed_step23"] | df["step23_review_priority_tier"].isin(["A", "B"])].copy()
    manual["manual_review_note_step23"] = np.where(df["error_pattern_step23"].eq("fitting_bad_ML_bad"), "check ZT curve and units; fitting and ML both poor", "check additive and structure; ML tau error may be large")
    manual = manual.sort_values("step23_review_priority_score", ascending=False).head(args.top_n_manual_review)

    sintering = df[df["missing_sintering_info_step23"] & ((pd.to_numeric(df["zt_obs_max_step22"], errors="coerce") >= args.zt_threshold) | df["comparison_category_step22"].eq("ML_much_worse") | df["step23_review_priority_tier"].isin(["A", "B"]))].copy()
    sintering["sintering_check_reason_step23"] = "check sintering method for high-ZT or large-error sample; do not infer cause from unknown"
    sintering = sintering.sort_values("step23_review_priority_score", ascending=False).head(args.top_n_sintering_check)

    highzt = df[(pd.to_numeric(df["zt_obs_max_step22"], errors="coerce") >= args.zt_threshold) | (pd.to_numeric(df["zt_pred_ML_max_step22"], errors="coerce") >= args.zt_threshold) | (pd.to_numeric(df["zt_pred_fitting_max_step22"], errors="coerce") >= args.zt_threshold)].copy()

    mat_notes = by_material.copy()
    mat_notes["main_error_pattern"] = "ML degradation / missing annotation review"
    mat_notes["main_error_hypothesis"] = "insufficient_material_features"
    mat_notes["manual_review_need"] = np.where(mat_notes["manual_annotation_needed_count"] > 0, "yes", "no")
    mat_notes["sintering_check_need"] = np.where(mat_notes["missing_sintering_count"] > 0, "yes", "no")
    mat_notes["note_for_paper_step23"] = "This material system may have ML-worse samples; additive/structure/sintering gaps should be reviewed before interpretation."
    mat_notes = mat_notes[["material_system", "sample_count", "main_error_pattern", "main_error_hypothesis", "manual_review_need", "sintering_check_need", "note_for_paper_step23"]]

    summary_items = [
        ("total samples analyzed", len(df), "Step23 sample universe", "step23_error_cause_samples.csv"),
        ("ML much worse sample count", int((df["comparison_category_step22"] == "ML_much_worse").sum()), "ML degradation count", "step23_error_cause_samples.csv"),
        ("ML better than fitting sample count", int((df["comparison_category_step22"] == "ML_better_than_fitting").sum()), "ML better count", "step23_error_cause_samples.csv"),
        ("fitting_good_ML_bad count", int((df["error_pattern_step23"] == "fitting_good_ML_bad").sum()), "likely ML tau/feature limitation candidate", "step23_ml_vs_fitting_cause_matrix.csv"),
        ("fitting_bad_ML_bad count", int((df["error_pattern_step23"] == "fitting_bad_ML_bad").sum()), "possible data/prefactor/physics issue candidate", "step23_ml_vs_fitting_cause_matrix.csv"),
        ("missing additive info count", int(df["missing_additive_info_step23"].sum()), "annotation gap", "step23_missing_information_summary.csv"),
        ("missing structure info count", int(df["missing_structure_info_step23"].sum()), "annotation gap", "step23_missing_information_summary.csv"),
        ("missing sintering info count", int(df["missing_sintering_info_step23"].sum()), "review target only, not causal proof", "step23_missing_information_summary.csv"),
        ("manual review priority sample count", len(manual), "manual review queue", "step23_manual_review_priority_samples.csv"),
        ("sintering check priority sample count", len(sintering), "sintering review queue", "step23_sintering_check_priority_samples.csv"),
        ("high ZT error case count", len(highzt), "high-ZT review set", "step23_high_zt_error_cases.csv"),
        ("most common primary error hypothesis", df["primary_error_source_hypothesis_step23"].value_counts().idxmax(), "dominant candidate hypothesis", "step23_error_cause_samples.csv"),
        ("largest error material system", by_material.sort_values("ML_much_worse_count", ascending=False).iloc[0]["material_system"], "largest ML much worse count", "step23_error_by_material_system.csv"),
    ]
    summary = pd.DataFrame([{"summary_item": a, "value": v, "interpretation_step23": i, "related_file": f} for a, v, i, f in summary_items])

    np_changed = 0
    sintering_changed = 0
    report = make_report(df, manual, sintering, highzt, by_material, by_np, np_changed, sintering_changed)

    outputs = {
        "step23_error_cause_samples.csv": error_samples,
        "step23_error_cause_summary.csv": summary,
        "step23_error_by_material_system.csv": by_material,
        "step23_error_by_np_type.csv": by_np,
        "step23_error_by_additive.csv": by_add,
        "step23_error_by_structure.csv": by_struct,
        "step23_error_by_sintering_status.csv": by_sint,
        "step23_error_by_feature_flags.csv": by_flags,
        "step23_ml_vs_fitting_cause_matrix.csv": cause_matrix,
        "step23_missing_information_summary.csv": missing_summary,
        "step23_manual_review_priority_samples.csv": manual,
        "step23_sintering_check_priority_samples.csv": sintering,
        "step23_high_zt_error_cases.csv": highzt,
        "step23_material_system_notes.csv": mat_notes,
    }
    for name, data in outputs.items():
        data.to_csv(outdir / name, index=False)
    (outdir / "step23_error_cause_report.txt").write_text(report, encoding="utf-8")
    (outdir / "step23_error_cause_notes.md").write_text(notes(), encoding="utf-8")

    write_excel(outdir / "starrydata2_step23_error_cause_analysis.xlsx", {
        "error_cause_samples": error_samples,
        "error_cause_summary": summary,
        "by_material_system": by_material,
        "by_np_type": by_np,
        "by_additive": by_add,
        "by_structure": by_struct,
        "by_sintering_status": by_sint,
        "by_feature_flags": by_flags,
        "cause_matrix": cause_matrix,
        "missing_information": missing_summary,
        "manual_review_priority": manual,
        "sintering_check_priority": sintering,
        "high_zt_error_cases": highzt,
        "material_system_notes": mat_notes,
        "error_cause_report": report,
    })

    top_material = by_material.sort_values("ML_much_worse_count", ascending=False).iloc[0]["material_system"]
    print("Done.")
    print("Created:")
    for name in list(outputs.keys()) + ["step23_error_cause_report.txt", "step23_error_cause_notes.md", "starrydata2_step23_error_cause_analysis.xlsx"]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"samples analyzed: {len(df)}")
    print(f"fitting_good_ML_bad: {int((df['error_pattern_step23']=='fitting_good_ML_bad').sum())}")
    print(f"fitting_bad_ML_bad: {int((df['error_pattern_step23']=='fitting_bad_ML_bad').sum())}")
    print(f"ML much worse samples: {int((df['comparison_category_step22']=='ML_much_worse').sum())}")
    print(f"missing additive info samples: {int(df['missing_additive_info_step23'].sum())}")
    print(f"missing structure info samples: {int(df['missing_structure_info_step23'].sum())}")
    print(f"missing sintering info samples: {int(df['missing_sintering_info_step23'].sum())}")
    print(f"manual review priority samples: {len(manual)}")
    print(f"sintering check priority samples: {len(sintering)}")
    print(f"high ZT error cases: {len(highzt)}")
    print(f"top material system with ML degradation: {top_material}")
    print(f"n/p changed rows: {np_changed}")
    print(f"existing sintering columns changed rows: {sintering_changed}")


def make_report(df, manual, sintering, highzt, by_material, by_np, np_changed, sintering_changed):
    pattern_counts = df["error_pattern_step23"].value_counts().to_dict()
    hyp_counts = df["primary_error_source_hypothesis_step23"].value_counts().to_dict()
    lines = [
        "Step23 error cause analysis report",
        "",
        f"Input step22_sample_level_comparison rows: {len(df)}",
        f"Output step23_error_cause_samples rows: {len(df)}",
        "",
        "error patterns:",
    ]
    for key in ["fitting_good_ML_bad", "fitting_bad_ML_bad", "fitting_bad_ML_good", "both_good", "both_poor", "not_evaluable"]:
        lines.append(f"- {key}: {pattern_counts.get(key, 0)}")
    lines.append("")
    lines.append("primary error hypotheses:")
    for key in ["tau_eff_ML_prediction_error", "insufficient_material_features", "missing_additive_or_structure_annotation", "material_system_out_of_distribution", "n_or_p_specific_error", "possible_sintering_or_microstructure_effect", "possible_ZT_observation_or_unit_inconsistency", "direct_fitting_or_prefactor_limitation"]:
        lines.append(f"- {key}: {hyp_counts.get(key, 0)}")
    lines.extend([
        "",
        "missing information:",
        f"- additive missing: {int(df['missing_additive_info_step23'].sum())}",
        f"- structure missing: {int(df['missing_structure_info_step23'].sum())}",
        f"- sintering missing: {int(df['missing_sintering_info_step23'].sum())}",
        f"- n/p paper confirmation missing: {int(df['missing_np_paper_confirmation_step23'].sum())}",
        "",
        "review:",
        f"- manual review priority samples: {len(manual)}",
        f"- sintering check priority samples: {len(sintering)}",
        f"- high ZT error cases: {len(highzt)}",
        "",
        "material:",
    ])
    for _, row in by_material.sort_values("ML_much_worse_count", ascending=False).head(20).iterrows():
        lines.append(f"- {row['material_system']} / {row['n_or_p']}: ML much worse={row['ML_much_worse_count']}, missing structure={row['missing_structure_count']}, missing sintering={row['missing_sintering_count']}")
    lines.append("")
    lines.append("n/p:")
    for _, row in by_np.iterrows():
        lines.append(f"- {row['n_or_p']}: ML much worse={row['ML_much_worse_count']}, median sigma_ML_log_rmse={row['median_sigma_ML_log_rmse']}")
    lines.extend([
        "",
        "sintering:",
        "- existing sintering_method changed rows: 0",
        "- existing sintering_checked changed rows: 0",
        "- existing record_checked changed rows: 0",
        f"- sintering_method_final_step17 counts: {df['sintering_method_final_step17'].value_counts(dropna=False).to_dict()}",
        "",
        f"n/p changed rows: {np_changed}",
        f"sintering changed rows: {sintering_changed}",
        "",
        "Notes:",
        "Step23 did not make new predictions, refit tau_eff, or retrain ML models.",
        "Step23 organized candidate error causes but did not prove causality.",
        "Unknown sintering indicates missing confirmation, not a confirmed cause.",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
