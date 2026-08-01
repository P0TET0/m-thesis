import argparse
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP23_DIR = "data/output/starrydata2_step23_error_cause_analysis"
DEFAULT_STEP21_DIR = "data/output/starrydata2_step21_pf_zt_ml_prediction"
DEFAULT_STEP22_DIR = "data/output/starrydata2_step22_fitting_vs_ml_comparison"
DEFAULT_STEP17_DIR = "data/output/starrydata2_step17_literature_review"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step24_material_candidates"
EXCEL_PREVIEW_ROWS = 100_000

STRING_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "composition",
    "material_system",
    "n_or_p",
]

PERIODIC_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}
RARE_METAL_ELEMENTS = {"Te", "Se", "Bi", "Sb", "Ge", "In", "Ga", "Hf", "Ta", "W", "Re", "Pt", "Pd", "Rh", "Ru", "Ir", "Os", "Ag", "Au", "Y", "La", "Ce", "Nd", "Sm", "Gd", "Dy", "Yb"}
TOXICITY_ELEMENTS = {"Pb", "Cd", "Hg", "Tl", "As", "Se", "Te", "Sb"}
URL_COLUMNS = {"doi_url", "url", "source_url"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Step24 material candidates from Step21-23 results.")
    parser.add_argument("--step23_dir", default=DEFAULT_STEP23_DIR)
    parser.add_argument("--step21_dir", default=DEFAULT_STEP21_DIR)
    parser.add_argument("--step22_dir", default=DEFAULT_STEP22_DIR)
    parser.add_argument("--step17_dir", default=DEFAULT_STEP17_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--kappa_threshold", type=float, default=2.0)
    parser.add_argument("--sigma_threshold", type=float, default=10000.0)
    parser.add_argument("--top_n_candidates", type=int, default=300)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {c: "string" for c in STRING_COLUMNS if c in header.columns}


def read_csv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")
        return None
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def require_columns(df, cols, name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def duplicate_count(df):
    if df is None or "sample_key" not in df.columns:
        return 0
    return int(df["sample_key"].duplicated().sum())


def first_by_key(df):
    if df is None or "sample_key" not in df.columns:
        return None
    return df.drop_duplicates("sample_key", keep="first").copy()


def merge_by_sample_key(base, other, label, report_lines, membership_col=None):
    if other is None:
        report_lines.append(f"{label}: not found")
        if membership_col:
            base[membership_col] = False
        return base
    dups = duplicate_count(other)
    report_lines.append(f"{label}: rows={len(other)}, duplicate sample_key rows={dups}; first row kept for duplicate keys")
    other = first_by_key(other)
    if membership_col:
        other[membership_col] = True
    rename = {c: f"{c}__{label}" for c in other.columns if c != "sample_key" and c in base.columns}
    merged = base.merge(other.rename(columns=rename), on="sample_key", how="left")
    if membership_col and membership_col not in merged.columns:
        merged[membership_col] = False
    if membership_col:
        merged[membership_col] = merged[membership_col].map(lambda x: bool(x) if pd.notna(x) else False)
    return merged


def to_num(df, col):
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def first_existing_numeric(df, cols):
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), values)
    return out


def str_col(df, col, default=""):
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].astype("string").fillna(default).astype(str)


def is_true(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def is_unknown(series):
    return series.isna() | series.astype(str).str.strip().str.lower().isin(["", "unknown", "nan", "<na>", "not_checked", "not checked", "none"])


def coalesce_columns(df, target, candidates):
    df[target] = first_existing_numeric(df, candidates)


def aggregate_rows(rows):
    if rows is None or rows.empty or "sample_key" not in rows.columns:
        return None
    g = rows.groupby("sample_key", dropna=False)
    agg = pd.DataFrame({"sample_key": g.size().index})
    agg["sigma_obs_max_step24"] = g["sigma_obs_S_per_m_step11"].max().values if "sigma_obs_S_per_m_step11" in rows.columns else np.nan
    agg["sigma_obs_median_step24"] = g["sigma_obs_S_per_m_step11"].median().values if "sigma_obs_S_per_m_step11" in rows.columns else np.nan
    pred_sigma_col = "sigma_pred_ML_for_pf_zt_S_per_m_step21"
    if pred_sigma_col not in rows.columns:
        pred_sigma_col = "sigma_pred_ML_all_samples_S_per_m_step20"
    agg["sigma_pred_ML_max_step24"] = g[pred_sigma_col].max().values if pred_sigma_col in rows.columns else np.nan
    agg["sigma_pred_ML_median_step24"] = g[pred_sigma_col].median().values if pred_sigma_col in rows.columns else np.nan
    agg["kappa_obs_min_step24"] = g["kappa_obs_W_per_mK_step11"].min().values if "kappa_obs_W_per_mK_step11" in rows.columns else np.nan
    agg["kappa_obs_median_step24"] = g["kappa_obs_W_per_mK_step11"].median().values if "kappa_obs_W_per_mK_step11" in rows.columns else np.nan
    if "seebeck_obs_V_per_K_step11" in rows.columns:
        rows = rows.copy()
        rows["seebeck_abs_step24_tmp"] = pd.to_numeric(rows["seebeck_obs_V_per_K_step11"], errors="coerce").abs()
        g = rows.groupby("sample_key", dropna=False)
        agg["seebeck_abs_max_step24"] = g["seebeck_abs_step24_tmp"].max().values
        agg["seebeck_abs_median_step24"] = g["seebeck_abs_step24_tmp"].median().values
    else:
        agg["seebeck_abs_max_step24"] = np.nan
        agg["seebeck_abs_median_step24"] = np.nan
    agg["zt_obs_max_from_rows_step24"] = g["zt_obs_dimensionless_step11"].max().values if "zt_obs_dimensionless_step11" in rows.columns else np.nan
    agg["zt_pred_ML_max_from_rows_step24"] = g["zt_pred_ML_step21"].max().values if "zt_pred_ML_step21" in rows.columns else np.nan
    agg["zt_calc_from_obs_max_from_rows_step24"] = g["zt_calc_from_obs_step11"].max().values if "zt_calc_from_obs_step11" in rows.columns else np.nan
    agg["temperature_min_step24"] = g["temperature_K"].min().values if "temperature_K" in rows.columns else np.nan
    agg["temperature_max_step24"] = g["temperature_K"].max().values if "temperature_K" in rows.columns else np.nan
    agg["temperature_span_step24"] = agg["temperature_max_step24"] - agg["temperature_min_step24"]
    agg["n_temperature_rows_step24"] = g.size().values
    return agg


def detect_elements(text):
    if pd.isna(text):
        return []
    tokens = re.findall(r"[A-Z][a-z]?", str(text))
    return sorted({t for t in tokens if t in PERIODIC_ELEMENTS})


def add_element_flags(df):
    elems = str_col(df, "composition").map(detect_elements)
    df["elements_detected_step24"] = elems.map(lambda xs: ";".join(xs))
    rare = elems.map(lambda xs: sorted(set(xs) & RARE_METAL_ELEMENTS))
    tox = elems.map(lambda xs: sorted(set(xs) & TOXICITY_ELEMENTS))
    df["rare_metal_elements_step24"] = rare.map(lambda xs: ";".join(xs))
    df["rare_metal_flag_step24"] = rare.map(bool)
    df["toxicity_attention_elements_step24"] = tox.map(lambda xs: ";".join(xs))
    df["toxicity_attention_flag_step24"] = tox.map(bool)
    df["contains_carbon_element_step24"] = elems.map(lambda xs: "C" in xs)
    return df


def add_candidate_flags(df, args):
    coalesce_columns(df, "zt_obs_max_step24", ["zt_obs_max_step24", "zt_obs_max_step22", "zt_obs_max_step21", "zt_obs_max_from_rows_step24", "zt_obs_max_step14"])
    coalesce_columns(df, "zt_pred_ML_max_step24", ["zt_pred_ML_max_step24", "zt_pred_ML_max_step22", "zt_pred_ML_max_step21", "zt_pred_ML_max_from_rows_step24"])
    coalesce_columns(df, "zt_pred_fitting_max_step24", ["zt_pred_fitting_max_step24", "zt_pred_fitting_max_step22", "zt_pred_fitting_max_step21", "zt_pred_max_step14"])
    coalesce_columns(df, "zt_calc_from_obs_max_step24", ["zt_calc_from_obs_max_step24", "zt_calc_from_obs_max_step22", "zt_calc_from_obs_max_step21", "zt_calc_from_obs_max_from_rows_step24", "zt_calc_from_obs_max_step14"])

    df["is_high_zt_observed_step24"] = to_num(df, "zt_obs_max_step24") >= args.zt_threshold
    df["is_high_zt_ml_predicted_step24"] = to_num(df, "zt_pred_ML_max_step24") >= args.zt_threshold
    df["is_high_zt_fitting_predicted_step24"] = to_num(df, "zt_pred_fitting_max_step24") >= args.zt_threshold
    df["is_low_kappa_step24"] = to_num(df, "kappa_obs_min_step24") <= args.kappa_threshold
    df["is_high_sigma_obs_step24"] = to_num(df, "sigma_obs_max_step24") >= args.sigma_threshold
    df["is_high_sigma_ml_step24"] = to_num(df, "sigma_pred_ML_max_step24") >= args.sigma_threshold
    df["is_low_kappa_high_sigma_step24"] = df["is_low_kappa_step24"] & (df["is_high_sigma_obs_step24"] | df["is_high_sigma_ml_step24"])
    df["is_low_rare_metal_attention_step24"] = ~df["rare_metal_flag_step24"].astype(bool)
    df["is_low_toxicity_attention_step24"] = ~df["toxicity_attention_flag_step24"].astype(bool)

    nano_kw = is_true(str_col(df, "nanocarbon_keyword_detected_step9"))
    nano_type = ~is_unknown(str_col(df, "nanocarbon_type_auto_step9"))
    text = (str_col(df, "material_system") + " " + str_col(df, "composition")).str.lower()
    carbon_words = text.str.contains(r"\b(?:carbon|cnt|graphene|graphite|nanotube|c60)\b", regex=True, na=False)
    df["is_nanocarbon_candidate_step24"] = nano_kw | nano_type | (df["contains_carbon_element_step24"].astype(bool) & carbon_words)

    df["needs_manual_review_step24"] = (
        df.get("in_manual_review_priority_step24", False).astype(bool)
        if "in_manual_review_priority_step24" in df.columns else False
    ) | is_true(str_col(df, "manual_annotation_needed_step23")) | is_true(str_col(df, "needs_manual_review_step15"))
    df["needs_sintering_check_step24"] = (
        df.get("in_sintering_check_priority_step24", False).astype(bool)
        if "in_sintering_check_priority_step24" in df.columns else False
    ) | is_true(str_col(df, "needs_sintering_check_later_step15")) | is_true(str_col(df, "needs_sintering_check_later_step14"))

    df["is_ml_supported_candidate_step24"] = df["is_high_zt_ml_predicted_step24"] | df["is_high_sigma_ml_step24"]
    df["is_fitting_supported_candidate_step24"] = df["is_high_zt_fitting_predicted_step24"] | df["is_high_zt_observed_step24"]
    df["is_balanced_recommended_candidate_step24"] = (
        (df["is_high_zt_observed_step24"] | df["is_high_zt_ml_predicted_step24"] | df["is_low_kappa_high_sigma_step24"])
        & (df["is_low_toxicity_attention_step24"] | df["needs_manual_review_step24"])
    )
    return df


def add_scores(df):
    components = []
    scores = []
    reasons = []
    cautions = []
    for _, row in df.iterrows():
        score = 0
        rs = []
        cs = ["downstream ML prediction is not unbiased evaluation"]

        def add(name, value, reason):
            nonlocal score
            score += value
            components.append({"sample_key": row["sample_key"], "score_component": name, "score_value": value, "score_reason": reason})
            if value > 0:
                rs.append(reason)

        if row.get("is_high_zt_observed_step24", False):
            add("observed_high_ZT", 40, "observed ZT >= threshold")
        if row.get("is_high_zt_ml_predicted_step24", False):
            add("ML_predicted_high_ZT", 25, "ML predicted ZT >= threshold")
        if row.get("is_high_zt_fitting_predicted_step24", False):
            add("fitting_predicted_high_ZT", 25, "fitting predicted ZT >= threshold")
        if row.get("is_low_kappa_step24", False):
            add("low_kappa", 20, "low thermal conductivity")
        if row.get("is_high_sigma_obs_step24", False):
            add("high_sigma_obs", 20, "high observed electrical conductivity")
        if row.get("is_high_sigma_ml_step24", False):
            add("high_sigma_ML", 10, "high ML predicted electrical conductivity")
        if row.get("is_low_kappa_high_sigma_step24", False):
            add("low_kappa_high_sigma", 30, "low kappa and high sigma")
        if row.get("is_nanocarbon_candidate_step24", False):
            add("nanocarbon", 25, "nanocarbon-related candidate")
        if row.get("is_low_rare_metal_attention_step24", False):
            add("low_rare_metal_attention", 15, "rare metal attention elements not detected")
        if row.get("is_low_toxicity_attention_step24", False):
            add("low_toxicity_attention", 15, "toxicity attention elements not detected")
        if str(row.get("paper_checked_step17", "")).strip().lower() in ["yes", "true", "checked", "1"]:
            add("manual_annotation_checked", 10, "manual annotation checked")
        if row.get("is_balanced_recommended_candidate_step24", False):
            add("balanced_recommended", 20, "balanced recommended candidate")

        if row.get("toxicity_attention_flag_step24", False):
            add("toxicity_attention", -20, "toxicity attention element detected")
            cs.append("toxicity attention element detected")
        if row.get("rare_metal_flag_step24", False):
            add("rare_metal_attention", -10, "rare metal attention element detected")
            cs.append("rare metal attention element detected")
        if is_unknown(pd.Series([row.get("structure_final_step17", np.nan)])).iloc[0]:
            add("structure_unknown", -10, "structure unknown")
            cs.append("structure unknown")
        if is_unknown(pd.Series([row.get("additive_final_step17", np.nan)])).iloc[0]:
            add("additive_unknown", -10, "additive unknown")
            cs.append("additive unknown")
        high_or_error = row.get("is_high_zt_observed_step24", False) or row.get("is_high_zt_ml_predicted_step24", False) or bool(row.get("comparison_category_step22", ""))
        if high_or_error and (is_unknown(pd.Series([row.get("sintering_method_final_step17", row.get("sintering_method", np.nan))])).iloc[0] or row.get("missing_sintering_info_step23", False) is True):
            add("sintering_unknown_high_error", -10, "sintering unknown for high-ZT/error sample")
            cs.append("sintering unknown")
        if pd.notna(row.get("sigma_log_rmse_gap_ML_minus_fitting_step22", np.nan)) and float(row.get("sigma_log_rmse_gap_ML_minus_fitting_step22", 0)) > 0.5:
            add("ML_worse_than_fitting", -15, "ML much worse than fitting")
            cs.append("ML much worse than fitting")
        if pd.notna(row.get("zt_obs_ML_mape_step22", np.nan)) and float(row.get("zt_obs_ML_mape_step22", 0)) > 100:
            add("large_ZT_error", -15, "large ZT error")
            cs.append("large ZT error")

        scores.append(score)
        reasons.append("; ".join(dict.fromkeys(rs)))
        cautions.append("; ".join(dict.fromkeys(cs)))

    df["candidate_score_step24"] = scores
    df["candidate_tier_step24"] = pd.cut(
        df["candidate_score_step24"],
        bins=[-math.inf, -0.000001, 24.999999, 49.999999, 79.999999, math.inf],
        labels=["low", "review", "C", "B", "A"],
    ).astype(str)
    df["candidate_reason_step24"] = reasons
    df["candidate_caution_step24"] = cautions
    return df, pd.DataFrame(components)


def ordered_columns(df):
    front = [
        "sample_key", "candidate_score_step24", "candidate_tier_step24", "candidate_reason_step24", "candidate_caution_step24",
        "DOI", "paper_title", "sample_id", "composition", "material_system", "n_or_p", "n_or_p_final_step17",
        "n_or_p_basis", "n_or_p_step6", "n_or_p_basis_step6", "n_or_p_confidence_step6",
        "zt_obs_max_step24", "zt_pred_ML_max_step24", "zt_pred_fitting_max_step24", "zt_calc_from_obs_max_step24",
        "zt_obs_max_step21", "zt_pred_ML_max_step21", "zt_pred_fitting_max_step21", "zt_calc_from_obs_max_step21",
        "zt_obs_max_step22", "zt_pred_ML_max_step22", "zt_pred_fitting_max_step22", "zt_calc_from_obs_max_step22",
        "sigma_obs_max_step24", "sigma_obs_median_step24", "sigma_pred_ML_max_step24", "sigma_pred_ML_median_step24",
        "kappa_obs_min_step24", "kappa_obs_median_step24", "seebeck_abs_max_step24", "seebeck_abs_median_step24",
        "zt_obs_max_from_rows_step24", "zt_pred_ML_max_from_rows_step24", "zt_calc_from_obs_max_from_rows_step24",
        "temperature_min_step24", "temperature_max_step24", "temperature_span_step24", "n_temperature_rows_step24",
        "is_high_zt_observed_step24", "is_high_zt_ml_predicted_step24", "is_high_zt_fitting_predicted_step24",
        "is_low_kappa_step24", "is_high_sigma_obs_step24", "is_high_sigma_ml_step24", "is_low_kappa_high_sigma_step24",
        "elements_detected_step24", "contains_carbon_element_step24",
        "rare_metal_elements_step24", "rare_metal_flag_step24", "toxicity_attention_elements_step24", "toxicity_attention_flag_step24",
        "is_low_rare_metal_attention_step24", "is_low_toxicity_attention_step24", "is_nanocarbon_candidate_step24",
        "is_ml_supported_candidate_step24", "is_fitting_supported_candidate_step24", "is_balanced_recommended_candidate_step24",
        "nanocarbon_keyword_detected_step9", "nanocarbon_type_auto_step9",
        "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9",
        "additive_final_step17", "additive_source_step17", "additive_confidence_final_step17",
        "structure_final_step17", "structure_source_step17", "structure_confidence_final_step17",
        "sintering_method_final_step17", "sintering_condition_final_step17", "sintering_checked_final_step17",
        "sintering_source_step17", "sintering_confidence_final_step17", "paper_checked_step17", "manual_review_status_step17",
        "comparison_category_step22", "primary_error_source_hypothesis_step23",
        "secondary_error_source_hypothesis_step23", "error_pattern_step23", "error_cause_note_step23",
        "sigma_fitting_log_rmse_step22", "sigma_ML_log_rmse_step22", "sigma_log_rmse_gap_ML_minus_fitting_step22",
        "pf_fitting_mape_step22", "pf_ML_mape_step22", "pf_mape_gap_ML_minus_fitting_step22",
        "zt_obs_fitting_mape_step22", "zt_obs_ML_mape_step22", "zt_obs_mape_gap_ML_minus_fitting_step22",
        "missing_additive_info_step23", "missing_structure_info_step23", "missing_sintering_info_step23",
        "manual_annotation_needed_step23", "step23_review_priority_score", "step23_review_priority_tier", "step23_review_reason",
        "in_manual_review_priority_step24", "in_sintering_check_priority_step24", "in_high_zt_error_step24",
        "needs_manual_review_step24", "needs_sintering_check_step24", "sintering_method", "sintering_checked", "record_checked",
        "evaluation_scope_step21", "downstream_evaluation_note_step21",
    ]
    urls = [c for c in df.columns if c in URL_COLUMNS or c.lower().endswith("_url")]
    cols = [c for c in front if c in df.columns]
    return df[cols + [c for c in urls if c not in cols]]


def sort_desc_safe(df, cols):
    present = [c for c in cols if c in df.columns]
    return df.sort_values(present, ascending=[False] * len(present), na_position="last") if present else df


def sort_mixed_safe(df, cols, ascending):
    present = [c for c in cols if c in df.columns]
    asc = [ascending[cols.index(c)] for c in present]
    return df.sort_values(present, ascending=asc, na_position="last") if present else df


def make_summaries(pool):
    def agg_bool(x):
        return int(x.fillna(False).astype(bool).sum())

    grouped = pool.groupby(["material_system", "n_or_p"], dropna=False).agg(
        sample_count=("sample_key", "count"),
        candidate_A_count=("candidate_tier_step24", lambda x: int((x == "A").sum())),
        candidate_B_count=("candidate_tier_step24", lambda x: int((x == "B").sum())),
        high_zt_count=("is_high_zt_observed_step24", agg_bool),
        low_kappa_high_sigma_count=("is_low_kappa_high_sigma_step24", agg_bool),
        low_rare_metal_attention_count=("is_low_rare_metal_attention_step24", agg_bool),
        low_toxicity_attention_count=("is_low_toxicity_attention_step24", agg_bool),
        nanocarbon_count=("is_nanocarbon_candidate_step24", agg_bool),
        manual_review_needed_count=("needs_manual_review_step24", agg_bool),
        sintering_check_needed_count=("needs_sintering_check_step24", agg_bool),
        median_candidate_score=("candidate_score_step24", "median"),
        max_zt_obs=("zt_obs_max_step24", "max"),
        max_zt_pred_ML=("zt_pred_ML_max_step24", "max"),
        min_kappa_obs=("kappa_obs_min_step24", "min"),
        max_sigma_obs=("sigma_obs_max_step24", "max"),
    ).reset_index()
    grouped["interpretation_step24"] = np.where(
        grouped["candidate_A_count"] + grouped["candidate_B_count"] > 0,
        "contains high-priority Step24 candidates",
        "screening pool material system",
    )
    grouped = grouped.sort_values(["candidate_A_count", "candidate_B_count", "sample_count"], ascending=[False, False, False])

    by_np = pool.groupby("n_or_p", dropna=False).agg(
        sample_count=("sample_key", "count"),
        candidate_A_count=("candidate_tier_step24", lambda x: int((x == "A").sum())),
        candidate_B_count=("candidate_tier_step24", lambda x: int((x == "B").sum())),
        high_zt_count=("is_high_zt_observed_step24", agg_bool),
        low_kappa_high_sigma_count=("is_low_kappa_high_sigma_step24", agg_bool),
        low_rare_metal_attention_count=("is_low_rare_metal_attention_step24", agg_bool),
        low_toxicity_attention_count=("is_low_toxicity_attention_step24", agg_bool),
        nanocarbon_count=("is_nanocarbon_candidate_step24", agg_bool),
        manual_review_needed_count=("needs_manual_review_step24", agg_bool),
        sintering_check_needed_count=("needs_sintering_check_step24", agg_bool),
        median_candidate_score=("candidate_score_step24", "median"),
    ).reset_index()
    return grouped, by_np


def notes_text():
    return """# Step24 Material Candidate Selection Notes

## Purpose
Step24 selects material candidates from existing Step21-23 fitting, ML, comparison, and error-analysis outputs.

## Candidate Criteria
Candidates are selected using observed/high predicted ZT, low thermal conductivity, high electrical conductivity, composition-based attention flags, nanocarbon keywords, and review priorities.

## High ZT Candidates
High ZT candidates satisfy observed, ML-predicted, or fitting-predicted ZT thresholds.

## Low Thermal Conductivity and High Electrical Conductivity Candidates
These candidates satisfy the configured kappa threshold and either observed or ML-predicted sigma threshold.

## Rare Metal Attention
Low rare metal attention means no configured rare-metal attention elements were detected from composition.

## Toxicity Attention
Low toxicity attention means no configured toxicity attention elements were detected from composition.

## Nanocarbon Candidates
Nanocarbon candidates are detected from Step9 annotations, nanocarbon type labels, or carbon-related keywords.

## Balanced Recommended Candidates
Balanced recommended candidates combine performance flags with lower toxicity attention or manual-review priority.

## Manual Review and Sintering Check
Manual review and sintering check outputs prioritize high-scoring candidates and samples with missing additive, structure, or sintering information.

## Important Caveats
Rare-metal-free and low-toxicity labels are provisional screening flags based on composition.
They are not final material safety or resource classifications.
Nanocarbon identification is based on available keywords and may miss cases without explicit annotations.
Many additive, structure, and sintering fields are still unknown.
Downstream ML predictions are for screening, not unbiased evaluation.
Step24 does not perform new prediction or model training.

## Next Step
Step25 should organize Step12-24 results into thesis-ready fitting, ML, comparison, error-analysis, and candidate-material tables.
"""


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"candidate_report": data.splitlines()})
            data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(len(v) for v in values) + 2, 60)


def build_report(args, input_counts, output_counts, tier_counts, flag_counts, material_summary, np_summary, duplicate_report, np_changed, sintering_changed, pool):
    lines = [
        "Step24 Material Candidate Selection Report",
        "",
        f"input downstream sample results rows: {input_counts.get('downstream_samples', 0)}",
        f"input downstream row predictions rows: {input_counts.get('downstream_rows', 0)}",
        f"input error cause samples rows: {input_counts.get('error_cause_samples', 0)}",
        f"output candidate_pool rows: {output_counts.get('candidate_pool', 0)}",
        "",
        "candidate counts:",
    ]
    for key in ["high_zt_candidates", "low_kappa_high_sigma_candidates", "low_rare_metal_candidates", "low_toxicity_candidates", "nanocarbon_candidates", "balanced_recommended_candidates", "manual_review_needed_candidates", "sintering_check_needed_candidates"]:
        lines.append(f"- {key}: {output_counts.get(key, 0)}")
    lines += [
        "",
        "thresholds:",
        f"- zt_threshold: {args.zt_threshold}",
        f"- kappa_threshold: {args.kappa_threshold}",
        f"- sigma_threshold: {args.sigma_threshold}",
        "",
        "candidate tier:",
    ]
    for tier in ["A", "B", "C", "review", "low"]:
        lines.append(f"- {tier}: {int(tier_counts.get(tier, 0))}")
    lines += ["", "top material systems by candidate A/B count:"]
    top_ab = material_summary.assign(candidate_AB_count=material_summary["candidate_A_count"] + material_summary["candidate_B_count"]).sort_values("candidate_AB_count", ascending=False).head(20)
    for _, row in top_ab.iterrows():
        lines.append(f"- {row['material_system']} / {row['n_or_p']}: {int(row['candidate_AB_count'])}")
    lines += ["", "top material systems by high ZT count:"]
    for _, row in material_summary.sort_values("high_zt_count", ascending=False).head(20).iterrows():
        lines.append(f"- {row['material_system']} / {row['n_or_p']}: {int(row['high_zt_count'])}")
    lines += ["", "top material systems by low kappa high sigma count:"]
    for _, row in material_summary.sort_values("low_kappa_high_sigma_count", ascending=False).head(20).iterrows():
        lines.append(f"- {row['material_system']} / {row['n_or_p']}: {int(row['low_kappa_high_sigma_count'])}")
    lines += [
        "",
        "flags:",
        f"- rare_metal_flag_step24=True: {flag_counts.get('rare_metal_flag_step24', 0)}",
        f"- toxicity_attention_flag_step24=True: {flag_counts.get('toxicity_attention_flag_step24', 0)}",
        f"- nanocarbon_candidate=True: {flag_counts.get('is_nanocarbon_candidate_step24', 0)}",
        "",
        "n/p:",
    ]
    by_np_ab = np_summary.assign(candidate_AB_count=np_summary["candidate_A_count"] + np_summary["candidate_B_count"])
    for _, row in by_np_ab.iterrows():
        lines.append(f"- {row['n_or_p']} candidate A/B count: {int(row['candidate_AB_count'])}")
    for _, row in by_np_ab.iterrows():
        lines.append(f"- {row['n_or_p']} high ZT count: {int(row['high_zt_count'])}")
    lines += [
        "",
        "missing information:",
        f"- additive unknown: {int(is_unknown(str_col(pool, 'additive_final_step17')).sum())}",
        f"- structure unknown: {int(is_unknown(str_col(pool, 'structure_final_step17')).sum())}",
        f"- sintering unknown: {int(is_unknown(str_col(pool, 'sintering_method')).sum())}",
        "",
        f"n/p changed rows: {np_changed}",
        f"sintering changed rows: {sintering_changed}",
        "",
        "duplicate sample_key checks:",
    ]
    lines.extend(f"- {line}" for line in duplicate_report)
    lines += [
        "",
        "flag counts:",
    ]
    for key, value in flag_counts.items():
        lines.append(f"- {key}=True: {value}")
    lines += [
        "",
        "Notes:",
        "Step24では新しい予測、tau_eff再fitting、ML再学習、PF/ZT再計算は行っていない。",
        "Step24では既存の予測・観測値を使って候補抽出を行った。",
        "low rare metal attention はレアメタルなし確定ではなく、注意元素が検出されないという意味である。",
        "low toxicity attention は弱毒性確定ではなく、毒性注意元素が検出されないという意味である。",
        "composition由来の元素判定は注意フラグであり、化学的・法的な安全性判定ではない。",
        "添加物・構造・焼結方法がunknownの場合は、情報不足として扱った。",
        "downstream ML predictions are for screening, not unbiased evaluation.",
    ]
    return "\n".join(lines) + "\n"


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    step23 = Path(args.step23_dir)
    step21 = Path(args.step21_dir)
    step22 = Path(args.step22_dir)
    step17 = Path(args.step17_dir)

    downstream = read_csv(step21 / "pf_zt_ml_downstream_sample_results_step21.csv", required=True)
    candidates21 = read_csv(step21 / "pf_zt_ml_candidate_samples_step21.csv", required=True)
    rows = read_csv(step21 / "thermoelectric_ml_downstream_predictions_step21.csv", required=True)
    primary21 = read_csv(step21 / "pf_zt_ml_primary_sample_results_step21.csv", required=False)
    err = read_csv(step23 / "step23_error_cause_samples.csv", required=True)
    manual = read_csv(step23 / "step23_manual_review_priority_samples.csv", required=False)
    sintering = read_csv(step23 / "step23_sintering_check_priority_samples.csv", required=False)
    highzt_err = read_csv(step23 / "step23_high_zt_error_cases.csv", required=False)
    comp22 = read_csv(step22 / "step22_sample_level_comparison.csv", required=False)
    ann17 = read_csv(step17 / "step17_annotated_samples.csv", required=False)

    require_columns(downstream, ["sample_key"], "pf_zt_ml_downstream_sample_results_step21.csv")
    require_columns(err, ["sample_key"], "step23_error_cause_samples.csv")
    require_columns(candidates21, ["sample_key"], "pf_zt_ml_candidate_samples_step21.csv")
    require_columns(rows, ["sample_key"], "thermoelectric_ml_downstream_predictions_step21.csv")

    duplicate_report = []
    input_counts = {
        "downstream_samples": len(downstream),
        "downstream_rows": len(rows),
        "error_cause_samples": len(err),
    }

    base_dups = duplicate_count(downstream)
    duplicate_report.append(f"pf_zt_ml_downstream_sample_results_step21.csv: rows={len(downstream)}, duplicate sample_key rows={base_dups}; first row kept")
    base_before = first_by_key(downstream)
    original_np = base_before[[c for c in ["sample_key", "n_or_p", "n_or_p_basis", "n_or_p_step6", "n_or_p_basis_step6", "n_or_p_confidence_step6"] if c in base_before.columns]].copy()
    original_sintering = base_before[[c for c in ["sample_key", "sintering_method", "sintering_checked", "record_checked"] if c in base_before.columns]].copy()

    pool = base_before.copy()
    row_agg = aggregate_rows(rows)
    pool = merge_by_sample_key(pool, row_agg, "row_agg_step24", duplicate_report)
    pool = merge_by_sample_key(pool, err, "step23", duplicate_report)
    pool = merge_by_sample_key(pool, comp22, "step22", duplicate_report)
    pool = merge_by_sample_key(pool, ann17, "step17", duplicate_report)
    pool = merge_by_sample_key(pool, manual, "manual_priority", duplicate_report, "in_manual_review_priority_step24")
    pool = merge_by_sample_key(pool, sintering, "sintering_priority", duplicate_report, "in_sintering_check_priority_step24")
    pool = merge_by_sample_key(pool, highzt_err, "high_zt_error", duplicate_report, "in_high_zt_error_step24")
    pool = merge_by_sample_key(pool, primary21, "primary_step21", duplicate_report)

    # Fill important fields from suffixed optional columns without changing existing base columns.
    for col in [
        "paper_title", "composition", "material_system", "n_or_p_final_step17", "n_or_p_basis", "n_or_p_step6",
        "n_or_p_basis_step6", "n_or_p_confidence_step6", "additive_final_step17", "structure_final_step17",
        "additive_source_step17", "additive_confidence_final_step17", "structure_source_step17",
        "structure_confidence_final_step17", "sintering_method_final_step17", "sintering_condition_final_step17",
        "sintering_checked_final_step17", "sintering_source_step17", "sintering_confidence_final_step17", "paper_checked_step17",
        "manual_review_status_step17", "nanocarbon_keyword_detected_step9", "nanocarbon_type_auto_step9",
        "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9", "comparison_category_step22",
        "primary_error_source_hypothesis_step23", "secondary_error_source_hypothesis_step23",
        "error_pattern_step23", "error_cause_note_step23", "missing_additive_info_step23",
        "missing_structure_info_step23", "missing_sintering_info_step23", "manual_annotation_needed_step23",
        "step23_review_priority_score", "step23_review_priority_tier", "step23_review_reason",
        "sigma_fitting_log_rmse_step22", "sigma_ML_log_rmse_step22", "sigma_log_rmse_gap_ML_minus_fitting_step22",
        "pf_fitting_mape_step22", "pf_ML_mape_step22", "pf_mape_gap_ML_minus_fitting_step22",
        "zt_obs_fitting_mape_step22", "zt_obs_ML_mape_step22", "zt_obs_mape_gap_ML_minus_fitting_step22",
    ]:
        if col not in pool.columns:
            pool[col] = pd.NA
        for suffix_col in [c for c in pool.columns if c.startswith(f"{col}__")]:
            pool[col] = pool[col].where(~is_unknown(pool[col]), pool[suffix_col])

    pool = add_element_flags(pool)
    pool = add_candidate_flags(pool, args)
    pool, score_breakdown = add_scores(pool)
    pool = ordered_columns(pool)

    high_zt = sort_mixed_safe(
        pool[pool["is_high_zt_observed_step24"] | pool["is_high_zt_ml_predicted_step24"] | pool["is_high_zt_fitting_predicted_step24"]].copy(),
        ["zt_obs_max_step24", "zt_pred_ML_max_step24", "candidate_score_step24"],
        [False, False, False],
    )
    low_kappa_high_sigma = sort_mixed_safe(
        pool[pool["is_low_kappa_high_sigma_step24"]].copy(),
        ["kappa_obs_min_step24", "sigma_obs_max_step24", "sigma_pred_ML_max_step24", "candidate_score_step24"],
        [True, False, False, False],
    )
    low_rare = sort_desc_safe(pool[pool["is_low_rare_metal_attention_step24"]].copy(), ["candidate_score_step24"])
    low_toxic = sort_desc_safe(pool[pool["is_low_toxicity_attention_step24"]].copy(), ["candidate_score_step24"])
    nano = sort_mixed_safe(pool[pool["is_nanocarbon_candidate_step24"]].copy(), ["candidate_score_step24", "zt_obs_max_step24", "zt_pred_ML_max_step24"], [False, False, False])
    balanced = sort_desc_safe(pool[(pool["candidate_tier_step24"].isin(["A", "B"])) | pool["is_balanced_recommended_candidate_step24"]].copy(), ["candidate_score_step24", "zt_obs_max_step24"]).head(args.top_n_candidates)
    ml_supported = sort_desc_safe(pool[pool["is_ml_supported_candidate_step24"]].copy(), ["candidate_score_step24", "zt_pred_ML_max_step24"])
    fitting_supported = sort_desc_safe(pool[pool["is_fitting_supported_candidate_step24"]].copy(), ["candidate_score_step24", "zt_pred_fitting_max_step24", "zt_obs_max_step24"])
    manual_needed = sort_desc_safe(pool[pool["needs_manual_review_step24"] | is_true(str_col(pool, "missing_additive_info_step23")) | is_true(str_col(pool, "missing_structure_info_step23")) | pool["candidate_tier_step24"].isin(["A", "B"])].copy(), ["candidate_score_step24"]).head(args.top_n_candidates)
    sintering_needed = sort_desc_safe(pool[pool["needs_sintering_check_step24"] | (is_true(str_col(pool, "missing_sintering_info_step23")) & pool["candidate_tier_step24"].isin(["A", "B"]))].copy(), ["candidate_score_step24"]).head(args.top_n_candidates)

    summary_material, summary_np = make_summaries(pool)

    np_compare = pool.merge(original_np, on="sample_key", how="left", suffixes=("", "__original"))
    np_changed = 0
    for col in ["n_or_p", "n_or_p_basis", "n_or_p_step6", "n_or_p_basis_step6", "n_or_p_confidence_step6"]:
        if col in np_compare.columns and f"{col}__original" in np_compare.columns:
            np_changed += int((np_compare[col].astype(str) != np_compare[f"{col}__original"].astype(str)).sum())
    sintering_compare = pool.merge(original_sintering, on="sample_key", how="left", suffixes=("", "__original"))
    sintering_changed = 0
    for col in ["sintering_method", "sintering_checked", "record_checked"]:
        if col in sintering_compare.columns and f"{col}__original" in sintering_compare.columns:
            sintering_changed += int((sintering_compare[col].astype(str) != sintering_compare[f"{col}__original"].astype(str)).sum())

    outputs = {
        "step24_candidate_pool.csv": pool,
        "step24_high_zt_candidates.csv": high_zt,
        "step24_low_kappa_high_sigma_candidates.csv": low_kappa_high_sigma,
        "step24_low_rare_metal_candidates.csv": low_rare,
        "step24_low_toxicity_candidates.csv": low_toxic,
        "step24_nanocarbon_candidates.csv": nano,
        "step24_balanced_recommended_candidates.csv": balanced,
        "step24_ml_supported_candidates.csv": ml_supported,
        "step24_fitting_supported_candidates.csv": fitting_supported,
        "step24_manual_review_needed_candidates.csv": manual_needed,
        "step24_sintering_check_needed_candidates.csv": sintering_needed,
        "step24_candidate_score_breakdown.csv": score_breakdown,
        "step24_candidate_summary_by_material.csv": summary_material,
        "step24_candidate_summary_by_np_type.csv": summary_np,
    }
    for name, df in outputs.items():
        write_csv(df, outdir / name)

    output_counts = {
        "candidate_pool": len(pool),
        "high_zt_candidates": len(high_zt),
        "low_kappa_high_sigma_candidates": len(low_kappa_high_sigma),
        "low_rare_metal_candidates": len(low_rare),
        "low_toxicity_candidates": len(low_toxic),
        "nanocarbon_candidates": len(nano),
        "balanced_recommended_candidates": len(balanced),
        "manual_review_needed_candidates": len(manual_needed),
        "sintering_check_needed_candidates": len(sintering_needed),
    }
    flag_cols = [
        "is_high_zt_observed_step24", "is_high_zt_ml_predicted_step24", "is_low_kappa_step24",
        "is_high_sigma_obs_step24", "is_low_kappa_high_sigma_step24", "is_low_rare_metal_attention_step24",
        "is_low_toxicity_attention_step24", "is_nanocarbon_candidate_step24", "rare_metal_flag_step24",
        "toxicity_attention_flag_step24",
    ]
    flag_counts = {c: int(pool[c].fillna(False).astype(bool).sum()) for c in flag_cols if c in pool.columns}
    tier_counts = pool["candidate_tier_step24"].value_counts().to_dict()
    duplicate_report.extend([
        f"step24_candidate_pool.csv duplicate sample_key rows: {duplicate_count(pool)}",
        f"step24_balanced_recommended_candidates.csv duplicate sample_key rows: {duplicate_count(balanced)}",
        f"step24_high_zt_candidates.csv duplicate sample_key rows: {duplicate_count(high_zt)}",
    ])

    report = build_report(args, input_counts, output_counts, tier_counts, flag_counts, summary_material, summary_np, duplicate_report, np_changed, sintering_changed, pool)
    (outdir / "step24_candidate_selection_report.txt").write_text(report, encoding="utf-8")
    (outdir / "step24_candidate_selection_notes.md").write_text(notes_text(), encoding="utf-8")

    write_excel(
        outdir / "starrydata2_step24_material_candidates.xlsx",
        {
            "candidate_pool": pool,
            "balanced_recommended": balanced,
            "high_zt_candidates": high_zt,
            "low_kappa_high_sigma": low_kappa_high_sigma,
            "low_rare_metal": low_rare,
            "low_toxicity": low_toxic,
            "nanocarbon_candidates": nano,
            "ml_supported": ml_supported,
            "fitting_supported": fitting_supported,
            "manual_review_needed": manual_needed,
            "sintering_check_needed": sintering_needed,
            "score_breakdown": score_breakdown,
            "summary_by_material": summary_material,
            "summary_by_np_type": summary_np,
            "candidate_report": report,
        },
    )

    print("Done.")
    print("Created:")
    for name in [
        "step24_candidate_pool.csv",
        "step24_high_zt_candidates.csv",
        "step24_low_kappa_high_sigma_candidates.csv",
        "step24_low_rare_metal_candidates.csv",
        "step24_low_toxicity_candidates.csv",
        "step24_nanocarbon_candidates.csv",
        "step24_balanced_recommended_candidates.csv",
        "step24_ml_supported_candidates.csv",
        "step24_fitting_supported_candidates.csv",
        "step24_manual_review_needed_candidates.csv",
        "step24_sintering_check_needed_candidates.csv",
        "step24_candidate_score_breakdown.csv",
        "step24_candidate_summary_by_material.csv",
        "step24_candidate_summary_by_np_type.csv",
        "step24_candidate_selection_report.txt",
        "step24_candidate_selection_notes.md",
        "starrydata2_step24_material_candidates.xlsx",
    ]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"candidate pool samples: {len(pool)}")
    print(f"balanced recommended candidates: {len(balanced)}")
    print(f"high ZT candidates: {len(high_zt)}")
    print(f"low kappa high sigma candidates: {len(low_kappa_high_sigma)}")
    print(f"low rare metal attention candidates: {len(low_rare)}")
    print(f"low toxicity attention candidates: {len(low_toxic)}")
    print(f"nanocarbon candidates: {len(nano)}")
    print(f"manual review needed candidates: {len(manual_needed)}")
    print(f"sintering check needed candidates: {len(sintering_needed)}")
    print(
        "candidate tier A/B/C/review/low: "
        f"{int(tier_counts.get('A', 0))}/{int(tier_counts.get('B', 0))}/{int(tier_counts.get('C', 0))}/"
        f"{int(tier_counts.get('review', 0))}/{int(tier_counts.get('low', 0))}"
    )
    print(f"n/p changed rows: {np_changed}")
    print(f"existing sintering columns changed rows: {sintering_changed}")


if __name__ == "__main__":
    main()
