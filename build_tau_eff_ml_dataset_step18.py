import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font


DEFAULT_STEP12_DIR = "data/output/starrydata2_step12_tau_fit"
DEFAULT_STEP13_DIR = "data/output/starrydata2_step13_sigma_validation"
DEFAULT_STEP15_DIR = "data/output/starrydata2_step15_pf_zt_error_analysis"
DEFAULT_STEP16_DIR = "data/output/starrydata2_step16_result_summary"
DEFAULT_STEP17_DIR = "data/output/starrydata2_step17_literature_review"
DEFAULT_STEP9_DIR = "data/output/starrydata2_step9_literature_annotations"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step18_tau_eff_ml_dataset"

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

REQUIRED_FILES = {
    "step12": "tau_fit_results_step12.csv",
    "step13": "tau_validation_primary_results_step13.csv",
    "step15": "pf_zt_error_samples_step15.csv",
    "step17": "step17_annotated_samples.csv",
}

OPTIONAL_FILES = {
    "step17_base": "step17_tau_eff_ml_annotation_base.csv",
    "step16_targets": "step16_next_step17_review_targets.csv",
    "best_candidates": "best_candidate_samples_step15.csv",
    "manual_candidates": "manual_review_candidates_step15.csv",
    "sintering_candidates": "sintering_check_candidates_step15.csv",
    "step9": "sample_literature_annotations_step9.csv",
}

TARGET_COLUMNS = [
    "target_log_tau_eff_step18",
    "target_tau_eff_step18",
    "target_available_step18",
    "target_quality_step18",
    "target_quality_note_step18",
    "use_for_tau_eff_ml_step18",
    "ml_exclusion_reason_step18",
]

LEAKAGE_COLUMNS_EXACT = {
    "tau_eff_step12",
    "log_tau_eff_step12",
    "target_tau_eff_step18",
    "target_log_tau_eff_step18",
    "sigma_fit_log_rmse_step12",
    "sigma_fit_mape_step12",
    "sigma_holdout_log_rmse_step12",
    "sigma_holdout_mape_step12",
    "validation_sigma_log_rmse_step13",
    "validation_sigma_mape_step13",
    "zt_pred_vs_obs_mape_step14",
    "zt_pred_vs_calc_mape_step14",
    "zt_obs_max_step14",
    "zt_pred_max_step14",
}

LEAKAGE_PATTERNS = [
    "tau_eff_step12",
    "log_tau_eff_step12",
    "target_tau_eff",
    "target_log_tau_eff",
    "sigma_fit_",
    "sigma_holdout_",
    "validation_sigma_",
    "zt_pred_",
    "zt_obs_",
    "zt_calc_",
    "pf_mape",
    "pf_log_rmse",
    "manual_review_priority_score",
    "problem_reason",
]

FEATURE_CATEGORICAL_RAW = [
    "material_system",
    "n_or_p_final_step17",
    "n_or_p",
    "n_or_p_confidence_final_step17",
    "n_or_p_confidence_step6",
    "fitting_source_actual_step10",
    "tau_eff_mode_step12",
    "additive_final_step17",
    "additive_source_step17",
    "additive_confidence_final_step17",
    "structure_final_step17",
    "structure_source_step17",
    "structure_confidence_final_step17",
    "nanocarbon_type_auto_step9",
    "sintering_method_final_step17",
    "sintering_checked_final_step17",
    "sintering_source_step17",
    "sintering_confidence_final_step17",
]

ONE_HOT_COLUMNS = [
    "material_system",
    "n_or_p_final_step17",
    "n_or_p",
    "fitting_source_actual_step10",
    "nanocarbon_type_auto_step9",
    "additive_final_step17",
    "structure_final_step17",
    "sintering_method_final_step17",
]

BINARY_FEATURE_COLUMNS = [
    "nanocarbon_keyword_detected_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "contains_C_step18",
    "contains_Bi_step18",
    "contains_Te_step18",
    "contains_Se_step18",
    "contains_Sb_step18",
    "contains_Pb_step18",
    "contains_Sn_step18",
    "contains_Ge_step18",
    "contains_Si_step18",
    "contains_Mg_step18",
    "contains_Co_step18",
    "contains_Fe_step18",
    "contains_Cu_step18",
    "contains_Ag_step18",
    "contains_In_step18",
    "contains_Ga_step18",
    "contains_Ca_step18",
    "contains_Na_step18",
    "contains_K_step18",
    "contains_O_step18",
    "contains_N_step18",
    "has_chalcogen_step18",
    "has_pnictogen_step18",
    "has_transition_metal_step18",
    "has_alkali_or_alkaline_step18",
    "has_heavy_element_step18",
    "has_toxic_attention_element_step18",
    "has_rare_metal_attention_element_step18",
]

NUMERIC_FEATURE_COLUMNS = [
    "element_count_step18",
]

METADATA_COLUMNS = [
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
    "n_or_p_final_step17",
    "n_or_p_confidence_final_step17",
    "tau_eff_step12",
    "log_tau_eff_step12",
    "tau_eff_unit_step12",
    "tau_eff_mode_step12",
    "fit_status_step12",
    "n_fit_rows_step12",
    "sigma_fit_log_rmse_step12",
    "sigma_fit_mape_step12",
    "validation_sigma_log_rmse_step13",
    "validation_sigma_mape_step13",
    "zt_obs_max_step14",
    "zt_pred_max_step14",
    "zt_pred_vs_obs_mape_step14",
    "zt_pred_vs_calc_mape_step14",
    "paper_checked_step17",
    "manual_review_status_step17",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "sintering_method_final_step17",
    "sintering_checked_final_step17",
]

EXCEL_PREVIEW_ROWS = 100_000

ELEMENT_SYMBOLS = [
    "Ac", "Ag", "Al", "Am", "Ar", "As", "At", "Au", "B", "Ba", "Be", "Bh", "Bi", "Bk",
    "Br", "C", "Ca", "Cd", "Ce", "Cf", "Cl", "Cm", "Cn", "Co", "Cr", "Cs", "Cu", "Db",
    "Ds", "Dy", "Er", "Es", "Eu", "F", "Fe", "Fl", "Fm", "Fr", "Ga", "Gd", "Ge", "H",
    "He", "Hf", "Hg", "Ho", "Hs", "I", "In", "Ir", "K", "Kr", "La", "Li", "Lr", "Lu",
    "Lv", "Mc", "Md", "Mg", "Mn", "Mo", "Mt", "N", "Na", "Nb", "Nd", "Ne", "Nh", "Ni",
    "No", "Np", "O", "Og", "Os", "P", "Pa", "Pb", "Pd", "Pm", "Po", "Pr", "Pt", "Pu",
    "Ra", "Rb", "Re", "Rf", "Rg", "Rh", "Rn", "Ru", "S", "Sb", "Sc", "Se", "Sg", "Si",
    "Sm", "Sn", "Sr", "Ta", "Tb", "Tc", "Te", "Th", "Ti", "Tl", "Tm", "Ts", "U", "V",
    "W", "Xe", "Y", "Yb", "Zn", "Zr",
]
ELEMENT_RE = re.compile("|".join(sorted(ELEMENT_SYMBOLS, key=len, reverse=True)))

CHALCOGEN = {"O", "S", "Se", "Te"}
PNICTOGEN = {"N", "P", "As", "Sb", "Bi"}
TRANSITION_METAL = {"Fe", "Co", "Ni", "Cu", "Zn", "Ag", "Cd", "Pt", "Pd", "Rh", "Ru", "Ir"}
ALKALI_OR_ALKALINE = {"Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba"}
HEAVY_ELEMENT = {"Pb", "Bi", "Sb", "Te", "Hg", "Tl"}
TOXIC_ATTENTION = {"Pb", "Cd", "Hg", "Tl", "As", "Se", "Te", "Sb"}
RARE_METAL_ATTENTION = {
    "Te", "Se", "Bi", "Sb", "Ge", "In", "Ga", "Hf", "Ta", "W", "Re", "Pt", "Pd",
    "Rh", "Ru", "Ir", "Os", "Ag", "Au", "Y", "La", "Ce", "Nd", "Sm", "Gd", "Dy", "Yb",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "t"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build Step18 tau_eff ML dataset from Step12 fitted labels.")
    parser.add_argument("--step12_dir", default=DEFAULT_STEP12_DIR)
    parser.add_argument("--step13_dir", default=DEFAULT_STEP13_DIR)
    parser.add_argument("--step15_dir", default=DEFAULT_STEP15_DIR)
    parser.add_argument("--step16_dir", default=DEFAULT_STEP16_DIR)
    parser.add_argument("--step17_dir", default=DEFAULT_STEP17_DIR)
    parser.add_argument("--step9_dir", default=DEFAULT_STEP9_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min_fit_rows", type=int, default=5)
    parser.add_argument("--max_sigma_fit_log_rmse", type=float, default=1.0)
    parser.add_argument("--max_validation_log_rmse", type=float, default=1.0)
    parser.add_argument("--use_only_fit_ok", type=parse_bool, default=True)
    parser.add_argument("--top_category_limit", type=int, default=30)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def read_csv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return None
    return pd.read_csv(path, dtype=dtype_for_existing(path), low_memory=False)


def first_by_sample_key(df, name, duplicate_notes):
    if df is None:
        return None
    if "sample_key" not in df.columns:
        duplicate_notes.append(f"{name}: no sample_key column; skipped")
        return None
    dup_count = int(df["sample_key"].duplicated().sum())
    if dup_count:
        duplicate_keys = df.loc[df["sample_key"].duplicated(), "sample_key"].dropna().astype(str).head(20).tolist()
        duplicate_notes.append(f"{name}: {dup_count} duplicate sample_key rows; first row used; examples={duplicate_keys}")
    return df.drop_duplicates("sample_key", keep="first").copy()


def merge_supplement(base, supplement, source_name, duplicate_notes):
    if supplement is None:
        return base
    supplement = first_by_sample_key(supplement, source_name, duplicate_notes)
    if supplement is None:
        return base

    base_cols = set(base.columns)
    keep_cols = ["sample_key"] + [c for c in supplement.columns if c != "sample_key" and c not in base_cols]

    for col in supplement.columns:
        if col == "sample_key":
            continue
        if col in base_cols:
            new_col = f"{col}__{source_name}"
            supplement = supplement.rename(columns={col: new_col})
            keep_cols.append(new_col)

    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in supplement.columns]))
    return base.merge(supplement[keep_cols], on="sample_key", how="left")


def coalesce_columns(df, target_col, candidate_cols, default=np.nan):
    series = None
    for col in candidate_cols:
        if col not in df.columns:
            continue
        if series is None:
            series = df[col].copy()
        else:
            series = series.where(series.notna() & (series.astype(str).str.strip() != ""), df[col])
    if series is None:
        df[target_col] = default
    else:
        df[target_col] = series.fillna(default)
    return df


def ensure_columns(df, columns, default=np.nan):
    for col in columns:
        if col not in df.columns:
            df[col] = default
    return df


def to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_bool_series(series):
    return series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0, "y": 1, "n": 0}
    ).fillna(0).astype(int)


def finite_positive(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0)


def finite(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values)


def build_target_flags(df, min_fit_rows, max_sigma_fit_log_rmse, max_validation_log_rmse, use_only_fit_ok):
    numeric_cols = [
        "tau_eff_step12",
        "log_tau_eff_step12",
        "n_fit_rows_step12",
        "sigma_fit_log_rmse_step12",
        "sigma_fit_mape_step12",
        "validation_sigma_log_rmse_step13",
        "validation_sigma_mape_step13",
    ]
    to_numeric(df, numeric_cols)

    df["target_log_tau_eff_step18"] = df["log_tau_eff_step12"]
    df["target_tau_eff_step18"] = df["tau_eff_step12"]

    fit_ok = df["fit_status_step12"].astype(str).str.lower().eq("ok")
    target_available = finite(df["log_tau_eff_step12"]) & finite_positive(df["tau_eff_step12"]) & fit_ok
    df["target_available_step18"] = target_available

    sigma_fit = pd.to_numeric(df["sigma_fit_log_rmse_step12"], errors="coerce")
    validation = pd.to_numeric(df["validation_sigma_log_rmse_step13"], errors="coerce")
    n_fit_rows = pd.to_numeric(df["n_fit_rows_step12"], errors="coerce")

    quality = np.full(len(df), "exclude", dtype=object)
    valid_base = target_available & (n_fit_rows >= min_fit_rows)
    high = valid_base & (sigma_fit <= 0.4) & (validation <= 0.4)
    medium = valid_base & (sigma_fit <= 0.8) & (validation <= 0.8)
    low = valid_base & ~(high | medium)
    quality[high] = "high"
    quality[medium] = "medium"
    quality[low] = "low"
    df["target_quality_step18"] = quality

    notes = []
    for _, row in df.iterrows():
        row_notes = []
        if not bool(row["target_available_step18"]):
            row_notes.append("missing or invalid target or fit_status not ok")
        if pd.isna(row.get("validation_sigma_log_rmse_step13")):
            row_notes.append("validation_sigma_log_rmse_step13 missing")
        if pd.to_numeric(pd.Series([row.get("n_fit_rows_step12")]), errors="coerce").iloc[0] < min_fit_rows:
            row_notes.append("too few fit rows")
        notes.append("; ".join(row_notes) if row_notes else "ok")
    df["target_quality_note_step18"] = notes

    validation_ok = validation.isna() | (validation <= max_validation_log_rmse)
    use = (
        target_available
        & (n_fit_rows >= min_fit_rows)
        & (sigma_fit <= max_sigma_fit_log_rmse)
        & validation_ok
    )
    if use_only_fit_ok:
        use &= fit_ok

    reasons = []
    for pos, (_, row) in enumerate(df.iterrows()):
        reason = "ok"
        if not bool(use.iloc[pos] if hasattr(use, "iloc") else use[pos]):
            if not bool(row["target_available_step18"]):
                reason = "missing target"
            elif use_only_fit_ok and str(row.get("fit_status_step12", "")).lower() != "ok":
                reason = "fit_status not ok"
            elif pd.to_numeric(pd.Series([row.get("n_fit_rows_step12")]), errors="coerce").iloc[0] < min_fit_rows:
                reason = "too few fit rows"
            elif pd.to_numeric(pd.Series([row.get("sigma_fit_log_rmse_step12")]), errors="coerce").iloc[0] > max_sigma_fit_log_rmse:
                reason = "large sigma fit error"
            elif (
                pd.notna(row.get("validation_sigma_log_rmse_step13"))
                and pd.to_numeric(pd.Series([row.get("validation_sigma_log_rmse_step13")]), errors="coerce").iloc[0]
                > max_validation_log_rmse
            ):
                reason = "large validation error"
            elif pd.isna(row.get("composition")) or str(row.get("composition")).strip() == "":
                reason = "missing composition"
            elif pd.isna(row.get("material_system")) or str(row.get("material_system")).strip().lower() in {"", "unknown", "nan"}:
                reason = "unknown material_system"
        reasons.append(reason)
    df["use_for_tau_eff_ml_step18"] = use
    df["ml_exclusion_reason_step18"] = reasons
    return df


def extract_elements(composition):
    if pd.isna(composition):
        return []
    text = str(composition)
    found = []
    for match in ELEMENT_RE.finditer(text):
        symbol = match.group(0)
        start, end = match.span()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if before.isalpha() or after.islower():
            continue
        found.append(symbol)
    return sorted(set(found), key=lambda x: ELEMENT_SYMBOLS.index(x) if x in ELEMENT_SYMBOLS else 999)


def add_element_features(df):
    elements = df["composition"].apply(extract_elements) if "composition" in df.columns else pd.Series([[]] * len(df))
    df["elements_detected_step18"] = elements.apply(lambda xs: ";".join(xs))
    df["element_count_step18"] = elements.apply(len)

    tracked = ["C", "Bi", "Te", "Se", "Sb", "Pb", "Sn", "Ge", "Si", "Mg", "Co", "Fe", "Cu", "Ag", "In", "Ga", "Ca", "Na", "K", "O", "N"]
    for symbol in tracked:
        df[f"contains_{symbol}_step18"] = elements.apply(lambda xs, s=symbol: int(s in xs))

    df["has_chalcogen_step18"] = elements.apply(lambda xs: int(bool(set(xs) & CHALCOGEN)))
    df["has_pnictogen_step18"] = elements.apply(lambda xs: int(bool(set(xs) & PNICTOGEN)))
    df["has_transition_metal_step18"] = elements.apply(lambda xs: int(bool(set(xs) & TRANSITION_METAL)))
    df["has_alkali_or_alkaline_step18"] = elements.apply(lambda xs: int(bool(set(xs) & ALKALI_OR_ALKALINE)))
    df["has_heavy_element_step18"] = elements.apply(lambda xs: int(bool(set(xs) & HEAVY_ELEMENT)))
    df["has_toxic_attention_element_step18"] = elements.apply(lambda xs: int(bool(set(xs) & TOXIC_ATTENTION)))
    df["has_rare_metal_attention_element_step18"] = elements.apply(lambda xs: int(bool(set(xs) & RARE_METAL_ATTENTION)))
    return df


def sanitize_category(value):
    text = str(value).strip().lower()
    if text in {"", "nan", "<na>", "none"}:
        text = "unknown"
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return text[:80] if text else "unknown"


def cap_categories(series, top_category_limit):
    clean = series.fillna("unknown").astype(str).str.strip()
    clean = clean.where(~clean.str.lower().isin({"", "nan", "<na>", "none"}), "unknown")
    top = set(clean.value_counts(dropna=False).head(top_category_limit).index)
    return clean.where(clean.isin(top), "other")


def is_leakage_column(col):
    lower = col.lower()
    if col in LEAKAGE_COLUMNS_EXACT:
        return True
    return any(pattern.lower() in lower for pattern in LEAKAGE_PATTERNS)


def build_feature_matrix(df, top_category_limit):
    feature_df = pd.DataFrame({"sample_key": df["sample_key"]})

    for col in NUMERIC_FEATURE_COLUMNS:
        if col in df.columns and not is_leakage_column(col):
            feature_df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BINARY_FEATURE_COLUMNS:
        if col in df.columns and not is_leakage_column(col):
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            else:
                feature_df[col] = normalize_bool_series(df[col])

    for col in ONE_HOT_COLUMNS:
        if col not in df.columns or is_leakage_column(col):
            continue
        capped = cap_categories(df[col], top_category_limit)
        for category in sorted(capped.dropna().unique(), key=str):
            out_col = f"{col}__{sanitize_category(category)}"
            if is_leakage_column(out_col):
                continue
            feature_df[out_col] = (capped == category).astype(int)

    leakage_cols = [c for c in feature_df.columns if c != "sample_key" and is_leakage_column(c)]
    if leakage_cols:
        feature_df = feature_df.drop(columns=leakage_cols)
    return feature_df, leakage_cols


def stable_hash_fraction(text, salt=""):
    raw = f"{salt}|{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def build_splits(df):
    splits = pd.DataFrame({"sample_key": df["sample_key"]})
    random_frac = df["sample_key"].astype(str).apply(lambda x: stable_hash_fraction(x, "random80"))
    splits["split_random_80_20_step18"] = np.where(random_frac < 0.8, "train", "test")

    random_frac_3 = df["sample_key"].astype(str).apply(lambda x: stable_hash_fraction(x, "random701515"))
    splits["split_random_70_15_15_step18"] = np.where(
        random_frac_3 < 0.7,
        "train",
        np.where(random_frac_3 < 0.85, "valid", "test"),
    )

    doi_keys = []
    doi_missing = 0
    for _, row in df.iterrows():
        doi = row.get("DOI")
        if pd.isna(doi) or str(doi).strip() == "":
            doi_missing += 1
            doi_keys.append(f"sample_key::{row['sample_key']}")
        else:
            doi_keys.append(f"doi::{str(doi).strip().lower()}")
    doi_frac = pd.Series(doi_keys).apply(lambda x: stable_hash_fraction(x, "doi80"))
    splits["split_doi_group_80_20_step18"] = np.where(doi_frac < 0.8, "train", "test")

    material = df["material_system"].fillna("unknown").astype(str).str.strip()
    top_materials = material[material.str.lower().ne("unknown")].value_counts().head(5).index.tolist()
    split_material = np.full(len(df), "not_assigned", dtype=object)
    for mat in top_materials:
        split_material[material.eq(mat)] = f"material_system_holdout_{sanitize_category(mat)}"
    splits["split_material_system_group_step18"] = split_material

    doi_leakage = 0
    if "DOI" in df.columns:
        tmp = pd.DataFrame({"DOI": df["DOI"], "split": splits["split_doi_group_80_20_step18"]})
        tmp = tmp[tmp["DOI"].notna() & tmp["DOI"].astype(str).str.strip().ne("")]
        doi_leakage = int(tmp.groupby("DOI")["split"].nunique().gt(1).sum())

    return splits, doi_missing, doi_leakage


def select_existing(df, columns):
    return df[[c for c in columns if c in df.columns]].copy()


def summarize_stats(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan, "p75": np.nan, "max": np.nan}
    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def value_summary(df, column, summary_type, denominator=None, top_n=None):
    if column not in df.columns:
        return pd.DataFrame(columns=["summary_type", "category", "count", "fraction", "note"])
    counts = df[column].fillna("missing").astype(str).value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    denom = denominator if denominator is not None else len(df)
    return pd.DataFrame(
        {
            "summary_type": summary_type,
            "category": counts.index.astype(str),
            "count": counts.values.astype(int),
            "fraction": [float(v / denom) if denom else 0.0 for v in counts.values],
            "note": "",
        }
    )


def build_data_quality_summary(dataset, splits):
    pieces = []
    summary_columns = [
        "fit_status_step12",
        "target_quality_step18",
        "use_for_tau_eff_ml_step18",
        "ml_exclusion_reason_step18",
        "material_system",
        "n_or_p_final_step17",
        "n_or_p",
        "paper_checked_step17",
        "manual_review_status_step17",
        "additive_source_step17",
        "structure_source_step17",
        "sintering_source_step17",
        "sintering_checked_final_step17",
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
    ]
    for col in summary_columns:
        pieces.append(value_summary(dataset, col, col))
    for col in ["split_random_80_20_step18", "split_doi_group_80_20_step18"]:
        pieces.append(value_summary(splits, col, col))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def build_feature_dictionary(dataset, feature_matrix):
    rows = []
    used = set(feature_matrix.columns) - {"sample_key"}

    descriptions = {
        "element_count_step18": "Number of unique elements detected from composition by regex.",
        "elements_detected_step18": "Semicolon-separated list of elements detected from composition.",
        "target_log_tau_eff_step18": "ML target based on log_tau_eff_step12.",
        "target_tau_eff_step18": "Original relative tau_eff target from Step12.",
    }

    all_cols = list(dict.fromkeys(list(dataset.columns) + list(feature_matrix.columns)))
    for col in all_cols:
        if col == "sample_key":
            feature_type = "metadata"
            source = "key"
            leakage = "none"
        elif col in TARGET_COLUMNS:
            feature_type = "target" if col.startswith("target_") else "quality"
            source = "step18"
            leakage = "target_leakage" if col.startswith("target_") else "low"
        elif col in used:
            source = "step18_feature_matrix"
            leakage = "none"
            if col in NUMERIC_FEATURE_COLUMNS:
                feature_type = "numeric"
            elif col in BINARY_FEATURE_COLUMNS or col.startswith("contains_") or col.startswith("has_"):
                feature_type = "binary"
            elif "__" in col:
                feature_type = "one_hot"
            else:
                feature_type = "numeric"
        elif is_leakage_column(col):
            source = infer_source(col)
            leakage = "target_leakage" if "tau_eff" in col else "evaluation_leakage"
            feature_type = "quality"
        elif col in FEATURE_CATEGORICAL_RAW:
            source = infer_source(col)
            leakage = "none"
            feature_type = "categorical"
        else:
            source = infer_source(col)
            leakage = "low" if col in METADATA_COLUMNS else "none"
            feature_type = "metadata"

        rows.append(
            {
                "feature_name": col,
                "feature_type": feature_type,
                "feature_source": source,
                "feature_description": descriptions.get(col, describe_feature(col)),
                "used_in_feature_matrix_step18": col in used,
                "leakage_risk": leakage,
                "note": "",
            }
        )
    return pd.DataFrame(rows)


def infer_source(col):
    for marker in ["step18", "step17", "step16", "step15", "step14", "step13", "step12", "step11", "step10", "step9", "step8", "step6"]:
        if marker in col:
            return marker
    if col in {"sample_key", "SID", "DOI", "doi_url", "sample_id", "paper_title", "year", "composition", "material_system", "n_or_p"}:
        return "sample_metadata"
    return "merged_input"


def describe_feature(col):
    if col.startswith("contains_"):
        return "Binary indicator that the composition contains the specified element."
    if col.startswith("has_"):
        return "Binary indicator for an element category detected from composition."
    if "__" in col:
        raw, cat = col.split("__", 1)
        return f"One-hot encoded category from {raw}: {cat}."
    if col.endswith("_final_step17"):
        return "Final annotation value carried from Step17 when available."
    return f"Column carried from {infer_source(col)}."


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"report": data.splitlines()})
            out = data.head(EXCEL_PREVIEW_ROWS).copy()
            out.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                width = min(max(len(v) for v in values) + 2, 60)
                ws.column_dimensions[col_cells[0].column_letter].width = width


def make_report(input_counts, output_counts, dataset, recommended, feature_matrix, splits, duplicate_notes, doi_missing, doi_leakage, leakage_cols):
    lines = []
    lines.append("Step18 tau_eff ML dataset report")
    lines.append("")
    lines.append(f"Input tau_fit_results_step12 rows: {input_counts.get('step12', 0)}")
    lines.append(f"Input tau_validation_primary_results_step13 rows: {input_counts.get('step13', 0)}")
    lines.append(f"Input pf_zt_error_samples_step15 rows: {input_counts.get('step15', 0)}")
    lines.append(f"Input step17_annotated_samples rows: {input_counts.get('step17', 0)}")
    lines.append("")
    lines.append(f"tau_eff_ml_dataset_step18 rows: {output_counts.get('dataset', 0)}")
    lines.append(f"tau_eff_ml_dataset_recommended_step18 rows: {output_counts.get('recommended', 0)}")
    lines.append(f"tau_eff_ml_feature_matrix_step18 rows: {output_counts.get('feature_matrix', 0)}")
    lines.append(f"tau_eff_ml_target_step18 rows: {output_counts.get('target', 0)}")
    lines.append(f"tau_eff_ml_excluded_samples_step18 rows: {output_counts.get('excluded', 0)}")
    lines.append("")
    lines.append("target:")
    for col in ["target_log_tau_eff_step18", "target_tau_eff_step18"]:
        stats = summarize_stats(dataset[col])
        text = ", ".join(f"{k}={v}" for k, v in stats.items())
        lines.append(f"- {col}: {text}")
    lines.append("")
    lines.append("target_quality_step18 counts:")
    lines.extend(format_counts(dataset, "target_quality_step18"))
    lines.append(f"use_for_tau_eff_ml_step18=True count: {int(dataset['use_for_tau_eff_ml_step18'].sum())}")
    lines.append("")
    lines.append("ml_exclusion_reason_step18 counts:")
    lines.extend(format_counts(dataset, "ml_exclusion_reason_step18"))
    lines.append("")
    lines.append("material_system recommended counts top 20:")
    lines.extend(format_counts(recommended, "material_system", top_n=20))
    lines.append("")
    lines.append("n_or_p_final_step17 recommended counts:")
    lines.extend(format_counts(recommended, "n_or_p_final_step17"))
    lines.append("nanocarbon_keyword_detected_step9 recommended counts:")
    lines.extend(format_counts(recommended, "nanocarbon_keyword_detected_step9"))
    lines.append("rare_metal_flag_auto_step9 recommended counts:")
    lines.extend(format_counts(recommended, "rare_metal_flag_auto_step9"))
    lines.append("toxicity_flag_auto_step9 recommended counts:")
    lines.extend(format_counts(recommended, "toxicity_flag_auto_step9"))
    lines.append("")
    lines.append("manual annotation:")
    for col in ["paper_checked_step17", "additive_source_step17", "structure_source_step17", "sintering_source_step17"]:
        lines.append(f"- {col}:")
        lines.extend(format_counts(dataset, col))
    lines.append("")
    lines.append("splits:")
    lines.append("- random 80/20:")
    lines.extend(format_counts(splits, "split_random_80_20_step18"))
    lines.append("- random 70/15/15:")
    lines.extend(format_counts(splits, "split_random_70_15_15_step18"))
    lines.append("- DOI group 80/20:")
    lines.extend(format_counts(splits, "split_doi_group_80_20_step18"))
    lines.append(f"- DOI missing rows assigned by sample_key: {doi_missing}")
    lines.append(f"- DOI leakage count: {doi_leakage}")
    lines.append("")
    feature_cols = [c for c in feature_matrix.columns if c != "sample_key"]
    one_hot_count = sum("__" in c for c in feature_cols)
    binary_count = sum(c in BINARY_FEATURE_COLUMNS or c.startswith("contains_") or c.startswith("has_") for c in feature_cols)
    numeric_count = sum(c in NUMERIC_FEATURE_COLUMNS for c in feature_cols)
    leakage_feature_count = sum(is_leakage_column(c) for c in feature_cols)
    lines.append("feature matrix:")
    lines.append(f"- feature count: {len(feature_cols)}")
    lines.append(f"- one-hot feature count: {one_hot_count}")
    lines.append(f"- binary feature count: {binary_count}")
    lines.append(f"- numeric feature count: {numeric_count}")
    lines.append(f"- leakage suspect feature count: {leakage_feature_count}")
    lines.append(f"- removed leakage columns: {json.dumps(leakage_cols, ensure_ascii=False)}")
    lines.append("")
    lines.append("sample_key duplicate checks:")
    for note in duplicate_notes:
        lines.append(f"- {note}")
    if not duplicate_notes:
        lines.append("- no input sample_key duplicates detected")
    for name, frame in [
        ("tau_eff_ml_dataset_step18.csv", dataset),
        ("tau_eff_ml_dataset_recommended_step18.csv", recommended),
        ("tau_eff_ml_feature_matrix_step18.csv", feature_matrix),
        ("tau_eff_ml_target_step18.csv", select_existing(dataset, ["sample_key"] + TARGET_COLUMNS)),
        ("tau_eff_ml_splits_step18.csv", splits),
    ]:
        dup = int(frame["sample_key"].duplicated().sum())
        lines.append(f"- {name}: duplicate sample_key rows={dup}")
    lines.append("")
    lines.append("Notes:")
    lines.append("Step18 does not train a machine learning model.")
    lines.append("Step18 prepares labels, features, metadata, and split information for tau_eff prediction.")
    lines.append("The target variable is target_log_tau_eff_step18 based on log_tau_eff_step12.")
    lines.append("tau_eff is a relative scale, not a physical relaxation time in seconds.")
    return "\n".join(lines) + "\n"


def format_counts(df, col, top_n=None):
    if col not in df.columns:
        return [f"  {col}: missing"]
    counts = df[col].fillna("missing").astype(str).value_counts(dropna=False)
    if top_n:
        counts = counts.head(top_n)
    return [f"  {idx}: {int(val)}" for idx, val in counts.items()]


def make_notes():
    return """# Step18 tau_eff ML Dataset Notes

## Purpose
Step18 prepares a one-row-per-sample machine learning dataset for predicting fitted tau_eff in Step19.

## Target Variable
The primary target is `target_log_tau_eff_step18`, copied from `log_tau_eff_step12`.
`target_tau_eff_step18` is retained for metadata and inspection.

## Input Features
Features include material metadata, final n/p annotations, additive and structure annotations, sintering annotations, nanocarbon flags, and regex-derived element indicators from composition.

## Data Exclusion Policy
Recommended ML rows require an available target, `fit_status_step12 == ok`, enough fitting rows, and fitting/validation errors within the configured thresholds. Rows that fail these checks remain in the full dataset with exclusion reasons.

## Train/Test Split Policy
Random splits are reproducible hashes of `sample_key`. DOI group splits hash DOI where available so samples from the same DOI stay in the same split; rows without DOI fall back to sample_key.

## Leakage Prevention
Fitting error columns, PF/ZT prediction results, and target columns are excluded from the feature matrix.

## Important Caveats
The target tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
Step18 does not train a model.
Step18 prepares the dataset for Step19.
Features derived from fitting errors or PF/ZT prediction results are not used as model inputs to avoid leakage.
DOI-based split is recommended for more reliable evaluation.

## Next Step
Step19 should train and evaluate models using `tau_eff_ml_dataset_recommended_step18.csv`, `tau_eff_ml_feature_matrix_step18.csv`, `tau_eff_ml_target_step18.csv`, and `tau_eff_ml_splits_step18.csv`.
"""


def print_done(output_counts, feature_count, target_counts, split_counts, doi_leakage, leakage_feature_count, sample_key_duplicates):
    created = [
        "tau_eff_ml_dataset_step18.csv",
        "tau_eff_ml_dataset_recommended_step18.csv",
        "tau_eff_ml_feature_matrix_step18.csv",
        "tau_eff_ml_target_step18.csv",
        "tau_eff_ml_metadata_step18.csv",
        "tau_eff_ml_splits_step18.csv",
        "tau_eff_ml_feature_dictionary_step18.csv",
        "tau_eff_ml_excluded_samples_step18.csv",
        "tau_eff_ml_data_quality_summary_step18.csv",
        "step18_tau_eff_ml_dataset_report.txt",
        "step18_tau_eff_ml_dataset_notes.md",
        "starrydata2_step18_tau_eff_ml_dataset.xlsx",
    ]
    print("Done.")
    print("Created:")
    for item in created:
        print(f"- {item}")
    print("")
    print("Summary:")
    print(f"all ML dataset samples: {output_counts.get('dataset', 0)}")
    print(f"recommended ML samples: {output_counts.get('recommended', 0)}")
    print(f"excluded samples: {output_counts.get('excluded', 0)}")
    print(f"feature matrix rows: {output_counts.get('feature_matrix', 0)}")
    print(f"feature count: {feature_count}")
    print(f"target available samples: {target_counts.get('available', 0)}")
    print(f"target high quality samples: {target_counts.get('high', 0)}")
    print(f"target medium quality samples: {target_counts.get('medium', 0)}")
    print(f"target low quality samples: {target_counts.get('low', 0)}")
    print(f"random train/test samples: {split_counts.get('random_train', 0)}/{split_counts.get('random_test', 0)}")
    print(f"DOI group train/test samples: {split_counts.get('doi_train', 0)}/{split_counts.get('doi_test', 0)}")
    print(f"DOI leakage count: {doi_leakage}")
    print(f"leakage feature count: {leakage_feature_count}")
    print(f"sample_key duplicates: {sample_key_duplicates}")


def main():
    args = parse_args()
    step12_dir = Path(args.step12_dir)
    step13_dir = Path(args.step13_dir)
    step15_dir = Path(args.step15_dir)
    step16_dir = Path(args.step16_dir)
    step17_dir = Path(args.step17_dir)
    step9_dir = Path(args.step9_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    duplicate_notes = []

    step12 = read_csv(step12_dir / REQUIRED_FILES["step12"], required=True)
    step13 = read_csv(step13_dir / REQUIRED_FILES["step13"], required=True)
    step15 = read_csv(step15_dir / REQUIRED_FILES["step15"], required=True)
    step17 = read_csv(step17_dir / REQUIRED_FILES["step17"], required=True)
    step9 = read_csv(step9_dir / OPTIONAL_FILES["step9"], required=False)
    step17_base = read_csv(step17_dir / OPTIONAL_FILES["step17_base"], required=False)
    step16_targets = read_csv(step16_dir / OPTIONAL_FILES["step16_targets"], required=False)
    best_candidates = read_csv(step15_dir / OPTIONAL_FILES["best_candidates"], required=False)
    manual_candidates = read_csv(step15_dir / OPTIONAL_FILES["manual_candidates"], required=False)
    sintering_candidates = read_csv(step15_dir / OPTIONAL_FILES["sintering_candidates"], required=False)

    input_counts = {
        "step12": len(step12),
        "step13": len(step13),
        "step15": len(step15),
        "step17": len(step17),
        "step16_targets": len(step16_targets) if step16_targets is not None else 0,
        "step9": len(step9) if step9 is not None else 0,
    }

    dataset = first_by_sample_key(step12, "step12", duplicate_notes)
    dataset = merge_supplement(dataset, step13, "step13", duplicate_notes)
    dataset = merge_supplement(dataset, step15, "step15", duplicate_notes)
    dataset = merge_supplement(dataset, step9, "step9", duplicate_notes)
    dataset = merge_supplement(dataset, step16_targets, "step16_targets", duplicate_notes)
    dataset = merge_supplement(dataset, step17_base, "step17_base", duplicate_notes)
    dataset = merge_supplement(dataset, step17, "step17", duplicate_notes)
    dataset = merge_supplement(dataset, best_candidates, "best_candidates", duplicate_notes)
    dataset = merge_supplement(dataset, manual_candidates, "manual_candidates", duplicate_notes)
    dataset = merge_supplement(dataset, sintering_candidates, "sintering_candidates", duplicate_notes)

    # Prefer Step17 final annotations, then Step17 base, Step9/manual/auto values, then existing Step12 values.
    coalesce_specs = {
        "n_or_p_final_step17": ["n_or_p_final_step17", "n_or_p_final_step17__step17_base", "n_or_p_final_step17__step17", "n_or_p", "n_or_p_step6"],
        "n_or_p_confidence_final_step17": ["n_or_p_confidence_final_step17", "n_or_p_confidence_final_step17__step17_base", "n_or_p_confidence_final_step17__step17", "n_or_p_confidence_step6"],
        "additive_final_step17": ["additive_final_step17", "additive_final_step17__step17_base", "additive_final_step17__step17", "additive_paper_manual_step17__step17", "additive_manual_step9", "additive_auto_step9"],
        "additive_source_step17": ["additive_source_step17", "additive_source_step17__step17_base", "additive_source_step17__step17"],
        "additive_confidence_final_step17": ["additive_confidence_final_step17", "additive_confidence_final_step17__step17_base", "additive_confidence_final_step17__step17", "additive_confidence_step17__step17"],
        "structure_final_step17": ["structure_final_step17", "structure_final_step17__step17_base", "structure_final_step17__step17", "structure_paper_manual_step17__step17", "structure_manual_step9", "structure_auto_step9"],
        "structure_source_step17": ["structure_source_step17", "structure_source_step17__step17_base", "structure_source_step17__step17"],
        "structure_confidence_final_step17": ["structure_confidence_final_step17", "structure_confidence_final_step17__step17_base", "structure_confidence_final_step17__step17", "structure_confidence_step17__step17"],
        "sintering_method_final_step17": ["sintering_method_final_step17", "sintering_method_final_step17__step17_base", "sintering_method_final_step17__step17", "sintering_method_paper_manual_step17__step17", "sintering_method"],
        "sintering_checked_final_step17": ["sintering_checked_final_step17", "sintering_checked_final_step17__step17_base", "sintering_checked_final_step17__step17", "sintering_checked"],
        "sintering_source_step17": ["sintering_source_step17", "sintering_source_step17__step17_base", "sintering_source_step17__step17"],
        "sintering_confidence_final_step17": ["sintering_confidence_final_step17", "sintering_confidence_final_step17__step17_base", "sintering_confidence_final_step17__step17", "sintering_confidence_step17__step17"],
        "paper_checked_step17": ["paper_checked_step17", "paper_checked_step17__step17_base", "paper_checked_step17__step17", "paper_checked_step9"],
        "manual_review_status_step17": ["manual_review_status_step17", "manual_review_status_step17__step17_base", "manual_review_status_step17__step17"],
    }
    for target, candidates in coalesce_specs.items():
        dataset = coalesce_columns(dataset, target, candidates, default="unknown")

    ensure_columns(dataset, FEATURE_CATEGORICAL_RAW + BINARY_FEATURE_COLUMNS + METADATA_COLUMNS, default=np.nan)
    dataset = build_target_flags(
        dataset,
        min_fit_rows=args.min_fit_rows,
        max_sigma_fit_log_rmse=args.max_sigma_fit_log_rmse,
        max_validation_log_rmse=args.max_validation_log_rmse,
        use_only_fit_ok=args.use_only_fit_ok,
    )
    dataset = add_element_features(dataset)

    feature_matrix, removed_leakage_cols = build_feature_matrix(dataset, args.top_category_limit)
    splits, doi_missing, doi_leakage = build_splits(dataset)

    target = select_existing(dataset, ["sample_key"] + TARGET_COLUMNS)
    metadata = select_existing(dataset, METADATA_COLUMNS)
    excluded = select_existing(
        dataset[~dataset["use_for_tau_eff_ml_step18"].astype(bool)].copy(),
        [
            "sample_key",
            "composition",
            "material_system",
            "n_or_p",
            "fit_status_step12",
            "n_fit_rows_step12",
            "sigma_fit_log_rmse_step12",
            "validation_sigma_log_rmse_step13",
            "target_available_step18",
            "target_quality_step18",
            "ml_exclusion_reason_step18",
        ],
    )

    recommended = dataset[
        dataset["use_for_tau_eff_ml_step18"].astype(bool)
        & dataset["target_available_step18"].astype(bool)
        & dataset["target_quality_step18"].isin(["high", "medium", "low"])
    ].copy()

    data_quality_summary = build_data_quality_summary(dataset, splits)
    feature_dictionary = build_feature_dictionary(dataset, feature_matrix)

    output_counts = {
        "dataset": len(dataset),
        "recommended": len(recommended),
        "feature_matrix": len(feature_matrix),
        "target": len(target),
        "metadata": len(metadata),
        "splits": len(splits),
        "excluded": len(excluded),
    }
    report = make_report(
        input_counts,
        output_counts,
        dataset,
        recommended,
        feature_matrix,
        splits,
        duplicate_notes,
        doi_missing,
        doi_leakage,
        removed_leakage_cols,
    )
    notes = make_notes()

    dataset.to_csv(output_dir / "tau_eff_ml_dataset_step18.csv", index=False)
    recommended.to_csv(output_dir / "tau_eff_ml_dataset_recommended_step18.csv", index=False)
    feature_matrix.to_csv(output_dir / "tau_eff_ml_feature_matrix_step18.csv", index=False)
    target.to_csv(output_dir / "tau_eff_ml_target_step18.csv", index=False)
    metadata.to_csv(output_dir / "tau_eff_ml_metadata_step18.csv", index=False)
    splits.to_csv(output_dir / "tau_eff_ml_splits_step18.csv", index=False)
    feature_dictionary.to_csv(output_dir / "tau_eff_ml_feature_dictionary_step18.csv", index=False)
    excluded.to_csv(output_dir / "tau_eff_ml_excluded_samples_step18.csv", index=False)
    data_quality_summary.to_csv(output_dir / "tau_eff_ml_data_quality_summary_step18.csv", index=False)
    (output_dir / "step18_tau_eff_ml_dataset_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step18_tau_eff_ml_dataset_notes.md").write_text(notes, encoding="utf-8")

    write_excel(
        output_dir / "starrydata2_step18_tau_eff_ml_dataset.xlsx",
        {
            "ml_dataset_recommended": recommended,
            "feature_matrix": feature_matrix,
            "target": target,
            "metadata": metadata,
            "splits": splits,
            "feature_dictionary": feature_dictionary,
            "excluded_samples": excluded,
            "data_quality_summary": data_quality_summary,
            "dataset_report": report,
        },
    )

    feature_cols = [c for c in feature_matrix.columns if c != "sample_key"]
    split_counts = splits["split_random_80_20_step18"].value_counts().to_dict()
    doi_counts = splits["split_doi_group_80_20_step18"].value_counts().to_dict()
    quality_counts = dataset["target_quality_step18"].value_counts().to_dict()
    sample_key_duplicates = int(dataset["sample_key"].duplicated().sum())
    leakage_feature_count = sum(is_leakage_column(c) for c in feature_cols)
    print_done(
        output_counts,
        len(feature_cols),
        {
            "available": int(dataset["target_available_step18"].sum()),
            "high": int(quality_counts.get("high", 0)),
            "medium": int(quality_counts.get("medium", 0)),
            "low": int(quality_counts.get("low", 0)),
        },
        {
            "random_train": int(split_counts.get("train", 0)),
            "random_test": int(split_counts.get("test", 0)),
            "doi_train": int(doi_counts.get("train", 0)),
            "doi_test": int(doi_counts.get("test", 0)),
        },
        doi_leakage,
        leakage_feature_count,
        sample_key_duplicates,
    )


if __name__ == "__main__":
    main()
