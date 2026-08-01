import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl.styles import Font
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_STEP18_DIR = "data/output/starrydata2_step18_tau_eff_ml_dataset"
DEFAULT_STEP17_DIR = "data/output/starrydata2_step17_literature_review"
DEFAULT_STEP23_DIR = "data/output/starrydata2_step23_error_cause_analysis"
DEFAULT_STEP19_DIR = "data/output/starrydata2_step19_tau_eff_ml_model"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step30_info_rich_tau_eff_ml"
EXCEL_PREVIEW_ROWS = 100_000
RANDOM_STATE = 42
MIN_SAMPLES = 200
MIN_PAPERS = 20

STRING_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "doi_url",
    "paper_id",
    "sample_id",
    "paper_title",
    "composition",
    "material_system",
    "n_or_p",
]

UNKNOWN_VALUES = {
    "",
    "nan",
    "<na>",
    "unknown",
    "none",
    "n/a",
    "na",
    "unchecked",
    "not checked",
    "not_checked",
    "not_reported",
}

LEAKAGE_PATTERNS = [
    "target",
    "tau_eff",
    "log_tau",
    "sigma",
    "power_factor",
    "pf_",
    "zt_",
    "pred",
    "prediction",
    "error",
    "residual",
    "rmse",
    "mae",
    "mape",
    "r2",
    "holdout",
    "validation",
    "fit_status",
    "fit_note",
    "split_",
    "quality",
    "exclusion",
    "comparison",
    "rank",
    "score",
]

LEVEL_DEFINITIONS = [
    (0, "all_recommended"),
    (1, "basic_material_info"),
    (2, "any_extra_info"),
    (3, "two_extra_info"),
    (4, "full_extra_info"),
]


class MeanRegressor:
    def __init__(self):
        self.mean_ = 0.0

    def fit(self, X, y):
        self.mean_ = float(np.nanmean(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_, dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step30: evaluate tau_eff ML on information-rich sample subsets."
    )
    parser.add_argument("--step18_dir", default=DEFAULT_STEP18_DIR)
    parser.add_argument("--step17_dir", default=DEFAULT_STEP17_DIR)
    parser.add_argument("--step23_dir", default=DEFAULT_STEP23_DIR)
    parser.add_argument("--step19_dir", default=DEFAULT_STEP19_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random_state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min_samples", type=int, default=MIN_SAMPLES)
    parser.add_argument("--min_papers", type=int, default=MIN_PAPERS)
    return parser.parse_args()


def dtype_for_existing(path):
    header = pd.read_csv(path, nrows=0)
    return {col: "string" for col in STRING_COLUMNS if col in header.columns}


def resolve_file(directory, preferred_name, fallback_tokens):
    directory = Path(directory)
    preferred = directory / preferred_name
    if preferred.exists():
        return preferred
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {directory}")
    files = list(directory.glob("*.csv"))
    token_re = [re.compile(re.escape(token), re.IGNORECASE) for token in fallback_tokens]
    scored = []
    for path in files:
        score = sum(1 for token in token_re if token.search(path.name))
        if score:
            scored.append((score, len(path.name), path))
    if not scored:
        raise FileNotFoundError(f"Could not find {preferred_name} or close CSV in {directory}")
    scored.sort(key=lambda x: (-x[0], x[1], x[2].name))
    return scored[0][2]


def read_csv(path, required=True, usecols=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input file not found: {path}")
        return None
    header = pd.read_csv(path, nrows=0)
    kwargs = {"dtype": dtype_for_existing(path), "low_memory": False}
    if usecols:
        kwargs["usecols"] = [col for col in usecols if col in header.columns]
    return pd.read_csv(path, **kwargs)


def assert_unique_sample_key(df, name):
    if "sample_key" not in df.columns:
        raise ValueError(f"{name} missing sample_key")
    dup = int(df["sample_key"].duplicated().sum())
    if dup:
        raise ValueError(f"{name} has duplicate sample_key rows: {dup}")


def first_by_key(df):
    if df is None or "sample_key" not in df.columns:
        return None
    return df.drop_duplicates("sample_key", keep="first").copy()


def unknown_scalar(value, np_mixed_unknown=False):
    if pd.isna(value):
        return True
    text = str(value).strip()
    if text.lower() in UNKNOWN_VALUES:
        return True
    if np_mixed_unknown and text.lower() == "mixed":
        return True
    return False


def known_series(series, np_mixed_unknown=False):
    return ~series.map(lambda x: unknown_scalar(x, np_mixed_unknown=np_mixed_unknown))


def normalize_np(value):
    if unknown_scalar(value, np_mixed_unknown=True):
        return ""
    text = str(value).strip().lower()
    if text in {"n", "n-type", "n type", "ntype"}:
        return "n"
    if text in {"p", "p-type", "p type", "ptype"}:
        return "p"
    return ""


def coalesce_columns(df, candidates, default="unknown", np_mixed_unknown=False):
    out = pd.Series(default, index=df.index, dtype=object)
    filled = pd.Series(False, index=df.index)
    for col in candidates:
        if col not in df.columns:
            continue
        values = df[col]
        ok = known_series(values, np_mixed_unknown=np_mixed_unknown)
        out = out.where(filled | ~ok, values)
        filled = filled | ok
    return out


def infer_material_system_from_composition(value):
    if unknown_scalar(value):
        return "unknown"
    text = str(value).strip()
    lower = text.lower()
    if any(token in lower for token in ["polyaniline", "pedot", "polymer"]):
        return "polymer"
    elements = set(re.findall(r"[A-Z][a-z]?", text))
    if not elements:
        return "unknown"
    if {"Co", "Sb"}.issubset(elements):
        return "skutterudite"
    if "Te" in elements and ({"Bi", "Sb"} & elements):
        return "bi_sb_te"
    if "Pb" in elements and ("Te" in elements or "Se" in elements or "S" in elements):
        return "pb_chalcogenide"
    if "Si" in elements and "Ge" in elements:
        return "si_ge"
    if "Mg" in elements and {"Si", "Sn", "Ge"} & elements:
        return "mg_silicide_stannide_germanide"
    if "Zn" in elements and "Sb" in elements:
        return "zn_antimonide"
    if "O" in elements:
        return "oxide"
    if {"S", "Se", "Te"} & elements:
        return "chalcogenide"
    if {"Sb", "Bi", "As", "P", "N"} & elements:
        return "pnictide"
    if "C" in elements and len(elements) <= 3:
        return "carbon_based"
    return "_".join(sorted(elements)[:4]).lower()


def merge_optional(base, df, columns, suffix):
    df = first_by_key(df)
    if df is None:
        return base
    keep = ["sample_key"] + [col for col in columns if col in df.columns and col != "sample_key"]
    sub = df[keep].copy()
    rename = {col: f"{col}__{suffix}" for col in sub.columns if col != "sample_key" and col in base.columns}
    sub = sub.rename(columns=rename)
    return base.merge(sub, on="sample_key", how="left")


def is_leakage_column(col):
    if col == "sample_key":
        return False
    lower = col.lower()
    return any(pattern in lower for pattern in LEAKAGE_PATTERNS)


def prepare_feature_matrix(feature_df):
    audit_rows = []
    candidate_cols = [col for col in feature_df.columns if col != "sample_key"]
    leak_cols = [col for col in candidate_cols if is_leakage_column(col)]
    safe_cols = [col for col in candidate_cols if col not in leak_cols]
    X = feature_df[safe_cols].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
        else:
            low = X[col].astype(str).str.strip().str.lower()
            values = set(low.dropna().unique())
            bool_values = {"true", "false", "1", "0", "yes", "no", "nan", "<na>", ""}
            if values.issubset(bool_values):
                X[col] = low.map({"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0})
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    all_missing_cols = [col for col in X.columns if X[col].isna().all()]
    X = X.drop(columns=all_missing_cols)
    medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    zero_variance_cols = X.columns[X.var(axis=0) <= 0].tolist()
    X = X.drop(columns=zero_variance_cols)

    for col in candidate_cols:
        if col in leak_cols:
            status = "excluded_leakage_pattern"
        elif col in all_missing_cols:
            status = "excluded_all_missing"
        elif col in zero_variance_cols:
            status = "excluded_zero_variance"
        elif col in X.columns:
            status = "used"
        else:
            status = "excluded"
        audit_rows.append({"feature_name": col, "status_step30": status, "used_in_model_step30": status == "used"})
    return X, pd.DataFrame(audit_rows), leak_cols


def choose_paper_identifier(data):
    candidates = ["DOI", "DOI__recommended", "DOI__metadata", "doi_url", "doi_url__metadata", "paper_id", "SID", "SID__metadata"]
    for col in candidates:
        if col in data.columns:
            s = data[col].astype("string").fillna("").str.strip()
            known = s[~s.str.lower().isin(UNKNOWN_VALUES)]
            if known.nunique() >= MIN_PAPERS:
                return col
    return None


def paper_id_series(data, column):
    if column is None:
        return pd.Series("", index=data.index, dtype=object)
    s = data[column].astype("string").fillna("").str.strip()
    s = s.mask(s.str.lower().isin(UNKNOWN_VALUES), "")
    return s


def safe_paper_count(series):
    s = series.astype("string").fillna("").str.strip()
    s = s[~s.str.lower().isin(UNKNOWN_VALUES) & s.ne("")]
    return int(s.nunique())


def add_info_flags(data):
    data = data.copy()
    data["composition_info_step30"] = coalesce_columns(
        data, ["composition", "composition__metadata", "composition__recommended", "composition__step17", "composition__step23"]
    )
    data["material_system_info_step30"] = coalesce_columns(
        data,
        [
            "material_system",
            "material_system__metadata",
            "material_system__recommended",
            "material_system__step17",
            "material_system__step23",
        ],
    )
    inferred_material = data["composition_info_step30"].map(infer_material_system_from_composition)
    data["material_system_source_step30"] = np.where(
        known_series(data["material_system_info_step30"]), "existing_material_system", "inferred_from_composition"
    )
    data["material_system_info_step30"] = data["material_system_info_step30"].where(
        known_series(data["material_system_info_step30"]), inferred_material
    )
    np_source = coalesce_columns(
        data,
        [
            "n_or_p_final_step17",
            "n_or_p_final_step17__metadata",
            "n_or_p_final_step17__step17",
            "n_or_p_final_step17__step23",
            "n_or_p",
            "n_or_p__metadata",
            "n_or_p__recommended",
            "n_or_p__step17",
            "n_or_p__step23",
        ],
        np_mixed_unknown=True,
    )
    data["n_or_p_info_step30"] = np_source.map(normalize_np)
    data["additive_info_step30"] = coalesce_columns(
        data,
        [
            "additive_final_step17",
            "additive_final_step17__step17",
            "additive_final_step17__step23",
            "additive_manual_step9",
            "additive_manual_step9__step17",
            "additive_auto_step9",
            "additive_auto_step9__step17",
        ],
    )
    data["structure_info_step30"] = coalesce_columns(
        data,
        [
            "structure_final_step17",
            "structure_final_step17__step17",
            "structure_final_step17__step23",
            "structure_manual_step9",
            "structure_manual_step9__step17",
            "structure_auto_step9",
            "structure_auto_step9__step17",
        ],
    )
    data["sintering_info_step30"] = coalesce_columns(
        data,
        [
            "sintering_method_final_step17",
            "sintering_method_final_step17__metadata",
            "sintering_method_final_step17__step17",
            "sintering_method_final_step17__step23",
            "sintering_method",
            "sintering_method__metadata",
            "sintering_method__recommended",
            "sintering_method__step17",
            "sintering_method__step23",
            "sintering_condition_final_step17",
            "sintering_condition_final_step17__step17",
            "sintering_condition_final_step17__step23",
        ],
    )

    data["has_composition_step30"] = known_series(data["composition_info_step30"])
    data["has_material_system_step30"] = known_series(data["material_system_info_step30"])
    data["has_np_type_step30"] = data["n_or_p_info_step30"].isin(["n", "p"])
    data["has_additive_info_step30"] = known_series(data["additive_info_step30"])
    data["has_structure_info_step30"] = known_series(data["structure_info_step30"])
    data["has_sintering_info_step30"] = known_series(data["sintering_info_step30"])
    data["extra_info_count_step30"] = data[
        ["has_additive_info_step30", "has_structure_info_step30", "has_sintering_info_step30"]
    ].sum(axis=1)

    level1 = data["has_composition_step30"] & data["has_material_system_step30"] & data["has_np_type_step30"]
    data["is_level0_all_recommended_step30"] = True
    data["is_level1_basic_material_info_step30"] = level1
    data["is_level2_any_extra_info_step30"] = level1 & (data["extra_info_count_step30"] >= 1)
    data["is_level3_two_extra_info_step30"] = level1 & (data["extra_info_count_step30"] >= 2)
    data["is_level4_full_extra_info_step30"] = level1 & (data["extra_info_count_step30"] >= 3)

    level = np.zeros(len(data), dtype=int)
    level = np.where(data["is_level1_basic_material_info_step30"], 1, level)
    level = np.where(data["is_level2_any_extra_info_step30"], 2, level)
    level = np.where(data["is_level3_two_extra_info_step30"], 3, level)
    level = np.where(data["is_level4_full_extra_info_step30"], 4, level)
    data["info_level_step30"] = level
    data["info_level_name_step30"] = data["info_level_step30"].map(dict(LEVEL_DEFINITIONS))
    return data


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan}
    spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman") if len(y_true) > 1 else np.nan
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "spearman": float(spearman) if pd.notna(spearman) else np.nan,
    }


def make_models(random_state):
    return {
        "baseline_mean": MeanRegressor(),
        "ridge_regression": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        "random_forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=1
        ),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
    }


def split_indices(level_df, split_method, random_state, paper_col):
    idx = np.arange(len(level_df))
    if split_method == "sample_random":
        train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=random_state)
        return train_idx, test_idx, ""
    if paper_col is None:
        return None, None, "paper identifier column was not available"
    groups = paper_id_series(level_df, paper_col)
    if (groups == "").any():
        return None, None, f"paper identifier {paper_col} contains missing values"
    if groups.nunique() < 2:
        return None, None, f"paper identifier {paper_col} has fewer than 2 groups"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(level_df, groups=groups))
    return train_idx, test_idx, ""


def evaluate_level(level_id, level_name, level_df, X_all, y_all, split_method, random_state, paper_col):
    train_idx, test_idx, skip_reason = split_indices(level_df, split_method, random_state + level_id, paper_col)
    if skip_reason:
        return [], pd.DataFrame(), skip_reason

    level_indices = level_df.index.to_numpy()
    train_abs = level_indices[train_idx]
    test_abs = level_indices[test_idx]
    X_train = X_all.loc[train_abs]
    X_test = X_all.loc[test_abs]
    y_train = y_all.loc[train_abs]
    y_test = y_all.loc[test_abs]

    if len(train_abs) == 0 or len(test_abs) == 0:
        return [], pd.DataFrame(), "train or test split is empty"

    paper_train = paper_id_series(level_df.iloc[train_idx], paper_col) if paper_col else pd.Series("", index=train_idx)
    paper_test = paper_id_series(level_df.iloc[test_idx], paper_col) if paper_col else pd.Series("", index=test_idx)
    train_papers = set(paper_train[paper_train.ne("")])
    test_papers = set(paper_test[paper_test.ne("")])
    doi_leakage = len(train_papers & test_papers) if split_method == "paper_group" else np.nan

    rows = []
    pred_frames = []
    for model_name, model in make_models(random_state + level_id).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = metric_dict(y_test, pred)
        rows.append(
            {
                "info_level_step30": level_id,
                "info_level_name_step30": level_name,
                "model_name": model_name,
                "split_method_step30": split_method,
                "rmse_log_tau_eff": metrics["rmse"],
                "mae_log_tau_eff": metrics["mae"],
                "r2_log_tau_eff": metrics["r2"],
                "spearman_log_tau_eff": metrics["spearman"],
                "train_sample_count": int(len(train_abs)),
                "test_sample_count": int(len(test_abs)),
                "train_doi_count": len(train_papers),
                "test_doi_count": len(test_papers),
                "doi_leakage_count": doi_leakage,
                "paper_identifier_column": paper_col or "",
                "skip_reason": "",
            }
        )
        pred_df = level_df.loc[test_abs, ["sample_key"]].copy()
        pred_df["info_level_step30"] = level_id
        pred_df["info_level_name_step30"] = level_name
        pred_df["model_name"] = model_name
        pred_df["split_method_step30"] = split_method
        pred_df["observed_log_tau_eff"] = y_test.values
        pred_df["predicted_log_tau_eff"] = pred
        pred_df["error_log_tau_eff"] = pred_df["predicted_log_tau_eff"] - pred_df["observed_log_tau_eff"]
        if paper_col and paper_col in level_df.columns:
            pred_df["paper_identifier_step30"] = level_df.loc[test_abs, paper_col].values
        pred_frames.append(pred_df)
    return rows, pd.concat(pred_frames, ignore_index=True), ""


def level_mask(data, level_id):
    if level_id == 0:
        return pd.Series(True, index=data.index)
    return data["info_level_step30"] >= level_id


def make_level_counts(data, paper_col, min_samples, min_papers):
    rows = []
    for level_id, level_name in LEVEL_DEFINITIONS:
        subset = data[level_mask(data, level_id)]
        paper_count = safe_paper_count(paper_id_series(subset, paper_col)) if paper_col else 0
        reasons = []
        if len(subset) < min_samples:
            reasons.append(f"sample_count<{min_samples}")
        if paper_count < min_papers:
            reasons.append(f"paper_count<{min_papers}")
        target = pd.to_numeric(subset["target_log_tau_eff_step18"], errors="coerce")
        rows.append(
            {
                "info_level_step30": level_id,
                "info_level_name_step30": level_name,
                "sample_count": int(len(subset)),
                "doi_count": int(paper_count),
                "material_system_known_rate": float(subset["has_material_system_step30"].mean()) if len(subset) else np.nan,
                "np_type_known_rate": float(subset["has_np_type_step30"].mean()) if len(subset) else np.nan,
                "additive_known_rate": float(subset["has_additive_info_step30"].mean()) if len(subset) else np.nan,
                "structure_known_rate": float(subset["has_structure_info_step30"].mean()) if len(subset) else np.nan,
                "sintering_known_rate": float(subset["has_sintering_info_step30"].mean()) if len(subset) else np.nan,
                "log_tau_eff_mean": float(target.mean()) if len(subset) else np.nan,
                "log_tau_eff_std": float(target.std()) if len(subset) else np.nan,
                "log_tau_eff_median": float(target.median()) if len(subset) else np.nan,
                "excluded_or_skip_reason": "; ".join(reasons),
                "is_trainable_step30": len(reasons) == 0,
            }
        )
    return pd.DataFrame(rows)


def make_missing_summary(data):
    rows = []
    fields = [
        ("composition", "has_composition_step30"),
        ("material_system", "has_material_system_step30"),
        ("n_or_p", "has_np_type_step30"),
        ("additive", "has_additive_info_step30"),
        ("structure", "has_structure_info_step30"),
        ("sintering", "has_sintering_info_step30"),
    ]
    for level_id, level_name in LEVEL_DEFINITIONS:
        subset = data[level_mask(data, level_id)]
        for field, flag in fields:
            known = int(subset[flag].sum()) if len(subset) else 0
            rows.append(
                {
                    "info_level_step30": level_id,
                    "info_level_name_step30": level_name,
                    "information_field": field,
                    "sample_count": int(len(subset)),
                    "known_count": known,
                    "missing_count": int(len(subset) - known),
                    "known_rate": float(known / len(subset)) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def select_best_models(comparison):
    rows = []
    for level_id, level_name in LEVEL_DEFINITIONS:
        sub = comparison[(comparison["info_level_step30"] == level_id) & comparison["skip_reason"].eq("")]
        if sub.empty:
            rows.append(
                {
                    "info_level_step30": level_id,
                    "info_level_name_step30": level_name,
                    "representative_model": "",
                    "selection_split_method": "",
                    "selection_criterion": "",
                    "rmse_log_tau_eff": np.nan,
                    "mae_log_tau_eff": np.nan,
                    "r2_log_tau_eff": np.nan,
                    "spearman_log_tau_eff": np.nan,
                    "selection_note": "not evaluated",
                }
            )
            continue
        paper_sub = sub[sub["split_method_step30"].eq("paper_group")]
        if not paper_sub.empty:
            cand = paper_sub.sort_values("rmse_log_tau_eff", na_position="last").iloc[0]
            criterion = "lowest paper_group RMSE"
        else:
            cand = sub[sub["split_method_step30"].eq("sample_random")].sort_values("rmse_log_tau_eff", na_position="last").iloc[0]
            criterion = "lowest sample_random RMSE because paper_group was unavailable"
        rows.append(
            {
                "info_level_step30": level_id,
                "info_level_name_step30": level_name,
                "representative_model": cand["model_name"],
                "selection_split_method": cand["split_method_step30"],
                "selection_criterion": criterion,
                "rmse_log_tau_eff": cand["rmse_log_tau_eff"],
                "mae_log_tau_eff": cand["mae_log_tau_eff"],
                "r2_log_tau_eff": cand["r2_log_tau_eff"],
                "spearman_log_tau_eff": cand["spearman_log_tau_eff"],
                "train_sample_count": cand["train_sample_count"],
                "test_sample_count": cand["test_sample_count"],
                "train_doi_count": cand["train_doi_count"],
                "test_doi_count": cand["test_doi_count"],
                "doi_leakage_count": cand["doi_leakage_count"],
                "selection_note": "",
            }
        )
    return pd.DataFrame(rows)


def make_interpretation(level_counts, best_summary):
    rows = []
    base = best_summary[best_summary["info_level_step30"].eq(0)]
    base_rmse = float(base["rmse_log_tau_eff"].iloc[0]) if not base.empty and pd.notna(base["rmse_log_tau_eff"].iloc[0]) else np.nan
    base_spear = (
        float(base["spearman_log_tau_eff"].iloc[0])
        if not base.empty and pd.notna(base["spearman_log_tau_eff"].iloc[0])
        else np.nan
    )
    for _, count_row in level_counts.iterrows():
        level_id = int(count_row["info_level_step30"])
        best = best_summary[best_summary["info_level_step30"].eq(level_id)]
        if best.empty or pd.isna(best["rmse_log_tau_eff"].iloc[0]):
            perf = "skipped"
            rmse_delta = np.nan
            spear_delta = np.nan
        else:
            rmse = float(best["rmse_log_tau_eff"].iloc[0])
            spear = float(best["spearman_log_tau_eff"].iloc[0]) if pd.notna(best["spearman_log_tau_eff"].iloc[0]) else np.nan
            rmse_delta = rmse - base_rmse if pd.notna(base_rmse) else np.nan
            spear_delta = spear - base_spear if pd.notna(base_spear) and pd.notna(spear) else np.nan
            if level_id == 0:
                perf = "baseline"
            elif pd.notna(rmse_delta) and rmse_delta < 0:
                perf = "RMSE improved versus Level 0"
            elif pd.notna(rmse_delta) and rmse_delta > 0:
                perf = "RMSE worsened versus Level 0"
            else:
                perf = "RMSE change not evaluable"
        data_note = "sufficient by Step30 threshold" if count_row["is_trainable_step30"] else "limited data; evaluation skipped"
        rows.append(
            {
                "info_level_step30": level_id,
                "info_level_name_step30": count_row["info_level_name_step30"],
                "performance_interpretation": perf,
                "rmse_delta_vs_level0": rmse_delta,
                "spearman_delta_vs_level0": spear_delta,
                "data_sufficiency_interpretation": data_note,
                "short_note": (
                    "Information-rich restriction is associated with lower RMSE."
                    if level_id > 0 and pd.notna(rmse_delta) and rmse_delta < 0
                    else "No RMSE improvement was observed; reduced sample size and other error sources remain plausible."
                    if level_id > 0 and pd.notna(rmse_delta)
                    else "Reference level or skipped level."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"text": data.splitlines()})
            data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(len(v) for v in values) + 2, 70)


def save_figure(fig, figures_dir, stem):
    png = figures_dir / f"{stem}.png"
    pdf = figures_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [
        {"figure_name": stem, "format": "png", "path": str(png)},
        {"figure_name": stem, "format": "pdf", "path": str(pdf)},
    ]


def make_step30_figures(level_counts, comparison, best_summary, missing_summary, predictions, output_dir):
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    made = []

    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
    level_labels = [f"L{int(x)}" for x in level_counts["info_level_step30"]]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(level_counts))
    width = 0.36
    ax1.bar(x - width / 2, level_counts["sample_count"], width=width, color="#4C78A8", label="samples")
    ax1.set_ylabel("Sample count")
    ax1.set_yscale("symlog", linthresh=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(level_labels)
    ax1.set_xlabel("Information level")
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, level_counts["doi_count"], width=width, color="#F28E2B", label="DOI/paper count")
    ax2.set_ylabel("DOI/paper count")
    ax2.set_yscale("symlog", linthresh=10)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax1.set_title("Step30 Level Size")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    made.extend(save_figure(fig, figures_dir, "step30_level_sample_doi_counts"))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    rmse = best_summary["rmse_log_tau_eff"].astype(float)
    bar_colors = [colors[i % len(colors)] if pd.notna(v) else "#BDBDBD" for i, v in enumerate(rmse)]
    ax.bar(level_labels, rmse.fillna(0), color=bar_colors)
    ax.set_ylabel("RMSE of log_tau_eff")
    ax.set_xlabel("Information level")
    ax.set_title("Representative Model RMSE by Level")
    if rmse.notna().any():
        ax.set_ylim(0, float(rmse.max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(rmse):
        text = "skip" if pd.isna(v) else f"{v:.3f}"
        ax.text(i, 0 if pd.isna(v) else v, text, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    made.extend(save_figure(fig, figures_dir, "step30_representative_rmse_by_level"))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    spearman = best_summary["spearman_log_tau_eff"].astype(float)
    bar_colors = [colors[i % len(colors)] if pd.notna(v) else "#BDBDBD" for i, v in enumerate(spearman)]
    ax.bar(level_labels, spearman.fillna(0), color=bar_colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Spearman correlation")
    ax.set_xlabel("Information level")
    ax.set_title("Representative Model Rank Agreement by Level")
    if spearman.notna().any():
        low = min(0.0, float(spearman.min()) * 1.18)
        high = max(0.1, float(spearman.max()) * 1.18)
        ax.set_ylim(low, high)
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(spearman):
        text = "skip" if pd.isna(v) else f"{v:.3f}"
        ax.text(i, 0 if pd.isna(v) else v, text, ha="center", va="bottom" if pd.isna(v) or v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    made.extend(save_figure(fig, figures_dir, "step30_representative_spearman_by_level"))

    eval_comp = comparison[
        comparison["skip_reason"].fillna("").eq("")
        & comparison["split_method_step30"].eq("paper_group")
        & comparison["model_name"].notna()
        & comparison["model_name"].ne("")
    ].copy()
    if not eval_comp.empty:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        pivot = eval_comp.pivot(index="model_name", columns="info_level_step30", values="rmse_log_tau_eff")
        pivot = pivot.reindex(["baseline_mean", "ridge_regression", "random_forest", "extra_trees", "gradient_boosting"])
        pivot.plot(kind="bar", ax=ax, color=colors[: len(pivot.columns)])
        ax.set_ylabel("RMSE of log_tau_eff")
        ax.set_xlabel("Model")
        ax.set_title("Paper-Group RMSE by Model and Level")
        ax.legend(title="Level")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        made.extend(save_figure(fig, figures_dir, "step30_paper_group_model_rmse"))

    heat = missing_summary.pivot_table(
        index="info_level_step30", columns="information_field", values="known_rate", aggfunc="first"
    )
    heat = heat.reindex(columns=["composition", "material_system", "n_or_p", "additive", "structure", "sintering"])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(heat.fillna(0).values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels([f"L{int(v)}" for v in heat.index])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            label = "" if pd.isna(val) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8)
    ax.set_title("Known Information Rate by Level")
    fig.colorbar(im, ax=ax, label="Known rate")
    fig.tight_layout()
    made.extend(save_figure(fig, figures_dir, "step30_known_information_rate_heatmap"))

    if not predictions.empty:
        plot_pred = predictions.copy()
        plot_pred["observed_log_tau_eff"] = pd.to_numeric(plot_pred["observed_log_tau_eff"], errors="coerce")
        plot_pred["predicted_log_tau_eff"] = pd.to_numeric(plot_pred["predicted_log_tau_eff"], errors="coerce")
        plot_pred = plot_pred.dropna(subset=["observed_log_tau_eff", "predicted_log_tau_eff"])
        if not plot_pred.empty:
            fig, ax = plt.subplots(figsize=(6.2, 6.0))
            for level_id, group in plot_pred.groupby("info_level_step30"):
                ax.scatter(
                    group["observed_log_tau_eff"],
                    group["predicted_log_tau_eff"],
                    s=12,
                    alpha=0.35,
                    label=f"L{int(level_id)}",
                )
            low = float(min(plot_pred["observed_log_tau_eff"].min(), plot_pred["predicted_log_tau_eff"].min()))
            high = float(max(plot_pred["observed_log_tau_eff"].max(), plot_pred["predicted_log_tau_eff"].max()))
            ax.plot([low, high], [low, high], color="#333333", linewidth=1.0)
            ax.set_xlabel("Observed log_tau_eff")
            ax.set_ylabel("Predicted log_tau_eff")
            ax.set_title("Representative Test Predictions")
            ax.legend(title="Level")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            made.extend(save_figure(fig, figures_dir, "step30_observed_vs_predicted_log_tau"))

    return pd.DataFrame(made)


def make_report(level_counts, comparison, best_summary, interpretation, missing_summary, feature_audit, input_paths, paper_col):
    lines = []
    lines.append("Step30 information-rich tau_eff ML evaluation")
    lines.append("")
    lines.append("Purpose")
    lines.append("- Step30 is an additional evaluation restricted to samples with richer material information.")
    lines.append("- Step30 does not refit tau_eff.")
    lines.append("- Step30 does not recalculate sigma, PF, or ZT.")
    lines.append("- Step30 uses Step18 target_log_tau_eff_step18 as the objective variable.")
    lines.append("- Step30 does not causally prove why ML performance decreases.")
    lines.append("- Existing material_system values were all or mostly unknown, so material_system_info_step30 uses existing values when available and otherwise a coarse composition-derived Step30-only fallback. Original columns are not changed.")
    lines.append("")
    lines.append("Inputs")
    for label, path in input_paths.items():
        lines.append(f"- {label}: {path}")
    lines.append(f"- paper identifier for DOI-group split/counts: {paper_col or 'not available'}")
    lines.append("")
    lines.append("Leakage check")
    leak = feature_audit[feature_audit["status_step30"].eq("excluded_leakage_pattern")]
    lines.append(f"- leakage-suspect feature columns excluded: {len(leak)}")
    if len(leak):
        lines.append("- excluded columns: " + ", ".join(leak["feature_name"].head(80).astype(str)))
        if len(leak) > 80:
            lines.append(f"- additional excluded columns omitted from report: {len(leak) - 80}")
    lines.append("")
    lines.append("Level counts")
    for _, row in level_counts.iterrows():
        reason = row["excluded_or_skip_reason"] if isinstance(row["excluded_or_skip_reason"], str) and row["excluded_or_skip_reason"] else "evaluated"
        lines.append(
            f"- Level {int(row['info_level_step30'])} {row['info_level_name_step30']}: "
            f"samples={int(row['sample_count'])}, DOI/paper count={int(row['doi_count'])}, reason={reason}"
        )
    lines.append("")
    lines.append("Representative models")
    for _, row in best_summary.iterrows():
        if not row.get("representative_model"):
            lines.append(f"- Level {int(row['info_level_step30'])}: skipped ({row.get('selection_note', '')})")
            continue
        lines.append(
            f"- Level {int(row['info_level_step30'])} {row['info_level_name_step30']}: "
            f"{row['representative_model']} by {row['selection_criterion']}; "
            f"RMSE={row['rmse_log_tau_eff']:.4f}, Spearman={row['spearman_log_tau_eff']:.4f}"
        )
    lines.append("")
    lines.append("Interpretation")
    lines.append(
        "- Restricting to information-rich samples can improve feature quality, but it also reduces training data size."
    )
    lines.append(
        "- If performance improves, the result supports insufficient information as a candidate contributor to lower ML performance."
    )
    lines.append(
        "- If performance does not improve, tau_eff definition, C(T) construction, data variability, and extrapolation difficulty also need to be considered."
    )
    for _, row in interpretation.iterrows():
        lines.append(
            f"- Level {int(row['info_level_step30'])}: {row['performance_interpretation']}; "
            f"{row['data_sufficiency_interpretation']}"
        )
    lines.append("")
    lines.append("Missing information summary")
    pivot = missing_summary.pivot_table(
        index=["info_level_step30", "info_level_name_step30"], columns="information_field", values="known_rate"
    )
    for idx, row in pivot.iterrows():
        level_id, level_name = idx
        rates = ", ".join(f"{col}={row[col]:.3f}" for col in row.index if pd.notna(row[col]))
        lines.append(f"- Level {level_id} {level_name}: {rates}")
    lines.append("")
    lines.append("Validation")
    max_paper_leakage = comparison.loc[
        comparison["split_method_step30"].eq("paper_group") & comparison["skip_reason"].eq(""), "doi_leakage_count"
    ].max()
    lines.append("- sample_key duplicates were checked before model evaluation.")
    lines.append("- feature matrix was checked for target/evaluation/prediction/error leakage patterns.")
    lines.append(f"- DOI leakage count in evaluated paper-group splits: {max_paper_leakage if pd.notna(max_paper_leakage) else 'not evaluated'}")
    lines.append("- Levels below the sample or DOI/paper threshold were recorded as skipped rather than raising an error.")
    lines.append("- Existing n_or_p, sintering_method, sintering_checked, and record_checked values were not modified.")
    lines.append("- Outputs are written only under the Step30 output directory.")
    return "\n".join(lines) + "\n"


def make_notes(best_summary, interpretation):
    lines = []
    lines.append("# Step30 Information-Rich tau_eff ML Notes")
    lines.append("")
    lines.append("Step30 is an additional evaluation restricted to samples with richer material information.")
    lines.append("It uses the Step18 `log_tau_eff` target and does not refit `tau_eff` or recalculate sigma, PF, or ZT.")
    lines.append("")
    lines.append(
        "This analysis does not causally prove the reason for lower ML performance. Restricting to information-rich samples may improve feature quality, but it also reduces the number of training samples."
    )
    lines.append("")
    for _, row in best_summary.iterrows():
        if row.get("representative_model"):
            lines.append(
                f"- Level {int(row['info_level_step30'])} ({row['info_level_name_step30']}): "
                f"representative model `{row['representative_model']}`, RMSE={row['rmse_log_tau_eff']:.4f}, "
                f"Spearman={row['spearman_log_tau_eff']:.4f}."
            )
    lines.append("")
    lines.append(
        "If performance improves for information-rich samples, this supports insufficient material information as a candidate contributor to ML performance loss."
    )
    lines.append(
        "If performance does not improve, other candidates remain important, including the tau_eff definition, C(T) construction, data variability, and extrapolation difficulty."
    )
    lines.append("")
    for _, row in interpretation.iterrows():
        lines.append(f"- Level {int(row['info_level_step30'])}: {row['short_note']}")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step18_dir = Path(args.step18_dir)
    step17_dir = Path(args.step17_dir)
    step23_dir = Path(args.step23_dir)
    step19_dir = Path(args.step19_dir)

    paths = {
        "recommended": resolve_file(step18_dir, "tau_eff_ml_dataset_recommended_step18.csv", ["recommended", "step18"]),
        "feature_matrix": resolve_file(step18_dir, "tau_eff_ml_feature_matrix_step18.csv", ["feature", "matrix", "step18"]),
        "target": resolve_file(step18_dir, "tau_eff_ml_target_step18.csv", ["target", "step18"]),
        "metadata": resolve_file(step18_dir, "tau_eff_ml_metadata_step18.csv", ["metadata", "step18"]),
        "splits": resolve_file(step18_dir, "tau_eff_ml_splits_step18.csv", ["splits", "step18"]),
        "step17_base": resolve_file(step17_dir, "step17_tau_eff_ml_annotation_base.csv", ["tau_eff", "annotation", "base"]),
        "step17_annotated": resolve_file(step17_dir, "step17_annotated_samples.csv", ["annotated", "samples"]),
        "step23_samples": resolve_file(step23_dir, "step23_error_cause_samples.csv", ["error", "cause", "samples"]),
        "step19_comparison": resolve_file(step19_dir, "tau_eff_ml_model_comparison_step19.csv", ["model", "comparison", "step19"]),
    }

    info_cols = [
        "sample_key",
        "SID",
        "DOI",
        "doi_url",
        "paper_id",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p",
        "n_or_p_final_step17",
        "additive_auto_step9",
        "additive_manual_step9",
        "additive_final_step17",
        "structure_auto_step9",
        "structure_manual_step9",
        "structure_final_step17",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "sintering_method_final_step17",
        "sintering_condition_final_step17",
        "sintering_checked_final_step17",
    ]

    recommended = read_csv(paths["recommended"], usecols=info_cols)
    feature = read_csv(paths["feature_matrix"])
    target = read_csv(paths["target"])
    metadata = read_csv(paths["metadata"], usecols=info_cols + ["target_quality_step18"])
    splits = read_csv(paths["splits"], required=False)
    step17_base = read_csv(paths["step17_base"], required=False, usecols=info_cols)
    step17_annotated = read_csv(paths["step17_annotated"], required=False, usecols=info_cols)
    step23 = read_csv(paths["step23_samples"], required=False, usecols=info_cols)

    for name, df in [("recommended", recommended), ("feature_matrix", feature), ("target", target), ("metadata", metadata)]:
        assert_unique_sample_key(df, name)

    data = recommended[["sample_key"]].copy()
    data = data.merge(target[["sample_key", "target_log_tau_eff_step18", "target_tau_eff_step18"]], on="sample_key", how="inner")
    data = data.merge(metadata, on="sample_key", how="left")
    data = merge_optional(data, recommended, info_cols, "recommended")
    data = merge_optional(data, splits, list(splits.columns) if splits is not None else [], "splits")
    data = merge_optional(data, step17_base, info_cols, "step17")
    data = merge_optional(data, step17_annotated, info_cols, "step17annotated")
    data = merge_optional(data, step23, info_cols, "step23")
    assert_unique_sample_key(data, "merged Step30 data")

    feature = feature[feature["sample_key"].isin(data["sample_key"])].copy()
    assert_unique_sample_key(feature, "feature_matrix filtered")
    feature = data[["sample_key"]].merge(feature, on="sample_key", how="left")
    X, feature_audit, leak_cols = prepare_feature_matrix(feature)

    data = add_info_flags(data)
    y = pd.to_numeric(data["target_log_tau_eff_step18"], errors="coerce")
    valid_target = np.isfinite(y)
    data = data.loc[valid_target].reset_index(drop=True)
    X = X.loc[valid_target].reset_index(drop=True)
    y = y.loc[valid_target].reset_index(drop=True)

    paper_col = choose_paper_identifier(data)
    level_counts = make_level_counts(data, paper_col, args.min_samples, args.min_papers)
    missing_summary = make_missing_summary(data)

    comparison_rows = []
    prediction_frames = []
    for _, count_row in level_counts.iterrows():
        level_id = int(count_row["info_level_step30"])
        level_name = count_row["info_level_name_step30"]
        subset = data[level_mask(data, level_id)]
        if not bool(count_row["is_trainable_step30"]):
            for split_method in ["sample_random", "paper_group"]:
                comparison_rows.append(
                    {
                        "info_level_step30": level_id,
                        "info_level_name_step30": level_name,
                        "model_name": "",
                        "split_method_step30": split_method,
                        "rmse_log_tau_eff": np.nan,
                        "mae_log_tau_eff": np.nan,
                        "r2_log_tau_eff": np.nan,
                        "spearman_log_tau_eff": np.nan,
                        "train_sample_count": 0,
                        "test_sample_count": 0,
                        "train_doi_count": 0,
                        "test_doi_count": 0,
                        "doi_leakage_count": np.nan,
                        "paper_identifier_column": paper_col or "",
                        "skip_reason": count_row["excluded_or_skip_reason"],
                    }
                )
            continue
        for split_method in ["sample_random", "paper_group"]:
            rows, pred, skip_reason = evaluate_level(
                level_id, level_name, subset, X, y, split_method, args.random_state, paper_col
            )
            if skip_reason:
                comparison_rows.append(
                    {
                        "info_level_step30": level_id,
                        "info_level_name_step30": level_name,
                        "model_name": "",
                        "split_method_step30": split_method,
                        "rmse_log_tau_eff": np.nan,
                        "mae_log_tau_eff": np.nan,
                        "r2_log_tau_eff": np.nan,
                        "spearman_log_tau_eff": np.nan,
                        "train_sample_count": 0,
                        "test_sample_count": 0,
                        "train_doi_count": 0,
                        "test_doi_count": 0,
                        "doi_leakage_count": np.nan,
                        "paper_identifier_column": paper_col or "",
                        "skip_reason": skip_reason,
                    }
                )
            else:
                comparison_rows.extend(rows)
                prediction_frames.append(pred)

    comparison = pd.DataFrame(comparison_rows)
    best_summary = select_best_models(comparison)

    all_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    representative_predictions = []
    for _, row in best_summary.iterrows():
        if not row.get("representative_model") or all_predictions.empty:
            continue
        mask = (
            all_predictions["info_level_step30"].eq(row["info_level_step30"])
            & all_predictions["model_name"].eq(row["representative_model"])
            & all_predictions["split_method_step30"].eq(row["selection_split_method"])
        )
        representative_predictions.append(all_predictions[mask].copy())
    predictions = pd.concat(representative_predictions, ignore_index=True) if representative_predictions else pd.DataFrame()

    interpretation = make_interpretation(level_counts, best_summary)
    figure_index = make_step30_figures(level_counts, comparison, best_summary, missing_summary, predictions, output_dir)

    flags_cols = [
        "sample_key",
        "composition_info_step30",
        "material_system_info_step30",
        "material_system_source_step30",
        "n_or_p_info_step30",
        "additive_info_step30",
        "structure_info_step30",
        "sintering_info_step30",
        "has_composition_step30",
        "has_material_system_step30",
        "has_np_type_step30",
        "has_additive_info_step30",
        "has_structure_info_step30",
        "has_sintering_info_step30",
        "extra_info_count_step30",
        "is_level0_all_recommended_step30",
        "is_level1_basic_material_info_step30",
        "is_level2_any_extra_info_step30",
        "is_level3_two_extra_info_step30",
        "is_level4_full_extra_info_step30",
        "info_level_step30",
        "info_level_name_step30",
    ]
    for col in ["n_or_p", "sintering_method", "sintering_checked", "record_checked", "DOI", "doi_url", "SID"]:
        if col in data.columns and col not in flags_cols:
            flags_cols.insert(1, col)
    flags = data[[col for col in flags_cols if col in data.columns]].copy()

    report = make_report(
        level_counts,
        comparison,
        best_summary,
        interpretation,
        missing_summary,
        feature_audit,
        {k: str(v) for k, v in paths.items()},
        paper_col,
    )
    notes = make_notes(best_summary, interpretation)

    outputs = {
        "info_rich_sample_flags_step30.csv": flags,
        "info_rich_level_counts_step30.csv": level_counts,
        "info_rich_ml_model_comparison_step30.csv": comparison,
        "info_rich_best_model_summary_step30.csv": best_summary,
        "info_rich_predictions_step30.csv": predictions,
        "info_rich_level_interpretation_step30.csv": interpretation,
        "info_rich_missing_information_summary_step30.csv": missing_summary,
        "info_rich_feature_leakage_audit_step30.csv": feature_audit,
        "info_rich_figure_index_step30.csv": figure_index,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)
    (output_dir / "step30_info_rich_tau_eff_ml_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step30_info_rich_tau_eff_ml_notes.md").write_text(notes, encoding="utf-8")
    write_excel(
        output_dir / "starrydata2_step30_info_rich_tau_eff_ml.xlsx",
        {
            "sample_flags": flags,
            "level_counts": level_counts,
            "model_comparison": comparison,
            "best_model_summary": best_summary,
            "predictions": predictions,
            "interpretation": interpretation,
            "missing_summary": missing_summary,
            "feature_audit": feature_audit,
            "figure_index": figure_index,
            "report": report,
            "notes": notes,
        },
    )

    print(f"Step30 complete: {output_dir}")
    print(f"Figures written: {output_dir / 'figures'}")
    print(level_counts[["info_level_step30", "info_level_name_step30", "sample_count", "doi_count", "excluded_or_skip_reason"]].to_string(index=False))
    print(best_summary[["info_level_step30", "representative_model", "selection_split_method", "rmse_log_tau_eff", "spearman_log_tau_eff"]].to_string(index=False))


if __name__ == "__main__":
    main()
