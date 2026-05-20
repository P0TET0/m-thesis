import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Font

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    joblib = None
    JOBLIB_AVAILABLE = False


DEFAULT_STEP18_DIR = "data/output/starrydata2_step18_tau_eff_ml_dataset"
DEFAULT_OUTPUT_DIR = "data/output/starrydata2_step19_tau_eff_ml_model"
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

LEAKAGE_PATTERNS = [
    "tau_eff_step12",
    "log_tau_eff_step12",
    "target_log_tau_eff_step18",
    "target_tau_eff_step18",
    "sigma_fit_log_rmse_step12",
    "sigma_fit_mape_step12",
    "sigma_holdout_log_rmse_step12",
    "sigma_holdout_mape_step12",
    "validation_sigma_log_rmse_step13",
    "validation_sigma_mape_step13",
    "zt_obs_max_step14",
    "zt_pred_max_step14",
    "zt_pred_vs_obs_mape_step14",
    "zt_pred_vs_calc_mape_step14",
    "pf_mape_step14",
    "pf_log_rmse_step14",
    "zt_error_analysis_category_step15",
    "manual_review_priority_score",
    "problem_reason",
]

SPLITS = [
    "split_random_80_20_step18",
    "split_random_70_15_15_step18",
    "split_doi_group_80_20_step18",
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
    parser = argparse.ArgumentParser(description="Train Step19 ML models to predict Step18 log_tau_eff.")
    parser.add_argument("--step18_dir", default=DEFAULT_STEP18_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary_split", default="split_doi_group_80_20_step18")
    parser.add_argument("--selection_split", default="split_random_70_15_15_step18")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_rows_for_permutation_importance", type=int, default=3000)
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


def is_leakage_column(col):
    lower = col.lower()
    return any(pattern.lower() in lower for pattern in LEAKAGE_PATTERNS)


def finite_series(series):
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values)


def true_series(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def safe_exp(values):
    return np.exp(np.clip(values, -50, 50))


def prepare_features(feature_df, candidate_cols, train_mask):
    audit_rows = []
    X_raw = feature_df[candidate_cols].copy()
    for col in candidate_cols:
        if X_raw[col].dtype == bool:
            X_raw[col] = X_raw[col].astype(int)
        else:
            low = X_raw[col].astype(str).str.strip().str.lower()
            if set(low.dropna().unique()).issubset({"true", "false", "1", "0", "yes", "no", "nan", "<na>", ""}):
                X_raw[col] = low.map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
            else:
                X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
        X_raw[col] = X_raw[col].replace([np.inf, -np.inf], np.nan)

    train = X_raw.loc[train_mask]
    all_missing_cols = [c for c in X_raw.columns if train[c].isna().all()]
    X_raw = X_raw.drop(columns=all_missing_cols)
    train = train.drop(columns=all_missing_cols)

    medians = train.median(numeric_only=True).fillna(0.0)
    X_filled = X_raw.fillna(medians).fillna(0.0)

    variances = X_filled.loc[train_mask].var(axis=0)
    zero_variance_cols = variances[variances <= 0].index.tolist()
    X_filled = X_filled.drop(columns=zero_variance_cols)

    for col in candidate_cols:
        status = "used"
        note = ""
        if col in all_missing_cols:
            status = "removed_all_missing"
        elif col in zero_variance_cols:
            status = "removed_zero_variance"
        elif col not in X_filled.columns:
            status = "removed"
        else:
            note = f"imputation_median={float(medians.get(col, 0.0))}"
        audit_rows.append(
            {
                "feature_name": col,
                "used_in_model_step19": col in X_filled.columns,
                "status": status,
                "imputation_value": float(medians.get(col, 0.0)) if col in medians.index else 0.0,
                "note": note,
            }
        )
    return X_filled, X_filled.columns.tolist(), medians.reindex(X_filled.columns).fillna(0.0).to_dict(), pd.DataFrame(audit_rows)


def make_models(random_state, X_train_selection, y_train_selection, X_valid_selection, y_valid_selection):
    model_notes = {}
    best_alpha = 1.0
    best_rmse = np.inf
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        model.fit(X_train_selection, y_train_selection)
        pred = model.predict(X_valid_selection)
        rmse = math.sqrt(mean_squared_error(y_valid_selection, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha

    model_notes["ridge_regression"] = f"Ridge alpha selected on validation split: {best_alpha}"
    return {
        "baseline_mean": MeanRegressor(),
        "ridge_regression": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=best_alpha))]),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }, model_notes


def metric_dict(y_true_log, y_pred_log):
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    ok = np.isfinite(y_true_log) & np.isfinite(y_pred_log)
    y_true_log = y_true_log[ok]
    y_pred_log = y_pred_log[ok]
    if len(y_true_log) == 0:
        return empty_metrics()

    err = y_pred_log - y_true_log
    abs_err = np.abs(err)
    y_true_tau = safe_exp(y_true_log)
    y_pred_tau = safe_exp(y_pred_log)
    denom = np.where(y_true_tau == 0, np.nan, y_true_tau)
    rel = np.abs(y_pred_tau - y_true_tau) / denom
    ratio = np.where(y_true_tau > 0, y_pred_tau / y_true_tau, np.nan)
    ratio_abs = np.maximum(ratio, 1 / ratio)
    pearson = pd.Series(y_true_log).corr(pd.Series(y_pred_log), method="pearson") if len(y_true_log) > 1 else np.nan
    spearman = pd.Series(y_true_log).corr(pd.Series(y_pred_log), method="spearman") if len(y_true_log) > 1 else np.nan
    return {
        "n_samples": int(len(y_true_log)),
        "log_tau_mae": float(mean_absolute_error(y_true_log, y_pred_log)),
        "log_tau_rmse": float(math.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "log_tau_r2": float(r2_score(y_true_log, y_pred_log)) if len(y_true_log) > 1 else np.nan,
        "log_tau_pearson": float(pearson) if pd.notna(pearson) else np.nan,
        "log_tau_spearman": float(spearman) if pd.notna(spearman) else np.nan,
        "tau_eff_mae": float(np.nanmean(np.abs(y_pred_tau - y_true_tau))),
        "tau_eff_rmse": float(math.sqrt(np.nanmean((y_pred_tau - y_true_tau) ** 2))),
        "tau_eff_mape": float(np.nanmean(rel)),
        "tau_eff_median_absolute_relative_error": float(np.nanmedian(rel)),
        "median_multiplicative_error": float(np.nanmedian(ratio_abs)),
        "within_log_0_25_rate": float(np.mean(abs_err <= 0.25)),
        "within_log_0_50_rate": float(np.mean(abs_err <= 0.50)),
        "within_tau_factor_2_rate": float(np.nanmean((ratio >= 0.5) & (ratio <= 2.0))),
        "within_tau_factor_3_rate": float(np.nanmean((ratio >= 1 / 3) & (ratio <= 3.0))),
        "within_factor_10_rate": float(np.nanmean((ratio >= 0.1) & (ratio <= 10.0))),
        "bias_log_tau": float(np.mean(err)),
        "median_abs_error_log_tau": float(np.median(abs_err)),
    }


def empty_metrics():
    keys = [
        "n_samples",
        "log_tau_mae",
        "log_tau_rmse",
        "log_tau_r2",
        "log_tau_pearson",
        "log_tau_spearman",
        "tau_eff_mae",
        "tau_eff_rmse",
        "tau_eff_mape",
        "tau_eff_median_absolute_relative_error",
        "median_multiplicative_error",
        "within_log_0_25_rate",
        "within_log_0_50_rate",
        "within_tau_factor_2_rate",
        "within_tau_factor_3_rate",
        "within_factor_10_rate",
        "bias_log_tau",
        "median_abs_error_log_tau",
    ]
    return {k: np.nan for k in keys}


def split_mask(data, split_name, role):
    return data[split_name].astype(str).str.lower().eq(role)


def doi_diagnostics(data, split_name):
    train = data[split_name].astype(str).eq("train")
    valid = data[split_name].astype(str).eq("valid")
    test = data[split_name].astype(str).eq("test")
    doi = data["DOI"].fillna("").astype(str).str.strip()
    train_doi = set(doi[train & doi.ne("")])
    test_doi = set(doi[test & doi.ne("")])
    overlap = train_doi & test_doi
    return {
        "split_name": split_name,
        "train_sample_count": int(train.sum()),
        "valid_sample_count": int(valid.sum()),
        "test_sample_count": int(test.sum()),
        "train_doi_count": len(train_doi),
        "test_doi_count": len(test_doi),
        "doi_overlap_count": len(overlap),
        "doi_leakage_flag": len(overlap) > 0,
        "missing_doi_count": int(doi.eq("").sum()),
        "note": "ok" if len(overlap) == 0 else "DOI appears in both train and test",
    }


def prediction_frame(data, y_pred, model_name, split_name):
    out = data[
        [
            "sample_key",
            "target_log_tau_eff_step18",
            "target_tau_eff_step18",
            "material_system",
            "n_or_p",
            "n_or_p_final_step17",
            "composition",
            "DOI",
            "target_quality_step18",
            "use_for_tau_eff_ml_step18",
        ]
    ].copy()
    out["model_name"] = model_name
    out["split_name"] = split_name
    out["split_role"] = data[split_name].values
    out["pred_log_tau_eff_step19"] = y_pred
    out["residual_log_tau_eff_step19"] = out["pred_log_tau_eff_step19"] - out["target_log_tau_eff_step18"]
    out["abs_error_log_tau_eff_step19"] = out["residual_log_tau_eff_step19"].abs()
    out["pred_tau_eff_step19"] = safe_exp(out["pred_log_tau_eff_step19"].astype(float))
    out["tau_eff_relative_error_step19"] = (
        (out["pred_tau_eff_step19"] - out["target_tau_eff_step18"]).abs() / out["target_tau_eff_step18"]
    )
    out["tau_eff_ratio_pred_true_step19"] = out["pred_tau_eff_step19"] / out["target_tau_eff_step18"]
    return out[
        [
            "sample_key",
            "model_name",
            "split_name",
            "split_role",
            "target_log_tau_eff_step18",
            "pred_log_tau_eff_step19",
            "residual_log_tau_eff_step19",
            "abs_error_log_tau_eff_step19",
            "target_tau_eff_step18",
            "pred_tau_eff_step19",
            "tau_eff_relative_error_step19",
            "tau_eff_ratio_pred_true_step19",
            "material_system",
            "n_or_p",
            "n_or_p_final_step17",
            "composition",
            "DOI",
            "target_quality_step18",
            "use_for_tau_eff_ml_step18",
        ]
    ]


def feature_importance_frame(model, model_name, X_test, y_test, feature_cols, feature_dictionary, random_state, max_rows):
    rows = []
    if model_name == "baseline_mean":
        return pd.DataFrame()

    base_model = model
    if isinstance(model, Pipeline) and "ridge" in model.named_steps:
        values = model.named_steps["ridge"].coef_
        importance_type = "absolute_coefficient"
    elif hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
        importance_type = "feature_importances_"
    else:
        values = None
        importance_type = "not_available"

    if values is not None:
        for feature, value in zip(feature_cols, values):
            rows.append(
                {
                    "model_name": model_name,
                    "importance_type": importance_type,
                    "feature_name": feature,
                    "importance_value": float(value),
                    "abs_importance_value": float(abs(value)),
                    "rank": 0,
                    "note": "",
                }
            )

    if len(X_test) > 0:
        sample_n = min(len(X_test), max_rows)
        X_perm = X_test.sample(sample_n, random_state=random_state) if sample_n < len(X_test) else X_test
        y_perm = y_test.loc[X_perm.index]
        try:
            perm = permutation_importance(
                model,
                X_perm,
                y_perm,
                n_repeats=3,
                random_state=random_state,
                scoring="neg_root_mean_squared_error",
                n_jobs=1,
            )
            for feature, value in zip(feature_cols, perm.importances_mean):
                rows.append(
                    {
                        "model_name": model_name,
                        "importance_type": "permutation_importance_primary_test",
                        "feature_name": feature,
                        "importance_value": float(value),
                        "abs_importance_value": float(abs(value)),
                        "rank": 0,
                        "note": f"n={sample_n}",
                    }
                )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "importance_type": "permutation_importance_primary_test",
                    "feature_name": "__permutation_failed__",
                    "importance_value": np.nan,
                    "abs_importance_value": np.nan,
                    "rank": 0,
                    "note": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank"] = out.groupby(["model_name", "importance_type"])["abs_importance_value"].rank(
        ascending=False, method="first"
    ).astype("Int64")
    if feature_dictionary is not None and "feature_name" in feature_dictionary.columns:
        keep = [c for c in ["feature_name", "feature_source", "feature_type"] if c in feature_dictionary.columns]
        out = out.merge(feature_dictionary[keep].drop_duplicates("feature_name"), on="feature_name", how="left")
    else:
        out["feature_source"] = ""
        out["feature_type"] = ""
    return out


def group_performance(best_primary_test, group_cols):
    rows = []
    for col in group_cols:
        if col not in best_primary_test.columns:
            continue
        for value, group in best_primary_test.groupby(best_primary_test[col].fillna("missing").astype(str)):
            if len(group) == 0:
                continue
            metrics = metric_dict(group["target_log_tau_eff_step18"], group["pred_log_tau_eff_step19"])
            rows.append(
                {
                    "group_name": col,
                    "group_value": value,
                    "n_samples": int(len(group)),
                    "log_tau_mae": metrics["log_tau_mae"],
                    "log_tau_rmse": metrics["log_tau_rmse"],
                    "log_tau_r2": metrics["log_tau_r2"],
                    "tau_eff_mape": metrics["tau_eff_mape"],
                    "within_tau_factor_2_rate": metrics["within_tau_factor_2_rate"],
                    "median_target_log_tau_eff": float(group["target_log_tau_eff_step18"].median()),
                    "median_pred_log_tau_eff": float(group["pred_log_tau_eff_step19"].median()),
                    "note": "primary split test; selected model",
                }
            )
    return pd.DataFrame(rows)


def residual_category(row):
    abs_err = row["abs_error_log_tau_eff_step19"]
    if abs_err <= 0.25:
        return "good"
    if abs_err <= 0.50:
        return "moderate_error"
    return "large_under_prediction" if row["pred_log_tau_eff_step19"] < row["target_log_tau_eff_step18"] else "large_over_prediction"


def problem_reason(row):
    reasons = []
    ratio = row["tau_eff_ratio_pred_true_step19"]
    if ratio < 1 / 3:
        reasons.append("large tau_eff underprediction")
    if ratio > 3:
        reasons.append("large tau_eff overprediction")
    if str(row.get("target_quality_step18", "")).lower() == "low":
        reasons.append("low target quality")
    if str(row.get("manual_review_status_step17", "")).lower() in {"", "unknown", "not_checked", "nan"}:
        reasons.append("manual annotation missing")
    if str(row.get("material_system", "")).lower() == "unknown":
        reasons.append("possible material-system outlier")
    if str(row.get("additive_final_step17", "")).lower() == "unknown" or str(row.get("structure_final_step17", "")).lower() == "unknown":
        reasons.append("possibly missing additive/structure feature")
    return "; ".join(dict.fromkeys(reasons))


def write_excel(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, data in sheets.items():
            if isinstance(data, str):
                data = pd.DataFrame({"report": data.splitlines()})
            data.head(EXCEL_PREVIEW_ROWS).to_excel(writer, sheet_name=sheet[:31], index=False)
            ws = writer.sheets[sheet[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for col_cells in ws.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in col_cells[:200]]
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(len(v) for v in values) + 2, 60)


def make_notes():
    return """# Step19 tau_eff ML Model Notes

## Purpose
Train machine learning models that predict fitted `log_tau_eff` from Step18 material features.

## Target
The target is `target_log_tau_eff_step18`. `tau_eff` is retained only as an exponentiated relative-scale value.

## Features
The model uses only columns from `tau_eff_ml_feature_matrix_step18.csv` after leakage checks, numeric conversion, imputation, and zero-variance filtering.

## Models
The run compares a mean baseline, Ridge regression, RandomForest, ExtraTrees, and GradientBoosting.

## Splits
Models are evaluated on random 80/20, random 70/15/15, and DOI group 80/20 splits.

## Selected Model
The selected model is chosen by validation RMSE on `split_random_70_15_15_step18`.

## Main Results
See `step19_tau_eff_ml_report.txt` and `tau_eff_ml_model_comparison_step19.csv`.

## Important Caveats
The model predicts log_tau_eff, not sigma directly.
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.
No sigma, PF, or ZT recalculation is performed in Step19.
The DOI group split is more reliable than random split for evaluating generalization.
Final all-sample predictions are for downstream screening, not unbiased evaluation.

## Next Step
Step20 should use the predicted tau_eff values to compute sigma_pred_ML, PF_pred_ML, and ZT_pred_ML.
"""


def make_report(input_counts, training_count, feature_count, removed_feature_count, missing_target_count, model_names, selected_summary, primary_metrics, model_comparison, feature_importance, group_perf, problem_samples, leakage_check, split_diag, output_counts):
    lines = []
    lines.append("Step19 tau_eff ML report")
    lines.append("")
    lines.append(f"Input recommended ML dataset rows: {input_counts['recommended']}")
    lines.append(f"Input feature matrix rows: {input_counts['feature_matrix']}")
    lines.append(f"Input target rows: {input_counts['target']}")
    lines.append(f"Input splits rows: {input_counts['splits']}")
    lines.append("")
    lines.append(f"Training samples: {training_count}")
    lines.append(f"Feature count: {feature_count}")
    lines.append(f"Removed feature count: {removed_feature_count}")
    lines.append(f"Target missing sample count: {missing_target_count}")
    lines.append("")
    lines.append(f"Executed models: {', '.join(model_names)}")
    lines.append(f"Selected model: {selected_summary['selected_model_name']}")
    lines.append(f"Selection split: {selected_summary['selection_split']}")
    lines.append(f"Selection metric: {selected_summary['selection_metric']}")
    lines.append(f"Selection validation log RMSE: {selected_summary['selection_metric_value']}")
    lines.append("")
    lines.append("Primary evaluation:")
    for key, value in primary_metrics.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Model comparison summary:")
    for _, row in model_comparison[
        (model_comparison["split_name"] == selected_summary["primary_split"])
        & (model_comparison["split_role"] == "test")
    ].sort_values("log_tau_rmse").iterrows():
        lines.append(
            f"- {row['model_name']}: primary test RMSE={row['log_tau_rmse']}, Spearman={row['log_tau_spearman']}"
        )
    lines.append("")
    lines.append("Random split test RMSE:")
    for _, row in model_comparison[
        (model_comparison["split_name"] == "split_random_80_20_step18")
        & (model_comparison["split_role"] == "test")
    ].sort_values("log_tau_rmse").iterrows():
        lines.append(f"- {row['model_name']}: {row['log_tau_rmse']}")
    lines.append("")
    lines.append("Feature importance top 20:")
    top_imp = feature_importance[
        feature_importance["importance_type"].isin(["feature_importances_", "absolute_coefficient"])
    ].sort_values("abs_importance_value", ascending=False).head(20)
    for _, row in top_imp.iterrows():
        lines.append(f"- {row['feature_name']}: {row['importance_value']}")
    lines.append("")
    lines.append("Group performance:")
    for group in ["material_system", "n_or_p", "nanocarbon_keyword_detected_step9"]:
        sub = group_perf[group_perf["group_name"] == group].sort_values("log_tau_rmse").head(10)
        lines.append(f"- {group}:")
        for _, row in sub.iterrows():
            lines.append(f"  {row['group_value']}: n={row['n_samples']}, RMSE={row['log_tau_rmse']}")
    lines.append("")
    lines.append("Problem samples:")
    lines.append(f"- large underprediction count: {int(problem_samples['tau_eff_ml_problem_reason_step19'].astype(str).str.contains('underprediction').sum()) if len(problem_samples) else 0}")
    lines.append(f"- large overprediction count: {int(problem_samples['tau_eff_ml_problem_reason_step19'].astype(str).str.contains('overprediction').sum()) if len(problem_samples) else 0}")
    lines.append(f"- needs_manual_review_for_ml_step19 count: {int(problem_samples['needs_manual_review_for_ml_step19'].astype(bool).sum()) if len(problem_samples) else 0}")
    lines.append("")
    lines.append("Leakage check:")
    lines.append(f"- suspicious feature count: {int(leakage_check['removed_columns'].astype(str).str.len().gt(2).sum())}")
    lines.append(f"- removed leakage feature count: {int(leakage_check.loc[leakage_check['checked_item'] == 'leakage_columns', 'removed_count'].iloc[0])}")
    lines.append("")
    lines.append("Split diagnostics:")
    for _, row in split_diag.iterrows():
        lines.append(f"- {row['split_name']}: train={row['train_sample_count']}, valid={row['valid_sample_count']}, test={row['test_sample_count']}, DOI leakage={row['doi_overlap_count']}")
    lines.append("")
    lines.append("Output row counts:")
    for key, value in output_counts.items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("Notes:")
    lines.append("Step19 trained machine learning models to predict log_tau_eff.")
    lines.append("Step19 did not recalculate sigma, PF, or ZT.")
    lines.append("tau_eff is relative scale and not physical seconds.")
    lines.append("Step20 will use predicted tau_eff to compute sigma_pred_ML, PF_pred_ML, and ZT_pred_ML.")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    step18_dir = Path(args.step18_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "model_artifacts_step19"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    recommended = read_csv(step18_dir / "tau_eff_ml_dataset_recommended_step18.csv")
    feature_matrix = read_csv(step18_dir / "tau_eff_ml_feature_matrix_step18.csv")
    target = read_csv(step18_dir / "tau_eff_ml_target_step18.csv")
    metadata = read_csv(step18_dir / "tau_eff_ml_metadata_step18.csv")
    splits = read_csv(step18_dir / "tau_eff_ml_splits_step18.csv")
    feature_dictionary = read_csv(step18_dir / "tau_eff_ml_feature_dictionary_step18.csv")
    all_dataset = read_csv(step18_dir / "tau_eff_ml_dataset_step18.csv", required=False)

    input_counts = {
        "recommended": len(recommended),
        "feature_matrix": len(feature_matrix),
        "target": len(target),
        "metadata": len(metadata),
        "splits": len(splits),
    }

    duplicate_count = int(feature_matrix["sample_key"].duplicated().sum())
    recommended_keys = set(recommended["sample_key"].astype(str))
    feature_cols_raw = [c for c in feature_matrix.columns if c != "sample_key"]
    leakage_cols = [c for c in feature_cols_raw if is_leakage_column(c)]
    candidate_cols = [c for c in feature_cols_raw if c not in leakage_cols]

    leakage_check = pd.DataFrame(
        [
            {
                "checked_item": "sample_key_duplicates",
                "status": "ok" if duplicate_count == 0 else "warning",
                "removed_columns": "[]",
                "removed_count": 0,
                "note": f"feature_matrix duplicate sample_key rows={duplicate_count}",
            },
            {
                "checked_item": "leakage_columns",
                "status": "ok" if not leakage_cols else "removed",
                "removed_columns": json.dumps(leakage_cols, ensure_ascii=False),
                "removed_count": len(leakage_cols),
                "note": "target/evaluation-derived columns removed from model inputs",
            },
        ]
    )

    ml_data = (
        feature_matrix[["sample_key"] + candidate_cols]
        .merge(target, on="sample_key", how="left")
        .merge(metadata, on="sample_key", how="left", suffixes=("", "__metadata"))
        .merge(splits, on="sample_key", how="left")
    )
    ml_data = ml_data[ml_data["sample_key"].astype(str).isin(recommended_keys)].copy()

    target_log = pd.to_numeric(ml_data["target_log_tau_eff_step18"], errors="coerce")
    target_tau = pd.to_numeric(ml_data["target_tau_eff_step18"], errors="coerce")
    use_for_ml = true_series(ml_data["use_for_tau_eff_ml_step18"])
    trainable = use_for_ml & np.isfinite(target_log) & np.isfinite(target_tau) & (target_tau > 0)
    ml_data["use_for_training_step19"] = trainable
    ml_data["training_exclusion_reason_step19"] = np.where(
        trainable,
        "ok",
        np.where(~use_for_ml, "use_for_tau_eff_ml_step18 is false", "missing target"),
    )
    missing_target_count = int((~np.isfinite(target_log)).sum())
    ml_data = ml_data[trainable].copy()
    ml_data["target_log_tau_eff_step18"] = pd.to_numeric(ml_data["target_log_tau_eff_step18"], errors="coerce")
    ml_data["target_tau_eff_step18"] = pd.to_numeric(ml_data["target_tau_eff_step18"], errors="coerce")

    selection_train_mask = split_mask(ml_data, args.selection_split, "train")
    X_all, feature_cols, imputation_values, input_audit = prepare_features(ml_data, candidate_cols, selection_train_mask)
    removed_feature_count = int((~input_audit["used_in_model_step19"]).sum()) + len(leakage_cols)
    y = ml_data["target_log_tau_eff_step18"].astype(float)

    X_sel_train = X_all.loc[selection_train_mask]
    y_sel_train = y.loc[selection_train_mask]
    X_sel_valid = X_all.loc[split_mask(ml_data, args.selection_split, "valid")]
    y_sel_valid = y.loc[split_mask(ml_data, args.selection_split, "valid")]
    model_templates, model_notes = make_models(args.random_state, X_sel_train, y_sel_train, X_sel_valid, y_sel_valid)

    split_diag = pd.DataFrame([doi_diagnostics(ml_data, s) for s in SPLITS])

    comparison_rows = []
    prediction_frames = []
    fitted_primary_models = {}
    selected_split_models = {}
    for model_name, template in model_templates.items():
        for split_name in SPLITS:
            train_mask = split_mask(ml_data, split_name, "train")
            if not train_mask.any():
                continue
            model = pickle.loads(pickle.dumps(template))
            model.fit(X_all.loc[train_mask], y.loc[train_mask])
            if split_name == args.primary_split:
                fitted_primary_models[model_name] = model
            if split_name == args.selection_split:
                selected_split_models[model_name] = model
            y_pred_all = model.predict(X_all)
            prediction_frames.append(prediction_frame(ml_data, y_pred_all, model_name, split_name))
            for role in ["train", "valid", "test"]:
                mask = split_mask(ml_data, split_name, role)
                if not mask.any():
                    continue
                metrics = metric_dict(y.loc[mask], y_pred_all[mask.values])
                row = {
                    "model_name": model_name,
                    "split_name": split_name,
                    "split_role": role,
                    "model_note": model_notes.get(model_name, ""),
                }
                row.update(metrics)
                comparison_rows.append(row)

    model_comparison = pd.DataFrame(comparison_rows)
    selection_rows = model_comparison[
        (model_comparison["split_name"] == args.selection_split)
        & (model_comparison["split_role"] == "valid")
    ].copy()
    selection_rows = selection_rows.sort_values(["log_tau_rmse", "log_tau_mae"], ascending=[True, True])
    selected_model = selection_rows.iloc[0]["model_name"]
    selection_metric_value = float(selection_rows.iloc[0]["log_tau_rmse"])
    model_comparison["is_selection_metric"] = (
        (model_comparison["split_name"] == args.selection_split)
        & (model_comparison["split_role"] == "valid")
    )
    model_comparison["is_primary_evaluation"] = (
        (model_comparison["split_name"] == args.primary_split)
        & (model_comparison["split_role"] == "test")
    )
    model_comparison["metric_rank"] = np.nan
    rank_mask = model_comparison["is_selection_metric"]
    model_comparison.loc[rank_mask, "metric_rank"] = model_comparison.loc[rank_mask, "log_tau_rmse"].rank(method="first")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    best_predictions = predictions[predictions["model_name"].eq(selected_model)].copy()
    best_predictions["selected_model_step19"] = True
    best_predictions["is_primary_split_step19"] = best_predictions["split_name"].eq(args.primary_split)
    best_predictions["prediction_note_step19"] = "split evaluation prediction; use held-out roles for evaluation"

    primary_model = fitted_primary_models[selected_model]
    primary_test_mask = split_mask(ml_data, args.primary_split, "test")
    primary_test_pred = best_predictions[
        best_predictions["split_name"].eq(args.primary_split) & best_predictions["split_role"].eq("test")
    ].copy()
    primary_test_pred = primary_test_pred.merge(
        metadata[
            [
                c
                for c in [
                    "sample_key",
                    "doi_url",
                    "sigma_fit_log_rmse_step12",
                    "validation_sigma_log_rmse_step13",
                    "zt_obs_max_step14",
                    "zt_pred_max_step14",
                    "manual_review_status_step17",
                    "paper_checked_step17",
                ]
                if c in metadata.columns
            ]
        ],
        on="sample_key",
        how="left",
    )
    for col in ["nanocarbon_keyword_detected_step9", "rare_metal_flag_auto_step9", "toxicity_flag_auto_step9", "additive_final_step17", "structure_final_step17"]:
        if col in recommended.columns and col not in primary_test_pred.columns:
            primary_test_pred = primary_test_pred.merge(recommended[["sample_key", col]], on="sample_key", how="left")

    feature_importance = feature_importance_frame(
        primary_model,
        selected_model,
        X_all.loc[primary_test_mask],
        y.loc[primary_test_mask],
        feature_cols,
        feature_dictionary,
        args.random_state,
        args.max_rows_for_permutation_importance,
    )

    group_cols = [
        "material_system",
        "n_or_p",
        "n_or_p_final_step17",
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "target_quality_step18",
    ]
    group_perf_input = primary_test_pred.copy()
    for col in group_cols:
        if col not in group_perf_input.columns and col in recommended.columns:
            group_perf_input = group_perf_input.merge(recommended[["sample_key", col]], on="sample_key", how="left")
    group_perf = group_performance(group_perf_input, group_cols)

    residual = primary_test_pred.copy()
    residual["residual_category_step19"] = residual.apply(residual_category, axis=1)
    residual["residual_note_step19"] = "selected model primary split test residual"
    residual_cols = [
        "sample_key",
        "material_system",
        "n_or_p",
        "composition",
        "DOI",
        "target_log_tau_eff_step18",
        "pred_log_tau_eff_step19",
        "residual_log_tau_eff_step19",
        "abs_error_log_tau_eff_step19",
        "target_tau_eff_step18",
        "pred_tau_eff_step19",
        "tau_eff_ratio_pred_true_step19",
        "target_quality_step18",
        "sigma_fit_log_rmse_step12",
        "validation_sigma_log_rmse_step13",
        "zt_obs_max_step14",
        "zt_pred_max_step14",
        "manual_review_status_step17",
        "paper_checked_step17",
        "residual_category_step19",
        "residual_note_step19",
    ]
    residual = residual[[c for c in residual_cols if c in residual.columns]].sort_values("abs_error_log_tau_eff_step19", ascending=False)

    problem = residual[
        (residual["abs_error_log_tau_eff_step19"] > 0.75)
        | (residual["tau_eff_ratio_pred_true_step19"] > 3)
        | (residual["tau_eff_ratio_pred_true_step19"] < 1 / 3)
    ].copy()
    problem = problem.merge(
        recommended[
            [
                c
                for c in ["sample_key", "additive_final_step17", "structure_final_step17", "manual_review_status_step17"]
                if c in recommended.columns
            ]
        ],
        on="sample_key",
        how="left",
        suffixes=("", "__rec"),
    )
    problem["tau_eff_ml_problem_reason_step19"] = problem.apply(problem_reason, axis=1)
    problem["needs_manual_review_for_ml_step19"] = True

    final_model = pickle.loads(pickle.dumps(selected_split_models[selected_model]))
    final_model.fit(X_all, y)
    all_source = all_dataset if all_dataset is not None else recommended
    all_features = feature_matrix[["sample_key"] + feature_cols].merge(all_source[["sample_key"]], on="sample_key", how="right")
    X_final_raw = all_features[feature_cols].copy()
    for col in feature_cols:
        X_final_raw[col] = pd.to_numeric(X_final_raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_final = X_final_raw.fillna(pd.Series(imputation_values)).fillna(0.0)
    pred_final_log = final_model.predict(X_final)
    final_predictions = pd.DataFrame(
        {
            "sample_key": all_features["sample_key"],
            "pred_log_tau_eff_final_model_step19": pred_final_log,
            "pred_tau_eff_final_model_step19": safe_exp(pred_final_log),
            "final_model_name_step19": selected_model,
            "final_model_training_scope_step19": "trained_on_all_recommended_ml_samples",
            "prediction_usage_note_step19": "for downstream screening; not for unbiased evaluation",
        }
    )
    meta_cols = [c for c in ["sample_key", "composition", "material_system", "n_or_p", "n_or_p_final_step17", "DOI", "doi_url"] if c in all_source.columns]
    final_predictions = final_predictions.merge(all_source[meta_cols].drop_duplicates("sample_key"), on="sample_key", how="left")

    primary_row = model_comparison[
        (model_comparison["model_name"] == selected_model)
        & (model_comparison["split_name"] == args.primary_split)
        & (model_comparison["split_role"] == "test")
    ].iloc[0]
    selected_summary = pd.DataFrame(
        [
            {
                "selected_model_name": selected_model,
                "selection_split": args.selection_split,
                "selection_metric": "valid log_tau_rmse",
                "selection_metric_value": selection_metric_value,
                "primary_split": args.primary_split,
                "primary_test_log_tau_rmse": primary_row["log_tau_rmse"],
                "primary_test_log_tau_mae": primary_row["log_tau_mae"],
                "primary_test_log_tau_r2": primary_row["log_tau_r2"],
                "primary_test_log_tau_spearman": primary_row["log_tau_spearman"],
                "primary_test_tau_eff_mape": primary_row["tau_eff_mape"],
                "primary_test_within_tau_factor_2_rate": primary_row["within_tau_factor_2_rate"],
                "recommended_ml_sample_count": len(ml_data),
                "feature_count": len(feature_cols),
                "training_note": "selected by validation RMSE; primary DOI test is reported separately",
                "caution": "tau_eff is relative scale, not physical seconds. Final all-sample predictions are for downstream screening and not unbiased evaluation.",
            }
        ]
    )

    metrics = model_comparison.copy()
    model_metrics_alias = metrics.rename(
        columns={
            "log_tau_mae": "mae_log_tau",
            "log_tau_rmse": "rmse_log_tau",
            "log_tau_r2": "r2_log_tau",
            "bias_log_tau": "bias_log_tau",
            "median_abs_error_log_tau": "median_abs_error_log_tau",
            "tau_eff_mae": "mae_tau",
            "tau_eff_rmse": "rmse_tau",
            "tau_eff_mape": "mape_tau",
        }
    )

    # Compatibility table for the first Step19 filename set.
    primary_tests = metrics[(metrics["split_name"] == args.primary_split) & (metrics["split_role"] == "test")].copy()
    primary_train = metrics[(metrics["split_name"] == args.primary_split) & (metrics["split_role"] == "train")][
        ["model_name", "log_tau_rmse"]
    ].rename(columns={"log_tau_rmse": "train_rmse_log_tau"})
    comp_alias = primary_tests.merge(primary_train, on="model_name", how="left")
    comp_alias["primary_split_name"] = args.primary_split
    comp_alias["test_rmse_log_tau"] = comp_alias["log_tau_rmse"]
    comp_alias["test_mae_log_tau"] = comp_alias["log_tau_mae"]
    comp_alias["test_r2_log_tau"] = comp_alias["log_tau_r2"]
    comp_alias["test_median_multiplicative_error"] = comp_alias["median_multiplicative_error"]
    comp_alias["test_within_factor_2_rate"] = comp_alias["within_tau_factor_2_rate"]
    comp_alias["test_within_factor_3_rate"] = comp_alias["within_tau_factor_3_rate"]
    comp_alias["test_within_factor_10_rate"] = comp_alias["within_factor_10_rate"]
    comp_alias["generalization_gap_rmse_log_tau"] = comp_alias["test_rmse_log_tau"] - comp_alias["train_rmse_log_tau"]
    comp_alias["rank_by_primary_rmse"] = comp_alias["test_rmse_log_tau"].rank(method="first")
    comp_alias["rank_by_primary_r2"] = comp_alias["test_r2_log_tau"].rank(ascending=False, method="first")
    comp_alias["selected_best_model"] = comp_alias["model_name"].eq(selected_model)
    comp_alias["selection_reason"] = "Selected by random 70/15/15 validation RMSE; DOI test reported as primary reliability check."
    comp_alias = comp_alias[
        [
            "model_name",
            "primary_split_name",
            "test_rmse_log_tau",
            "test_mae_log_tau",
            "test_r2_log_tau",
            "test_median_multiplicative_error",
            "test_within_factor_2_rate",
            "test_within_factor_3_rate",
            "test_within_factor_10_rate",
            "train_rmse_log_tau",
            "generalization_gap_rmse_log_tau",
            "rank_by_primary_rmse",
            "rank_by_primary_r2",
            "selected_best_model",
            "selection_reason",
        ]
    ]

    output_counts = {
        "model_comparison": len(model_comparison),
        "predictions": len(predictions),
        "best_model_predictions": len(best_predictions),
        "final_all_samples_predictions": len(final_predictions),
        "feature_importance": len(feature_importance),
        "problem_samples": len(problem),
    }
    primary_metrics = {
        "primary split": args.primary_split,
        "DOI leakage count": int(split_diag.loc[split_diag["split_name"].eq(args.primary_split), "doi_overlap_count"].iloc[0]),
        "primary test sample count": int(primary_row["n_samples"]),
        "primary test log_tau MAE": primary_row["log_tau_mae"],
        "primary test log_tau RMSE": primary_row["log_tau_rmse"],
        "primary test log_tau R2": primary_row["log_tau_r2"],
        "primary test log_tau Pearson": primary_row["log_tau_pearson"],
        "primary test log_tau Spearman": primary_row["log_tau_spearman"],
        "primary test tau_eff MAPE": primary_row["tau_eff_mape"],
        "primary test within factor 2 rate": primary_row["within_tau_factor_2_rate"],
        "primary test within factor 3 rate": primary_row["within_tau_factor_3_rate"],
    }
    report = make_report(
        input_counts,
        len(ml_data),
        len(feature_cols),
        removed_feature_count,
        missing_target_count,
        list(model_templates.keys()),
        selected_summary.iloc[0].to_dict(),
        primary_metrics,
        model_comparison,
        feature_importance,
        group_perf,
        problem,
        leakage_check,
        split_diag,
        output_counts,
    )
    notes = make_notes()

    model_comparison.to_csv(output_dir / "tau_eff_ml_model_comparison_step19.csv", index=False)
    metrics.to_csv(output_dir / "tau_eff_ml_metrics_step19.csv", index=False)
    predictions.to_csv(output_dir / "tau_eff_ml_predictions_step19.csv", index=False)
    best_predictions.to_csv(output_dir / "tau_eff_ml_best_model_predictions_step19.csv", index=False)
    final_predictions.to_csv(output_dir / "tau_eff_ml_final_all_samples_predictions_step19.csv", index=False)
    feature_importance.to_csv(output_dir / "tau_eff_ml_feature_importance_step19.csv", index=False)
    group_perf.to_csv(output_dir / "tau_eff_ml_group_performance_step19.csv", index=False)
    residual.to_csv(output_dir / "tau_eff_ml_residual_analysis_step19.csv", index=False)
    problem.to_csv(output_dir / "tau_eff_ml_problem_samples_step19.csv", index=False)
    selected_summary.to_csv(output_dir / "tau_eff_ml_selected_model_summary_step19.csv", index=False)
    leakage_check.to_csv(output_dir / "tau_eff_ml_leakage_check_step19.csv", index=False)
    split_diag.to_csv(output_dir / "tau_eff_ml_split_diagnostics_step19.csv", index=False)
    input_audit.to_csv(output_dir / "tau_eff_ml_model_input_audit_step19.csv", index=False)
    (output_dir / "step19_tau_eff_ml_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step19_tau_eff_ml_notes.md").write_text(notes, encoding="utf-8")

    # Compatibility outputs for the alternate Step19 instruction block.
    model_metrics_alias.to_csv(output_dir / "tau_eff_ml_model_metrics_step19.csv", index=False)
    best_primary_test_alias = primary_test_pred.rename(
        columns={
            "pred_log_tau_eff_step19": "log_tau_eff_pred_step19",
            "pred_tau_eff_step19": "tau_eff_pred_step19",
            "residual_log_tau_eff_step19": "log_tau_error_step19",
            "tau_eff_ratio_pred_true_step19": "tau_eff_multiplicative_error_step19",
            "abs_error_log_tau_eff_step19": "absolute_log_tau_error_step19",
        }
    )
    best_primary_test_alias.to_csv(output_dir / "tau_eff_ml_test_predictions_step19.csv", index=False)
    comp_alias.to_csv(output_dir / "tau_eff_ml_model_comparison_primary_step19.csv", index=False)
    selected_summary.rename(columns={"selected_model_name": "best_model_name"}).to_csv(
        output_dir / "tau_eff_ml_best_model_summary_step19.csv", index=False
    )
    residual.to_csv(output_dir / "tau_eff_ml_error_analysis_step19.csv", index=False)
    (output_dir / "step19_tau_eff_ml_training_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "step19_tau_eff_ml_training_notes.md").write_text(notes, encoding="utf-8")

    if JOBLIB_AVAILABLE:
        joblib.dump(primary_model, artifact_dir / "best_model_step19.joblib")
        joblib.dump(final_model, artifact_dir / "final_model_trained_on_all_recommended_step19.joblib")
        joblib.dump(primary_model, output_dir / "best_tau_eff_model_step19.joblib")
    (artifact_dir / "feature_columns_step19.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (artifact_dir / "model_config_step19.json").write_text(
        json.dumps(
            {
                "selected_model": selected_model,
                "primary_split": args.primary_split,
                "selection_split": args.selection_split,
                "random_state": args.random_state,
                "joblib_available": JOBLIB_AVAILABLE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "best_model_feature_columns_step19.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (output_dir / "best_model_imputation_values_step19.json").write_text(
        json.dumps(imputation_values, indent=2), encoding="utf-8"
    )

    write_excel(
        output_dir / "starrydata2_step19_tau_eff_ml_model.xlsx",
        {
            "model_comparison": model_comparison,
            "metrics": metrics,
            "best_model_predictions": best_predictions,
            "feature_importance": feature_importance,
            "group_performance": group_perf,
            "residual_analysis": residual,
            "problem_samples": problem,
            "selected_model_summary": selected_summary,
            "leakage_check": leakage_check,
            "ml_report": report,
        },
    )
    write_excel(
        output_dir / "starrydata2_step19_tau_eff_ml_training.xlsx",
        {
            "model_comparison": comp_alias,
            "model_metrics": model_metrics_alias,
            "test_predictions": best_primary_test_alias,
            "feature_importance": feature_importance,
            "best_model_summary": selected_summary,
            "error_analysis": residual,
            "split_diagnostics": split_diag,
            "model_input_audit": input_audit,
            "training_report": report,
        },
    )

    baseline_primary = model_comparison[
        (model_comparison["model_name"] == "baseline_mean")
        & (model_comparison["split_name"] == args.primary_split)
        & (model_comparison["split_role"] == "test")
    ].iloc[0]
    print("Done.")
    print("Created:")
    for name in [
        "tau_eff_ml_model_comparison_step19.csv",
        "tau_eff_ml_metrics_step19.csv",
        "tau_eff_ml_predictions_step19.csv",
        "tau_eff_ml_best_model_predictions_step19.csv",
        "tau_eff_ml_final_all_samples_predictions_step19.csv",
        "tau_eff_ml_feature_importance_step19.csv",
        "tau_eff_ml_group_performance_step19.csv",
        "tau_eff_ml_residual_analysis_step19.csv",
        "tau_eff_ml_problem_samples_step19.csv",
        "tau_eff_ml_selected_model_summary_step19.csv",
        "tau_eff_ml_leakage_check_step19.csv",
        "step19_tau_eff_ml_report.txt",
        "step19_tau_eff_ml_notes.md",
        "starrydata2_step19_tau_eff_ml_model.xlsx",
    ]:
        print(f"- {name}")
    print("")
    print("Summary:")
    print(f"training samples: {len(ml_data)}")
    print(f"feature count: {len(feature_cols)}")
    print(f"selected model: {selected_model}")
    print(f"selection validation log RMSE: {selection_metric_value}")
    print(f"primary DOI test log RMSE: {primary_row['log_tau_rmse']}")
    print(f"primary DOI test R2: {primary_row['log_tau_r2']}")
    print(f"primary DOI test Spearman: {primary_row['log_tau_spearman']}")
    print(f"primary DOI test tau_eff MAPE: {primary_row['tau_eff_mape']}")
    print(f"primary DOI within factor 2 rate: {primary_row['within_tau_factor_2_rate']}")
    print(f"baseline DOI test log RMSE: {baseline_primary['log_tau_rmse']}")
    print(f"best model improvement over baseline: {baseline_primary['log_tau_rmse'] - primary_row['log_tau_rmse']}")
    print(f"problem samples: {len(problem)}")
    print(f"leakage feature count: {len(leakage_cols)}")
    print(f"DOI leakage count: {int(split_diag.loc[split_diag['split_name'].eq(args.primary_split), 'doi_overlap_count'].iloc[0])}")


if __name__ == "__main__":
    main()
