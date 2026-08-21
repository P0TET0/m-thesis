#!/usr/bin/env python3
"""Rebuild a fair, common DOI-group CV for exp005 ML/DF and exp006.

The two predictive methods are refit in every fold on the same training-sample
universe.  All samples belonging to one DOI stay in one fold.  DOI groups that
contain reported high-ZT samples are greedily spread across the five folds.

exp005 DF is retained as a per-sample fitting reference: it uses the fitted
Step12 tau_eff and therefore is not an out-of-sample learning model.  It is
evaluated on the same fold-partitioned rows, temperatures, S, kappa, and
reported ZT as the two out-of-fold predictive methods.
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import compute_pf_zt_from_exp006_sigma as pfzt
from compare_exp005_exp006_common_cohort_zt import (
    add_target_metadata,
    add_targeted_prefactors,
    normalized_electrical_baseline,
)


ROOT = Path(__file__).resolve().parent
STEP18 = ROOT / "data" / "output" / "starrydata2_step18_tau_eff_ml_dataset"
DEFAULT_EXP006 = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step5a_validation_rows_with_splits.parquet"
)
DEFAULT_CURVES = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step5_core_properties"
    / "candidate_core_curves_step5.csv"
)
DEFAULT_TAU_FIT = (
    ROOT
    / "data"
    / "output"
    / "starrydata2_step12_tau_fit"
    / "tau_fit_results_step12.csv"
)
DEFAULT_OUTPUT = ROOT / "data" / "output" / "exp005_exp006_common_doi_group_cv"

N_FOLDS = 5
SEED = 42
ZT_THRESHOLD = 1.0
TOLERANCE_K = 2.0
EPS = 1.0e-12

METHOD_LABELS = {
    "exp005_df": "exp005 DF\n(per-sample fit reference)",
    "exp005_ml": "exp005 ML\n(DOI-group OOF)",
    "exp006": "exp006\n(DOI-group OOF)",
}
METHOD_COLORS = {
    "exp005_df": "#4C78A8",
    "exp005_ml": "#59A14F",
    "exp006": "#E15759",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp006-rows", type=Path, default=DEFAULT_EXP006)
    parser.add_argument("--curve-source", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--tau-fit-results", type=Path, default=DEFAULT_TAU_FIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_doi(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip().str.casefold()
    out = out.str.replace(r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
    out = out.str.replace(r"^doi:\s*", "", regex=True)
    return out.str.rstrip("/.")


def true_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def read_step18() -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_csv(STEP18 / "tau_eff_ml_feature_matrix_step18.csv", low_memory=False)
    target = pd.read_csv(STEP18 / "tau_eff_ml_target_step18.csv", low_memory=False)
    metadata = pd.read_csv(
        STEP18 / "tau_eff_ml_metadata_step18.csv",
        usecols=["sample_key", "DOI", "composition", "material_system", "n_or_p"],
        low_memory=False,
    )
    data = features.merge(target, on="sample_key", how="inner", validate="one_to_one")
    data = data.merge(metadata, on="sample_key", how="left", validate="one_to_one")
    valid = (
        true_series(data["use_for_tau_eff_ml_step18"])
        & finite(data["target_log_tau_eff_step18"]).notna()
        & finite(data["target_tau_eff_step18"]).gt(0)
    )
    data = data.loc[valid].copy()
    data["sample_key"] = data["sample_key"].astype(str)
    data["doi_normalized_exp005"] = normalize_doi(data["DOI"])
    feature_cols = [column for column in features.columns if column != "sample_key"]
    return data, feature_cols


def read_exp006(path: Path) -> pd.DataFrame:
    columns = [
        "row_id",
        "paper_id",
        "doi",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_family_raw",
        "T_K",
        "S_uV_per_K",
        "sigma_S_per_m",
        "carrier_type",
        "F0_eta",
        "sigma0_S_per_m",
        "log10_sigma0_S_per_m",
        "T_bin_index",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "T_bin_label",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "material_group_key",
        "is_valid_sigma0_bool",
        "is_conservative_valid_sigma0_bool",
    ]
    frame = pd.read_parquet(path, columns=columns)
    frame["sample_key"] = frame["sample_key"].astype(str)
    frame["doi_normalized_exp006"] = normalize_doi(frame["doi"])
    return frame


def attach_curve_observations(rows: pd.DataFrame, curve_source: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    observations, curve_stats = pfzt.load_curve_observations(
        curve_source, set(rows["sample_key"]), 3
    )
    matched, match_stats = pfzt.match_curve_observations(
        rows[["sample_key", "T_K"]], observations, TOLERANCE_K
    )
    out = pd.concat([rows.reset_index(drop=True), matched.reset_index(drop=True)], axis=1)
    out["seebeck_obs_V_per_K"] = finite(out["S_uV_per_K"]) * 1.0e-6
    return out, {**curve_stats, **match_stats}


def build_common_universe(
    exp005: pd.DataFrame, exp006: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    exp005_keys = set(exp005["sample_key"])
    rows = exp006.loc[exp006["sample_key"].isin(exp005_keys)].copy()
    doi_map = exp005[["sample_key", "doi_normalized_exp005"]]
    rows = rows.merge(doi_map, on="sample_key", how="left", validate="many_to_one")
    doi_ok = (
        rows["doi_normalized_exp005"].ne("")
        & rows["doi_normalized_exp006"].ne("")
        & rows["doi_normalized_exp005"].eq(rows["doi_normalized_exp006"])
    )
    mismatched_samples = int(rows.loc[~doi_ok, "sample_key"].nunique())
    rows = rows.loc[doi_ok].copy()
    common_keys = set(rows["sample_key"])
    samples = exp005.loc[exp005["sample_key"].isin(common_keys)].copy()
    samples = samples.drop_duplicates("sample_key").reset_index(drop=True)
    stats = {
        "exp005_trainable_samples": int(len(exp005)),
        "exp006_valid_rows_input": int(len(exp006)),
        "common_samples_before_reported_filter": int(len(samples)),
        "common_dois_before_reported_filter": int(samples["doi_normalized_exp005"].nunique()),
        "common_exp006_rows_before_reported_filter": int(len(rows)),
        "doi_mismatch_samples_removed": mismatched_samples,
    }
    return samples, rows, stats


def sample_observation_summary(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["ZT_obs"] = finite(work["ZT_obs"])
    work["has_kappa_zt"] = finite(work["kappa_obs_W_per_mK"]).gt(0) & work["ZT_obs"].notna()
    return (
        work.groupby("sample_key", sort=False)
        .agg(
            max_ZT_obs=("ZT_obs", "max"),
            reported_ZT_rows=("ZT_obs", lambda s: int(s.notna().sum())),
            evaluable_rows=("has_kappa_zt", "sum"),
        )
        .reset_index()
    )


def assign_doi_folds(samples: pd.DataFrame, n_folds: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2")
    work = samples.copy()
    work["is_high_ZT_sample"] = finite(work["max_ZT_obs"]).ge(ZT_THRESHOLD)
    work["is_evaluable_sample"] = finite(work["evaluable_rows"]).gt(0)
    grouped = (
        work.groupby("doi_normalized_exp005", sort=False)
        .agg(
            sample_count=("sample_key", "nunique"),
            evaluable_sample_count=("is_evaluable_sample", "sum"),
            high_ZT_sample_count=("is_high_ZT_sample", "sum"),
            max_ZT_obs=("max_ZT_obs", "max"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    grouped["tie_break"] = rng.random(len(grouped))
    grouped = grouped.sort_values(
        ["high_ZT_sample_count", "evaluable_sample_count", "sample_count", "max_ZT_obs", "tie_break"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    totals = {
        "high": np.zeros(n_folds, dtype=float),
        "eval": np.zeros(n_folds, dtype=float),
        "samples": np.zeros(n_folds, dtype=float),
        "dois": np.zeros(n_folds, dtype=float),
    }
    target = {
        "high": max(float(grouped["high_ZT_sample_count"].sum()) / n_folds, 1.0),
        "eval": max(float(grouped["evaluable_sample_count"].sum()) / n_folds, 1.0),
        "samples": max(float(grouped["sample_count"].sum()) / n_folds, 1.0),
        "dois": max(float(len(grouped)) / n_folds, 1.0),
    }
    assigned: list[int] = []
    for row in grouped.itertuples(index=False):
        scores: list[float] = []
        for fold in range(n_folds):
            new_high = totals["high"][fold] + row.high_ZT_sample_count
            new_eval = totals["eval"][fold] + row.evaluable_sample_count
            new_samples = totals["samples"][fold] + row.sample_count
            new_dois = totals["dois"][fold] + 1
            score = (
                10.0 * (new_high / target["high"]) ** 2
                + 4.0 * (new_eval / target["eval"]) ** 2
                + 2.0 * (new_samples / target["samples"]) ** 2
                + (new_dois / target["dois"]) ** 2
            )
            scores.append(score)
        best = int(np.argmin(scores))
        assigned.append(best)
        totals["high"][best] += row.high_ZT_sample_count
        totals["eval"][best] += row.evaluable_sample_count
        totals["samples"][best] += row.sample_count
        totals["dois"][best] += 1
    grouped["fold"] = assigned
    fold_map = grouped.set_index("doi_normalized_exp005")["fold"]
    work["fold"] = work["doi_normalized_exp005"].map(fold_map).astype(int)
    fold_summary = (
        work.groupby("fold")
        .agg(
            DOI_count=("doi_normalized_exp005", "nunique"),
            sample_count=("sample_key", "nunique"),
            evaluable_sample_count=("is_evaluable_sample", "sum"),
            high_ZT_sample_count=("is_high_ZT_sample", "sum"),
            max_ZT_obs=("max_ZT_obs", "max"),
        )
        .reset_index()
    )
    return work, fold_summary


def numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame[columns].copy()
    for column in columns:
        if output[column].dtype == bool:
            output[column] = output[column].astype(int)
            continue
        low = output[column].astype(str).str.strip().str.lower()
        unique = set(low.dropna().unique())
        if unique.issubset({"true", "false", "1", "0", "yes", "no", "nan", "<na>", ""}):
            output[column] = low.map(
                {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}
            )
        else:
            output[column] = pd.to_numeric(output[column], errors="coerce")
        output[column] = output[column].replace([np.inf, -np.inf], np.nan)
    return output


def exp005_ml_oof(samples: pd.DataFrame, feature_cols: list[str], seed: int) -> pd.DataFrame:
    base = numeric_features(samples, feature_cols)
    target = finite(samples["target_log_tau_eff_step18"])
    records: list[pd.DataFrame] = []
    for fold in sorted(samples["fold"].unique()):
        train_mask = samples["fold"].ne(fold)
        test_mask = samples["fold"].eq(fold)
        X_train = base.loc[train_mask].copy()
        X_test = base.loc[test_mask].copy()
        all_missing = X_train.columns[X_train.isna().all()].tolist()
        X_train = X_train.drop(columns=all_missing)
        X_test = X_test.drop(columns=all_missing)
        medians = X_train.median(numeric_only=True).fillna(0.0)
        X_train = X_train.fillna(medians).fillna(0.0)
        X_test = X_test.fillna(medians).fillna(0.0)
        zero_variance = X_train.columns[X_train.var(axis=0).le(0)].tolist()
        X_train = X_train.drop(columns=zero_variance)
        X_test = X_test.drop(columns=zero_variance)
        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
        model.fit(X_train, target.loc[train_mask])
        predicted_log = model.predict(X_test)
        records.append(
            pd.DataFrame(
                {
                    "sample_key": samples.loc[test_mask, "sample_key"].to_numpy(),
                    "fold": int(fold),
                    "target_log_tau_eff_step18": target.loc[test_mask].to_numpy(),
                    "target_tau_eff_step18": np.exp(target.loc[test_mask].to_numpy()),
                    "pred_log_tau_eff_exp005_ml_oof": predicted_log,
                    "pred_tau_eff_exp005_ml_oof": np.exp(np.clip(predicted_log, -50, 50)),
                    "exp005_ml_feature_count": X_train.shape[1],
                    "exp005_ml_train_sample_count": int(train_mask.sum()),
                    "exp005_ml_train_DOI_count": int(
                        samples.loc[train_mask, "doi_normalized_exp005"].nunique()
                    ),
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def exp006_reference_for_fold(train_rows: pd.DataFrame) -> pd.DataFrame:
    train = train_rows.loc[true_series(train_rows["is_conservative_valid_sigma0_bool"])].copy()
    group_cols = [
        "material_group_key",
        "carrier_type",
        "T_bin_index",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "T_bin_label",
    ]
    sample_cols = group_cols + ["validation_sample_group_id"]
    sample_values = (
        train.groupby(sample_cols, dropna=False, sort=False)
        .agg(sample_log10_sigma0=("log10_sigma0_S_per_m", "median"))
        .reset_index()
    )
    counts = (
        train.groupby(group_cols, dropna=False, sort=False)
        .agg(
            train_row_count=("row_id", "size"),
            train_sample_count=("validation_sample_group_id", "nunique"),
            train_paper_count=("validation_paper_group_id", "nunique"),
        )
        .reset_index()
    )
    reference = (
        sample_values.groupby(group_cols, dropna=False, sort=False)
        .agg(log10_sigma0_ref_S_per_m=("sample_log10_sigma0", "median"))
        .reset_index()
        .merge(counts, on=group_cols, how="left", validate="one_to_one")
    )
    reference["is_reference_bin_candidate"] = (
        reference["train_row_count"].ge(3)
        & reference["train_sample_count"].ge(3)
        & reference["train_paper_count"].ge(1)
    )
    reference["sigma0_ref_S_per_m"] = 10.0 ** finite(reference["log10_sigma0_ref_S_per_m"])
    return reference


def exp006_oof(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = [
        "material_group_key",
        "carrier_type",
        "T_bin_index",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "T_bin_label",
    ]
    predictions: list[pd.DataFrame] = []
    reference_summaries: list[dict[str, int]] = []
    for fold in sorted(rows["fold"].unique()):
        train = rows.loc[rows["fold"].ne(fold)]
        test = rows.loc[rows["fold"].eq(fold) & true_series(rows["is_valid_sigma0_bool"])].copy()
        reference = exp006_reference_for_fold(train)
        test = test.merge(reference, on=group_cols, how="left", validate="many_to_one")
        ok = (
            test["is_reference_bin_candidate"].eq(True)
            & finite(test["sigma0_ref_S_per_m"]).gt(0)
            & finite(test["F0_eta"]).gt(0)
        )
        test["prediction_status_exp006_oof"] = np.where(ok, "ok", "unavailable")
        test["sigma_pred_exp006_oof_S_per_m"] = np.where(
            ok, finite(test["sigma0_ref_S_per_m"]) * finite(test["F0_eta"]), np.nan
        )
        predictions.append(test)
        reference_summaries.append(
            {
                "fold": int(fold),
                "train_rows": int(len(train)),
                "train_samples": int(train["sample_key"].nunique()),
                "train_DOIs": int(train["doi_normalized_exp005"].nunique()),
                "reference_bins_total": int(len(reference)),
                "reference_bins_eligible": int(reference["is_reference_bin_candidate"].sum()),
                "test_rows_all_valid": int(len(test)),
                "test_rows_prediction_ok": int(ok.sum()),
                "test_samples_prediction_ok": int(test.loc[ok, "sample_key"].nunique()),
            }
        )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(reference_summaries)


def add_exp005_predictions(
    rows: pd.DataFrame,
    sample_oof: pd.DataFrame,
    curve_source: Path,
    tau_fit_results: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    baseline, baseline_stats = normalized_electrical_baseline(curve_source, tau_fit_results)
    tau_cols = [
        "sample_key",
        "target_tau_eff_step18",
        "pred_tau_eff_exp005_ml_oof",
        "pred_log_tau_eff_exp005_ml_oof",
    ]
    out = rows.merge(sample_oof[tau_cols], on="sample_key", how="left", validate="many_to_one")
    out = out.rename(columns={"T_K": "temperature_K"})
    out = add_target_metadata(out, baseline)
    out = add_targeted_prefactors(out, baseline)
    prefactor = finite(out["prefactor_C_S_per_m_step12_rebuilt"])
    out["sigma_pred_exp005_df_S_per_m"] = prefactor * finite(out["target_tau_eff_step18"])
    out["sigma_pred_exp005_ml_S_per_m"] = prefactor * finite(out["pred_tau_eff_exp005_ml_oof"])
    return out, baseline_stats


def calculate_zt(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    temperature = finite(out["temperature_K"])
    seebeck = finite(out["seebeck_obs_V_per_K"])
    kappa = finite(out["kappa_obs_W_per_mK"])
    common = temperature.gt(0) & seebeck.notna() & kappa.gt(0)
    sigma_columns = {
        "exp005_df": "sigma_pred_exp005_df_S_per_m",
        "exp005_ml": "sigma_pred_exp005_ml_S_per_m",
        "exp006": "sigma_pred_exp006_oof_S_per_m",
    }
    for method, sigma_column in sigma_columns.items():
        sigma = finite(out[sigma_column])
        valid = common & sigma.gt(0)
        out[f"ZT_pred_{method}"] = np.where(
            valid, np.square(seebeck) * sigma * temperature / kappa, np.nan
        )
    out["ZT_obs"] = finite(out["ZT_obs"])
    out["strict_common_row"] = out["ZT_obs"].notna()
    for method in METHOD_LABELS:
        out["strict_common_row"] &= finite(out[f"ZT_pred_{method}"]).notna()
    return out


def aggregate_samples(rows: pd.DataFrame) -> pd.DataFrame:
    strict = rows.loc[rows["strict_common_row"]].copy()
    agg: dict[str, tuple[str, str]] = {
        "fold": ("fold", "first"),
        "DOI": ("doi", "first"),
        "doi_normalized": ("doi_normalized_exp005", "first"),
        "composition": ("composition", "first"),
        "material_system": ("material_system", "first"),
        "n_or_p": ("n_or_p", "first"),
        "row_count": ("row_id", "size"),
        "max_ZT_obs": ("ZT_obs", "max"),
    }
    for method in METHOD_LABELS:
        agg[f"max_ZT_pred_{method}"] = (f"ZT_pred_{method}", "max")
    out = strict.groupby("sample_key", sort=False).agg(**agg).reset_index()
    for method in METHOD_LABELS:
        out[f"error_max_ZT_{method}"] = out[f"max_ZT_pred_{method}"] - out["max_ZT_obs"]
        out[f"abs_error_max_ZT_{method}"] = out[f"error_max_ZT_{method}"].abs()
    return out


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    actual = finite(actual)
    predicted = finite(predicted)
    ok = actual.notna() & predicted.notna()
    actual = actual.loc[ok].to_numpy(float)
    predicted = predicted.loc[ok].to_numpy(float)
    if len(actual) == 0:
        return {"n": 0}
    error = predicted - actual
    positive = (actual > 0) & (predicted > 0)
    log_error = np.log10(predicted[positive] / actual[positive])
    ratio = predicted[positive] / actual[positive]
    return {
        "n": int(len(actual)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
        "log10_RMSE": float(np.sqrt(np.mean(log_error**2))) if len(log_error) else np.nan,
        "within_factor_2": float(np.mean((ratio >= 0.5) & (ratio <= 2.0))) if len(ratio) else np.nan,
        "Pearson": float(pd.Series(actual).corr(pd.Series(predicted), method="pearson")) if len(actual) > 1 else np.nan,
        "Spearman": float(pd.Series(actual).corr(pd.Series(predicted), method="spearman")) if len(actual) > 1 else np.nan,
    }


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    a = finite(actual).ge(ZT_THRESHOLD)
    p = finite(predicted).ge(ZT_THRESHOLD)
    tp = int((a & p).sum())
    fp = int((~a & p).sum())
    fn = int((a & ~p).sum())
    tn = int((~a & ~p).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


def build_metrics(rows: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_rows = rows.loc[rows["strict_common_row"]]
    overall: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for method in METHOD_LABELS:
        for level, frame, actual_col, pred_col in [
            ("row", strict_rows, "ZT_obs", f"ZT_pred_{method}"),
            ("sample_max", samples, "max_ZT_obs", f"max_ZT_pred_{method}"),
        ]:
            record: dict[str, object] = {"method": method, "evaluation_level": level}
            record.update(regression_metrics(frame[actual_col], frame[pred_col]))
            if level == "sample_max":
                record.update(classification_metrics(frame[actual_col], frame[pred_col]))
            overall.append(record)
        for fold, frame in samples.groupby("fold"):
            record = {"method": method, "fold": int(fold), "evaluation_level": "sample_max"}
            record.update(regression_metrics(frame["max_ZT_obs"], frame[f"max_ZT_pred_{method}"]))
            record.update(classification_metrics(frame["max_ZT_obs"], frame[f"max_ZT_pred_{method}"]))
            folds.append(record)
    return pd.DataFrame(overall), pd.DataFrame(folds)


def shared_limits(samples: pd.DataFrame) -> tuple[float, float]:
    columns = ["max_ZT_obs", *[f"max_ZT_pred_{m}" for m in METHOD_LABELS]]
    values = pd.concat([finite(samples[c]) for c in columns], ignore_index=True).dropna()
    upper = max(1.1, float(values.quantile(0.995)) * 1.08)
    return -0.03, upper


def plot_parity(samples: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1), sharex=True, sharey=True)
    low, high = shared_limits(samples)
    primary = metrics.loc[metrics["evaluation_level"].eq("sample_max")].set_index("method")
    for ax, method in zip(axes, METHOD_LABELS):
        x = finite(samples["max_ZT_obs"])
        y = finite(samples[f"max_ZT_pred_{method}"])
        high_mask = x.ge(ZT_THRESHOLD)
        ax.scatter(x.loc[~high_mask], y.loc[~high_mask], s=13, alpha=0.35, color=METHOD_COLORS[method], edgecolors="none")
        ax.scatter(x.loc[high_mask], y.loc[high_mask], s=32, alpha=0.85, color="#F28E2B", edgecolors="black", linewidths=0.25, label="reported max ZT >= 1")
        ax.plot([low, high], [low, high], linestyle="--", color="#555555", linewidth=1.1)
        ax.axvline(ZT_THRESHOLD, color="#888888", linewidth=0.8, linestyle=":")
        ax.axhline(ZT_THRESHOLD, color="#888888", linewidth=0.8, linestyle=":")
        row = primary.loc[method]
        ax.text(0.04, 0.96, f"log-RMSE = {row['log10_RMSE']:.3f}\nSpearman = {row['Spearman']:.3f}\nF1(ZT>=1) = {row['F1']:.3f}", transform=ax.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        ax.set_title(METHOD_LABELS[method], fontsize=12)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Reported sample max ZT")
    axes[0].set_ylabel("Predicted sample max ZT")
    axes[-1].legend(loc="lower right", frameon=True, fontsize=9)
    fig.suptitle("Common DOI-group 5-fold CV: same samples, temperatures, S, kappa, and reported ZT", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_full_range_parity(samples: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    """Show every point; per-panel y limits prevent DF outliers hiding ML detail."""
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 5.1), sharex=True)
    x = finite(samples["max_ZT_obs"])
    x_low = min(-0.002, float(x.min()) * 1.05)
    x_high = float(x.max()) * 1.06
    primary = metrics.loc[metrics["evaluation_level"].eq("sample_max")].set_index("method")
    for ax, method in zip(axes, METHOD_LABELS):
        y = finite(samples[f"max_ZT_pred_{method}"])
        high_mask = x.ge(ZT_THRESHOLD)
        ax.scatter(x.loc[~high_mask], y.loc[~high_mask], s=12, alpha=0.32, color=METHOD_COLORS[method], edgecolors="none")
        ax.scatter(x.loc[high_mask], y.loc[high_mask], s=30, alpha=0.85, color="#F28E2B", edgecolors="black", linewidths=0.25)
        y_high = float(y.max()) * 1.06
        diagonal_high = min(x_high, y_high)
        ax.plot([0, diagonal_high], [0, diagonal_high], linestyle="--", color="#555555", linewidth=1.1)
        ax.axvline(ZT_THRESHOLD, color="#888888", linewidth=0.8, linestyle=":")
        ax.axhline(ZT_THRESHOLD, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(min(-0.002, float(y.min()) * 1.05), y_high)
        ax.set_yscale("symlog", linthresh=0.03, linscale=0.8)
        row = primary.loc[method]
        ax.text(0.04, 0.96, f"all n = {int(row['n']):,}\nmax predicted ZT = {y.max():.3g}", transform=ax.transAxes, va="top", fontsize=10, bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"})
        ax.set_title(METHOD_LABELS[method], fontsize=12)
        ax.grid(alpha=0.18)
        ax.set_xlabel("Reported sample max ZT (full range)")
        ax.set_ylabel("Predicted sample max ZT (symlog; full range)")
    fig.suptitle("Common DOI-group 5-fold CV: full-range parity (no ZT<1 restriction, no point clipping)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_performance(metrics: pd.DataFrame, path: Path) -> None:
    primary = metrics.loc[metrics["evaluation_level"].eq("sample_max")].set_index("method")
    specs = [
        ("log10_RMSE", "log10 RMSE (lower is better)", False),
        ("within_factor_2", "Within factor 2 (higher is better)", True),
        ("Spearman", "Spearman rank correlation (higher is better)", True),
        ("F1", "High-ZT F1 at ZT >= 1 (higher is better)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.0))
    methods = list(METHOD_LABELS)
    labels = [METHOD_LABELS[m].replace("\n", " ") for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    for ax, (column, title, higher) in zip(axes.flat, specs):
        values = primary.loc[methods, column].astype(float).to_numpy()
        bars = ax.bar(labels, values, color=colors, width=0.68)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        if higher:
            ax.set_ylim(0, max(1.0, float(np.nanmax(values)) * 1.15))
        else:
            ax.set_ylim(0, float(np.nanmax(values)) * 1.20)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=10)
        ax.tick_params(axis="x", labelrotation=10)
    fig.suptitle("Common DOI-group CV performance on the strict shared reported-ZT cohort", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fold_diagnostics(fold_summary: pd.DataFrame, fold_metrics: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.2), sharex=True)
    folds = fold_summary["fold"].astype(int).to_numpy()
    width = 0.34
    axes[0].bar(folds - width / 2, fold_summary["evaluable_sample_count"], width=width, color="#76B7B2", label="evaluable samples")
    ax2 = axes[0].twinx()
    ax2.bar(folds + width / 2, fold_summary["high_ZT_sample_count"], width=width, color="#F28E2B", label="high-ZT samples")
    axes[0].set_ylabel("Evaluable samples")
    ax2.set_ylabel("Reported max ZT >= 1 samples")
    axes[0].set_title("DOI fold balance")
    axes[0].grid(axis="y", alpha=0.18)
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    for method in METHOD_LABELS:
        subset = fold_metrics.loc[fold_metrics["method"].eq(method)].sort_values("fold")
        axes[1].plot(subset["fold"], subset["log10_RMSE"], marker="o", linewidth=2, color=METHOD_COLORS[method], label=METHOD_LABELS[method].replace("\n", " "))
    axes[1].set_xlabel("DOI fold")
    axes[1].set_ylabel("Sample-max log10 RMSE")
    axes[1].set_title("Performance stability across held-out DOI folds")
    axes[1].set_xticks(folds)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for values in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def write_notes(
    path: Path,
    stats: dict[str, int],
    fold_summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    primary = metrics.loc[metrics["evaluation_level"].eq("sample_max")].set_index("method")
    lines = [
        "# Common DOI-group cross-validation (exp005 DF / exp005 ML / exp006)",
        "",
        "## Primary result",
        "",
        f"- Common trainable universe: {stats['common_samples_before_reported_filter']:,} samples / {stats['common_dois_before_reported_filter']:,} DOI groups.",
        f"- Strict shared reported-ZT evaluation: {len(rows.loc[rows['strict_common_row']]):,} rows / {len(samples):,} samples / {samples['doi_normalized'].nunique():,} DOI groups.",
        f"- No ZT<1 cutoff is applied: the strict shared sample-max range is {samples['max_ZT_obs'].min():.4g} to {samples['max_ZT_obs'].max():.4g}; high-ZT samples (max ZT >= 1) = {int(samples['max_ZT_obs'].ge(ZT_THRESHOLD).sum())}.",
        "",
        "| method | sample-max log10 RMSE | factor-2 rate | Spearman | high-ZT F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_LABELS:
        row = primary.loc[method]
        lines.append(
            f"| {METHOD_LABELS[method].replace(chr(10), ' ')} | {row['log10_RMSE']:.4f} | {row['within_factor_2']:.4f} | {row['Spearman']:.4f} | {row['F1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## What was made common",
            "",
            "- The sample universe is the intersection of exp005 Step18 trainable samples and exp006 Step5A valid sigma0 rows, with normalized DOI agreement required.",
            "- One DOI is assigned to exactly one of five folds; no DOI appears in both train and test within a fold.",
            "- Reported high-ZT DOI groups were explicitly balanced across folds.",
            "- exp005 ML uses the historical GradientBoostingRegressor settings and fold-local feature imputation/variance filtering.",
            "- exp006 rebuilds material-family + carrier + 25 K-bin sample-median sigma0 references from fold training rows only. Eligibility remains >=3 rows, >=3 samples, >=1 DOI.",
            f"- Kappa and reported ZT are matched independently from the same original Step5 curves by same-sample nearest temperature within +/-{TOLERANCE_K:g} K.",
            "- PF/ZT for every method uses the same row temperature, observed S, and observed kappa.",
            "- Primary metrics use only strict rows where reported ZT and all three method predictions are available.",
            "",
            "## Important interpretation",
            "",
            "- exp005 ML and exp006 are genuine out-of-fold predictions under the common DOI split.",
            "- exp005 DF is a per-sample fitting reference, not a cross-sample predictive model: it uses the test sample's fitted Step12 tau_eff. Fold labels only partition its evaluation rows. Treat DF as an attainable fit/reference ceiling, not as a deployable OOF predictor.",
            "- The Step12 empirical prefactor is rebuilt from the original Step5 electrical curves with the historical leave-one-sample-out hierarchy. This preserves exp005's original target definition; it is shared by exp005 DF and ML.",
            "- Exact-bin exp006 prediction coverage is reported separately. Missing exp006 references are not silently replaced by a global fallback.",
            "",
            "## Fold balance",
            "",
            markdown_table(fold_summary),
            "",
            "## exp006 fold reference coverage",
            "",
            markdown_table(reference_summary),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def publish(stage: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; rerun with --overwrite: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(stage), str(destination))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    stage_parent = Path(tempfile.mkdtemp(prefix="common_doi_cv_", dir=output.parent))
    stage = stage_parent / output.name
    stage.mkdir(parents=True)
    try:
        exp005, feature_cols = read_step18()
        exp006 = read_exp006(args.exp006_rows.resolve())
        samples, rows, stats = build_common_universe(exp005, exp006)
        rows, observation_stats = attach_curve_observations(rows, args.curve_source.resolve())
        stats.update(observation_stats)
        observed = sample_observation_summary(rows)
        samples = samples.merge(observed, on="sample_key", how="left", validate="one_to_one")
        samples, fold_summary = assign_doi_folds(samples, args.folds, args.seed)
        fold_map = samples[["sample_key", "fold", "composition"]]
        rows = rows.merge(fold_map, on="sample_key", how="inner", validate="many_to_one")

        sample_oof = exp005_ml_oof(samples, feature_cols, args.seed)
        exp006_rows, reference_summary = exp006_oof(rows)
        model_rows, baseline_stats = add_exp005_predictions(
            exp006_rows,
            sample_oof,
            args.curve_source.resolve(),
            args.tau_fit_results.resolve(),
        )
        stats.update(baseline_stats)
        model_rows = calculate_zt(model_rows)
        sample_results = aggregate_samples(model_rows)
        metrics, fold_metrics = build_metrics(model_rows, sample_results)

        assignments = samples[
            [
                "sample_key",
                "DOI",
                "doi_normalized_exp005",
                "composition",
                "material_system",
                "n_or_p",
                "max_ZT_obs",
                "reported_ZT_rows",
                "evaluable_rows",
                "is_high_ZT_sample",
                "is_evaluable_sample",
                "fold",
            ]
        ].rename(columns={"doi_normalized_exp005": "doi_normalized"})

        write_csv(assignments, stage / "common_doi_cv_fold_assignments.csv")
        write_csv(fold_summary, stage / "common_doi_cv_fold_balance.csv")
        write_csv(sample_oof, stage / "exp005_ml_oof_tau_predictions.csv")
        write_csv(reference_summary, stage / "exp006_oof_reference_coverage.csv")
        write_csv(model_rows, stage / "common_doi_cv_row_predictions.csv")
        write_csv(sample_results, stage / "common_doi_cv_sample_max_predictions.csv")
        write_csv(metrics, stage / "common_doi_cv_metrics.csv")
        write_csv(fold_metrics, stage / "common_doi_cv_fold_metrics.csv")

        plot_parity(sample_results, metrics, stage / "figure_common_doi_cv_sample_max_parity.png")
        plot_full_range_parity(
            sample_results,
            metrics,
            stage / "figure_common_doi_cv_sample_max_parity_full_range.png",
        )
        plot_performance(metrics, stage / "figure_common_doi_cv_performance_summary.png")
        plot_fold_diagnostics(
            fold_summary,
            fold_metrics,
            stage / "figure_common_doi_cv_fold_stability_and_balance.png",
        )
        write_notes(
            stage / "README_common_doi_group_cv.md",
            stats,
            fold_summary,
            reference_summary,
            model_rows,
            sample_results,
            metrics,
        )
        write_csv(pd.DataFrame([stats]), stage / "common_doi_cv_input_counts.csv")
        publish(stage, output, args.overwrite)
    except Exception:
        shutil.rmtree(stage_parent, ignore_errors=True)
        raise
    else:
        shutil.rmtree(stage_parent, ignore_errors=True)
    print(f"output: {output}")
    print(metrics.loc[metrics['evaluation_level'].eq('sample_max')].to_string(index=False))


if __name__ == "__main__":
    main()
