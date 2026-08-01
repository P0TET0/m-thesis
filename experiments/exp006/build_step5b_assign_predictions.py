import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step5a_validation_rows_with_splits.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step5a_validation_rows_with_splits.csv"

SPLIT_SCHEMES = ["sample_holdout", "paper_holdout"]
SUBSETS = ["all_valid", "conservative_valid"]
GROUP_SCHEMES = ["global", "material_family"]
CURVE_METHODS = ["row_median", "sample_median"]
PREDICTION_STATUSES = {
    "ok",
    "missing_reference_bin",
    "unreliable_reference_bin",
    "invalid_sigma0_ref",
    "invalid_F0_eta",
}

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "T_K",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "carrier_type",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
    "sigma_source",
    "match_method",
    "sample_holdout_split",
    "paper_holdout_split",
    "sample_cv_fold",
    "paper_cv_fold",
]

CORE_COLUMNS = [
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "T_bin_center_K",
    "carrier_type",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "sample_holdout_split",
    "paper_holdout_split",
]

NUMERIC_COLUMNS = [
    "T_K",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
]

DROP_COLUMNS = [
    "row_id",
    "reject_reason",
    "T_K",
    "T_bin_center_K",
    "carrier_type",
    "sigma_S_per_m",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "paper_id",
    "sample_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_family_raw",
]

REFERENCE_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "material_group_key",
    "material_group_label",
    "carrier_type",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "log10_sigma0_ref_S_per_m",
    "sigma0_ref_S_per_m",
    "sigma0_raw_median_S_per_m",
    "log10_sigma0_q25",
    "log10_sigma0_q75",
    "log10_sigma0_iqr",
    "log10_sigma0_min",
    "log10_sigma0_max",
    "log10_sigma0_mean",
    "log10_sigma0_std",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "is_reference_bin_candidate",
    "reliability_level",
]

PREDICTION_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "prediction_status",
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
    "T_K",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "carrier_type",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "is_reference_bin_candidate",
    "reliability_level",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "squared_log10_sigma_pred_over_exp",
    "sigma_source",
    "match_method",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign Step5B train-only sigma0 reference predictions to test rows.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step5b] {message}", flush=True)


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def resolve_input(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Step5A validation rows not found: {explicit}")
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step5A validation rows not found in experiments/exp006/data/processed")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def validate_columns(df: pd.DataFrame) -> None:
    missing_core = sorted(set(CORE_COLUMNS) - set(df.columns))
    if missing_core:
        raise ValueError(f"input missing required analysis columns: {missing_core}")
    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""


def reject_reason(row: pd.Series) -> str:
    if not bool(row["is_valid_sigma0_bool"]):
        return "is_valid_sigma0_not_true"
    positive_columns = [
        ("sigma_S_per_m", "invalid_sigma"),
        ("F0_eta", "invalid_F0_eta"),
        ("sigma0_S_per_m", "invalid_sigma0"),
        ("T_K", "invalid_T_K"),
    ]
    for column, reason in positive_columns:
        value = row.get(column, np.nan)
        if not np.isfinite(value) or value <= 0:
            return reason
    finite_columns = [
        ("log10_sigma_S_per_m", "invalid_log10_sigma"),
        ("log10_sigma0_S_per_m", "invalid_log10_sigma0"),
        ("T_bin_center_K", "invalid_T_bin_center"),
    ]
    for column, reason in finite_columns:
        if not np.isfinite(row.get(column, np.nan)):
            return reason
    if str(row.get("carrier_type", "")) not in {"p", "n"}:
        return "invalid_carrier_type"
    if clean_text(row.get("validation_sample_group_id")) == "":
        return "missing_validation_sample_group_id"
    if clean_text(row.get("validation_paper_group_id")) == "":
        return "missing_validation_paper_group_id"
    if str(row.get("sample_holdout_split", "")) not in {"train", "test"}:
        return "invalid_sample_holdout_split"
    if str(row.get("paper_holdout_split", "")) not in {"train", "test"}:
        return "invalid_paper_holdout_split"
    return ""


def filter_usable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["is_valid_sigma0_bool"] = as_bool(work["is_valid_sigma0"])
    work["is_conservative_valid_sigma0_bool"] = as_bool(work["is_conservative_valid_sigma0"])
    for column in NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["reject_reason"] = work.apply(reject_reason, axis=1)
    usable = work[work["reject_reason"].eq("")].copy()
    dropped = work[~work["reject_reason"].eq("")].copy()
    dropped_out = pd.DataFrame(columns=DROP_COLUMNS)
    if not dropped.empty:
        for column in DROP_COLUMNS:
            dropped_out[column] = dropped[column] if column in dropped.columns else ""
    return usable.drop(columns=["reject_reason"]), dropped_out


def build_configs() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split_scheme in SPLIT_SCHEMES:
        for reference_subset in SUBSETS:
            for eval_subset in SUBSETS:
                for group_scheme in GROUP_SCHEMES:
                    for curve_method in CURVE_METHODS:
                        config_id = (
                            f"{split_scheme}__ref_{reference_subset}__eval_{eval_subset}"
                            f"__{group_scheme}__{curve_method}"
                        )
                        rows.append(
                            {
                                "config_id": config_id,
                                "split_scheme": split_scheme,
                                "reference_source_subset": reference_subset,
                                "eval_target_subset": eval_subset,
                                "group_scheme": group_scheme,
                                "curve_method": curve_method,
                            }
                        )
    return pd.DataFrame(rows)


def subset_mask(df: pd.DataFrame, subset_name: str) -> pd.Series:
    if subset_name == "all_valid":
        return df["is_valid_sigma0_bool"]
    if subset_name == "conservative_valid":
        return df["is_conservative_valid_sigma0_bool"]
    raise ValueError(f"unknown subset: {subset_name}")


def split_column(split_scheme: str) -> str:
    if split_scheme == "sample_holdout":
        return "sample_holdout_split"
    if split_scheme == "paper_holdout":
        return "paper_holdout_split"
    raise ValueError(f"unknown split_scheme: {split_scheme}")


def key_columns(group_scheme: str) -> list[str]:
    if group_scheme == "global":
        return ["carrier_type", "T_bin_center_K"]
    if group_scheme == "material_family":
        return ["material_group_key", "carrier_type", "T_bin_center_K"]
    raise ValueError(f"unknown group_scheme: {group_scheme}")


def prepare_group_keys(df: pd.DataFrame, group_scheme: str, for_prediction: bool = False) -> pd.DataFrame:
    out = df.copy()
    if group_scheme == "global":
        out["material_group_label"] = "ALL_MATERIALS"
        if for_prediction:
            out["material_group_key_for_prediction"] = "ALL"
        else:
            out["material_group_key"] = "ALL"
    else:
        out["material_group_key"] = out["material_group_key"].map(lambda value: clean_text(value) or "unknown_material_family")
        out["material_group_label"] = out["material_group_key"]
        if for_prediction:
            out["material_group_key_for_prediction"] = out["material_group_key"]
    return out


def reliability_level(row: pd.Series) -> str:
    if not bool(row["is_reference_bin_candidate"]):
        return "insufficient"
    if int(row["train_sample_count"]) >= 10 and int(row["train_paper_count"]) >= 3:
        return "high"
    if int(row["train_sample_count"]) >= 5 and int(row["train_paper_count"]) >= 2:
        return "medium"
    return "low"


def aggregate_reference_values(group: pd.DataFrame, values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    log_ref = float(values.median())
    return {
        "log10_sigma0_ref_S_per_m": log_ref,
        "sigma0_ref_S_per_m": 10.0**log_ref,
        "sigma0_raw_median_S_per_m": float(pd.to_numeric(group["sigma0_S_per_m"], errors="coerce").median()),
        "log10_sigma0_q25": float(values.quantile(0.25)),
        "log10_sigma0_q75": float(values.quantile(0.75)),
        "log10_sigma0_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "log10_sigma0_min": float(values.min()),
        "log10_sigma0_max": float(values.max()),
        "log10_sigma0_mean": float(values.mean()),
        "log10_sigma0_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "train_row_count": int(len(group)),
        "train_sample_count": int(group["validation_sample_group_id"].nunique(dropna=True)),
        "train_paper_count": int(group["validation_paper_group_id"].nunique(dropna=True)),
    }


def make_reference_rows(train: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    if train.empty:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    group_scheme = str(config["group_scheme"])
    curve_method = str(config["curve_method"])
    keyed = prepare_group_keys(train, group_scheme)
    grouping_cols = [
        "material_group_key",
        "material_group_label",
        "carrier_type",
        "T_bin_index",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "T_bin_label",
    ]
    rows: list[dict[str, Any]] = []
    if curve_method == "row_median":
        for keys, group in keyed.groupby(grouping_cols, dropna=False, sort=False):
            row = dict(zip(grouping_cols, keys))
            row.update(aggregate_reference_values(group, group["log10_sigma0_S_per_m"]))
            rows.append(row)
    elif curve_method == "sample_median":
        sample_cols = grouping_cols + ["validation_sample_group_id"]
        sample_values = (
            keyed.groupby(sample_cols, dropna=False, sort=False)
            .agg(sample_level_log10_sigma0_median=("log10_sigma0_S_per_m", "median"))
            .reset_index()
        )
        for keys, sample_group in sample_values.groupby(grouping_cols, dropna=False, sort=False):
            mask = np.ones(len(keyed), dtype=bool)
            for column, value in zip(grouping_cols, keys):
                mask &= keyed[column].eq(value).to_numpy()
            original_group = keyed[mask]
            row = dict(zip(grouping_cols, keys))
            row.update(aggregate_reference_values(original_group, sample_group["sample_level_log10_sigma0_median"]))
            rows.append(row)
    else:
        raise ValueError(f"unknown curve_method: {curve_method}")
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    for column in ["config_id", "split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"]:
        out[column] = config[column]
    return out


def assign_reference_flags(curves: pd.DataFrame, min_rows: int, min_samples: int, min_papers: int) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame(columns=REFERENCE_COLUMNS)
    out = curves.copy()
    out["is_reference_bin_candidate"] = (
        (out["train_row_count"] >= min_rows)
        & (out["train_sample_count"] >= min_samples)
        & (out["train_paper_count"] >= min_papers)
    )
    out["reliability_level"] = out.apply(reliability_level, axis=1)
    return out[REFERENCE_COLUMNS].sort_values(
        ["config_id", "material_group_key", "carrier_type", "T_bin_center_K"]
    ).reset_index(drop=True)


def build_reference_curves(rows: pd.DataFrame, configs: pd.DataFrame, min_rows: int, min_samples: int, min_papers: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, config in configs.iterrows():
        log(f"building reference: {config['config_id']}")
        split_col = split_column(str(config["split_scheme"]))
        train = rows[rows[split_col].eq("train") & subset_mask(rows, str(config["reference_source_subset"]))].copy()
        frames.append(make_reference_rows(train, config))
    curves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=REFERENCE_COLUMNS)
    return assign_reference_flags(curves, min_rows, min_samples, min_papers)


def merge_keys(group_scheme: str) -> list[str]:
    return ["config_id", *key_columns(group_scheme)]


def assign_predictions_for_config(rows: pd.DataFrame, reference: pd.DataFrame, config: pd.Series) -> pd.DataFrame:
    split_col = split_column(str(config["split_scheme"]))
    test = rows[rows[split_col].eq("test") & subset_mask(rows, str(config["eval_target_subset"]))].copy()
    group_scheme = str(config["group_scheme"])
    test = prepare_group_keys(test, group_scheme, for_prediction=True)
    for column in ["config_id", "split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"]:
        test[column] = config[column]
    ref = reference[reference["config_id"].eq(config["config_id"])].copy()
    keep_ref = [
        "config_id",
        "material_group_key",
        "carrier_type",
        "T_bin_center_K",
        "log10_sigma0_ref_S_per_m",
        "sigma0_ref_S_per_m",
        "train_row_count",
        "train_sample_count",
        "train_paper_count",
        "is_reference_bin_candidate",
        "reliability_level",
    ]
    merged = test.merge(ref[keep_ref], on=merge_keys(group_scheme), how="left", suffixes=("", "_ref"))
    status = np.full(len(merged), "ok", dtype=object)
    missing_ref = merged["sigma0_ref_S_per_m"].isna() & merged["is_reference_bin_candidate"].isna()
    status[missing_ref.to_numpy()] = "missing_reference_bin"
    unreliable = (~missing_ref) & (~merged["is_reference_bin_candidate"].eq(True))
    status[unreliable.to_numpy()] = "unreliable_reference_bin"
    sigma0_ref = pd.to_numeric(merged["sigma0_ref_S_per_m"], errors="coerce")
    invalid_sigma0_ref = (~missing_ref) & (~unreliable) & (~np.isfinite(sigma0_ref) | (sigma0_ref <= 0))
    status[invalid_sigma0_ref.to_numpy()] = "invalid_sigma0_ref"
    f0 = pd.to_numeric(merged["F0_eta"], errors="coerce")
    invalid_f0 = (~missing_ref) & (~unreliable) & (~invalid_sigma0_ref) & (~np.isfinite(f0) | (f0 <= 0))
    status[invalid_f0.to_numpy()] = "invalid_F0_eta"
    merged["prediction_status"] = status
    ok = merged["prediction_status"].eq("ok")
    sigma_exp = pd.to_numeric(merged["sigma_S_per_m"], errors="coerce")
    merged["sigma_pred_S_per_m"] = np.nan
    merged["log10_sigma_pred_S_per_m"] = np.nan
    merged["sigma_pred_over_exp"] = np.nan
    merged["log10_sigma_pred_over_exp"] = np.nan
    merged["abs_log10_sigma_pred_over_exp"] = np.nan
    merged["squared_log10_sigma_pred_over_exp"] = np.nan
    merged.loc[ok, "sigma_pred_S_per_m"] = sigma0_ref[ok] * f0[ok]
    pred = pd.to_numeric(merged["sigma_pred_S_per_m"], errors="coerce")
    merged.loc[ok, "log10_sigma_pred_S_per_m"] = np.log10(pred[ok])
    merged.loc[ok, "sigma_pred_over_exp"] = pred[ok] / sigma_exp[ok]
    ratio = pd.to_numeric(merged["sigma_pred_over_exp"], errors="coerce")
    merged.loc[ok, "log10_sigma_pred_over_exp"] = np.log10(ratio[ok])
    log_ratio = pd.to_numeric(merged["log10_sigma_pred_over_exp"], errors="coerce")
    merged.loc[ok, "abs_log10_sigma_pred_over_exp"] = log_ratio[ok].abs()
    merged.loc[ok, "squared_log10_sigma_pred_over_exp"] = log_ratio[ok] ** 2
    for column in PREDICTION_COLUMNS:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[PREDICTION_COLUMNS]


def assign_test_predictions(rows: pd.DataFrame, reference: pd.DataFrame, configs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, config in configs.iterrows():
        log(f"assigning predictions: {config['config_id']}")
        frames.append(assign_predictions_for_config(rows, reference, config))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)


def build_coverage_summary(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    config_cols = ["config_id", "split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"]
    for keys, group in predictions.groupby(config_cols, dropna=False, sort=False):
        row = dict(zip(config_cols, keys))
        ok = group["prediction_status"].eq("ok")
        ref_config = reference[reference["config_id"].eq(row["config_id"])]
        status_counts = group["prediction_status"].value_counts().sort_index().to_dict()
        row.update(
            {
                "test_rows": len(group),
                "prediction_ok_rows": int(ok.sum()),
                "prediction_unavailable_rows": int((~ok).sum()),
                "coverage_fraction": float(ok.mean()) if len(group) else np.nan,
                "p_test_rows": int(group["carrier_type"].eq("p").sum()),
                "n_test_rows": int(group["carrier_type"].eq("n").sum()),
                "p_prediction_ok_rows": int((ok & group["carrier_type"].eq("p")).sum()),
                "n_prediction_ok_rows": int((ok & group["carrier_type"].eq("n")).sum()),
                "reference_bins_total": len(ref_config),
                "reference_bins_reliable": int(ref_config["is_reference_bin_candidate"].sum()) if not ref_config.empty else 0,
                "prediction_status_counts": str(status_counts),
                "T_bin_count_test": group["T_bin_center_K"].nunique(),
                "material_family_count_test": group["material_group_key_for_prediction"].nunique(),
                "sample_count_test": group["validation_sample_group_id"].nunique(),
                "paper_count_test": group["validation_paper_group_id"].nunique(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_unavailable_summary(unavailable: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "config_id",
        "split_scheme",
        "reference_source_subset",
        "eval_target_subset",
        "group_scheme",
        "curve_method",
        "prediction_status",
        "carrier_type",
        "T_bin_center_K",
        "material_group_key_for_prediction",
    ]
    if unavailable.empty:
        return pd.DataFrame(columns=[*cols, "row_count"])
    return unavailable.groupby(cols, dropna=False).size().reset_index(name="row_count")


def default_config_id(group_scheme: str) -> str:
    return f"sample_holdout__ref_conservative_valid__eval_all_valid__{group_scheme}__sample_median"


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(
    output_dir: Path,
    suffix: str,
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    valid: pd.DataFrame,
    unavailable: pd.DataFrame,
    default: pd.DataFrame,
    global_default: pd.DataFrame,
    coverage: pd.DataFrame,
    unavailable_summary: pd.DataFrame,
    dropped: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_statuses: dict[str, str] = {}
    parquet_frames = {
        "step5b_train_reference_curve_bins": reference,
        "step5b_test_predictions": predictions,
        "step5b_test_predictions_valid": valid,
        "step5b_test_predictions_default": default,
        "step5b_test_predictions_global_default": global_default,
    }
    for base, frame in parquet_frames.items():
        frame.to_csv(output_dir / output_name(base, suffix, "csv"), index=False, encoding="utf-8-sig")
        ok, error = save_parquet(frame, output_dir / output_name(base, suffix, "parquet"))
        parquet_statuses[output_name(base, suffix, "parquet")] = "saved" if ok else f"not saved: {error}"
    unavailable.to_csv(output_dir / output_name("step5b_test_predictions_unavailable", suffix, "csv"), index=False, encoding="utf-8-sig")
    coverage.to_csv(output_dir / output_name("step5b_prediction_coverage_by_config", suffix, "csv"), index=False, encoding="utf-8-sig")
    unavailable_summary.to_csv(output_dir / output_name("step5b_prediction_unavailable_summary", suffix, "csv"), index=False, encoding="utf-8-sig")
    dropped.to_csv(output_dir / output_name("step5b_dropped_rows", suffix, "csv"), index=False, encoding="utf-8-sig")
    return parquet_statuses


def build_dropped_output(dropped: pd.DataFrame) -> pd.DataFrame:
    if dropped.empty:
        return pd.DataFrame(columns=DROP_COLUMNS)
    for column in DROP_COLUMNS:
        if column not in dropped.columns:
            dropped[column] = ""
    return dropped[DROP_COLUMNS]


def check_no_leakage(rows: pd.DataFrame, predictions: pd.DataFrame, config: pd.Series) -> bool:
    split_col = split_column(str(config["split_scheme"]))
    train = rows[rows[split_col].eq("train")]
    test_predictions = predictions[predictions["config_id"].eq(config["config_id"])]
    if str(config["split_scheme"]) == "sample_holdout":
        return train["validation_sample_group_id"].isin(test_predictions["validation_sample_group_id"]).sum() == 0
    return train["validation_paper_group_id"].isin(test_predictions["validation_paper_group_id"]).sum() == 0


def run_sanity_checks(
    input_rows: int,
    rows: pd.DataFrame,
    dropped: pd.DataFrame,
    configs: pd.DataFrame,
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    valid: pd.DataFrame,
    unavailable: pd.DataFrame,
    default: pd.DataFrame,
    global_default: pd.DataFrame,
    coverage: pd.DataFrame,
    full_run: bool,
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["input_rows_equal_used_plus_dropped"] = input_rows == len(rows) + len(dropped)
    checks["config_count_32"] = len(configs) == 32
    checks["reference_config_id_exists"] = "config_id" in reference.columns and reference["config_id"].notna().all()
    checks["prediction_config_id_exists"] = "config_id" in predictions.columns and predictions["config_id"].notna().all()
    checks["sample_holdout_no_leakage"] = all(
        check_no_leakage(rows, predictions, config)
        for _, config in configs[configs["split_scheme"].eq("sample_holdout")].iterrows()
    )
    checks["paper_holdout_no_leakage"] = all(
        check_no_leakage(rows, predictions, config)
        for _, config in configs[configs["split_scheme"].eq("paper_holdout")].iterrows()
    )
    checks["prediction_status_allowed"] = set(predictions["prediction_status"].dropna()).issubset(PREDICTION_STATUSES)
    ok = predictions["prediction_status"].eq("ok")
    not_ok = ~ok
    checks["ok_sigma0_ref_positive"] = bool(
        np.isfinite(predictions.loc[ok, "sigma0_ref_S_per_m"]).all()
        and (predictions.loc[ok, "sigma0_ref_S_per_m"] > 0).all()
    )
    checks["ok_F0_positive"] = bool(
        np.isfinite(predictions.loc[ok, "F0_eta"]).all() and (predictions.loc[ok, "F0_eta"] > 0).all()
    )
    checks["ok_sigma_pred_positive"] = bool(
        np.isfinite(predictions.loc[ok, "sigma_pred_S_per_m"]).all()
        and (predictions.loc[ok, "sigma_pred_S_per_m"] > 0).all()
    )
    checks["ok_log10_sigma_pred_finite"] = bool(np.isfinite(predictions.loc[ok, "log10_sigma_pred_S_per_m"]).all())
    checks["ok_ratio_positive"] = bool(
        np.isfinite(predictions.loc[ok, "sigma_pred_over_exp"]).all()
        and (predictions.loc[ok, "sigma_pred_over_exp"] > 0).all()
    )
    checks["ok_log10_ratio_finite"] = bool(np.isfinite(predictions.loc[ok, "log10_sigma_pred_over_exp"]).all())
    checks["not_ok_prediction_values_nan"] = bool(predictions.loc[not_ok, "sigma_pred_S_per_m"].isna().all())
    checks["sigma_pred_formula"] = bool(
        np.allclose(
            predictions.loc[ok, "sigma_pred_S_per_m"],
            predictions.loc[ok, "sigma0_ref_S_per_m"] * predictions.loc[ok, "F0_eta"],
            rtol=1e-10,
            atol=0.0,
        )
    )
    checks["log10_ratio_formula"] = bool(
        np.allclose(
            predictions.loc[ok, "log10_sigma_pred_over_exp"],
            np.log10(predictions.loc[ok, "sigma_pred_S_per_m"] / predictions.loc[ok, "sigma_S_per_m"]),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    checks["coverage_fraction_range"] = coverage["coverage_fraction"].dropna().between(0.0, 1.0).all()
    checks["coverage_config_id_unique"] = coverage["config_id"].is_unique
    checks["default_file_exists_nonempty"] = len(default) > 0
    checks["global_default_file_exists_nonempty"] = len(global_default) > 0
    checks["valid_file_only_ok"] = valid.empty or valid["prediction_status"].eq("ok").all()
    checks["unavailable_file_only_not_ok"] = unavailable.empty or ~unavailable["prediction_status"].eq("ok").any()
    if full_run:
        checks["full_prediction_ok_nonzero"] = int(ok.sum()) > 0
        checks["full_default_ok_nonzero"] = int(default["prediction_status"].eq("ok").sum()) > 0
    else:
        if int(ok.sum()) == 0:
            warnings.append("small test produced zero ok predictions")
        if int(default["prediction_status"].eq("ok").sum()) == 0:
            warnings.append("small test produced zero ok default predictions")
    for _, config in configs.iterrows():
        split_col = split_column(str(config["split_scheme"]))
        test_ids = set(rows[rows[split_col].eq("test")]["row_id"].astype(str))
        pred_ids = set(predictions[predictions["config_id"].eq(config["config_id"])]["row_id"].astype(str))
        if not pred_ids.issubset(test_ids):
            checks[f"test_rows_only_{config['config_id']}"] = False
            break
    else:
        checks["test_predictions_from_test_rows_only"] = True
    failures = [name for name, ok_value in checks.items() if not ok_value]
    return checks, failures, warnings


def numeric_summary(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"min={values.min():.6g}, median={values.median():.6g}, max={values.max():.6g}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    text = df.copy()
    text.columns = [str(col) for col in text.columns]
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *body])


def coverage_compare(coverage: pd.DataFrame, column: str) -> dict[str, float]:
    if coverage.empty:
        return {}
    return coverage.groupby(column)["coverage_fraction"].median().to_dict()


def write_report(
    report_path: Path,
    input_path: Path,
    input_rows: int,
    rows: pd.DataFrame,
    dropped: pd.DataFrame,
    configs: pd.DataFrame,
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    valid: pd.DataFrame,
    unavailable: pd.DataFrame,
    default: pd.DataFrame,
    global_default: pd.DataFrame,
    coverage: pd.DataFrame,
    checks: dict[str, bool],
    warnings: list[str],
    parquet_statuses: dict[str, str],
    elapsed_sec: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = predictions["prediction_status"].value_counts().to_dict()
    cov = coverage["coverage_fraction"].dropna()
    default_cov = coverage[coverage["config_id"].eq(default_config_id("material_family"))]["coverage_fraction"]
    global_default_cov = coverage[coverage["config_id"].eq(default_config_id("global"))]["coverage_fraction"]
    unavailable_reason_preview = (
        unavailable["prediction_status"].value_counts().head(10).to_frame("row_count").reset_index(names="prediction_status")
        if not unavailable.empty
        else pd.DataFrame(columns=["prediction_status", "row_count"])
    )
    lines = [
        "# Step5B Prediction Assignment Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- validation rows used: {len(rows)}",
        f"- dropped rows: {len(dropped)}",
        f"- config_count: {len(configs)}",
        f"- train reference curve bins: {len(reference)}",
        f"- reliable reference curve bins: {int(reference['is_reference_bin_candidate'].sum()) if not reference.empty else 0}",
        f"- test prediction rows: {len(predictions)}",
        f"- prediction_status counts: {status_counts}",
        f"- prediction_status == ok rows: {len(valid)}",
        f"- prediction_status != ok rows: {len(unavailable)}",
        f"- coverage_fraction summary: {numeric_summary(cov)}",
        f"- default coverage_fraction: {default_cov.iloc[0] if not default_cov.empty else 'n/a'}",
        f"- global default coverage_fraction: {global_default_cov.iloc[0] if not global_default_cov.empty else 'n/a'}",
        f"- sample_holdout coverage median: {coverage_compare(coverage, 'split_scheme').get('sample_holdout', 'n/a')}",
        f"- paper_holdout coverage median: {coverage_compare(coverage, 'split_scheme').get('paper_holdout', 'n/a')}",
        f"- global/material_family coverage median: {coverage_compare(coverage, 'group_scheme')}",
        f"- reference subset coverage median: {coverage_compare(coverage, 'reference_source_subset')}",
        f"- curve method coverage median: {coverage_compare(coverage, 'curve_method')}",
        f"- default ok rows: {int(default['prediction_status'].eq('ok').sum()) if not default.empty else 0}",
        f"- global default ok rows: {int(global_default['prediction_status'].eq('ok').sum()) if not global_default.empty else 0}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for name, status in parquet_statuses.items():
        lines.append(f"- {name}: {status}")
    if not dropped.empty:
        lines.extend(["", "## Dropped Reasons", "", str(dropped["reject_reason"].value_counts().to_dict())])
    lines.extend(
        [
            "",
            "## Prediction Unavailable Reasons",
            "",
            dataframe_to_markdown(unavailable_reason_preview),
            "",
            "## Coverage By Config",
            "",
            dataframe_to_markdown(coverage),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- Step5B builds reference curves from train rows only.")
    lines.append("- Step4 full-data reference curves are not read for independent validation.")
    lines.append("- Test-row sigma0_S_per_m is retained for diagnostics, but not used to compute sigma_pred.")
    lines.append("- This step creates point-level error columns; aggregate metrics such as MAE, RMSE, and factor accuracy belong to Step5C.")
    lines.append("- Step5C should use step5b_test_predictions_valid.csv for accuracy summaries.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    report_path = args.report or (REPORT_DIR / output_name("step5b_prediction_assignment_report", args.output_suffix, "md"))
    full_run = args.max_rows is None

    log("loading validation rows with splits...")
    df = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        df = df.head(args.max_rows).copy()
    input_rows = len(df)
    log(f"input rows: {input_rows}")
    log("validating required columns...")
    validate_columns(df)
    log("filtering usable validation rows...")
    rows, dropped_raw = filter_usable_rows(df)
    dropped = build_dropped_output(dropped_raw)
    log("preparing 32 prediction configs...")
    configs = build_configs()
    log("building train-only reference curves...")
    reference = build_reference_curves(rows, configs, args.min_rows_per_bin, args.min_samples_per_bin, args.min_papers_per_bin)
    log("assigning reference bins to test rows...")
    predictions = assign_test_predictions(rows, reference, configs)
    log("computing sigma_pred for available rows...")
    valid = predictions[predictions["prediction_status"].eq("ok")].copy()
    unavailable = predictions[~predictions["prediction_status"].eq("ok")].copy()
    default = predictions[predictions["config_id"].eq(default_config_id("material_family"))].copy()
    global_default = predictions[predictions["config_id"].eq(default_config_id("global"))].copy()
    log("building coverage summaries...")
    coverage = build_coverage_summary(predictions, reference)
    unavailable_summary = build_unavailable_summary(unavailable)
    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(
        input_rows,
        rows,
        dropped,
        configs,
        reference,
        predictions,
        valid,
        unavailable,
        default,
        global_default,
        coverage,
        full_run,
    )
    if failures:
        for failure in failures:
            print(f"[step5b] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("writing outputs...")
    parquet_statuses = write_outputs(
        args.output,
        args.output_suffix,
        reference,
        predictions,
        valid,
        unavailable,
        default,
        global_default,
        coverage,
        unavailable_summary,
        dropped,
    )
    write_report(
        report_path,
        input_path,
        input_rows,
        rows,
        dropped,
        configs,
        reference,
        predictions,
        valid,
        unavailable,
        default,
        global_default,
        coverage,
        checks,
        warnings,
        parquet_statuses,
        time.time() - started,
    )
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
