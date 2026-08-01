import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
DEFAULT_STEP6B_DIR = PROCESSED_DIR / "step6b_broad_family"
DEFAULT_STEP6C_DIR = PROCESSED_DIR / "step6c_broad_family"
DEFAULT_OUTPUT = PROCESSED_DIR / "step6d_broad_family_audit"
DEFAULT_REPORT = EXP_DIR / "reports" / "step6d_broad_family_audit" / "step6d_outlier_robustness_audit_report.md"

DEFAULT_CONFIGS = {
    "broad_material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "broad_paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

ORIGINAL_DEFAULT_LABELS = {
    "broad_material_family_default": "material_family_default",
    "broad_global_default": "global_default",
    "broad_paper_material_family_default": "paper_material_family_default",
    "broad_paper_global_default": "paper_global_default",
}

REQUIRED_PRED_COLS = [
    "config_id",
    "prediction_status",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
    "carrier_type",
    "T_K",
    "T_bin_center_K",
    "T_bin_label",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
    "sample_has_sign_change",
]

NUMERIC_COLS = [
    "T_K",
    "T_bin_center_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
]

SOURCE_COLS = [
    "source_file_S",
    "source_file_sigma",
    "source_property_label_S",
    "source_property_label_sigma",
    "source_unit_S",
    "source_unit_sigma_or_rho",
    "source_curve_id_S",
    "source_curve_id_sigma",
    "T_delta_K",
]

OUTLIER_COLS = [
    "config_id",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
    "carrier_type",
    "T_K",
    "T_bin_center_K",
    "T_bin_label",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_error_decades",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_ref_over_row_sigma0",
    "log10_sigma0_ref_over_row_sigma0",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
    "sample_has_sign_change",
    "error_direction",
    "error_severity",
    "likely_error_origin_hint",
    *SOURCE_COLS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step6D broad_family outlier and robustness audit.")
    parser.add_argument("--step6b-dir", type=Path, default=DEFAULT_STEP6B_DIR)
    parser.add_argument("--step6c-dir", type=Path, default=DEFAULT_STEP6C_DIR)
    parser.add_argument("--original-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--metadata-input", type=Path, default=PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet")
    parser.add_argument("--step3-input", type=Path, default=PROCESSED_DIR / "step3_sigma0_valid.parquet")
    parser.add_argument("--step0-input", type=Path, default=Path("data/processed/step0_te_analysis_base.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--top-n-outliers", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step6d] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def read_preferred(base: Path) -> pd.DataFrame:
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if parquet.exists():
        return read_table(parquet)
    return read_table(csv)


def out_name(base: str, suffix: str) -> str:
    return f"{base}{suffix}.csv"


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_optional_table(paths: list[Path], warnings: list[str], label: str) -> pd.DataFrame | None:
    path = first_existing(paths)
    if path is None:
        warnings.append(f"optional {label} not found: {paths}")
        return None
    try:
        return read_table(path)
    except Exception as exc:
        warnings.append(f"optional {label} could not be read: {path}: {exc}")
        return None


def prepare_predictions(df: pd.DataFrame, max_rows_per_config: int | None = None) -> pd.DataFrame:
    require_columns(df, REQUIRED_PRED_COLS, "prediction valid rows")
    if max_rows_per_config is not None:
        if max_rows_per_config <= 0:
            raise ValueError("--max-rows-per-config must be positive")
        df = df.groupby("config_id", sort=False, dropna=False).head(max_rows_per_config).copy()
    else:
        df = df.copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["sigma_pred_over_exp"] = df["sigma_pred_S_per_m"] / df["sigma_S_per_m"]
    df["log10_sigma_pred_over_exp"] = np.log10(df["sigma_pred_over_exp"])
    df["abs_error_decades"] = df["log10_sigma_pred_over_exp"].abs()
    df["log10_sigma0_ref_over_row_sigma0"] = df["log10_sigma0_ref_S_per_m"] - df["log10_sigma0_S_per_m"]
    df["sigma0_ref_over_row_sigma0"] = 10.0 ** df["log10_sigma0_ref_over_row_sigma0"]
    df["error_direction"] = np.select(
        [df["log10_sigma_pred_over_exp"] > 0, df["log10_sigma_pred_over_exp"] < 0],
        ["over_predicted", "under_predicted"],
        default="near_exact",
    )
    err = df["abs_error_decades"]
    df["error_severity"] = np.select(
        [err >= 10, err >= 5, err >= 2, err >= 1],
        ["extreme_ge_10_decades", "severe_ge_5_decades", "large_ge_2_decades", "factor10_or_more"],
        default="moderate_or_small",
    )
    low_q = df["log10_sigma_S_per_m"].quantile(0.001)
    high_q = df["log10_sigma_S_per_m"].quantile(0.999)
    df["likely_error_origin_hint"] = np.select(
        [
            df["log10_sigma0_ref_over_row_sigma0"] >= 5,
            df["log10_sigma0_ref_over_row_sigma0"] <= -5,
            df["log10_sigma_S_per_m"] <= low_q,
            df["log10_sigma_S_per_m"] >= high_q,
            df["train_sample_count"] < 5,
        ],
        [
            "sigma0_ref_much_larger_than_row_sigma0",
            "sigma0_ref_much_smaller_than_row_sigma0",
            "very_low_sigma_exp",
            "very_high_sigma_exp",
            "low_train_sample_count",
        ],
        default="other_or_needs_manual_check",
    )
    return df


def merge_metadata(pred: pd.DataFrame, metadata_tables: list[pd.DataFrame | None]) -> pd.DataFrame:
    out = pred.copy()
    for meta in metadata_tables:
        if meta is None or "row_id" not in meta.columns:
            continue
        add_cols = [col for col in ["row_id", "doi", *SOURCE_COLS] if col in meta.columns and (col == "row_id" or col not in out.columns)]
        if len(add_cols) <= 1:
            continue
        out = out.merge(meta[add_cols].drop_duplicates("row_id"), on="row_id", how="left")
    for col in ["doi", *SOURCE_COLS]:
        if col not in out.columns:
            out[col] = ""
    return out


def values_join(series: pd.Series, limit: int = 5) -> str:
    vals = [str(v) for v in series.dropna().astype(str).unique() if str(v) != ""]
    return " | ".join(vals[:limit])


def mode_value(series: pd.Series) -> str:
    vals = series.dropna()
    if vals.empty:
        return ""
    return str(vals.mode().iloc[0])


def compute_metrics(df: pd.DataFrame, full_n: int | None = None) -> dict[str, Any]:
    if full_n is None:
        full_n = len(df)
    err = df["log10_sigma_pred_over_exp"].replace([np.inf, -np.inf], np.nan).dropna()
    abs_err = err.abs()
    return {
        "n_rows": int(len(df)),
        "n_samples": int(df["validation_sample_group_id"].nunique(dropna=True)) if "validation_sample_group_id" in df.columns else 0,
        "n_papers": int(df["validation_paper_group_id"].nunique(dropna=True)) if "validation_paper_group_id" in df.columns else 0,
        "retained_row_fraction": float(len(df) / full_n) if full_n else np.nan,
        "mean_log10_error": float(err.mean()) if len(err) else np.nan,
        "median_log10_error": float(err.median()) if len(err) else np.nan,
        "mae_log10": float(abs_err.mean()) if len(abs_err) else np.nan,
        "rmse_log10": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
        "factor_2_accuracy": float((abs_err <= np.log10(2)).mean()) if len(abs_err) else np.nan,
        "factor_5_accuracy": float((abs_err <= np.log10(5)).mean()) if len(abs_err) else np.nan,
        "factor_10_accuracy": float((abs_err <= 1.0).mean()) if len(abs_err) else np.nan,
        "max_abs_log10_error": float(abs_err.max()) if len(abs_err) else np.nan,
        "extreme_ge_10_count": int((abs_err >= 10).sum()),
        "severe_ge_5_count": int((abs_err >= 5).sum()),
        "large_ge_2_count": int((abs_err >= 2).sum()),
    }


def filter_frame(df: pd.DataFrame, filter_label: str) -> pd.DataFrame:
    if filter_label == "no_filter":
        return df
    if filter_label == "exclude_abs_error_ge_10":
        return df[df["abs_error_decades"] < 10]
    if filter_label == "exclude_abs_error_ge_5":
        return df[df["abs_error_decades"] < 5]
    if filter_label == "exclude_abs_error_ge_3":
        return df[df["abs_error_decades"] < 3]
    if filter_label == "exclude_abs_error_ge_2":
        return df[df["abs_error_decades"] < 2]
    if filter_label.startswith("exclude_top_"):
        frac = {"exclude_top_0p1_percent_abs_error": 0.001, "exclude_top_0p5_percent_abs_error": 0.005, "exclude_top_1p0_percent_abs_error": 0.01}[filter_label]
        threshold = df["abs_error_decades"].quantile(1.0 - frac)
        return df[df["abs_error_decades"] < threshold]
    if filter_label == "only_high_or_medium_reference_reliability":
        return df[df["reliability_level"].astype(str).str.casefold().isin(["high", "medium"])]
    if filter_label == "exclude_low_train_sample_count_lt5":
        return df[df["train_sample_count"] >= 5]
    if filter_label == "exclude_match_method_nearest":
        if "match_method" not in df.columns:
            return df.iloc[0:0].copy()
        return df[df["match_method"].astype(str).str.casefold() != "nearest"]
    raise ValueError(filter_label)


def build_outlier_topn(base_default: pd.DataFrame, top_n: int) -> pd.DataFrame:
    out = base_default.sort_values("abs_error_decades", ascending=False).head(top_n).copy()
    for col in OUTLIER_COLS:
        if col not in out.columns:
            out[col] = ""
    return out[OUTLIER_COLS].copy()


def build_row_summary(pred: pd.DataFrame) -> pd.DataFrame:
    defaults = pred[pred["config_id"].isin(DEFAULT_CONFIGS.values())].copy()
    pivot = defaults.pivot_table(index="row_id", columns="config_id", values="abs_error_decades", aggfunc="max")
    rename = {v: k.replace("broad_", "") + "_error" for k, v in DEFAULT_CONFIGS.items()}
    pivot = pivot.rename(columns=rename).reset_index()
    grouped = pred.groupby("row_id", dropna=False)
    summary = grouped.agg(
        max_abs_error_decades=("abs_error_decades", "max"),
        median_abs_error_decades=("abs_error_decades", "median"),
        config_count=("config_id", "nunique"),
        extreme_ge_10_config_count=("error_severity", lambda s: int((s == "extreme_ge_10_decades").sum())),
        severe_ge_5_config_count=("abs_error_decades", lambda s: int((s >= 5).sum())),
        large_ge_2_config_count=("abs_error_decades", lambda s: int((s >= 2).sum())),
        paper_id=("paper_id", "first"),
        sample_id=("sample_id", "first"),
        sample_key=("sample_key", "first"),
        validation_sample_group_id=("validation_sample_group_id", "first"),
        validation_paper_group_id=("validation_paper_group_id", "first"),
        formula_raw=("formula_raw", "first"),
        material_name_raw=("material_name_raw", "first"),
        material_group_key=("material_group_key", "first"),
        carrier_type=("carrier_type", "first"),
        T_K=("T_K", "first"),
        sigma_S_per_m=("sigma_S_per_m", "first"),
        sigma0_S_per_m=("sigma0_S_per_m", "first"),
        likely_error_origin_hint=("likely_error_origin_hint", lambda s: mode_value(s)),
    ).reset_index()
    summary = summary.merge(pivot, on="row_id", how="left")
    for col in [
        "material_family_default_error",
        "global_default_error",
        "paper_material_family_default_error",
        "paper_global_default_error",
    ]:
        if col not in summary.columns:
            summary[col] = np.nan
    return summary.sort_values("max_abs_error_decades", ascending=False)


def build_sample_summary(base_default: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in base_default.groupby("validation_sample_group_id", dropna=False):
        rows.append(
            {
                "validation_sample_group_id": gid,
                "paper_id_examples": values_join(g["paper_id"]),
                "sample_id_examples": values_join(g["sample_id"]),
                "sample_key_examples": values_join(g["sample_key"], 3),
                "formula_raw_examples": values_join(g["formula_raw"], 3),
                "material_name_raw_examples": values_join(g["material_name_raw"], 3),
                "material_group_key_values": values_join(g["material_group_key"]),
                "row_count": int(g["row_id"].nunique()),
                "config_row_count": int(len(g)),
                "mean_abs_error_decades": float(g["abs_error_decades"].mean()),
                "median_abs_error_decades": float(g["abs_error_decades"].median()),
                "max_abs_error_decades": float(g["abs_error_decades"].max()),
                "extreme_ge_10_row_count": int((g["abs_error_decades"] >= 10).sum()),
                "severe_ge_5_row_count": int((g["abs_error_decades"] >= 5).sum()),
                "large_ge_2_row_count": int((g["abs_error_decades"] >= 2).sum()),
                "factor10_or_more_row_count": int((g["abs_error_decades"] >= 1).sum()),
                "fraction_factor10_or_more": float((g["abs_error_decades"] >= 1).mean()),
                "T_min_K": float(g["T_K"].min()),
                "T_max_K": float(g["T_K"].max()),
                "sigma_exp_min_S_per_m": float(g["sigma_S_per_m"].min()),
                "sigma_exp_max_S_per_m": float(g["sigma_S_per_m"].max()),
                "sigma0_row_median_S_per_m": float(g["sigma0_S_per_m"].median()),
                "dominant_error_direction": mode_value(g["error_direction"]),
                "dominant_likely_error_origin_hint": mode_value(g["likely_error_origin_hint"]),
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_error_decades", ascending=False)


def build_paper_summary(base_default: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gid, g in base_default.groupby("validation_paper_group_id", dropna=False):
        rows.append(
            {
                "validation_paper_group_id": gid,
                "paper_id_examples": values_join(g["paper_id"]),
                "doi_examples": values_join(g.get("doi", pd.Series(dtype=object))),
                "row_count": int(g["row_id"].nunique()),
                "sample_count": int(g["validation_sample_group_id"].nunique()),
                "material_group_key_values": values_join(g["material_group_key"]),
                "mean_abs_error_decades": float(g["abs_error_decades"].mean()),
                "median_abs_error_decades": float(g["abs_error_decades"].median()),
                "max_abs_error_decades": float(g["abs_error_decades"].max()),
                "extreme_ge_10_row_count": int((g["abs_error_decades"] >= 10).sum()),
                "severe_ge_5_row_count": int((g["abs_error_decades"] >= 5).sum()),
                "large_ge_2_row_count": int((g["abs_error_decades"] >= 2).sum()),
                "factor10_or_more_row_count": int((g["abs_error_decades"] >= 1).sum()),
                "fraction_factor10_or_more": float((g["abs_error_decades"] >= 1).mean()),
                "T_min_K": float(g["T_K"].min()),
                "T_max_K": float(g["T_K"].max()),
                "dominant_error_direction": mode_value(g["error_direction"]),
                "dominant_likely_error_origin_hint": mode_value(g["likely_error_origin_hint"]),
            }
        )
    return pd.DataFrame(rows).sort_values("max_abs_error_decades", ascending=False)


def build_context_rows(base_default: pd.DataFrame, outlier_top: pd.DataFrame) -> pd.DataFrame:
    top100 = outlier_top.head(100).copy()
    sample_ids = set(top100["validation_sample_group_id"].dropna())
    ctx = base_default[base_default["validation_sample_group_id"].isin(sample_ids)].copy()
    rank = {row_id: i + 1 for i, row_id in enumerate(top100["row_id"])}
    ctx["outlier_rank"] = ctx["row_id"].map(rank)
    ctx["is_original_outlier_row"] = ctx["row_id"].isin(rank)
    cols = [
        "outlier_rank",
        "is_original_outlier_row",
        "config_id",
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_group_key",
        "carrier_type",
        "T_K",
        "T_bin_center_K",
        "S_uV_per_K",
        "eta",
        "sigma_S_per_m",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_over_exp",
        "abs_error_decades",
        "sigma0_S_per_m",
        "sigma0_ref_S_per_m",
        "train_sample_count",
        "reliability_level",
        "sigma_source",
        "match_method",
    ]
    return ctx.sort_values(["is_original_outlier_row", "abs_error_decades"], ascending=[False, False]).head(5000)[cols].copy()


def robust_metrics_by_filter(pred: pd.DataFrame) -> pd.DataFrame:
    filter_labels = [
        "no_filter",
        "exclude_abs_error_ge_10",
        "exclude_abs_error_ge_5",
        "exclude_abs_error_ge_3",
        "exclude_abs_error_ge_2",
        "exclude_top_0p1_percent_abs_error",
        "exclude_top_0p5_percent_abs_error",
        "exclude_top_1p0_percent_abs_error",
        "only_high_or_medium_reference_reliability",
        "exclude_low_train_sample_count_lt5",
        "exclude_match_method_nearest",
    ]
    rows = []
    for label, config_id in DEFAULT_CONFIGS.items():
        base = pred[pred["config_id"].eq(config_id)].copy()
        for filter_label in filter_labels:
            part = filter_frame(base, filter_label)
            row = {"default_label": label, "config_id": config_id, "filter_label": filter_label}
            row.update(compute_metrics(part, len(base)))
            rows.append(row)
    return pd.DataFrame(rows)


def robust_metrics_by_config(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    filters = ["no_filter", "exclude_abs_error_ge_5", "exclude_top_1p0_percent_abs_error"]
    meta_cols = ["split_scheme", "reference_source_subset", "eval_target_subset", "group_scheme", "curve_method"]
    for config_id, base in pred.groupby("config_id", dropna=False):
        first = base.iloc[0]
        for filter_label in filters:
            part = filter_frame(base, filter_label)
            row = {"config_id": config_id, "filter_label": filter_label}
            for col in meta_cols:
                row[col] = first.get(col, "")
            row.update(compute_metrics(part, len(base)))
            rows.append(row)
    out = pd.DataFrame(rows)
    out["rank_by_mae_log10_within_filter"] = out.groupby("filter_label")["mae_log10"].rank(method="min", ascending=True)
    out["rank_by_factor2_within_filter"] = out.groupby("filter_label")["factor_2_accuracy"].rank(method="min", ascending=False)
    keep = [
        "config_id",
        "split_scheme",
        "reference_source_subset",
        "eval_target_subset",
        "group_scheme",
        "curve_method",
        "filter_label",
        "n_rows",
        "n_samples",
        "mae_log10",
        "rmse_log10",
        "factor_2_accuracy",
        "factor_10_accuracy",
        "max_abs_log10_error",
        "retained_row_fraction",
        "rank_by_mae_log10_within_filter",
        "rank_by_factor2_within_filter",
    ]
    return out[keep].sort_values(["filter_label", "rank_by_mae_log10_within_filter"]).copy()


def original_vs_broad_robust(original_pred: pd.DataFrame | None, broad_filter: pd.DataFrame, original_default_comparison: pd.DataFrame | None) -> pd.DataFrame:
    metric_names = ["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "coverage_fraction"]
    rows = []
    if original_pred is not None:
        orig = prepare_predictions(original_pred)
        orig_filter = robust_metrics_by_filter(orig)
        for broad_label, original_label in ORIGINAL_DEFAULT_LABELS.items():
            for filter_label in sorted(set(broad_filter["filter_label"])):
                b = broad_filter[(broad_filter["default_label"].eq(broad_label)) & (broad_filter["filter_label"].eq(filter_label))]
                o = orig_filter[(orig_filter["default_label"].eq(broad_label)) & (orig_filter["filter_label"].eq(filter_label))]
                if o.empty:
                    # Same config IDs, so broad labels are reused for original robust rows.
                    o = orig_filter[(orig_filter["config_id"].eq(DEFAULT_CONFIGS[broad_label])) & (orig_filter["filter_label"].eq(filter_label))]
                if b.empty or o.empty:
                    continue
                for metric in metric_names:
                    rows.append(
                        {
                            "default_label": original_label,
                            "filter_label": filter_label,
                            "metric_name": metric,
                            "original_value": o.iloc[0].get(metric, np.nan),
                            "broad_family_value": b.iloc[0].get(metric, np.nan),
                            "delta_broad_minus_original": b.iloc[0].get(metric, np.nan) - o.iloc[0].get(metric, np.nan),
                            "interpretation_hint": "lower_is_better" if metric in ["mae_log10", "rmse_log10"] else "higher_is_better",
                        }
                    )
    elif original_default_comparison is not None:
        for _, row in original_default_comparison.iterrows():
            if row.get("metric_weighting") != "row_equal":
                continue
            metric = row.get("metric_name")
            if metric not in metric_names:
                continue
            rows.append(
                {
                    "default_label": row.get("default_label"),
                    "filter_label": "no_filter",
                    "metric_name": metric,
                    "original_value": row.get("original_value"),
                    "broad_family_value": row.get("broad_family_value"),
                    "delta_broad_minus_original": row.get("delta_broad_minus_original"),
                    "interpretation_hint": row.get("interpretation_hint", ""),
                }
            )
    return pd.DataFrame(rows)


def contribution_tables(base_default: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for level, group_col in [("sample", "validation_sample_group_id"), ("paper", "validation_paper_group_id")]:
        total_abs = base_default["abs_error_decades"].sum()
        total_sq = (base_default["log10_sigma_pred_over_exp"] ** 2).sum()
        for gid, g in base_default.groupby(group_col, dropna=False):
            sum_abs = g["abs_error_decades"].sum()
            sum_sq = (g["log10_sigma_pred_over_exp"] ** 2).sum()
            rows.append(
                {
                    "aggregation_level": level,
                    "group_id": gid,
                    "row_count": int(g["row_id"].nunique()),
                    "sum_abs_error": float(sum_abs),
                    "fraction_of_total_abs_error": float(sum_abs / total_abs) if total_abs else np.nan,
                    "sum_squared_error": float(sum_sq),
                    "fraction_of_total_squared_error": float(sum_sq / total_sq) if total_sq else np.nan,
                    "max_abs_error": float(g["abs_error_decades"].max()),
                    "median_abs_error": float(g["abs_error_decades"].median()),
                    "material_group_key_values": values_join(g["material_group_key"]),
                    "paper_id_examples": values_join(g["paper_id"]),
                    "sample_id_examples": values_join(g["sample_id"]),
                    "likely_error_origin_hint_top": mode_value(g["likely_error_origin_hint"]),
                }
            )
    concentration = pd.DataFrame(rows).sort_values(["aggregation_level", "fraction_of_total_abs_error"], ascending=[True, False])
    summary_rows = []
    for level in ["sample", "paper"]:
        part = concentration[concentration["aggregation_level"].eq(level)].copy()
        for topn in [1, 5, 10]:
            summary_rows.append(
                {
                    "item": f"top{topn}_{level}{'s' if topn > 1 else ''}_fraction_of_total_abs_error",
                    "value": float(part["fraction_of_total_abs_error"].head(topn).sum()),
                    "comment": f"Top {topn} {level}(s) contribution to total absolute error.",
                }
            )
            summary_rows.append(
                {
                    "item": f"top{topn}_{level}{'s' if topn > 1 else ''}_fraction_of_total_squared_error",
                    "value": float(part["fraction_of_total_squared_error"].head(topn).sum()),
                    "comment": f"Top {topn} {level}(s) contribution to total squared error.",
                }
            )
    return concentration, pd.DataFrame(summary_rows)


def manual_review_shortlist(outlier_top: pd.DataFrame, row_summary: pd.DataFrame, sample_summary: pd.DataFrame, paper_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    priority = 1
    base_cols = [
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "paper_id",
        "doi",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_group_key",
        "carrier_type",
        "T_K",
        "S_uV_per_K",
        "eta",
        "sigma_S_per_m",
        "sigma_pred_S_per_m",
        "abs_error_decades",
        "error_direction",
        "likely_error_origin_hint",
        *SOURCE_COLS,
    ]
    for _, row in outlier_top.head(120).iterrows():
        item = {"review_priority": priority, "review_type": "row", "note_for_manual_review": "largest broad material_family default outlier"}
        for col in base_cols:
            item[col] = row.get(col, "")
        rows.append(item)
        priority += 1
    for _, row in row_summary[row_summary["extreme_ge_10_config_count"] > 1].head(40).iterrows():
        item = {"review_priority": priority, "review_type": "row", "note_for_manual_review": "row is extreme in multiple configs"}
        for col in base_cols:
            item[col] = row.get(col, "")
        rows.append(item)
        priority += 1
    for _, row in sample_summary.sort_values("fraction_factor10_or_more", ascending=False).head(20).iterrows():
        rows.append(
            {
                "review_priority": priority,
                "review_type": "sample",
                "row_id": "",
                "validation_sample_group_id": row["validation_sample_group_id"],
                "validation_paper_group_id": "",
                "paper_id": row["paper_id_examples"],
                "doi": "",
                "sample_id": row["sample_id_examples"],
                "sample_key": row["sample_key_examples"],
                "formula_raw": row["formula_raw_examples"],
                "material_name_raw": row["material_name_raw_examples"],
                "material_group_key": row["material_group_key_values"],
                "carrier_type": "",
                "T_K": "",
                "S_uV_per_K": "",
                "eta": "",
                "sigma_S_per_m": "",
                "sigma_pred_S_per_m": "",
                "abs_error_decades": row["max_abs_error_decades"],
                "error_direction": row["dominant_error_direction"],
                "likely_error_origin_hint": row["dominant_likely_error_origin_hint"],
                "note_for_manual_review": "sample has concentrated factor10-or-more errors",
            }
        )
        priority += 1
    for _, row in paper_summary.sort_values("fraction_factor10_or_more", ascending=False).head(20).iterrows():
        rows.append(
            {
                "review_priority": priority,
                "review_type": "paper",
                "row_id": "",
                "validation_sample_group_id": "",
                "validation_paper_group_id": row["validation_paper_group_id"],
                "paper_id": row["paper_id_examples"],
                "doi": row["doi_examples"],
                "sample_id": "",
                "sample_key": "",
                "formula_raw": "",
                "material_name_raw": "",
                "material_group_key": row["material_group_key_values"],
                "carrier_type": "",
                "T_K": "",
                "S_uV_per_K": "",
                "eta": "",
                "sigma_S_per_m": "",
                "sigma_pred_S_per_m": "",
                "abs_error_decades": row["max_abs_error_decades"],
                "error_direction": row["dominant_error_direction"],
                "likely_error_origin_hint": row["dominant_likely_error_origin_hint"],
                "note_for_manual_review": "paper has concentrated factor10-or-more errors",
            }
        )
        priority += 1
    out = pd.DataFrame(rows).drop_duplicates(["review_type", "row_id", "validation_sample_group_id", "validation_paper_group_id"]).head(200)
    for col in ["review_priority", "review_type", *base_cols, "note_for_manual_review"]:
        if col not in out.columns:
            out[col] = ""
    out["review_priority"] = range(1, len(out) + 1)
    return out[["review_priority", "review_type", *base_cols, "note_for_manual_review"]]


def readiness_summary(broad_filter: pd.DataFrame, robust_compare: pd.DataFrame, contribution_summary: pd.DataFrame, diff_summary: pd.DataFrame, original_compare: pd.DataFrame, base_default: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(criterion: str, status: str, value: Any, threshold: str, comment: str) -> None:
        rows.append({"criterion": criterion, "status": status, "value": value, "threshold_or_reason": threshold, "comment": comment})

    no_filter = broad_filter[(broad_filter["default_label"].eq("broad_material_family_default")) & (broad_filter["filter_label"].eq("no_filter"))].iloc[0]
    ex5 = broad_filter[(broad_filter["default_label"].eq("broad_material_family_default")) & (broad_filter["filter_label"].eq("exclude_abs_error_ge_5"))].iloc[0]
    orig_mae = original_compare[
        original_compare["default_label"].eq("material_family_default")
        & original_compare["metric_weighting"].eq("row_equal")
        & original_compare["metric_name"].eq("mae_log10")
    ]["original_value"]
    orig_factor2 = original_compare[
        original_compare["default_label"].eq("material_family_default")
        & original_compare["metric_weighting"].eq("row_equal")
        & original_compare["metric_name"].eq("factor_2_accuracy")
    ]["original_value"]
    broad_delta_mae = original_compare[
        original_compare["default_label"].eq("material_family_default")
        & original_compare["metric_weighting"].eq("row_equal")
        & original_compare["metric_name"].eq("mae_log10")
    ]["delta_broad_minus_original"]
    broad_delta_f2 = original_compare[
        original_compare["default_label"].eq("material_family_default")
        & original_compare["metric_weighting"].eq("row_equal")
        & original_compare["metric_name"].eq("factor_2_accuracy")
    ]["delta_broad_minus_original"]
    diff_frac = diff_summary[diff_summary["comparison_label"].eq("sample_holdout_material_family_vs_global")]["different_prediction_fraction"]
    top1_sample = contribution_summary[contribution_summary["item"].eq("top1_sample_fraction_of_total_abs_error")]["value"]
    top1_paper = contribution_summary[contribution_summary["item"].eq("top1_paper_fraction_of_total_abs_error")]["value"]
    extreme_count = int((base_default["abs_error_decades"] >= 10).sum())
    max_abs = float(base_default["abs_error_decades"].max())
    coverage = float(no_filter.get("retained_row_fraction", np.nan))
    # Use Step5B coverage if available in no_filter rows; retained fraction is always 1 for no_filter.
    default_metric_coverage = read_value_from_original_compare_like(original_compare, "material_family_default", "coverage_fraction", "broad_family_value")
    if np.isfinite(default_metric_coverage):
        coverage = default_metric_coverage
    add("coverage_is_high", "pass" if coverage >= 0.95 else "fail", coverage, ">= 0.95", "Broad material_family default coverage.")
    add("material_family_differs_from_global", "pass" if not diff_frac.empty and diff_frac.iloc[0] > 0.1 else "fail", diff_frac.iloc[0] if not diff_frac.empty else np.nan, "> 0.1", "Checks whether broad grouping changes predictions.")
    add("mae_improved_vs_original", "pass" if not broad_delta_mae.empty and broad_delta_mae.iloc[0] < -0.05 else "fail", broad_delta_mae.iloc[0] if not broad_delta_mae.empty else np.nan, "< -0.05", "Broad minus original MAE.")
    add("factor2_improved_vs_original", "pass" if not broad_delta_f2.empty and broad_delta_f2.iloc[0] > 0.02 else "fail", broad_delta_f2.iloc[0] if not broad_delta_f2.empty else np.nan, "> 0.02", "Broad minus original factor2.")
    orig_mae_value = orig_mae.iloc[0] if not orig_mae.empty else np.nan
    add("robust_mae_remains_improved_after_excluding_extreme_outliers", "pass" if np.isfinite(orig_mae_value) and ex5["mae_log10"] < orig_mae_value else "fail", ex5["mae_log10"], "exclude_abs_error_ge_5 broad MAE < original no_filter MAE", "Robustness after removing severe outliers.")
    add("not_dominated_by_single_sample_abs_error", "pass" if not top1_sample.empty and top1_sample.iloc[0] < 0.20 else "caution", top1_sample.iloc[0] if not top1_sample.empty else np.nan, "< 0.20", "Top sample share of absolute error.")
    add("not_dominated_by_single_paper_abs_error", "pass" if not top1_paper.empty and top1_paper.iloc[0] < 0.30 else "caution", top1_paper.iloc[0] if not top1_paper.empty else np.nan, "< 0.30", "Top paper share of absolute error.")
    add("extreme_outliers_exist", "caution" if extreme_count > 0 else "pass", extreme_count, "caution if > 0", "Extreme outliers are expected to be audited, not treated as fatal.")
    add("manual_review_needed", "caution" if extreme_count > 0 or max_abs > 5 else "pass", max_abs, "caution if extreme count > 0 or max_abs > 5", "Manual review of shortlist is required before final reporting.")
    action = "Use broad_family as a main candidate only with explicit outlier audit; review shortlist and consider reporting robust metrics alongside no-filter metrics."
    add("recommended_next_action", "caution", action, "manual decision", "Next decision point.")
    return pd.DataFrame(rows)


def read_value_from_original_compare_like(compare: pd.DataFrame, label: str, metric: str, col: str) -> float:
    row = compare[
        compare["default_label"].eq(label)
        & compare["metric_weighting"].eq("row_equal")
        & compare["metric_name"].eq(metric)
    ]
    if row.empty or col not in row.columns:
        return np.nan
    return float(row[col].iloc[0])


def df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def build_report(report: Path, warnings: list[str], input_rows: dict[str, Any], outputs: dict[str, Path], robust_filter: pd.DataFrame, outliers: pd.DataFrame, sample_summary: pd.DataFrame, paper_summary: pd.DataFrame, contribution_summary: pd.DataFrame, readiness: pd.DataFrame, checks: dict[str, bool], elapsed: float) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    default_metrics = robust_filter[
        robust_filter["default_label"].eq("broad_material_family_default")
        & robust_filter["filter_label"].isin(["no_filter", "exclude_abs_error_ge_5", "exclude_top_1p0_percent_abs_error"])
    ]
    lines = [
        "# Step6D Outlier Robustness Audit Report",
        "",
        "## Inputs",
        "",
        *[f"- {k}: {v}" for k, v in input_rows.items()],
        "",
        "## Outputs",
        "",
        *[f"- {k}: {v}" for k, v in outputs.items()],
        "",
        "## Broad Material Family Default Metrics",
        "",
        df_to_markdown(default_metrics, 20),
        "",
        "## Largest Outliers",
        "",
        df_to_markdown(outliers[["row_id", "paper_id", "sample_id", "formula_raw", "material_group_key", "T_K", "sigma_S_per_m", "sigma_pred_S_per_m", "abs_error_decades", "error_direction", "likely_error_origin_hint"]], 10),
        "",
        "## Sample Concentration",
        "",
        df_to_markdown(sample_summary.head(10), 10),
        "",
        "## Paper Concentration",
        "",
        df_to_markdown(paper_summary.head(10), 10),
        "",
        "## Error Contribution Summary",
        "",
        df_to_markdown(contribution_summary, 30),
        "",
        "## Readiness Summary",
        "",
        df_to_markdown(readiness, 20),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Sanity Checks",
            "",
            *[f"- {name}: {ok}" for name, ok in checks.items()],
            "",
            "## Notes",
            "",
            "- This Step6D audits existing Step6B/Step6C outputs only.",
            "- No new sigma predictions are computed.",
            "- Starrydata2 raw data and Step4 full-data reference curves are not read.",
            "- Extreme-outlier exclusion is a sensitivity analysis, not a final data deletion decision.",
            "",
            "## Next Actions",
            "",
            "- Manually inspect the shortlist paper/sample rows.",
            "- Decide whether to report broad_family no-filter and robust metrics together.",
            "- Compare formula_system_collapsed if a second repaired grouping is needed.",
            "- Select final tables and figures for reporting.",
            "",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity(pred: pd.DataFrame, base_default: pd.DataFrame, outputs: dict[str, Path], report: Path) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "prediction_valid_all_ok": pred["prediction_status"].eq("ok").all(),
        "sigma_positive_finite": np.isfinite(pred["sigma_S_per_m"]).all() and (pred["sigma_S_per_m"] > 0).all(),
        "sigma_pred_positive_finite": np.isfinite(pred["sigma_pred_S_per_m"]).all() and (pred["sigma_pred_S_per_m"] > 0).all(),
        "sigma_pred_over_exp_consistent": np.allclose(pred["sigma_pred_over_exp"], pred["sigma_pred_S_per_m"] / pred["sigma_S_per_m"], rtol=1e-10, atol=1e-12),
        "log_error_consistent": np.allclose(pred["log10_sigma_pred_over_exp"], np.log10(pred["sigma_pred_over_exp"]), rtol=1e-10, atol=1e-12),
        "sigma0_ratio_equals_prediction_error": np.allclose(pred["log10_sigma0_ref_over_row_sigma0"], pred["log10_sigma_pred_over_exp"], rtol=1e-10, atol=1e-10),
        "default_4_configs_exist": set(DEFAULT_CONFIGS.values()).issubset(set(pred["config_id"])),
        "broad_material_family_default_nonempty": len(base_default) > 0,
        "report_created": report.exists() and report.stat().st_size > 0,
        "did_not_read_step4_full_data_reference_curve": True,
        "did_not_read_raw_data": True,
        "did_not_compute_new_sigma_pred": True,
    }
    for label, path in outputs.items():
        checks[f"{label}_created"] = path.exists() and path.stat().st_size > 0
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def main() -> None:
    started = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    log("loading Step6B broad_family predictions...")
    broad_pred = read_preferred(args.step6b_dir / "step5b_test_predictions_valid")
    broad_metrics = read_table(args.step6b_dir / "step5c_metrics_by_config.csv")
    default_comparison = read_table(args.step6b_dir / "step5c_default_comparison.csv")
    broad_default_summary = read_table(args.step6b_dir / "step6b_broad_family_default_metrics_summary.csv")
    broad_vs_original = read_table(args.step6b_dir / "step6b_broad_family_vs_original_default_metrics_comparison.csv")
    diff_summary = read_table(args.step6b_dir / "step6b_material_family_vs_global_prediction_diff_summary.csv")

    log("loading Step6C diagnostics...")
    _ = read_table(args.step6c_dir / "step6c_visual_diagnostics_summary.csv")
    _ = read_table(args.step6c_dir / "step6c_broad_largest_error_diagnostics_top100.csv")

    log("loading original comparison data...")
    original_pred = load_optional_table(
        [args.original_dir / "step5b_test_predictions_valid.parquet", args.original_dir / "step5b_test_predictions_valid.csv"],
        warnings,
        "original prediction valid rows",
    )
    original_default_comparison = load_optional_table([args.original_dir / "step5c_default_comparison.csv"], warnings, "original default comparison")

    log("loading optional metadata...")
    metadata = load_optional_table([args.metadata_input, args.metadata_input.with_suffix(".csv")], warnings, "metadata input")
    step3 = load_optional_table([args.step3_input, args.step3_input.with_suffix(".csv")], warnings, "step3 input")
    step0 = load_optional_table([args.step0_input, args.step0_input.with_suffix(".csv")], warnings, "step0 input")

    log("validating required columns...")
    require_columns(broad_pred, REQUIRED_PRED_COLS, "Step6B broad predictions")

    log("computing diagnostic error columns...")
    pred = prepare_predictions(broad_pred, args.max_rows_per_config)
    pred = merge_metadata(pred, [metadata, step3, step0])
    base_default = pred[pred["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])].copy()

    log("building outlier topN table...")
    outliers = build_outlier_topn(base_default, args.top_n_outliers)
    outlier_path = args.output / out_name("step6d_outlier_rows_topN", args.output_suffix)
    outliers.to_csv(outlier_path, index=False, encoding="utf-8-sig")

    log("summarizing outliers by row_id...")
    row_summary = build_row_summary(pred)
    row_summary_path = args.output / out_name("step6d_outlier_summary_by_row_id", args.output_suffix)
    row_summary.to_csv(row_summary_path, index=False, encoding="utf-8-sig")

    log("summarizing outliers by sample...")
    sample_summary = build_sample_summary(base_default)
    sample_summary_path = args.output / out_name("step6d_outlier_summary_by_sample", args.output_suffix)
    sample_summary.to_csv(sample_summary_path, index=False, encoding="utf-8-sig")

    log("summarizing outliers by paper...")
    paper_summary = build_paper_summary(base_default)
    paper_summary_path = args.output / out_name("step6d_outlier_summary_by_paper", args.output_suffix)
    paper_summary.to_csv(paper_summary_path, index=False, encoding="utf-8-sig")

    log("building top outlier sample context rows...")
    context_rows = build_context_rows(base_default, outliers)
    context_path = args.output / out_name("step6d_top_outlier_sample_context_rows", args.output_suffix)
    context_rows.to_csv(context_path, index=False, encoding="utf-8-sig")

    log("computing robust metrics by filter...")
    robust_filter = robust_metrics_by_filter(pred)
    robust_filter_path = args.output / out_name("step6d_robust_metrics_by_filter", args.output_suffix)
    robust_filter.to_csv(robust_filter_path, index=False, encoding="utf-8-sig")

    log("computing robust metrics by config...")
    robust_config = robust_metrics_by_config(pred)
    robust_config_path = args.output / out_name("step6d_robust_metrics_by_config", args.output_suffix)
    robust_config.to_csv(robust_config_path, index=False, encoding="utf-8-sig")

    log("comparing original vs broad robust metrics...")
    robust_compare = original_vs_broad_robust(original_pred, robust_filter, original_default_comparison)
    robust_compare_path = args.output / out_name("step6d_original_vs_broad_robust_metrics_comparison", args.output_suffix)
    robust_compare.to_csv(robust_compare_path, index=False, encoding="utf-8-sig")

    log("computing error contribution concentration...")
    contribution, contribution_summary = contribution_tables(base_default)
    contribution_path = args.output / out_name("step6d_error_contribution_concentration", args.output_suffix)
    contribution_summary_path = args.output / out_name("step6d_error_contribution_summary", args.output_suffix)
    contribution.to_csv(contribution_path, index=False, encoding="utf-8-sig")
    contribution_summary.to_csv(contribution_summary_path, index=False, encoding="utf-8-sig")

    log("building manual review shortlist...")
    shortlist = manual_review_shortlist(outliers, row_summary, sample_summary, paper_summary)
    shortlist_path = args.output / out_name("step6d_manual_review_shortlist", args.output_suffix)
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8-sig")

    log("building readiness summary...")
    readiness = readiness_summary(robust_filter, robust_compare, contribution_summary, diff_summary, broad_vs_original, base_default)
    readiness_path = args.output / out_name("step6d_broad_family_main_result_readiness_summary", args.output_suffix)
    readiness.to_csv(readiness_path, index=False, encoding="utf-8-sig")

    outputs = {
        "outlier_rows_topN": outlier_path,
        "outlier_summary_by_row_id": row_summary_path,
        "outlier_summary_by_sample": sample_summary_path,
        "outlier_summary_by_paper": paper_summary_path,
        "top_outlier_sample_context_rows": context_path,
        "robust_metrics_by_filter": robust_filter_path,
        "robust_metrics_by_config": robust_config_path,
        "original_vs_broad_robust_metrics_comparison": robust_compare_path,
        "error_contribution_concentration": contribution_path,
        "error_contribution_summary": contribution_summary_path,
        "manual_review_shortlist": shortlist_path,
        "readiness_summary": readiness_path,
    }
    input_rows = {
        "broad_prediction_rows": len(broad_pred),
        "audit_prediction_rows_after_optional_limit": len(pred),
        "broad_material_family_default_rows": len(base_default),
        "broad_metrics_rows": len(broad_metrics),
        "broad_default_summary_rows": len(broad_default_summary),
        "default_comparison_rows": len(default_comparison),
        "original_prediction_rows": len(original_pred) if original_pred is not None else "not_available",
    }

    log("writing report...")
    build_report(args.report, warnings, input_rows, outputs, robust_filter, outliers, sample_summary, paper_summary, contribution_summary, readiness, {}, time.time() - started)

    log("running sanity checks...")
    checks, failures = run_sanity(pred, base_default, outputs, args.report)
    if failures:
        build_report(args.report, warnings, input_rows, outputs, robust_filter, outliers, sample_summary, paper_summary, contribution_summary, readiness, checks, time.time() - started)
        for failure in failures:
            print(f"[step6d] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    build_report(args.report, warnings, input_rows, outputs, robust_filter, outliers, sample_summary, paper_summary, contribution_summary, readiness, checks, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
