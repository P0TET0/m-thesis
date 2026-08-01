import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step5b_test_predictions_valid.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step5b_test_predictions_valid.csv"
DEFAULT_COVERAGE = PROCESSED_DIR / "step5b_prediction_coverage_by_config.csv"
DEFAULT_UNAVAILABLE = PROCESSED_DIR / "step5b_test_predictions_unavailable.csv"

CONFIG_KEYS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
]

REQUIRED_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "prediction_status",
    "row_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
    "T_K",
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

CORE_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "prediction_status",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "T_K",
    "T_bin_center_K",
    "carrier_type",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "squared_log10_sigma_pred_over_exp",
]

NUMERIC_COLUMNS = [
    "T_K",
    "T_bin_center_K",
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
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "squared_log10_sigma_pred_over_exp",
]

DROP_COLUMNS = [
    "config_id",
    "row_id",
    "reject_reason",
    "prediction_status",
    "T_K",
    "carrier_type",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "F0_eta",
    "log10_sigma_pred_over_exp",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
]

METRIC_COLUMNS = [
    "n_rows",
    "n_samples",
    "n_papers",
    "n_material_families",
    "n_T_bins",
    "mean_log10_error",
    "median_log10_error",
    "mae_log10",
    "rmse_log10",
    "std_log10_error",
    "q05_log10_error",
    "q25_log10_error",
    "q75_log10_error",
    "q95_log10_error",
    "max_abs_log10_error",
    "overprediction_fraction",
    "underprediction_fraction",
    "near_exact_fraction",
    "factor_2_accuracy",
    "factor_3_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "median_abs_factor_error",
    "mean_abs_factor_error_equiv",
    "sigma_exp_median_S_per_m",
    "sigma_pred_median_S_per_m",
    "eta_median",
    "S_abs_median_uV_per_K",
    "T_min_K",
    "T_max_K",
    "train_sample_count_median",
    "train_paper_count_median",
]

LARGEST_ERROR_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
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
    "carrier_type",
    "T_K",
    "T_bin_center_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_log10_sigma_pred_over_exp",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step5C prediction evaluation metrics.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--unavailable", type=Path, default=DEFAULT_UNAVAILABLE)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--min-eval-rows", type=int, default=30)
    parser.add_argument("--min-eval-samples", type=int, default=5)
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step5c] {message}", flush=True)


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
        raise FileNotFoundError(f"Step5B valid predictions not found: {explicit}")
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step5B valid predictions not found in experiments/exp006/data/processed")


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
    if row.get("prediction_status") != "ok":
        return "prediction_status_not_ok"
    for column, reason in [
        ("sigma_S_per_m", "invalid_sigma_exp"),
        ("sigma_pred_S_per_m", "invalid_sigma_pred"),
        ("F0_eta", "invalid_F0_eta"),
    ]:
        value = row.get(column, np.nan)
        if not np.isfinite(value) or value <= 0:
            return reason
    for column, reason in [
        ("log10_sigma_pred_over_exp", "invalid_log10_error"),
        ("abs_log10_sigma_pred_over_exp", "invalid_abs_log10_error"),
        ("squared_log10_sigma_pred_over_exp", "invalid_squared_log10_error"),
    ]:
        if not np.isfinite(row.get(column, np.nan)):
            return reason
    if str(row.get("carrier_type", "")) not in {"p", "n"}:
        return "invalid_carrier_type"
    return ""


def filter_evaluable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    for column in NUMERIC_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    if "material_family_raw" in work.columns:
        work["material_family_raw"] = work["material_family_raw"].where(
            work["material_family_raw"].map(clean_text).ne(""),
            work["material_group_key"],
        )
    work["reject_reason"] = work.apply(reject_reason, axis=1)
    usable = work[work["reject_reason"].eq("")].copy()
    dropped = work[~work["reject_reason"].eq("")].copy()
    dropped_out = pd.DataFrame(columns=DROP_COLUMNS)
    if not dropped.empty:
        for column in DROP_COLUMNS:
            dropped_out[column] = dropped[column] if column in dropped.columns else ""
    return usable.drop(columns=["reject_reason"]), dropped_out


def assign_eta_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, np.inf]
    labels = ["[1, 2)", "[2, 5)", "[5, 10)", "[10, 20)", "[20, 50)", "[50, inf)"]
    out["eta_bin_label"] = pd.cut(out["eta"], bins=bins, labels=labels, right=False, include_lowest=True)
    left_map = dict(zip(labels, bins[:-1]))
    right_map = dict(zip(labels, bins[1:]))
    out["eta_bin_label"] = out["eta_bin_label"].astype(str)
    out["eta_bin_left"] = out["eta_bin_label"].map(left_map)
    out["eta_bin_right"] = out["eta_bin_label"].map(right_map)
    return out


def eval_group_reliability(n_rows: int, n_samples: int, n_papers: int, min_rows: int, min_samples: int) -> tuple[bool, str]:
    reliable = n_rows >= min_rows and n_samples >= min_samples
    if not reliable:
        return False, "insufficient"
    if n_samples >= 30 and n_papers >= 5:
        return True, "high"
    if n_samples >= 10 and n_papers >= 2:
        return True, "medium"
    return True, "low"


def metric_values(group: pd.DataFrame, min_rows: int, min_samples: int) -> dict[str, Any]:
    e = pd.to_numeric(group["log10_sigma_pred_over_exp"], errors="coerce")
    abs_e = pd.to_numeric(group["abs_log10_sigma_pred_over_exp"], errors="coerce")
    sq_e = pd.to_numeric(group["squared_log10_sigma_pred_over_exp"], errors="coerce")
    n_rows = int(len(group))
    n_samples = int(group["validation_sample_group_id"].nunique(dropna=True))
    n_papers = int(group["validation_paper_group_id"].nunique(dropna=True))
    is_rel, rel_label = eval_group_reliability(n_rows, n_samples, n_papers, min_rows, min_samples)
    return {
        "n_rows": n_rows,
        "n_samples": n_samples,
        "n_papers": n_papers,
        "n_material_families": int(group["material_group_key"].nunique(dropna=True)),
        "n_T_bins": int(group["T_bin_center_K"].nunique(dropna=True)),
        "mean_log10_error": float(e.mean()),
        "median_log10_error": float(e.median()),
        "mae_log10": float(abs_e.mean()),
        "rmse_log10": float(math.sqrt(sq_e.mean())),
        "std_log10_error": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
        "q05_log10_error": float(e.quantile(0.05)),
        "q25_log10_error": float(e.quantile(0.25)),
        "q75_log10_error": float(e.quantile(0.75)),
        "q95_log10_error": float(e.quantile(0.95)),
        "max_abs_log10_error": float(abs_e.max()),
        "overprediction_fraction": float((e > 0).mean()),
        "underprediction_fraction": float((e < 0).mean()),
        "near_exact_fraction": float((abs_e <= 0.05).mean()),
        "factor_2_accuracy": float((abs_e <= math.log10(2)).mean()),
        "factor_3_accuracy": float((abs_e <= math.log10(3)).mean()),
        "factor_5_accuracy": float((abs_e <= math.log10(5)).mean()),
        "factor_10_accuracy": float((abs_e <= 1.0).mean()),
        "median_abs_factor_error": float(10.0 ** abs_e.median()),
        "mean_abs_factor_error_equiv": float(10.0 ** abs_e.mean()),
        "sigma_exp_median_S_per_m": float(pd.to_numeric(group["sigma_S_per_m"], errors="coerce").median()),
        "sigma_pred_median_S_per_m": float(pd.to_numeric(group["sigma_pred_S_per_m"], errors="coerce").median()),
        "eta_median": float(pd.to_numeric(group["eta"], errors="coerce").median()),
        "S_abs_median_uV_per_K": float(pd.to_numeric(group["S_abs_uV_per_K"], errors="coerce").median()),
        "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
        "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        "train_sample_count_median": float(pd.to_numeric(group["train_sample_count"], errors="coerce").median()),
        "train_paper_count_median": float(pd.to_numeric(group["train_paper_count"], errors="coerce").median()),
        "is_reliable_eval_group": is_rel,
        "eval_group_reliability": rel_label,
    }


def sample_equal_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    sample_keys = keys + ["validation_sample_group_id"]
    rows: list[dict[str, Any]] = []
    for values, group in df.groupby(sample_keys, dropna=False, sort=False):
        row = dict(zip(sample_keys, values if isinstance(values, tuple) else (values,)))
        row.update(
            {
                "row_id": group["row_id"].iloc[0],
                "validation_paper_group_id": group["validation_paper_group_id"].iloc[0],
                "material_group_key": group["material_group_key"].iloc[0],
                "T_bin_center_K": group["T_bin_center_K"].median(),
                "log10_sigma_pred_over_exp": group["log10_sigma_pred_over_exp"].median(),
                "sigma_S_per_m": group["sigma_S_per_m"].median(),
                "sigma_pred_S_per_m": group["sigma_pred_S_per_m"].median(),
                "eta": group["eta"].median(),
                "S_abs_uV_per_K": group["S_abs_uV_per_K"].median(),
                "T_K": group["T_K"].median(),
                "train_sample_count": group["train_sample_count"].median(),
                "train_paper_count": group["train_paper_count"].median(),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_log10_sigma_pred_over_exp"] = out["log10_sigma_pred_over_exp"].abs()
    out["squared_log10_sigma_pred_over_exp"] = out["log10_sigma_pred_over_exp"] ** 2
    return out


def compute_metrics(df: pd.DataFrame, group_keys: list[str], min_rows: int, min_samples: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for weighting in ["row_equal", "sample_equal"]:
        source = df if weighting == "row_equal" else sample_equal_rows(df, group_keys)
        rows: list[dict[str, Any]] = []
        if source.empty:
            continue
        for values, group in source.groupby(group_keys, dropna=False, sort=False):
            key_values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(group_keys, key_values))
            row["metric_weighting"] = weighting
            row.update(metric_values(group, min_rows, min_samples))
            rows.append(row)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=[*group_keys, "metric_weighting", *METRIC_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def load_coverage(path: Path | None, warnings: list[str]) -> pd.DataFrame:
    if path is None or not path.exists():
        warnings.append("coverage file is missing; coverage merge and coverage-conditioned ranking are degraded")
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def merge_coverage(metrics: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or coverage.empty:
        return metrics
    coverage_cols = [
        *CONFIG_KEYS,
        "test_rows",
        "prediction_ok_rows",
        "prediction_unavailable_rows",
        "coverage_fraction",
        "reference_bins_total",
        "reference_bins_reliable",
    ]
    present = [col for col in coverage_cols if col in coverage.columns]
    return metrics.merge(coverage[present].drop_duplicates(CONFIG_KEYS), on=CONFIG_KEYS, how="left")


def default_configs() -> list[str]:
    return [
        "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
        "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
        "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
        "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    ]


def build_default_comparison(metrics_by_config: pd.DataFrame) -> pd.DataFrame:
    return metrics_by_config[metrics_by_config["config_id"].isin(default_configs())].copy()


def build_ranking(metrics_by_config: pd.DataFrame, has_coverage: bool, warnings: list[str]) -> pd.DataFrame:
    ranked = metrics_by_config[
        metrics_by_config["metric_weighting"].eq("row_equal") & metrics_by_config["is_reliable_eval_group"].eq(True)
    ].copy()
    if has_coverage and "coverage_fraction" in ranked.columns:
        ranked = ranked[ranked["coverage_fraction"] >= 0.95].copy()
    else:
        warnings.append("coverage_fraction is unavailable; ranking ignores coverage condition")
    if ranked.empty:
        return ranked
    ranked["rank_by_mae_log10"] = ranked["mae_log10"].rank(method="min", ascending=True).astype(int)
    ranked["rank_by_rmse_log10"] = ranked["rmse_log10"].rank(method="min", ascending=True).astype(int)
    ranked["rank_by_factor_2_accuracy"] = ranked["factor_2_accuracy"].rank(method="min", ascending=False).astype(int)
    ranked["rank_by_factor_10_accuracy"] = ranked["factor_10_accuracy"].rank(method="min", ascending=False).astype(int)
    return ranked.sort_values(["rank_by_mae_log10", "rank_by_factor_2_accuracy"]).reset_index(drop=True)


def largest_error_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("abs_log10_sigma_pred_over_exp", ascending=False).head(1000).copy()
    for column in LARGEST_ERROR_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    return out[LARGEST_ERROR_COLUMNS]


def sanity_checks(
    rows: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    coverage: pd.DataFrame,
    min_rows: int,
    min_samples: int,
    full_run: bool,
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["prediction_status_ok"] = rows["prediction_status"].eq("ok").all()
    checks["sigma_exp_positive"] = bool(np.isfinite(rows["sigma_S_per_m"]).all() and (rows["sigma_S_per_m"] > 0).all())
    checks["sigma_pred_positive"] = bool(np.isfinite(rows["sigma_pred_S_per_m"]).all() and (rows["sigma_pred_S_per_m"] > 0).all())
    checks["F0_positive"] = bool(np.isfinite(rows["F0_eta"]).all() and (rows["F0_eta"] > 0).all())
    checks["log10_error_finite"] = bool(np.isfinite(rows["log10_sigma_pred_over_exp"]).all())
    checks["abs_error_consistent"] = bool(
        np.allclose(rows["abs_log10_sigma_pred_over_exp"], rows["log10_sigma_pred_over_exp"].abs(), rtol=1e-10)
    )
    checks["squared_error_consistent"] = bool(
        np.allclose(rows["squared_log10_sigma_pred_over_exp"], rows["log10_sigma_pred_over_exp"] ** 2, rtol=1e-10)
    )
    checks["ratio_consistent"] = bool(
        np.allclose(rows["sigma_pred_over_exp"], rows["sigma_pred_S_per_m"] / rows["sigma_S_per_m"], rtol=1e-10)
    )
    checks["log10_ratio_consistent"] = bool(
        np.allclose(rows["log10_sigma_pred_over_exp"], np.log10(rows["sigma_pred_over_exp"]), rtol=1e-10)
    )
    all_metrics = [frame for frame in outputs.values() if not frame.empty and "mae_log10" in frame.columns]
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    if not metrics.empty:
        factor_cols = ["factor_2_accuracy", "factor_3_accuracy", "factor_5_accuracy", "factor_10_accuracy"]
        checks["factor_accuracy_range"] = metrics[factor_cols].apply(lambda s: s.between(0, 1).all()).all()
        checks["mae_nonnegative"] = (metrics["mae_log10"] >= 0).all()
        checks["rmse_nonnegative"] = (metrics["rmse_log10"] >= 0).all()
        checks["max_abs_ge_mae"] = (metrics["max_abs_log10_error"] + 1e-12 >= metrics["mae_log10"]).all()
        checks["n_rows_positive"] = (metrics["n_rows"] > 0).all()
        expected_reliable = (metrics["n_rows"] >= min_rows) & (metrics["n_samples"] >= min_samples)
        checks["is_reliable_eval_group_rule"] = (metrics["is_reliable_eval_group"] == expected_reliable).all()
        expected_level = []
        for _, row in metrics.iterrows():
            _, level = eval_group_reliability(int(row["n_rows"]), int(row["n_samples"]), int(row["n_papers"]), min_rows, min_samples)
            expected_level.append(level)
        checks["eval_group_reliability_rule"] = (metrics["eval_group_reliability"].to_numpy() == np.array(expected_level)).all()
    metrics_by_config = outputs["metrics_by_config"]
    config_count = metrics_by_config["config_id"].nunique() if not metrics_by_config.empty else 0
    if full_run:
        checks["metrics_by_config_has_32_configs"] = config_count == 32
    elif config_count != 32:
        warnings.append(f"small test metrics_by_config has {config_count} configs, not 32")
    default = outputs["default_comparison"]
    expected_default_rows = {(config_id, weighting) for config_id in default_configs() for weighting in ["row_equal", "sample_equal"]}
    found_default_rows = set(zip(default.get("config_id", []), default.get("metric_weighting", [])))
    checks["default_comparison_complete"] = expected_default_rows.issubset(found_default_rows)
    checks["ranking_nonempty"] = not outputs["config_ranking"].empty
    checks["largest_error_rows_limit"] = len(outputs["largest_abs_error_rows"]) <= 1000
    if not coverage.empty and "coverage_fraction" in coverage.columns:
        checks["coverage_fraction_range"] = pd.to_numeric(coverage["coverage_fraction"], errors="coerce").dropna().between(0, 1).all()
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures, warnings


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(output_dir: Path, suffix: str, outputs: dict[str, pd.DataFrame], dropped: pd.DataFrame) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses: dict[str, str] = {}
    for base, frame in outputs.items():
        frame.to_csv(output_dir / output_name(f"step5c_{base}", suffix, "csv"), index=False, encoding="utf-8-sig")
    dropped.to_csv(output_dir / output_name("step5c_dropped_rows", suffix, "csv"), index=False, encoding="utf-8-sig")
    ok, error = save_parquet(outputs["metrics_by_config"], output_dir / output_name("step5c_metrics_by_config", suffix, "parquet"))
    statuses[output_name("step5c_metrics_by_config", suffix, "parquet")] = "saved" if ok else f"not saved: {error}"
    return statuses


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def default_metric_rows(default: pd.DataFrame, config_id: str) -> pd.DataFrame:
    cols = [
        "config_id",
        "metric_weighting",
        "mae_log10",
        "rmse_log10",
        "median_log10_error",
        "factor_2_accuracy",
        "factor_5_accuracy",
        "factor_10_accuracy",
        "coverage_fraction",
    ]
    sub = default[default["config_id"].eq(config_id)].copy()
    for col in cols:
        if col not in sub.columns:
            sub[col] = np.nan
    return sub[cols]


def write_report(
    report_path: Path,
    input_path: Path,
    input_rows: int,
    eval_rows: pd.DataFrame,
    dropped: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    checks: dict[str, bool],
    warnings: list[str],
    parquet_statuses: dict[str, str],
    args: argparse.Namespace,
    elapsed_sec: float,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    default = outputs["default_comparison"]
    ranking = outputs["config_ranking"]
    largest = outputs["largest_abs_error_rows"]
    lines = [
        "# Step5C Evaluation Metrics Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- evaluated_rows: {len(eval_rows)}",
        f"- dropped_rows: {len(dropped)}",
        f"- config_count: {eval_rows['config_id'].nunique()}",
        f"- metric_weighting: {sorted(outputs['metrics_by_config']['metric_weighting'].unique().tolist())}",
        f"- min_eval_rows: {args.min_eval_rows}",
        f"- min_eval_samples: {args.min_eval_samples}",
        f"- metrics_by_config rows: {len(outputs['metrics_by_config'])}",
        f"- metrics_by_carrier_type rows: {len(outputs['metrics_by_carrier_type'])}",
        f"- metrics_by_material_family rows: {len(outputs['metrics_by_material_family'])}",
        f"- metrics_by_temperature_bin rows: {len(outputs['metrics_by_temperature_bin'])}",
        f"- metrics_by_eta_bin rows: {len(outputs['metrics_by_eta_bin'])}",
        f"- metrics_by_reliability_level rows: {len(outputs['metrics_by_reliability_level'])}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for name, status in parquet_statuses.items():
        lines.append(f"- {name}: {status}")
    if not dropped.empty:
        lines.extend(["", "## Dropped Reasons", "", str(dropped["reject_reason"].value_counts().to_dict())])
    lines.extend(["", "## Default Comparison", "", df_to_markdown(default)])
    for title, config_id in [
        ("Material Family Default", default_configs()[0]),
        ("Global Default", default_configs()[1]),
        ("Paper Material Family Default", default_configs()[2]),
        ("Paper Global Default", default_configs()[3]),
    ]:
        lines.extend(["", f"## {title}", "", df_to_markdown(default_metric_rows(default, config_id))])
    best_mae = ranking.sort_values("rank_by_mae_log10").head(10)
    best_f2 = ranking.sort_values("rank_by_factor_2_accuracy").head(10)
    cols = ["config_id", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "coverage_fraction"]
    lines.extend(["", "## Best Configs By MAE", "", df_to_markdown(best_mae[[c for c in cols if c in best_mae.columns]])])
    lines.extend(["", "## Best Configs By Factor 2 Accuracy", "", df_to_markdown(best_f2[[c for c in cols if c in best_f2.columns]])])
    comparison_cols = ["split_scheme", "group_scheme", "curve_method", "reference_source_subset", "carrier_type", "mae_log10", "factor_2_accuracy"]
    carrier = outputs["metrics_by_carrier_type"]
    eta = outputs["metrics_by_eta_bin"]
    temp = outputs["metrics_by_temperature_bin"]
    rel = outputs["metrics_by_reliability_level"]
    lines.extend(
        [
            "",
            "## Comparison Notes",
            "",
            f"- split_scheme median mae_log10: {outputs['metrics_by_config'].groupby('split_scheme')['mae_log10'].median().to_dict()}",
            f"- group_scheme median mae_log10: {outputs['metrics_by_config'].groupby('group_scheme')['mae_log10'].median().to_dict()}",
            f"- curve_method median mae_log10: {outputs['metrics_by_config'].groupby('curve_method')['mae_log10'].median().to_dict()}",
            f"- reference_source_subset median mae_log10: {outputs['metrics_by_config'].groupby('reference_source_subset')['mae_log10'].median().to_dict()}",
            f"- p/n median mae_log10: {carrier.groupby('carrier_type')['mae_log10'].median().to_dict() if not carrier.empty else {}}",
            f"- eta bin median mae_log10: {eta.groupby('eta_bin_label')['mae_log10'].median().to_dict() if not eta.empty else {}}",
            f"- temperature bin median mae_log10: {temp.groupby('T_bin_label')['mae_log10'].median().to_dict() if not temp.empty else {}}",
            f"- reliability_level median mae_log10: {rel.groupby('reliability_level')['mae_log10'].median().to_dict() if not rel.empty else {}}",
            f"- largest abs_log10 error: {largest['abs_log10_sigma_pred_over_exp'].max() if not largest.empty else 'n/a'}",
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
    lines.append("- Main metric is log10(sigma_pred / sigma_exp).")
    lines.append("- Sigma spans orders of magnitude, so log error is used instead of ordinary absolute error.")
    lines.append("- Step5B train-only sigma0_ref is used; Step4 full-data curves are not used for independent validation.")
    lines.append("- Test-side sigma0_S_per_m is not used to create predictions.")
    lines.append("- Step5D should create predicted-vs-experimental, error distribution, config comparison, eta, temperature, and material-family plots.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    report_path = args.report or (REPORT_DIR / output_name("step5c_evaluation_metrics_report", args.output_suffix, "md"))
    full_run = args.max_rows_per_config is None
    warnings: list[str] = []

    log("loading valid test predictions...")
    df = read_table(input_path)
    if args.max_rows_per_config is not None:
        if args.max_rows_per_config <= 0:
            raise ValueError("--max-rows-per-config must be positive")
        df = df.groupby("config_id", dropna=False, sort=False).head(args.max_rows_per_config).copy()
    input_rows = len(df)
    log(f"input rows: {input_rows}")
    log("loading coverage summary...")
    coverage = load_coverage(args.coverage, warnings)
    log("validating required columns...")
    validate_columns(df)
    log("filtering evaluable rows...")
    eval_rows, dropped = filter_evaluable_rows(df)
    log("assigning eta bins...")
    eval_rows = assign_eta_bins(eval_rows)

    log("computing metrics by config...")
    metrics_by_config = compute_metrics(eval_rows, CONFIG_KEYS, args.min_eval_rows, args.min_eval_samples)
    metrics_by_config = merge_coverage(metrics_by_config, coverage)
    log("computing metrics by carrier type...")
    metrics_by_carrier_type = compute_metrics(eval_rows, CONFIG_KEYS + ["carrier_type"], args.min_eval_rows, args.min_eval_samples)
    log("computing metrics by material family...")
    metrics_by_material_family = compute_metrics(
        eval_rows, CONFIG_KEYS + ["material_group_key", "material_family_raw"], args.min_eval_rows, args.min_eval_samples
    )
    log("computing metrics by temperature bin...")
    metrics_by_temperature_bin = compute_metrics(
        eval_rows, CONFIG_KEYS + ["T_bin_center_K", "T_bin_label"], args.min_eval_rows, args.min_eval_samples
    )
    log("computing metrics by eta bin...")
    metrics_by_eta_bin = compute_metrics(
        eval_rows, CONFIG_KEYS + ["eta_bin_label", "eta_bin_left", "eta_bin_right"], args.min_eval_rows, args.min_eval_samples
    )
    log("computing metrics by reliability level...")
    metrics_by_reliability_level = compute_metrics(
        eval_rows, CONFIG_KEYS + ["reliability_level"], args.min_eval_rows, args.min_eval_samples
    )
    metrics_by_sigma_source = compute_metrics(eval_rows, CONFIG_KEYS + ["sigma_source"], args.min_eval_rows, args.min_eval_samples)
    metrics_by_match_method = compute_metrics(eval_rows, CONFIG_KEYS + ["match_method"], args.min_eval_rows, args.min_eval_samples)
    log("building default comparison...")
    default_comparison = build_default_comparison(metrics_by_config)
    log("building config ranking...")
    config_ranking = build_ranking(metrics_by_config, not coverage.empty, warnings)
    log("collecting largest error rows...")
    largest_abs_error_rows = largest_error_rows(eval_rows)
    outputs = {
        "metrics_by_config": metrics_by_config,
        "metrics_by_carrier_type": metrics_by_carrier_type,
        "metrics_by_material_family": metrics_by_material_family,
        "metrics_by_temperature_bin": metrics_by_temperature_bin,
        "metrics_by_eta_bin": metrics_by_eta_bin,
        "metrics_by_reliability_level": metrics_by_reliability_level,
        "metrics_by_sigma_source": metrics_by_sigma_source,
        "metrics_by_match_method": metrics_by_match_method,
        "default_comparison": default_comparison,
        "config_ranking": config_ranking,
        "largest_abs_error_rows": largest_abs_error_rows,
    }
    log("running sanity checks...")
    checks, failures, sanity_warnings = sanity_checks(eval_rows, outputs, coverage, args.min_eval_rows, args.min_eval_samples, full_run)
    warnings.extend(sanity_warnings)
    if failures:
        for failure in failures:
            print(f"[step5c] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("writing outputs...")
    parquet_statuses = write_outputs(args.output, args.output_suffix, outputs, dropped)
    write_report(
        report_path,
        input_path,
        input_rows,
        eval_rows,
        dropped,
        outputs,
        checks,
        warnings,
        parquet_statuses,
        args,
        time.time() - started,
    )
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
