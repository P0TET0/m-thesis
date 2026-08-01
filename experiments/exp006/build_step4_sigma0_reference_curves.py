import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step3_sigma0_valid.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step3_sigma0_valid.csv"

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
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
]

DROP_COLUMNS = [
    "row_id",
    "reject_reason",
    "T_K",
    "carrier_type",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
    "paper_id",
    "sample_id",
    "sample_key",
    "material_family_raw",
]

CURVE_KEY_COLUMNS = [
    "source_subset",
    "group_scheme",
    "material_group_key",
    "carrier_type",
    "curve_method",
    "T_bin_center_K",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step4 sigma0(T) reference curve bins.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--bin-width-k", type=float, default=100.0)
    parser.add_argument("--bin-start-k", type=float, default=50.0)
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step4] {message}", flush=True)


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
        raise FileNotFoundError(f"Step3 sigma0 valid input not found: {explicit}")
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError("Step3 sigma0 valid input not found in experiments/exp006/data/processed")


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def clean_group_value(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, float) and pd.isna(value):
        return fallback
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "n/a"}:
        return fallback
    return text


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"input missing required columns: {missing}")


def reject_reason(row: pd.Series) -> str:
    if not bool(row.get("is_valid_sigma0_bool", False)):
        return "is_valid_sigma0_not_true"
    if not np.isfinite(row.get("T_K_num", np.nan)) or row.get("T_K_num", np.nan) <= 0:
        return "invalid_T_K"
    if not np.isfinite(row.get("sigma0_num", np.nan)) or row.get("sigma0_num", np.nan) <= 0:
        return "invalid_sigma0"
    if not np.isfinite(row.get("log10_sigma0_num", np.nan)):
        return "invalid_log10_sigma0"
    if str(row.get("carrier_type", "")) not in {"p", "n"}:
        return "invalid_carrier_type"
    return ""


def filter_usable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["is_valid_sigma0_bool"] = as_bool(work["is_valid_sigma0"])
    work["is_conservative_valid_sigma0_bool"] = as_bool(work["is_conservative_valid_sigma0"])
    work["T_K_num"] = pd.to_numeric(work["T_K"], errors="coerce")
    work["sigma0_num"] = pd.to_numeric(work["sigma0_S_per_m"], errors="coerce")
    work["log10_sigma0_num"] = pd.to_numeric(work["log10_sigma0_S_per_m"], errors="coerce")
    work["reject_reason"] = work.apply(reject_reason, axis=1)
    usable = work[work["reject_reason"].eq("")].copy()
    dropped = work[~work["reject_reason"].eq("")].copy()
    dropped_out = pd.DataFrame(columns=DROP_COLUMNS)
    if not dropped.empty:
        for column in DROP_COLUMNS:
            dropped_out[column] = dropped[column] if column in dropped.columns else ""
    return usable.drop(columns=["reject_reason"]), dropped_out


def assign_temperature_bins(df: pd.DataFrame, bin_width: float, bin_start: float) -> pd.DataFrame:
    out = df.copy()
    if bin_width <= 0:
        raise ValueError("--bin-width-k must be positive")
    t = pd.to_numeric(out["T_K"], errors="coerce")
    idx = np.floor((t - bin_start) / bin_width).astype(int)
    left = bin_start + idx * bin_width
    right = left + bin_width
    center = (left + right) / 2.0
    out["T_bin_index"] = idx
    out["T_bin_left_K"] = left
    out["T_bin_right_K"] = right
    out["T_bin_center_K"] = center
    out["T_bin_label"] = [
        f"{left_val:g}_{right_val:g}K" for left_val, right_val in zip(left, right)
    ]
    out["material_family_clean"] = out["material_family_raw"].map(
        lambda value: clean_group_value(value, "unknown_material_family")
    )
    return out


def add_group_columns(df: pd.DataFrame, group_scheme: str) -> pd.DataFrame:
    out = df.copy()
    out["group_scheme"] = group_scheme
    if group_scheme == "global":
        out["material_group_key"] = "ALL"
        out["material_group_label"] = "ALL_MATERIALS"
    elif group_scheme == "material_family":
        out["material_group_key"] = out["material_family_clean"]
        out["material_group_label"] = out["material_family_clean"]
    else:
        raise ValueError(f"unknown group_scheme: {group_scheme}")
    return out


def reliability_level(row: pd.Series) -> str:
    if not bool(row["is_reference_bin_candidate"]):
        return "insufficient"
    if int(row["sample_count"]) >= 10 and int(row["paper_count"]) >= 3:
        return "high"
    if int(row["sample_count"]) >= 5 and int(row["paper_count"]) >= 2:
        return "medium"
    return "low"


def aggregate_curve_values(group: pd.DataFrame, values: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    log_sigma = pd.to_numeric(group["log10_sigma_S_per_m"], errors="coerce")
    sigma = pd.to_numeric(group["sigma_S_per_m"], errors="coerce")
    log_ref = float(values.median())
    return {
        "log10_sigma0_ref_S_per_m": log_ref,
        "sigma0_ref_S_per_m": 10.0 ** log_ref,
        "sigma0_raw_median_S_per_m": float(pd.to_numeric(group["sigma0_S_per_m"], errors="coerce").median()),
        "log10_sigma0_q25": float(values.quantile(0.25)),
        "log10_sigma0_q75": float(values.quantile(0.75)),
        "log10_sigma0_iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "log10_sigma0_min": float(values.min()),
        "log10_sigma0_max": float(values.max()),
        "log10_sigma0_mean": float(values.mean()),
        "log10_sigma0_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "sigma0_min_S_per_m": float(pd.to_numeric(group["sigma0_S_per_m"], errors="coerce").min()),
        "sigma0_max_S_per_m": float(pd.to_numeric(group["sigma0_S_per_m"], errors="coerce").max()),
        "T_median_K": float(pd.to_numeric(group["T_K"], errors="coerce").median()),
        "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
        "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        "eta_median": float(pd.to_numeric(group["eta"], errors="coerce").median()),
        "eta_min": float(pd.to_numeric(group["eta"], errors="coerce").min()),
        "eta_max": float(pd.to_numeric(group["eta"], errors="coerce").max()),
        "S_abs_median_uV_per_K": float(pd.to_numeric(group["S_abs_uV_per_K"], errors="coerce").median()),
        "sigma_median_S_per_m": float(sigma.median()),
        "log10_sigma_median_S_per_m": float(log_sigma.median()),
        "row_count": int(len(group)),
        "sample_count": int(group["sample_group_id"].nunique(dropna=True)),
        "paper_count": int(group["paper_id"].nunique(dropna=True)),
        "formula_count": int(group["formula_raw"].nunique(dropna=True)),
    }


def make_curve_rows_for_method(df: pd.DataFrame, curve_method: str) -> list[dict[str, Any]]:
    key_cols = [
        "source_subset",
        "group_scheme",
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
        for keys, group in df.groupby(key_cols, dropna=False, sort=False):
            row = dict(zip(key_cols, keys))
            row["curve_method"] = curve_method
            row.update(aggregate_curve_values(group, group["log10_sigma0_S_per_m"]))
            rows.append(row)
        return rows
    if curve_method == "sample_median":
        sample_key_cols = key_cols + ["sample_group_id"]
        sample_values = (
            df.groupby(sample_key_cols, dropna=False, sort=False)
            .agg(sample_log10_sigma0=("log10_sigma0_S_per_m", "median"))
            .reset_index()
        )
        for keys, sample_group in sample_values.groupby(key_cols, dropna=False, sort=False):
            mask = np.ones(len(df), dtype=bool)
            for col, value in zip(key_cols, keys):
                mask &= df[col].eq(value).to_numpy()
            original_group = df[mask]
            row = dict(zip(key_cols, keys))
            row["curve_method"] = curve_method
            row.update(aggregate_curve_values(original_group, sample_group["sample_log10_sigma0"]))
            rows.append(row)
        return rows
    raise ValueError(f"unknown curve_method: {curve_method}")


def build_curve_bins(binned: pd.DataFrame) -> pd.DataFrame:
    log("building global curves...")
    frames: list[pd.DataFrame] = []
    all_valid = binned.copy()
    all_valid["source_subset"] = "all_valid"
    frames.append(all_valid)
    conservative = binned[binned["is_conservative_valid_sigma0_bool"]].copy()
    conservative["source_subset"] = "conservative_valid"
    frames.append(conservative)

    rows: list[dict[str, Any]] = []
    for source_df in frames:
        if source_df.empty:
            continue
        for group_scheme in ["global", "material_family"]:
            if group_scheme == "material_family":
                log("building material_family curves...")
            grouped_source = add_group_columns(source_df, group_scheme)
            for method in ["row_median", "sample_median"]:
                if method == "row_median":
                    log("computing row_median curves...")
                else:
                    log("computing sample_median curves...")
                rows.extend(make_curve_rows_for_method(grouped_source, method))
    return pd.DataFrame(rows)


def assign_reliability(curves: pd.DataFrame, min_rows: int, min_samples: int, min_papers: int) -> pd.DataFrame:
    out = curves.copy()
    out["is_reference_bin_candidate"] = (
        (out["row_count"] >= min_rows)
        & (out["sample_count"] >= min_samples)
        & (out["paper_count"] >= min_papers)
    )
    out["reliability_level"] = out.apply(reliability_level, axis=1)
    out["recommended_default"] = (
        out["source_subset"].eq("conservative_valid")
        & out["curve_method"].eq("sample_median")
        & out["is_reference_bin_candidate"]
    )
    ordered = [
        "source_subset",
        "group_scheme",
        "material_group_key",
        "material_group_label",
        "carrier_type",
        "curve_method",
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
        "sigma0_min_S_per_m",
        "sigma0_max_S_per_m",
        "T_median_K",
        "T_min_K",
        "T_max_K",
        "eta_median",
        "eta_min",
        "eta_max",
        "S_abs_median_uV_per_K",
        "sigma_median_S_per_m",
        "log10_sigma_median_S_per_m",
        "row_count",
        "sample_count",
        "paper_count",
        "formula_count",
        "is_reference_bin_candidate",
        "reliability_level",
        "recommended_default",
    ]
    return out[ordered].sort_values(CURVE_KEY_COLUMNS).reset_index(drop=True)


def build_coverage(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    key_cols = ["source_subset", "group_scheme", "material_group_key", "material_group_label", "carrier_type", "curve_method"]
    for keys, group in curves.groupby(key_cols, dropna=False, sort=False):
        row = dict(zip(key_cols, keys))
        row.update(
            {
                "total_curve_bins": len(group),
                "reliable_curve_bins": int(group["is_reference_bin_candidate"].sum()),
                "high_reliability_bins": int(group["reliability_level"].eq("high").sum()),
                "medium_reliability_bins": int(group["reliability_level"].eq("medium").sum()),
                "low_reliability_bins": int(group["reliability_level"].eq("low").sum()),
                "insufficient_bins": int(group["reliability_level"].eq("insufficient").sum()),
                "total_rows": int(group["row_count"].sum()),
                "total_samples": int(group["sample_count"].sum()),
                "total_papers": int(group["paper_count"].sum()),
                "T_bin_center_min_K": group["T_bin_center_K"].min(),
                "T_bin_center_max_K": group["T_bin_center_K"].max(),
                "T_min_K": group["T_min_K"].min(),
                "T_max_K": group["T_max_K"].max(),
                "log10_sigma0_ref_min": group["log10_sigma0_ref_S_per_m"].min(),
                "log10_sigma0_ref_max": group["log10_sigma0_ref_S_per_m"].max(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_sanity_checks(input_rows: int, binned: pd.DataFrame, dropped: pd.DataFrame, curves: pd.DataFrame, reliable: pd.DataFrame, default: pd.DataFrame, min_rows: int, min_samples: int, min_papers: int, full_run: bool) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["input_rows_equal_used_plus_dropped"] = input_rows == len(binned) + len(dropped)
    checks["used_T_positive_finite"] = bool(np.isfinite(binned["T_K"]).all() and (binned["T_K"] > 0).all())
    checks["used_sigma0_positive_finite"] = bool(np.isfinite(binned["sigma0_S_per_m"]).all() and (binned["sigma0_S_per_m"] > 0).all())
    checks["used_log10_sigma0_finite"] = bool(np.isfinite(binned["log10_sigma0_S_per_m"]).all())
    checks["T_inside_bins"] = bool(((binned["T_bin_left_K"] <= binned["T_K"]) & (binned["T_K"] < binned["T_bin_right_K"])).all())
    checks["T_bin_center_consistent"] = bool(np.allclose(binned["T_bin_center_K"], (binned["T_bin_left_K"] + binned["T_bin_right_K"]) / 2.0))
    checks["carrier_type_p_or_n_only"] = set(binned["carrier_type"].dropna()).issubset({"p", "n"})
    checks["curve_key_unique"] = not curves.duplicated(CURVE_KEY_COLUMNS).any()
    checks["sigma0_ref_consistent"] = bool(np.allclose(curves["sigma0_ref_S_per_m"], 10.0 ** curves["log10_sigma0_ref_S_per_m"]))
    checks["counts_positive"] = bool(((curves["row_count"] > 0) & (curves["sample_count"] > 0) & (curves["paper_count"] > 0)).all())
    expected_candidate = (
        (curves["row_count"] >= min_rows)
        & (curves["sample_count"] >= min_samples)
        & (curves["paper_count"] >= min_papers)
    )
    checks["reference_candidate_rule"] = bool((curves["is_reference_bin_candidate"] == expected_candidate).all())
    expected_reliability = curves.apply(reliability_level, axis=1)
    checks["reliability_level_rule"] = bool((curves["reliability_level"] == expected_reliability).all())
    checks["reliable_file_rule"] = bool(reliable.empty or reliable["is_reference_bin_candidate"].all())
    checks["default_file_rule"] = bool(default.empty or default["recommended_default"].all())
    expected_default = (
        curves["source_subset"].eq("conservative_valid")
        & curves["curve_method"].eq("sample_median")
        & curves["is_reference_bin_candidate"]
    )
    checks["recommended_default_rule"] = bool((curves["recommended_default"] == expected_default).all())
    q_ok = (curves["log10_sigma0_q25"] <= curves["log10_sigma0_ref_S_per_m"]) & (
        curves["log10_sigma0_ref_S_per_m"] <= curves["log10_sigma0_q75"]
    )
    checks["q25_ref_q75_mostly_consistent"] = bool(q_ok.all())
    if not q_ok.all():
        warnings.append(f"{int((~q_ok).sum())} bins have q25/ref/q75 numerical ordering issues")
    checks["reference_curve_bins_nonempty"] = len(curves) > 0
    if full_run:
        checks["full_run_reliable_nonempty"] = len(reliable) > 0
        checks["full_run_default_nonempty"] = len(default) > 0
    else:
        if reliable.empty:
            warnings.append("small test produced no reliable bins")
        if default.empty:
            warnings.append("small test produced no recommended_default bins")
    failures = [name for name, ok in checks.items() if not ok and not name.endswith("mostly_consistent")]
    return checks, failures, warnings


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(output_dir: Path, suffix: str, binned: pd.DataFrame, curves: pd.DataFrame, reliable: pd.DataFrame, default: pd.DataFrame, coverage: pd.DataFrame, dropped: pd.DataFrame) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_statuses: dict[str, str] = {}
    parquet_frames = {
        "step4_sigma0_binned_input_rows": binned,
        "step4_sigma0_reference_curve_bins": curves,
        "step4_sigma0_reference_curve_reliable": reliable,
        "step4_sigma0_reference_curve_default": default,
    }
    for base, frame in parquet_frames.items():
        frame.to_csv(output_dir / output_name(base, suffix, "csv"), index=False, encoding="utf-8-sig")
        ok, error = save_parquet(frame, output_dir / output_name(base, suffix, "parquet"))
        parquet_statuses[output_name(base, suffix, "parquet")] = "saved" if ok else f"not saved: {error}"
    coverage.to_csv(output_dir / output_name("step4_sigma0_curve_coverage_by_group", suffix, "csv"), index=False, encoding="utf-8-sig")
    dropped.to_csv(output_dir / output_name("step4_sigma0_dropped_rows", suffix, "csv"), index=False, encoding="utf-8-sig")
    return parquet_statuses


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
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(report_path: Path, input_path: Path, input_rows: int, binned: pd.DataFrame, dropped: pd.DataFrame, curves: pd.DataFrame, reliable: pd.DataFrame, default: pd.DataFrame, coverage: pd.DataFrame, checks: dict[str, bool], warnings: list[str], parquet_statuses: dict[str, str], args: argparse.Namespace, elapsed_sec: float) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    global_overview = coverage[coverage["group_scheme"].eq("global")].head(20)
    material_overview = coverage[coverage["group_scheme"].eq("material_family")].sort_values("reliable_curve_bins", ascending=False).head(10)
    conservative_sample = curves[
        curves["source_subset"].eq("conservative_valid") & curves["curve_method"].eq("sample_median")
    ].head(20)
    lines = [
        "# Step4 Sigma0 Reference Curve Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {input_rows}",
        f"- rows used for curves: {len(binned)}",
        f"- dropped rows: {len(dropped)}",
        f"- bin_width_K: {args.bin_width_k:g}",
        f"- bin_start_K: {args.bin_start_k:g}",
        f"- T_bin_center range: {numeric_summary(binned['T_bin_center_K']) if not binned.empty else 'n/a'}",
        f"- source_subset row counts: {binned.get('source_subset', pd.Series(dtype=str)).value_counts().to_dict() if 'source_subset' in binned else 'n/a'}",
        f"- group_scheme curve bin counts: {curves['group_scheme'].value_counts().to_dict()}",
        f"- curve_method curve bin counts: {curves['curve_method'].value_counts().to_dict()}",
        f"- carrier_type curve bin counts: {curves['carrier_type'].value_counts().to_dict()}",
        f"- is_reference_bin_candidate == True bins: {int(curves['is_reference_bin_candidate'].sum())}",
        f"- recommended_default == True bins: {int(curves['recommended_default'].sum())}",
        f"- reliability_level counts: {curves['reliability_level'].value_counts().to_dict()}",
        f"- log10_sigma0_ref_S_per_m summary: {numeric_summary(curves['log10_sigma0_ref_S_per_m'])}",
        f"- sigma0_ref_S_per_m summary: {numeric_summary(curves['sigma0_ref_S_per_m'])}",
        f"- sample_count summary: {numeric_summary(curves['sample_count'])}",
        f"- paper_count summary: {numeric_summary(curves['paper_count'])}",
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
            "## Global Curve Overview",
            "",
            dataframe_to_markdown(global_overview),
            "",
            "## Material Family Curve Overview Top 10",
            "",
            dataframe_to_markdown(material_overview),
            "",
            "## Conservative Valid + Sample Median Preview",
            "",
            dataframe_to_markdown(conservative_sample),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Warnings And Step5 Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- Step4 uses all available data to build reference curves; this is not an independent validation.")
    lines.append("- Step5 should evaluate sigma prediction with train/test splits to avoid leakage.")
    lines.append("- Step5 can compare global and material_family reference curves.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    report_path = args.report or (REPORT_DIR / output_name("step4_sigma0_reference_curve_report", args.output_suffix, "md"))
    full_run = args.max_rows is None

    log("loading step3 sigma0 valid rows...")
    df = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        df = df.head(args.max_rows).copy()
    input_rows = len(df)
    log(f"input rows: {input_rows}")
    log("validating required columns...")
    validate_required_columns(df)
    log("filtering rows usable for reference curves...")
    usable, dropped = filter_usable_rows(df)
    log("assigning 100 K temperature bins...")
    binned = assign_temperature_bins(usable, args.bin_width_k, args.bin_start_k)
    curves = build_curve_bins(binned)
    if curves.empty:
        raise RuntimeError("reference curve bins are empty")
    log("assigning reliability flags...")
    curves = assign_reliability(curves, args.min_rows_per_bin, args.min_samples_per_bin, args.min_papers_per_bin)
    reliable = curves[curves["is_reference_bin_candidate"]].copy()
    default = curves[curves["recommended_default"]].copy()
    coverage = build_coverage(curves)
    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(
        input_rows,
        binned,
        dropped,
        curves,
        reliable,
        default,
        args.min_rows_per_bin,
        args.min_samples_per_bin,
        args.min_papers_per_bin,
        full_run,
    )
    if failures:
        for failure in failures:
            print(f"[step4] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("writing outputs...")
    parquet_statuses = write_outputs(args.output, args.output_suffix, binned, curves, reliable, default, coverage, dropped)
    write_report(report_path, input_path, input_rows, binned, dropped, curves, reliable, default, coverage, checks, warnings, parquet_statuses, args, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
