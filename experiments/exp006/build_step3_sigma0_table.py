import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

DEFAULT_INPUT_PARQUET = PROCESSED_DIR / "step2_eta_ge1_candidates.parquet"
DEFAULT_INPUT_CSV = PROCESSED_DIR / "step2_eta_ge1_candidates.csv"

SIGMA0_MODEL = "sigma0_equals_sigma_over_F0_eta"

CORE_REQUIRED_COLUMNS = [
    "row_id",
    "T_K",
    "S_uV_per_K",
    "carrier_type",
    "sigma_S_per_m",
    "eta",
    "eta_status",
    "eta_ge_1",
    "F0_eta",
    "is_valid_for_sigma0_step3",
    "is_conservative_main_analysis",
]

EXPECTED_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "sample_label",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "S_V_per_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "carrier_type",
    "sigma_S_per_m",
    "sigma_source",
    "match_method",
    "eta",
    "eta_status",
    "eta_ge_1",
    "F0_eta",
    "F1_eta",
    "is_valid_for_sigma0_step3",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
    "eta_model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute sigma0 from Step2B eta >= 1 candidates.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step3] {message}", flush=True)


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
        raise FileNotFoundError(f"Step2 eta >= 1 candidate input not found: {explicit}")
    if DEFAULT_INPUT_PARQUET.exists():
        return DEFAULT_INPUT_PARQUET
    if DEFAULT_INPUT_CSV.exists():
        return DEFAULT_INPUT_CSV
    raise FileNotFoundError(
        "Step2 eta >= 1 candidate input not found. Expected "
        "experiments/exp006/data/processed/step2_eta_ge1_candidates.parquet or .csv"
    )


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing_core = sorted(set(CORE_REQUIRED_COLUMNS) - set(df.columns))
    if missing_core:
        raise ValueError(f"input is missing required analysis columns: {missing_core}")
    out = df.copy()
    for column in EXPECTED_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    for column in ["T_K", "S_uV_per_K", "S_V_per_K", "S_abs_uV_per_K", "sigma_S_per_m", "eta", "F0_eta", "F1_eta"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def assign_sigma0_status(df: pd.DataFrame) -> pd.Series:
    eta_status_ok = df["eta_status"].astype(str).eq("ok")
    eta_ge1 = as_bool(df["eta_ge_1"])
    step2_valid = as_bool(df["is_valid_for_sigma0_step3"])
    eta = pd.to_numeric(df["eta"], errors="coerce")
    f0 = pd.to_numeric(df["F0_eta"], errors="coerce")
    sigma = pd.to_numeric(df["sigma_S_per_m"], errors="coerce")

    status = np.full(len(df), "ok", dtype=object)
    status[~eta_status_ok] = "invalid_eta_status"
    mask = eta_status_ok & (~eta_ge1 | ~(eta >= 1.0))
    status[mask] = "eta_lt_1"
    mask = eta_status_ok & eta_ge1 & (eta >= 1.0) & (~np.isfinite(f0) | (f0 <= 0))
    status[mask] = "invalid_F0_eta"
    mask = eta_status_ok & eta_ge1 & (eta >= 1.0) & np.isfinite(f0) & (f0 > 0) & (~np.isfinite(sigma) | (sigma <= 0))
    status[mask] = "invalid_sigma"
    mask = (
        eta_status_ok
        & eta_ge1
        & (eta >= 1.0)
        & np.isfinite(f0)
        & (f0 > 0)
        & np.isfinite(sigma)
        & (sigma > 0)
        & ~step2_valid
    )
    status[mask] = "invalid_step2_flag"
    return pd.Series(status, index=df.index)


def compute_sigma0(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sigma0_calc_status"] = assign_sigma0_status(out)
    valid = out["sigma0_calc_status"].eq("ok")
    out["sigma0_S_per_m"] = np.nan
    out.loc[valid, "sigma0_S_per_m"] = out.loc[valid, "sigma_S_per_m"] / out.loc[valid, "F0_eta"]
    out["log10_sigma0_S_per_m"] = np.nan
    out.loc[valid, "log10_sigma0_S_per_m"] = np.log10(out.loc[valid, "sigma0_S_per_m"])
    out["log10_sigma_S_per_m"] = np.nan
    positive_sigma = np.isfinite(out["sigma_S_per_m"]) & (out["sigma_S_per_m"] > 0)
    out.loc[positive_sigma, "log10_sigma_S_per_m"] = np.log10(out.loc[positive_sigma, "sigma_S_per_m"])
    out["sigma_reconstructed_S_per_m"] = np.nan
    out.loc[valid, "sigma_reconstructed_S_per_m"] = out.loc[valid, "sigma0_S_per_m"] * out.loc[valid, "F0_eta"]
    out["log10_sigma_reconstructed_S_per_m"] = np.nan
    out.loc[valid, "log10_sigma_reconstructed_S_per_m"] = np.log10(out.loc[valid, "sigma_reconstructed_S_per_m"])
    out["sigma0_reconstruction_log_error"] = np.nan
    out.loc[valid, "sigma0_reconstruction_log_error"] = np.log10(
        out.loc[valid, "sigma_reconstructed_S_per_m"] / out.loc[valid, "sigma_S_per_m"]
    )
    out["sigma0_calc_model"] = SIGMA0_MODEL
    out["is_valid_sigma0"] = valid
    out["is_conservative_valid_sigma0"] = valid & as_bool(out["is_conservative_main_analysis"])
    return out


def group_summary(group: pd.DataFrame) -> dict[str, Any]:
    valid = group[group["is_valid_sigma0"]]
    return {
        "row_count": len(group),
        "valid_sigma0_count": len(valid),
        "T_min_K": group["T_K"].min(),
        "T_max_K": group["T_K"].max(),
        "eta_min": valid["eta"].min() if not valid.empty else np.nan,
        "eta_median": valid["eta"].median() if not valid.empty else np.nan,
        "eta_max": valid["eta"].max() if not valid.empty else np.nan,
        "S_abs_median_uV_per_K": group["S_abs_uV_per_K"].median() if "S_abs_uV_per_K" in group else np.nan,
        "sigma_median_S_per_m": valid["sigma_S_per_m"].median() if not valid.empty else np.nan,
        "log10_sigma_median_S_per_m": valid["log10_sigma_S_per_m"].median() if not valid.empty else np.nan,
        "sigma0_median_S_per_m": valid["sigma0_S_per_m"].median() if not valid.empty else np.nan,
        "log10_sigma0_median_S_per_m": valid["log10_sigma0_S_per_m"].median() if not valid.empty else np.nan,
        "sigma0_min_S_per_m": valid["sigma0_S_per_m"].min() if not valid.empty else np.nan,
        "sigma0_max_S_per_m": valid["sigma0_S_per_m"].max() if not valid.empty else np.nan,
        "log10_sigma0_min_S_per_m": valid["log10_sigma0_S_per_m"].min() if not valid.empty else np.nan,
        "log10_sigma0_max_S_per_m": valid["log10_sigma0_S_per_m"].max() if not valid.empty else np.nan,
    }


def build_sample_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (sample_group_id, carrier_type), group in df.groupby(["sample_group_id", "carrier_type"], dropna=False, sort=False):
        row = {
            "sample_group_id": sample_group_id,
            "carrier_type": carrier_type,
            "paper_id": group["paper_id"].iloc[0],
            "sample_id": group["sample_id"].iloc[0],
            "sample_key": group["sample_key"].iloc[0],
            "formula_raw": group["formula_raw"].iloc[0],
            "material_name_raw": group["material_name_raw"].iloc[0],
            "material_family_raw": group["material_family_raw"].iloc[0],
            **group_summary(group),
            "sample_has_sign_change": bool(as_bool(group["sample_has_sign_change"]).any()),
        }
        row["is_conservative_sample"] = not row["sample_has_sign_change"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, carrier_type), group in df.groupby(["material_family_raw", "carrier_type"], dropna=False, sort=False):
        row = {
            "material_family_raw": family,
            "carrier_type": carrier_type,
            **group_summary(group),
            "paper_count": group["paper_id"].nunique(dropna=True),
            "sample_count": group["sample_group_id"].nunique(dropna=True),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def run_sanity_checks(input_rows: int, df: pd.DataFrame, valid_df: pd.DataFrame, conservative_df: pd.DataFrame) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []
    checks["output_rows_equal_input_rows"] = len(df) == input_rows
    checks["row_id_unique"] = df["row_id"].is_unique
    if not df["eta_status"].astype(str).eq("ok").all():
        warnings.append("not all eta_status values are ok")
    if not as_bool(df["eta_ge_1"]).all():
        warnings.append("not all eta_ge_1 values are True")
    valid = df["is_valid_sigma0"]
    checks["valid_sigma0_positive_finite"] = bool(
        valid.empty or (np.isfinite(df.loc[valid, "sigma0_S_per_m"]).all() and (df.loc[valid, "sigma0_S_per_m"] > 0).all())
    )
    checks["valid_log10_sigma0_finite"] = bool(valid.empty or np.isfinite(df.loc[valid, "log10_sigma0_S_per_m"]).all())
    checks["valid_reconstructed_sigma_positive_finite"] = bool(
        valid.empty
        or (
            np.isfinite(df.loc[valid, "sigma_reconstructed_S_per_m"]).all()
            and (df.loc[valid, "sigma_reconstructed_S_per_m"] > 0).all()
        )
    )
    max_abs_error = df.loc[valid, "sigma0_reconstruction_log_error"].abs().max() if valid.any() else 0.0
    checks["reconstruction_log_error_le_1e_10"] = bool(max_abs_error <= 1e-10)
    checks["status_ok_matches_is_valid_sigma0"] = bool((df["sigma0_calc_status"].eq("ok") == df["is_valid_sigma0"]).all())
    checks["valid_output_only_valid_rows"] = bool(valid_df.empty or valid_df["is_valid_sigma0"].all())
    checks["conservative_output_only_conservative_valid_rows"] = bool(
        conservative_df.empty or conservative_df["is_conservative_valid_sigma0"].all()
    )
    checks["sigma0_formula_consistent"] = bool(
        valid.empty
        or np.allclose(
            df.loc[valid, "sigma0_S_per_m"],
            df.loc[valid, "sigma_S_per_m"] / df.loc[valid, "F0_eta"],
            rtol=1e-10,
            atol=0.0,
        )
    )
    checks["F0_positive_for_valid"] = bool(valid.empty or (df.loc[valid, "F0_eta"] > 0).all())
    checks["sigma_positive_for_valid"] = bool(valid.empty or (df.loc[valid, "sigma_S_per_m"] > 0).all())
    checks["log10_sigma0_consistent"] = bool(
        valid.empty
        or np.allclose(
            df.loc[valid, "log10_sigma0_S_per_m"],
            np.log10(df.loc[valid, "sigma0_S_per_m"]),
            rtol=1e-12,
            atol=1e-12,
        )
    )
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures, warnings


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_outputs(df: pd.DataFrame, sample_summary: pd.DataFrame, family_summary: pd.DataFrame, output_dir: Path, suffix: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = df[df["is_valid_sigma0"]].copy()
    conservative = df[df["is_conservative_valid_sigma0"]].copy()
    failed = df[~df["is_valid_sigma0"]].copy()
    statuses: dict[str, str] = {}
    for base, frame in {
        "step3_sigma0_calculated": df,
        "step3_sigma0_valid": valid,
        "step3_conservative_sigma0_valid": conservative,
    }.items():
        frame.to_csv(output_dir / output_name(base, suffix, "csv"), index=False, encoding="utf-8-sig")
        ok, error = save_parquet(frame, output_dir / output_name(base, suffix, "parquet"))
        statuses[output_name(base, suffix, "parquet")] = "saved" if ok else f"not saved: {error}"
    failed.to_csv(output_dir / output_name("step3_sigma0_failed", suffix, "csv"), index=False, encoding="utf-8-sig")
    sample_summary.to_csv(output_dir / output_name("step3_sigma0_summary_by_sample", suffix, "csv"), index=False, encoding="utf-8-sig")
    family_summary.to_csv(output_dir / output_name("step3_sigma0_summary_by_material_family", suffix, "csv"), index=False, encoding="utf-8-sig")
    return statuses


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


def write_report(report_path: Path, input_path: Path, df: pd.DataFrame, sample_summary: pd.DataFrame, family_summary: pd.DataFrame, checks: dict[str, bool], warnings: list[str], parquet_statuses: dict[str, str], elapsed_sec: float, max_rows: int | None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    valid = df[df["is_valid_sigma0"]]
    max_abs_error = valid["sigma0_reconstruction_log_error"].abs().max() if not valid.empty else np.nan
    carrier_counts = valid.groupby("carrier_type").size().reset_index(name="valid_sigma0_count")
    sigma_source_counts = valid.groupby("sigma_source").size().reset_index(name="valid_sigma0_count")
    match_counts = valid.groupby("match_method").size().reset_index(name="valid_sigma0_count")
    material_top = family_summary.sort_values("valid_sigma0_count", ascending=False).head(10)
    lines = [
        "# Step3 Sigma0 Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {len(df)}",
        f"- max_rows: {max_rows if max_rows is not None else 'none'}",
        f"- sigma0_calc_status counts: {df['sigma0_calc_status'].value_counts(dropna=False).to_dict()}",
        f"- is_valid_sigma0 == True rows: {int(df['is_valid_sigma0'].sum())}",
        f"- is_conservative_valid_sigma0 == True rows: {int(df['is_conservative_valid_sigma0'].sum())}",
        f"- eta summary: {numeric_summary(valid['eta'])}",
        f"- F0_eta summary: {numeric_summary(valid['F0_eta'])}",
        f"- sigma_S_per_m summary: {numeric_summary(valid['sigma_S_per_m'])}",
        f"- sigma0_S_per_m summary: {numeric_summary(valid['sigma0_S_per_m'])}",
        f"- log10_sigma0_S_per_m summary: {numeric_summary(valid['log10_sigma0_S_per_m'])}",
        f"- sigma0_reconstruction_log_error summary: {numeric_summary(valid['sigma0_reconstruction_log_error'])}",
        f"- sigma0_reconstruction_log_error max_abs: {max_abs_error:.6g}",
        f"- sample summary rows: {len(sample_summary)}",
        f"- material family summary rows: {len(family_summary)}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for name, status in parquet_statuses.items():
        lines.append(f"- {name}: {status}")
    lines.extend(
        [
            "",
            "## Valid Sigma0 By Carrier Type",
            "",
            dataframe_to_markdown(carrier_counts),
            "",
            "## Valid Sigma0 By Material Family Top 10",
            "",
            dataframe_to_markdown(material_top),
            "",
            "## Valid Sigma0 By Sigma Source",
            "",
            dataframe_to_markdown(sigma_source_counts),
            "",
            "## Valid Sigma0 By Match Method",
            "",
            dataframe_to_markdown(match_counts),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok in checks.items():
        lines.append(f"- {name}: {ok}")
    lines.extend(["", "## Warnings And Step4 Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- Step4 should build 100 K temperature bins before calculating median curves.")
    lines.append("- Compare median log10_sigma0 as well as median sigma0 because sigma0 spans many orders of magnitude.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_input(args.input)
    report_path = args.report or (REPORT_DIR / output_name("step3_sigma0_report", args.output_suffix, "md"))

    log("loading step2 eta >= 1 candidates...")
    df = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        df = df.head(args.max_rows).copy()
    log(f"input rows: {len(df)}")

    log("validating required columns...")
    prepared = validate_and_prepare(df)
    log("assigning sigma0 calculation status...")
    log("computing sigma0...")
    calculated = compute_sigma0(prepared)
    log("reconstructing sigma for sanity check...")
    log("building sample summary...")
    sample_summary = build_sample_summary(calculated)
    log("building material family summary...")
    family_summary = build_family_summary(calculated)
    valid = calculated[calculated["is_valid_sigma0"]].copy()
    conservative = calculated[calculated["is_conservative_valid_sigma0"]].copy()
    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(len(df), calculated, valid, conservative)
    if failures:
        for failure in failures:
            print(f"[step3] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    log("writing outputs...")
    parquet_statuses = write_outputs(calculated, sample_summary, family_summary, args.output, args.output_suffix)
    write_report(report_path, input_path, calculated, sample_summary, family_summary, checks, warnings, parquet_statuses, time.time() - started, args.max_rows)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
