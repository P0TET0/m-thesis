import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

K_B_OVER_E_UV_PER_K = 86.17333262145
ETA_MODEL = "SPB_acoustic_phonon_lookup"

STEP1_REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "S_V_per_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "carrier_type",
    "sigma_S_per_m",
    "match_method",
    "sigma_source",
    "is_usable_for_eta",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
]

LOOKUP_REQUIRED_COLUMNS = [
    "eta",
    "F0_eta",
    "F1_eta",
    "s_model",
    "S_abs_uV_per_K",
]

ETA_STATUSES = {
    "ok",
    "out_of_range_low_S",
    "out_of_range_high_S",
    "invalid_S",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign eta to Step1 eta input candidates using the Step2A lookup table."
    )
    parser.add_argument("--input", type=Path, default=None, help="Step1 eta input candidates")
    parser.add_argument("--lookup", type=Path, default=None, help="Step2A eta lookup table")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR, help="output directory")
    parser.add_argument("--report", type=Path, default=None, help="report markdown path")
    parser.add_argument("--max-rows", type=int, default=None, help="use only the first N rows")
    parser.add_argument("--output-suffix", default="", help="suffix inserted before output extensions")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step2b] {message}", flush=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def resolve_first_existing(explicit: Path | None, candidates: list[Path], label: str) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"{label} file not found: {explicit}")
    for path in candidates:
        if path.exists():
            return path
    candidate_text = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"{label} file not found. Checked:\n{candidate_text}")


def resolve_step1_input(explicit: Path | None) -> Path:
    return resolve_first_existing(
        explicit,
        [
            PROJECT_ROOT / "data" / "processed" / "step1_eta_input_candidates.parquet",
            PROJECT_ROOT / "data" / "processed" / "step1_eta_input_candidates.csv",
            PROCESSED_DIR / "step1_eta_input_candidates.parquet",
            PROCESSED_DIR / "step1_eta_input_candidates.csv",
        ],
        "Step1 eta input candidates",
    )


def resolve_lookup(explicit: Path | None) -> Path:
    return resolve_first_existing(
        explicit,
        [
            PROCESSED_DIR / "step2_eta_lookup_table.parquet",
            PROCESSED_DIR / "step2_eta_lookup_table.csv",
            PROJECT_ROOT / "data" / "processed" / "step2_eta_lookup_table.parquet",
            PROJECT_ROOT / "data" / "processed" / "step2_eta_lookup_table.csv",
        ],
        "Step2A eta lookup table",
    )


def validate_required_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def drop_sigma0_columns(df: pd.DataFrame) -> pd.DataFrame:
    sigma0_columns = [
        col
        for col in df.columns
        if "sigma0" in str(col).casefold()
        and str(col) not in {"is_valid_for_sigma0_step3"}
    ]
    return df.drop(columns=sigma0_columns) if sigma0_columns else df


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def prepare_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    out = lookup.copy()
    for column in LOOKUP_REQUIRED_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=LOOKUP_REQUIRED_COLUMNS).sort_values("eta").reset_index(drop=True)
    if out.empty:
        raise ValueError("lookup table has no valid numeric rows")
    if not np.all(np.diff(out["eta"].to_numpy()) > 0):
        raise ValueError("lookup eta must be strictly increasing")
    if not np.all(np.diff(out["s_model"].to_numpy()) <= 1e-12):
        raise ValueError("lookup s_model must be monotonically decreasing")
    return out


def assign_eta(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    out = drop_sigma0_columns(df.copy())
    for column in ["S_V_per_K", "S_uV_per_K", "S_abs_uV_per_K", "sigma_S_per_m", "T_K"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["S_abs_uV_per_K"] = out["S_abs_uV_per_K"].where(
        np.isfinite(out["S_abs_uV_per_K"]),
        out["S_uV_per_K"].abs(),
    )
    out["s_abs_dimensionless"] = out["S_abs_uV_per_K"] / K_B_OVER_E_UV_PER_K

    lookup_eta = lookup["eta"].to_numpy(dtype=float)
    lookup_s = lookup["s_model"].to_numpy(dtype=float)
    lookup_f0 = lookup["F0_eta"].to_numpy(dtype=float)
    lookup_f1 = lookup["F1_eta"].to_numpy(dtype=float)
    lookup_s_uV = lookup["S_abs_uV_per_K"].to_numpy(dtype=float)
    s_min = float(np.nanmin(lookup_s))
    s_max = float(np.nanmax(lookup_s))

    s_values = out["s_abs_dimensionless"].to_numpy(dtype=float)
    finite_positive_s = np.isfinite(s_values) & (s_values > 0)
    status = np.full(len(out), "invalid_S", dtype=object)
    status[finite_positive_s & (s_values < s_min)] = "out_of_range_low_S"
    status[finite_positive_s & (s_values > s_max)] = "out_of_range_high_S"
    ok = finite_positive_s & (s_values >= s_min) & (s_values <= s_max)
    status[ok] = "ok"
    out["eta_status"] = status

    eta = np.full(len(out), np.nan, dtype=float)
    # np.interp requires ascending x. s_model decreases with eta, so reverse both.
    eta[ok] = np.interp(s_values[ok], lookup_s[::-1], lookup_eta[::-1])
    out["eta"] = eta
    out["eta_is_finite"] = np.isfinite(out["eta"])

    out["F0_eta"] = np.nan
    out["F1_eta"] = np.nan
    out["S_model_abs_uV_per_K"] = np.nan
    ok_eta = out["eta"].to_numpy(dtype=float)
    out.loc[ok, "F0_eta"] = np.interp(ok_eta[ok], lookup_eta, lookup_f0)
    out.loc[ok, "F1_eta"] = np.interp(ok_eta[ok], lookup_eta, lookup_f1)
    out.loc[ok, "S_model_abs_uV_per_K"] = np.interp(ok_eta[ok], lookup_eta, lookup_s_uV)

    signed = np.full(len(out), np.nan, dtype=float)
    carrier = out["carrier_type"].astype(str).to_numpy()
    model_abs = out["S_model_abs_uV_per_K"].to_numpy(dtype=float)
    signed[(carrier == "p") & np.isfinite(model_abs)] = model_abs[(carrier == "p") & np.isfinite(model_abs)]
    signed[(carrier == "n") & np.isfinite(model_abs)] = -model_abs[(carrier == "n") & np.isfinite(model_abs)]
    out["S_model_signed_uV_per_K"] = signed
    out["S_eta_abs_error_uV_per_K"] = out["S_abs_uV_per_K"] - out["S_model_abs_uV_per_K"]
    out["eta_ge_1"] = (out["eta_status"] == "ok") & (out["eta"] >= 1.0)
    out["is_valid_for_sigma0_step3"] = (
        (out["eta_status"] == "ok")
        & (out["eta"] >= 1.0)
        & np.isfinite(out["sigma_S_per_m"])
        & (out["sigma_S_per_m"] > 0)
    )
    out["eta_model"] = ETA_MODEL
    return out


def build_family_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (family, carrier), group in df.groupby(["material_family_raw", "carrier_type"], dropna=False, sort=False):
        eta_ok = group[group["eta_status"] == "ok"]
        rows.append(
            {
                "material_family_raw": family,
                "carrier_type": carrier,
                "row_count": len(group),
                "eta_ok_count": int((group["eta_status"] == "ok").sum()),
                "eta_ge1_count": int(group["eta_ge_1"].sum()),
                "sigma0_step3_valid_count": int(group["is_valid_for_sigma0_step3"].sum()),
                "eta_min": eta_ok["eta"].min() if not eta_ok.empty else np.nan,
                "eta_median": eta_ok["eta"].median() if not eta_ok.empty else np.nan,
                "eta_max": eta_ok["eta"].max() if not eta_ok.empty else np.nan,
                "S_abs_min_uV_per_K": group["S_abs_uV_per_K"].min(),
                "S_abs_median_uV_per_K": group["S_abs_uV_per_K"].median(),
                "S_abs_max_uV_per_K": group["S_abs_uV_per_K"].max(),
                "T_min_K": group["T_K"].min(),
                "T_max_K": group["T_K"].max(),
                "paper_count": group["paper_id"].nunique(dropna=True),
                "sample_count": group["sample_group_id"].nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows)


def run_sanity_checks(
    input_rows: int,
    df: pd.DataFrame,
    ge1: pd.DataFrame,
    conservative_ge1: pd.DataFrame,
) -> tuple[dict[str, bool], list[str], list[str]]:
    checks: dict[str, bool] = {}
    warnings: list[str] = []

    checks["output_rows_equal_input_rows"] = len(df) == input_rows
    checks["row_id_unique"] = df["row_id"].is_unique
    checks["carrier_type_p_or_n_only"] = set(df["carrier_type"].dropna()).issubset({"p", "n"})
    checks["S_abs_matches_abs_signed_S"] = bool(
        np.allclose(df["S_abs_uV_per_K"], df["S_uV_per_K"].abs(), rtol=1e-10, atol=1e-8)
    )
    checks["s_abs_dimensionless_consistent"] = bool(
        np.allclose(
            df["s_abs_dimensionless"],
            df["S_abs_uV_per_K"] / K_B_OVER_E_UV_PER_K,
            rtol=1e-12,
            atol=1e-12,
        )
    )
    checks["eta_status_not_missing"] = bool(df["eta_status"].notna().all())
    checks["eta_status_allowed"] = set(df["eta_status"].dropna()).issubset(ETA_STATUSES)

    ok = df["eta_status"] == "ok"
    not_ok = ~ok
    checks["eta_finite_for_ok"] = bool(np.isfinite(df.loc[ok, "eta"]).all())
    checks["eta_nan_for_not_ok"] = bool(df.loc[not_ok, "eta"].isna().all())
    checks["F0_positive_finite_for_ok"] = bool(
        np.isfinite(df.loc[ok, "F0_eta"]).all() and (df.loc[ok, "F0_eta"] > 0).all()
    )
    max_abs_error = df.loc[ok, "S_eta_abs_error_uV_per_K"].abs().max() if ok.any() else 0.0
    checks["S_eta_abs_error_le_0_1_uV_for_ok"] = bool(max_abs_error <= 0.1)
    checks["eta_ge_1_rule"] = bool((df["eta_ge_1"] == (ok & (df["eta"] >= 1.0))).all())
    expected_valid = ok & (df["eta"] >= 1.0) & np.isfinite(df["sigma_S_per_m"]) & (df["sigma_S_per_m"] > 0)
    checks["is_valid_for_sigma0_step3_rule"] = bool((df["is_valid_for_sigma0_step3"] == expected_valid).all())
    checks["ge1_candidates_rule"] = bool(ge1.empty or ge1["is_valid_for_sigma0_step3"].all())
    checks["conservative_ge1_candidates_rule"] = bool(
        conservative_ge1.empty
        or (
            conservative_ge1["is_valid_for_sigma0_step3"].all()
            and as_bool(conservative_ge1["is_conservative_main_analysis"]).all()
        )
    )
    sigma0_columns = [
        col
        for col in df.columns
        if "sigma0" in str(col).casefold()
        and str(col) not in {"is_valid_for_sigma0_step3"}
    ]
    checks["no_sigma0_columns"] = len(sigma0_columns) == 0
    if sigma0_columns:
        warnings.append(f"sigma0-like columns found: {sigma0_columns}")

    failures = [name for name, ok_value in checks.items() if not ok_value]
    return checks, failures, warnings


def save_parquet(df: pd.DataFrame, path: Path) -> tuple[bool, str]:
    try:
        df.to_parquet(path, index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def output_name(base: str, suffix: str, ext: str) -> str:
    return f"{base}{suffix}.{ext}"


def write_outputs(
    df: pd.DataFrame,
    ge1: pd.DataFrame,
    conservative_ge1: pd.DataFrame,
    failed: pd.DataFrame,
    family_counts: pd.DataFrame,
    output_dir: Path,
    suffix: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "step2_eta_calculated": df,
        "step2_eta_ge1_candidates": ge1,
        "step2_conservative_eta_ge1_candidates": conservative_ge1,
    }
    statuses: dict[str, str] = {}
    for base, frame in frames.items():
        csv_path = output_dir / output_name(base, suffix, "csv")
        parquet_path = output_dir / output_name(base, suffix, "parquet")
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        ok, error = save_parquet(frame, parquet_path)
        statuses[parquet_path.name] = "saved" if ok else f"not saved: {error}"
    failed.to_csv(
        output_dir / output_name("step2_eta_failed_or_out_of_range", suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    family_counts.to_csv(
        output_dir / output_name("step2_eta_counts_by_material_family", suffix, "csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return statuses


def numeric_summary(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"min={values.min():.6g}, median={values.median():.6g}, max={values.max():.6g}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "n/a"
    text_df = df.copy()
    text_df.columns = [str(col) for col in text_df.columns]
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text_df.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text_df.columns) + " |" for _, row in text_df.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(
    report_path: Path,
    input_path: Path,
    lookup_path: Path,
    lookup: pd.DataFrame,
    df: pd.DataFrame,
    ge1: pd.DataFrame,
    conservative_ge1: pd.DataFrame,
    family_counts: pd.DataFrame,
    checks: dict[str, bool],
    warnings: list[str],
    parquet_statuses: dict[str, str],
    elapsed_sec: float,
    max_rows: int | None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    status_counts = df["eta_status"].value_counts(dropna=False).to_dict()
    carrier_ge1 = df[df["eta_ge_1"]].groupby("carrier_type").size().reset_index(name="eta_ge1_count")
    material_top = (
        family_counts.sort_values("eta_ge1_count", ascending=False)
        .head(10)[["material_family_raw", "carrier_type", "eta_ge1_count", "row_count", "sample_count", "paper_count"]]
        if not family_counts.empty
        else pd.DataFrame()
    )
    ok = df["eta_status"] == "ok"
    max_error = df.loc[ok, "S_eta_abs_error_uV_per_K"].abs().max() if ok.any() else np.nan
    eta_ge1_threshold = lookup.loc[lookup["eta"] >= 1.0, "S_abs_uV_per_K"].max()
    lines = [
        "# Step2B Eta Assignment Report",
        "",
        "## Summary",
        "",
        f"- input_file: {input_path}",
        f"- input_rows: {len(df)}",
        f"- max_rows: {max_rows if max_rows is not None else 'none'}",
        f"- lookup_table_file: {lookup_path}",
        f"- lookup eta range: {lookup['eta'].min():.6g} to {lookup['eta'].max():.6g}",
        f"- lookup S_abs_uV_per_K range: {lookup['S_abs_uV_per_K'].min():.6g} to {lookup['S_abs_uV_per_K'].max():.6g}",
        f"- eta_status counts: {status_counts}",
        f"- eta ok rows: {int(ok.sum())}",
        f"- eta failed or out-of-range rows: {int((~ok).sum())}",
        f"- eta >= 1 rows: {int(df['eta_ge_1'].sum())}",
        f"- is_valid_for_sigma0_step3 == True rows: {int(df['is_valid_for_sigma0_step3'].sum())}",
        f"- conservative eta >= 1 candidate rows: {len(conservative_ge1)}",
        f"- eta summary: {numeric_summary(df.loc[ok, 'eta'])}",
        f"- S_abs_uV_per_K summary: {numeric_summary(df['S_abs_uV_per_K'])}",
        f"- S_eta_abs_error_uV_per_K summary: {numeric_summary(df.loc[ok, 'S_eta_abs_error_uV_per_K'])}",
        f"- max abs S_eta_abs_error_uV_per_K: {max_error:.6g}",
        f"- eta >= 1 roughly corresponds to S_abs_uV_per_K <= {eta_ge1_threshold:.6g}",
        f"- elapsed_seconds: {elapsed_sec:.2f}",
        "",
        "## Parquet Status",
        "",
    ]
    for filename, status in parquet_statuses.items():
        lines.append(f"- {filename}: {status}")
    lines.extend(
        [
            "",
            "## Carrier Type Eta >= 1 Counts",
            "",
            dataframe_to_markdown(carrier_ge1),
            "",
            "## Material Family Eta >= 1 Top 10",
            "",
            dataframe_to_markdown(material_top),
            "",
            "## Sanity Check",
            "",
        ]
    )
    for name, ok_value in checks.items():
        lines.append(f"- {name}: {ok_value}")
    lines.extend(["", "## Warnings And Step3 Notes", ""])
    if warnings:
        for warning in warnings:
            lines.append(f"- WARNING: {warning}")
    else:
        lines.append("- WARNING: none")
    lines.append("- Step3 can compute sigma0 as sigma_S_per_m / F0_eta for is_valid_for_sigma0_step3 rows.")
    lines.append("- sigma0 is intentionally not computed in Step2B.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    input_path = resolve_step1_input(args.input)
    lookup_path = resolve_lookup(args.lookup)
    report_path = args.report or (REPORT_DIR / output_name("step2_eta_report", args.output_suffix, "md"))

    log("loading step1 eta input candidates...")
    step1 = read_table(input_path)
    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        step1 = step1.head(args.max_rows).copy()
    log(f"input rows: {len(step1)}")

    log("loading eta lookup table...")
    lookup = prepare_lookup(read_table(lookup_path))
    log(f"lookup rows: {len(lookup)}")

    log("validating required columns...")
    validate_required_columns(step1, STEP1_REQUIRED_COLUMNS, "Step1 eta input candidates")
    validate_required_columns(lookup, LOOKUP_REQUIRED_COLUMNS, "Step2A eta lookup table")

    log("computing s_abs_dimensionless...")
    log("interpolating eta...")
    calculated = assign_eta(step1, lookup)

    log("reconstructing S from eta...")
    log("assigning eta_ge_1 and sigma0 step3 flags...")
    ge1 = calculated[calculated["is_valid_for_sigma0_step3"]].copy()
    conservative_ge1 = calculated[
        calculated["is_valid_for_sigma0_step3"] & as_bool(calculated["is_conservative_main_analysis"])
    ].copy()
    failed = calculated[calculated["eta_status"] != "ok"].copy()
    family_counts = build_family_counts(calculated)

    log("running sanity checks...")
    checks, failures, warnings = run_sanity_checks(len(step1), calculated, ge1, conservative_ge1)
    if failures:
        for failure in failures:
            print(f"[step2b] FAIL: {failure}", flush=True)
        raise SystemExit(1)

    log("writing outputs...")
    parquet_statuses = write_outputs(
        calculated,
        ge1,
        conservative_ge1,
        failed,
        family_counts,
        args.output,
        args.output_suffix,
    )
    write_report(
        report_path,
        input_path,
        lookup_path,
        lookup,
        calculated,
        ge1,
        conservative_ge1,
        family_counts,
        checks,
        warnings,
        parquet_statuses,
        time.time() - started,
        args.max_rows,
    )

    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
