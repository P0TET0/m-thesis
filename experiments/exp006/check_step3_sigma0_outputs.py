import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

REQUIRED_COLUMNS = [
    "row_id",
    "sigma_S_per_m",
    "eta",
    "eta_status",
    "eta_ge_1",
    "F0_eta",
    "is_valid_for_sigma0_step3",
    "is_conservative_main_analysis",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_reconstructed_S_per_m",
    "log10_sigma_reconstructed_S_per_m",
    "sigma0_reconstruction_log_error",
    "sigma0_calc_status",
    "sigma0_calc_model",
    "is_valid_sigma0",
    "is_conservative_valid_sigma0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step3 sigma0 outputs.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "step3_sigma0_calculated.csv")
    parser.add_argument("--valid", type=Path, default=PROCESSED_DIR / "step3_sigma0_valid.csv")
    parser.add_argument(
        "--conservative-valid",
        type=Path,
        default=PROCESSED_DIR / "step3_conservative_sigma0_valid.csv",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.input, args.valid, args.conservative_valid]:
        if not path.exists():
            failures.append(f"missing required output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    df = pd.read_csv(args.input, low_memory=False)
    valid_df = pd.read_csv(args.valid, low_memory=False)
    conservative_df = pd.read_csv(args.conservative_valid, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        failures.append(f"missing required columns: {missing}")

    if not df.empty and not missing:
        valid = as_bool(df["is_valid_sigma0"])
        conservative = as_bool(df["is_conservative_valid_sigma0"])
        if not df["row_id"].is_unique:
            failures.append("row_id is not unique")
        if not (df["sigma0_calc_status"].eq("ok") == valid).all():
            failures.append("sigma0_calc_status == ok does not match is_valid_sigma0")
        if valid.any():
            sigma0 = pd.to_numeric(df.loc[valid, "sigma0_S_per_m"], errors="coerce")
            log_sigma0 = pd.to_numeric(df.loc[valid, "log10_sigma0_S_per_m"], errors="coerce")
            sigma = pd.to_numeric(df.loc[valid, "sigma_S_per_m"], errors="coerce")
            f0 = pd.to_numeric(df.loc[valid, "F0_eta"], errors="coerce")
            reconstructed = pd.to_numeric(df.loc[valid, "sigma_reconstructed_S_per_m"], errors="coerce")
            log_error = pd.to_numeric(df.loc[valid, "sigma0_reconstruction_log_error"], errors="coerce")
            if not (np.isfinite(sigma0).all() and (sigma0 > 0).all()):
                failures.append("valid sigma0 values are not positive finite")
            if not np.isfinite(log_sigma0).all():
                failures.append("valid log10_sigma0 values are not finite")
            if not (np.isfinite(reconstructed).all() and (reconstructed > 0).all()):
                failures.append("valid reconstructed sigma values are not positive finite")
            if log_error.abs().max() > 1e-10:
                failures.append(f"reconstruction log error exceeds 1e-10: {log_error.abs().max()}")
            if not np.allclose(sigma0, sigma / f0, rtol=1e-10, atol=0.0):
                failures.append("sigma0 != sigma_S_per_m / F0_eta")
            if not (f0 > 0).all():
                failures.append("F0_eta is not positive for valid rows")
            if not (sigma > 0).all():
                failures.append("sigma_S_per_m is not positive for valid rows")
            if not np.allclose(log_sigma0, np.log10(sigma0), rtol=1e-12, atol=1e-12):
                failures.append("log10_sigma0 is inconsistent")
        if not valid_df.empty and not as_bool(valid_df["is_valid_sigma0"]).all():
            failures.append("valid output contains invalid rows")
        if not conservative_df.empty and not as_bool(conservative_df["is_conservative_valid_sigma0"]).all():
            failures.append("conservative-valid output contains invalid rows")

    print(f"rows: {len(df)}")
    print(f"sigma0_calc_status counts: {df['sigma0_calc_status'].value_counts(dropna=False).to_dict() if 'sigma0_calc_status' in df else {}}")
    print(f"valid sigma0 rows: {len(valid_df)}")
    print(f"conservative valid sigma0 rows: {len(conservative_df)}")
    if not df.empty and "sigma0_reconstruction_log_error" in df:
        valid = as_bool(df["is_valid_sigma0"])
        max_abs_error = pd.to_numeric(df.loc[valid, "sigma0_reconstruction_log_error"], errors="coerce").abs().max()
        print(f"max abs sigma0_reconstruction_log_error: {max_abs_error}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step3 sigma0 output checks passed")


if __name__ == "__main__":
    main()
