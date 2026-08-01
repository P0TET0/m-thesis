import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

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
    "s_abs_dimensionless",
    "eta",
    "eta_status",
    "eta_is_finite",
    "F0_eta",
    "F1_eta",
    "S_model_abs_uV_per_K",
    "S_model_signed_uV_per_K",
    "S_eta_abs_error_uV_per_K",
    "eta_ge_1",
    "is_valid_for_sigma0_step3",
    "eta_model",
]

K_B_OVER_E_UV_PER_K = 86.17333262145
ETA_STATUSES = {"ok", "out_of_range_low_S", "out_of_range_high_S", "invalid_S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step2B eta assignment outputs.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "step2_eta_calculated.csv")
    parser.add_argument("--ge1", type=Path, default=PROCESSED_DIR / "step2_eta_ge1_candidates.csv")
    parser.add_argument(
        "--conservative-ge1",
        type=Path,
        default=PROCESSED_DIR / "step2_conservative_eta_ge1_candidates.csv",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.input, args.ge1, args.conservative_ge1]:
        if not path.exists():
            fail(f"missing required output: {path}", failures)
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)

    df = pd.read_csv(args.input, low_memory=False)
    ge1 = pd.read_csv(args.ge1, low_memory=False)
    conservative = pd.read_csv(args.conservative_ge1, low_memory=False)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        fail(f"missing required columns: {missing}", failures)

    if not df.empty and not missing:
        if not df["row_id"].is_unique:
            fail("row_id is not unique", failures)
        if not set(df["carrier_type"].dropna()).issubset({"p", "n"}):
            fail("carrier_type contains values other than p/n", failures)
        s_abs = pd.to_numeric(df["S_abs_uV_per_K"], errors="coerce")
        s_signed = pd.to_numeric(df["S_uV_per_K"], errors="coerce")
        s_dim = pd.to_numeric(df["s_abs_dimensionless"], errors="coerce")
        if not np.allclose(s_abs, s_signed.abs(), rtol=1e-10, atol=1e-8):
            fail("S_abs_uV_per_K is not abs(S_uV_per_K)", failures)
        if not np.allclose(s_dim, s_abs / K_B_OVER_E_UV_PER_K, rtol=1e-12, atol=1e-12):
            fail("s_abs_dimensionless is inconsistent", failures)
        if not df["eta_status"].notna().all():
            fail("eta_status has missing values", failures)
        if not set(df["eta_status"].dropna()).issubset(ETA_STATUSES):
            fail("eta_status has unexpected values", failures)

        ok = df["eta_status"] == "ok"
        eta = pd.to_numeric(df["eta"], errors="coerce")
        f0 = pd.to_numeric(df["F0_eta"], errors="coerce")
        sigma = pd.to_numeric(df["sigma_S_per_m"], errors="coerce")
        if not np.isfinite(eta[ok]).all():
            fail("eta is not finite for eta_status == ok", failures)
        if not eta[~ok].isna().all():
            fail("eta is not NaN for eta_status != ok", failures)
        if not (np.isfinite(f0[ok]).all() and (f0[ok] > 0).all()):
            fail("F0_eta is not positive finite for eta_status == ok", failures)
        max_abs_error = pd.to_numeric(df.loc[ok, "S_eta_abs_error_uV_per_K"], errors="coerce").abs().max()
        if pd.notna(max_abs_error) and max_abs_error > 0.1:
            fail(f"S_eta_abs_error_uV_per_K exceeds 0.1 uV/K: {max_abs_error}", failures)
        eta_ge_1 = as_bool(df["eta_ge_1"])
        if not (eta_ge_1 == (ok & (eta >= 1.0))).all():
            fail("eta_ge_1 does not match eta >= 1 rule", failures)
        valid = as_bool(df["is_valid_for_sigma0_step3"])
        expected_valid = ok & (eta >= 1.0) & np.isfinite(sigma) & (sigma > 0)
        if not (valid == expected_valid).all():
            fail("is_valid_for_sigma0_step3 does not match definition", failures)
        sigma0_columns = [
            col
            for col in df.columns
            if "sigma0" in str(col).casefold()
            and str(col) not in {"is_valid_for_sigma0_step3"}
        ]
        if sigma0_columns:
            fail("sigma0-like columns exist in output", failures)

    if not ge1.empty and not as_bool(ge1["is_valid_for_sigma0_step3"]).all():
        fail("ge1 output contains rows not valid for sigma0 step3", failures)
    if not conservative.empty:
        if not as_bool(conservative["is_valid_for_sigma0_step3"]).all():
            fail("conservative-ge1 output contains rows not valid for sigma0 step3", failures)
        if not as_bool(conservative["is_conservative_main_analysis"]).all():
            fail("conservative-ge1 output contains non-conservative rows", failures)

    print(f"rows: {len(df)}")
    print(f"eta_status counts: {df['eta_status'].value_counts(dropna=False).to_dict() if 'eta_status' in df else {}}")
    print(f"eta >= 1 candidates: {len(ge1)}")
    print(f"conservative eta >= 1 candidates: {len(conservative)}")
    if not df.empty and "S_eta_abs_error_uV_per_K" in df:
        ok = df["eta_status"] == "ok"
        max_abs_error = pd.to_numeric(df.loc[ok, "S_eta_abs_error_uV_per_K"], errors="coerce").abs().max()
        print(f"max abs S_eta_abs_error_uV_per_K: {max_abs_error}")
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)
    print("step2 eta output checks passed")


if __name__ == "__main__":
    main()
