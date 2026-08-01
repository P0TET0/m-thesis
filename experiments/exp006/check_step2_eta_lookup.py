import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_LOOKUP = EXP_DIR / "data" / "processed" / "step2_eta_lookup_table.csv"

REQUIRED_COLUMNS = [
    "eta",
    "F0_eta",
    "F1_eta",
    "s_model",
    "S_abs_V_per_K",
    "S_abs_uV_per_K",
]

REFERENCE_VALUES_UV = {
    0.0: 204.5,
    1.0: 150.9,
    2.0: 112.4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step2A eta lookup table.")
    parser.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    return parser.parse_args()


def interpolate_s_uV(df: pd.DataFrame, eta_value: float) -> float:
    return float(np.interp(eta_value, df["eta"].to_numpy(), df["S_abs_uV_per_K"].to_numpy()))


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    if not args.lookup.exists():
        print(f"FAIL: lookup file not found: {args.lookup}")
        raise SystemExit(1)

    df = pd.read_csv(args.lookup)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        failures.append(f"missing required columns: {missing}")

    if not df.empty and not missing:
        eta = pd.to_numeric(df["eta"], errors="coerce").to_numpy()
        f0 = pd.to_numeric(df["F0_eta"], errors="coerce").to_numpy()
        f1 = pd.to_numeric(df["F1_eta"], errors="coerce").to_numpy()
        s_model = pd.to_numeric(df["s_model"], errors="coerce").to_numpy()
        s_v = pd.to_numeric(df["S_abs_V_per_K"], errors="coerce").to_numpy()
        s_uV = pd.to_numeric(df["S_abs_uV_per_K"], errors="coerce").to_numpy()
        if not np.all(np.diff(eta) > 0):
            failures.append("eta is not strictly increasing")
        if not (np.isfinite(f0).all() and (f0 > 0).all()):
            failures.append("F0_eta contains non-finite or non-positive values")
        if not (np.isfinite(f1).all() and (f1 >= 0).all()):
            failures.append("F1_eta contains non-finite or negative values")
        if not (np.isfinite(s_model).all() and (s_model > 0).all()):
            failures.append("s_model contains non-finite or non-positive values")
        if not (np.isfinite(s_uV).all() and (s_uV > 0).all()):
            failures.append("S_abs_uV_per_K contains non-finite or non-positive values")
        if not np.allclose(s_uV, s_v * 1e6, rtol=1e-12, atol=1e-9):
            failures.append("S_abs_uV_per_K is not S_abs_V_per_K * 1e6")
        if not np.all(np.diff(s_uV) <= 1e-9):
            failures.append("S_abs_uV_per_K is not monotonically decreasing with eta")
        eta_min = float(np.nanmin(eta))
        eta_max = float(np.nanmax(eta))
        for eta_ref, expected in REFERENCE_VALUES_UV.items():
            if eta_min <= eta_ref <= eta_max:
                actual = interpolate_s_uV(df, eta_ref)
                if abs(actual - expected) > 1.0:
                    failures.append(
                        f"S_abs_uV_per_K at eta={eta_ref:g} is {actual:.6g}, expected about {expected:.6g}"
                    )
            else:
                failures.append(f"eta={eta_ref:g} is outside the lookup grid")

    print(f"rows: {len(df)}")
    if not df.empty and "eta" in df.columns:
        print(f"eta range: {df['eta'].min()} to {df['eta'].max()}")
    if not df.empty and "S_abs_uV_per_K" in df.columns:
        print(f"S_abs_uV_per_K range: {df['S_abs_uV_per_K'].min()} to {df['S_abs_uV_per_K'].max()}")
        print(f"S_abs_uV_per_K at eta=1: {interpolate_s_uV(df, 1.0):.6g}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step2 eta lookup checks passed")


if __name__ == "__main__":
    main()
