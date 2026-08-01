import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = EXP_DIR / "data" / "processed" / "step1_te_carrier_classified.csv"

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "S_V_per_K",
    "S_uV_per_K",
    "sigma_S_per_m",
    "match_method",
    "sigma_source",
    "S_abs_uV_per_K",
    "carrier_type",
    "carrier_type_rule",
    "is_usable_for_eta",
    "sample_group_id",
    "n_points_sample",
    "n_p_points_sample",
    "n_n_points_sample",
    "n_unknown_points_sample",
    "sample_has_sign_change",
    "sample_carrier_behavior",
    "is_conservative_main_analysis",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step1 carrier classification outputs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--zero-threshold-uV", type=float, default=1.0)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin(["true", "1", "yes"])


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_dir = input_path.parent
    failures: list[str] = []

    required_files = [
        input_path,
        output_dir / "step1_eta_input_candidates.csv",
        output_dir / "step1_conservative_main_candidates.csv",
        output_dir / "step1_sample_sign_summary.csv",
        output_dir / "step1_carrier_counts_by_material_family.csv",
    ]
    for path in required_files:
        if not path.exists():
            fail(f"missing required output: {path}", failures)
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)

    df = pd.read_csv(input_path, low_memory=False)
    eta = pd.read_csv(output_dir / "step1_eta_input_candidates.csv", low_memory=False)
    conservative = pd.read_csv(output_dir / "step1_conservative_main_candidates.csv", low_memory=False)
    summary = pd.read_csv(output_dir / "step1_sample_sign_summary.csv", low_memory=False)

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        fail(f"missing required columns: {missing}", failures)

    if not df.empty:
        if not df["row_id"].is_unique:
            fail("row_id is not unique", failures)
        s_v = pd.to_numeric(df["S_V_per_K"], errors="coerce")
        s_uv = pd.to_numeric(df["S_uV_per_K"], errors="coerce")
        s_abs = pd.to_numeric(df["S_abs_uV_per_K"], errors="coerce")
        t_k = pd.to_numeric(df["T_K"], errors="coerce")
        sigma = pd.to_numeric(df["sigma_S_per_m"], errors="coerce")
        if not np.isfinite(s_v).all():
            fail("S_V_per_K contains non-finite values", failures)
        if not np.isfinite(s_uv).all():
            fail("S_uV_per_K contains non-finite values", failures)
        if not np.allclose(s_uv, s_v * 1e6, rtol=1e-6, atol=1e-9):
            fail("S_uV_per_K is not consistent with S_V_per_K * 1e6", failures)
        if not np.allclose(s_abs, s_uv.abs(), rtol=1e-12, atol=1e-12):
            fail("S_abs_uV_per_K is not abs(S_uV_per_K)", failures)
        allowed = {"p", "n", "unknown_near_zero"}
        if not set(df["carrier_type"].dropna()).issubset(allowed):
            fail("carrier_type contains unexpected values", failures)
        expected_carrier = np.select(
            [s_uv > args.zero_threshold_uV, s_uv < -args.zero_threshold_uV, s_uv.abs() <= args.zero_threshold_uV],
            ["p", "n", "unknown_near_zero"],
            default="unknown_near_zero",
        )
        if not (df["carrier_type"].to_numpy() == expected_carrier).all():
            fail("carrier_type does not match threshold rule", failures)
        usable = as_bool(df["is_usable_for_eta"])
        if not (usable == df["carrier_type"].isin(["p", "n"])).all():
            fail("is_usable_for_eta does not match carrier_type rule", failures)
        sign_change = as_bool(df["sample_has_sign_change"])
        conservative_flag = as_bool(df["is_conservative_main_analysis"])
        expected_conservative = df["carrier_type"].isin(["p", "n"]) & ~sign_change
        if not (conservative_flag == expected_conservative).all():
            fail("is_conservative_main_analysis does not match definition", failures)
        if not (np.isfinite(t_k).all() and (t_k > 0).all()):
            fail("T_K contains non-finite or non-positive values", failures)
        if not (np.isfinite(sigma).all() and (sigma > 0).all()):
            fail("sigma_S_per_m contains non-finite or non-positive values", failures)

    if not eta.empty and not set(eta["carrier_type"].dropna()).issubset({"p", "n"}):
        fail("eta candidates contain non p/n rows", failures)
    if not conservative.empty and as_bool(conservative["sample_has_sign_change"]).any():
        fail("conservative candidates contain sign-changing sample rows", failures)
    if not summary.empty:
        expected = summary["n_p_points_sample"].astype(int).gt(0) & summary["n_n_points_sample"].astype(int).gt(0)
        actual = as_bool(summary["sample_has_sign_change"])
        if not (expected == actual).all():
            fail("sample summary sign-change flag is inconsistent", failures)

    print(f"rows: {len(df)}")
    print(f"eta candidates: {len(eta)}")
    print(f"conservative candidates: {len(conservative)}")
    print(f"samples: {len(summary)}")
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)
    print("step1 output checks passed")


if __name__ == "__main__":
    main()
