import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

CURVE_KEY_COLUMNS = [
    "source_subset",
    "group_scheme",
    "material_group_key",
    "carrier_type",
    "curve_method",
    "T_bin_center_K",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step4 sigma0 reference curve outputs.")
    parser.add_argument("--curve", type=Path, default=PROCESSED_DIR / "step4_sigma0_reference_curve_bins.csv")
    parser.add_argument("--reliable", type=Path, default=PROCESSED_DIR / "step4_sigma0_reference_curve_reliable.csv")
    parser.add_argument("--default", type=Path, default=PROCESSED_DIR / "step4_sigma0_reference_curve_default.csv")
    parser.add_argument("--binned-rows", type=Path, default=PROCESSED_DIR / "step4_sigma0_binned_input_rows.csv")
    parser.add_argument("--dropped", type=Path, default=PROCESSED_DIR / "step4_sigma0_dropped_rows.csv")
    parser.add_argument("--min-rows-per-bin", type=int, default=3)
    parser.add_argument("--min-samples-per-bin", type=int, default=3)
    parser.add_argument("--min-papers-per-bin", type=int, default=1)
    return parser.parse_args()


def reliability_level(row: pd.Series) -> str:
    if not bool(row["is_reference_bin_candidate"]):
        return "insufficient"
    if int(row["sample_count"]) >= 10 and int(row["paper_count"]) >= 3:
        return "high"
    if int(row["sample_count"]) >= 5 and int(row["paper_count"]) >= 2:
        return "medium"
    return "low"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.curve, args.reliable, args.default, args.binned_rows, args.dropped]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    curves = pd.read_csv(args.curve, low_memory=False)
    reliable = pd.read_csv(args.reliable, low_memory=False)
    default = pd.read_csv(args.default, low_memory=False)
    binned = pd.read_csv(args.binned_rows, low_memory=False)
    dropped = pd.read_csv(args.dropped, low_memory=False)

    if curves.empty:
        failures.append("reference curve bins are empty")
    if not binned.empty:
        if not (np.isfinite(pd.to_numeric(binned["T_K"], errors="coerce")).all() and (pd.to_numeric(binned["T_K"], errors="coerce") > 0).all()):
            failures.append("binned rows contain invalid T_K")
        if not (np.isfinite(pd.to_numeric(binned["sigma0_S_per_m"], errors="coerce")).all() and (pd.to_numeric(binned["sigma0_S_per_m"], errors="coerce") > 0).all()):
            failures.append("binned rows contain invalid sigma0")
        if not np.isfinite(pd.to_numeric(binned["log10_sigma0_S_per_m"], errors="coerce")).all():
            failures.append("binned rows contain invalid log10_sigma0")
        t = pd.to_numeric(binned["T_K"], errors="coerce")
        left = pd.to_numeric(binned["T_bin_left_K"], errors="coerce")
        right = pd.to_numeric(binned["T_bin_right_K"], errors="coerce")
        center = pd.to_numeric(binned["T_bin_center_K"], errors="coerce")
        if not ((left <= t) & (t < right)).all():
            failures.append("T_K is outside assigned bins")
        if not np.allclose(center, (left + right) / 2.0):
            failures.append("T_bin_center_K is inconsistent")
        if not set(binned["carrier_type"].dropna()).issubset({"p", "n"}):
            failures.append("binned rows contain carrier_type outside p/n")
    if not curves.empty:
        if curves.duplicated(CURVE_KEY_COLUMNS).any():
            failures.append("curve key duplicates exist")
        if not np.allclose(curves["sigma0_ref_S_per_m"], 10.0 ** curves["log10_sigma0_ref_S_per_m"]):
            failures.append("sigma0_ref_S_per_m is inconsistent")
        if not ((curves["row_count"] > 0) & (curves["sample_count"] > 0) & (curves["paper_count"] > 0)).all():
            failures.append("row/sample/paper counts must be positive")
        expected_candidate = (
            (curves["row_count"] >= args.min_rows_per_bin)
            & (curves["sample_count"] >= args.min_samples_per_bin)
            & (curves["paper_count"] >= args.min_papers_per_bin)
        )
        if not (as_bool(curves["is_reference_bin_candidate"]) == expected_candidate).all():
            failures.append("is_reference_bin_candidate rule mismatch")
        expected_reliability = curves.apply(reliability_level, axis=1)
        if not (curves["reliability_level"] == expected_reliability).all():
            failures.append("reliability_level rule mismatch")
        expected_default = (
            curves["source_subset"].eq("conservative_valid")
            & curves["curve_method"].eq("sample_median")
            & as_bool(curves["is_reference_bin_candidate"])
        )
        if not (as_bool(curves["recommended_default"]) == expected_default).all():
            failures.append("recommended_default rule mismatch")
        q_ok = (curves["log10_sigma0_q25"] <= curves["log10_sigma0_ref_S_per_m"]) & (
            curves["log10_sigma0_ref_S_per_m"] <= curves["log10_sigma0_q75"]
        )
        if not q_ok.all():
            print(f"WARNING: {(~q_ok).sum()} q25/ref/q75 ordering issues")
    if not reliable.empty and not as_bool(reliable["is_reference_bin_candidate"]).all():
        failures.append("reliable file contains non-reference bins")
    if not default.empty and not as_bool(default["recommended_default"]).all():
        failures.append("default file contains non-default bins")

    print(f"binned rows: {len(binned)}")
    print(f"dropped rows: {len(dropped)}")
    print(f"curve bins: {len(curves)}")
    print(f"reliable bins: {len(reliable)}")
    print(f"recommended_default bins: {len(default)}")
    if not curves.empty:
        print(f"group_scheme counts: {curves['group_scheme'].value_counts().to_dict()}")
        print(f"curve_method counts: {curves['curve_method'].value_counts().to_dict()}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step4 sigma0 reference curve checks passed")


if __name__ == "__main__":
    main()
