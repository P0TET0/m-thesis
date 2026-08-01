import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_PROCESSED = EXP_DIR / "data" / "processed"

REQUIRED_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_label",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "T_S_K",
    "T_sigma_K",
    "T_delta_K",
    "S_V_per_K",
    "S_uV_per_K",
    "S_sign",
    "sigma_S_per_m",
    "rho_ohm_m",
    "sigma_source",
    "match_method",
    "source_file_S",
    "source_file_sigma",
    "source_property_label_S",
    "source_property_label_sigma",
    "source_unit_S",
    "source_unit_sigma_or_rho",
    "source_curve_id_S",
    "source_curve_id_sigma",
    "source_notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check step0 thermoelectric analysis outputs.")
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--match-tol-k", type=float, default=1.0)
    return parser.parse_args()


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> None:
    args = parse_args()
    processed = args.processed
    table_path = processed / "step0_te_analysis_base.csv"
    reject_path = processed / "step0_rejected_rows.csv"
    duplicate_path = processed / "step0_duplicate_candidates.csv"
    schema_path = processed / "step0_schema_detected.json"

    failures: list[str] = []
    for path in [table_path, reject_path, duplicate_path, schema_path]:
        if not path.exists():
            fail(f"missing required output: {path}", failures)
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)

    df = pd.read_csv(table_path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        fail(f"missing required columns: {missing}", failures)
    if not df.empty:
        numeric_checks = {
            "T_K": lambda s: np.isfinite(s).all() and (s > 0).all(),
            "T_S_K": lambda s: np.isfinite(s).all() and (s > 0).all(),
            "T_sigma_K": lambda s: np.isfinite(s).all() and (s > 0).all(),
            "S_V_per_K": lambda s: np.isfinite(s).all(),
            "S_uV_per_K": lambda s: np.isfinite(s).all(),
            "sigma_S_per_m": lambda s: np.isfinite(s).all() and (s > 0).all(),
        }
        for column, check in numeric_checks.items():
            values = pd.to_numeric(df[column], errors="coerce")
            if not check(values):
                fail(f"invalid numeric column: {column}", failures)
        rho = pd.to_numeric(df["rho_ohm_m"], errors="coerce").dropna()
        if not rho.empty and not (rho > 0).all():
            fail("rho_ohm_m contains non-positive values", failures)
        if not np.allclose(
            pd.to_numeric(df["S_uV_per_K"], errors="coerce"),
            pd.to_numeric(df["S_V_per_K"], errors="coerce") * 1e6,
        ):
            fail("S_uV_per_K is not S_V_per_K * 1e6", failures)
        non_interp = df[df["match_method"] != "interpolated"]
        if not non_interp.empty and not (
            pd.to_numeric(non_interp["T_delta_K"], errors="coerce") <= args.match_tol_k + 1e-12
        ).all():
            fail("non-interpolated rows exceed match tolerance", failures)
        converted = df[df["sigma_source"] == "resistivity_converted"]
        if not converted.empty and not np.allclose(
            pd.to_numeric(converted["sigma_S_per_m"], errors="coerce"),
            1.0 / pd.to_numeric(converted["rho_ohm_m"], errors="coerce"),
            rtol=1e-10,
            atol=1e-12,
        ):
            fail("resistivity_converted rows do not satisfy sigma = 1 / rho", failures)
        allowed_sources = {"conductivity_direct", "resistivity_converted"}
        unknown_sources = sorted(set(df["sigma_source"].dropna()) - allowed_sources)
        if unknown_sources:
            fail(f"unknown sigma_source values: {unknown_sources}", failures)
        allowed_methods = {"exact", "nearest", "interpolated"}
        unknown_methods = sorted(set(df["match_method"].dropna()) - allowed_methods)
        if unknown_methods:
            fail(f"unknown match_method values: {unknown_methods}", failures)
        if not df["row_id"].is_unique:
            fail("row_id is not unique", failures)
        if df[["S_V_per_K", "sigma_S_per_m"]].isna().any().any():
            fail("rows with missing S or sigma exist", failures)

    print(f"rows: {len(df)}")
    print(f"reject rows: {len(pd.read_csv(reject_path))}")
    print(f"duplicate candidates: {len(pd.read_csv(duplicate_path))}")
    if failures:
        for item in failures:
            print(f"FAIL: {item}")
        raise SystemExit(1)
    print("step0 output checks passed")


if __name__ == "__main__":
    main()
