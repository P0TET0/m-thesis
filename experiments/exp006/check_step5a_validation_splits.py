import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

REQUIRED_ROW_COLUMNS = [
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "sample_holdout_split",
    "paper_holdout_split",
    "sample_cv_fold",
    "paper_cv_fold",
    "is_sample_holdout_train",
    "is_sample_holdout_test",
    "is_paper_holdout_train",
    "is_paper_holdout_test",
    "eval_eligible_all_valid",
    "eval_eligible_conservative_valid",
    "T_K",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "carrier_type",
    "sigma_S_per_m",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
]

REQUIRED_COVERAGE_COLUMNS = [
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "train_rows",
    "test_rows",
    "test_rows_with_reference",
    "test_rows_without_reference",
    "coverage_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step5A validation split outputs.")
    parser.add_argument("--rows", type=Path, default=PROCESSED_DIR / "step5a_validation_rows_with_splits.csv")
    parser.add_argument(
        "--sample-assignments",
        type=Path,
        default=PROCESSED_DIR / "step5a_sample_group_split_assignments.csv",
    )
    parser.add_argument(
        "--paper-assignments",
        type=Path,
        default=PROCESSED_DIR / "step5a_paper_group_split_assignments.csv",
    )
    parser.add_argument("--summary", type=Path, default=PROCESSED_DIR / "step5a_split_summary.csv")
    parser.add_argument("--coverage", type=Path, default=PROCESSED_DIR / "step5a_holdout_coverage_preflight.csv")
    parser.add_argument("--dropped", type=Path, default=PROCESSED_DIR / "step5a_dropped_rows.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--require-full-run", action="store_true")
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "na", "n/a"}:
        return ""
    return text


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def check_group_single_value(df: pd.DataFrame, group_col: str, value_col: str) -> bool:
    if df.empty:
        return False
    return df.groupby(group_col)[value_col].nunique(dropna=False).max() == 1


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [
        args.rows,
        args.sample_assignments,
        args.paper_assignments,
        args.summary,
        args.coverage,
        args.dropped,
    ]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    rows = pd.read_csv(args.rows, low_memory=False)
    sample_assignments = pd.read_csv(args.sample_assignments, low_memory=False)
    paper_assignments = pd.read_csv(args.paper_assignments, low_memory=False)
    summary = pd.read_csv(args.summary, low_memory=False)
    coverage = pd.read_csv(args.coverage, low_memory=False)
    dropped = pd.read_csv(args.dropped, low_memory=False)

    require_columns(rows, REQUIRED_ROW_COLUMNS, "validation rows", failures)
    require_columns(coverage, REQUIRED_COVERAGE_COLUMNS, "coverage preflight", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if rows.empty:
        failures.append("validation rows are empty")
    if not rows["row_id"].is_unique:
        failures.append("row_id is not unique")
    if rows["validation_sample_group_id"].map(clean_text).eq("").any():
        failures.append("validation_sample_group_id contains missing values")
    if rows["validation_paper_group_id"].map(clean_text).eq("").any():
        failures.append("validation_paper_group_id contains missing values")

    if not set(rows["sample_holdout_split"].dropna()).issubset({"train", "test"}):
        failures.append("sample_holdout_split contains values outside train/test")
    if not set(rows["paper_holdout_split"].dropna()).issubset({"train", "test"}):
        failures.append("paper_holdout_split contains values outside train/test")
    if not check_group_single_value(rows, "validation_sample_group_id", "sample_holdout_split"):
        failures.append("sample holdout leakage: a sample group appears in multiple splits")
    if not check_group_single_value(rows, "validation_paper_group_id", "paper_holdout_split"):
        failures.append("paper holdout leakage: a paper group appears in multiple splits")

    sample_fold = numeric(rows, "sample_cv_fold")
    paper_fold = numeric(rows, "paper_cv_fold")
    if not sample_fold.between(0, args.n_folds - 1).all():
        failures.append("sample_cv_fold is outside expected range")
    if not paper_fold.between(0, args.n_folds - 1).all():
        failures.append("paper_cv_fold is outside expected range")
    if not check_group_single_value(rows, "validation_sample_group_id", "sample_cv_fold"):
        failures.append("sample group has multiple CV folds")
    if not check_group_single_value(rows, "validation_paper_group_id", "paper_cv_fold"):
        failures.append("paper group has multiple CV folds")

    t = numeric(rows, "T_K")
    left = numeric(rows, "T_bin_left_K")
    right = numeric(rows, "T_bin_right_K")
    if not ((left <= t) & (t < right)).all():
        failures.append("T_K is outside assigned T bin")
    if not set(rows["carrier_type"].dropna()).issubset({"p", "n"}):
        failures.append("carrier_type contains values outside p/n")

    for column in ["sigma_S_per_m", "F0_eta", "sigma0_S_per_m"]:
        values = numeric(rows, column)
        if not (np.isfinite(values).all() and (values > 0).all()):
            failures.append(f"{column} contains non-finite or non-positive values")
    log_sigma0 = numeric(rows, "log10_sigma0_S_per_m")
    if not np.isfinite(log_sigma0).all():
        failures.append("log10_sigma0_S_per_m contains non-finite values")

    if (as_bool(rows["is_sample_holdout_train"]) != rows["sample_holdout_split"].eq("train")).any():
        failures.append("is_sample_holdout_train is inconsistent")
    if (as_bool(rows["is_sample_holdout_test"]) != rows["sample_holdout_split"].eq("test")).any():
        failures.append("is_sample_holdout_test is inconsistent")
    if (as_bool(rows["is_paper_holdout_train"]) != rows["paper_holdout_split"].eq("train")).any():
        failures.append("is_paper_holdout_train is inconsistent")
    if (as_bool(rows["is_paper_holdout_test"]) != rows["paper_holdout_split"].eq("test")).any():
        failures.append("is_paper_holdout_test is inconsistent")
    if not as_bool(rows["eval_eligible_all_valid"]).all():
        failures.append("eval_eligible_all_valid is not all True")

    coverage_fraction = numeric(coverage, "coverage_fraction").dropna()
    if not coverage_fraction.between(0.0, 1.0).all():
        failures.append("coverage_fraction is outside 0..1")
    with_ref = numeric(coverage, "test_rows_with_reference")
    without_ref = numeric(coverage, "test_rows_without_reference")
    test_rows = numeric(coverage, "test_rows")
    if not ((with_ref + without_ref) == test_rows).all():
        failures.append("coverage test row counts are inconsistent")
    if coverage.empty:
        failures.append("coverage preflight is empty")

    if args.require_full_run:
        if set(rows["sample_holdout_split"].dropna()) != {"train", "test"}:
            failures.append("full run sample holdout does not contain both train and test")
        if set(rows["paper_holdout_split"].dropna()) != {"train", "test"}:
            failures.append("full run paper holdout does not contain both train and test")

    print(f"validation rows: {len(rows)}")
    print(f"dropped rows: {len(dropped)}")
    print(f"sample groups: {rows['validation_sample_group_id'].nunique()}")
    print(f"paper groups: {rows['validation_paper_group_id'].nunique()}")
    print(f"sample assignment rows: {len(sample_assignments)}")
    print(f"paper assignment rows: {len(paper_assignments)}")
    print(f"split summary rows: {len(summary)}")
    print(f"coverage rows: {len(coverage)}")
    print(f"sample holdout rows: {rows['sample_holdout_split'].value_counts().to_dict()}")
    print(f"paper holdout rows: {rows['paper_holdout_split'].value_counts().to_dict()}")
    default_coverage = coverage[
        coverage["split_scheme"].eq("sample_holdout")
        & coverage["reference_source_subset"].eq("conservative_valid")
        & coverage["eval_target_subset"].eq("all_valid")
        & coverage["group_scheme"].eq("material_family")
        & coverage["curve_method"].eq("sample_median")
    ]
    if not default_coverage.empty:
        print(f"default coverage_fraction: {default_coverage['coverage_fraction'].iloc[0]}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step5a validation split checks passed")


if __name__ == "__main__":
    main()
