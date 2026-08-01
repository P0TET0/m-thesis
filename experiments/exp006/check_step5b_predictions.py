import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"

PREDICTION_STATUSES = {
    "ok",
    "missing_reference_bin",
    "unreliable_reference_bin",
    "invalid_sigma0_ref",
    "invalid_F0_eta",
}

REQUIRED_PREDICTION_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "prediction_status",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key_for_prediction",
    "carrier_type",
    "T_bin_center_K",
    "F0_eta",
    "sigma_S_per_m",
    "sigma0_ref_S_per_m",
    "is_reference_bin_candidate",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
]

REQUIRED_REFERENCE_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "material_group_key",
    "carrier_type",
    "T_bin_center_K",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "is_reference_bin_candidate",
    "reliability_level",
]

REQUIRED_COVERAGE_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "test_rows",
    "prediction_ok_rows",
    "prediction_unavailable_rows",
    "coverage_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step5B prediction outputs.")
    parser.add_argument("--predictions", type=Path, default=PROCESSED_DIR / "step5b_test_predictions.csv")
    parser.add_argument("--valid", type=Path, default=PROCESSED_DIR / "step5b_test_predictions_valid.csv")
    parser.add_argument("--coverage", type=Path, default=PROCESSED_DIR / "step5b_prediction_coverage_by_config.csv")
    parser.add_argument("--reference", type=Path, default=PROCESSED_DIR / "step5b_train_reference_curve_bins.csv")
    parser.add_argument("--dropped", type=Path, default=PROCESSED_DIR / "step5b_dropped_rows.csv")
    parser.add_argument(
        "--unavailable",
        type=Path,
        default=PROCESSED_DIR / "step5b_test_predictions_unavailable.csv",
    )
    parser.add_argument(
        "--default",
        type=Path,
        default=PROCESSED_DIR / "step5b_test_predictions_default.csv",
    )
    parser.add_argument(
        "--global-default",
        type=Path,
        default=PROCESSED_DIR / "step5b_test_predictions_global_default.csv",
    )
    parser.add_argument("--require-full-run", action="store_true")
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def default_config_id(group_scheme: str) -> str:
    return f"sample_holdout__ref_conservative_valid__eval_all_valid__{group_scheme}__sample_median"


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [
        args.predictions,
        args.valid,
        args.coverage,
        args.reference,
        args.dropped,
        args.unavailable,
        args.default,
        args.global_default,
    ]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    predictions = pd.read_csv(args.predictions, low_memory=False)
    valid = pd.read_csv(args.valid, low_memory=False)
    coverage = pd.read_csv(args.coverage, low_memory=False)
    reference = pd.read_csv(args.reference, low_memory=False)
    dropped = pd.read_csv(args.dropped, low_memory=False)
    unavailable = pd.read_csv(args.unavailable, low_memory=False)
    default = pd.read_csv(args.default, low_memory=False)
    global_default = pd.read_csv(args.global_default, low_memory=False)

    require_columns(predictions, REQUIRED_PREDICTION_COLUMNS, "predictions", failures)
    require_columns(reference, REQUIRED_REFERENCE_COLUMNS, "reference", failures)
    require_columns(coverage, REQUIRED_COVERAGE_COLUMNS, "coverage", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if predictions.empty:
        failures.append("predictions are empty")
    if reference.empty:
        failures.append("reference curve bins are empty")
    if coverage.empty:
        failures.append("coverage summary is empty")

    if coverage["config_id"].nunique() != 32:
        failures.append(f"expected 32 configs in coverage, got {coverage['config_id'].nunique()}")
    if not coverage["config_id"].is_unique:
        failures.append("coverage config_id is not unique")
    if not set(predictions["prediction_status"].dropna()).issubset(PREDICTION_STATUSES):
        failures.append("prediction_status contains unexpected values")
    if not valid.empty and not valid["prediction_status"].eq("ok").all():
        failures.append("valid file contains non-ok predictions")
    if not unavailable.empty and unavailable["prediction_status"].eq("ok").any():
        failures.append("unavailable file contains ok predictions")

    ok = predictions["prediction_status"].eq("ok")
    not_ok = ~ok
    for column in ["sigma0_ref_S_per_m", "F0_eta", "sigma_pred_S_per_m", "sigma_pred_over_exp"]:
        values = numeric(predictions.loc[ok], column)
        if not (np.isfinite(values).all() and (values > 0).all()):
            failures.append(f"ok rows contain invalid {column}")
    for column in ["log10_sigma_pred_S_per_m", "log10_sigma_pred_over_exp"]:
        values = numeric(predictions.loc[ok], column)
        if not np.isfinite(values).all():
            failures.append(f"ok rows contain invalid {column}")
    if not predictions.loc[not_ok, "sigma_pred_S_per_m"].isna().all():
        failures.append("non-ok rows have sigma_pred_S_per_m values")

    pred = numeric(predictions.loc[ok], "sigma_pred_S_per_m")
    sigma0_ref = numeric(predictions.loc[ok], "sigma0_ref_S_per_m")
    f0 = numeric(predictions.loc[ok], "F0_eta")
    if not np.allclose(pred, sigma0_ref * f0, rtol=1e-10, atol=0.0):
        failures.append("sigma_pred formula mismatch")
    sigma_exp = numeric(predictions.loc[ok], "sigma_S_per_m")
    log_ratio = numeric(predictions.loc[ok], "log10_sigma_pred_over_exp")
    if not np.allclose(log_ratio, np.log10(pred / sigma_exp), rtol=1e-10, atol=1e-12):
        failures.append("log10 prediction ratio formula mismatch")

    coverage_fraction = numeric(coverage, "coverage_fraction").dropna()
    if not coverage_fraction.between(0.0, 1.0).all():
        failures.append("coverage_fraction is outside 0..1")
    if not (
        numeric(coverage, "prediction_ok_rows") + numeric(coverage, "prediction_unavailable_rows")
        == numeric(coverage, "test_rows")
    ).all():
        failures.append("coverage row counts are inconsistent")

    if default.empty:
        failures.append("default prediction file is empty")
    elif default["config_id"].nunique() != 1 or default["config_id"].iloc[0] != default_config_id("material_family"):
        failures.append("default prediction file has the wrong config")
    if global_default.empty:
        failures.append("global default prediction file is empty")
    elif global_default["config_id"].nunique() != 1 or global_default["config_id"].iloc[0] != default_config_id("global"):
        failures.append("global default prediction file has the wrong config")

    if not np.allclose(
        numeric(reference, "sigma0_ref_S_per_m"),
        10.0 ** numeric(reference, "log10_sigma0_ref_S_per_m"),
        rtol=1e-10,
        atol=0.0,
    ):
        failures.append("reference sigma0_ref is inconsistent with log10_sigma0_ref")
    if not set(reference["reliability_level"].dropna()).issubset({"high", "medium", "low", "insufficient"}):
        failures.append("reference reliability_level contains unexpected values")

    if args.require_full_run:
        if int(ok.sum()) == 0:
            failures.append("full run has zero ok predictions")
        if int(default["prediction_status"].eq("ok").sum()) == 0:
            failures.append("full run default has zero ok predictions")
        if int(global_default["prediction_status"].eq("ok").sum()) == 0:
            failures.append("full run global default has zero ok predictions")

    print(f"prediction rows: {len(predictions)}")
    print(f"valid prediction rows: {len(valid)}")
    print(f"unavailable prediction rows: {len(unavailable)}")
    print(f"dropped rows: {len(dropped)}")
    print(f"reference bins: {len(reference)}")
    print(f"reliable reference bins: {int(as_bool(reference['is_reference_bin_candidate']).sum())}")
    print(f"coverage configs: {coverage['config_id'].nunique()}")
    print(f"prediction_status counts: {predictions['prediction_status'].value_counts().to_dict()}")
    print(f"coverage_fraction summary: min={coverage_fraction.min()}, median={coverage_fraction.median()}, max={coverage_fraction.max()}")
    print(f"default ok rows: {int(default['prediction_status'].eq('ok').sum())}")
    print(f"global default ok rows: {int(global_default['prediction_status'].eq('ok').sum())}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step5b prediction checks passed")


if __name__ == "__main__":
    main()
