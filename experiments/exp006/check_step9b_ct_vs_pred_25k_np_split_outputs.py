import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
STEP9A_DIR = EXP_DIR / "data" / "processed" / "step9a_25k_bin_broad_family"
DEFAULT_PREDICTIONS = STEP9A_DIR / "step5b_test_predictions_valid.parquet"
DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid"
    "__material_family__sample_median"
)

SUMMARY_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "prediction_points",
    "old_ct_points",
    "T_pred_min_K",
    "T_pred_max_K",
    "T_old_ct_min_K",
    "T_old_ct_max_K",
    "sigma_pred_median_S_per_m",
    "old_C_T_median_S_per_m",
    "median_log10_pred_over_oldCT_nearest",
    "warning",
]
FIGURE_INDEX_COLUMNS = [
    "figure_id",
    "material_group_key",
    "carrier_type",
    "figure_type",
    "figure_path_png",
    "figure_path_pdf",
    "title",
    "n_prediction_points",
    "n_old_ct_points",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step9B C(T) vs sigma_pred outputs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure-index", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    return parser.parse_args()


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & values.gt(0)


def suffix_from_summary(path: Path) -> str:
    match = re.fullmatch(r"step9b_summary_by_group_carrier(.*)\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected summary filename: {path.name}")
    return match.group(1)


def current_step9a_manifest() -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(STEP9A_DIR).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in STEP9A_DIR.rglob("*")
        if path.is_file()
    }


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.summary, args.figure_index, args.report, args.predictions]:
        if not path.exists():
            failures.append(f"missing required path: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    suffix = suffix_from_summary(args.summary)
    output = args.summary.parent
    paths = {
        "prediction_rows": output / f"step9b_prediction_rows_used{suffix}.csv",
        "old_curves": output / f"step9b_old_ct_curves_no_pn{suffix}.csv",
        "nearest": output / f"step9b_nearest_comparison_table{suffix}.csv",
        "mapping": output / f"step9b_material_mapping{suffix}.csv",
        "unmatched": output / f"step9b_unmatched_old_material_labels{suffix}.csv",
        "protection": output / f"step9b_step9a_protection_manifest{suffix}.csv",
    }
    for label, path in paths.items():
        if not path.exists():
            failures.append(f"missing {label}: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    summary = pd.read_csv(args.summary, low_memory=False)
    figure_index = pd.read_csv(args.figure_index, low_memory=False)
    used = pd.read_csv(paths["prediction_rows"], low_memory=False)
    old = pd.read_csv(paths["old_curves"], low_memory=False)
    nearest = pd.read_csv(paths["nearest"], low_memory=False)
    protection = pd.read_csv(paths["protection"], low_memory=False)

    if list(summary.columns) != SUMMARY_COLUMNS:
        failures.append("summary columns do not match specification")
    if list(figure_index.columns) != FIGURE_INDEX_COLUMNS:
        failures.append("figure index columns do not match specification")
    expected_combinations = {
        (group, carrier)
        for group in summary["material_group_key"].drop_duplicates()
        for carrier in ["p", "n"]
    }
    found_combinations = set(zip(summary["material_group_key"], summary["carrier_type"]))
    if found_combinations != expected_combinations:
        failures.append("summary does not contain p and n rows for every target group")
    if len(figure_index) == 0:
        failures.append("no figures were indexed")
    if set(figure_index["carrier_type"]) - {"p", "n"}:
        failures.append("figure index contains a carrier other than p/n")
    indexed_combinations = set(zip(figure_index["material_group_key"], figure_index["carrier_type"]))
    if indexed_combinations != expected_combinations:
        failures.append("separate p and n main figures were not created for every target group")
    for _, row in figure_index.iterrows():
        png = Path(row["figure_path_png"])
        pdf = Path(row["figure_path_pdf"])
        if not png.exists() or png.stat().st_size == 0:
            failures.append(f"missing/empty PNG: {png}")
        if not pdf.exists() or pdf.stat().st_size == 0:
            failures.append(f"missing/empty PDF: {pdf}")
        if "measured sigma and sigma0_ref are not plotted" not in str(row["description"]):
            failures.append(f"figure description does not exclude measured sigma/sigma0_ref: {row['figure_id']}")
        if "no_pn" not in str(row["figure_type"]):
            failures.append(f"figure is not marked as using no-p/n old C(T): {row['figure_id']}")

    if used.empty:
        failures.append("prediction rows used are empty")
    if not finite_positive(used["sigma_pred_S_per_m"]).all():
        failures.append("used sigma_pred contains non-finite or non-positive values")
    if not (np.isfinite(pd.to_numeric(used["T_K"], errors="coerce")) & (pd.to_numeric(used["T_K"], errors="coerce") > 0)).all():
        failures.append("used prediction temperature contains invalid values")
    if set(used["carrier_type"]) - {"p", "n"}:
        failures.append("used prediction rows contain invalid carrier_type")
    if used["material_group_key"].isna().any() or used["material_group_key"].astype(str).str.strip().eq("").any():
        failures.append("used prediction rows contain missing material_group_key")

    source = pd.read_parquet(
        args.predictions,
        columns=["config_id", "prediction_status", "row_id", "sigma_pred_S_per_m"],
    )
    if not source["config_id"].astype(str).eq(args.config_id).any():
        failures.append("requested config_id is absent from prediction source")
    source = source[
        source["config_id"].astype(str).eq(args.config_id)
        & source["row_id"].astype(str).isin(used["row_id"].astype(str))
    ][["row_id", "prediction_status", "sigma_pred_S_per_m"]].drop_duplicates("row_id")
    joined = used[["row_id", "sigma_pred_S_per_m"]].merge(
        source,
        on="row_id",
        how="left",
        suffixes=("_used", "_source"),
    )
    if joined["prediction_status"].isna().any() or not joined["prediction_status"].eq("ok").all():
        failures.append("used rows are not all existing prediction_status=ok source rows")
    if not np.allclose(
        pd.to_numeric(joined["sigma_pred_S_per_m_used"], errors="coerce"),
        pd.to_numeric(joined["sigma_pred_S_per_m_source"], errors="coerce"),
        rtol=1e-12,
        atol=0.0,
    ):
        failures.append("used sigma_pred differs from Step9A source; new values may have been calculated")

    if old.empty or not finite_positive(old["old_C_T_S_per_m"]).all():
        failures.append("old C(T) curves are empty or invalid")
    if "carrier_type" in old.columns or "n_or_p" in old.columns:
        failures.append("old C(T) curve output retains a p/n split")
    if old.duplicated(["material_group_key_mapped", "T_K"]).any():
        failures.append("old C(T) has duplicate material/temperature points after aggregation")
    if not old["old_ct_parse_status"].eq("ok_pn_aggregated").all():
        failures.append("old C(T) parse status does not confirm p/n aggregation")
    if not old["source_file"].map(lambda value: Path(str(value)).exists()).all():
        failures.append("old C(T) source file does not exist")

    if not nearest.empty:
        expected_ratio = (
            pd.to_numeric(nearest["log10_sigma_pred_S_per_m"], errors="coerce")
            - pd.to_numeric(nearest["log10_old_C_T_S_per_m"], errors="coerce")
        )
        if not np.allclose(
            pd.to_numeric(nearest["log10_pred_over_oldCT"], errors="coerce"),
            expected_ratio,
            rtol=1e-12,
            atol=1e-12,
        ):
            failures.append("nearest comparison log10 ratio is inconsistent")
        if (pd.to_numeric(nearest["T_delta_K"], errors="coerce") < 0).any():
            failures.append("nearest comparison contains negative T_delta_K")

    if protection.empty or not protection["unchanged"].astype(str).str.casefold().isin({"true", "1"}).all():
        failures.append("Step9A protection manifest reports a changed file")
    else:
        current = current_step9a_manifest()
        for _, row in protection.iterrows():
            key = str(row["relative_path"])
            expected = (int(row["size_after"]), int(row["mtime_ns_after"]))
            if current.get(key) != expected:
                failures.append(f"protected Step9A file changed after Step9B: {key}")
                break

    report_text = args.report.read_text(encoding="utf-8")
    required_phrases = [
        str(args.config_id),
        "p/n-unsplit",
        "Measured sigma is not included",
        "sigma0_ref is not included",
        "No new sigma_pred was calculated",
        "Step4 full-data reference curves were not used",
        "Starrydata2 raw data was not read",
    ]
    for phrase in required_phrases:
        if phrase not in report_text:
            failures.append(f"report missing required statement: {phrase}")

    png_count = sum(Path(path).exists() for path in figure_index["figure_path_png"])
    pdf_count = sum(Path(path).exists() for path in figure_index["figure_path_pdf"])
    print(f"summary rows: {len(summary)}")
    print(f"target material groups: {summary['material_group_key'].nunique()}")
    print(f"prediction rows used: {len(used)}")
    print(f"old C(T) points: {len(old)}")
    print(f"nearest comparison rows: {len(nearest)}")
    print(f"PNG figures: {png_count}")
    print(f"PDF figures: {pdf_count}")
    if not nearest.empty:
        print(f"median log10(sigma_pred / old C(T)): {nearest['log10_pred_over_oldCT'].median()}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step9b C(T) vs sigma_pred checks passed")


if __name__ == "__main__":
    main()
