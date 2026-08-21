"""Validate Step9B 25 K sigma0_ref(T) versus Step31 C(T) outputs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PARSE_SUMMARY_COLUMNS = [
    "step31_script_path",
    "step31_report_path",
    "detected_output_file",
    "detected_ct_column",
    "detected_temperature_column",
    "detected_material_column",
    "detected_carrier_column",
    "detection_status",
    "notes",
]
SIGMA_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "T_bin_center_K",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
]
CT_COLUMNS = [
    "source_file",
    "material_group_key",
    "carrier_type",
    "T_K",
    "step31_C_T_S_per_m",
    "log10_step31_C_T_S_per_m",
    "parse_status",
]
COMPARISON_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "T_K_25k",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "T_K_step31",
    "step31_C_T_S_per_m",
    "log10_step31_C_T_S_per_m",
    "T_delta_K",
    "log10_sigma0ref25k_over_step31ct",
    "match_method",
]
SUMMARY_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "sigma0ref25k_points",
    "step31_ct_points",
    "comparison_points",
    "T_min_K",
    "T_max_K",
    "sigma0ref25k_median_S_per_m",
    "step31_ct_median_S_per_m",
    "median_log10_sigma0ref25k_over_step31ct",
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
    "n_sigma0ref25k_points",
    "n_step31_ct_points",
    "description",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--figure-index", type=Path, required=True)
    parser.add_argument("--parse-summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def suffix_from_summary(path: Path) -> str:
    match = re.fullmatch(r"step9b_summary_by_group_carrier(.*)\.csv", path.name)
    if not match:
        raise ValueError(f"Unexpected summary filename: {path.name}")
    return match.group(1)


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & values.gt(0)


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    try:
        suffix = suffix_from_summary(args.summary)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    output_dir = args.summary.parent
    paths = {
        "summary": args.summary,
        "figure_index": args.figure_index,
        "parse_summary": args.parse_summary,
        "report": args.report,
        "sigma": output_dir / f"step9b_25k_sigma0ref_curves{suffix}.csv",
        "ct": output_dir / f"step9b_step31_ct_curves_normalized{suffix}.csv",
        "comparison": output_dir / f"step9b_curve_comparison_table{suffix}.csv",
        "mapping": output_dir / f"step9b_material_mapping{suffix}.csv",
        "unmatched": output_dir / f"step9b_unmatched_material_labels{suffix}.csv",
    }
    for label, path in paths.items():
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            failures.append(f"missing or empty {label}: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    summary = pd.read_csv(paths["summary"], keep_default_na=False)
    figures = pd.read_csv(paths["figure_index"], keep_default_na=False)
    parse_summary = pd.read_csv(paths["parse_summary"], keep_default_na=False)
    sigma = pd.read_csv(paths["sigma"], low_memory=False)
    ct = pd.read_csv(paths["ct"], low_memory=False)
    comparison = pd.read_csv(paths["comparison"], low_memory=False)
    mapping = pd.read_csv(paths["mapping"], keep_default_na=False)
    unmatched = pd.read_csv(paths["unmatched"], keep_default_na=False)

    expected_columns = [
        (summary, SUMMARY_COLUMNS, "summary"),
        (figures, FIGURE_INDEX_COLUMNS, "figure index"),
        (parse_summary, PARSE_SUMMARY_COLUMNS, "parse summary"),
        (sigma, SIGMA_COLUMNS, "sigma0_ref curves"),
        (ct, CT_COLUMNS, "normalized Step31 curves"),
        (comparison, COMPARISON_COLUMNS, "comparison table"),
    ]
    for frame, expected, label in expected_columns:
        if frame.columns.tolist() != expected:
            failures.append(f"{label} columns do not match the specification")

    if len(parse_summary) != 1 or not parse_summary["detection_status"].eq("ok").all():
        failures.append("Step31 parse summary does not contain one successful detection")
    else:
        row = parse_summary.iloc[0]
        for column in ["step31_script_path", "step31_report_path", "detected_output_file"]:
            source = Path(str(row[column]))
            if not source.exists() or source.stat().st_size == 0:
                failures.append(f"detected Step31 source is missing/empty: {source}")
        for column in [
            "detected_ct_column",
            "detected_temperature_column",
            "detected_material_column",
            "detected_carrier_column",
        ]:
            if not str(row[column]).strip():
                failures.append(f"Step31 detection left {column} blank")

    if sigma.empty:
        failures.append("Step9A 25 K sigma0_ref curves are empty")
    else:
        if not finite_positive(sigma["sigma0_ref_S_per_m"]).all():
            failures.append("sigma0_ref contains non-finite or non-positive values")
        temperatures = pd.to_numeric(sigma["T_bin_center_K"], errors="coerce")
        if not np.isfinite(temperatures).all() or not np.isclose(np.mod(temperatures, 25.0), 0.0).all():
            failures.append("sigma0_ref temperatures are not a finite 25 K grid")
        expected_log = np.log10(pd.to_numeric(sigma["sigma0_ref_S_per_m"], errors="coerce"))
        if not np.allclose(expected_log, pd.to_numeric(sigma["log10_sigma0_ref_S_per_m"], errors="coerce")):
            failures.append("sigma0_ref log column is inconsistent")
        if set(sigma["carrier_type"].astype(str)) - {"p", "n"}:
            failures.append("sigma0_ref contains a carrier other than p/n")

    valid_ct = ct[ct["parse_status"].astype(str).str.startswith("ok")]
    if valid_ct.empty or not finite_positive(valid_ct["step31_C_T_S_per_m"]).all():
        failures.append("Step31 C(T) valid curves are empty or invalid")
    if set(valid_ct["carrier_type"].astype(str)) - {"p", "n"}:
        failures.append("Step31 valid curves contain a carrier other than p/n")
    if valid_ct.duplicated(["material_group_key", "carrier_type", "T_K"]).any():
        failures.append("Step31 normalized curves contain duplicate group/carrier/temperature rows")

    if comparison.empty:
        failures.append("comparison table is empty")
    else:
        if not comparison["match_method"].eq("nearest_temperature").all():
            failures.append("comparison table uses a method other than nearest_temperature")
        expected_delta = (
            pd.to_numeric(comparison["T_K_25k"], errors="coerce")
            - pd.to_numeric(comparison["T_K_step31"], errors="coerce")
        ).abs()
        if not np.allclose(expected_delta, pd.to_numeric(comparison["T_delta_K"], errors="coerce")):
            failures.append("comparison T_delta_K is inconsistent")
        expected_ratio = (
            pd.to_numeric(comparison["log10_sigma0_ref_S_per_m"], errors="coerce")
            - pd.to_numeric(comparison["log10_step31_C_T_S_per_m"], errors="coerce")
        )
        if not np.allclose(
            expected_ratio,
            pd.to_numeric(comparison["log10_sigma0ref25k_over_step31ct"], errors="coerce"),
        ):
            failures.append("comparison log10 ratio is inconsistent")

    if summary.empty:
        failures.append("summary is empty")
    else:
        if set(summary["carrier_type"].astype(str)) - {"p", "n"}:
            failures.append("summary contains a carrier other than p/n")
        per_group = summary.groupby("material_group_key")["carrier_type"].apply(set)
        if any(carriers != {"p", "n"} for carriers in per_group):
            failures.append("summary does not contain both p and n for every target group")
        for row in summary.itertuples(index=False):
            missing_side = row.sigma0ref25k_points == 0 or row.step31_ct_points == 0
            if missing_side and not str(row.warning).strip():
                failures.append(f"missing-side combination lacks warning: {row.material_group_key}, {row.carrier_type}")
            if missing_side and row.comparison_points != 0:
                failures.append(f"missing-side combination has comparisons: {row.material_group_key}, {row.carrier_type}")

    if figures.empty or not (figures["figure_type"] == "overlay").any():
        failures.append("at least one overlay figure was not created")
    if not (figures["figure_type"] == "log_ratio_nearest").any():
        failures.append("at least one log-ratio figure was not created")
    successful = set(
        zip(
            summary.loc[summary["comparison_points"].gt(0), "material_group_key"],
            summary.loc[summary["comparison_points"].gt(0), "carrier_type"],
        )
    )
    indexed = set(zip(figures["material_group_key"], figures["carrier_type"]))
    if indexed != successful:
        failures.append("figure index combinations differ from successful comparison combinations")
    if len(figures) != 2 * len(successful):
        failures.append("each successful combination does not have exactly two figure types")
    for row in figures.itertuples(index=False):
        for figure_path in [Path(row.figure_path_png), Path(row.figure_path_pdf)]:
            if not figure_path.exists() or figure_path.stat().st_size == 0:
                failures.append(f"missing/empty indexed figure: {figure_path}")

    expected_mapping_columns = [
        "source_label",
        "normalized_material_group_key",
        "mapping_status",
        "mapping_rule",
    ]
    if mapping.columns.tolist() != expected_mapping_columns:
        failures.append("material mapping columns do not match the specification")
    if unmatched.columns.tolist() != expected_mapping_columns:
        failures.append("unmatched material label columns do not match mapping schema")
    if not unmatched.empty and not unmatched["mapping_status"].eq("unmatched").all():
        failures.append("unmatched label table contains a matched label")

    all_columns = [column for frame in [sigma, ct, comparison, summary, figures] for column in frame.columns]
    if any("sigma_pred" in column for column in all_columns):
        failures.append("an output unexpectedly contains a sigma_pred column")
    report_text = args.report.read_text(encoding="utf-8")
    required_report_phrases = [
        "25 K sigma0_ref(T)",
        "Step31 SS2026 C(T)",
        "No new sigma_pred was calculated.",
        "Starrydata2 raw data was not read.",
        "Existing Step0-Step9A and Step31 outputs were not overwritten.",
    ]
    for phrase in required_report_phrases:
        if phrase not in report_text:
            failures.append(f"report is missing required phrase: {phrase}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"Step9B output check failed with {len(failures)} issue(s).")
        raise SystemExit(1)
    print(
        "PASS: Step9B outputs are complete and consistent "
        f"({len(summary)} summary rows, {len(comparison)} comparisons, {len(figures)} figures indexed)."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
