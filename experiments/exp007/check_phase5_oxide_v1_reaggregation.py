"""Independent checks for exp007 Phase 5 oxide_v1 reaggregation outputs."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
ATOL = 1e-12

REQUIRED = [
    DATA / "phase5_oxide_v1_inventory_summary.csv",
    DATA / "phase5_oxide_v1_overall_scope_metrics.csv",
    DATA / "phase5_oxide_v1_group_metrics.csv",
    DATA / "phase5_oxide_v1_primary_group_metrics.csv",
    DATA / "phase5_oxide_v1_carrier_metrics.csv",
    DATA / "phase5_oxide_v1_temperature_metrics.csv",
    DATA / "phase5_oxide_v1_sample_metrics.csv",
    DATA / "phase5_oxide_v1_outlier_samples.csv",
    DATA / "phase5_oxide_v1_existing_prediction_rows.csv",
    DATA / "phase5_input_manifest.csv",
    DATA / "phase5_validation_checks.csv",
    REPORTS / "phase5_oxide_v1_reaggregation.md",
    REPORTS / "phase5_validation_log.txt",
    REPORTS / "phase5_workbook_validation_log.txt",
    REPORTS / "phase5_oxide_v1_reaggregation.xlsx",
]

EXPECTED_SHEETS = {
    "Summary",
    "Primary groups",
    "Scope comparison",
    "All group scopes",
    "Carrier metrics",
    "Temperature metrics",
    "Outlier samples",
    "Sample metrics",
    "Row predictions",
    "Validation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def workbook_sheet_names(path: Path) -> set[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return {node.attrib["name"] for node in root.findall("x:sheets/x:sheet", namespace)}


def workbook_has_error_text(path: Path) -> bool:
    pattern = re.compile(rb"#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!")
    with zipfile.ZipFile(path) as archive:
        return any(
            pattern.search(archive.read(name))
            for name in archive.namelist()
            if name.startswith("xl/") and name.endswith(".xml")
        )


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    check("required_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 5 outputs; run the build steps first.")

    phase1 = pd.read_csv(
        DATA / "phase1_sigma_residual_rows.csv",
        usecols=["row_id", "is_evaluable", "is_oxide_broad", "residual_log10_pred_over_observed"],
        low_memory=False,
    )
    phase4 = pd.read_csv(
        DATA / "phase4_oxide_v1_row_assignments.csv",
        usecols=["row_id", "oxide_v1_group_key", "oxide_v1_default_model_eligible"],
        low_memory=False,
    )
    rows = pd.read_csv(DATA / "phase5_oxide_v1_existing_prediction_rows.csv", low_memory=False)
    samples = pd.read_csv(DATA / "phase5_oxide_v1_sample_metrics.csv", low_memory=False)
    outliers = pd.read_csv(DATA / "phase5_oxide_v1_outlier_samples.csv", low_memory=False)
    overall = pd.read_csv(DATA / "phase5_oxide_v1_overall_scope_metrics.csv", low_memory=False)
    groups = pd.read_csv(DATA / "phase5_oxide_v1_group_metrics.csv", low_memory=False)
    primary = pd.read_csv(DATA / "phase5_oxide_v1_primary_group_metrics.csv", low_memory=False)
    carrier = pd.read_csv(DATA / "phase5_oxide_v1_carrier_metrics.csv", low_memory=False)
    temperature = pd.read_csv(DATA / "phase5_oxide_v1_temperature_metrics.csv", low_memory=False)
    validation = pd.read_csv(DATA / "phase5_validation_checks.csv")
    manifest = pd.read_csv(DATA / "phase5_input_manifest.csv")

    phase1_oxide = phase1.loc[as_bool(phase1["is_evaluable"]) & as_bool(phase1["is_oxide_broad"])].copy()
    check("all_build_validation_passes", validation["status"].eq("PASS").all(), f"failed={int(validation['status'].eq('FAIL').sum())}")
    check("row_count_reconciles", len(rows) == len(phase1_oxide) == len(phase4) == 3437, f"phase5={len(rows)}; phase1={len(phase1_oxide)}; phase4={len(phase4)}")
    check("row_ids_match_phase1", set(rows["row_id"].astype(str)) == set(phase1_oxide["row_id"].astype(str)), f"unique={rows['row_id'].nunique()}")
    check("row_ids_unique", rows["row_id"].is_unique, f"duplicates={int(rows['row_id'].duplicated().sum())}")
    check("taxonomy_fixed", rows["taxonomy_version"].eq("oxide_v1").all() and rows["taxonomy_rules_sha256"].nunique() == 1, str(rows["taxonomy_rules_sha256"].iloc[0]))
    check("classification_complete", rows["oxide_v1_group_key"].notna().all(), f"missing={int(rows['oxide_v1_group_key'].isna().sum())}")
    check("sample_count_reconciles", len(samples) == rows["validation_sample_group_id"].nunique() == 610, f"samples={len(samples)}")
    check("sample_ids_unique", samples["validation_sample_group_id"].is_unique, f"duplicates={int(samples['validation_sample_group_id'].duplicated().sum())}")
    check("no_sample_conflicts", not as_bool(samples["sample_assignment_conflict"]).any(), f"conflicts={int(as_bool(samples['sample_assignment_conflict']).sum())}")
    check("sample_status_counts_frozen", samples["oxide_v1_inclusion_status"].value_counts().to_dict() == {"include": 553, "review": 36, "exclude": 21}, str(samples["oxide_v1_inclusion_status"].value_counts().to_dict()))
    check("eligible_rows_frozen", int(as_bool(rows["oxide_v1_default_model_eligible"]).sum()) == 3032, f"rows={int(as_bool(rows['oxide_v1_default_model_eligible']).sum())}")

    source_residual = phase1_oxide.set_index("row_id")["residual_log10_pred_over_observed"]
    aligned_residual = rows["row_id"].map(source_residual)
    max_residual_diff = float(np.max(np.abs(rows["residual_log10_pred_over_observed"].to_numpy(float) - aligned_residual.to_numpy(float))))
    check("prediction_residuals_unchanged", max_residual_diff <= ATOL, f"max_abs_difference={max_residual_diff:.3e}")
    phase4_group = phase4.set_index("row_id")["oxide_v1_group_key"]
    check("group_assignments_match_phase4", rows.set_index("row_id")["oxide_v1_group_key"].sort_index().equals(phase4_group.sort_index()), "row-level group key")

    scope_counts = overall.set_index("phase5_analysis_scope")["n_rows"].astype(int).to_dict()
    check("overall_scope_rows_partition", scope_counts == {"all_assigned": 3437, "default_eligible": 3032, "review_only": 263, "excluded_only": 142}, str(scope_counts))
    group_scope_sums = groups.groupby("phase5_analysis_scope")["n_rows"].sum().astype(int).to_dict()
    check("group_scope_rows_reconcile", group_scope_sums == scope_counts, str(group_scope_sums))
    check("primary_has_all_groups", len(primary) == 14 and primary["oxide_v1_group_key"].is_unique, f"groups={len(primary)}")
    check("primary_rows_reconcile", int(primary["n_rows"].sum()) == 3032, f"sum={int(primary['n_rows'].sum())}")
    check("carrier_rows_reconcile", int(carrier["n_rows"].sum()) == 3032, f"sum={int(carrier['n_rows'].sum())}")
    check("temperature_rows_reconcile", int(temperature["n_rows"].sum()) == 3032, f"sum={int(temperature['n_rows'].sum())}")

    eligible = rows.loc[as_bool(rows["oxide_v1_default_model_eligible"])]
    independent_rows = []
    for group_key, subset in eligible.groupby("oxide_v1_group_key"):
        sample_medians = subset.groupby("validation_sample_group_id")["residual_log10_pred_over_observed"].median()
        independent_rows.append(
            {
                "oxide_v1_group_key": group_key,
                "n_rows_check": len(subset),
                "n_samples_check": subset["validation_sample_group_id"].nunique(),
                "row_mae_check": subset["residual_log10_pred_over_observed"].abs().mean(),
                "sample_equal_mae_check": sample_medians.abs().mean(),
            }
        )
    independent = pd.DataFrame(independent_rows).set_index("oxide_v1_group_key")
    reported = primary.set_index("oxide_v1_group_key")
    common = independent.index.intersection(reported.index)
    row_mae_diff = float(np.max(np.abs(independent.loc[common, "row_mae_check"] - reported.loc[common, "row_mae_log10"])))
    sample_mae_diff = float(np.max(np.abs(independent.loc[common, "sample_equal_mae_check"] - reported.loc[common, "sample_equal_mae_log10"])))
    check("primary_row_mae_recomputed", row_mae_diff <= ATOL, f"max_abs_difference={row_mae_diff:.3e}")
    check("primary_sample_equal_mae_recomputed", sample_mae_diff <= ATOL, f"max_abs_difference={sample_mae_diff:.3e}")
    check("supported_rank_scope", primary.loc[primary["sample_equal_mae_high_error_rank"].notna(), "descriptive_support_minimum_30rows_3samples_3papers"].map(lambda x: str(x).casefold() == "true").all(), f"ranked={int(primary['sample_equal_mae_high_error_rank'].notna().sum())}")

    expected_outliers = samples.loc[
        samples["median_residual_log10"].abs().gt(1.0 + ATOL)
        | as_bool(samples["sample_has_beyond_factor_10_row"])
    ]
    check("outlier_rule_reproduced", set(outliers["validation_sample_group_id"].astype(str)) == set(expected_outliers["validation_sample_group_id"].astype(str)), f"outliers={len(outliers)}")

    current_hashes: list[str | None] = []
    missing_manifest_sources: list[str] = []
    for item in manifest.itertuples(index=False):
        source = ROOT / str(item.path)
        if not source.is_file():
            missing_manifest_sources.append(str(item.path))
            current_hashes.append(None)
        else:
            current_hashes.append(sha256(source))
    check("manifest_sources_exist", not missing_manifest_sources, f"missing={missing_manifest_sources}")
    check("manifest_hashes_match", current_hashes == manifest["sha256"].tolist(), "current SHA-256 versus recorded")

    workbook = REPORTS / "phase5_oxide_v1_reaggregation.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        bad_zip_member = archive.testzip()
    sheets = workbook_sheet_names(workbook)
    check("workbook_zip_integrity", bad_zip_member is None, f"bad_member={bad_zip_member}")
    check("workbook_sheets_complete", sheets == EXPECTED_SHEETS, f"sheets={sorted(sheets)}")
    check("workbook_has_no_error_text", not workbook_has_error_text(workbook), "scan of xl/*.xml")
    check("workbook_nonempty", workbook.stat().st_size > 100_000, f"size={workbook.stat().st_size}")
    check("workbook_validation_passes", "status=PASS" in (REPORTS / "phase5_workbook_validation_log.txt").read_text(encoding="utf-8"), "phase5_workbook_validation_log.txt")

    report = (REPORTS / "phase5_oxide_v1_reaggregation.md").read_text(encoding="utf-8")
    check("report_records_no_retraining", "Model fitting: none" in report and "do not estimate the performance" in report, "scope and interpretation limit")

    passed = all(condition for _, condition, _ in checks)
    log_lines = [
        "Phase 5 independent checks",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"failed={sum(not condition for _, condition, _ in checks)}",
    ]
    log_lines.extend(
        f"{name}: {'PASS' if condition else 'FAIL'} | {detail}"
        for name, condition, detail in checks
    )
    (REPORTS / "phase5_check_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines[:4]))
    if not passed:
        failed = [name for name, condition, _ in checks if not condition]
        raise SystemExit(f"Phase 5 checks failed: {failed}")


if __name__ == "__main__":
    main()
