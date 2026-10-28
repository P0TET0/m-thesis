"""Independent checks for exp007 Phase 6 oxide_v1 re-prediction outputs."""

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
ATOL = 2e-12

REQUIRED = [
    DATA / "phase6_oxide_v1_all_row_assignments.csv",
    DATA / "phase6_oxide_v1_train_reference_bins.csv",
    DATA / "phase6_oxide_v1_test_predictions.csv",
    DATA / "phase6_prediction_comparison_rows.csv",
    DATA / "phase6_overall_metrics.csv",
    DATA / "phase6_paired_metrics.csv",
    DATA / "phase6_metric_deltas.csv",
    DATA / "phase6_oxide_scope_metrics.csv",
    DATA / "phase6_oxide_group_metrics.csv",
    DATA / "phase6_oxide_group_coverage.csv",
    DATA / "phase6_sample_prediction_changes.csv",
    DATA / "phase6_inventory_summary.csv",
    DATA / "phase6_input_manifest.csv",
    DATA / "phase6_validation_checks.csv",
    REPORTS / "phase6_oxide_v1_reprediction.md",
    REPORTS / "phase6_validation_log.txt",
    REPORTS / "phase6_workbook_validation_log.txt",
    REPORTS / "phase6_oxide_v1_reprediction.xlsx",
]

EXPECTED_SHEETS = {
    "Summary",
    "Oxide scopes",
    "Oxide groups",
    "Group coverage",
    "Sample changes",
    "Prediction rows",
    "Oxide assignments",
    "Reference bins",
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


def metric_values(residual: pd.Series) -> tuple[float, float, float, float]:
    values = pd.to_numeric(residual, errors="coerce").dropna().to_numpy(float)
    absolute = np.abs(values)
    return (
        float(np.mean(absolute)),
        float(np.sqrt(np.mean(values**2))),
        float(np.mean(absolute <= np.log10(2.0) + 1e-12)),
        float(np.mean(absolute <= 1.0 + 1e-12)),
    )


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
        raise SystemExit("Missing Phase 6 outputs; run all build steps first.")

    assignments = pd.read_csv(DATA / "phase6_oxide_v1_all_row_assignments.csv", low_memory=False)
    reference = pd.read_csv(DATA / "phase6_oxide_v1_train_reference_bins.csv", low_memory=False)
    predictions = pd.read_csv(DATA / "phase6_oxide_v1_test_predictions.csv", low_memory=False)
    comparison = pd.read_csv(DATA / "phase6_prediction_comparison_rows.csv", low_memory=False)
    overall = pd.read_csv(DATA / "phase6_overall_metrics.csv")
    paired = pd.read_csv(DATA / "phase6_paired_metrics.csv")
    deltas = pd.read_csv(DATA / "phase6_metric_deltas.csv")
    scopes = pd.read_csv(DATA / "phase6_oxide_scope_metrics.csv")
    groups = pd.read_csv(DATA / "phase6_oxide_group_metrics.csv")
    coverage = pd.read_csv(DATA / "phase6_oxide_group_coverage.csv")
    samples = pd.read_csv(DATA / "phase6_sample_prediction_changes.csv", low_memory=False)
    validation = pd.read_csv(DATA / "phase6_validation_checks.csv")
    manifest = pd.read_csv(DATA / "phase6_input_manifest.csv")
    phase4 = pd.read_csv(DATA / "phase4_oxide_v1_row_assignments.csv", usecols=["row_id", "oxide_v1_group_key"])

    check("all_build_checks_pass", validation["status"].eq("PASS").all(), f"failed={int(validation['status'].ne('PASS').sum())}")
    check("assignment_row_count", len(assignments) == 17715, f"rows={len(assignments)}")
    check("assignment_row_ids_unique", assignments["row_id"].is_unique, f"duplicates={int(assignments['row_id'].duplicated().sum())}")
    check("taxonomy_fixed", assignments["taxonomy_version"].eq("oxide_v1").all() and assignments["taxonomy_rules_sha256"].nunique() == 1, str(assignments["taxonomy_rules_sha256"].iloc[0]))
    heldout = assignments.loc[assignments["sample_holdout_split"].eq("test"), ["row_id", "oxide_v1_group_key"]].set_index("row_id").sort_index()
    frozen = phase4.set_index("row_id").sort_index()
    check("heldout_assignment_count", len(heldout) == 3437, f"rows={len(heldout)}")
    check("heldout_assignments_match_phase4", heldout.equals(frozen), "row-level oxide_v1 group keys")

    check("prediction_row_count", len(predictions) == 19008 and predictions["row_id"].is_unique, f"rows={len(predictions)}")
    status_counts = predictions["prediction_status"].value_counts().to_dict()
    check("prediction_status_counts", status_counts == {"ok": 18890, "unreliable_reference_bin": 70, "missing_reference_bin": 48}, str(status_counts))
    ok = predictions["prediction_status"].eq("ok")
    expected_prediction = predictions.loc[ok, "sigma0_ref_S_per_m"] * predictions.loc[ok, "F0_eta"]
    formula_diff = np.abs(predictions.loc[ok, "sigma_pred_S_per_m"] - expected_prediction)
    check("prediction_formula", bool(np.allclose(predictions.loc[ok, "sigma_pred_S_per_m"], expected_prediction, rtol=1e-10, atol=0.0)), f"max_abs_diff={float(formula_diff.max()):.3e}")
    check("reference_bins_nonempty", len(reference) > 0 and as_bool(reference["is_reference_bin_candidate"]).any(), f"bins={len(reference)}; reliable={int(as_bool(reference['is_reference_bin_candidate']).sum())}")

    check("comparison_row_count", len(comparison) == 19008 and comparison["row_id"].is_unique, f"rows={len(comparison)}")
    check("paired_row_count", int(as_bool(comparison["paired_prediction_available"]).sum()) == 18890, f"rows={int(as_bool(comparison['paired_prediction_available']).sum())}")
    nonoxide = comparison.loc[~as_bool(comparison["is_phase0_broad_oxide"]) & as_bool(comparison["paired_prediction_available"])]
    max_nonoxide_delta = float(nonoxide["delta_log10_prediction_phase6_minus_baseline"].abs().max())
    check("nonoxide_predictions_unchanged", max_nonoxide_delta <= ATOL, f"rows={len(nonoxide)}; max_abs_delta={max_nonoxide_delta:.3e}")
    oxide_changed = comparison.loc[as_bool(comparison["is_phase0_broad_oxide"]) & as_bool(comparison["paired_prediction_available"]), "delta_log10_prediction_phase6_minus_baseline"].abs().gt(ATOL)
    check("oxide_predictions_materially_changed", int(oxide_changed.sum()) > 0, f"changed={int(oxide_changed.sum())}")

    paired_rows = comparison.loc[as_bool(comparison["paired_prediction_available"])].copy()
    independently_computed: dict[str, tuple[float, float, float, float]] = {
        "broad_family_100K": metric_values(paired_rows["baseline_residual_log10"]),
        "oxide_v1_100K": metric_values(paired_rows["phase6_residual_log10"]),
    }
    reported = paired.loc[paired["metric_weighting"].eq("row_equal")].set_index("prediction_variant")
    for variant, expected in independently_computed.items():
        found = tuple(float(reported.loc[variant, column]) for column in ["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy"])
        difference = float(np.max(np.abs(np.asarray(found) - np.asarray(expected))))
        check(f"paired_metrics_recomputed_{variant}", difference <= 1e-12, f"max_abs_difference={difference:.3e}")

    phase0 = pd.read_csv(DATA / "phase0_baseline_metrics.csv")
    frozen_metric = phase0[(phase0["variant"].eq("broad_family_100K")) & phase0["metric_weighting"].eq("row_equal")].iloc[0]
    baseline_metric = overall[(overall["prediction_variant"].eq("broad_family_100K")) & overall["metric_weighting"].eq("row_equal")].iloc[0]
    check("baseline_overall_reproduces_phase0", abs(float(baseline_metric["mae_log10"]) - float(frozen_metric["mae_log10"])) <= 1e-12 and int(baseline_metric["prediction_ok_rows"]) == int(frozen_metric["prediction_ok_rows"]), f"mae_diff={abs(float(baseline_metric['mae_log10']) - float(frozen_metric['mae_log10'])):.3e}")
    check("overall_table_complete", len(overall) == 4 and len(paired) == 4 and len(deltas) == 32, f"overall={len(overall)}; paired={len(paired)}; deltas={len(deltas)}")
    check("scope_rows_complete", set(scopes["analysis_scope"]) == {"all_oxide_assigned", "default_eligible", "review_only", "excluded_only"}, str(sorted(scopes["analysis_scope"].unique())))
    check("group_coverage_reconciles", int(coverage["test_rows"].sum()) == 3437 and len(coverage) == 14, f"rows={int(coverage['test_rows'].sum())}; groups={len(coverage)}")
    check("group_metrics_scopes", set(groups["analysis_scope"]) == {"all_assigned", "default_eligible"}, str(sorted(groups["analysis_scope"].unique())))
    check("sample_rows_unique", samples["validation_sample_group_id"].is_unique and len(samples) == paired_rows["validation_sample_group_id"].nunique(), f"samples={len(samples)}")

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

    workbook = REPORTS / "phase6_oxide_v1_reprediction.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        bad_zip_member = archive.testzip()
    sheets = workbook_sheet_names(workbook)
    check("workbook_zip_integrity", bad_zip_member is None, f"bad_member={bad_zip_member}")
    check("workbook_sheets_complete", sheets == EXPECTED_SHEETS, f"sheets={sorted(sheets)}")
    check("workbook_has_no_error_text", not workbook_has_error_text(workbook), "scan of xl/*.xml")
    check("workbook_nonempty", workbook.stat().st_size > 100_000, f"size={workbook.stat().st_size}")
    check("workbook_validation_passes", "status=PASS" in (REPORTS / "phase6_workbook_validation_log.txt").read_text(encoding="utf-8"), "phase6_workbook_validation_log.txt")

    report = (REPORTS / "phase6_oxide_v1_reprediction.md").read_text(encoding="utf-8")
    check("report_documents_single_change", "Changed factor" in report and "Fixed factors" in report and "classification change is the only experimental intervention" in report, "controlled-comparison statements")

    passed = all(condition for _, condition, _ in checks)
    lines = [
        "Phase 6 independent checks",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"failed={sum(not condition for _, condition, _ in checks)}",
    ]
    lines.extend(f"{name}: {'PASS' if condition else 'FAIL'} | {detail}" for name, condition, detail in checks)
    (REPORTS / "phase6_check_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    if not passed:
        failed = [name for name, condition, _ in checks if not condition]
        raise SystemExit(f"Phase 6 checks failed: {failed}")


if __name__ == "__main__":
    main()
