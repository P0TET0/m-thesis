"""Independent checks for exp007 Phase 7 classification evaluation outputs."""

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
FIGURES = EXP007 / "figures"
ATOL = 2e-12

REQUIRED = [
    DATA / "phase7_paired_metric_evaluation.csv",
    DATA / "phase7_bootstrap_intervals.csv",
    DATA / "phase7_oxide_group_evaluation.csv",
    DATA / "phase7_stratified_evaluation.csv",
    DATA / "phase7_primary_sample_changes.csv",
    DATA / "phase7_sample_change_summary.csv",
    DATA / "phase7_coverage_loss_rows.csv",
    DATA / "phase7_coverage_loss_summary.csv",
    DATA / "phase7_decision_summary.csv",
    DATA / "phase7_inventory_summary.csv",
    DATA / "phase7_input_manifest.csv",
    DATA / "phase7_validation_checks.csv",
    REPORTS / "phase7_classification_evaluation.md",
    REPORTS / "phase7_validation_log.txt",
    REPORTS / "phase7_workbook_validation_log.txt",
    REPORTS / "phase7_classification_evaluation.xlsx",
    FIGURES / "figure_phase7_primary_metrics_before_after.png",
    FIGURES / "figure_phase7_group_mae_coverage_tradeoff.png",
    FIGURES / "figure_phase7_sample_error_changes.png",
]

EXPECTED_SHEETS = {
    "Summary",
    "Primary metrics",
    "Bootstrap CIs",
    "Oxide groups",
    "Stratified",
    "Sample changes",
    "Coverage losses",
    "Coverage summary",
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


def metric_values(residual: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(residual, errors="coerce").dropna().to_numpy(float)
    absolute = np.abs(values)
    mean_error = float(values.mean())
    return {
        "mean_log10_error": mean_error,
        "abs_mean_log10_error": abs(mean_error),
        "mae_log10": float(absolute.mean()),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "factor_2_accuracy": float(np.mean(absolute <= np.log10(2.0) + ATOL)),
        "factor_5_accuracy": float(np.mean(absolute <= np.log10(5.0) + ATOL)),
        "factor_10_accuracy": float(np.mean(absolute <= 1.0 + ATOL)),
    }


def workbook_sheet_names(path: Path) -> set[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return {node.attrib["name"] for node in root.findall("x:sheets/x:sheet", namespace)}


def workbook_has_error_text(path: Path) -> bool:
    pattern = re.compile(rb"#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!")
    with zipfile.ZipFile(path) as archive:
        return any(pattern.search(archive.read(name)) for name in archive.namelist() if name.startswith("xl/") and name.endswith(".xml"))


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    check("required_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 7 outputs; run all build steps first.")

    comparison = pd.read_csv(DATA / "phase6_prediction_comparison_rows.csv", low_memory=False)
    metrics = pd.read_csv(DATA / "phase7_paired_metric_evaluation.csv")
    bootstrap = pd.read_csv(DATA / "phase7_bootstrap_intervals.csv")
    groups = pd.read_csv(DATA / "phase7_oxide_group_evaluation.csv")
    stratified = pd.read_csv(DATA / "phase7_stratified_evaluation.csv")
    samples = pd.read_csv(DATA / "phase7_primary_sample_changes.csv", low_memory=False)
    sample_summary = pd.read_csv(DATA / "phase7_sample_change_summary.csv")
    lost = pd.read_csv(DATA / "phase7_coverage_loss_rows.csv", low_memory=False)
    coverage = pd.read_csv(DATA / "phase7_coverage_loss_summary.csv")
    decision = pd.read_csv(DATA / "phase7_decision_summary.csv")
    validation = pd.read_csv(DATA / "phase7_validation_checks.csv")
    manifest = pd.read_csv(DATA / "phase7_input_manifest.csv")

    check("all_build_checks_pass", validation["status"].eq("PASS").all(), f"failed={int(validation['status'].ne('PASS').sum())}")
    paired = as_bool(comparison["paired_prediction_available"])
    oxide = as_bool(comparison["is_phase0_broad_oxide"])
    eligible = as_bool(comparison["oxide_v1_default_model_eligible"])
    primary_mask = paired & oxide & eligible
    primary = comparison.loc[primary_mask].copy()
    check("phase6_comparison_frozen", len(comparison) == 19008 and int(paired.sum()) == 18890, f"rows={len(comparison)}; paired={int(paired.sum())}")
    check("primary_cohort_frozen", len(primary) == 2984 and primary["validation_sample_group_id"].nunique() == 552, f"rows={len(primary)}; samples={primary['validation_sample_group_id'].nunique()}")

    reported = metrics[(metrics["analysis_scope"].eq("oxide_default_eligible")) & metrics["metric_weighting"].eq("row_equal")].set_index("metric_name")
    independently_computed = {
        "baseline": metric_values(primary["baseline_residual_log10"]),
        "oxide_v1": metric_values(primary["phase6_residual_log10"]),
    }
    metric_differences = []
    for metric in independently_computed["baseline"]:
        metric_differences.extend(
            [
                abs(float(reported.loc[metric, "baseline_value"]) - independently_computed["baseline"][metric]),
                abs(float(reported.loc[metric, "oxide_v1_value"]) - independently_computed["oxide_v1"][metric]),
            ]
        )
    check("primary_metrics_recomputed", max(metric_differences) <= ATOL, f"max_abs_difference={max(metric_differences):.3e}")

    sample_independent = (
        primary.groupby("validation_sample_group_id", sort=False)
        .agg(baseline=("baseline_residual_log10", "median"), oxide_v1=("phase6_residual_log10", "median"))
        .reset_index()
    )
    sample_independent["delta"] = sample_independent["oxide_v1"].abs() - sample_independent["baseline"].abs()
    improved = int((sample_independent["delta"] < -ATOL).sum())
    worsened = int((sample_independent["delta"] > ATOL).sum())
    unchanged = len(sample_independent) - improved - worsened
    primary_summary = sample_summary[sample_summary["summary_level"].eq("primary_scope")].iloc[0]
    check("sample_change_counts_recomputed", (improved, worsened, unchanged) == (int(primary_summary["improved_samples"]), int(primary_summary["worsened_samples"]), int(primary_summary["unchanged_samples"])), f"counts={(improved, worsened, unchanged)}")
    check("sample_change_rows_match", len(samples) == len(sample_independent) and samples["validation_sample_group_id"].is_unique, f"rows={len(samples)}")

    check("bootstrap_shape", len(bootstrap) == 42 and set(bootstrap["bootstrap_replicates"]) == {2000}, f"rows={len(bootstrap)}")
    check("bootstrap_intervals_contain_point", bool((bootstrap["ci95_lower"] <= bootstrap["point_delta"] + ATOL).all() and (bootstrap["point_delta"] <= bootstrap["ci95_upper"] + ATOL).all()), "lower <= point <= upper")
    primary_bootstrap = bootstrap[(bootstrap["analysis_scope"].eq("oxide_default_eligible")) & bootstrap["metric_weighting"].eq("row_equal")].set_index("metric_name")
    check("primary_error_intervals_support_improvement", primary_bootstrap.loc[["mae_log10", "rmse_log10"], "interval_interpretation"].eq("supported_improvement").all(), str(primary_bootstrap.loc[["mae_log10", "rmse_log10"], ["ci95_lower", "ci95_upper"]].to_dict("index")))

    check("group_rows_reconcile", int(groups["paired_rows"].sum()) == len(primary), f"rows={int(groups['paired_rows'].sum())}")
    check("group_samples_have_support_labels", groups["support_note"].notna().all() and len(groups) == 12, f"groups={len(groups)}")
    check("stratified_dimensions_complete", set(stratified["dimension"]) == {"oxide_v1_group", "carrier_type", "temperature_bin", "experimental_sigma_decade"}, str(sorted(stratified["dimension"].unique())))

    baseline_ok = as_bool(comparison["baseline_prediction_available"])
    new_ok = as_bool(comparison["phase6_prediction_available"])
    independently_lost = comparison.loc[baseline_ok & ~new_ok, "row_id"].astype(str).sort_values().tolist()
    check("coverage_loss_rows_recomputed", independently_lost == lost["row_id"].astype(str).sort_values().tolist(), f"rows={len(lost)}")
    primary_lost = lost.loc[as_bool(lost["oxide_v1_default_model_eligible"])]
    check("primary_coverage_loss_count", len(primary_lost) == 48, f"rows={len(primary_lost)}")
    primary_coverage = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    check("primary_coverage_recomputed", int(primary_coverage["test_rows"]) == int((oxide & eligible).sum()) == 3032 and int(primary_coverage["oxide_v1_prediction_ok_rows"]) == 2984, f"test={primary_coverage['test_rows']}; ok={primary_coverage['oxide_v1_prediction_ok_rows']}")

    verdict = decision.loc[decision["item"].eq("overall_verdict"), "result"].iloc[0]
    check("decision_summary_matches_evidence", verdict == "classification_improves_large_error_control_with_small_coverage_cost", verdict)

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

    for figure_name in [
        "figure_phase7_primary_metrics_before_after.png",
        "figure_phase7_group_mae_coverage_tradeoff.png",
        "figure_phase7_sample_error_changes.png",
    ]:
        figure = FIGURES / figure_name
        with figure.open("rb") as handle:
            signature = handle.read(8)
        check(f"figure_valid_{figure.stem}", signature == b"\x89PNG\r\n\x1a\n" and figure.stat().st_size > 10_000, f"size={figure.stat().st_size}")

    workbook = REPORTS / "phase7_classification_evaluation.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        bad_zip_member = archive.testzip()
    sheets = workbook_sheet_names(workbook)
    check("workbook_zip_integrity", bad_zip_member is None, f"bad_member={bad_zip_member}")
    check("workbook_sheets_complete", sheets == EXPECTED_SHEETS, f"sheets={sorted(sheets)}")
    check("workbook_has_no_error_text", not workbook_has_error_text(workbook), "scan of xl/*.xml")
    check("workbook_nonempty", workbook.stat().st_size > 100_000, f"size={workbook.stat().st_size}")
    check("workbook_validation_passes", "status=PASS" in (REPORTS / "phase7_workbook_validation_log.txt").read_text(encoding="utf-8"), "phase7_workbook_validation_log.txt")

    report = (REPORTS / "phase7_classification_evaluation.md").read_text(encoding="utf-8")
    check("report_documents_design", "Primary cohort" in report and "Sample-equal" in report and "Coverage is evaluated separately" in report, "paired, sample-equal, and coverage statements")
    check("report_documents_limits", "external database" in report and "composition-only taxonomy" in report, "interpretation limits")

    passed = all(condition for _, condition, _ in checks)
    lines = [
        "Phase 7 independent checks",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"failed={sum(not condition for _, condition, _ in checks)}",
    ]
    lines.extend(f"{name}: {'PASS' if condition else 'FAIL'} | {detail}" for name, condition, detail in checks)
    (REPORTS / "phase7_check_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]))
    if not passed:
        failed = [name for name, condition, _ in checks if not condition]
        raise SystemExit(f"Phase 7 checks failed: {failed}")


if __name__ == "__main__":
    main()
